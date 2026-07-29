#!/usr/bin/env ruby
# prep/franken_moe_cuda_gate.rb — toy#134 gates for the CUDA twin of the
# spec-callable MoE runner (`toy train franken-moe --device cuda`).
# CUDA curves are PLATFORM-SCOPED: determinism/structure legs only,
# never compared to CPU fixtures (the standing CUDA discipline).
#
#   1. LANES: dense chain / dense dfa-experts / top1 fully-DFA /
#      bp-router / bp-spine all train (finite, deterministic 2x).
#      Dense rides vendor-patch 0012 (cublasSger k==1 out_prod
#      fallback — the gating-path backward crashed cublas before it).
#   2. TOP1 COLLAPSE + bp-spine step-1 forward-identity, on-device.
#   3. BATCH: the toy#133 order-swap isolation null on CUDA.
#   4. COMPOSE: E=8 wide corpus batch --no-shadow-free lane + eval;
#      ckpt round-trip (write stages through the CPU session; reload
#      eval byte-equals the training run's own end-of-run eval).

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

CLEAN = { "SPINEL_DIR" => nil, "SPINEL_SKIP_PIN_CHECK" => nil }
SMOKE = "data/fineweb_gpt2_smoke.bin"
abort "franken_moe_cuda_gate: #{SMOKE} missing — generate it: uv run prep/pretokenize_pack.py --tokens 200_000 --out #{SMOKE}" unless File.file?(File.join(ROOT, SMOKE))

def run_cli(args, run_dir)
  env = CLEAN.merge("TOY_RUN_ID" => "moe-cuda-gate")
  argv = [TOY, "train", "franken-moe", "--device", "cuda"] + args
  argv += ["--out", run_dir] if run_dir
  out, st = Open3.capture2e(env, *argv, chdir: ROOT)
  abort "franken_moe_cuda_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def losses(out)
  out.lines.select { |l| l.start_with?("step ") }.map { |l| l[/loss=(\S+)/, 1] }
end

failures = []

# ---- 1. lanes: train + deterministic 2x ----
LANES = [
  ["dense-chain",  %w[--steps 6 --seed 0]],
  ["dfa-experts",  %w[--steps 6 --seed 0 --moe-policy dfa-experts]],
  ["top1-dfa",     %w[--steps 6 --seed 0 --routing top1]],
  ["bp-router",    %w[--steps 6 --seed 0 --routing top1 --moe-policy bp-router-dfa-experts]],
  ["bp-spine",     %w[--steps 6 --seed 0 --routing top1 --moe-policy bp-spine]],
]
lane_first = {}
LANES.each do |name, args|
  a = losses(run_cli(args, nil))
  b = losses(run_cli(args, nil))
  failures << "#{name}: not deterministic" unless a == b && a.length == 6
  failures << "#{name}: NaN" if a.map(&:to_f).any?(&:nan?)
  lane_first[name] = a.first
end
failures << "lanes: dfa-experts identical to chain (policy dead on cuda)" if lane_first.empty?
puts failures.empty? ? "  ok: 5 lanes train deterministically on CUDA (dense rides vendor-patch 0012)" : "  FAIL: lanes"

# ---- 2. top1 collapse + bp-spine forward-identity ----
n0 = failures.length
Dir.mktmpdir("moe_cuda_top1") do |dir|
  run_cli(%w[--steps 40 --seed 0 --routing top1], dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  shares = evs.select { |e| e["kind"] == "route" }.map { |e| e["shares"][0] }
  tail = shares.last(10)
  failures << "top1: no collapse on cuda (tail #{tail.uniq.inspect})" unless tail.uniq.length == 1 && (tail.first == 0.0 || tail.first == 1.0)
end
failures << "bp-spine: step-1 forward differs from fully-dfa (detach not identity on cuda)" unless lane_first["bp-spine"] == lane_first["top1-dfa"]
puts failures.length == n0 ? "  ok: top1 collapses; bp-spine step-1 byte-equals fully-dfa on-device" : "  FAIL: top1/spine"

# ---- 3. the order-swap batch isolation null (toy#133, on CUDA) ----
n0 = failures.length
Dir.mktmpdir("moe_cuda_batch") do |dir|
  raw = File.binread(File.join(ROOT, SMOKE))
  w = raw[16, 32 * 4]
  x = raw[16 + 5000 * 4, 32 * 4]
  File.binwrite(File.join(dir, "w.bin"), w)
  File.binwrite(File.join(dir, "x.bin"), x)
  File.binwrite(File.join(dir, "wx.bin"), w + x)
  File.binwrite(File.join(dir, "xw.bin"), x + w)
  get1 = lambda do |args|
    losses(run_cli(args, nil)).first.to_f
  end
  base = ["--steps", "1", "--seed", "0", "--context", "32", "--vocab", "50257"]
  lw = get1.call(base + ["--corpus", File.join(dir, "w.bin")])
  lx = get1.call(base + ["--corpus", File.join(dir, "x.bin")])
  lwx = get1.call(base + ["--corpus", File.join(dir, "wx.bin"), "--batch", "2"])
  lxw = get1.call(base + ["--corpus", File.join(dir, "xw.bin"), "--batch", "2"])
  mean = (lw + lx) / 2.0
  failures << "batch: [W,X] #{lwx} != mean #{mean} on cuda" unless (lwx - mean).abs < 1.0e-4
  failures << "batch: order-swap #{lxw} != #{lwx} (cross-window leak on cuda)" unless (lxw - lwx).abs < 1.0e-4
end
puts failures.length == n0 ? "  ok: order-swap batch isolation null holds on CUDA" : "  FAIL: batch null"

# ---- 4. the F9 composition + ckpt round-trip ----
n0 = failures.length
Dir.mktmpdir("moe_cuda_f9") do |dir|
  f9 = %w[--steps 8 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 32 --shape wide
          --experts 8 --routing top1 --moe-policy bp-spine --moe-aux 0.15 --batch 2
          --lr 0.005 --warmup 4]
  ev = %w[--eval-corpus data/fineweb_gpt2_smoke.bin --eval-tokens 256 --eval-offset 150000]
  out = run_cli(f9 + ev + %w[--ckpt-every 4], dir)
  failures << "f9: short/NaN" unless losses(out).length == 8 && losses(out).map(&:to_f).none?(&:nan?)
  train_ce = out.lines.select { |l| l.start_with?("eval_ce:") }
  failures << "f9: no eval_ce" unless train_ce.length == 1
  ck = File.join(dir, "weights", "step_8.gguf")
  failures << "f9: missing checkpoint" unless File.file?(ck)
  # the reload must mirror the EVAL-GRAPH params (--context/--batch) —
  # weights are ctx/batch-independent but the eval windowing is not
  # (found live: batch 2 vs unbatched = same per-token CE, different
  # window partition -> 1e-7 mean drift + different windows= count).
  rl = run_cli(%w[--steps 0 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 32 --shape wide
                  --experts 8 --routing top1 --moe-policy bp-spine --batch 2] + ["--load-ckpt", ck] + ev, nil)
  rl_ce = rl.lines.select { |l| l.start_with?("eval_ce:") }
  failures << "f9: ckpt round-trip eval differs\ntrain: #{train_ce.join}reload: #{rl_ce.join}" unless rl_ce == train_ce
end
puts failures.length == n0 ? "  ok: F9 composition (wide/E8/bp-spine/aux/batch/lr/warmup) + eval + ckpt round-trip byte-equal on CUDA" : "  FAIL: f9 leg"

if failures.empty?
  puts "GATE PASS [franken-moe-cuda]: lanes(5, det-2x) + collapse/spine-identity + batch order-swap null + F9 composition/ckpt round-trip (toy#134; vendor-patch 0012)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken-moe-cuda]: #{f}" }
  exit 1
end
