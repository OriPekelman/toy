#!/usr/bin/env ruby
# prep/franken_llama_gate.rb — toy#112 gates for the spec-callable franken
# runner (libexec/toy-train-franken-llama, `toy train franken`).
#
#   1. F0 BYTE-PARITY: empty policy, STEPS=5 SEED=0 — the stdout loss
#      curve must byte-equal prep/fixtures/train_baseline.txt (the same
#      fixture `toy train from-scratch` is gated on): the franken runner
#      with no policy IS the from-scratch trainer.
#   2. BUNDLE STRUCTURE (policy run, TAO_RUN_DIR): valid JSONL; run_start
#      first with schema toy/v1 + a `franken` object carrying the exact
#      policy/b axes; one step event per step; align events == 12 ×
#      steps when FRANKEN_ALIGN=1 (2 layers × [4q+4k+4v] on the dfa
#      layer... exactly the dfa-wired count) with finite cos in [-1,1]
#      and non-negative norms; run_end last; weights/step_<N>.gguf.
#   3. DFA EFFECT: the chain,dfa curve differs from the F0 curve.
#   4. BYTE-REPRO: two identical policy runs — identical stdout.

ROOT   = File.expand_path("..", __dir__)
RUNNER = File.join(ROOT, "libexec", "toy-train-franken-llama")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_baseline.txt")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

def run_franken_llama(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  if run_dir
    FileUtils.mkdir_p(File.join(run_dir, "weights"))
    env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "franken-gate")
  end
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "franken_llama_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-franken-llama")
  unless build_st.success? && File.executable?(RUNNER)
    warn "franken_llama_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []

# ---- 1. F0 byte-parity ----
f0_out = run_franken_llama({}, nil)
f0_curve = f0_out.lines.select { |l| l.start_with?("step ") }
expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
if f0_curve == expect
  puts "  ok: F0 — empty-policy curve byte-equals train_baseline.txt (#{f0_curve.length} steps)"
else
  failures << "F0: curve != train_baseline.txt\ngot:  #{f0_curve.join}want: #{expect.join}"
end

# ---- 2 + 3. policy run: bundle structure + dfa effect ----
Dir.mktmpdir("franken_llama_gate") do |dir|
  pol_out = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa",
                                "FRANKEN_B_SEED" => "42",
                                "FRANKEN_ALIGN"  => "1" }, dir)
  pol_curve = pol_out.lines.select { |l| l.start_with?("step ") }
  failures << "dfa-effect: policy curve identical to F0" if pol_curve == f0_curve

  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    failures << "bundle: first event not run_start" unless events.first && events.first["kind"] == "run_start"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    rs = events.first || {}
    failures << "bundle: schema != toy/v1" unless rs["schema"] == "toy/v1"
    # toy#129 item 4: derived cost accounting rides run_start
    co = rs["cost"]
    if co.nil?
      failures << "bundle: run_start has no cost object (toy#129)"
    else
      failures << "bundle: cost fields not positive (#{co.inspect})" unless %w[total_params active_params flops_per_token].all? { |k| co[k].is_a?(Numeric) && co[k] > 0 }
      failures << "bundle: llama cost active != total (#{co.inspect})" unless co["active_params"] == co["total_params"]
    end
    fr = rs["franken"]
    if fr.nil?
      failures << "bundle: run_start has no franken object"
    else
      failures << "bundle: franken.policy != [0,1] (got #{fr['policy'].inspect})" unless fr["policy"] == [0, 1]
      failures << "bundle: franken.b_seed != 42" unless fr["b_seed"] == 42
    end
    steps_ev = events.count { |e| e["kind"] == "step" }
    failures << "bundle: #{steps_ev} step events (want 5)" unless steps_ev == 5
    aligns = events.select { |e| e["kind"] == "align" }
    failures << "bundle: #{aligns.length} align events (want 60 = 12 dfa weights x 5 steps)" unless aligns.length == 60
    bad = aligns.count do |e|
      c = e["cos"]
      !c.is_a?(Numeric) || c.to_f.nan? || c.to_f.abs > 1.0001 ||
        !e["dfa_norm"].is_a?(Numeric) || e["dfa_norm"] < 0 ||
        !e["bp_norm"].is_a?(Numeric) || e["bp_norm"] < 0
    end
    failures << "bundle: #{bad} malformed align events" unless bad == 0
    # the engine-scale telemetry sanity: dfa signal is alive
    live = aligns.count { |e| e["dfa_norm"] > 0 }
    failures << "bundle: dfa_norm not positive anywhere" unless live == aligns.length
  else
    failures << "bundle: no events.jsonl"
  end
  ckpt = File.join(dir, "weights", "step_5.gguf")
  failures << "bundle: missing checkpoint #{ckpt}" unless File.file?(ckpt)
  # toy#112 gap closed: flow.json (the toy#25 self-describing bundle)
  fj = File.join(dir, "flow.json")
  if File.file?(fj)
    flow = JSON.parse(File.read(fj)) rescue nil
    if flow.nil? || flow["format"] != "toy/v1" || !flow["nodes"].is_a?(Array) || flow["nodes"].empty?
      failures << "bundle: flow.json invalid (format/nodes)"
    end
  else
    failures << "bundle: no flow.json"
  end
  puts "  ok: bundle — run_start(franken provenance) + 5 steps + 60 align + run_end + checkpoint; dfa curve differs" if failures.empty?
end

# ---- seed!=0 parity (toy#113): franken empty-policy must equal
# toy-train AT THE SAME NONZERO SEED, compared DYNAMICALLY (no fixture:
# seed!=0 curves are toy-version-scoped; only seed=0 is frozen). This
# tripwires the whole-program numeric-stream divergence class — at
# f7cea71 the two units compiled DIFFERENT xorshift/Box-Muller streams
# from identical source (seed=0 masked it).
FS_RUNNER = File.join(ROOT, "libexec", "toy-train")
fs_out, fs_st = Open3.capture2e({ "STEPS" => "8", "SEED" => "1" }, FS_RUNNER, chdir: ROOT)
fr_out = run_franken_llama({ "STEPS" => "8", "SEED" => "1" }, nil)
fs_curve = fs_out.lines.select { |l| l.start_with?("step ") }
fr_curve = fr_out.lines.select { |l| l.start_with?("step ") }
if fs_st.success? && fs_curve.length == 8 && fs_curve == fr_curve
  puts "  ok: seed=1 parity — franken empty-policy byte-equals toy-train (8 steps, dynamic)"
else
  failures << "seed1-parity: curves differ or toy-train failed\nfs: #{fs_curve.join}fr: #{fr_curve.join}"
end

# ---- TOY_RUN_ID passthrough (toy#115): the CLI's controlled env must
# forward a caller-supplied run id into run_start.run_id (the tao#flow
# contract) instead of substituting the internal counter.
Dir.mktmpdir("franken_rid_gate") do |dir|
  _, st = Open3.capture2e({ "TOY_RUN_ID" => "gate/rid/check" },
                          File.join(ROOT, "bin", "toy"), "train", "franken",
                          "--steps", "1", "--seed", "0", "--out", dir, chdir: ROOT)
  rid = st.success? ? (JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)["run_id"] rescue nil) : nil
  if rid == "gate/rid/check"
    puts "  ok: TOY_RUN_ID passthrough (CLI controlled env honors caller id)"
  else
    failures << "run-id: TOY_RUN_ID not honored (got #{rid.inspect})"
  end
end

# ---- llama-shape maskbp byte-null (toy#117): maskbp:-1 (gate saturated
# to exactly 1.0) must byte-equal toy-train — the leg that would have
# caught the B-buffer scratch-reuse explosion (rig shape provably
# doesn't cover it; B leaves are per-step re-uploads now).
fs8 = Open3.capture2e({ "STEPS" => "8", "SEED" => "0" }, FS_RUNNER, chdir: ROOT)[0]
mb8 = run_franken_llama({ "FRANKEN_POLICY" => "maskbp:-1,maskbp:-1",
                          "FRANKEN_B_SEED" => "42", "STEPS" => "8" }, nil)
fs8c = fs8.lines.select { |l| l.start_with?("step ") }
mb8c = mb8.lines.select { |l| l.start_with?("step ") }
if fs8c.length == 8 && fs8c == mb8c
  puts "  ok: maskbp(-1) byte-equals toy-train at the llama shape (toy#117 pin)"
else
  failures << "maskbp-null: curves differ\nfs: #{fs8c.join}mb: #{mb8c.join}"
end

# ---- toy#122: F6 long-horizon surface — corpus stream + align thinning ----
# corpus arm: deterministic, actually streams (differs from the fixed-seq
# feed), and leaves the byte-gated default path untouched (leg 1 pins it).
c1 = run_franken_llama({ "STEPS" => "8", "CORPUS" => "data/ts_seqs.bin" }, nil)
c2 = run_franken_llama({ "STEPS" => "8", "CORPUS" => "data/ts_seqs.bin" }, nil)
c_curve = c1.lines.select { |l| l.start_with?("step ") }
failures << "corpus: not deterministic" unless c1 == c2
failures << "corpus: only #{c_curve.length} steps" unless c_curve.length == 8
d_curve = run_franken_llama({ "STEPS" => "8" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "corpus: curve identical to fixed-seq feed (stream not live)" if c_curve == d_curve
puts failures.empty? ? "  ok: --corpus streams (deterministic, differs from fixed-seq; default feed untouched)" : "  FAIL: corpus leg"

# align-every thinning: N=3 over 6 steps -> emissions at steps 1,4 ->
# 12 weights x 2 = 24 align events; N=1 == legacy per-step (72).
Dir.mktmpdir("franken_ae_gate") do |dir|
  run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42",
                      "FRANKEN_ALIGN" => "1", "FRANKEN_ALIGN_EVERY" => "3",
                      "STEPS" => "6" }, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  aligns = evs.select { |e| e["kind"] == "align" }
  steps_seen = aligns.map { |e| e["step"] }.uniq.sort
  failures << "align-every: #{aligns.length} events (want 24)" unless aligns.length == 24
  failures << "align-every: wrong steps #{steps_seen.inspect} (want [1,4])" unless steps_seen == [1, 4]
  failures << "align-every: step events thinned too (#{evs.count { |e| e['kind'] == 'step' }})" unless evs.count { |e| e["kind"] == "step" } == 6
end
puts failures.empty? ? "  ok: --align-every thins align emissions (24 @ N=3/6 steps; step events untouched)" : "  FAIL: align-every leg"

# ---- toy#126: --lr / --warmup (the F7b LR-sweep surface) ----
# --lr: deterministic, actually moves the curve (default 0.001 stays
# byte-null via leg 1). --warmup: linear ramp reaches LR at step N —
# pinned via the step events' lr field (the ramp is Ruby-side; the
# events are the observable), and the ramped curve differs from flat.
lr_env = { "STEPS" => "6", "FRANKEN_LR" => "0.01" }
lr1 = run_franken_llama(lr_env, nil).lines.select { |l| l.start_with?("step ") }
lr2 = run_franken_llama(lr_env, nil).lines.select { |l| l.start_with?("step ") }
d6  = run_franken_llama({ "STEPS" => "6" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "lr: not deterministic" unless lr1 == lr2 && lr1.length == 6
failures << "lr: curve identical to default 0.001 (knob dead)" if lr1 == d6
wu_env = lr_env.merge("FRANKEN_WARMUP" => "4")
wu1 = nil
Dir.mktmpdir("franken_lr_gate") do |dir|
  wu1 = run_franken_llama(wu_env, dir).lines.select { |l| l.start_with?("step ") }
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  lrs = evs.select { |e| e["kind"] == "step" }.map { |e| e["lr"] }
  want = [1, 2, 3, 4].map { |t| 0.01 * (t.to_f / 4.0) } + [0.01, 0.01]
  ramp_ok = lrs.length == 6 && lrs.zip(want).all? { |g, w| g.is_a?(Numeric) && (g - w).abs < 1.0e-12 }
  failures << "warmup: step-event lr ramp #{lrs.inspect} (want #{want.inspect})" unless ramp_ok
  failures << "warmup: run_start config warmup/lr wrong" unless evs.first.dig("config", "warmup") == 4 && (evs.first.dig("config", "lr").to_f - 0.01).abs < 1.0e-12
end
wu2 = run_franken_llama(wu_env, nil).lines.select { |l| l.start_with?("step ") }
failures << "warmup: not deterministic" unless wu1 == wu2
failures << "warmup: curve identical to flat lr (ramp dead)" if wu1 == lr1
puts failures.empty? ? "  ok: --lr moves the curve (deterministic); --warmup ramps to LR @N (events pin the ramp, curve differs from flat)" : "  FAIL: lr/warmup leg"

# ---- toy#124: shape presets, byte-pinned per preset ----
%w[wide deep].each do |sh|
  fx = File.join(ROOT, "prep", "fixtures", "franken_#{sh}_baseline.txt")
  exp = File.readlines(fx).reject { |l| l.start_with?("#") || l.strip.empty? }.map(&:chomp)
  got = run_franken_llama({ "FRANKEN_SHAPE" => sh, "STEPS" => "5" }, nil)
        .lines.select { |l| l.start_with?("step ") }.map(&:chomp)
  if got == exp
    puts "  ok: --shape #{sh} byte-equals its recorded baseline (5 steps)"
  else
    failures << "shape-#{sh}: curve != #{File.basename(fx)}\ngot:  #{got.join(' | ')}\nwant: #{exp.join(' | ')}"
  end
  s1 = run_franken_llama({ "FRANKEN_SHAPE" => sh, "STEPS" => "5", "SEED" => "1" }, nil)
  s2 = run_franken_llama({ "FRANKEN_SHAPE" => sh, "STEPS" => "5", "SEED" => "1" }, nil)
  failures << "shape-#{sh}: seed=1 not deterministic" unless s1 == s2
end

# ---- toy#129 item 1: TOYC pack header + context/vocab unpinning ----
# The committed GPT-2 smoke pack (data/fineweb_gpt2_smoke.bin, TOYC v1
# vocab 50257) drives the instrument's vocab from the HEADER; context
# comes from FRANKEN_CONTEXT. Deterministic; provenance carries both;
# a conflicting FRANKEN_VOCAB fails loud; a headerless pack + explicit
# FRANKEN_VOCAB=627 byte-equals the implicit-627 run (the flag is a
# name for what already happens, not a new code path).
abort "franken gate: data/fineweb_gpt2_smoke.bin missing — generate it: uv run prep/pretokenize_pack.py --tokens 200_000 --out data/fineweb_gpt2_smoke.bin" unless File.file?(File.join(ROOT, "data", "fineweb_gpt2_smoke.bin"))
pk_env = { "STEPS" => "3", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_CONTEXT" => "64" }
pk1 = nil
Dir.mktmpdir("franken_pack_gate") do |dir|
  pk1 = run_franken_llama(pk_env, dir)
  rsp = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "pack: model.vocab #{rsp.dig('model', 'vocab').inspect} (want 50257 from the TOYC header)" unless rsp.dig("model", "vocab") == 50257
  failures << "pack: config.context #{rsp.dig('config', 'context').inspect} (want 64)" unless rsp.dig("config", "context") == 64
end
pk2 = run_franken_llama(pk_env, nil)
pk1_c = pk1.lines.select { |l| l.start_with?("step ") }
pk2_c = pk2.lines.select { |l| l.start_with?("step ") }
failures << "pack: not deterministic" unless pk1_c == pk2_c && pk1_c.length == 3
l0 = pk1_c.first[/loss=(\S+)/, 1].to_f
failures << "pack: first loss #{l0} not ~ln(50257)" unless l0 > 9.0 && l0 < 12.5
_oc, stc = Open3.capture2e({ "STEPS" => "1", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_VOCAB" => "100" }, RUNNER, chdir: ROOT)
failures << "pack: conflicting FRANKEN_VOCAB not rejected" if stc.success?
hv1 = run_franken_llama({ "STEPS" => "3", "CORPUS" => "data/ts_seqs.bin", "FRANKEN_VOCAB" => "627" }, nil)
hv0 = run_franken_llama({ "STEPS" => "3", "CORPUS" => "data/ts_seqs.bin" }, nil)
failures << "pack: explicit FRANKEN_VOCAB=627 differs from implicit (headerless null broken)" unless hv1.lines.select { |l| l.start_with?("step ") } == hv0.lines.select { |l| l.start_with?("step ") }
puts failures.empty? ? "  ok: TOYC pack — vocab 50257 from header, ctx 64, deterministic; vocab-conflict rejected; headerless --vocab null" : "  FAIL: pack leg"

# ---- toy#133: --batch (B corpus windows per step) ----
# THE isolation null: with distinct windows W and X, batched [W,X] and
# the order-swap [X,W] must BOTH equal the mean of the two B=1 losses —
# any cross-window attention leak breaks order-invariance first (the
# lesson: a mis-allocated mask read zeros and B=2 ran fully unmasked;
# this leg would have caught it). Plus B=1 flag-null and B=4
# determinism.
n0 = failures.length
Dir.mktmpdir("franken_batch_gate") do |dir|
  raw = File.binread(File.join(ROOT, "data", "fineweb_gpt2_smoke.bin"))
  w = raw[16, 32 * 4]
  x = raw[16 + 5000 * 4, 32 * 4]
  File.binwrite(File.join(dir, "w.bin"), w)
  File.binwrite(File.join(dir, "x.bin"), x)
  File.binwrite(File.join(dir, "wx.bin"), w + x)
  File.binwrite(File.join(dir, "xw.bin"), x + w)
  get1 = lambda do |env|
    out = run_franken_llama(env, nil)
    out.lines.select { |l| l.start_with?("step ") }.first[/loss=(\S+)/, 1].to_f
  end
  base_env = { "STEPS" => "1", "FRANKEN_VOCAB" => "50257" }
  lw = get1.call(base_env.merge("CORPUS" => File.join(dir, "w.bin")))
  lx = get1.call(base_env.merge("CORPUS" => File.join(dir, "x.bin")))
  lwx = get1.call(base_env.merge("CORPUS" => File.join(dir, "wx.bin"), "FRANKEN_BATCH" => "2"))
  lxw = get1.call(base_env.merge("CORPUS" => File.join(dir, "xw.bin"), "FRANKEN_BATCH" => "2"))
  mean = (lw + lx) / 2.0
  failures << "batch: [W,X] #{lwx} != mean #{mean} (leak or wrong labels)" unless (lwx - mean).abs < 1.0e-4
  failures << "batch: order-swap [X,W] #{lxw} != [W,X] #{lwx} (cross-window leak)" unless (lxw - lwx).abs < 1.0e-4
end
bn1 = run_franken_llama({ "STEPS" => "4", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_BATCH" => "1" }, nil)
bn0 = run_franken_llama({ "STEPS" => "4", "CORPUS" => "data/fineweb_gpt2_smoke.bin" }, nil)
failures << "batch: FRANKEN_BATCH=1 differs from no-flag (flag-null broken)" unless bn1.lines.select { |l| l.start_with?("step ") } == bn0.lines.select { |l| l.start_with?("step ") }
b4a = run_franken_llama({ "STEPS" => "4", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_BATCH" => "4" }, nil).lines.select { |l| l.start_with?("step ") }
b4b = run_franken_llama({ "STEPS" => "4", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_BATCH" => "4" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "batch: B=4 not deterministic" unless b4a == b4b && b4a.length == 4
failures << "batch: B=4 NaN" if b4a.any? { |l| l[/loss=(\S+)/, 1].to_f.nan? }
_ob, stb = Open3.capture2e({ "STEPS" => "1", "FRANKEN_BATCH" => "2" }, RUNNER, chdir: ROOT)
failures << "batch: B>1 without corpus not rejected" if stb.success?
puts failures.length == n0 ? "  ok: --batch — order-swap isolation null exact-class, B=1 flag-null, B=4 deterministic, corpus guard" : "  FAIL: batch leg"

# ---- toy#136 (K1): --act / --rope / --schedule ----
# situ-glu + nope each train deterministically and move the curve (the
# swiglu/rope defaults stay byte-null via F0); cosine's per-step lr is
# pinned via step events (formula-checked at both ends); provenance
# carries all three axes.
n0 = failures.length
k1_pairs = [
  ["situ-glu", { "FRANKEN_ACT" => "situ-glu" }],
  ["nope",     { "FRANKEN_NOPE" => "1" }],
]
d6k = run_franken_llama({ "STEPS" => "6" }, nil).lines.select { |l| l.start_with?("step ") }
k1_pairs.each do |name, env|
  a = run_franken_llama({ "STEPS" => "6" }.merge(env), nil).lines.select { |l| l.start_with?("step ") }
  b = run_franken_llama({ "STEPS" => "6" }.merge(env), nil).lines.select { |l| l.start_with?("step ") }
  failures << "k1 #{name}: not deterministic" unless a == b && a.length == 6
  failures << "k1 #{name}: NaN" if a.any? { |l| l[/loss=(\S+)/, 1].to_f.nan? }
  failures << "k1 #{name}: curve identical to default (axis dead)" if a == d6k
end
Dir.mktmpdir("franken_k1_gate") do |dir|
  run_franken_llama({ "STEPS" => "6", "FRANKEN_ACT" => "situ-glu", "FRANKEN_NOPE" => "1",
                      "FRANKEN_SCHEDULE" => "cosine", "FRANKEN_LR" => "0.01" }, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  cfg = evs.first["config"] || {}
  failures << "k1: provenance act #{cfg['act'].inspect}" unless cfg["act"] == "situ-glu"
  failures << "k1: provenance rope #{cfg['rope'].inspect}" unless cfg["rope"] == "nope"
  failures << "k1: provenance schedule #{cfg['schedule'].inspect}" unless cfg["schedule"] == "cosine"
  lrs = evs.select { |e| e["kind"] == "step" }.map { |e| e["lr"] }
  # cosine, no warmup: lr_0 = LR (cos(0)=1); lr_5 = min + 0.5*(LR-min)*(1+cos(pi*5/6))
  min_lr = 0.001
  want_last = min_lr + 0.5 * (0.01 - min_lr) * (1.0 + Math.cos(Math::PI * 5.0 / 6.0))
  failures << "k1: cosine lr[0] #{lrs.first} != 0.01" unless (lrs.first - 0.01).abs < 1.0e-12
  failures << "k1: cosine lr[5] #{lrs.last} != #{want_last}" unless (lrs.last - want_last).abs < 1.0e-9
  failures << "k1: cosine lr not decreasing" unless lrs.each_cons(2).all? { |x, y| y < x }
end
puts failures.length == n0 ? "  ok: K1 axes — situ-glu + nope move the curve deterministically; cosine lr event-pinned; provenance carries act/rope/schedule" : "  FAIL: K1 leg"

# ---- toy#137 (K2b): KDA_LAYERS — a Kimi Delta Attention layer ----
# Absent = all-attention, byte-null (leg 1's F0 fixture pins it). With
# a KDA layer: it TRAINS (the whole layer — projections, low-rank decay
# pair, per-head A_h, full-rank gate, W_o — rides one autodiff sweep),
# deterministically, and the curve differs from all-attention. A dfa
# POLICY on that layer must fail loud (the credit-assignment wiring is
# attention-shaped; "KDA under DFA" is its own K-series question), and
# (the engine also fails loud when both lists claim one layer, but this
# runner exposes KDA_LAYERS only, so that guard is not probed here.)
n0 = failures.length
kda_env = { "STEPS" => "6", "SEED" => "0", "KDA_LAYERS" => "1" }
ka = run_franken_llama(kda_env, nil).lines.select { |l| l.start_with?("step ") }
kb = run_franken_llama(kda_env, nil).lines.select { |l| l.start_with?("step ") }
failures << "kda: not deterministic" unless ka == kb && ka.length == 6
kl = ka.map { |l| l[/loss=(\S+)/, 1].to_f }
failures << "kda: NaN" if kl.any?(&:nan?)
failures << "kda: did not train (#{kl.first} -> #{kl.last})" unless kl.last < kl.first - 0.1
d6kda = run_franken_llama({ "STEPS" => "6", "SEED" => "0" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "kda: curve identical to all-attention (layer kind dead)" if ka == d6kda
_ok, stk = Open3.capture2e({ "STEPS" => "2", "KDA_LAYERS" => "1", "FRANKEN_POLICY" => "chain,dfa",
                             "FRANKEN_B_SEED" => "42" }, RUNNER, chdir: ROOT)
failures << "kda: dfa policy on a KDA layer not rejected" if stk.success?
# (the both-lists guard lives in build_gdn_flags! but is NOT reachable
# from this runner — the franken runner wires KDA_LAYERS only, no
# GDN_LAYERS — so it is asserted where it is reachable, not here.)
Dir.mktmpdir("franken_kda_gate") do |dir|
  run_franken_llama({ "STEPS" => "2", "SEED" => "0", "KDA_LAYERS" => "1" }, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  failures << "kda: provenance kda_layers #{evs.first.dig('config', 'kda_layers').inspect}" unless evs.first.dig("config", "kda_layers") == "1"
end
# toy#137 K2c: ShortConv identity null + the batch guard + the cost
# split. The conv is identity-inited (tap0=1, taps1..3=0), so a
# conv-ON run's STEP-1 loss must BYTE-EQUAL a conv-OFF run's — then
# diverge as the taps train. That is a forward-identity null on a
# mechanism whose whole risk is the shifted-view plumbing.
kc_on  = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "KDA_LAYERS" => "1" }, nil).lines.select { |l| l.start_with?("step ") }
kc_off = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "KDA_LAYERS" => "1", "KDA_CONV" => "0" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "kda-conv: step-1 differs from conv-off (identity init broken)\non: #{kc_on.first}off: #{kc_off.first}" unless kc_on.first == kc_off.first
failures << "kda-conv: curve identical to conv-off (taps never train)" if kc_on == kc_off
_ob3, stb3 = Open3.capture2e({ "STEPS" => "1", "KDA_LAYERS" => "1", "FRANKEN_BATCH" => "4",
                               "CORPUS" => "data/fineweb_gpt2_smoke.bin" }, RUNNER, chdir: ROOT)
failures << "kda: --batch > 1 with KDA layers not rejected" if stb3.success?
Dir.mktmpdir("franken_kda_cost") do |dir|
  FileUtils.mkdir_p(File.join(dir, "weights"))
  run_franken_llama({ "STEPS" => "1", "SEED" => "0", "KDA_LAYERS" => "1" }, dir)
  ck = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)["cost"]
  Dir.mktmpdir("franken_attn_cost") do |d2|
    FileUtils.mkdir_p(File.join(d2, "weights"))
    run_franken_llama({ "STEPS" => "1", "SEED" => "0" }, d2)
    ca = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)["cost"]
    failures << "kda-cost: flops not lower than all-attention (#{ck['flops_per_token']} vs #{ca['flops_per_token']})" unless ck["flops_per_token"] < ca["flops_per_token"]
    failures << "kda-cost: params identical to all-attention (formula not kind-aware)" if ck["total_params"] == ca["total_params"]
  end
end
# toy#138 K3a: the hybrid layer pattern — K3's 3 KDA : 1 global with
# the FINAL layer always global. At deep (L=6): KDA at 0,1,2,4 and
# attention at 3,5. Asserted STRUCTURALLY (cost accounting counts
# layer kinds, so params/flops pin the split) rather than by curve.
hy = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                         "FRANKEN_LAYER_PATTERN" => "hybrid" }, nil).lines.select { |l| l.start_with?("step ") }
hy2 = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                          "FRANKEN_LAYER_PATTERN" => "hybrid" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "hybrid: not deterministic" unless hy == hy2 && hy.length == 3
failures << "hybrid: NaN" if hy.any? { |l| l[/loss=(\S+)/, 1].to_f.nan? }
Dir.mktmpdir("franken_hy") do |dir|
  FileUtils.mkdir_p(File.join(dir, "weights"))
  run_franken_llama({ "STEPS" => "1", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                      "FRANKEN_LAYER_PATTERN" => "hybrid" }, dir)
  hj = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "hybrid: provenance layer_pattern #{hj.dig('config', 'layer_pattern').inspect}" unless hj.dig("config", "layer_pattern") == "hybrid"
  Dir.mktmpdir("franken_hy_ref") do |d2|
    FileUtils.mkdir_p(File.join(d2, "weights"))
    # the same 6-layer shape with an EXPLICIT 0,1,2,4 kda list must give
    # the identical cost split — that is the pattern's contract.
    run_franken_llama({ "STEPS" => "1", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                        "KDA_LAYERS" => "0,1,2,4" }, d2)
    rj = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)
    failures << "hybrid: cost differs from explicit 0,1,2,4 (wrong layer split)" unless hj["cost"] == rj["cost"]
  end
end
_oh, sth = Open3.capture2e({ "STEPS" => "1", "FRANKEN_LAYER_PATTERN" => "hybrid",
                             "KDA_LAYERS" => "0" }, RUNNER, chdir: ROOT)
failures << "hybrid: pattern + explicit kda-layers not rejected" if sth.success?
puts failures.length == n0 ? "  ok: KDA_LAYERS — a KDA layer trains through the engine (#{kl.first.round(3)} -> #{kl.last.round(3)}), deterministic, differs from all-attention; conv identity-null + batch guard + linear-attention cost; hybrid 3:1 pattern == explicit split; dfa-policy rejected" : "  FAIL: KDA leg"

# ---- toy#129 item 2: --no-shadow ----
# THE null (the roadmap's own criterion): applied updates byte-identical
# shadow vs no-shadow at the same policy/seed — the mode changes the
# graph SHAPE (no chain grad-accs on dfa weights), never the numbers.
# Plus: summary says shadow=off, provenance carries shadow=false, and
# the two impossible asks fail loud (align needs the acc; mask modes
# read it).
ns_env = { "STEPS" => "6", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }
sh6 = run_franken_llama(ns_env, nil)
ns6 = run_franken_llama(ns_env.merge("FRANKEN_NO_SHADOW" => "1"), nil)
sh6_c = sh6.lines.select { |l| l.start_with?("step ") }
ns6_c = ns6.lines.select { |l| l.start_with?("step ") }
failures << "no-shadow: applied updates differ from the shadow build\nsh: #{sh6_c.join}ns: #{ns6_c.join}" unless sh6_c == ns6_c && ns6_c.length == 6
failures << "no-shadow: summary does not say shadow=off" unless ns6.include?("shadow=off")
failures << "no-shadow: shadow summary does not say shadow=on" unless sh6.include?("shadow=on")
Dir.mktmpdir("franken_ns_gate") do |dir|
  run_franken_llama({ "STEPS" => "2", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42", "FRANKEN_NO_SHADOW" => "1" }, dir)
  rs0 = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "no-shadow: provenance franken.shadow #{rs0.dig('franken', 'shadow').inspect} (want false)" unless rs0.dig("franken", "shadow") == false
end
_o1, st1 = Open3.capture2e({ "STEPS" => "1", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42", "FRANKEN_ALIGN" => "1", "FRANKEN_NO_SHADOW" => "1" }, RUNNER, chdir: ROOT)
failures << "no-shadow: ALIGN + NO_SHADOW not rejected" if st1.success?
_o2, st2 = Open3.capture2e({ "STEPS" => "1", "FRANKEN_POLICY" => "chain,maskdfa:0.5", "FRANKEN_B_SEED" => "42", "FRANKEN_NO_SHADOW" => "1" }, RUNNER, chdir: ROOT)
failures << "no-shadow: mask mode + NO_SHADOW not rejected" if st2.success?
puts failures.empty? ? "  ok: --no-shadow — applied updates byte-equal shadow (the null), provenance shadow=false, align/mask guards fail loud" : "  FAIL: no-shadow leg"

# ---- toy#129 item 3 (enabling seam): --ckpt-every ----
# Mid-run checkpoints at boundaries + THE null: the write (downloads +
# a fresh plain-storage session, freed) must not disturb the training
# sched — curve byte-equals the no-ckpt run.
ck_base = run_franken_llama({ "STEPS" => "5", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
Dir.mktmpdir("franken_ck_gate") do |dir|
  ck_out = run_franken_llama({ "STEPS" => "5", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42", "FRANKEN_CKPT_EVERY" => "2" }, dir)
  ck_c = ck_out.lines.select { |l| l.start_with?("step ") }
  base_c = ck_base.lines.select { |l| l.start_with?("step ") }
  failures << "ckpt-every: curve differs from no-ckpt run (write disturbs training)" unless ck_c == base_c
  %w[step_2.gguf step_4.gguf step_5.gguf].each do |ck|
    failures << "ckpt-every: missing weights/#{ck}" unless File.file?(File.join(dir, "weights", ck))
  end
end
puts failures.empty? ? "  ok: --ckpt-every — boundary checkpoints written; curve byte-equals no-ckpt (write is sched-null)" : "  FAIL: ckpt-every leg"

# ---- 4. byte-repro ----
r1 = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
r2 = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
failures << "byte-repro: outputs differ" unless r1 == r2
puts "  ok: byte-repro — two policy runs identical" if r1 == r2

if failures.empty?
  puts "GATE PASS [franken-llama]: F0 byte-parity + seed!=0 parity + bundle/provenance/align + dfa-effect + corpus/align-every (toy#122) + shape presets (toy#124) + lr/warmup (toy#126) + no-shadow/pack-header/ckpt-every (toy#129) + batch (toy#133) + K1 axes (toy#136) + KDA layer (toy#137) + byte-repro (toy#112/#113)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken-llama]: #{f}" }
  exit 1
end
