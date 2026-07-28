#!/usr/bin/env ruby
# prep/eval_ce_gate.rb — toy#130 gates for the CE-over-pack evaluator
# (libexec/toy-eval-ce, `toy eval ce`).
#
#   1. LADDER SANITY: train a 5-step franken checkpoint on the GPT-2
#      smoke pack, eval it on a held-out offset — CE lands in the
#      ln(50257)-neighborhood band (an untrained-but-finite model).
#   2. DETERMINISM: two identical evals — byte-equal eval_ce lines.
#   3. BUNDLE: run_start(name=eval-ce, config) + eval event + run_end.
#   4. TOYC VOCAB MISMATCH: a pack whose header disagrees with the
#      model's vocab is REJECTED (the toy#129 header contract).
#   5. OOB GUARD: a headerless pack with an out-of-range token id is
#      REJECTED (would read past the embed table).
#   6. EOF: an offset past the pack end → zero windows → exit 1.
#
# Needs data/fineweb_gpt2_smoke.bin (ungenerated in git per the
# toy#123 precedent) — the abort below names the generation command.

ROOT    = File.expand_path("..", __dir__)
EVAL_CE = File.join(ROOT, "libexec", "toy-eval-ce")
TRAINER = File.join(ROOT, "libexec", "toy-train-franken-llama")
SMOKE   = "data/fineweb_gpt2_smoke.bin"

require "open3"
require "json"
require "tmpdir"
require "fileutils"

abort "eval_ce_gate: #{SMOKE} missing — generate it: uv run prep/pretokenize_pack.py --tokens 200_000 --out #{SMOKE}" unless File.file?(File.join(ROOT, SMOKE))

def run_eval(env)
  out, st = Open3.capture2e(env, EVAL_CE, chdir: ROOT)
  [out, st]
end

failures = []

Dir.mktmpdir("eval_ce_gate") do |dir|
  FileUtils.mkdir_p(File.join(dir, "weights"))
  t_out, t_st = Open3.capture2e(
    { "STEPS" => "5", "SEED" => "0", "CORPUS" => SMOKE,
      "TAO_RUN_DIR" => dir, "TOY_RUN_ID" => "eval-ce-gate" },
    TRAINER, chdir: ROOT)
  abort "eval_ce_gate: trainer failed:\n#{t_out.lines.last(8).join}" unless t_st.success?
  ckpt = File.join(dir, "weights", "step_5.gguf")
  abort "eval_ce_gate: trainer wrote no #{ckpt}" unless File.file?(ckpt)

  # ---- 1 + 2. sanity band + determinism ----
  ev_env = { "GGUF" => ckpt, "PACK" => SMOKE, "CONTEXT" => "32",
             "EVAL_TOKENS" => "512", "EVAL_OFFSET" => "100000" }
  o1, s1 = run_eval(ev_env)
  o2, s2 = run_eval(ev_env)
  l1 = o1.lines.select { |l| l.start_with?("eval_ce:") }
  l2 = o2.lines.select { |l| l.start_with?("eval_ce:") }
  failures << "eval: runner failed\n#{o1.lines.last(5).join}" unless s1.success? && s2.success?
  failures << "eval: not deterministic\n1: #{l1.join}2: #{l2.join}" unless l1 == l2 && l1.length == 1
  if l1.length == 1
    ce = l1.first[/ce=(\S+)/, 1].to_f
    failures << "eval: ce #{ce} outside the sanity band (5, 12.5)" unless ce > 5.0 && ce < 12.5
    failures << "eval: wrong window count #{l1.first}" unless l1.first.include?("windows=16 tokens=512")
  end
  puts failures.empty? ? "  ok: 5-step checkpoint evals deterministically (#{l1.first.to_s.strip})" : "  FAIL: sanity/determinism"

  # ---- 3. bundle ----
  n0 = failures.length
  Dir.mktmpdir("eval_ce_bundle") do |edir|
    run_eval(ev_env.merge("TAO_RUN_DIR" => edir, "TOY_RUN_ID" => "ce-gate"))
    evs = File.readlines(File.join(edir, "events.jsonl")).map { |l| JSON.parse(l) }
    rs = evs.first || {}
    failures << "bundle: first event not run_start/eval-ce" unless rs["kind"] == "run_start" && rs["name"] == "eval-ce"
    failures << "bundle: run_id lost" unless rs["run_id"] == "ce-gate"
    failures << "bundle: config missing" unless rs.dig("config", "context") == 32 && rs.dig("config", "vocab") == 50257
    ev = evs.find { |e| e["kind"] == "eval" }
    failures << "bundle: no eval event" if ev.nil?
    failures << "bundle: eval event malformed" unless ev && ev["loss"].is_a?(Numeric) && ev["windows"] == 16
    failures << "bundle: last event not run_end" unless evs.last && evs.last["kind"] == "run_end"
    puts failures.length == n0 ? "  ok: bundle — run_start(config) + eval + run_end" : "  FAIL: bundle"
  end

  # ---- 4. TOYC vocab mismatch ----
  n0 = failures.length
  bad_pack = File.join(dir, "bad_vocab.bin")
  File.binwrite(bad_pack, "TOYC" + [1, 999, 0].pack("L<3") + [1, 2, 3, 4] .pack("l<4") * 64)
  _o3, s3 = run_eval({ "GGUF" => ckpt, "PACK" => bad_pack, "CONTEXT" => "32", "EVAL_TOKENS" => "64" })
  failures << "mismatch: TOYC vocab 999 vs model 50257 not rejected" if s3.success?
  puts failures.length == n0 ? "  ok: TOYC vocab mismatch rejected" : "  FAIL: mismatch"

  # ---- 5. OOB token guard (headerless) ----
  n0 = failures.length
  oob_pack = File.join(dir, "oob.bin")
  # 64 tokens; id 60000 sits INSIDE window 0 (idx 10) so the guard must see it.
  File.binwrite(oob_pack, ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 60000] + [4] * 53).pack("l<*"))
  _o4, s4 = run_eval({ "GGUF" => ckpt, "PACK" => oob_pack, "CONTEXT" => "32", "EVAL_TOKENS" => "64" })
  failures << "oob: token id 60000 not rejected" if s4.success?
  puts failures.length == n0 ? "  ok: OOB token id rejected (headerless pack)" : "  FAIL: oob"

  # ---- 6. offset past EOF → zero windows → fail ----
  n0 = failures.length
  _o5, s5 = run_eval(ev_env.merge("EVAL_OFFSET" => "300000"))
  failures << "eof: offset past pack end not rejected" if s5.success?
  puts failures.length == n0 ? "  ok: offset past EOF fails loud (zero windows)" : "  FAIL: eof"
end

if failures.empty?
  puts "GATE PASS [eval-ce]: sanity band + determinism + bundle + TOYC-mismatch + OOB + EOF guards (toy#130)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [eval-ce]: #{f}" }
  exit 1
end
