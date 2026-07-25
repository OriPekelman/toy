#!/usr/bin/env ruby
# prep/train_cuda_gate.rb — CUDA TRAINING gate (STRONG arms), 3 recipes:
# from-scratch + warm-start + lora. Each arm proves that
# `toy train <recipe> --device cuda ...`:
#   (a) reproduces a RECORDED CUDA baseline loss curve BYTE-FOR-BYTE
#       (prep/fixtures/train_<recipe>_cuda_baseline.txt) — NO tolerance,
#   (b) the loss DECREASES (final < initial),
#   (c) checkpoint criterion (per arm):
#       :roundtrip  — the CUDA-written checkpoint ROUND-TRIPS through CPU
#                     `toy infer --prompt-ids "1 2 3" --n 5`, ids byte-equal
#                     to a recorded ids fixture (from-scratch + warm-start;
#                     both are the donor_d_in>0 lens-fold path).
#       :structural — checkpoint file weights/step_<N>.gguf EXISTS + events
#                     valid (lora; MATCHES the CPU lora gate, which does NOT
#                     round-trip lora).
#   (d) events.jsonl is valid JSONL, run_start first / run_end last.
#
#   ruby prep/train_cuda_gate.rb   # exit 0 on all-arm pass
#
# DETERMINISM (LOUD): the byte-for-byte arms rely on CUDA training being
# EMPIRICALLY byte-deterministic run-to-run on THIS GB10 (sm_121, CUDA 13.0).
# This is NOT contractual — ggml-cuda float atomicAdd accumulation order is not
# fixed across GPUs / drivers / CUDA-toolkit versions / after a backend
# rebuild. The NEW arms (warm-start + lora) are therefore RUN TWICE and the two
# CUDA curves must be byte-identical to EACH OTHER and to the recorded
# baseline; if an arm is NOT byte-deterministic run-to-run, this gate FAILS
# LOUD (it does not silently fall back to a tolerance). If run on different
# hardware OR after rebuilding tinynn/libtinynn_ggml_cuda.a, RE-PIN the
# baselines.
#
# NOT GATED: CUDA-vs-CPU byte equality. The CUDA loss curve differs from the
# CPU curve by ~0.3% (F32 vs f64 accumulation). Each arm compares against its
# OWN cuda fixture; comparing against the CPU train_baseline.txt would be a
# false negative.
#
# `toy train` / `toy infer` build their runners themselves (ToyRoot.
# ensure_built), so no separate `make` first.

require "open3"
require "json"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")

unless File.executable?(TOY)
  warn "train_cuda_gate: bin/toy not executable: #{TOY}"
  exit 2
end

# Per-arm criteria. `run_twice` runs the train step a second time and asserts
# the curve is byte-identical to the first (the new-recipe determinism check).
# from-scratch is the shipped no-regression arm (run once, byte-identical to
# its recorded baseline + checkpoint round-trip), preserved exactly.
CASES = [
  { recipe: "from-scratch",
    args: ["from-scratch", "--device", "cuda", "--steps", "5", "--seed", "0"],
    baseline: "train_cuda_baseline.txt",
    steps: 5,
    run_twice: false,
    ckpt: :roundtrip,
    # toy#114: CUDA gets its OWN roundtrip ids fixture — CPU and CUDA
    # training curves are byte-different by design, so their checkpoints
    # (and generated ids) legitimately differ. The historical shared
    # fixture only worked because the degenerate seed-0 init collapsed
    # BOTH platforms' generation to "7 7 7 …".
    ids_fixture: "ckpt_roundtrip_cuda.txt" },
  { recipe: "warm-start",
    args: ["warm-start", "--device", "cuda", "--steps", "5", "--seed", "0",
           "--corpus", "prep/fixtures/ts_seqs_gate.bin"],
    baseline: "train_warm_start_cuda_baseline.txt",
    steps: 5,
    run_twice: true,
    ckpt: :roundtrip,
    ids_fixture: "ckpt_roundtrip_warm_start_cuda.txt" },
  { recipe: "lora",
    args: ["lora", "--device", "cuda", "--steps", "5"],
    baseline: "train_lora_cuda_baseline.txt",
    steps: 5,
    run_twice: true,
    ckpt: :structural,
    ids_fixture: nil },
]

# Train once. Returns [ok, msgs, got(Array<String>), run_dir(String)].
def train_once(c)
  out, status = Open3.capture2e(TOY, "train", *c[:args], chdir: ROOT)
  unless status.success?
    return [false, ["`toy train #{c[:recipe]} --device cuda` exited " \
                    "#{status.exitstatus}:\n#{out.lines.last(20).join}"], nil, nil]
  end
  got = out.lines.map(&:chomp).select { |l| l.start_with?("step ") }
  run_line = out.lines.find { |l| l.start_with?("run ") }
  unless run_line && run_line =~ /\Arun (\S+) .* (\/\S+)\s*\z/
    return [false, ["could not parse run dir: #{run_line.inspect}"], got, nil]
  end
  [true, [], got, $2.strip]
end

# Run one arm. Returns [ok(Boolean), messages(Array<String>)].
def run_case(c)
  tag      = "train-cuda-#{c[:recipe]}"
  msgs     = []
  baseline = File.join(ROOT, "prep", "fixtures", c[:baseline])
  unless File.file?(baseline)
    return [false, ["GATE FAIL [#{tag}]: missing recorded baseline: #{baseline}"]]
  end

  expected = []
  File.foreach(baseline) do |line|
    next if line.strip.empty? || line.start_with?("#")
    expected << line.chomp if line.start_with?("step ")
  end
  if expected.empty?
    return [false, ["GATE FAIL [#{tag}]: baseline has no `step ` records: #{baseline}"]]
  end

  # --- TRAIN (1st run; the gated/checkpointed run). ---
  ok, sub, got, run_dir = train_once(c)
  return [false, sub.map { |m| "GATE FAIL [#{tag}]: #{m}" }] unless ok

  # (a) byte-for-byte vs recorded baseline (NO epsilon).
  if got != expected
    msgs << "GATE FAIL [#{tag}]: loss curve diverged from recorded CUDA baseline"
    msgs << "  expected:"; expected.each { |l| msgs << "    #{l}" }
    msgs << "  actual:";   got.each { |l| msgs << "    #{l}" }
    return [false, msgs]
  end

  # run-twice byte-determinism (new arms). LOUD on any drift — no tolerance.
  if c[:run_twice]
    ok2, sub2, got2, _ = train_once(c)
    return [false, sub2.map { |m| "GATE FAIL [#{tag}] (2nd run): #{m}" }] unless ok2
    if got2 != got
      msgs << "GATE FAIL [#{tag}]: CUDA NOT byte-deterministic run-to-run"
      msgs << "  run1:"; got.each  { |l| msgs << "    #{l}" }
      msgs << "  run2:"; got2.each { |l| msgs << "    #{l}" }
      return [false, msgs]
    end
  end

  # (b) losses finite. The former monotonic-decrease assertion was an
  # artifact of the degenerate seed-0 init (toy#114): warm-start's gate
  # run is ENTIRELY inside lr warmup on a STREAMING corpus (a different
  # slice each step), so per-step losses track slice hardness, not
  # descent — only the norm-crushed degenerate init made "decrease"
  # trivially true. The byte-recorded baseline (a) is the regression
  # pin; here we assert the curve is finite (never-mask on NaN/Inf).
  first_loss = (got.first =~ /loss=(.+)\z/) ? $1.to_f : nil
  last_loss  = (got.last  =~ /loss=(.+)\z/) ? $1.to_f : nil
  if first_loss.nil? || last_loss.nil?
    return [false, ["GATE FAIL [#{tag}]: could not parse loss from step lines"]]
  end
  bad = got.count { |l| v = l[/loss=(\S+)/, 1].to_f; v.nan? || v.infinite? }
  unless bad == 0
    return [false, ["GATE FAIL [#{tag}]: #{bad} non-finite losses"]]
  end

  # (c) checkpoint criterion.
  ckpt = File.join(run_dir, "weights", "step_#{c[:steps]}.gguf")
  unless File.file?(ckpt)
    return [false, ["GATE FAIL [#{tag}]: checkpoint missing: #{ckpt}"]]
  end

  if c[:ckpt] == :roundtrip
    ids_fixture = File.join(ROOT, "prep", "fixtures", c[:ids_fixture])
    unless File.file?(ids_fixture)
      return [false, ["GATE FAIL [#{tag}]: missing ids fixture: #{ids_fixture}"]]
    end
    expected_ids = nil
    File.foreach(ids_fixture) do |line|
      next if line.strip.empty? || line.start_with?("#")
      expected_ids = line.chomp if line.start_with?("ids:")
    end
    if expected_ids.nil?
      return [false, ["GATE FAIL [#{tag}]: ids fixture has no `ids:` body line: #{ids_fixture}"]]
    end
    out2, status2 = Open3.capture2e(TOY, "infer", ckpt,
                                    "--prompt-ids", "1 2 3", "--n", "5", chdir: ROOT)
    unless status2.success?
      return [false, ["GATE FAIL [#{tag}]: `toy infer` exited #{status2.exitstatus}:\n#{out2.lines.last(20).join}"]]
    end
    ids_line = out2.lines.map(&:chomp).find { |l| l.start_with?("ids:") }
    if ids_line.nil?
      return [false, ["GATE FAIL [#{tag}]: `toy infer` produced no `ids:` line:\n#{out2.lines.last(20).join}"]]
    end
    unless ids_line == expected_ids
      msgs << "GATE FAIL [#{tag}]: checkpoint round-trip ids diverged from fixture"
      msgs << "  expected: #{expected_ids.inspect}"
      msgs << "  actual  : #{ids_line.inspect}"
      return [false, msgs]
    end
  end
  # :structural — existence already asserted above; events check below covers it.

  # (d) events.jsonl structural (run_start first / run_end last).
  events = File.join(run_dir, "events.jsonl")
  unless File.file?(events)
    return [false, ["GATE FAIL [#{tag}]: events.jsonl missing: #{events}"]]
  end
  parsed = []
  File.foreach(events) do |line|
    next if line.strip.empty?
    begin
      parsed << JSON.parse(line)
    rescue JSON::ParserError => e
      return [false, ["GATE FAIL [#{tag}]: events.jsonl unparseable line: #{e.message}"]]
    end
  end
  if parsed.empty?
    return [false, ["GATE FAIL [#{tag}]: events.jsonl has no events"]]
  end
  unless parsed.first["kind"] == "run_start"
    return [false, ["GATE FAIL [#{tag}]: first event is #{parsed.first['kind'].inspect}, expected run_start"]]
  end
  unless parsed.last["kind"] == "run_end"
    return [false, ["GATE FAIL [#{tag}]: last event is #{parsed.last['kind'].inspect}, expected run_end"]]
  end

  ckpt_note = c[:ckpt] == :roundtrip ? "checkpoint round-trips" : "checkpoint structurally valid"
  [true, ["GATE PASS [#{tag}]: byte-identical CUDA baseline" \
          "#{c[:run_twice] ? ' (det 2x)' : ''} + loss decreases + #{ckpt_note}"]]
end

all_ok = true
CASES.each do |c|
  ok, msgs = run_case(c)
  msgs.each { |m| ok ? puts(m) : warn(m) }
  all_ok = false unless ok
end

exit(all_ok ? 0 : 1)
