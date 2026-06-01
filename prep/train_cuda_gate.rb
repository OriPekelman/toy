#!/usr/bin/env ruby
# prep/train_cuda_gate.rb — CUDA from-scratch TRAINING gate (STRONG arm).
#
# Proves that `toy train from-scratch --device cuda --steps 5 --seed 0`:
#   (a) reproduces a RECORDED CUDA baseline loss curve BYTE-FOR-BYTE
#       (prep/fixtures/train_cuda_baseline.txt) — NO tolerance/epsilon,
#   (b) the loss DECREASES (final < initial),
#   (c) the CUDA-written checkpoint ROUND-TRIPS through CPU `toy infer`,
#       producing token ids byte-equal to the SHARED fixture
#       prep/fixtures/ckpt_roundtrip_baseline.txt (greedy argmax discretizes
#       the F32/f64 low-bit diff, so the ids fixture is shared with the CPU
#       gate).
#
#   ruby prep/train_cuda_gate.rb   # exit 0 on all-pass
#
# DETERMINISM (LOUD): the byte-for-byte arm relies on CUDA from-scratch
# training being EMPIRICALLY byte-deterministic run-to-run on THIS GB10
# (sm_121, CUDA 13.0). This is NOT contractual — ggml-cuda float atomicAdd
# accumulation order is not fixed across GPUs / drivers / CUDA-toolkit
# versions / after a backend rebuild. If run on different hardware OR after
# rebuilding tinynn/libtinynn_ggml_cuda.a, RE-PIN train_cuda_baseline.txt.
#
# NOT GATED: CUDA-vs-CPU byte equality. The CUDA loss curve differs from the
# CPU curve by ~0.3% (F32 vs f64 accumulation, docs/archive/
# p1-grad-bisection-2026-05-22.md). This gate has its OWN fixture; comparing
# against the CPU train_baseline.txt would be a false negative.
#
# `toy train` / `toy infer` build their runners themselves (ToyRoot.
# ensure_built — `make libexec/toy-train-cuda`), so no separate `make` first.

require "open3"
require "json"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")

unless File.executable?(TOY)
  warn "train_cuda_gate: bin/toy not executable: #{TOY}"
  exit 2
end

BASELINE = File.join(ROOT, "prep", "fixtures", "train_cuda_baseline.txt")
unless File.file?(BASELINE)
  warn "train_cuda_gate: missing recorded baseline: #{BASELINE}"
  exit 2
end

# Load the recorded baseline: the bare "step N: loss=…" lines, in order.
expected = []
File.foreach(BASELINE) do |line|
  next if line.strip.empty? || line.start_with?("#")
  expected << line.chomp if line.start_with?("step ")
end
if expected.empty?
  warn "train_cuda_gate: baseline has no `step ` records: #{BASELINE}"
  exit 2
end

IDS_FIXTURE = File.join(ROOT, "prep", "fixtures", "ckpt_roundtrip_baseline.txt")
unless File.file?(IDS_FIXTURE)
  warn "train_cuda_gate: missing ids fixture: #{IDS_FIXTURE}"
  exit 2
end
expected_ids = nil
File.foreach(IDS_FIXTURE) do |line|
  next if line.strip.empty? || line.start_with?("#")
  expected_ids = line.chomp if line.start_with?("ids:")
end
if expected_ids.nil?
  warn "train_cuda_gate: ids fixture has no `ids:` body line: #{IDS_FIXTURE}"
  exit 2
end

# --- 1. TRAIN on CUDA (seeded, deterministic) ----------------------------
out, status = Open3.capture2e(TOY, "train", "from-scratch", "--device", "cuda",
                              "--steps", "5", "--seed", "0", chdir: ROOT)
unless status.success?
  warn "train_cuda_gate: `toy train --device cuda` exited #{status.exitstatus}:"
  warn out.lines.last(20).join
  exit 1
end

# --- 2(a). STRONG byte arm: loss curve byte-for-byte ---------------------
got = out.lines.map(&:chomp).select { |l| l.start_with?("step ") }
if got != expected
  warn "GATE FAIL [train-cuda]: loss curve diverged from recorded CUDA baseline"
  warn "  expected:"
  expected.each { |l| warn "    #{l}" }
  warn "  actual:"
  got.each { |l| warn "    #{l}" }
  exit 1
end

# --- 2(b). loss-decrease: net-new honest invariant -----------------------
first_loss = (got.first =~ /loss=(.+)\z/) ? $1.to_f : nil
last_loss  = (got.last  =~ /loss=(.+)\z/) ? $1.to_f : nil
if first_loss.nil? || last_loss.nil?
  warn "GATE FAIL [train-cuda]: could not parse loss from step lines"
  exit 1
end
unless last_loss < first_loss
  warn "GATE FAIL [train-cuda]: loss did not decrease (initial=#{first_loss}, final=#{last_loss})"
  exit 1
end

# --- 3(c). CHECKPOINT round-trip through CPU `toy infer` -----------------
run_line = out.lines.find { |l| l.start_with?("run ") }
unless run_line && run_line =~ /\Arun (\S+) .* (\/\S+)\s*\z/
  warn "train_cuda_gate: could not parse run dir: #{run_line.inspect}"
  exit 1
end
run_dir = $2.strip
ckpt    = File.join(run_dir, "weights", "step_5.gguf")
unless File.file?(ckpt)
  warn "train_cuda_gate: checkpoint missing: #{ckpt}"
  exit 1
end

out2, status2 = Open3.capture2e(TOY, "infer", ckpt,
                                "--prompt-ids", "1 2 3", "--n", "5", chdir: ROOT)
unless status2.success?
  warn "train_cuda_gate: `toy infer` exited #{status2.exitstatus}:"
  warn out2.lines.last(20).join
  exit 1
end

ids_line = out2.lines.map(&:chomp).find { |l| l.start_with?("ids:") }
if ids_line.nil?
  warn "train_cuda_gate: `toy infer` produced no `ids:` line; output was:"
  warn out2.lines.last(20).join
  exit 1
end
unless ids_line == expected_ids
  warn "GATE FAIL [train-cuda]: checkpoint round-trip ids diverged from shared fixture"
  warn "  expected: #{expected_ids.inspect}"
  warn "  actual  : #{ids_line.inspect}"
  exit 1
end

# --- 4. STRUCTURAL: events.jsonl valid JSONL (run_start first/run_end last)
events = File.join(run_dir, "events.jsonl")
if File.file?(events)
  parsed = []
  File.foreach(events) do |line|
    next if line.strip.empty?
    begin
      parsed << JSON.parse(line)
    rescue JSON::ParserError => e
      warn "GATE FAIL [train-cuda]: events.jsonl unparseable line: #{e.message}"
      exit 1
    end
  end
  if parsed.empty?
    warn "GATE FAIL [train-cuda]: events.jsonl has no events"
    exit 1
  end
  unless parsed.first["kind"] == "run_start"
    warn "GATE FAIL [train-cuda]: first event is #{parsed.first['kind'].inspect}, expected run_start"
    exit 1
  end
  unless parsed.last["kind"] == "run_end"
    warn "GATE FAIL [train-cuda]: last event is #{parsed.last['kind'].inspect}, expected run_end"
    exit 1
  end
else
  warn "GATE FAIL [train-cuda]: events.jsonl missing: #{events}"
  exit 1
end

puts "GATE PASS [train-cuda]: byte-identical CUDA baseline + loss decreases + checkpoint round-trips"
exit 0
