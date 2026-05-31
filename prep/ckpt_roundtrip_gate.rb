#!/usr/bin/env ruby
# prep/ckpt_roundtrip_gate.rb — deterministic train→infer ROUND-TRIP gate.
#
# Proves that a checkpoint WRITTEN by `toy train from-scratch` is LOADABLE
# and DECODABLE by `toy infer`, and that the generated token IDs are
# BYTE-IDENTICAL to a recorded fixture. This is the discipline check that
# the from-scratch checkpoint is a STANDARD fused-llama GGUF (token_embd
# folded from the projection lens + fused per-head attention), not the
# old training-graph naming that crashed realize_for_mmap.
#
#   ruby prep/ckpt_roundtrip_gate.rb   # exit 0 on byte-for-byte match
#
# DETERMINISM: train --steps 5 --seed 0 is seeded (deterministic weights);
# infer uses a FIXED numeric prompt ("1 2 3", all ids < vocab=627, the
# from-scratch model has NO tokenizer) + greedy argmax (the default,
# Sampler.argmax — no rand/temperature/seed). So the "ids:" line is
# reproducible run-to-run. Do NOT loosen to a "no-NaN" weak check — the
# whole point is byte-identical generated ids.
#
# `toy train` / `toy infer` build their runners themselves (ToyRoot.
# ensure_built), so no separate `make` is needed first — only bin/toy.
# CPU-only: no CUDA arm.

require "open3"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")

unless File.executable?(TOY)
  warn "ckpt_roundtrip_gate: bin/toy not executable: #{TOY}"
  exit 2
end

FIXTURE = File.join(ROOT, "prep", "fixtures", "ckpt_roundtrip_baseline.txt")
unless File.file?(FIXTURE)
  warn "ckpt_roundtrip_gate: missing fixture: #{FIXTURE}"
  exit 2
end

# Recorded fixture body: the single non-comment "ids: …" line.
expected = nil
File.foreach(FIXTURE) do |line|
  next if line.strip.empty? || line.start_with?("#")
  expected = line.chomp if line.start_with?("ids:")
end
if expected.nil?
  warn "ckpt_roundtrip_gate: fixture has no `ids:` body line: #{FIXTURE}"
  exit 2
end

# --- 1. TRAIN (seeded, deterministic) ------------------------------------
out, status = Open3.capture2e(TOY, "train", "from-scratch",
                              "--steps", "5", "--seed", "0", chdir: ROOT)
unless status.success?
  warn "ckpt_roundtrip_gate: `toy train` exited #{status.exitstatus}:"
  warn out.lines.last(20).join
  exit 1
end

# --- 2. LOCATE the checkpoint --------------------------------------------
run_line = out.lines.find { |l| l.start_with?("run ") }
unless run_line && run_line =~ /\Arun (\S+) .* (\/\S+)\s*\z/
  warn "ckpt_roundtrip_gate: could not parse run dir: #{run_line.inspect}"
  exit 1
end
run_dir = $2.strip
ckpt    = File.join(run_dir, "weights", "step_5.gguf")
unless File.file?(ckpt)
  warn "ckpt_roundtrip_gate: checkpoint missing: #{ckpt}"
  exit 1
end

# --- 3. INFER (numeric ids, greedy, deterministic) -----------------------
out2, status2 = Open3.capture2e(TOY, "infer", ckpt,
                                "--prompt-ids", "1 2 3", "--n", "5", chdir: ROOT)
unless status2.success?
  warn "ckpt_roundtrip_gate: `toy infer` exited #{status2.exitstatus}:"
  warn out2.lines.last(20).join
  exit 1
end

ids_line = out2.lines.map(&:chomp).find { |l| l.start_with?("ids:") }
if ids_line.nil?
  warn "ckpt_roundtrip_gate: `toy infer` produced no `ids:` line; output was:"
  warn out2.lines.last(20).join
  exit 1
end

# --- 4. BYTE-COMPARE -----------------------------------------------------
puts "fixture : #{File.basename(FIXTURE)}"
puts "expected: #{expected.inspect}"
puts "actual  : #{ids_line.inspect}"
if ids_line == expected
  puts "GATE PASS [ckpt-roundtrip]: train→infer generated ids byte-identical"
  exit 0
else
  warn "GATE FAIL [ckpt-roundtrip]: generated ids diverged from recorded baseline"
  warn "  expected: #{expected.inspect}"
  warn "  actual  : #{ids_line.inspect}"
  exit 1
end
