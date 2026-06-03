#!/usr/bin/env ruby
# prep/full_finetune_gate.rb — the 6th realize-gate (F3 full fine-tune).
#
# P2's accepted ceiling left realize_for_full_finetune un-decomposed and
# un-gated. This gate records the CE loss curve from the engine's
# full_finetune realize+train path and re-verifies it byte-for-byte, so the
# lift of the per-block alloc onto TransformerBlock#alloc_full_finetune_f32_weights!
# is provably behavior-preserving (record from the INLINE path FIRST, then lift).
#
# MODEL-GATED: full_finetune loads a REAL model's weights (no random init), so
# this needs data/smollm2-135m-native.gguf — a gitignored dev artifact present
# on gx10 / Mac dev boxes. When absent the gate SKIPs loudly (exit 0) rather
# than failing a fixture-only checkout. Train losses are ggml-internal →
# byte-exact across CPU backends + machines (see the cross-platform gate note),
# so the recorded curve is portable wherever the same model bytes are present.
#
#   ruby prep/full_finetune_gate.rb            # compare to recorded baseline
#   ruby prep/full_finetune_gate.rb --record   # (re)record the baseline

require "open3"

ROOT     = File.expand_path("..", __dir__)
GGUF     = File.join(ROOT, "data", "smollm2-135m-native.gguf")
RUNNER   = File.join(ROOT, "examples", "smoke_full_finetune")
BASELINE = File.join(ROOT, "prep", "fixtures", "full_finetune_baseline.txt")
RECORD   = ARGV.include?("--record")

unless File.exist?(GGUF)
  puts "SKIP [full-finetune]: model absent (#{GGUF})."
  puts "  full_finetune trains REAL weights; this gate needs the gitignored dev GGUF."
  exit 0
end

build_out, build_st = Open3.capture2e("make", "examples/smoke_full_finetune", chdir: ROOT)
unless build_st.success? && File.exist?(RUNNER)
  puts "GATE FAIL [full-finetune]: build failed"
  puts build_out
  exit 1
end

got, run_st = Open3.capture2e(RUNNER, chdir: ROOT)
unless run_st.success?
  puts "GATE FAIL [full-finetune]: runner exited #{run_st.exitstatus}"
  puts got
  exit 1
end
got = got.strip

if RECORD
  File.write(BASELINE, got + "\n")
  puts "RECORDED [full-finetune]: #{BASELINE} (#{got.lines.size} steps) from the inline realize path"
  puts got
  exit 0
end

unless File.exist?(BASELINE)
  puts "GATE FAIL [full-finetune]: no baseline at #{BASELINE} — run with --record first."
  exit 1
end

want = File.read(BASELINE).strip
if got == want
  puts "GATE PASS [full-finetune]: full fine-tune CE curve byte-identical to recorded inline-path baseline"
  exit 0
end

puts "GATE FAIL [full-finetune]: CE curve diverged from baseline"
puts "--- expected ---"
puts want
puts "--- actual ---"
puts got
exit 1
