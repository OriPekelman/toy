#!/usr/bin/env ruby
# prep/gpt2_train_gate.rb — byte-exact gate for the minimal GPT-2 trainer.
#
# Records the CE loss curve from prep/gpt2_train_min.rb (the inline GPT-2
# forward+CE+backward+AdamW proof — wte+wpe embeddings, composite LayerNorm,
# single-head causal attention, GELU FFN, tied output) and re-verifies it
# byte-for-byte. This is the "record-from-inline-first" reference for the
# eventual engine-integrated `toy train --arch gpt2`: it pins the exact loss
# trajectory the two vendored backward kernels (ggml_gelu_back, ggml_norm_back;
# vendor-patches/0007) produce, so a later refactor onto the engine is provably
# behavior-preserving.
#
# Random-init (no model file) — always runs. Train losses are ggml-internal
# CE → byte-exact across CPU backends + machines on aarch64 (the gx10-canonical
# float-gate note); the curve is portable wherever the same ggml bytes are
# built. The trainer is deterministic (seeded LCG weights + fixed synthetic
# data); verified by running twice.
#
#   ruby prep/gpt2_train_gate.rb            # compare to recorded baseline
#   ruby prep/gpt2_train_gate.rb --record   # (re)record the baseline

require "open3"

ROOT     = File.expand_path("..", __dir__)
RUNNER   = File.join(ROOT, "libexec", "gpt2-train-min")
BASELINE = File.join(ROOT, "prep", "fixtures", "gpt2_train_baseline.txt")
RECORD   = ARGV.include?("--record")

build_out, build_st = Open3.capture2e("make", "gpt2-train-min", chdir: ROOT)
unless build_st.success? && File.exist?(RUNNER)
  puts "GATE FAIL [gpt2-train]: build failed"
  puts build_out
  exit 1
end

got, run_st = Open3.capture2e(RUNNER, chdir: ROOT)
unless run_st.success?
  puts "GATE FAIL [gpt2-train]: runner exited #{run_st.exitstatus}"
  puts got
  exit 1
end
# Keep only the deterministic curve lines (step CE + summary); drop any
# incidental output so the fixture is the trajectory, nothing else.
got = got.lines.select { |l| l =~ /\A(step|initial|final|ratio|VERDICT)/ }.join.strip

if RECORD
  File.write(BASELINE, got + "\n")
  puts "RECORDED [gpt2-train]: #{BASELINE} (#{got.lines.count { |l| l.start_with?('step') }} step lines)"
  puts got
  exit 0
end

unless File.exist?(BASELINE)
  puts "GATE FAIL [gpt2-train]: no baseline at #{BASELINE} — run with --record first"
  exit 1
end
want = File.read(BASELINE).strip

if got == want
  puts "GATE PASS [gpt2-train]: CE curve byte-identical to baseline"
  puts "  " + got.lines.last.strip
  exit 0
else
  puts "GATE FAIL [gpt2-train]: CE curve diverged from #{BASELINE}"
  want.lines.zip(got.lines).each_with_index do |(w, g), i|
    next if w == g
    puts "  line #{i + 1}:"
    puts "    want: #{w.to_s.strip}"
    puts "    got:  #{g.to_s.strip}"
  end
  exit 1
end
