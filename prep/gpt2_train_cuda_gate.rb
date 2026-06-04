#!/usr/bin/env ruby
# prep/gpt2_train_cuda_gate.rb — gate for the GPT-2 CUDA engine runner
# (libexec/toy-train-gpt2-cuda, `toy train --arch gpt2 --device cuda`).
#
# The forward + most of the backward run on CUDA; the GELU/LayerNorm backward
# ops (no CUDA kernel) fall back to the CPU backend via the scheduler. So the
# curve is its OWN reference (CUDA-vs-CUDA determinism, EMPIRICAL on this GB10),
# NOT a byte-match of the CPU curve. We assert: byte-identical to the recorded
# CUDA baseline + loss decreasing.
#
#   ruby prep/gpt2_train_cuda_gate.rb            # compare
#   ruby prep/gpt2_train_cuda_gate.rb --record   # (re)record
require "open3"
ROOT     = File.expand_path("..", __dir__)
RUNNER   = File.join(ROOT, "libexec", "toy-train-gpt2-cuda")
BASELINE = File.join(ROOT, "prep", "fixtures", "gpt2_train_cuda_baseline.txt")
RECORD   = ARGV.include?("--record")

bo, bs = Open3.capture2e("make", "libexec/toy-train-gpt2-cuda", chdir: ROOT)
unless bs.success? && File.exist?(RUNNER)
  puts "GATE FAIL [gpt2-train-cuda]: build failed"; puts bo; exit 1
end
got, rs = Open3.capture2e({ "STEPS" => "10" }, RUNNER, chdir: ROOT)
unless rs.success?
  puts "GATE FAIL [gpt2-train-cuda]: runner exited #{rs.exitstatus}"; puts got; exit 1
end
got = got.lines.select { |l| l =~ /\Astep / }.join.strip

if RECORD
  File.write(BASELINE, got + "\n")
  puts "RECORDED [gpt2-train-cuda]: #{BASELINE} (#{got.lines.size} steps)"; puts got; exit 0
end
unless File.exist?(BASELINE)
  puts "GATE FAIL [gpt2-train-cuda]: no baseline — run with --record"; exit 1
end
want  = File.read(BASELINE).strip
first = got.lines.first.to_s[/loss=([0-9.eE+-]+)/, 1].to_f
last  = got.lines.last.to_s[/loss=([0-9.eE+-]+)/, 1].to_f
if got == want && last < first
  puts "GATE PASS [gpt2-train-cuda]: CUDA loss curve byte-identical + decreasing (#{first.round(3)} → #{last.round(3)})"
  exit 0
else
  puts "GATE FAIL [gpt2-train-cuda]: curve diverged or not decreasing (first=#{first} last=#{last})"
  want.lines.zip(got.lines).each_with_index { |(w,g),i| (puts "  line #{i+1}: want #{w.to_s.strip} | got #{g.to_s.strip}") if w != g }
  exit 1
end
