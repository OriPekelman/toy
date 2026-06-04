#!/usr/bin/env ruby
# prep/gpt2_train_engine_gate.rb — byte-exact gate for the GPT-2 engine runner
# (libexec/toy-train-gpt2 → Toy::LLM::Engine::GPT2SeqEngine, the `toy train
# --arch gpt2` compute path). Trains a GPT-2-shape model from scratch on the
# from-scratch corpus and asserts the loss curve is byte-identical to
# prep/fixtures/gpt2_train_engine_baseline.txt. ggml-internal CE is byte-exact
# on aarch64; the runner is deterministic (seeded LCG + fixed corpus line).
# This complements gate-gpt2 (the inline trainer) by gating the ENGINE path.
#
#   ruby prep/gpt2_train_engine_gate.rb            # compare
#   ruby prep/gpt2_train_engine_gate.rb --record   # (re)record
require "open3"
ROOT     = File.expand_path("..", __dir__)
RUNNER   = File.join(ROOT, "libexec", "toy-train-gpt2")
BASELINE = File.join(ROOT, "prep", "fixtures", "gpt2_train_engine_baseline.txt")
RECORD   = ARGV.include?("--record")

bo, bs = Open3.capture2e("make", "libexec/toy-train-gpt2", chdir: ROOT)
unless bs.success? && File.exist?(RUNNER)
  puts "GATE FAIL [gpt2-train-engine]: build failed"; puts bo; exit 1
end
got, rs = Open3.capture2e({ "STEPS" => "10" }, RUNNER, chdir: ROOT)
unless rs.success?
  puts "GATE FAIL [gpt2-train-engine]: runner exited #{rs.exitstatus}"; puts got; exit 1
end
got = got.lines.select { |l| l =~ /\Astep / }.join.strip

if RECORD
  File.write(BASELINE, got + "\n")
  puts "RECORDED [gpt2-train-engine]: #{BASELINE} (#{got.lines.size} steps)"; puts got; exit 0
end
unless File.exist?(BASELINE)
  puts "GATE FAIL [gpt2-train-engine]: no baseline — run with --record"; exit 1
end
want = File.read(BASELINE).strip
# Sanity: loss must actually decrease (not a frozen/zero curve).
first = got.lines.first.to_s[/loss=([0-9.eE+-]+)/, 1].to_f
last  = got.lines.last.to_s[/loss=([0-9.eE+-]+)/, 1].to_f
if got == want && last < first
  puts "GATE PASS [gpt2-train-engine]: loss curve byte-identical + decreasing (#{first.round(3)} → #{last.round(3)})"
  exit 0
else
  puts "GATE FAIL [gpt2-train-engine]: curve diverged or not decreasing"
  puts "  first=#{first} last=#{last}"
  want.lines.zip(got.lines).each_with_index { |(w,g),i| (puts "  line #{i+1}: want #{w.to_s.strip} | got #{g.to_s.strip}") if w != g }
  exit 1
end
