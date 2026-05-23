# Bench: inference token throughput.
# SmolLM2-135M, generates N tokens after a 4-token prompt prefill;
# reports tokens/sec on the generation-only phase (excluding prefill
# and warm-up first-step graph realize).
#
# Output format:
#   BENCH infer_toks_per_sec <float>
#   BENCH infer_step_ms <float>

require_relative "../lib/arch"
require_relative "../lib/transformer_lm"

GGUF  = ENV["GGUF"]  || "data/smollm2-135m-f32.gguf"
N_NEW = (ENV["N_NEW"] || "32").to_i

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "BENCH infer_toks_per_sec 0"
  puts "BENCH infer_step_ms 0"
  exit 1
end

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)

prompt = [6403, 1980, 253, 655, 28]

# Warm-up: 4 tokens to absorb graph realize cost.
lm.generate(prompt, 4)

t0 = Time.now
ids = lm.generate(prompt, N_NEW)
elapsed = (Time.now - t0).to_f

new_toks = ids.length - prompt.length
if new_toks <= 0; new_toks = N_NEW; end
puts "BENCH infer_toks_per_sec " + (new_toks.to_f / elapsed).to_s
puts "BENCH infer_step_ms " + (elapsed * 1000.0 / new_toks.to_f).to_s
