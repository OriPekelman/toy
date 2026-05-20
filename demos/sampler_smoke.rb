# Sampler smoke — runs ONE configured generation. Parameterize via ENV
# to test different sampling configs without rebuilding.
#
# Defaults: SmolLM2-135M, T=0.7, top_k=40, seed=42, 6 new tokens.
#
# Other configs to try (each as a separate run; multi-config-per-binary
# currently crashes — probably exhausts a Spinel-side resource):
#   MODE=greedy   ./demos/sampler_smoke
#   MODE=temp    ./demos/sampler_smoke         # temperature only
#   MODE=topk    ./demos/sampler_smoke         # +top_k
#   MODE=topp    ./demos/sampler_smoke         # +top_p (small vocab only)
#   SEED=123     ./demos/sampler_smoke         # different seed

require_relative "../lib/arch"
require_relative "../lib/transformer_lm"
require_relative "../lib/sampler"

GGUF = ENV["GGUF"] || "data/smollm2-135m-native.gguf"
MODE = ENV["MODE"] || "topk"
SEED = (ENV["SEED"] || "42").to_i
N_NEW = (ENV["N_NEW"] || "8").to_i

arch = Arch.from_gguf(GGUF)
lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)
prompt = [15155, 2917, 253, 1429]

puts "mode=" + MODE + " seed=" + SEED.to_s + " vocab=" + arch.vocab_size.to_s

cfg = SamplerConfig.new
cfg.seed = SEED
if MODE == "greedy"
  cfg.temperature = 0.0   # → argmax via Sampler.pick
elsif MODE == "topk"
  cfg.temperature = 0.7
  cfg.top_k = 40
elsif MODE == "topp"
  cfg.temperature = 0.7
  cfg.top_p = 0.9
elsif MODE == "all"
  cfg.temperature = 0.7
  cfg.top_k = 40
  cfg.top_p = 0.9
else
  cfg.temperature = 0.7
end
ids = lm.generate(prompt, N_NEW, cfg)

print "ids:"
i = 0
while i < ids.length
  print " " + ids[i].to_s
  i = i + 1
end
puts ""
