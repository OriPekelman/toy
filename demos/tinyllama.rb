# demos/tinyllama_native.rb — TinyLlama-1.1B via the native Mat path
# (no FFI). Slow but useful for debugging the FFI-only issue.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/models/toy_smollm2_loader"
require_relative "../lib/toy/train/training"

GGUF     = "data/tinyllama-1.1b-f32.gguf"
IDS_PATH = "data/tinyllama_prompt_ids.txt"

cfg   = SmolLM2ConfigLoader.read(GGUF)
model = Toy::SmolLM2.new(cfg)
GGUFLoad.load_toy_smollm2(model, GGUF)
puts "has_untied_output=" + model.has_untied_output.to_s

raw = ["?"]
raw.pop
File.open(IDS_PATH, "r") do |f|
  f.each_line { |line| raw.push(line.chomp) }
end
ids = parse_ids(raw[0])
puts "prompt: " + ids.length.to_s + " tokens, first=" + ids[0].to_s

# ONE forward pass on the prompt — just see what the logits look like.
t0 = Time.now
logits = model.forward(ids, 0)
puts "forward: " + ((Time.now - t0) * 1000.0).to_s + " ms"
puts "logits shape [" + logits.nrows.to_s + ", " + logits.ncols.to_s + "]"
last = logits.nrows - 1
base = last * logits.ncols
puts "last-row logits [0..9]: " +
     logits.flat[base + 0].to_s + ", " + logits.flat[base + 1].to_s + ", " +
     logits.flat[base + 2].to_s + ", " + logits.flat[base + 3].to_s + ", " +
     logits.flat[base + 4].to_s
puts "argmax over last row:"
best_i = 0
best_v = logits.flat[base]
j = 1
while j < cfg.vocab
  v = logits.flat[base + j]
  if v > best_v
    best_v = v
    best_i = j
  end
  j = j + 1
end
puts "  best_i=" + best_i.to_s + " best_v=" + best_v.to_s
