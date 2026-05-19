# demos/gpt2.rb — DistilGPT2 / GPT-2 inference through Toy::GPT2.
#
# Native Mat forward (no FFI). Reads pre-tokenized prompt IDs from
# data/prompt_ids.txt (see prep/tokens.py), runs greedy generation,
# writes the full ID sequence back. The model itself is ~80 lines
# of Ruby in lib/toy_gpt2.rb; this demo is the glue.

require_relative "../lib/toy"
require_relative "../lib/toy_gpt2"
require_relative "../lib/toy_gpt2_loader"
require_relative "../lib/training"   # parse_ids: comma- or space-separated int line

# --- config from GGUF ----
GGUF = "data/distilgpt2-f32.gguf"
ext  = GPT2ConfigLoader.read(GGUF)
cfg  = Toy::GPT2Config.new(ext.vocab_size, ext.d_model, ext.n_heads,
                           ext.d_ff, ext.n_layers, ext.context_length)
# --- build + load ----
model = Toy::GPT2.new(cfg)
GGUFLoad.load_toy_gpt2(model, GGUF)
puts ""
puts ""
puts model.algorithm_card
puts ""

# --- read prompt ids ----
raw = ["?"]
raw.pop
File.open("data/prompt_ids.txt", "r") do |f|
  f.each_line { |line| raw.push(line.chomp) }
end
ids = parse_ids(raw[0])
puts "prompt: " + ids.length.to_s + " tokens"

# --- greedy generation ----
N_NEW = 8
gen = 0
t0  = Time.now
while gen < N_NEW
  logits = model.forward(ids, 0)               # [T, V]
  last   = logits.nrows - 1
  base   = last * logits.ncols
  # argmax over last row
  best_i = 0
  best_v = logits.flat[base]
  j = 1
  while j < logits.ncols
    v = logits.flat[base + j]
    if v > best_v
      best_v = v
      best_i = j
    end
    j += 1
  end
  ids.push(best_i)
  gen += 1
end
dt = ((Time.now - t0) * 1000.0).round(1)
puts "generated " + N_NEW.to_s + " tokens in " + dt.to_s + " ms"

# --- write back ----
File.open("data/prompt_ids.txt", "w") do |out|
  n = ids.length
  k = 0
  while k < n
    out.write(ids[k].to_s)
    if k < n - 1
      out.write(" ")
    end
    k += 1
  end
  out.write("\n")
end
puts "wrote data/prompt_ids.txt"
