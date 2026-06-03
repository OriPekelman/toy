# demos/smollm2.rb — SmolLM2-135M inference through Toy::SmolLM2.
#
# Native Mat forward (no FFI). Mirrors demos/gpt2.rb. Tokenization
# is host-side via prep/smollm2_tokens.py:
#   ./prep/smollm2_tokens.py encode "Once upon a time"
#   ./demos/smollm2
#   ./prep/smollm2_tokens.py decode

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy/train/training"   # parse_ids

GGUF     = "data/smollm2-135m-f32.gguf"
IDS_PATH = "data/smollm2_prompt_ids.txt"
N_NEW    = 8

# --- config from GGUF ---
cfg = SmolLM2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " n_q=" + cfg.n_heads.to_s + " n_kv=" + cfg.n_kv.to_s +
     " L=" + cfg.n_layers.to_s + " d_ff=" + cfg.d_ff.to_s +
     " rope_base=" + cfg.rope_base.to_s

# --- build + load ---
model = Toy::SmolLM2.new(cfg)
GGUFLoad.load_toy_smollm2(model, GGUF)
puts ""
puts ""
puts model.algorithm_card
puts ""

# --- read prompt ids ---
raw = ["?"]
raw.pop
File.open(IDS_PATH, "r") do |f|
  f.each_line { |line| raw.push(line.chomp) }
end
ids = parse_ids(raw[0])
puts "prompt: " + ids.length.to_s + " tokens"

# --- greedy generation ---
gen = 0
t0  = Time.now
while gen < N_NEW
  logits = model.forward(ids, 0)              # [T, V]
  last   = logits.nrows - 1
  base   = last * logits.ncols
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

# --- write back ---
File.open(IDS_PATH, "w") do |out|
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
puts "wrote " + IDS_PATH
