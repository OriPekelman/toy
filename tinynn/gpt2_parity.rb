# Parity probe: one forward pass at distilgpt2 shape, full prompt
# loaded from data/prompt_ids.txt, dump the last-row logits (next-token
# distribution) to data/ours_logits.txt as space-separated floats.
#
# Pair with prep/parity.py which produces data/ref_logits.txt from
# HF transformers on the same input, then compares the two.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/models/gpt2"
require_relative "../lib/toy/io/gguf_load"
require_relative "../lib/toy/train/training"   # parse_ids

VOCAB    = 50257
D_MODEL  = 768
D_FF     = 3072
N_HEADS  = 12
N_LAYERS = 6
CONTEXT  = 1024
IDS_PATH = "data/prompt_ids.txt"
GGUF     = "data/distilgpt2-f32.gguf"
OUT      = "data/ours_logits.txt"

def read_ids(path)
  raw = ["?"]
  raw.pop
  File.open(path, "r") do |f|
    f.each_line { |line| raw.push(line.chomp) }
  end
  parse_ids(raw[0])
end

ids = read_ids(IDS_PATH)
puts "prompt: " + ids.length.to_s + " tokens"

model = GPT2LM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
ok = GGUFLoad.load_gpt2(model, GGUF)
if !ok
  puts "load failed"
else
  t0 = Time.now
  logits = model.forward(ids, 0)
  dt = ((Time.now - t0) * 1000).to_s
  puts "forward: " + dt + " ms  (shape " + logits.nrows.to_s + " x " +
       logits.ncols.to_s + ")"

  # Write the last row (next-token logits). Space-separated, one line.
  last = logits.nrows - 1
  File.open(OUT, "w") do |f|
    v = 0
    while v < VOCAB
      f.write(logits.flat[last * VOCAB + v].to_s)
      if v < VOCAB - 1
        f.write(" ")
      end
      v = v + 1
    end
    f.write("\n")
  end
  puts "wrote " + VOCAB.to_s + " logits to " + OUT
  puts "compare with: ./prep/parity.py compare"

  # Quick argmax + max-logit peek (cheap and Spinel-safe).
  best   = 0
  best_v = logits.flat[last * VOCAB]
  v = 1
  while v < VOCAB
    val = logits.flat[last * VOCAB + v]
    if val > best_v
      best_v = val
      best   = v
    end
    v = v + 1
  end
  puts "ours argmax: id=" + best.to_s + "  logit=" + best_v.to_s
end
