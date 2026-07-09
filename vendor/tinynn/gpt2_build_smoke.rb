# Build-smoke for GPT2LM: instantiate the class with toy dims, run one
# forward pass on a 3-token sequence. Just confirms it Spinel-compiles
# and the call shapes are consistent; values are meaningless (random
# init, no real weights loaded yet).

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/models/gpt2"

VOCAB    = 32
D_MODEL  = 16
D_FF     = 32
N_HEADS  = 4
N_LAYERS = 2
CONTEXT  = 8

srand(42)
model = GPT2LM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
model.token_embed.fill_random(0.1)
model.pos_embed.fill_random(0.05)

ids = [3, 7, 1]
logits = model.forward(ids, 0)

puts "shape: " + logits.nrows.to_s + " x " + logits.ncols.to_s
puts "expected: " + ids.length.to_s + " x " + VOCAB.to_s

# Argmax over the last row (next-token prediction).
last = logits.nrows - 1
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
puts "argmax(last row)=" + best.to_s + "  value=" + best_v.to_s
puts "logits[0..4]: " + logits.flat[0].to_s + ", " + logits.flat[1].to_s + ", " + logits.flat[2].to_s + ", " + logits.flat[3].to_s
