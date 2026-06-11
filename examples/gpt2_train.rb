# examples/gpt2_train.rb — train a GPT-2 from scratch via the engine API.
#
# The GPT-2 counterpart to examples/train_from_scratch.rb: a clean read of the
# Toy::LLM::Engine::GPT2SeqEngine surface —
#   engine.realize!(vocab, d_model, n_heads, d_ff, n_layers, context, seed)
#   engine.step!(token_ids, positions, labels, hp, is_first?) -> loss
# builds the whole forward + CE + backward + AdamW graph (wte+wpe learned
# positional embeddings, LayerNorm, multi-head causal attention, GELU FFN, tied
# output) and drives one training step. The LayerNorm + GELU backward ride the
# vendored ggml kernels (vendor-patches/0007).
#
# For the CLI surface (train on the from-scratch corpus, write a run dir):
#   toy train from-scratch --arch gpt2            # CPU
#   toy train from-scratch --arch gpt2 --device cuda
#
# This file memorizes a fixed synthetic sequence so the loss visibly collapses
# (CE -> ~0) in a few seconds — the clearest demonstration that the kernels
# train. Build + run:
#   make examples/gpt2_train && ./examples/gpt2_train
#
# REQUIRE-PATH NOTE: from examples/ the lib paths are "../lib/..." (one level
# up). A wrong path is silently ignored by Spinel → TinyNN/Mat unloaded →
# emit-0 → CE=0. (Bit us once; see docs/notes/gpt2-engine-spinel-blocker.md.)

require_relative "../lib/toy"
require_relative "../lib/tinynn"
require_relative "../lib/toy/llm/engine/gpt2_seq_engine"
require_relative "../lib/toy/llm/adamw"

VOCAB   = 64
D_MODEL = 32
N_HEADS = 4
D_FF    = 64
N_LAYERS = 2
CONTEXT = 16
STEPS   = (ENV["STEPS"] || "120").to_i
SEED    = (ENV["SEED"]  || "0").to_i

puts "GPT-2 from scratch: vocab=" + VOCAB.to_s + " d=" + D_MODEL.to_s +
     " heads=" + N_HEADS.to_s + " d_ff=" + D_FF.to_s + " L=" + N_LAYERS.to_s +
     " ctx=" + CONTEXT.to_s

engine = Toy::LLM::Engine::GPT2SeqEngine.new
engine.realize!(VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS, CONTEXT, SEED)

# A fixed, memorizable token sequence + its shift-by-one next-token targets.
seq_ids   = [0]; seq_ids.pop
positions = [0]; positions.pop
k = 0
while k < CONTEXT
  seq_ids.push((k * 11 + 5) % VOCAB)
  positions.push(k)
  k = k + 1
end

m_labels = Mat.new(CONTEXT, VOCAB)
zj = 0
while zj < CONTEXT * VOCAB; m_labels.flat[zj] = 0.0; zj = zj + 1; end
lk = 0
while lk < CONTEXT
  tgt = (lk + 1 < CONTEXT) ? seq_ids[lk + 1] : seq_ids[lk]
  m_labels.flat[lk * VOCAB + tgt] = 1.0
  lk = lk + 1
end

# NAMED AdamW (byte-identical to the old hand-filled m_hp): lr=0.01,
# beta2=0.999, per-step 1/(1-beta^t) bias correction.
adamw = Toy::AdamW.for_lora   # gpt2 graph reads the lora hp convention
adamw.lr = 0.01

first_loss = 0.0
last_loss  = 0.0
step = 1
while step <= STEPS
  m_hp = adamw.hp(step)
  loss = engine.step!(seq_ids, positions, m_labels, m_hp, step == 1)
  first_loss = loss if step == 1
  last_loss  = loss
  if step <= 3 || step % 20 == 0 || step == STEPS
    puts "step " + step.to_s.rjust(4) + ": CE=" + loss.to_s
  end
  step = step + 1
end

puts "initial CE = " + first_loss.to_s
puts "final   CE = " + last_loss.to_s
if last_loss < first_loss * 0.1
  puts "VERDICT: GPT-2 trains — CE collapsed (LayerNorm + GELU backward via the vendored kernels)"
  exit 0
else
  puts "VERDICT: not learning (final/initial = " + (last_loss / first_loss).to_s + ")"
  exit 1
end
