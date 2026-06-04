# lib/toy/run/train_gpt2.rb — Spinel-compiled GPT-2 from-scratch TRAINING runner.
#
# ⚠️ WIP — BLOCKED on a Spinel poly-degradation (toy#32 class). The
# GPT2SeqEngine + this runner COMPILE and TRAIN CORRECTLY at small dims (proven
# by prep/gpt2_engine_smoke.rb: VOCAB=32 → 72 weights, CE 3.46→0.96), but at
# realistic dims (VOCAB=627) the engine's @g_weights array and the label Mat
# degrade to EMPTY/ZERO in this compilation unit → CE=0, no training. It is
# SIZE-dependent, not env-vs-literal: literal VOCAB=627 fails the same way; the
# llama engine works at 627 because its unit constrains Mat/Array types
# differently (it doesn't build many diverse-dim Ruby Mats). NOT wired into the
# `toy train` CLI yet (so nothing user-facing is broken). The byte-exact INLINE
# trainer (prep/gpt2_train_min.rb, `make gate-gpt2`) is the working reference.
# Fix hypotheses for the next pass: (a) replace Mat+upload_row_major init with
# flat-Array + tnn_upload_from_float_array (the llama upload_random_init!
# pattern, no Ruby Mat in the hot path); (b) reduce Mat polymorphism in the
# engine; (c) pin the realize! dim params. See docs/notes/gpt2-backward-patches.md.
#
# The lib-side compute for `toy train from-scratch --arch gpt2`. A SEPARATE
# binary (libexec/toy-train-gpt2) from the llama runner (lib/toy/run/train.rb):
# compiling the GPT-2 realize path alongside the llama one would make Spinel
# merge the engine/cfg receiver types (landmine #16) — the same reason the LoRA
# recipe is its own binary. This keeps both realize paths monomorphic and
# protects the llama byte-exact gates from GPT-2 churn.
#
# Env (mirrors the llama from-scratch runner):
#   STEPS   — training steps (default "5")
#   SEED    — random-init seed (default "0")
#   plus the shape knobs below (VOCAB/D_MODEL/N_HEADS/D_FF/N_LAYERS/CONTEXT).
#
# Output: "step <N>: loss=<float>" per step on STDOUT. Trains on the first line
# of data/ts_seqs.txt (the from-scratch ground-truth corpus). Backward of the
# LayerNorm + GELU rides the two vendored kernels (vendor-patches/0007). CPU
# only (this slice); the CUDA/Metal mirrors come after the CPU gate.

require_relative "../toy"
require_relative "../tinynn"
require_relative "../llm/labels"
require_relative "../llm/adamw"
require_relative "../llm/engine/gpt2_seq_engine"

STEPS    = (ENV["STEPS"]    || "5").to_i
SEED     = (ENV["SEED"]     || "0").to_i
LR       = (ENV["LR"]       || "0.001").to_f
# Shape dims are LITERAL constants, NOT ENV.to_i. Critical (toy#32 poly-degrade
# class): a runtime-Int VOCAB/CONTEXT degrades the label one-hot indexing
# (m.flat[k*VOCAB + tgt]) to emit-0, silently zeroing the labels → CE=0. Spinel
# needs the concrete-Int literal to keep the numerical path monomorphic. The
# llama from-scratch runner is literal-shaped for the same reason; env-driven
# shapes are a follow-up once the dims are pinned. From-scratch gate shape.
VOCAB    = 627
D_MODEL  = 64
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

engine = Toy::LLM::Engine::GPT2SeqEngine.new
engine.realize!(VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS, CONTEXT, SEED)

# First corpus line → CONTEXT token ids, zero-padded (from-scratch ground truth).
raw     = File.read("data/ts_seqs.txt")
lines   = raw.split("\n")
parts   = lines[0].split(" ")
seq_ids = [0]; seq_ids.pop
k = 0
while k < CONTEXT
  if k < parts.length
    seq_ids.push(parts[k].to_i)
  else
    seq_ids.push(0)
  end
  k = k + 1
end

positions = [0]; positions.pop
p = 0
while p < CONTEXT
  positions.push(p)
  p = p + 1
end

# Shift-by-one next-token one-hot labels (target = next token, self at last pos).
# Built INLINE (not Toy::Labels.next_token): the cross-module call degrades to
# emitting an all-zero Mat in this compilation unit (the toy#32 poly-degrade
# class — seq_ids/Mat writes lose their concrete arm across the module
# boundary, so CE came out 0). Inlining keeps seq_ids/Mat monomorphic here.
m_labels = Mat.new(CONTEXT, VOCAB)
zj = 0
while zj < CONTEXT * VOCAB; m_labels.flat[zj] = 0.0; zj = zj + 1; end
lk = 0
while lk < CONTEXT
  tgt = (lk + 1 < CONTEXT) ? seq_ids[lk + 1].to_i : seq_ids[lk].to_i
  m_labels.flat[lk * VOCAB + tgt] = 1.0
  lk = lk + 1
end

# AdamW hyper-params built INLINE (cross-module value objects degrade in this
# unit — see the labels note). slots 5/6 = 1/(1-beta^t) (bias correction),
# matching the gated inline GPT-2 trainer's dynamics.
m_hp = Mat.new(1, 7)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0

step = 0
while step < STEPS
  sp1 = (step + 1).to_f
  m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** sp1))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** sp1))
  loss = engine.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  step = step + 1
end
