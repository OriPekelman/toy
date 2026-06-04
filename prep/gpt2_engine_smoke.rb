# prep/gpt2_engine_smoke.rb — isolate GPT2SeqEngine vs the corpus path.
# Drives the engine with the SAME synthetic data as the gated inline trainer
# (prep/gpt2_train_min.rb). If CE drops here, the engine class is sound and the
# runner's corpus/labels are the suspect; if CE is 0/stuck, the engine class is.
require_relative "../lib/toy"
require_relative "../lib/tinynn"
require_relative "../lib/toy/llm/labels"
require_relative "../lib/toy/llm/engine/gpt2_seq_engine"

VOCAB = 32; D_MODEL = 16; N_HEADS = 4; D_FF = 32; N_LAYERS = 2; CONTEXT = 8

engine = Toy::LLM::Engine::GPT2SeqEngine.new
engine.realize!(VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS, CONTEXT, 1234)
puts "realize_backward rc = " + engine.g_rb_rc.to_s + " (1=ok)  weights=" + engine.g_weights.length.to_s

seq_ids = [0]; seq_ids.pop
positions = [0]; positions.pop
ti = 0
while ti < CONTEXT
  seq_ids.push((ti * 7 + 3) % VOCAB)
  positions.push(ti)
  ti = ti + 1
end
m_labels = Toy::Labels.next_token(seq_ids, VOCAB, CONTEXT, 1)

# count nonzero labels (sanity)
nz = 0; j = 0
while j < CONTEXT * VOCAB; nz = nz + (m_labels.flat[j] > 0.5 ? 1 : 0); j = j + 1; end
puts "label one-hots set: " + nz.to_s + " (expect " + CONTEXT.to_s + ")"

m_hp = Mat.new(1, 7)
m_hp.flat[0] = 0.01; m_hp.flat[1] = 0.9; m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8; m_hp.flat[4] = 0.0

step = 1
while step <= 20
  m_hp.flat[5] = 1.0 / (1.0 - (0.9 ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  loss = engine.step!(seq_ids, positions, m_labels, m_hp, step == 1)
  if step <= 3 || step == 20
    puts "step " + step.to_s + ": CE=" + loss.to_s + " (compute rc=" + engine.g_cb_rc.to_s + ")"
  end
  step = step + 1
end
