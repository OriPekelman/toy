# prep/smokes/smoke_full_finetune.rb — full fine-tune (F3) loss-curve driver.
#
# The 6th realize-gate's compiled runner (CPU). Drives the engine's
# realize_for_full_finetune + build_training_step on a real GGUF and prints
# a deterministic CE curve. full_finetune loads ALL per-block weights from
# the GGUF (no random init), so given a fixed model + tokens + hyper-params
# the curve is bit-deterministic — the basis for prep/full_finetune_gate.rb,
# which records this curve from the INLINE realize path and re-verifies it
# byte-for-byte after the per-block alloc lifts onto TransformerBlock.
#
# Model-gated: needs a real GGUF (default data/smollm2-135m-native.gguf, a
# gitignored dev artifact — the gate SKIPs loudly when it is absent). Train
# losses are ggml-internal → byte-exact across CPU backends + machines.

require_relative "../../lib/toy"
require_relative "../../lib/toy/models/toy_smollm2"
require_relative "../../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../../lib/toy/llm/engine/llama_seq_engine"
require_relative "../../lib/toy/llm/adamw"

GGUF      = ENV["GGUF"]   || "data/smollm2-135m-native.gguf"
STEPS     = (ENV["STEPS"] || "8").to_i
LR        = (ENV["LR"]    || "0.0005").to_f
TARGET_ID = (ENV["TARGET_ID"] || "99").to_i
TOKENS    = [12092, 4845, 253, 1429]

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
gguf  = TinyNN.tnn_gguf_load(GGUF)

seq = Toy::LLM::Engine::LlamaSeqEngine.new
seq.enable_full_finetune!
seq.realize_for_full_finetune(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)

r = seq.build_training_step
t_loss = r[0]; t_labels = r[1]; t_hp = r[2]

m_labels = Mat.new(TOKENS.length, cfg.vocab)
i = 0
while i < TOKENS.length * cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
ti = 0
while ti < TOKENS.length
  m_labels.flat[ti * cfg.vocab + TARGET_ID] = 1.0
  ti = ti + 1
end

# NAMED AdamW (byte-identical to the old hand-filled m_hp Mat). lora-style
# bias correction: beta2=0.999 and slots 5/6 = 1/(1-beta^t) per step.
adamw = Toy::AdamW.for_lora   # full-finetune graph reads the lora hp convention
adamw.lr = LR

positions = [0, 1, 2, 3]
step = 1
while step <= STEPS
  m_hp = adamw.hp(step)
  if step == 1
    TinyNN.tnn_graph_reset(seq.sess)
  else
    TinyNN.tnn_graph_reset_grads_only(seq.sess)
  end
  TinyNN.upload_int_array(seq.sess, seq.t_seq_token_ids, TOKENS)
  TinyNN.upload_int_array(seq.sess, seq.t_seq_positions, positions)
  TinyNN.upload_row_major(seq.sess, t_labels, m_labels)
  TinyNN.upload_row_major(seq.sess, t_hp,     m_hp)
  TinyNN.tnn_compute_backward(seq.sess)
  TinyNN.tnn_download(seq.sess, t_loss)
  puts "step " + step.to_s + ": CE=" + TinyNN.tnn_scratch_get(seq.sess, 0).to_s
  step = step + 1
end
