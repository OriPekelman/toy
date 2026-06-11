# Task #70 experiment: test the sched-aliasing hypothesis. Same
# graph + setup as demos/smollm2_lora_train_ce.rb, but with EVERY
# node in graph_b pinned as output (via tnn_pin_all_graph_b_nodes).
# Pinning forbids the sched from reusing any intermediate's buffer
# slot — which would block any aliasing-induced corruption in the
# long backward chain.
#
# Compare loss trajectory against the non-pinned CE smoke:
#   - If pinned CPU now converges like CUDA → sched aliasing IS the
#     root cause of the CPU/CUDA divergence. Document + open a
#     ggml-side issue. Fix would be either (a) the pinning workaround
#     as a primitive callers opt into, or (b) audit ggml's sched.cpp
#     to find why CPU sched aliases nodes that have downstream
#     consumers but no set_output.
#   - If pinned CPU still doesn't converge → sched aliasing is not
#     the cause. Pivot to FFN (once Spinel #626 #1 unblocks the bisect
#     smoke) or multi-layer composition.
#
# This is a diagnostic — pinning every node inflates memory and disables
# the sched's buffer-reuse optimization, so this is not a path we'd
# ship to production. It's the cleanest way to test the hypothesis.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
MAX_T     = (ENV["MAX_T"]    || "32").to_i
RANK      = (ENV["RANK"]     || "16").to_i
SEED      = (ENV["SEED"]     || "42").to_i
STEPS     = (ENV["STEPS"]    || "20").to_i
LR        = (ENV["LR"]       || "0.5").to_f
TARGET_ID = (ENV["TARGET_ID"]|| "99").to_i

PROMPT    = [12092, 4845, 253, 1429]
TRAIN_POS = PROMPT.length

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s
puts "training (PINNED): LR=" + LR.to_s + " STEPS=" + STEPS.to_s +
     " TARGET_ID=" + TARGET_ID.to_s + " r=" + RANK.to_s

gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)

puts ""
puts "=== Prefill " + PROMPT.length.to_s + " tokens ==="
i = 0
while i < PROMPT.length
  SmolLM2KV.decode_step(kv, PROMPT[i], i)
  i = i + 1
end
last_prompt_id = PROMPT[PROMPT.length - 1]

puts ""
puts "=== Build training graph ==="
sess = kv.sess
TinyNN.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)

t_labels = TinyNN.tnn_input_2d_f32(sess, 1, cfg.vocab)
t_hp_v   = TinyNN.tnn_input_1d_f32(sess, 2)
t_loss = TinyNN.tnn_cross_entropy_loss(sess, step_h.kv_step_logits, t_labels)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)

TinyNN.tnn_build_forward_only(sess, t_loss)
TinyNN.tnn_build_backward(sess)

li = 0
while li < cfg.n_layers
  blk = kv.kv_blocks_ffi[li]
  hq = 0
  while hq < cfg.n_heads
    t_a = blk.t_w_lora_a_q[hq]
    t_b = blk.t_w_lora_b_q[hq]
    t_grad_a = TinyNN.tnn_tensor_grad(sess, t_a)
    t_grad_b = TinyNN.tnn_tensor_grad(sess, t_b)
    t_opt_a = TinyNN.tnn_opt_step_sgd(sess, t_a, t_grad_a, t_hp_v)
    t_opt_b = TinyNN.tnn_opt_step_sgd(sess, t_b, t_grad_b, t_hp_v)
    TinyNN.tnn_extend_backward_graph(sess, t_opt_a)
    TinyNN.tnn_extend_backward_graph(sess, t_opt_b)
    hq = hq + 1
  end
  li = li + 1
end

# THE EXPERIMENT — pin every node in graph_b BEFORE realize_backward.
n_pinned = TinyNN.tnn_pin_all_graph_b_nodes(sess)
puts "  pinned " + n_pinned.to_s + " graph_b nodes as output."

TinyNN.tnn_realize_backward(sess)
puts "  graph realized."

m_labels = Mat.new(1, cfg.vocab)
i = 0
while i < cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
m_labels.flat[TARGET_ID] = 1.0
m_hp_v = Mat.new(1, 2)
m_hp_v.flat[0] = LR
m_hp_v.flat[1] = 0.0

puts ""
puts "=== Train " + STEPS.to_s + " SGD steps ==="
losses = []
s = 0
while s < STEPS
  TinyNN.tnn_graph_reset(sess)
  TinyNN.upload_int_array(sess, step_h.t_token_id, [last_prompt_id])
  TinyNN.upload_int_array(sess, step_h.t_pos,      [TRAIN_POS])
  TinyNN.upload_row_major(sess, t_labels, m_labels)
  TinyNN.upload_row_major(sess, t_hp_v,   m_hp_v)
  rc = TinyNN.tnn_compute_backward(sess)
  if rc != 0; puts "compute_backward rc=" + rc.to_s; exit 1; end
  TinyNN.tnn_download(sess, t_loss)
  loss_v = TinyNN.tnn_scratch_get(sess, 0)
  losses.push(loss_v)
  puts "  step " + (s + 1).to_s + ": CE=" + loss_v.to_s
  s = s + 1
end

puts ""
puts "initial CE = " + losses[0].to_s
puts "final   CE = " + losses[losses.length - 1].to_s
puts "diff       = " + (losses[0] - losses[losses.length - 1]).to_s
