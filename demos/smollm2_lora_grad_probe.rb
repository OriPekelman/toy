# Task #70 diagnostic — dump LoRA-A/B grad magnitudes per layer to
# find where the CPU vs CUDA backward chain diverges.
#
# Setup: identical to demos/smollm2_lora_train_ce.rb. Run ONE
# compute_backward step on the requested backend (BACKEND=cpu | cuda),
# then for each layer download layer.head[0]'s LoRA-A and LoRA-B
# grad tensors and print:
#   - max(|elem|) of A's grad
#   - max(|elem|) of B's grad
#
# Compare line-by-line across the two backends. If layer L has matched
# magnitudes for all L >= K but diverges below K, the bug is in the
# backward propagation between layer K and K-1 (or the op that lives
# at that boundary). If ALL layers diverge from the start (layer 29
# closest to logits), the bug is in the CE backward / unembed path.
#
# Run:
#   BACKEND=cuda ./demos/smollm2_lora_grad_probe_cuda > /tmp/cuda_grads.txt
#   BACKEND=cpu  ./demos/smollm2_lora_grad_probe       > /tmp/cpu_grads.txt
#   diff /tmp/cpu_grads.txt /tmp/cuda_grads.txt

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/models/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
MAX_T     = (ENV["MAX_T"]    || "32").to_i
RANK      = (ENV["RANK"]     || "16").to_i
SEED      = (ENV["SEED"]     || "42").to_i
TARGET_ID = (ENV["TARGET_ID"]|| "99").to_i

PROMPT    = [12092, 4845, 253, 1429]
TRAIN_POS = PROMPT.length

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "# GradProbe CPU: " + GGUF + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s

gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)

i = 0
while i < PROMPT.length
  SmolLM2KV.decode_step(kv, PROMPT[i], i)
  i = i + 1
end

sess = kv.sess
TinyNN.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)
t_labels = TinyNN.tnn_input_2d_f32(sess, 1, cfg.vocab)
t_hp_v   = TinyNN.tnn_input_1d_f32(sess, 2)
t_loss = TinyNN.tnn_cross_entropy_loss(sess, step_h.kv_step_logits, t_labels)
TinyNN.tnn_set_output(t_loss); TinyNN.tnn_set_loss(t_loss)
TinyNN.tnn_build_forward_only(sess, t_loss)
TinyNN.tnn_build_backward(sess)

# Collect grad handles for every layer's head-0 A and B (don't add
# opt_step nodes — we want bare grads, not post-update params).
# Also mark the grad tensors as output so the sched doesn't alias
# their data with a non-output node.
grad_a = []
grad_b = []
li = 0
while li < cfg.n_layers
  blk = kv.kv_blocks_ffi[li]
  hq = 0
  while hq < cfg.n_heads
    t_a = blk.t_w_lora_a_q[hq]
    t_b = blk.t_w_lora_b_q[hq]
    t_grad_a = TinyNN.tnn_tensor_grad(sess, t_a)
    t_grad_b = TinyNN.tnn_tensor_grad(sess, t_b)
    if hq == 0
      grad_a.push(t_grad_a)
      grad_b.push(t_grad_b)
      TinyNN.tnn_set_output(t_grad_a)
      TinyNN.tnn_set_output(t_grad_b)
    end
    # Still wire opt_steps to keep the graph identical to the train
    # smoke (so the same ops execute). The opt_step writes the param
    # in place but we're only reading grads, so the update is harmless.
    t_opt_a = TinyNN.tnn_opt_step_sgd(sess, t_a, t_grad_a, t_hp_v)
    t_opt_b = TinyNN.tnn_opt_step_sgd(sess, t_b, t_grad_b, t_hp_v)
    TinyNN.tnn_extend_backward_graph(sess, t_opt_a)
    TinyNN.tnn_extend_backward_graph(sess, t_opt_b)
    hq = hq + 1
  end
  li = li + 1
end

TinyNN.tnn_realize_backward(sess)

m_labels = Mat.new(1, cfg.vocab)
i = 0
while i < cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
m_labels.flat[TARGET_ID] = 1.0
m_hp_v = Mat.new(1, 2)
m_hp_v.flat[0] = 0.0    # LR=0 so opt_step doesn't perturb (we still read grads BEFORE opt_step uses them)
m_hp_v.flat[1] = 0.0

TinyNN.tnn_graph_reset(sess)
TinyNN.upload_int_array(sess, step_h.t_token_id, [PROMPT[PROMPT.length - 1]])
TinyNN.upload_int_array(sess, step_h.t_pos,      [TRAIN_POS])
TinyNN.upload_row_major(sess, t_labels, m_labels)
TinyNN.upload_row_major(sess, t_hp_v,   m_hp_v)
TinyNN.tnn_compute_backward(sess)

# Download each layer's head-0 A and B grad; compute max-abs.
def max_abs_of(sess, t, n)
  TinyNN.tnn_download(sess, t)
  m = 0.0
  i = 0
  while i < n
    v = TinyNN.tnn_scratch_get(sess, i)
    av = v
    if av < 0.0; av = -av; end
    if av > m; m = av; end
    i = i + 1
  end
  m
end

d_head = cfg.d_model / cfg.n_heads
n_a = RANK * cfg.d_model        # (r, d_model)
n_b = d_head * RANK             # (d_head, r)
puts "# layer  maxabs(gradA)  maxabs(gradB)"
li = 0
while li < cfg.n_layers
  ga = max_abs_of(sess, grad_a[li], n_a)
  gb = max_abs_of(sess, grad_b[li], n_b)
  puts li.to_s + " " + ga.to_s + " " + gb.to_s
  li = li + 1
end
