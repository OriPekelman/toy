# F1.2 step 6b — multi-POSITION SFT training on CUDA.
#
# Cycles between TWO training graphs at different positions in the
# sequence, training on next-token prediction at each. Shared persistent
# tensors (LoRA-A, LoRA-B, Adam m/v) flow between the two graphs;
# only the forward/backward chain is rebuilt.
#
# The Adam m/v tensors live in ctx_w (allocated by the cache via
# `enable_lora_q_adamw!` before realize_for_mmap), so they survive
# `tnn_reset_for_rebuild` between position-switch rebuilds. Previously
# m/v lived in the compute ctx and were freed on rebuild → cycle 2
# diverged to NaN. The persistent variant is the F1.2 step 6b fix.
#
# Limitation: each position-switch tears down graph_b and re-sched-
# allocs (sched can't keep two cgraphs alive simultaneously). The cost
# is ~6 ms build + a few hundred ms sched-alloc per switch — fine for
# 2-position cycle, not viable for true full-sequence SFT (which is
# what 6d would need). For real SFT we need a SEQUENCE-mode forward
# graph (FullForwardFFICache-shape) that processes all T positions
# in one compute. That's the M3-shaped prerequisite (task #75) for
# step 6c/6d.
#
# Also: each rebuild writes K/V into the cache at its position via
# cpy, so the KV state isn't stable across the cycle. The model sees
# inconsistent contexts. This smoke is a mechanical validation
# ("multi-pos training plumbing works") not a real SFT signal.
#
# Run: ./demos/smollm2_lora_sft_multipos_cuda

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/models/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
MAX_T     = (ENV["MAX_T"]    || "32").to_i
RANK      = (ENV["RANK"]     || "16").to_i
SEED      = (ENV["SEED"]     || "42").to_i
CYCLES    = (ENV["CYCLES"]   || "10").to_i
LR        = (ENV["LR"]       || "0.001").to_f
BETA1     = 0.9
BETA2     = 0.999
EPS       = 1.0e-8
WD        = 0.0

# Prefix is 4 tokens; we train at pos 4 AND pos 5.
# At pos 4, predict TARGET_4; at pos 5, predict TARGET_5.
PROMPT   = [12092, 4845, 253, 1429]
TARGET_4 = 99
TARGET_5 = 1234

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s
puts "multipos SFT: " + CYCLES.to_s + " cycles (pos4 → pos5), LR=" + LR.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICacheCuda.new
kv.enable_lora_q!(RANK)
kv.enable_lora_q_adamw!                   # F1.2 step 6b: persistent m/v
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)

# Initial prefill of all 4 prompt tokens (writes K/V at pos 0..3).
i = 0
while i < PROMPT.length
  SmolLM2KVCuda.decode_step(kv, PROMPT[i], i)
  i = i + 1
end
last_prompt_id = PROMPT[PROMPT.length - 1]

# Adam state — flattened into the same idx scheme used in
# build_training_graph (li * n_heads + hq). The cache owns the
# persistent tensors; we just gather pointers into linear arrays
# for the inner loop.
m_tensors_a = []
v_tensors_a = []
m_tensors_b = []
v_tensors_b = []
sess = kv.sess
li = 0
while li < cfg.n_layers
  blk_init = kv.kv_blocks_ffi[li]
  hq = 0
  while hq < cfg.n_heads
    m_tensors_a.push(blk_init.t_w_lora_a_q_m[hq])
    v_tensors_a.push(blk_init.t_w_lora_a_q_v[hq])
    m_tensors_b.push(blk_init.t_w_lora_b_q_m[hq])
    v_tensors_b.push(blk_init.t_w_lora_b_q_v[hq])
    hq = hq + 1
  end
  li = li + 1
end

# Helper — builds a fresh training graph at the given pos, wires
# opt_step_adamw for every LoRA pair to the SHARED m/v tensors, and
# realizes. Returns step handle + loss tensor + labels tensor + hp
# tensor. Caller uploads inputs and runs compute_backward.
def build_training_graph(kv, cfg, train_pos, m_a_arr, v_a_arr, m_b_arr, v_b_arr)
  sess = kv.sess
  TinyNNCuda.tnn_reset_for_rebuild(sess)
  step_h = kv.build_decode_step(train_pos)
  t_labels = TinyNNCuda.tnn_input_2d_f32(sess, 1, cfg.vocab)
  t_hp_v   = TinyNNCuda.tnn_input_1d_f32(sess, 7)
  t_loss = TinyNNCuda.tnn_cross_entropy_loss(sess, step_h.kv_step_logits, t_labels)
  TinyNNCuda.tnn_set_output(t_loss); TinyNNCuda.tnn_set_loss(t_loss)
  TinyNNCuda.tnn_build_forward_only(sess, t_loss)
  TinyNNCuda.tnn_build_backward(sess)
  idx = 0
  li = 0
  while li < cfg.n_layers
    blk = kv.kv_blocks_ffi[li]
    hq = 0
    while hq < cfg.n_heads
      t_a = blk.t_w_lora_a_q[hq]
      t_b = blk.t_w_lora_b_q[hq]
      t_grad_a = TinyNNCuda.tnn_tensor_grad(sess, t_a)
      t_grad_b = TinyNNCuda.tnn_tensor_grad(sess, t_b)
      t_opt_a = TinyNNCuda.tnn_opt_step_adamw(sess, t_a, t_grad_a,
                                                m_a_arr[idx], v_a_arr[idx], t_hp_v)
      t_opt_b = TinyNNCuda.tnn_opt_step_adamw(sess, t_b, t_grad_b,
                                                m_b_arr[idx], v_b_arr[idx], t_hp_v)
      TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_a)
      TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_b)
      idx = idx + 1
      hq = hq + 1
    end
    li = li + 1
  end
  TinyNNCuda.tnn_realize_backward(sess)
  [step_h, t_loss, t_labels, t_hp_v]
end

def train_one_step(sess, step_h, t_loss, t_labels, t_hp_v,
                    input_token_id, train_pos, target_id, step_t, vocab,
                    is_first_step)
  if is_first_step
    TinyNNCuda.tnn_graph_reset(sess)
  else
    TinyNNCuda.tnn_graph_reset_grads_only(sess)
  end
  beta1h = 1.0 / (1.0 - (0.9 ** step_t.to_f))
  beta2h = 1.0 / (1.0 - (0.999 ** step_t.to_f))
  m_hp = Mat.new(1, 7)
  m_hp.flat[0] = 0.001; m_hp.flat[1] = 0.9; m_hp.flat[2] = 0.999
  m_hp.flat[3] = 1.0e-8; m_hp.flat[4] = 0.0
  m_hp.flat[5] = beta1h; m_hp.flat[6] = beta2h
  m_labels = Mat.new(1, vocab)
  i = 0
  while i < vocab; m_labels.flat[i] = 0.0; i = i + 1; end
  m_labels.flat[target_id] = 1.0
  TinyNNCuda.upload_int_array(sess, step_h.t_token_id, [input_token_id])
  TinyNNCuda.upload_int_array(sess, step_h.t_pos,      [train_pos])
  TinyNNCuda.upload_row_major(sess, t_labels, m_labels)
  TinyNNCuda.upload_row_major(sess, t_hp_v,   m_hp)
  TinyNNCuda.tnn_compute_backward(sess)
  TinyNNCuda.tnn_download(sess, t_loss)
  TinyNNCuda.tnn_scratch_get(sess, 0)
end

losses_pos4 = []
losses_pos5 = []
global_step = 0

cy = 0
while cy < CYCLES
  # Train at pos 4 (input = last_prompt_id at pos 4, predict TARGET_4)
  step4 = build_training_graph(kv, cfg, 4, m_tensors_a, v_tensors_a, m_tensors_b, v_tensors_b)
  global_step = global_step + 1
  loss_4 = train_one_step(sess, step4[0], step4[1], step4[2], step4[3],
                            last_prompt_id, 4, TARGET_4, global_step, cfg.vocab,
                            global_step == 1)
  losses_pos4.push(loss_4)

  # Train at pos 5 (after pos-4 step, K/V at pos 4 is whatever the
  # last graph wrote. Input token for pos 5 is TARGET_4 — pretending
  # the model "saw" its prediction. Predict TARGET_5).
  step5 = build_training_graph(kv, cfg, 5, m_tensors_a, v_tensors_a, m_tensors_b, v_tensors_b)
  global_step = global_step + 1
  loss_5 = train_one_step(sess, step5[0], step5[1], step5[2], step5[3],
                            TARGET_4, 5, TARGET_5, global_step, cfg.vocab,
                            false)
  losses_pos5.push(loss_5)

  puts "  cycle " + (cy + 1).to_s + ": pos4 CE=" + loss_4.to_s +
       "  pos5 CE=" + loss_5.to_s
  cy = cy + 1
end

puts ""
init4 = losses_pos4[0]
final4 = losses_pos4[losses_pos4.length - 1]
init5 = losses_pos5[0]
final5 = losses_pos5[losses_pos5.length - 1]
puts "pos4: " + init4.to_s + " → " + final4.to_s
puts "pos5: " + init5.to_s + " → " + final5.to_s

# Acceptance: BOTH positions show meaningful decrease AND no NaN.
# Previously this smoke FAILED because Adam m/v lived in the compute
# ctx and were freed at every position-switch rebuild → cycle 2
# diverged to NaN. F1.2 step 6b moved m/v to ctx_w (persistent) via
# `kv.enable_lora_q_adamw!`, fixing the divergence.
if final4.to_s == "NaN" || final5.to_s == "NaN"
  puts "VERDICT: FAIL (NaN — Adam m/v not persisting across position-switch rebuild)"; exit 1
end
if final4 >= 0.7 * init4
  puts "VERDICT: FAIL (pos4 didn't converge: " + init4.to_s + " → " + final4.to_s + ")"; exit 1
end
if final5 >= 0.7 * init5
  puts "VERDICT: FAIL (pos5 didn't converge: " + init5.to_s + " → " + final5.to_s + ")"; exit 1
end
puts "VERDICT: PASS (multi-pos AdamW SFT works; both positions converged)"
