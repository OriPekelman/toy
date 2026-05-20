# Task #70 bisect step 5: add CONCAT to the chain. Two attention
# heads, each with its own LoRA on Q; per-head outputs are concat'd
# along dim=0 before an O projection.
#
# Inlined (no arrays-of-tensors) to dodge Spinel's cross-class type
# inference (see Spinel #626 sub-issue 1).

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

K       = 8
D_MODEL = K * 2
T_CTX   = 4
MAX_T   = 16
R       = 4
STEPS   = 30
LR      = 0.001

sess = TinyNN.tnn_session_new(0)
$seed = 42
def next_normal(scale)
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u1 = ($seed.to_f + 1.0) / 2147483648.0
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u2 = ($seed.to_f + 1.0) / 2147483648.0
  scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

# Head 0
t_W_q0 = TinyNN.tnn_input_2d_f32_persistent(sess, K, D_MODEL)
t_A0   = TinyNN.tnn_input_2d_f32_persistent(sess, R, D_MODEL)
t_B0   = TinyNN.tnn_input_2d_f32_persistent(sess, K, R)
t_K0   = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T, K)
t_V0   = TinyNN.tnn_input_2d_f32_persistent(sess, K, MAX_T)
# Head 1
t_W_q1 = TinyNN.tnn_input_2d_f32_persistent(sess, K, D_MODEL)
t_A1   = TinyNN.tnn_input_2d_f32_persistent(sess, R, D_MODEL)
t_B1   = TinyNN.tnn_input_2d_f32_persistent(sess, K, R)
t_K1   = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T, K)
t_V1   = TinyNN.tnn_input_2d_f32_persistent(sess, K, MAX_T)
t_W_o  = TinyNN.tnn_input_2d_f32_persistent(sess, D_MODEL, D_MODEL)
t_hp   = TinyNN.tnn_input_1d_f32_persistent(sess, 2)
TinyNN.tnn_set_param(t_A0); TinyNN.tnn_set_param(t_B0)
TinyNN.tnn_set_param(t_A1); TinyNN.tnn_set_param(t_B1)
TinyNN.tnn_finalize_weights(sess)

t_x   = TinyNN.tnn_input_2d_f32(sess, 1, D_MODEL)
t_pos = TinyNN.tnn_input_1d_i32(sess, 1)
bytes_d_head = K * 4
bytes_max_T  = MAX_T * 4
scale_v      = 1.0 / Math.sqrt(K.to_f)

# Head 0 forward
t_ax0    = TinyNN.tnn_matmul(sess, t_A0, t_x)
t_bax0   = TinyNN.tnn_matmul(sess, t_B0, t_ax0)
t_qbase0 = TinyNN.tnn_matmul(sess, t_W_q0, t_x)
t_qraw0  = TinyNN.tnn_add(sess, t_qbase0, t_bax0)
t_q0     = TinyNN.tnn_rope_ext(sess, t_qraw0, t_pos, K, 10000.0)
t_Kv0    = TinyNN.tnn_view_2d(sess, t_K0, K, T_CTX, bytes_d_head, 0)
t_Vv0    = TinyNN.tnn_view_2d(sess, t_V0, T_CTX, K, bytes_max_T, 0)
t_sc0    = TinyNN.tnn_matmul(sess, t_Kv0, t_q0)
t_scl0   = TinyNN.tnn_scale(sess, t_sc0, scale_v)
t_attn0  = TinyNN.tnn_softmax(sess, t_scl0)
t_head0  = TinyNN.tnn_matmul(sess, t_Vv0, t_attn0)

# Head 1 forward
t_ax1    = TinyNN.tnn_matmul(sess, t_A1, t_x)
t_bax1   = TinyNN.tnn_matmul(sess, t_B1, t_ax1)
t_qbase1 = TinyNN.tnn_matmul(sess, t_W_q1, t_x)
t_qraw1  = TinyNN.tnn_add(sess, t_qbase1, t_bax1)
t_q1     = TinyNN.tnn_rope_ext(sess, t_qraw1, t_pos, K, 10000.0)
t_Kv1    = TinyNN.tnn_view_2d(sess, t_K1, K, T_CTX, bytes_d_head, 0)
t_Vv1    = TinyNN.tnn_view_2d(sess, t_V1, T_CTX, K, bytes_max_T, 0)
t_sc1    = TinyNN.tnn_matmul(sess, t_Kv1, t_q1)
t_scl1   = TinyNN.tnn_scale(sess, t_sc1, scale_v)
t_attn1  = TinyNN.tnn_softmax(sess, t_scl1)
t_head1  = TinyNN.tnn_matmul(sess, t_Vv1, t_attn1)

t_concat = TinyNN.tnn_concat(sess, t_head0, t_head1, 0)
t_out    = TinyNN.tnn_matmul(sess, t_W_o, t_concat)
t_o_sq   = TinyNN.tnn_mul(sess, t_out, t_out)
t_loss   = TinyNN.tnn_sum(sess, t_o_sq)

TinyNN.tnn_set_output(t_A0); TinyNN.tnn_set_output(t_B0)
TinyNN.tnn_set_output(t_A1); TinyNN.tnn_set_output(t_B1)
TinyNN.tnn_set_output(t_loss); TinyNN.tnn_set_loss(t_loss)

TinyNN.tnn_build_forward_only(sess, t_loss)
TinyNN.tnn_build_backward(sess)
t_grad_A0 = TinyNN.tnn_tensor_grad(sess, t_A0)
t_grad_B0 = TinyNN.tnn_tensor_grad(sess, t_B0)
t_grad_A1 = TinyNN.tnn_tensor_grad(sess, t_A1)
t_grad_B1 = TinyNN.tnn_tensor_grad(sess, t_B1)
t_opt_A0  = TinyNN.tnn_opt_step_sgd(sess, t_A0, t_grad_A0, t_hp)
t_opt_B0  = TinyNN.tnn_opt_step_sgd(sess, t_B0, t_grad_B0, t_hp)
t_opt_A1  = TinyNN.tnn_opt_step_sgd(sess, t_A1, t_grad_A1, t_hp)
t_opt_B1  = TinyNN.tnn_opt_step_sgd(sess, t_B1, t_grad_B1, t_hp)
TinyNN.tnn_extend_backward_graph(sess, t_opt_A0)
TinyNN.tnn_extend_backward_graph(sess, t_opt_B0)
TinyNN.tnn_extend_backward_graph(sess, t_opt_A1)
TinyNN.tnn_extend_backward_graph(sess, t_opt_B1)
TinyNN.tnn_realize_backward(sess)

# Upload weights — same seed sequence as before (consumed in declaration order).
def upload_init_weights(sess, t_W_q0, t_A0, t_B0, t_K0, t_V0,
                          t_W_q1, t_A1, t_B1, t_K1, t_V1, t_W_o,
                          t_hp, t_x, k, d_model, r, max_t, lr)
  m = Mat.new(k, d_model); i = 0
  while i < k * d_model; m.flat[i] = next_normal(0.5); i = i + 1; end
  TinyNN.upload_row_major(sess, t_W_q0, m)
  ma = Mat.new(r, d_model); i = 0
  while i < r * d_model; ma.flat[i] = next_normal(0.1); i = i + 1; end
  TinyNN.upload_row_major(sess, t_A0, ma)
  mb = Mat.new(k, r); i = 0
  while i < k * r; mb.flat[i] = 0.0; i = i + 1; end
  TinyNN.upload_row_major(sess, t_B0, mb)
  mk = Mat.new(max_t, k); i = 0
  while i < max_t * k; mk.flat[i] = next_normal(0.3); i = i + 1; end
  TinyNN.upload_row_major(sess, t_K0, mk)
  mv = Mat.new(k, max_t); i = 0
  while i < k * max_t; mv.flat[i] = next_normal(0.3); i = i + 1; end
  TinyNN.upload_row_major(sess, t_V0, mv)

  m1 = Mat.new(k, d_model); i = 0
  while i < k * d_model; m1.flat[i] = next_normal(0.5); i = i + 1; end
  TinyNN.upload_row_major(sess, t_W_q1, m1)
  ma1 = Mat.new(r, d_model); i = 0
  while i < r * d_model; ma1.flat[i] = next_normal(0.1); i = i + 1; end
  TinyNN.upload_row_major(sess, t_A1, ma1)
  mb1 = Mat.new(k, r); i = 0
  while i < k * r; mb1.flat[i] = 0.0; i = i + 1; end
  TinyNN.upload_row_major(sess, t_B1, mb1)
  mk1 = Mat.new(max_t, k); i = 0
  while i < max_t * k; mk1.flat[i] = next_normal(0.3); i = i + 1; end
  TinyNN.upload_row_major(sess, t_K1, mk1)
  mv1 = Mat.new(k, max_t); i = 0
  while i < k * max_t; mv1.flat[i] = next_normal(0.3); i = i + 1; end
  TinyNN.upload_row_major(sess, t_V1, mv1)

  m_wo = Mat.new(d_model, d_model); i = 0
  while i < d_model * d_model; m_wo.flat[i] = next_normal(0.3); i = i + 1; end
  TinyNN.upload_row_major(sess, t_W_o, m_wo)
  m_hp = Mat.new(1, 2); m_hp.flat[0] = lr; m_hp.flat[1] = 0.0
  TinyNN.upload_row_major(sess, t_hp, m_hp)
  m_x = Mat.new(d_model, 1); i = 0
  while i < d_model; m_x.flat[i] = 0.5 + i.to_f * 0.3; i = i + 1; end
  TinyNN.upload_row_major(sess, t_x, m_x)
  m_x
end

m_x = upload_init_weights(sess, t_W_q0, t_A0, t_B0, t_K0, t_V0,
                                t_W_q1, t_A1, t_B1, t_K1, t_V1, t_W_o,
                                t_hp, t_x, K, D_MODEL, R, MAX_T, LR)

losses = []
s = 0
while s < STEPS
  TinyNN.tnn_graph_reset(sess)
  TinyNN.upload_row_major(sess, t_x, m_x)
  TinyNN.upload_int_array(sess, t_pos, [0])
  TinyNN.tnn_compute_backward(sess)
  TinyNN.tnn_download(sess, t_loss)
  losses.push(TinyNN.tnn_scratch_get(sess, 0))
  if s == 0 || (s + 1) % 5 == 0
    puts "step " + (s + 1).to_s + ": loss=" + losses[s].to_s
  end
  s = s + 1
end
TinyNN.tnn_session_free(sess)
puts ""
puts "initial=" + losses[0].to_s + " final=" + losses[STEPS - 1].to_s
