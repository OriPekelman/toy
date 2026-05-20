# Task #70 bisect step 6: full single-block (attention concat + FFN).
# Tests if SwiGLU FFN backward (silu + mul + matmul) introduces the
# CPU/CUDA divergence.

require_relative "../lib/transformer"
require_relative "../lib/tinynn_cuda"

K       = 8
D_MODEL = K * 2
D_FF    = 32
T_CTX   = 4
MAX_T   = 16
R       = 4
STEPS   = 30
LR      = 0.001

sess = TinyNNCuda.tnn_session_new(0)
$seed = 42
def next_normal(scale)
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u1 = ($seed.to_f + 1.0) / 2147483648.0
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u2 = ($seed.to_f + 1.0) / 2147483648.0
  scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

t_W_q0 = TinyNNCuda.tnn_input_2d_f32_persistent(sess, K, D_MODEL)
t_A0   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, R, D_MODEL)
t_B0   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, K, R)
t_K0   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, MAX_T, K)
t_V0   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, K, MAX_T)
t_W_q1 = TinyNNCuda.tnn_input_2d_f32_persistent(sess, K, D_MODEL)
t_A1   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, R, D_MODEL)
t_B1   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, K, R)
t_K1   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, MAX_T, K)
t_V1   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, K, MAX_T)
t_W_o    = TinyNNCuda.tnn_input_2d_f32_persistent(sess, D_MODEL, D_MODEL)
t_rn2    = TinyNNCuda.tnn_input_1d_f32_persistent(sess, D_MODEL)
t_w_gate = TinyNNCuda.tnn_input_2d_f32_persistent(sess, D_FF, D_MODEL)
t_w_up   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, D_FF, D_MODEL)
t_w_down = TinyNNCuda.tnn_input_2d_f32_persistent(sess, D_MODEL, D_FF)
t_hp     = TinyNNCuda.tnn_input_1d_f32_persistent(sess, 2)
TinyNNCuda.tnn_set_param(t_A0); TinyNNCuda.tnn_set_param(t_B0)
TinyNNCuda.tnn_set_param(t_A1); TinyNNCuda.tnn_set_param(t_B1)
TinyNNCuda.tnn_finalize_weights(sess)

t_x   = TinyNNCuda.tnn_input_2d_f32(sess, 1, D_MODEL)
t_pos = TinyNNCuda.tnn_input_1d_i32(sess, 1)
bytes_d_head = K * 4
bytes_max_T  = MAX_T * 4
scale_v      = 1.0 / Math.sqrt(K.to_f)

# Head 0
t_ax0 = TinyNNCuda.tnn_matmul(sess, t_A0, t_x)
t_bax0 = TinyNNCuda.tnn_matmul(sess, t_B0, t_ax0)
t_qbase0 = TinyNNCuda.tnn_matmul(sess, t_W_q0, t_x)
t_qraw0 = TinyNNCuda.tnn_add(sess, t_qbase0, t_bax0)
t_q0 = TinyNNCuda.tnn_rope_ext(sess, t_qraw0, t_pos, K, 10000.0)
t_Kv0 = TinyNNCuda.tnn_view_2d(sess, t_K0, K, T_CTX, bytes_d_head, 0)
t_Vv0 = TinyNNCuda.tnn_view_2d(sess, t_V0, T_CTX, K, bytes_max_T, 0)
t_sc0 = TinyNNCuda.tnn_matmul(sess, t_Kv0, t_q0)
t_scl0 = TinyNNCuda.tnn_scale(sess, t_sc0, scale_v)
t_attn0 = TinyNNCuda.tnn_softmax(sess, t_scl0)
t_head0 = TinyNNCuda.tnn_matmul(sess, t_Vv0, t_attn0)
# Head 1
t_ax1 = TinyNNCuda.tnn_matmul(sess, t_A1, t_x)
t_bax1 = TinyNNCuda.tnn_matmul(sess, t_B1, t_ax1)
t_qbase1 = TinyNNCuda.tnn_matmul(sess, t_W_q1, t_x)
t_qraw1 = TinyNNCuda.tnn_add(sess, t_qbase1, t_bax1)
t_q1 = TinyNNCuda.tnn_rope_ext(sess, t_qraw1, t_pos, K, 10000.0)
t_Kv1 = TinyNNCuda.tnn_view_2d(sess, t_K1, K, T_CTX, bytes_d_head, 0)
t_Vv1 = TinyNNCuda.tnn_view_2d(sess, t_V1, T_CTX, K, bytes_max_T, 0)
t_sc1 = TinyNNCuda.tnn_matmul(sess, t_Kv1, t_q1)
t_scl1 = TinyNNCuda.tnn_scale(sess, t_sc1, scale_v)
t_attn1 = TinyNNCuda.tnn_softmax(sess, t_scl1)
t_head1 = TinyNNCuda.tnn_matmul(sess, t_Vv1, t_attn1)

t_concat = TinyNNCuda.tnn_concat(sess, t_head0, t_head1, 0)
t_attn_out = TinyNNCuda.tnn_matmul(sess, t_W_o, t_concat)
t_x_attn = TinyNNCuda.tnn_add(sess, t_x, t_attn_out)        # residual

# FFN block: rms_norm → SwiGLU → matmul down
t_h2     = TinyNNCuda.tnn_rms_norm(sess, t_x_attn, t_rn2, 1.0e-5)
t_gate   = TinyNNCuda.tnn_matmul(sess, t_w_gate, t_h2)
t_up     = TinyNNCuda.tnn_matmul(sess, t_w_up, t_h2)
t_gate_s = TinyNNCuda.tnn_silu(sess, t_gate)
t_swglu  = TinyNNCuda.tnn_mul(sess, t_gate_s, t_up)
t_ffn    = TinyNNCuda.tnn_matmul(sess, t_w_down, t_swglu)
t_x_out  = TinyNNCuda.tnn_add(sess, t_x_attn, t_ffn)

t_o_sq = TinyNNCuda.tnn_mul(sess, t_x_out, t_x_out)
t_loss = TinyNNCuda.tnn_sum(sess, t_o_sq)

TinyNNCuda.tnn_set_output(t_A0); TinyNNCuda.tnn_set_output(t_B0)
TinyNNCuda.tnn_set_output(t_A1); TinyNNCuda.tnn_set_output(t_B1)
TinyNNCuda.tnn_set_output(t_loss); TinyNNCuda.tnn_set_loss(t_loss)

TinyNNCuda.tnn_build_forward_only(sess, t_loss)
TinyNNCuda.tnn_build_backward(sess)
t_gA0 = TinyNNCuda.tnn_tensor_grad(sess, t_A0)
t_gB0 = TinyNNCuda.tnn_tensor_grad(sess, t_B0)
t_gA1 = TinyNNCuda.tnn_tensor_grad(sess, t_A1)
t_gB1 = TinyNNCuda.tnn_tensor_grad(sess, t_B1)
TinyNNCuda.tnn_extend_backward_graph(sess, TinyNNCuda.tnn_opt_step_sgd(sess, t_A0, t_gA0, t_hp))
TinyNNCuda.tnn_extend_backward_graph(sess, TinyNNCuda.tnn_opt_step_sgd(sess, t_B0, t_gB0, t_hp))
TinyNNCuda.tnn_extend_backward_graph(sess, TinyNNCuda.tnn_opt_step_sgd(sess, t_A1, t_gA1, t_hp))
TinyNNCuda.tnn_extend_backward_graph(sess, TinyNNCuda.tnn_opt_step_sgd(sess, t_B1, t_gB1, t_hp))
TinyNNCuda.tnn_realize_backward(sess)

# Init
m = Mat.new(K, D_MODEL); i = 0
while i < K * D_MODEL; m.flat[i] = next_normal(0.5); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_W_q0, m)
ma = Mat.new(R, D_MODEL); i = 0
while i < R * D_MODEL; ma.flat[i] = next_normal(0.1); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_A0, ma)
mb = Mat.new(K, R); i = 0
while i < K * R; mb.flat[i] = 0.0; i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_B0, mb)
mk = Mat.new(MAX_T, K); i = 0
while i < MAX_T * K; mk.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_K0, mk)
mv = Mat.new(K, MAX_T); i = 0
while i < K * MAX_T; mv.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_V0, mv)
m1 = Mat.new(K, D_MODEL); i = 0
while i < K * D_MODEL; m1.flat[i] = next_normal(0.5); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_W_q1, m1)
ma1 = Mat.new(R, D_MODEL); i = 0
while i < R * D_MODEL; ma1.flat[i] = next_normal(0.1); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_A1, ma1)
mb1 = Mat.new(K, R); i = 0
while i < K * R; mb1.flat[i] = 0.0; i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_B1, mb1)
mk1 = Mat.new(MAX_T, K); i = 0
while i < MAX_T * K; mk1.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_K1, mk1)
mv1 = Mat.new(K, MAX_T); i = 0
while i < K * MAX_T; mv1.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_V1, mv1)
m_wo = Mat.new(D_MODEL, D_MODEL); i = 0
while i < D_MODEL * D_MODEL; m_wo.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_W_o, m_wo)
m_rn2 = Mat.new(1, D_MODEL); i = 0
while i < D_MODEL; m_rn2.flat[i] = 1.0; i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_rn2, m_rn2)
m_gate = Mat.new(D_FF, D_MODEL); i = 0
while i < D_FF * D_MODEL; m_gate.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_w_gate, m_gate)
m_up = Mat.new(D_FF, D_MODEL); i = 0
while i < D_FF * D_MODEL; m_up.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_w_up, m_up)
m_down = Mat.new(D_MODEL, D_FF); i = 0
while i < D_MODEL * D_FF; m_down.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_w_down, m_down)
m_hp = Mat.new(1, 2); m_hp.flat[0] = LR; m_hp.flat[1] = 0.0
TinyNNCuda.upload_row_major(sess, t_hp, m_hp)
m_x = Mat.new(D_MODEL, 1); i = 0
while i < D_MODEL; m_x.flat[i] = 0.5 + i.to_f * 0.3; i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_x, m_x)

losses = []
s = 0
while s < STEPS
  TinyNNCuda.tnn_graph_reset(sess)
  TinyNNCuda.upload_row_major(sess, t_x, m_x)
  TinyNNCuda.upload_int_array(sess, t_pos, [0])
  TinyNNCuda.tnn_compute_backward(sess)
  TinyNNCuda.tnn_download(sess, t_loss)
  losses.push(TinyNNCuda.tnn_scratch_get(sess, 0))
  if s == 0 || (s + 1) % 5 == 0
    puts "step " + (s + 1).to_s + ": loss=" + losses[s].to_s
  end
  s = s + 1
end
TinyNNCuda.tnn_session_free(sess)
puts ""
puts "initial=" + losses[0].to_s + " final=" + losses[STEPS - 1].to_s
