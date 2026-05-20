# Task #70 bisect step 4: replace contiguous K/V with views of larger
# persistent buffers. Matches the SmolLM2 KV-cache pattern where
# K_hist = view_2d(t_K_cache, d_head, pos+1, ...).

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

K     = 8
T_CTX = 4
MAX_T = 16
R     = 4
STEPS = 30
LR    = 0.001

sess = TinyNN.tnn_session_new(0)
$seed = 42
def next_normal(scale)
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u1 = ($seed.to_f + 1.0) / 2147483648.0
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u2 = ($seed.to_f + 1.0) / 2147483648.0
  scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

t_W_q  = TinyNN.tnn_input_2d_f32_persistent(sess, K, K)
t_A    = TinyNN.tnn_input_2d_f32_persistent(sess, R, K)
t_B    = TinyNN.tnn_input_2d_f32_persistent(sess, K, R)
# Larger backing buffers; K/V "history" is a view of the first T_CTX rows/cols.
t_K_cache = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T, K)   # (K, MAX_T)
t_V_cache = TinyNN.tnn_input_2d_f32_persistent(sess, K, MAX_T)   # (MAX_T, K)
t_hp   = TinyNN.tnn_input_1d_f32_persistent(sess, 2)
TinyNN.tnn_set_param(t_A); TinyNN.tnn_set_param(t_B)
TinyNN.tnn_finalize_weights(sess)

t_x    = TinyNN.tnn_input_2d_f32(sess, 1, K)
t_pos  = TinyNN.tnn_input_1d_i32(sess, 1)

t_ax    = TinyNN.tnn_matmul(sess, t_A, t_x)
t_bax   = TinyNN.tnn_matmul(sess, t_B, t_ax)
t_qbase = TinyNN.tnn_matmul(sess, t_W_q, t_x)
t_qraw  = TinyNN.tnn_add(sess, t_qbase, t_bax)
t_q     = TinyNN.tnn_rope_ext(sess, t_qraw, t_pos, K, 10000.0)

# View slice — same pattern as SmolLM2's build_attention_qhead_step.
bytes_d_head = K * 4         # sizeof(float) * d_head
bytes_max_T  = MAX_T * 4
t_K_view = TinyNN.tnn_view_2d(sess, t_K_cache, K, T_CTX, bytes_d_head, 0)
t_V_view = TinyNN.tnn_view_2d(sess, t_V_cache, T_CTX, K, bytes_max_T, 0)

scale_v = 1.0 / Math.sqrt(K.to_f)
t_scores = TinyNN.tnn_matmul(sess, t_K_view, t_q)
t_scaled = TinyNN.tnn_scale(sess, t_scores, scale_v)
t_attn   = TinyNN.tnn_softmax(sess, t_scaled)
t_head   = TinyNN.tnn_matmul(sess, t_V_view, t_attn)
t_h_sq   = TinyNN.tnn_mul(sess, t_head, t_head)
t_loss   = TinyNN.tnn_sum(sess, t_h_sq)

TinyNN.tnn_set_output(t_A); TinyNN.tnn_set_output(t_B)
TinyNN.tnn_set_output(t_loss); TinyNN.tnn_set_loss(t_loss)

TinyNN.tnn_build_forward_only(sess, t_loss)
TinyNN.tnn_build_backward(sess)
t_grad_A = TinyNN.tnn_tensor_grad(sess, t_A)
t_grad_B = TinyNN.tnn_tensor_grad(sess, t_B)
t_opt_A  = TinyNN.tnn_opt_step_sgd(sess, t_A, t_grad_A, t_hp)
t_opt_B  = TinyNN.tnn_opt_step_sgd(sess, t_B, t_grad_B, t_hp)
TinyNN.tnn_extend_backward_graph(sess, t_opt_A)
TinyNN.tnn_extend_backward_graph(sess, t_opt_B)
TinyNN.tnn_realize_backward(sess)

m_W = Mat.new(K, K); i = 0
while i < K * K; m_W.flat[i] = next_normal(0.5); i = i + 1; end
TinyNN.upload_row_major(sess, t_W_q, m_W)
m_A = Mat.new(R, K); i = 0
while i < R * K; m_A.flat[i] = next_normal(0.1); i = i + 1; end
TinyNN.upload_row_major(sess, t_A, m_A)
m_B = Mat.new(K, R); i = 0
while i < K * R; m_B.flat[i] = 0.0; i = i + 1; end
TinyNN.upload_row_major(sess, t_B, m_B)
# Fill K_cache + V_cache; the parts beyond T_CTX are unused noise.
m_K = Mat.new(MAX_T, K); i = 0
while i < MAX_T * K; m_K.flat[i] = next_normal(0.3); i = i + 1; end
TinyNN.upload_row_major(sess, t_K_cache, m_K)
m_V = Mat.new(K, MAX_T); i = 0
while i < K * MAX_T; m_V.flat[i] = next_normal(0.3); i = i + 1; end
TinyNN.upload_row_major(sess, t_V_cache, m_V)
m_hp = Mat.new(1, 2); m_hp.flat[0] = LR; m_hp.flat[1] = 0.0
TinyNN.upload_row_major(sess, t_hp, m_hp)
m_x = Mat.new(K, 1); i = 0
while i < K; m_x.flat[i] = 0.5 + i.to_f * 0.3; i = i + 1; end
TinyNN.upload_row_major(sess, t_x, m_x)

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
