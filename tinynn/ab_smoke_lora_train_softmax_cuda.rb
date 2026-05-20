# Task #70 bisect step 3: add full attention chain (no view yet).
#
# Forward:  q_raw = (W + B@A) @ x        ; q_eff = rope(q_raw)
#           k = w_k @ x_static           ; v = w_v @ x_static
#           scores = mul_mat(k, q_eff)   ; scaled = scale(scores, 1/sqrt(d))
#           attn = softmax(scaled)
#           head = mul_mat(v, attn)
#           loss = sum(head²)
#
# Exercises: softmax_back, scale_back (trivial), mul_mat through a
# k/v intermediate, plus rope_ext_back. No views, no concat, no
# diag_mask. If CPU/CUDA agree we know it's a view-or-concat issue.

require_relative "../lib/transformer"
require_relative "../lib/tinynn_cuda"

K     = 8         # d_head
T_CTX = 4         # length of "K/V cache"
R     = 4
STEPS = 30
LR    = 0.001

sess = TinyNNCuda.tnn_session_new(0)
$seed = 42
def next_normal(scale)
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u1 = ($seed.to_f + 1.0) / 2147483648.0
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u2 = ($seed.to_f + 1.0) / 2147483648.0
  scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

# Persistent: W_q (frozen), A, B (trainable LoRA), K (frozen "cache"), V (frozen "cache").
t_W_q  = TinyNNCuda.tnn_input_2d_f32_persistent(sess, K, K)        # (K, K) — d_head out
t_A    = TinyNNCuda.tnn_input_2d_f32_persistent(sess, R, K)
t_B    = TinyNNCuda.tnn_input_2d_f32_persistent(sess, K, R)
t_Kc   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, T_CTX, K)    # (K, T) — fake K cache
t_Vc   = TinyNNCuda.tnn_input_2d_f32_persistent(sess, K, T_CTX)    # (T, K) — fake V cache  (transposed layout matches SmolLM2)
t_hp   = TinyNNCuda.tnn_input_1d_f32_persistent(sess, 2)
TinyNNCuda.tnn_set_param(t_A); TinyNNCuda.tnn_set_param(t_B)
TinyNNCuda.tnn_finalize_weights(sess)

t_x    = TinyNNCuda.tnn_input_2d_f32(sess, 1, K)
t_pos  = TinyNNCuda.tnn_input_1d_i32(sess, 1)

t_ax    = TinyNNCuda.tnn_matmul(sess, t_A, t_x)
t_bax   = TinyNNCuda.tnn_matmul(sess, t_B, t_ax)
t_qbase = TinyNNCuda.tnn_matmul(sess, t_W_q, t_x)
t_qraw  = TinyNNCuda.tnn_add(sess, t_qbase, t_bax)
t_q     = TinyNNCuda.tnn_rope_ext(sess, t_qraw, t_pos, K, 10000.0)

scale_v = 1.0 / Math.sqrt(K.to_f)
t_scores = TinyNNCuda.tnn_matmul(sess, t_Kc, t_q)               # ne=[T, 1]
t_scaled = TinyNNCuda.tnn_scale(sess, t_scores, scale_v)
t_attn   = TinyNNCuda.tnn_softmax(sess, t_scaled)              # along ne0=T
t_head   = TinyNNCuda.tnn_matmul(sess, t_Vc, t_attn)            # ne=[K, 1]
t_h_sq   = TinyNNCuda.tnn_mul(sess, t_head, t_head)
t_loss   = TinyNNCuda.tnn_sum(sess, t_h_sq)

TinyNNCuda.tnn_set_output(t_A); TinyNNCuda.tnn_set_output(t_B)
TinyNNCuda.tnn_set_output(t_loss); TinyNNCuda.tnn_set_loss(t_loss)

TinyNNCuda.tnn_build_forward_only(sess, t_loss)
TinyNNCuda.tnn_build_backward(sess)
t_grad_A = TinyNNCuda.tnn_tensor_grad(sess, t_A)
t_grad_B = TinyNNCuda.tnn_tensor_grad(sess, t_B)
t_opt_A  = TinyNNCuda.tnn_opt_step_sgd(sess, t_A, t_grad_A, t_hp)
t_opt_B  = TinyNNCuda.tnn_opt_step_sgd(sess, t_B, t_grad_B, t_hp)
TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_A)
TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_B)
TinyNNCuda.tnn_realize_backward(sess)

m_W = Mat.new(K, K); i = 0
while i < K * K; m_W.flat[i] = next_normal(0.5); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_W_q, m_W)
m_A = Mat.new(R, K); i = 0
while i < R * K; m_A.flat[i] = next_normal(0.1); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_A, m_A)
m_B = Mat.new(K, R); i = 0
while i < K * R; m_B.flat[i] = 0.0; i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_B, m_B)
m_K = Mat.new(T_CTX, K); i = 0
while i < T_CTX * K; m_K.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_Kc, m_K)
m_V = Mat.new(K, T_CTX); i = 0
while i < K * T_CTX; m_V.flat[i] = next_normal(0.3); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_Vc, m_V)
m_hp = Mat.new(1, 2); m_hp.flat[0] = LR; m_hp.flat[1] = 0.0
TinyNNCuda.upload_row_major(sess, t_hp, m_hp)
m_x = Mat.new(K, 1); i = 0
while i < K; m_x.flat[i] = 0.5 + i.to_f * 0.3; i = i + 1; end
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
