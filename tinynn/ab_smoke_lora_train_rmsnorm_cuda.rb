# Task #70 bisect step 1: toy LoRA training + rms_norm. If CPU and
# CUDA still match here, rms_norm chain is innocent.
#
# Forward:  y = (W + B@A) @ x;  yn = rms_norm(y, gamma);  loss = sum(yn²)
# Backward: ... rms_norm_back ... matmul_back ... opt_step on A, B.
#
# Same shape (K=8 OUT=8 R=4 60 steps) as ab_smoke_lora_train. CPU and
# CUDA should agree to <= ULP if rms_norm backward is symmetric.

require_relative "../lib/transformer"
require_relative "../lib/tinynn_cuda"

K     = 8
OUT   = 8
R     = 4
STEPS = 60
LR    = 0.002

sess = TinyNNCuda.tnn_session_new(0)

$seed = 42
def next_normal(scale)
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u1 = ($seed.to_f + 1.0) / 2147483648.0
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u2 = ($seed.to_f + 1.0) / 2147483648.0
  scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

t_W     = TinyNNCuda.tnn_input_2d_f32_persistent(sess, OUT, K)
t_A     = TinyNNCuda.tnn_input_2d_f32_persistent(sess, R,   K)
t_B     = TinyNNCuda.tnn_input_2d_f32_persistent(sess, OUT, R)
t_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(sess, OUT)
t_hp    = TinyNNCuda.tnn_input_1d_f32_persistent(sess, 2)
TinyNNCuda.tnn_set_param(t_A); TinyNNCuda.tnn_set_param(t_B)
TinyNNCuda.tnn_finalize_weights(sess)

t_x     = TinyNNCuda.tnn_input_2d_f32(sess, 1, K)
t_ax    = TinyNNCuda.tnn_matmul(sess, t_A, t_x)
t_bax   = TinyNNCuda.tnn_matmul(sess, t_B, t_ax)
t_ybase = TinyNNCuda.tnn_matmul(sess, t_W, t_x)
t_y     = TinyNNCuda.tnn_add(sess, t_ybase, t_bax)
t_yn    = TinyNNCuda.tnn_rms_norm(sess, t_y, t_gamma, 1.0e-5)
t_yn_sq = TinyNNCuda.tnn_mul(sess, t_yn, t_yn)
t_loss  = TinyNNCuda.tnn_sum(sess, t_yn_sq)

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

m_W = Mat.new(OUT, K); i = 0
while i < OUT * K; m_W.flat[i] = next_normal(0.5); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_W, m_W)
m_A = Mat.new(R, K); i = 0
while i < R * K; m_A.flat[i] = next_normal(0.1); i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_A, m_A)
m_B = Mat.new(OUT, R); i = 0
while i < OUT * R; m_B.flat[i] = 0.0; i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_B, m_B)
m_gamma = Mat.new(1, OUT); i = 0
while i < OUT; m_gamma.flat[i] = 1.0; i = i + 1; end
TinyNNCuda.upload_row_major(sess, t_gamma, m_gamma)
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
  TinyNNCuda.tnn_compute_backward(sess)
  TinyNNCuda.tnn_download(sess, t_loss)
  losses.push(TinyNNCuda.tnn_scratch_get(sess, 0))
  if s == 0 || (s + 1) % 10 == 0
    puts "step " + (s + 1).to_s + ": loss=" + losses[s].to_s
  end
  s = s + 1
end
TinyNNCuda.tnn_session_free(sess)

puts ""
puts "initial=" + losses[0].to_s + " final=" + losses[STEPS - 1].to_s
