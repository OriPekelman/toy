# SGD sanity check — same toy graph as ab_smoke_train_step but uses
# tnn_opt_step_sgd (no Adam momentum). Lets us isolate gradient
# direction from optimizer state.
#
# w = w - lr * grad - lr * wd * w
#
# Toy: y = W @ x; loss = sum(y²); W is param. With x fixed positive
# and gradient = 2*y*x^T (always pointing toward larger |y|),
# SGD subtracting lr * gradient should reduce |y| → reduce loss.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

K = 3
OUT = 2
LR = 0.01
STEPS = 30

sess = TinyNN.tnn_session_new(0)

t_W  = TinyNN.tnn_input_2d_f32_persistent(sess, OUT, K)
t_hp = TinyNN.tnn_input_1d_f32_persistent(sess, 2)   # [alpha, wd]
TinyNN.tnn_set_param(t_W)
TinyNN.tnn_finalize_weights(sess)

t_x = TinyNN.tnn_input_2d_f32(sess, 1, K)

t_y    = TinyNN.tnn_matmul(sess, t_W, t_x)
t_y2   = TinyNN.tnn_mul(sess, t_y, t_y)
t_loss = TinyNN.tnn_sum(sess, t_y2)

TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)
TinyNN.tnn_realize(sess, t_loss)

rc = TinyNN.tnn_build_backward(sess)
if rc != 0; puts "build_backward rc=" + rc.to_s; exit 1; end

t_grad = TinyNN.tnn_tensor_grad(sess, t_W)
t_opt  = TinyNN.tnn_opt_step_sgd(sess, t_W, t_grad, t_hp)
TinyNN.tnn_extend_backward_graph(sess, t_opt)
TinyNN.tnn_realize_backward(sess)

m_W = Mat.new(K, OUT)
m_W.flat[0] = -0.2
m_W.flat[1] =  0.3
m_W.flat[2] =  0.5
m_W.flat[3] = -0.1
m_W.flat[4] =  0.4
m_W.flat[5] =  0.1
TinyNN.stage_row_major_and_upload(sess, t_W, m_W)
TinyNN.tnn_graph_reset(sess)

m_x = Mat.new(1, K)
m_x.flat[0] = 1.0
m_x.flat[1] = 2.0
m_x.flat[2] = 3.0

m_hp = Mat.new(1, 2)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.0      # weight decay
TinyNN.stage_row_major_and_upload(sess, t_hp, m_hp)

# Diagnostic: dump uploaded W and x BEFORE first compute
TinyNN.stage_row_major_and_upload(sess, t_x, m_x)
TinyNN.tnn_download(sess, t_W)
puts "W pre-compute: " + (0..5).map { |i| TinyNN.tnn_scratch_get(sess, i) }.inspect

t = 1
while t <= STEPS
  TinyNN.stage_row_major_and_upload(sess, t_x, m_x)
  rc = TinyNN.tnn_compute_backward(sess)
  if rc != 0; puts "compute rc=" + rc.to_s; exit 1; end
  TinyNN.tnn_download(sess, t_loss)
  loss = TinyNN.tnn_scratch_get(sess, 0)
  if t == 1 || t == 2 || t == 3 || t % 5 == 0 || t == STEPS
    TinyNN.tnn_download(sess, t_W)
    w0 = TinyNN.tnn_scratch_get(sess, 0)
    w5 = TinyNN.tnn_scratch_get(sess, 5)
    TinyNN.tnn_download(sess, t_grad)
    g0 = TinyNN.tnn_scratch_get(sess, 0)
    g5 = TinyNN.tnn_scratch_get(sess, 5)
    puts "step " + t.to_s + ": loss=" + loss.to_s + " W[0]=" + w0.to_s + " W[5]=" + w5.to_s + " grad[0]=" + g0.to_s + " grad[5]=" + g5.to_s
  end
  t = t + 1
end

TinyNN.tnn_session_free(sess)
