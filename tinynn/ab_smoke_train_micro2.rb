# F1.1 micro-smoke #2: introduce matmul.
#
# y = matmul(W, x);  W is 1×1, x is 1×1; both behave as scalars.
# loss = y * y.
#
#   Hand: y = W*x = 1*2 = 2; loss = 4
#         ∂L/∂y = 2y = 4
#         ∂L/∂W = ∂L/∂y * x = 4 * 2 = 8
#         SGD W' = 1 - 0.01*8 = 0.92
#
# If W → 0.92 → matmul backward works at scalar shape; sum reduction
# is the next thing to bisect.
# If W → something else → matmul backward path has a shape bug.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

LR = 0.01

sess = TinyNN.tnn_session_new(0)
t_W  = TinyNN.tnn_input_2d_f32_persistent(sess, 1, 1)
t_hp = TinyNN.tnn_input_1d_f32_persistent(sess, 2)
TinyNN.tnn_set_param(t_W)
TinyNN.tnn_finalize_weights(sess)

t_x  = TinyNN.tnn_input_2d_f32(sess, 1, 1)
t_y  = TinyNN.tnn_matmul(sess, t_W, t_x)
t_loss = TinyNN.tnn_mul(sess, t_y, t_y)
TinyNN.tnn_set_output(t_y)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)
TinyNN.tnn_build_forward_only(sess, t_loss)

TinyNN.tnn_build_backward(sess)
t_grad = TinyNN.tnn_tensor_grad(sess, t_W)
t_opt = TinyNN.tnn_opt_step_sgd(sess, t_W, t_grad, t_hp)
TinyNN.tnn_extend_backward_graph(sess, t_opt)
TinyNN.tnn_realize_backward(sess)

# W=1, x=2, hp=[lr, wd]
m_W = Mat.new(1, 1); m_W.flat[0] = 1.0
TinyNN.stage_row_major_and_upload(sess, t_W, m_W)
m_hp = Mat.new(1, 2); m_hp.flat[0] = LR; m_hp.flat[1] = 0.0
TinyNN.stage_row_major_and_upload(sess, t_hp, m_hp)
TinyNN.tnn_graph_reset(sess)

m_x = Mat.new(1, 1); m_x.flat[0] = 2.0
TinyNN.stage_row_major_and_upload(sess, t_x, m_x)

# Diagnostic: read W and x back before compute
TinyNN.tnn_download(sess, t_W)
puts "pre-compute W = " + TinyNN.tnn_scratch_get(sess, 0).to_s
TinyNN.tnn_download(sess, t_x)
puts "pre-compute x = " + TinyNN.tnn_scratch_get(sess, 0).to_s

rc = TinyNN.tnn_compute_backward(sess)
puts "compute rc=" + rc.to_s

TinyNN.tnn_download(sess, t_W)
w_after = TinyNN.tnn_scratch_get(sess, 0)
TinyNN.tnn_download(sess, t_loss)
loss_v = TinyNN.tnn_scratch_get(sess, 0)
TinyNN.tnn_download(sess, t_grad)
grad_v = TinyNN.tnn_scratch_get(sess, 0)

puts ""
puts "After 1 SGD step (W=1, x=2):"
puts "  W       = " + w_after.to_s + "  (expected 0.92)"
puts "  loss    = " + loss_v.to_s + "  (expected 4.0)"
puts "  grad    = " + grad_v.to_s + "  (expected 8.0)"

diff = (w_after - 0.92).abs
puts ""
if diff < 1.0e-5
  puts "VERDICT: matmul backward works at scalar shape."
  puts "  Implication: bug is in sum-reduce or larger-shape matmul."
else
  puts "VERDICT: matmul backward off by " + diff.to_s
  puts "  Implication: shape-related issue in matmul gradient chain."
end

TinyNN.tnn_session_free(sess)
