# F1.1 micro #5 — F0.4-shaped setup at small shape. Backward only.
# No optimizer. Just to see if loss reads correctly.
#
# Same as micro #4 (forward-only matmul) + tnn_build_backward.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

K = 2

sess = TinyNN.tnn_session_new(0)
t_W = TinyNN.tnn_input_2d_f32_persistent(sess, 1, K)
TinyNN.tnn_set_param(t_W)
TinyNN.tnn_finalize_weights(sess)

t_x = TinyNN.tnn_input_2d_f32(sess, 1, K)
t_y = TinyNN.tnn_matmul(sess, t_W, t_x)
t_loss = TinyNN.tnn_mul(sess, t_y, t_y)
TinyNN.tnn_set_output(t_y)         # protect y's slot from sched aliasing
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)
TinyNN.tnn_build_forward_only(sess, t_loss)

TinyNN.tnn_build_backward(sess)
TinyNN.tnn_realize_backward(sess)

# Try graph_reset BEFORE uploads — in case reset clobbers them.
TinyNN.tnn_graph_reset(sess)

m_W = Mat.new(K, 1); m_W.flat[0] = 1.0; m_W.flat[1] = 2.0
TinyNN.stage_row_major_and_upload(sess, t_W, m_W)
m_x = Mat.new(K, 1); m_x.flat[0] = 3.0; m_x.flat[1] = 4.0
TinyNN.stage_row_major_and_upload(sess, t_x, m_x)

rc = TinyNN.tnn_compute_backward(sess)
puts "compute_backward rc=" + rc.to_s

TinyNN.tnn_download(sess, t_y)
puts "y    = " + TinyNN.tnn_scratch_get(sess, 0).to_s + "   (expected 11)"
TinyNN.tnn_download(sess, t_loss)
puts "loss = " + TinyNN.tnn_scratch_get(sess, 0).to_s + "   (expected 121)"
t_grad = TinyNN.tnn_tensor_grad(sess, t_W)
TinyNN.tnn_download(sess, t_grad)
puts "grad = [" + TinyNN.tnn_scratch_get(sess, 0).to_s + ", " + TinyNN.tnn_scratch_get(sess, 1).to_s + "]   (expected [66, 88])"

TinyNN.tnn_session_free(sess)
