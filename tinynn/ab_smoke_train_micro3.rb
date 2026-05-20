# F1.1 micro-smoke #3: matmul with k=2 (instead of k=1 in micro2).
#
# W ne=[k=2, m=1] (2 elements), x ne=[k=2, n=1] (2 elements).
# y = matmul(W, x) → ne=[1, 1] (scalar-shaped 1-element).
# loss = y * y.
#
# Hand:  W = [1, 2], x = [3, 4]
#        y = 1*3 + 2*4 = 11
#        loss = 121
#        ∂L/∂W[0] = 2*y*x[0] = 2*11*3 = 66
#        ∂L/∂W[1] = 2*y*x[1] = 2*11*4 = 88
#        SGD W' = W - lr*grad
#        At lr=0.001: W'=[1-0.066, 2-0.088] = [0.934, 1.912]
require_relative "../lib/transformer"
require_relative "../lib/tinynn"

K = 2
LR = 0.001

sess = TinyNN.tnn_session_new(0)
# tnn_input_2d_f32_persistent(sess, rows, cols) → ne=[cols, rows].
# For W ne=[K, 1] we want rows=1, cols=K.
t_W  = TinyNN.tnn_input_2d_f32_persistent(sess, 1, K)
t_hp = TinyNN.tnn_input_1d_f32_persistent(sess, 2)
TinyNN.tnn_set_param(t_W)
TinyNN.tnn_finalize_weights(sess)

t_x  = TinyNN.tnn_input_2d_f32(sess, 1, K)               # ne=[K, 1]
t_y  = TinyNN.tnn_matmul(sess, t_W, t_x)                 # ne=[1, 1]
t_loss = TinyNN.tnn_mul(sess, t_y, t_y)
TinyNN.tnn_set_output(t_y)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)
TinyNN.tnn_realize(sess, t_loss)

rc = TinyNN.tnn_build_backward(sess)
puts "build rc=" + rc.to_s
t_grad = TinyNN.tnn_tensor_grad(sess, t_W)
t_opt = TinyNN.tnn_opt_step_sgd(sess, t_W, t_grad, t_hp)
rc = TinyNN.tnn_extend_backward_graph(sess, t_opt)
puts "extend rc=" + rc.to_s
rc = TinyNN.tnn_realize_backward(sess)
puts "realize rc=" + rc.to_s

m_W = Mat.new(K, 1); m_W.flat[0] = 1.0; m_W.flat[1] = 2.0
TinyNN.stage_row_major_and_upload(sess, t_W, m_W)
m_hp = Mat.new(1, 2); m_hp.flat[0] = LR; m_hp.flat[1] = 0.0
TinyNN.stage_row_major_and_upload(sess, t_hp, m_hp)
TinyNN.tnn_graph_reset(sess)

m_x = Mat.new(K, 1); m_x.flat[0] = 3.0; m_x.flat[1] = 4.0
TinyNN.stage_row_major_and_upload(sess, t_x, m_x)

# Pre-compute peek
TinyNN.tnn_download(sess, t_W)
puts "pre W = [" + TinyNN.tnn_scratch_get(sess, 0).to_s + ", " + TinyNN.tnn_scratch_get(sess, 1).to_s + "]"
TinyNN.tnn_download(sess, t_x)
puts "pre x = [" + TinyNN.tnn_scratch_get(sess, 0).to_s + ", " + TinyNN.tnn_scratch_get(sess, 1).to_s + "]"

rc = TinyNN.tnn_compute_backward(sess)
puts "compute rc=" + rc.to_s

TinyNN.tnn_download(sess, t_W)
puts ""
puts "Expected: y=11, loss=121, grad=[66, 88], W'=[0.934, 1.912]"
puts "  W       = [" + TinyNN.tnn_scratch_get(sess, 0).to_s + ", " + TinyNN.tnn_scratch_get(sess, 1).to_s + "]"
TinyNN.tnn_download(sess, t_y)
puts "  y       = " + TinyNN.tnn_scratch_get(sess, 0).to_s
TinyNN.tnn_download(sess, t_loss)
puts "  loss    = " + TinyNN.tnn_scratch_get(sess, 0).to_s
TinyNN.tnn_download(sess, t_grad)
puts "  grad    = [" + TinyNN.tnn_scratch_get(sess, 0).to_s + ", " + TinyNN.tnn_scratch_get(sess, 1).to_s + "]"

TinyNN.tnn_session_free(sess)
