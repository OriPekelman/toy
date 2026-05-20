# F1.1 micro #4 — forward-only with matmul. No backward, no optimizer.
# Just: y = matmul(W, x); loss = y * y; download loss.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

K = 2

sess = TinyNN.tnn_session_new(0)
t_W = TinyNN.tnn_input_2d_f32_persistent(sess, 1, K)
TinyNN.tnn_finalize_weights(sess)

t_x = TinyNN.tnn_input_2d_f32(sess, 1, K)
t_y = TinyNN.tnn_matmul(sess, t_W, t_x)
t_loss = TinyNN.tnn_mul(sess, t_y, t_y)
TinyNN.tnn_set_output(t_y)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_realize(sess, t_loss)

m_W = Mat.new(K, 1); m_W.flat[0] = 1.0; m_W.flat[1] = 2.0
TinyNN.stage_row_major_and_upload(sess, t_W, m_W)

m_x = Mat.new(K, 1); m_x.flat[0] = 3.0; m_x.flat[1] = 4.0
TinyNN.stage_row_major_and_upload(sess, t_x, m_x)

TinyNN.tnn_compute(sess)

TinyNN.tnn_download(sess, t_y)
puts "y     = " + TinyNN.tnn_scratch_get(sess, 0).to_s + "  (expected 11)"
TinyNN.tnn_download(sess, t_loss)
puts "loss  = " + TinyNN.tnn_scratch_get(sess, 0).to_s + "  (expected 121)"

TinyNN.tnn_session_free(sess)
