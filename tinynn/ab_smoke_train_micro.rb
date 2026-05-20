# F1.1 micro-smoke: bisects the autograd value-mismatch.
#
# Smallest possible training step: single scalar param W, loss = W².
# No matmul. No sum-reduce. No multi-element.
#
#   Hand:  ∂(W²)/∂W = 2W
#          One SGD step at lr=0.01 from W=1.0:
#          W' = 1.0 - 0.01 * (2 * 1.0) = 0.98
#
# Three possible outcomes:
#   (a) W reads back as 0.98          → ggml autograd works; the bug
#                                        in ab_smoke_train_step is in
#                                        our matmul/sum chain.
#   (b) W reads back as something else → ggml or our wrapper bug at
#                                        the basement; everything
#                                        above this is downstream.
#   (c) compile/runtime fail           → tooling regression.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

LR = 0.01

sess = TinyNN.tnn_session_new(0)

# 1×1 param (a 1-element 2-D tensor is the smallest shape we can build
# with tnn_input_2d_f32_persistent).
t_W  = TinyNN.tnn_input_2d_f32_persistent(sess, 1, 1)
t_hp = TinyNN.tnn_input_1d_f32_persistent(sess, 2)   # SGD: [alpha, wd]
TinyNN.tnn_set_param(t_W)
TinyNN.tnn_finalize_weights(sess)

# Loss = W * W. tnn_mul is element-wise on same-shape tensors.
t_loss = TinyNN.tnn_mul(sess, t_W, t_W)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)
TinyNN.tnn_realize(sess, t_loss)

# Backward graph + SGD step in-graph.
rc = TinyNN.tnn_build_backward(sess)
if rc != 0; puts "build_backward rc=" + rc.to_s; exit 1; end
t_grad = TinyNN.tnn_tensor_grad(sess, t_W)
t_opt  = TinyNN.tnn_opt_step_sgd(sess, t_W, t_grad, t_hp)
TinyNN.tnn_extend_backward_graph(sess, t_opt)
TinyNN.tnn_realize_backward(sess)

# Upload W=1.0, hp=[lr, wd].
m_W = Mat.new(1, 1)
m_W.flat[0] = 1.0
TinyNN.stage_row_major_and_upload(sess, t_W, m_W)

m_hp = Mat.new(1, 2)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.0
TinyNN.stage_row_major_and_upload(sess, t_hp, m_hp)

# Reset grads (and any momenta — SGD has none, but reset also seeds
# loss_grad = 1 which we need).
TinyNN.tnn_graph_reset(sess)

# Pre-compute W (should still be 1.0).
TinyNN.tnn_download(sess, t_W)
puts "W before compute: " + TinyNN.tnn_scratch_get(sess, 0).to_s

# Run one step.
rc = TinyNN.tnn_compute_backward(sess)
if rc != 0; puts "compute rc=" + rc.to_s; exit 1; end

# Read W (post-step), loss (this iteration), grad (this iteration).
TinyNN.tnn_download(sess, t_W)
w_after = TinyNN.tnn_scratch_get(sess, 0)
TinyNN.tnn_download(sess, t_loss)
loss_v = TinyNN.tnn_scratch_get(sess, 0)
TinyNN.tnn_download(sess, t_grad)
grad_v = TinyNN.tnn_scratch_get(sess, 0)

puts ""
puts "After 1 SGD step at lr=" + LR.to_s + ":"
puts "  W       = " + w_after.to_s + "  (expected 0.98)"
puts "  loss    = " + loss_v.to_s + "  (expected 1.0)"
puts "  grad    = " + grad_v.to_s + "  (expected 2.0)"

# Verdict
expected_w = 0.98
diff = (w_after - expected_w).abs
puts ""
if diff < 1.0e-5
  puts "VERDICT (a): ggml autograd works at scalar shape."
  puts "  Implication: bug in ab_smoke_train_step is in matmul or sum chain."
else
  puts "VERDICT (b): basement-level mismatch — |W - 0.98| = " + diff.to_s
  puts "  Implication: deeper issue. Investigate ggml side or our wrapper."
end

TinyNN.tnn_session_free(sess)
