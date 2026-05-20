# F1.1 — in-graph training step (mnist-example pattern).
#
# Builds forward + backward + adam_step as one cgraph. No host readback
# of intermediates: only the scalar loss is downloaded each iteration.
# The optimizer updates W in-place inside the compute call.
#
# Toy graph: y = W @ x; loss = sum(y²); W is the only param.
# Expected: loss → 0 as adam learns to drive y → 0.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

K = 3
OUT = 2
LR = 0.05
STEPS = 50

sess = TinyNN.tnn_session_new(0)

# Persistent param + Adam moment buffers (same shape as W).
# tnn_input_2d_f32_persistent(sess, rows, cols) → ne=[cols, rows].
t_W = TinyNN.tnn_input_2d_f32_persistent(sess, OUT, K)   # ne=[K, OUT]
t_m = TinyNN.tnn_input_2d_f32_persistent(sess, OUT, K)
t_v = TinyNN.tnn_input_2d_f32_persistent(sess, OUT, K)

# AdamW hyperparameter tensor: 7 floats — alpha, beta1, beta2, eps, wd,
# beta1h (=1-beta1^t), beta2h (=1-beta2^t).
t_hp = TinyNN.tnn_input_1d_f32_persistent(sess, 7)

# Mark W as the only trainable param.
TinyNN.tnn_set_param(t_W)

TinyNN.tnn_finalize_weights(sess)

# Compute-side x (constant input).
t_x = TinyNN.tnn_input_2d_f32(sess, 1, K)                # ne=[K, 1]

# Forward: y = W @ x, loss = sum(y²)
t_y    = TinyNN.tnn_matmul(sess, t_W, t_x)               # ne=[OUT, 1]
t_y2   = TinyNN.tnn_mul(sess, t_y, t_y)
t_loss = TinyNN.tnn_sum(sess, t_y2)

TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)
TinyNN.tnn_realize(sess, t_loss)

# Initial W upload (deterministic).
$seed = 12345
def next_normal(scale)
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u1 = ($seed.to_f + 1.0) / 2147483648.0
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u2 = ($seed.to_f + 1.0) / 2147483648.0
  scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

# Build backward graph (no allocation yet)
rc = TinyNN.tnn_build_backward(sess)
puts "build_backward rc=" + rc.to_s
if rc != 0; exit 1; end

# Grab W's gradient handle and build the adam-step node
t_grad = TinyNN.tnn_tensor_grad(sess, t_W)
if t_grad == nil
  puts "FAIL: no gradient for W"; exit 1
end
t_opt = TinyNN.tnn_opt_step_adamw(sess, t_W, t_grad, t_m, t_v, t_hp)
if t_opt == nil
  puts "FAIL: opt_step_adamw returned nil"; exit 1
end

rc = TinyNN.tnn_extend_backward_graph(sess, t_opt)
puts "extend rc=" + rc.to_s
if rc != 0; exit 1; end

# This RESETS the sched, so any upload to sched-managed tensors must
# happen AFTER. Persistent tensors (W, m, v, hp) keep their values
# because they have their own backend buffer.
rc = TinyNN.tnn_realize_backward(sess)
puts "realize_backward rc=" + rc.to_s
if rc != 0; exit 1; end

# Persistent tensors: upload AFTER realize so the buffers are stable.
m_W = Mat.new(K, OUT)
i = 0
while i < K * OUT
  m_W.flat[i] = next_normal(0.5)
  i = i + 1
end
TinyNN.stage_row_major_and_upload(sess, t_W, m_W)

# Zero gradient accumulators + Adam momenta in one shot — ggml_graph_reset
# walks the cgraph, zeros src[2]/src[3] (m, v) for any OPT_STEP_ADAMW
# node, zeros gradient accs, and sets loss-grad = 1. Call once.
rc = TinyNN.tnn_graph_reset(sess)
puts "graph_reset rc=" + rc.to_s
if rc != 0; exit 1; end

# x is fixed across iterations in this smoke; in real training,
# upload fresh per-batch inputs each step.
m_x = Mat.new(1, K)
m_x.flat[0] = 1.0
m_x.flat[1] = 2.0
m_x.flat[2] = 3.0

# Training loop
beta1 = 0.9
beta2 = 0.999
eps   = 1.0e-8
wd    = 0.0
t = 1
while t <= STEPS
  # ggml's adamw expects beta1h, beta2h as MULTIPLIERS — i.e.
  # 1/(1-beta^t). The kernel computes:  mh = m * beta1h,
  # vh = sqrt(v * beta2h) + eps. With these multipliers, at step t
  # mh recovers the bias-corrected first moment m_hat exactly.
  bch1 = 1.0 / (1.0 - (beta1 ** t))
  bch2 = 1.0 / (1.0 - (beta2 ** t))
  m_hp = Mat.new(1, 7)
  m_hp.flat[0] = LR
  m_hp.flat[1] = beta1
  m_hp.flat[2] = beta2
  m_hp.flat[3] = eps
  m_hp.flat[4] = wd
  m_hp.flat[5] = bch1
  m_hp.flat[6] = bch2
  TinyNN.stage_row_major_and_upload(sess, t_hp, m_hp)
  # x is sched-allocated; needs re-upload each step (sched alloc may
  # reuse its buffer between iterations).
  TinyNN.stage_row_major_and_upload(sess, t_x, m_x)

  rc = TinyNN.tnn_compute_backward(sess)
  if rc != 0
    puts "step " + t.to_s + ": compute_backward rc=" + rc.to_s
    exit 1
  end

  TinyNN.tnn_download(sess, t_loss)
  loss = TinyNN.tnn_scratch_get(sess, 0)
  if t == 1 || t == 2 || t % 5 == 0 || t == STEPS
    # Also peek at W to confirm it's actually being updated
    TinyNN.tnn_download(sess, t_W)
    w0 = TinyNN.tnn_scratch_get(sess, 0)
    w5 = TinyNN.tnn_scratch_get(sess, 5)
    puts "step " + t.to_s + ": loss=" + loss.to_s + "  W[0]=" + w0.to_s + " W[5]=" + w5.to_s
  end
  t = t + 1
end

TinyNN.tnn_session_free(sess)
