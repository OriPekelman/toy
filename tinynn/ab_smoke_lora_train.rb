# F1.2 step 1 — multi-step LoRA convergence via the F1.1 in-graph
# optimizer. Same task as F1.0 (demos/lora_smoke.rb: regress y to zero
# under a LoRA-shaped reparameterisation) but executed through the
# autograd + opt_step_sgd primitives instead of hand-coded backward.
#
# Setup:
#   x       : K-vector (frozen input)
#   W_base  : OUT × K (frozen — no set_param)
#   A       : R × K    (trainable LoRA factor)
#   B       : OUT × R  (trainable LoRA factor, zero-init standard LoRA)
#   y       = (W_base + B @ A) @ x
#   loss    = sum(y * y)
#
# At step 0 with B=0, y = W_base @ x and loss = ||W_base @ x||². As
# (A, B) train, B@A learns to cancel W_base @ x, driving loss down.
#
# Validates:
#   - Multiple param tensors in one graph_b (A and B both get
#     grad_accs + opt_step nodes)
#   - tnn_build_forward_only → build_backward → extend × 2 →
#     realize_backward → graph_reset → compute_backward repeated
#   - SGD multi-step convergence (graph_reset every iter zeros grads
#     and re-sets loss_grad; SGD has no momenta to clobber)
#   - tnn_set_output(A) + tnn_set_output(B) so we can read updated
#     params after each compute (avoids sched aliasing — same footgun
#     as F1.1's micro5)
#
# Run: ruby tinynn/ab_smoke_lora_train.rb

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

K     = 8       # input dim
OUT   = 8       # output dim
R     = 4       # LoRA rank
STEPS = 60
LR    = 0.002

sess = TinyNN.tnn_session_new(0)

$seed = 42
def next_normal(scale)
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u1 = ($seed.to_f + 1.0) / 2147483648.0
  $seed = ($seed * 1103515245 + 12345) & 0x7FFFFFFF
  u2 = ($seed.to_f + 1.0) / 2147483648.0
  scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

# Persistent params + frozen base. tnn_input_2d_f32_persistent(sess,
# rows, cols) gives ne=[cols, rows], so for W_base ne=[K, OUT] we
# create (OUT, K) — matches ggml_mul_mat(W, x) where W ne=[K, OUT]
# multiplies an x of ne=[K, 1] to produce ne=[OUT, 1].
t_W = TinyNN.tnn_input_2d_f32_persistent(sess, OUT, K)
t_A = TinyNN.tnn_input_2d_f32_persistent(sess, R,   K)
t_B = TinyNN.tnn_input_2d_f32_persistent(sess, OUT, R)
t_hp = TinyNN.tnn_input_1d_f32_persistent(sess, 2)   # [lr, wd]

# Mark A and B trainable; W stays frozen.
TinyNN.tnn_set_param(t_A)
TinyNN.tnn_set_param(t_B)
TinyNN.tnn_finalize_weights(sess)

# Compute-side input.
t_x = TinyNN.tnn_input_2d_f32(sess, 1, K)

# Forward: y = W·x + B·(A·x); loss = sum(y * y).
t_ax    = TinyNN.tnn_matmul(sess, t_A, t_x)      # ne=[1, R]
t_bax   = TinyNN.tnn_matmul(sess, t_B, t_ax)     # ne=[1, OUT]
t_ybase = TinyNN.tnn_matmul(sess, t_W, t_x)      # ne=[1, OUT]
t_y     = TinyNN.tnn_add(sess, t_ybase, t_bax)
t_y_sq  = TinyNN.tnn_mul(sess, t_y, t_y)
t_loss  = TinyNN.tnn_sum(sess, t_y_sq)             # scalar

# Protect tensors we want to read after compute from sched aliasing
# (F1.1 footgun: any intermediate forward node may share a slot with
# a backward intermediate unless marked output).
TinyNN.tnn_set_output(t_A)
TinyNN.tnn_set_output(t_B)
TinyNN.tnn_set_output(t_y)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)

rc = TinyNN.tnn_build_forward_only(sess, t_loss)
if rc != 0; puts "build_forward_only rc=" + rc.to_s; exit 1; end

rc = TinyNN.tnn_build_backward(sess)
if rc != 0; puts "build_backward rc=" + rc.to_s; exit 1; end

# Attach opt_step_sgd to each param. The grad tensors are created by
# build_backward; we look them up via tnn_tensor_grad.
t_grad_A = TinyNN.tnn_tensor_grad(sess, t_A)
t_grad_B = TinyNN.tnn_tensor_grad(sess, t_B)
t_opt_A  = TinyNN.tnn_opt_step_sgd(sess, t_A, t_grad_A, t_hp)
t_opt_B  = TinyNN.tnn_opt_step_sgd(sess, t_B, t_grad_B, t_hp)
TinyNN.tnn_extend_backward_graph(sess, t_opt_A)
TinyNN.tnn_extend_backward_graph(sess, t_opt_B)

rc = TinyNN.tnn_realize_backward(sess)
if rc != 0; puts "realize_backward rc=" + rc.to_s; exit 1; end

# Initialise weights. W_base = small Gaussian. A = small Gaussian.
# B = zero (standard LoRA init: at step 0, B@A = 0 so the adapter
# is identity, allowing fine-tuning to start from the base model).
m_W = Mat.new(OUT, K)
i = 0
while i < OUT * K
  m_W.flat[i] = next_normal(0.5)
  i = i + 1
end
TinyNN.stage_row_major_and_upload(sess, t_W, m_W)

m_A = Mat.new(R, K)
i = 0
while i < R * K
  m_A.flat[i] = next_normal(0.1)
  i = i + 1
end
TinyNN.stage_row_major_and_upload(sess, t_A, m_A)

m_B = Mat.new(OUT, R)
i = 0
while i < OUT * R
  m_B.flat[i] = 0.0
  i = i + 1
end
TinyNN.stage_row_major_and_upload(sess, t_B, m_B)

m_hp = Mat.new(1, 2)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.0
TinyNN.stage_row_major_and_upload(sess, t_hp, m_hp)

m_x = Mat.new(K, 1)
i = 0
while i < K
  m_x.flat[i] = 0.5 + i.to_f * 0.3
  i = i + 1
end
# t_x is non-persistent (compute-side); upload AFTER realize_backward,
# AFTER any sched_reset, just like the F1.1 micros do.
TinyNN.stage_row_major_and_upload(sess, t_x, m_x)

losses = []
s = 0
while s < STEPS
  TinyNN.tnn_graph_reset(sess)
  # x is in sched-allocated buffer; sched_reset between calls could
  # invalidate it. tnn_graph_reset doesn't sched_reset (just zeros
  # grads / sets loss_grad / clears momenta) so x survives. But for
  # safety re-upload x every step — cheap, and immune to future
  # changes in graph_reset's scope.
  TinyNN.stage_row_major_and_upload(sess, t_x, m_x)

  rc = TinyNN.tnn_compute_backward(sess)
  if rc != 0; puts "compute_backward rc=" + rc.to_s; exit 1; end

  TinyNN.tnn_download(sess, t_loss)
  loss_v = TinyNN.tnn_scratch_get(sess, 0)
  losses.push(loss_v)

  if s == 0 || (s + 1) % 5 == 0
    puts "step " + (s + 1).to_s + ": loss=" + loss_v.to_s
  end
  s = s + 1
end

TinyNN.tnn_session_free(sess)

# Acceptance: monotonic-ish decrease. Allow a small wobble — SGD on a
# non-convex composite reparameterisation can wiggle. Strict gate:
# final < 0.1 × initial.
puts ""
puts "initial loss = " + losses[0].to_s
puts "final   loss = " + losses[STEPS - 1].to_s
puts "ratio        = " + (losses[STEPS - 1] / losses[0]).to_s

if losses[STEPS - 1] < 0.1 * losses[0]
  puts "VERDICT: PASS (loss < 10% of initial)"
else
  puts "VERDICT: FAIL (loss did not converge)"; exit 1
end
