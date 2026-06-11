# Validates the dual-cgraph + persistent-weights design.
#
# Layout:
#   ctx_w (persistent): t_w, t_m, t_v, t_params -- stable backend buffer
#   ctx   (compute):    t_h (input), t_grad (input)
#   graph   (a): forward = scale(t_w, 1.0)              -- just reads t_w
#   graph_b (b): opt_step_adamw(t_w, t_grad, t_m, t_v, t_params)
#                                                       -- mutates t_w
#
# Sequence:
#   1. upload initial t_w, zero t_m / t_v, set t_params
#   2. switch_a + compute -> result1 should equal t_w
#   3. switch_b + upload grad + compute_b -> t_w mutated in place
#   4. switch_a + compute -> result2 should equal NEW t_w (differs from result1)

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

rows = 3
cols = 4

# === Session setup ===
sess = TinyNN.tnn_session_new(0)

# Persistent tensors (in ctx_w).
t_w       = TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols)
t_m       = TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols)
t_v       = TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols)
t_params  = TinyNN.tnn_input_1d_f32_persistent(sess, 7)
TinyNN.tnn_set_param(t_w)
TinyNN.tnn_finalize_weights(sess)

# Compute tensor (in ctx).
t_grad = TinyNN.tnn_input_2d_f32(sess, rows, cols)

# graph_a: read t_w via a scale-by-1 op (ggml needs an op to produce
# an output tensor we can download).
t_w_view = TinyNN.tnn_scale(sess, t_w, 1.0)
TinyNN.tnn_set_output(t_w_view)
TinyNN.tnn_realize(sess, t_w_view)

# graph_b: opt_step_adamw mutates t_w in place.
t_w_after = TinyNN.tnn_opt_step_adamw(sess, t_w, t_grad, t_m, t_v, t_params)
TinyNN.tnn_realize_b(sess, t_w_after)

# === Step 1: initial uploads ===
def fill(rows, cols, fn)
  m = Mat.new(rows, cols)
  n = rows * cols
  i = 0
  while i < n
    m.flat[i] = fn.call(i)
    i = i + 1
  end
  m
end

n_elem = rows * cols
w_init = Mat.new(rows, cols)
m_init = Mat.new(rows, cols)
v_init = Mat.new(rows, cols)
grad   = Mat.new(rows, cols)
i = 0
while i < n_elem
  w_init.flat[i] = i.to_f * 0.1 - 0.5
  m_init.flat[i] = 0.0
  v_init.flat[i] = 0.0
  grad.flat[i]   = (i.to_f - 6.0) * 0.05
  i = i + 1
end

TinyNN.upload_row_major(sess, t_w, w_init)
TinyNN.upload_row_major(sess, t_m, m_init)
TinyNN.upload_row_major(sess, t_v, v_init)

# adamw_params: alpha, b1, b2, eps, wd, beta1h, beta2h
lr   = 0.01
b1   = 0.9
b2   = 0.999
eps  = 1.0e-8
omc1 = 1.0 - b1
omc2 = 1.0 - b2
TinyNN.tnn_scratch_set(sess, 0, lr)
TinyNN.tnn_scratch_set(sess, 1, b1)
TinyNN.tnn_scratch_set(sess, 2, b2)
TinyNN.tnn_scratch_set(sess, 3, eps)
TinyNN.tnn_scratch_set(sess, 4, 0.0)
TinyNN.tnn_scratch_set(sess, 5, 1.0 / omc1)
TinyNN.tnn_scratch_set(sess, 6, 1.0 / omc2)
TinyNN.tnn_upload(sess, t_params)

# === Step 2: compute graph_a, observe initial t_w ===
TinyNN.tnn_compute(sess)
result1 = TinyNN.download_row_major(sess, t_w_view, rows, cols)

# === Step 3: switch to graph_b, upload grad, compute (mutates t_w) ===
TinyNN.tnn_switch_b(sess)
TinyNN.upload_row_major(sess, t_grad, grad)
TinyNN.tnn_compute_b(sess)

# Read back t_w post-adam (it's persistent so no switch needed -- the
# backend buffer is stable).
w_after = TinyNN.download_row_major(sess, t_w, rows, cols)

# === Step 4: switch back to graph_a, recompute ===
TinyNN.tnn_switch_a(sess)
TinyNN.tnn_compute(sess)
result2 = TinyNN.download_row_major(sess, t_w_view, rows, cols)

# === Verify ===
def cmp(label, a, b, tol_lo, tol_hi)
  max_d = 0.0
  n = a.nrows * a.ncols
  i = 0
  while i < n
    d = a.flat[i] - b.flat[i]
    if d < 0
      d = -d
    end
    if d > max_d
      max_d = d
    end
    i = i + 1
  end
  ok = max_d >= tol_lo && max_d <= tol_hi
  puts label + " max_abs_diff=" + max_d.to_s + " " + (ok ? "OK" : "FAIL")
  ok
end

# result1 should equal w_init (scale by 1).
ok1 = cmp("result1 vs w_init (should match)",        result1, w_init,  0.0, 1.0e-5)
# w_after should DIFFER from w_init (adam updated it).
ok2 = cmp("w_after vs w_init (should differ)",       w_after, w_init,  1.0e-5, 1.0)
# result2 should equal w_after (we re-scale w by 1).
ok3 = cmp("result2 vs w_after (should match)",       result2, w_after, 0.0, 1.0e-5)
# result2 should DIFFER from result1 (since w_after != w_init).
ok4 = cmp("result2 vs result1 (should differ)",      result2, result1, 1.0e-5, 1.0)

puts "STEP B OK: " + (ok1 && ok2 && ok3 && ok4).to_s
TinyNN.tnn_session_free(sess)
