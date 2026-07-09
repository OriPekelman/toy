# Phase F0.4 — ggml_build_backward_expand autograd smoke.
#
# Toy graph:   y = W · x        (matmul; W is the only param)
#              L = sum(y * y)   (squared L2 norm)
# Hand grad:   dL/dW = 2 · y · xᵀ
#
# Marks W as param, marks L as loss, builds backward, computes,
# reads W's gradient via tnn_tensor_grad, compares to the analytical
# reference.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

# Tiny shape: x is [k=3, T=1]; W is [k=3, out=2]; y = W·x → ne=[out=2, T=1]
K = 3
OUT = 2

# Inputs (known values for reproducibility)
xv = [1.0, 2.0, 3.0]                 # k=3
wv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 2×3 row-major: row0=[.1,.2,.3], row1=[.4,.5,.6]

sess = TinyNN.tnn_session_new(0)

# Persistent W (param) — finalized as a real backend tensor.
# Note: tnn_input_2d_f32(sess, rows, cols) → ne=[cols, rows].
# For ggml_mul_mat(A=W, B=x) we want W ne=[K, OUT] and x ne=[K, 1].
t_W = TinyNN.tnn_input_2d_f32_persistent(sess, OUT, K)  # ne=[K, OUT]
TinyNN.tnn_set_param(t_W)
TinyNN.tnn_finalize_weights(sess)

# Compute-side x — ne=[K, 1]
t_x = TinyNN.tnn_input_2d_f32(sess, 1, K)               # ne=[K, 1]

# Forward: y = W · x.   ggml mul_mat: matmul(W, x) → ne=[OUT, 1]
t_y = TinyNN.tnn_matmul(sess, t_W, t_x)

# Loss = sum(y²) = y · y → scalar. Compute via mul + sum-1d.
t_y2  = TinyNN.tnn_mul(sess, t_y, t_y)
t_loss = TinyNN.tnn_sum(sess, t_y2)

TinyNN.tnn_set_output(t_y)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)
TinyNN.tnn_realize(sess, t_loss)

# Upload W and x
m_W = Mat.new(K, OUT)
i = 0
while i < wv.length
  m_W.flat[i] = wv[i]
  i = i + 1
end
TinyNN.stage_row_major_and_upload(sess, t_W, m_W)

m_x = Mat.new(1, K)
i = 0
while i < xv.length
  m_x.flat[i] = xv[i]
  i = i + 1
end
TinyNN.stage_row_major_and_upload(sess, t_x, m_x)

# Build backward graph (no allocation yet)
rc = TinyNN.tnn_build_backward(sess)
puts "build_backward rc=" + rc.to_s
if rc != 0; exit 1; end
# No opt_step extension here — just verify backward-only still works
rc = TinyNN.tnn_realize_backward(sess)
puts "realize_backward rc=" + rc.to_s
if rc != 0; exit 1; end
rc = TinyNN.tnn_compute_backward(sess)
puts "compute_backward rc=" + rc.to_s
if rc != 0; exit 1; end

# Read forward y and loss
TinyNN.tnn_download(sess, t_y)
yv = [TinyNN.tnn_scratch_get(sess, 0), TinyNN.tnn_scratch_get(sess, 1)]
puts "y = " + yv.inspect

TinyNN.tnn_download(sess, t_loss)
lv = TinyNN.tnn_scratch_get(sess, 0)
puts "loss = " + lv.to_s

# Read gradient tensor for W
t_W_grad = TinyNN.tnn_tensor_grad(sess, t_W)
if t_W_grad == nil
  puts "FAIL: no gradient for W"; exit 1
end
TinyNN.tnn_download(sess, t_W_grad)
gv = []
i = 0
while i < K * OUT
  gv.push(TinyNN.tnn_scratch_get(sess, i))
  i = i + 1
end
puts "dL/dW (ggml) = " + gv.inspect

# Reference: dL/dW = 2 * y * xᵀ.  W is OUT×K row-major in our Mat
# convention; ggml stores it ne=[K, OUT] column-major. The gradient
# read from ggml will match ggml's layout.
puts ""
puts "expected gradient (2 * y[i] * x[j]) for i in OUT, j in K:"
i = 0
while i < OUT
  row = []
  j = 0
  while j < K
    row.push(2.0 * yv[i] * xv[j])
    j = j + 1
  end
  puts "  row" + i.to_s + " = " + row.inspect
  i = i + 1
end

TinyNN.tnn_session_free(sess)
