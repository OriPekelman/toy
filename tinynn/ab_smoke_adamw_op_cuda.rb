# A/B parity for ggml's built-in opt_step_adamw (in-place update).
# Matches the project's adam_step_mat when weight-decay = 0:
#   m_new = m*b1 + g*(1-b1)
#   v_new = v*b2 + g*g*(1-b2)
#   mh = m_new * beta1h  (beta1h = 1/omc1)
#   vh = sqrt(v_new * beta2h) + eps  (beta2h = 1/omc2)
#   p  = p*(1 - alpha*wd) - alpha * mh / vh    (with wd=0 -> p = p - alpha*mh/vh)

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn_cuda"

def adam_native(param, grad, m, v, lr, b1, b2, eps, omc1, omc2)
  np = Mat.new(param.nrows, param.ncols)
  nm = Mat.new(param.nrows, param.ncols)
  nv = Mat.new(param.nrows, param.ncols)
  one_minus_b1 = 1.0 - b1
  one_minus_b2 = 1.0 - b2
  n = param.nrows * param.ncols
  i = 0
  while i < n
    gi = grad.flat[i]
    new_m = b1 * m.flat[i] + one_minus_b1 * gi
    new_v = b2 * v.flat[i] + one_minus_b2 * gi * gi
    nm.flat[i] = new_m
    nv.flat[i] = new_v
    m_hat = new_m / omc1
    v_hat = new_v / omc2
    np.flat[i] = param.flat[i] - lr * m_hat / (Math.sqrt(v_hat) + eps)
    i = i + 1
  end
  [np, nm, nv]
end

rows = 3
cols = 4
param = Mat.new(rows, cols)
grad  = Mat.new(rows, cols)
mst   = Mat.new(rows, cols)
vst   = Mat.new(rows, cols)
n = rows * cols
i = 0
while i < n
  param.flat[i] = i.to_f * 0.1 - 0.5
  grad.flat[i]  = (i.to_f - 6.0) * 0.05
  mst.flat[i]   = i.to_f * 0.02
  vst.flat[i]   = i.to_f * 0.01
  i = i + 1
end

lr   = 0.001
b1   = 0.9
b2   = 0.999
eps  = 1.0e-8
omc1 = 1.0 - 0.9
omc2 = 1.0 - 0.999
ref  = adam_native(param, grad, mst, vst, lr, b1, b2, eps, omc1, omc2)
nat_p = ref[0]
nat_m = ref[1]
nat_v = ref[2]

# Build a one-shot ggml graph: opt_step_adamw mutates `t_a` in place;
# we read back t_a, t_m, t_v.
sess = TinyNNCuda.tnn_session_new(1)
t_a    = TinyNNCuda.tnn_input_2d_f32(sess, rows, cols)
t_grad = TinyNNCuda.tnn_input_2d_f32(sess, rows, cols)
t_m    = TinyNNCuda.tnn_input_2d_f32(sess, rows, cols)
t_v    = TinyNNCuda.tnn_input_2d_f32(sess, rows, cols)
t_par  = TinyNNCuda.tnn_input_1d_f32(sess, 7)

TinyNNCuda.tnn_set_param(t_a)
t_out = TinyNNCuda.tnn_opt_step_adamw(sess, t_a, t_grad, t_m, t_v, t_par)
TinyNNCuda.tnn_set_output(t_a)
TinyNNCuda.tnn_set_output(t_m)
TinyNNCuda.tnn_set_output(t_v)
TinyNNCuda.tnn_realize(sess, t_out)

# Upload param, grad, m, v as row-major (Mat data matches ggml layout for
# small 2-D inputs we feed as raw bytes).
TinyNNCuda.upload_row_major(sess, t_a,    param)
TinyNNCuda.upload_row_major(sess, t_grad, grad)
TinyNNCuda.upload_row_major(sess, t_m,    mst)
TinyNNCuda.upload_row_major(sess, t_v,    vst)

# adamw_params: 7 floats [alpha, beta1, beta2, eps, wd, beta1h, beta2h]
# Match project: wd=0; beta1h = 1/omc1, beta2h = 1/omc2.
beta1h = 1.0 / omc1
beta2h = 1.0 / omc2
TinyNNCuda.tnn_scratch_set(sess, 0, lr)
TinyNNCuda.tnn_scratch_set(sess, 1, b1)
TinyNNCuda.tnn_scratch_set(sess, 2, b2)
TinyNNCuda.tnn_scratch_set(sess, 3, eps)
TinyNNCuda.tnn_scratch_set(sess, 4, 0.0)
TinyNNCuda.tnn_scratch_set(sess, 5, beta1h)
TinyNNCuda.tnn_scratch_set(sess, 6, beta2h)
TinyNNCuda.tnn_upload(sess, t_par)

TinyNNCuda.tnn_compute(sess)

ffi_p = TinyNNCuda.download_row_major(sess, t_a, rows, cols)
ffi_m = TinyNNCuda.download_row_major(sess, t_m, rows, cols)
ffi_v = TinyNNCuda.download_row_major(sess, t_v, rows, cols)

def cmp(name, a, b, tol)
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
  puts name + ": max-abs-diff=" + max_d.to_s + " " + (max_d < tol ? "OK" : "FAIL")
end

cmp("param", nat_p, ffi_p, 1.0e-4)
cmp("m",     nat_m, ffi_m, 1.0e-5)
cmp("v",     nat_v, ffi_v, 1.0e-5)
