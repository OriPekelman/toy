# Consolidated CUDA parity smoke: matmul, add, gelu, rms_norm, softmax,
# scale, ffn_pipeline. Compares each TinyNNCuda op to the native Mat
# implementation (or an inline reference for gelu/rms_norm/softmax).
# Run via `make ab-smoke-all-cuda` once `make setup-ggml-cuda` has
# produced vendor/ggml/build-cuda.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn_cuda"

# --- reference helpers ---

def gelu_native(x)
  c = 0.7978845608028654
  out = Mat.new(x.nrows, x.ncols)
  n = x.nrows * x.ncols
  i = 0
  while i < n
    v = x.flat[i]
    u = c * (v + 0.044715 * v * v * v)
    out.flat[i] = 0.5 * v * (1.0 + Math.tanh(u))
    i = i + 1
  end
  out
end

def softmax_native(x)
  out = Mat.new(x.nrows, x.ncols)
  r = 0
  while r < x.nrows
    m = x.flat[r * x.ncols]
    c = 1
    while c < x.ncols
      v = x.flat[r * x.ncols + c]
      if v > m
        m = v
      end
      c = c + 1
    end
    s = 0.0
    c = 0
    while c < x.ncols
      e = Math.exp(x.flat[r * x.ncols + c] - m)
      out.flat[r * x.ncols + c] = e
      s = s + e
      c = c + 1
    end
    c = 0
    while c < x.ncols
      out.flat[r * x.ncols + c] = out.flat[r * x.ncols + c] / s
      c = c + 1
    end
    r = r + 1
  end
  out
end

def rms_norm_native(x, gamma, eps)
  out = Mat.new(x.nrows, x.ncols)
  r = 0
  while r < x.nrows
    sum_sq = 0.0
    c = 0
    while c < x.ncols
      v = x.flat[r * x.ncols + c]
      sum_sq = sum_sq + v * v
      c = c + 1
    end
    mean_sq = sum_sq / x.ncols.to_f
    rms = Math.sqrt(mean_sq + eps)
    c = 0
    while c < x.ncols
      out.flat[r * x.ncols + c] = x.flat[r * x.ncols + c] / rms * gamma[c]
      c = c + 1
    end
    r = r + 1
  end
  out
end

def scale_native(a, s)
  out = Mat.new(a.nrows, a.ncols)
  n = a.nrows * a.ncols
  i = 0
  while i < n
    out.flat[i] = a.flat[i] * s
    i = i + 1
  end
  out
end

def cmp(name, native, ffi, tol)
  n = native.nrows * native.ncols
  ok = true
  i = 0
  while i < n
    d = native.flat[i] - ffi.flat[i]
    if d < 0
      d = -d
    end
    if d > tol
      ok = false
    end
    i = i + 1
  end
  puts name + ": match=" + ok.to_s + "  (n=" + n.to_s + ", tol=" + tol.to_s + ")"
end

# --- matmul: 2x3 @ 3x2 ---
a = Mat.new(2, 3)
a.flat[0] = 1.0; a.flat[1] = 2.0; a.flat[2] = 3.0
a.flat[3] = 4.0; a.flat[4] = 5.0; a.flat[5] = 6.0
b = Mat.new(3, 2)
b.flat[0] = 7.0;  b.flat[1] = 8.0
b.flat[2] = 9.0;  b.flat[3] = 10.0
b.flat[4] = 11.0; b.flat[5] = 12.0
cmp("matmul",  a.matmul(b), TinyNNCuda.matmul(a, b), 1.0e-4)

# --- add: 2x3 ---
b_add = Mat.new(2, 3)
b_add.flat[0] = -1.0; b_add.flat[1] = 0.5; b_add.flat[2] = 2.0
b_add.flat[3] =  3.0; b_add.flat[4] = -4.0; b_add.flat[5] = 5.0
cmp("add",     a.plus(b_add),    TinyNNCuda.add(a, b_add), 1.0e-4)

# --- gelu: 1x5 ---
g = Mat.new(1, 5)
g.flat[0] = 0.0; g.flat[1] = 0.5; g.flat[2] = 3.0; g.flat[3] = -0.5; g.flat[4] = -3.0
cmp("gelu",    gelu_native(g),  TinyNNCuda.gelu(g),       1.0e-3)

# --- rms_norm: 2x4 ---
x = Mat.new(2, 4)
x.flat[0] = 1.0;  x.flat[1] = 2.0;  x.flat[2] = -1.0; x.flat[3] = 0.5
x.flat[4] = -0.5; x.flat[5] = 3.0;  x.flat[6] = 0.0;  x.flat[7] = 1.5
gamma = [1.0, 0.5, 2.0, 1.5]
eps = 0.00001
cmp("rms_norm",rms_norm_native(x, gamma, eps), TinyNNCuda.rms_norm(x, gamma, eps), 1.0e-3)

# --- softmax: 2x4 ---
sm = Mat.new(2, 4)
sm.flat[0] = 1.0; sm.flat[1] = 2.0; sm.flat[2] = 3.0; sm.flat[3] = 4.0
sm.flat[4] = 0.5; sm.flat[5] = 0.5; sm.flat[6] = 0.5; sm.flat[7] = 0.5
cmp("softmax", softmax_native(sm), TinyNNCuda.softmax(sm), 1.0e-4)

# --- scale: 2x3 ---
cmp("scale",   scale_native(a, 0.5), TinyNNCuda.scale(a, 0.5), 1.0e-5)

# --- matmul_t: (3,4) and (5,4) -> (3,5) ---
mata = Mat.new(3, 4)
matb = Mat.new(5, 4)
i = 0
while i < 12
  mata.flat[i] = i.to_f * 0.1 - 0.5
  i = i + 1
end
i = 0
while i < 20
  matb.flat[i] = i.to_f * 0.07 - 0.4
  i = i + 1
end
cmp("matmul_t", mata.matmul_t(matb), TinyNNCuda.matmul_t(mata, matb), 1.0e-3)

# --- t_matmul: (4,3) and (4,5) -> (3,5) ---
mata2 = Mat.new(4, 3)
matbt = Mat.new(4, 5)
i = 0
while i < 12
  mata2.flat[i] = i.to_f * 0.13 - 0.4
  i = i + 1
end
i = 0
while i < 20
  matbt.flat[i] = i.to_f * 0.09 - 0.3
  i = i + 1
end
cmp("t_matmul", mata2.t_matmul(matbt), TinyNNCuda.t_matmul(mata2, matbt), 1.0e-3)

# --- softmax_back: (3, 4) ---
xsm = Mat.new(3, 4)
i = 0
while i < 12
  xsm.flat[i] = (i.to_f * 0.4 - 2.0)
  i = i + 1
end
a_out = TinyNNCuda.softmax(xsm)
dysm = Mat.new(3, 4)
i = 0
while i < 12
  dysm.flat[i] = (i.to_f * 0.2 - 1.0)
  i = i + 1
end
# Native softmax-back reference.
def softmax_back_native(soft, dy)
  out = Mat.new(soft.nrows, soft.ncols)
  r = 0
  while r < soft.nrows
    nc = soft.ncols
    dot = 0.0
    c = 0
    while c < nc
      dot = dot + soft.flat[r * nc + c] * dy.flat[r * nc + c]
      c = c + 1
    end
    c = 0
    while c < nc
      ai = soft.flat[r * nc + c]
      di = dy.flat[r * nc + c]
      out.flat[r * nc + c] = ai * (di - dot)
      c = c + 1
    end
    r = r + 1
  end
  out
end
cmp("softmax_back", softmax_back_native(a_out, dysm), TinyNNCuda.softmax_back(a_out, dysm), 1.0e-3)

# --- embed_lookup + embed_back ---
tab = Mat.new(5, 3)
i = 0
while i < 5
  c = 0
  while c < 3
    tab.flat[i * 3 + c] = (i * 10 + c).to_f
    c = c + 1
  end
  i = i + 1
end
idx_arr = [2, 0, 4, 2]
def embed_lookup_native(t, idx)
  d = t.ncols
  n = idx.length
  out = Mat.new(n, d)
  i = 0
  while i < n
    tok = idx[i]
    c = 0
    while c < d
      out.flat[i * d + c] = t.flat[tok * d + c]
      c = c + 1
    end
    i = i + 1
  end
  out
end
def embed_back_native(d_out, idx, vocab)
  d = d_out.ncols
  n = idx.length
  out = Mat.new(vocab, d)
  i = 0
  while i < n
    tok = idx[i]
    c = 0
    while c < d
      out.flat[tok * d + c] = out.flat[tok * d + c] + d_out.flat[i * d + c]
      c = c + 1
    end
    i = i + 1
  end
  out
end
cmp("embed_lookup", embed_lookup_native(tab, idx_arr), TinyNNCuda.embed_lookup(tab, idx_arr), 1.0e-4)
d_emb = Mat.new(4, 3)
i = 0
while i < 12
  d_emb.flat[i] = (i + 1).to_f * 0.5
  i = i + 1
end
cmp("embed_back", embed_back_native(d_emb, idx_arr, 5), TinyNNCuda.embed_back(d_emb, idx_arr, 5), 1.0e-4)

# --- sgd_step: (4, 3) ---
sp = Mat.new(4, 3)
sg = Mat.new(4, 3)
i = 0
while i < 12
  sp.flat[i] = i.to_f * 0.5 - 2.0
  sg.flat[i] = (i.to_f - 6.0) * 0.1
  i = i + 1
end
def sgd_native(param, grad, lr)
  out = Mat.new(param.nrows, param.ncols)
  n = param.nrows * param.ncols
  i = 0
  while i < n
    out.flat[i] = param.flat[i] - lr * grad.flat[i]
    i = i + 1
  end
  out
end
cmp("sgd_step", sgd_native(sp, sg, 0.05), TinyNNCuda.sgd_step(sp, sg, 0.05), 1.0e-4)

# --- ffn_pipeline: (3,4)*(4,6)*(6,4) ---
h_p  = Mat.new(3, 4)
w1_p = Mat.new(4, 6)
w2_p = Mat.new(6, 4)
n_h = 12
i = 0
while i < n_h
  h_p.flat[i] = i.to_f * 0.1 - 0.5
  i = i + 1
end
n_w1 = 24
i = 0
while i < n_w1
  w1_p.flat[i] = i.to_f * 0.05 - 0.3
  i = i + 1
end
n_w2 = 24
i = 0
while i < n_w2
  w2_p.flat[i] = i.to_f * 0.07 - 0.4
  i = i + 1
end
pre = h_p.matmul(w1_p)
hid = gelu_native(pre)
nat = hid.matmul(w2_p)
cmp("ffn_pipeline", nat, TinyNNCuda.ffn_pipeline(h_p, w1_p, w2_p), 1.0e-3)

# --- gelu_back ---
def gelu_back_native(x, dh)
  c = 0.7978845608028654
  k = 0.044715
  out = Mat.new(x.nrows, x.ncols)
  n = x.nrows * x.ncols
  i = 0
  while i < n
    xi = x.flat[i]
    di = dh.flat[i]
    xi2 = xi * xi
    u = c * (xi + k * xi * xi2)
    tu = Math.tanh(u)
    sech2 = 1.0 - tu * tu
    dudx = c * (1.0 + 3.0 * k * xi2)
    dgelu = 0.5 * (1.0 + tu) + 0.5 * xi * sech2 * dudx
    out.flat[i] = di * dgelu
    i = i + 1
  end
  out
end
gb_x = Mat.new(1, 5)
gb_x.flat[0] =  0.0; gb_x.flat[1] = 0.5; gb_x.flat[2] = 3.0; gb_x.flat[3] = -0.5; gb_x.flat[4] = -3.0
gb_d = Mat.new(1, 5)
gb_d.flat[0] = 1.0; gb_d.flat[1] = -0.5; gb_d.flat[2] = 2.0; gb_d.flat[3] = -1.0; gb_d.flat[4] = 0.5
cmp("gelu_back", gelu_back_native(gb_x, gb_d), TinyNNCuda.gelu_back(gb_x, gb_d), 1.0e-3)

# --- cross_entropy_grad ---
def ce_grad_native(logits, targets, n_pred)
  sm = Mat.new(logits.nrows, logits.ncols)
  r = 0
  while r < logits.nrows
    m = logits.flat[r * logits.ncols]
    c = 1
    while c < logits.ncols
      v = logits.flat[r * logits.ncols + c]
      if v > m
        m = v
      end
      c = c + 1
    end
    s = 0.0
    c = 0
    while c < logits.ncols
      e = Math.exp(logits.flat[r * logits.ncols + c] - m)
      sm.flat[r * logits.ncols + c] = e
      s = s + e
      c = c + 1
    end
    c = 0
    while c < logits.ncols
      sm.flat[r * logits.ncols + c] = sm.flat[r * logits.ncols + c] / s
      c = c + 1
    end
    r = r + 1
  end
  out = Mat.new(logits.nrows, logits.ncols)
  inv = 1.0 / n_pred.to_f
  i = 0
  while i < n_pred
    v = 0
    while v < logits.ncols
      term = sm.flat[i * logits.ncols + v]
      if v == targets[i]
        term = term - 1.0
      end
      out.flat[i * logits.ncols + v] = term * inv
      v = v + 1
    end
    i = i + 1
  end
  out
end
ce_logits = Mat.new(3, 5)
i = 0
while i < 15
  ce_logits.flat[i] = i.to_f * 0.3 - 2.0
  i = i + 1
end
ce_targets = [4, 0, 2]
cmp("ce_grad", ce_grad_native(ce_logits, ce_targets, 3), TinyNNCuda.cross_entropy_grad(ce_logits, ce_targets, 3), 1.0e-3)

# --- adam_step ---
ad_p = Mat.new(3, 4)
ad_g = Mat.new(3, 4)
ad_m = Mat.new(3, 4)
ad_v = Mat.new(3, 4)
i = 0
while i < 12
  ad_p.flat[i] = i.to_f * 0.1 - 0.5
  ad_g.flat[i] = (i.to_f - 6.0) * 0.05
  ad_m.flat[i] = i.to_f * 0.02
  ad_v.flat[i] = i.to_f * 0.01
  i = i + 1
end
def adam_native(p, g, mm, vv, lr, b1, b2, eps, omc1, omc2)
  np = Mat.new(p.nrows, p.ncols)
  one_minus_b1 = 1.0 - b1
  one_minus_b2 = 1.0 - b2
  n = p.nrows * p.ncols
  i = 0
  while i < n
    gi = g.flat[i]
    new_m = b1 * mm.flat[i] + one_minus_b1 * gi
    new_v = b2 * vv.flat[i] + one_minus_b2 * gi * gi
    m_hat = new_m / omc1
    v_hat = new_v / omc2
    np.flat[i] = p.flat[i] - lr * m_hat / (Math.sqrt(v_hat) + eps)
    i = i + 1
  end
  np
end
nat_ap = adam_native(ad_p, ad_g, ad_m, ad_v, 0.001, 0.9, 0.999, 1.0e-8, 0.1, 0.001)
ffi_ap = TinyNNCuda.adam_step(ad_p, ad_g, ad_m, ad_v, 0.001, 0.9, 0.999, 1.0e-8, 0.1, 0.001).param
cmp("adam_step param", nat_ap, ffi_ap, 1.0e-4)
