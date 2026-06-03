# Bigger-shape parity tests at the transformer's actual dimensions.
# Run via `make ab-smoke-big`.
#
# Shapes mirrored from train_tinystories' model:
#   T = 64 (context_length)
#   d_model = 32, d_ff = 128, vocab = 5000
#
# What each test exercises:
#   matmul (64, 32) * (32, 128)    FFN first projection
#   matmul (64, 32) * (32, 5000)   tied unembed (largest matmul)
#   softmax on (64, 64)            attention scores
#   rms_norm on (64, 32)           pre-norm
#
# Tolerance is loose (1e-2 on the big matmuls) because per-element f32
# rounding accumulates over the inner-product k-dim.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

def fill_lcg(m, seed)
  # Cheap deterministic LCG to avoid all-equal patterns that hide
  # transposition bugs. Maps to [-1, 1).
  s = seed
  n = m.nrows * m.ncols
  i = 0
  while i < n
    s = (s * 1103515245 + 12345) & 0x7fffffff
    m.flat[i] = (s % 1000).to_f / 500.0 - 1.0
    i = i + 1
  end
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

def max_abs_diff(a, b)
  d = 0.0
  n = a.nrows * a.ncols
  i = 0
  while i < n
    e = a.flat[i] - b.flat[i]
    if e < 0
      e = -e
    end
    if e > d
      d = e
    end
    i = i + 1
  end
  d
end

# --- matmul (64, 32) * (32, 128) ---
puts "=== matmul (64, 32) * (32, 128) ==="
h    = Mat.new(64, 32);  fill_lcg(h,    1)
w_ff = Mat.new(32, 128); fill_lcg(w_ff, 2)
t0 = Time.now
nat = h.matmul(w_ff)
t1 = Time.now
ffi = TinyNN.matmul(h, w_ff)
t2 = Time.now
puts "  native: " + ((t1 - t0) * 1000.0).to_s + " ms"
puts "  ffi:    " + ((t2 - t1) * 1000.0).to_s + " ms"
puts "  max-abs-diff: " + max_abs_diff(nat, ffi).to_s

# --- matmul (64, 32) * (32, 5000)  — tied unembed shape ---
puts "=== matmul (64, 32) * (32, 5000) (tied unembed) ==="
e = Mat.new(32, 5000); fill_lcg(e, 3)
t0 = Time.now
nat = h.matmul(e)
t1 = Time.now
ffi = TinyNN.matmul(h, e)
t2 = Time.now
puts "  native: " + ((t1 - t0) * 1000.0).to_s + " ms"
puts "  ffi:    " + ((t2 - t1) * 1000.0).to_s + " ms"
puts "  max-abs-diff: " + max_abs_diff(nat, ffi).to_s

# --- softmax on (64, 64) ---
puts "=== softmax on (64, 64) (attention scores) ==="
scores = Mat.new(64, 64); fill_lcg(scores, 4)
t0 = Time.now
nat = softmax_native(scores)
t1 = Time.now
ffi = TinyNN.softmax(scores)
t2 = Time.now
puts "  native: " + ((t1 - t0) * 1000.0).to_s + " ms"
puts "  ffi:    " + ((t2 - t1) * 1000.0).to_s + " ms"
puts "  max-abs-diff: " + max_abs_diff(nat, ffi).to_s

# --- rms_norm on (64, 32) ---
puts "=== rms_norm on (64, 32) (pre-norm) ==="
x_in = Mat.new(64, 32); fill_lcg(x_in, 5)
gamma = Array.new(32, 1.0)
i = 0
while i < 32
  gamma[i] = (i.to_f * 0.1) + 0.5
  i = i + 1
end
eps = 0.00001
t0 = Time.now
nat = rms_norm_native(x_in, gamma, eps)
t1 = Time.now
ffi = TinyNN.rms_norm(x_in, gamma, eps)
t2 = Time.now
puts "  native: " + ((t1 - t0) * 1000.0).to_s + " ms"
puts "  ffi:    " + ((t2 - t1) * 1000.0).to_s + " ms"
puts "  max-abs-diff: " + max_abs_diff(nat, ffi).to_s
