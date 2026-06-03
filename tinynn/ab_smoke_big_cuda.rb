# CUDA variant of ab_smoke_big.rb. Same shapes and references, but
# every op goes through TinyNNCuda (cudaMemcpy host->device round-trip
# included in the FFI timing).
#
# A warm-up call runs before timing to amortize backend_init / first
# kernel-launch JIT costs. With the persistent engine refactor the
# warm-up is the only call that pays the device-init bill; subsequent
# calls reuse the cached backend.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn_cuda"

def fill_lcg(m, seed)
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

# Warm-up: one tiny matmul to amortize backend init / driver / cudaMalloc
# initial costs. Subsequent calls reuse the cached engine.
puts "warm-up..."
warm_a = Mat.new(2, 2)
warm_a.flat[0] = 1.0; warm_a.flat[1] = 2.0
warm_a.flat[2] = 3.0; warm_a.flat[3] = 4.0
_ = TinyNNCuda.matmul(warm_a, warm_a)
puts "warm-up done"

# --- matmul (64, 32) * (32, 128) ---
puts "=== matmul (64, 32) * (32, 128) ==="
h    = Mat.new(64, 32);  fill_lcg(h,    1)
w_ff = Mat.new(32, 128); fill_lcg(w_ff, 2)
t0 = Time.now
nat = h.matmul(w_ff)
t1 = Time.now
ffi = TinyNNCuda.matmul(h, w_ff)
t2 = Time.now
puts "  native: " + ((t1 - t0) * 1000.0).to_s + " ms"
puts "  cuda:   " + ((t2 - t1) * 1000.0).to_s + " ms"
puts "  max-abs-diff: " + max_abs_diff(nat, ffi).to_s

# --- matmul (64, 32) * (32, 5000) ---
puts "=== matmul (64, 32) * (32, 5000) (tied unembed) ==="
e = Mat.new(32, 5000); fill_lcg(e, 3)
t0 = Time.now
nat = h.matmul(e)
t1 = Time.now
ffi = TinyNNCuda.matmul(h, e)
t2 = Time.now
puts "  native: " + ((t1 - t0) * 1000.0).to_s + " ms"
puts "  cuda:   " + ((t2 - t1) * 1000.0).to_s + " ms"
puts "  max-abs-diff: " + max_abs_diff(nat, ffi).to_s

# --- softmax on (64, 64) ---
puts "=== softmax on (64, 64) (attention scores) ==="
scores = Mat.new(64, 64); fill_lcg(scores, 4)
t0 = Time.now
nat = softmax_native(scores)
t1 = Time.now
ffi = TinyNNCuda.softmax(scores)
t2 = Time.now
puts "  native: " + ((t1 - t0) * 1000.0).to_s + " ms"
puts "  cuda:   " + ((t2 - t1) * 1000.0).to_s + " ms"
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
ffi = TinyNNCuda.rms_norm(x_in, gamma, eps)
t2 = Time.now
puts "  native: " + ((t1 - t0) * 1000.0).to_s + " ms"
puts "  cuda:   " + ((t2 - t1) * 1000.0).to_s + " ms"
puts "  max-abs-diff: " + max_abs_diff(nat, ffi).to_s
