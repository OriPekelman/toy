# CUDA version of persistent_bench.rb. Same shapes; runs on the local
# GPU. Expected outcome: one-shot loses badly (per-call cudaMalloc),
# persistent wins because backend buffers are allocated once.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn_cuda"

T   = 8
DM  = 16
DFF = 32
ITERS = 50

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

h    = Mat.new(T, DM)
w_ff = Mat.new(DM, DFF)
fill_lcg(h,    1)
fill_lcg(w_ff, 2)

# Warm-up: amortise CUDA first-launch cost away from the measurement.
_ = TinyNNCuda.matmul(h, w_ff)

# ----- one-shot -----
t0 = Time.now
i = 0
while i < ITERS
  _ = TinyNNCuda.matmul(h, w_ff)
  i = i + 1
end
t1 = Time.now
puts "one-shot " + ITERS.to_s + " x TinyNNCuda.matmul: " + ((t1 - t0) * 1000.0).to_s + " ms"

# ----- persistent -----
sess = TinyNNCuda.persistent_new(1)
ta = TinyNNCuda.alloc_2d(sess, T, DM)
tb = TinyNNCuda.alloc_2d(sess, w_ff.ncols, w_ff.nrows)
tc = TinyNNCuda.build_matmul(sess, ta, tb)
TinyNNCuda.realize(sess, tc)
TinyNNCuda.upload_transposed(sess, tb, w_ff)

t0 = Time.now
i = 0
while i < ITERS
  TinyNNCuda.upload_row_major(sess, ta, h)
  TinyNNCuda.compute(sess)
  _ = TinyNNCuda.download_matmul(sess, tc, T, DFF)
  i = i + 1
end
t1 = Time.now
puts "persistent " + ITERS.to_s + " x compute: " + ((t1 - t0) * 1000.0).to_s + " ms"

last_ffi = TinyNNCuda.download_matmul(sess, tc, T, DFF)
nat = h.matmul(w_ff)
ok = true
max_d = 0.0
n_check = T * DFF
i = 0
while i < n_check
  d = last_ffi.flat[i] - nat.flat[i]
  if d < 0
    d = -d
  end
  if d > max_d
    max_d = d
  end
  if d > 1.0e-2
    ok = false
  end
  i = i + 1
end
puts "parity vs native: max-abs-diff=" + max_d.to_s + " match=" + ok.to_s

TinyNNCuda.persistent_free(sess)
