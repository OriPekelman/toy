# Persistent-session bench: build a 2-matmul graph once, run it N
# times against changing inputs.  Compares against the one-shot pattern
# (TinyNN.matmul called twice per iter), which builds + tears down a
# new ggml graph + scheduler-alloc each time.
#
# At the toy LM's FFN shapes (T=8 d_model=16 d_ff=32) this should
# show the persistent pattern winning by 1-2 orders of magnitude on
# the per-iteration cost — most of which is currently graph alloc.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

T   = 8
DM  = 16
DFF = 32
ITERS = 50

# --- input + weight matrices (deterministic LCG fill) ---
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

# ----- one-shot baseline (current API) -----
t0 = Time.now
i = 0
while i < ITERS
  _ = TinyNN.matmul(h, w_ff)
  i = i + 1
end
t1 = Time.now
puts "one-shot " + ITERS.to_s + " x TinyNN.matmul: " + ((t1 - t0) * 1000.0).to_s + " ms"

# ----- persistent session -----
sess = TinyNN.persistent_new(0)
ta = TinyNN.alloc_2d(sess, T, DM)
# B uploaded transposed (ggml mul_mat is A*B^T natively).
tb = TinyNN.alloc_2d(sess, w_ff.ncols, w_ff.nrows)
tc = TinyNN.build_matmul(sess, ta, tb)
TinyNN.realize(sess, tc)

# Upload weight ONCE.
TinyNN.upload_transposed(sess, tb, w_ff)

t0 = Time.now
i = 0
while i < ITERS
  TinyNN.upload_row_major(sess, ta, h)
  TinyNN.compute(sess)
  _ = TinyNN.download_matmul(sess, tc, T, DFF)
  i = i + 1
end
t1 = Time.now
puts "persistent " + ITERS.to_s + " x compute: " + ((t1 - t0) * 1000.0).to_s + " ms"

# Verify last result matches native h.matmul(w_ff).
last_ffi = TinyNN.download_matmul(sess, tc, T, DFF)
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
  if d > 1.0e-4
    ok = false
  end
  i = i + 1
end
puts "parity vs native: max-abs-diff=" + max_d.to_s + " match=" + ok.to_s

TinyNN.persistent_free(sess)
