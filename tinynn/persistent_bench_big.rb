# Persistent-session bench at real-LLM shapes: T=64, d_model=512, vocab=32K.
# Tells us whether the persistent API earns its complexity at scale, or
# just at the toy shapes. Native Mat#matmul is included for reference.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

T     = 64
DM    = 512
VOCAB = 32000
ITERS = 5     # vocab=32K means each matmul = ~8 MB output, keep loop small

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

h  = Mat.new(T, DM)
w  = Mat.new(DM, VOCAB)   # tied-unembed-shaped
fill_lcg(h, 1)
fill_lcg(w, 2)

puts "=== shapes: h=(" + T.to_s + "," + DM.to_s + ")  w=(" + DM.to_s + "," + VOCAB.to_s + ")  iters=" + ITERS.to_s + " ==="

# Native baseline.
t0 = Time.now
i = 0
while i < ITERS
  _ = h.matmul(w)
  i = i + 1
end
t1 = Time.now
puts "native " + ITERS.to_s + " x Mat#matmul: " + ((t1 - t0) * 1000.0).to_s + " ms"

# Persistent FFI session.
sess = TinyNN.persistent_new(0)
ta = TinyNN.alloc_2d(sess, T, DM)
tb = TinyNN.alloc_2d(sess, w.ncols, w.nrows)
tc = TinyNN.build_matmul(sess, ta, tb)
TinyNN.realize(sess, tc)
TinyNN.upload_transposed(sess, tb, w)

t0 = Time.now
i = 0
while i < ITERS
  TinyNN.upload_row_major(sess, ta, h)
  TinyNN.compute(sess)
  _ = TinyNN.download_matmul(sess, tc, T, VOCAB)
  i = i + 1
end
t1 = Time.now
puts "persistent " + ITERS.to_s + " x compute(+ up/down): " + ((t1 - t0) * 1000.0).to_s + " ms"

TinyNN.persistent_free(sess)
