# A/B parity for the chained-graph FFNFFICache: pre, hidden, out from
# the persistent session vs hand-rolled native FFN values.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

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

# FFN dimensions: T=3, d_model=4, d_ff=6.
t_seq   = 3
d_model = 4
d_ff    = 6

h  = Mat.new(t_seq, d_model)
w1 = Mat.new(d_model, d_ff)
w2 = Mat.new(d_ff, d_model)

n_h = t_seq * d_model
i = 0
while i < n_h
  h.flat[i] = i.to_f * 0.1 - 0.5
  i = i + 1
end

n_w1 = d_model * d_ff
i = 0
while i < n_w1
  w1.flat[i] = i.to_f * 0.05 - 0.3
  i = i + 1
end

n_w2 = d_ff * d_model
i = 0
while i < n_w2
  w2.flat[i] = i.to_f * 0.07 - 0.4
  i = i + 1
end

pre_n    = h.matmul(w1)
hidden_n = gelu_native(pre_n)
out_n    = hidden_n.matmul(w2)

cache = FFNFFICache.new
cache.realize_for(t_seq, d_model, d_ff)

TinyNN.upload_row_major(cache.sess, cache.t_h, h)
TinyNN.stage_transposed_and_upload(cache.sess, cache.t_w1_t, w1)
TinyNN.stage_transposed_and_upload(cache.sess, cache.t_w2_t, w2)
TinyNN.tnn_compute(cache.sess)
pre_f    = TinyNN.download_row_major(cache.sess, cache.t_pre,    t_seq, d_ff)
hidden_f = TinyNN.download_row_major(cache.sess, cache.t_hidden, t_seq, d_ff)
out_f    = TinyNN.download_row_major(cache.sess, cache.t_out,    t_seq, d_model)

def cmp(label, a, b, tol)
  n = a.nrows * a.ncols
  max = 0.0
  i = 0
  while i < n
    d = a.flat[i] - b.flat[i]
    if d < 0
      d = -d
    end
    if d > max
      max = d
    end
    i = i + 1
  end
  puts label + " max_abs_diff=" + max.to_s + (max < tol ? " OK" : " FAIL")
  max < tol
end

ok1 = cmp("pre    ", pre_n,    pre_f,    1.0e-4)
# ggml_gelu uses an f16 lookup table on CPU: parity to ~1e-3 absolute.
ok2 = cmp("hidden ", hidden_n, hidden_f, 1.0e-3)
ok3 = cmp("out    ", out_n,    out_f,    1.0e-3)

puts "all match: " + (ok1 && ok2 && ok3).to_s
puts "pre_n[0..5]: " + pre_n.flat[0].to_s + " " + pre_n.flat[1].to_s + " " + pre_n.flat[2].to_s + " " + pre_n.flat[3].to_s + " " + pre_n.flat[4].to_s + " " + pre_n.flat[5].to_s
puts "pre_f[0..5]: " + pre_f.flat[0].to_s + " " + pre_f.flat[1].to_s + " " + pre_f.flat[2].to_s + " " + pre_f.flat[3].to_s + " " + pre_f.flat[4].to_s + " " + pre_f.flat[5].to_s
puts "pre dims: nrows=" + pre_n.nrows.to_s + " ncols=" + pre_n.ncols.to_s
