# A/B parity for embedding lookup + scatter-add.
#
# embed_lookup(table, indices)        -> table[indices[i]] for each i
# embed_back(d_out, indices, vocab)   -> result[indices[i]] += d_out[i]
#
# Tests both ops at toy shapes so we exercise the FFI int32 scratch path.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

def embed_lookup_native(table, indices)
  d = table.ncols
  n = indices.length
  out = Mat.new(n, d)
  i = 0
  while i < n
    tok = indices[i]
    c = 0
    while c < d
      out.flat[i * d + c] = table.flat[tok * d + c]
      c = c + 1
    end
    i = i + 1
  end
  out
end

def embed_back_native(d_out, indices, vocab)
  d = d_out.ncols
  n = indices.length
  out = Mat.new(vocab, d)
  i = 0
  while i < n
    tok = indices[i]
    c = 0
    while c < d
      out.flat[tok * d + c] = out.flat[tok * d + c] + d_out.flat[i * d + c]
      c = c + 1
    end
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
  puts name + ": shape " + native.nrows.to_s + "x" + native.ncols.to_s + " match=" + ok.to_s + "  (n=" + n.to_s + ")"
end

# Table: vocab=5, d_model=3. Fill with row-pattern: table[r,c] = r*10 + c.
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

idx = [2, 0, 4, 2]   # repeats index 2 to exercise scatter accumulation
n_idx = 4

# --- forward: gather ---
n_lookup = embed_lookup_native(tab, idx)
f_lookup = TinyNN.embed_lookup(tab, idx)
puts "lookup native[0,0]=" + n_lookup.flat[0].to_s + " ffi[0,0]=" + f_lookup.flat[0].to_s
cmp("embed_lookup", n_lookup, f_lookup, 1.0e-5)

# --- backward: scatter ---
d_out = Mat.new(n_idx, 3)
i = 0
while i < n_idx * 3
  d_out.flat[i] = (i + 1).to_f * 0.5   # 0.5, 1.0, 1.5, ...
  i = i + 1
end
n_back = embed_back_native(d_out, idx, 5)
f_back = TinyNN.embed_back(d_out, idx, 5)
puts "back native row2=" + n_back.flat[6].to_s + "," + n_back.flat[7].to_s + "," + n_back.flat[8].to_s
puts "back ffi    row2=" + f_back.flat[6].to_s + "," + f_back.flat[7].to_s + "," + f_back.flat[8].to_s
cmp("embed_back",   n_back,   f_back,   1.0e-4)
