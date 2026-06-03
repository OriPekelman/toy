# A/B parity for softmax_back.
# Native reference: dx_i = a_i * (dy_i - Σ_k a_k*dy_k)   per row
#
# TODO: rms_norm_back's TinyNN wrapper compiles and runs but its output
# doesn't match ggml's documented formula (or any I've reproduced from
# the source). Probably either an arg-ordering subtlety I'm missing or
# a backend dispatch issue; leaving the FFI function in place and
# coming back to it. Not blocking the rest of the backward op set.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

def softmax_back_native(softmax_out, dy)
  out = Mat.new(softmax_out.nrows, softmax_out.ncols)
  r = 0
  while r < softmax_out.nrows
    nc = softmax_out.ncols
    dot = 0.0
    c = 0
    while c < nc
      dot = dot + softmax_out.flat[r * nc + c] * dy.flat[r * nc + c]
      c = c + 1
    end
    c = 0
    while c < nc
      ai = softmax_out.flat[r * nc + c]
      di = dy.flat[r * nc + c]
      out.flat[r * nc + c] = ai * (di - dot)
      c = c + 1
    end
    r = r + 1
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

# --- softmax_back: (3, 4). Need actual softmax output as input. ---
xsm = Mat.new(3, 4)
i = 0
while i < 12
  xsm.flat[i] = (i.to_f * 0.4 - 2.0)
  i = i + 1
end
# Use the project's own softmax via FFI to get the output (already verified by ab-smoke-softmax).
a_out = TinyNN.softmax(xsm)
dysm = Mat.new(3, 4)
i = 0
while i < 12
  dysm.flat[i] = (i.to_f * 0.2 - 1.0)
  i = i + 1
end
nat2 = softmax_back_native(a_out, dysm)
ffi2 = TinyNN.softmax_back(a_out, dysm)
puts "native[0]=" + nat2.flat[0].to_s + " ffi[0]=" + ffi2.flat[0].to_s
cmp("softmax_back", nat2, ffi2, 1.0e-3)
