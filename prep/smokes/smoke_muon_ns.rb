#!/usr/bin/env ruby
# prep/smokes/smoke_muon_ns.rb — toy#139: Muon's Newton–Schulz
# orthogonalization, proven by its DEFINING PROPERTIES.
#
# There is no reference implementation to byte-match here, so the gate
# asserts what orthogonalization MEANS — and one of the properties is
# the very thing the F9m experiment is about:
#
#   1. NO-ROTATION: an already-orthonormal input comes back along the
#      SAME axes — diagonal entries in the band, everything else ~0.
#      (Not "unchanged": the Frobenius normalization rescales an
#      orthonormal M×N input to singular values 1/sqrt(M), and the
#      quintic then pulls them back into a band around 1 rather than
#      exactly to 1. Preserved DIRECTION is the invariant; exact
#      preservation is not, and asserting it would be asserting a
#      property Newton–Schulz does not have.) Transposed index
#      bookkeeping fails this immediately.
#   2. BAND: for a random matrix, Y Yᵀ ≈ I — every singular value
#      pushed near 1. The quintic trades exactness for speed, so the
#      assertion is a BAND (diag in [0.5,1.5], off-diag small), not
#      equality.
#   3. SCALE INVARIANCE: orth(c·X) == orth(X) for c > 0, to within
#      float noise. THIS is Muon's "geometry not magnitude" claim, and
#      it is exactly the AdamW confound tao#139 wants isolated — if it
#      did not hold, the experiment would be measuring something else.
#   4. NON-TRIVIAL: the output is not just a rescaled input (a
#      degenerate iteration that returned c·X would pass 2 and 3).
#
# Spinel hygiene: while loops, popped-empty literals, no interpolation.

require_relative "../../lib/toy"
require_relative "../../lib/toy/ffi/tinynn"
require_relative "../../lib/toy/llm/primitives/muon"

N = 6      # ne0 — matrix columns
M = 4      # ne1 — matrix rows

def zeros(n)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(0.0)
    i = i + 1
  end
  a
end

def fillv(n, seed)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(((((i + seed) * 1103515245 + 12345) % 1000) - 500).to_f * 0.002)
    i = i + 1
  end
  a
end

sess = TinyNN.tnn_session_new(0)
TinyNN.tnn_session_set_graph_capacity(sess, 262144)

t_x    = TinyNN.tnn_input_2d_f32_persistent(sess, M, N)   # ne=[N,M]
t_xs   = TinyNN.tnn_input_2d_f32_persistent(sess, M, N)   # 100x scaled twin
t_orth = TinyNN.tnn_input_2d_f32_persistent(sess, M, N)   # orthonormal rows
TinyNN.tnn_finalize_weights(sess)

xv = fillv(N * M, 17)
TinyNN.tnn_upload_from_float_array(sess, t_x, xv, N * M)
xs = zeros(N * M)
i = 0
while i < N * M
  xs[i] = xv[i] * 100.0
  i = i + 1
end
TinyNN.tnn_upload_from_float_array(sess, t_xs, xs, N * M)
# Rows = the first M standard basis vectors: M[i, j] = 1 iff i == j.
ov = zeros(N * M)
i = 0
while i < M
  ov[i * N + i] = 1.0
  i = i + 1
end
TinyNN.tnn_upload_from_float_array(sess, t_orth, ov, N * M)

y_r = Toy::LLM::Primitives::Muon.orth(sess, t_x,    N, M)
y_s = Toy::LLM::Primitives::Muon.orth(sess, t_xs,   N, M)
y_o = Toy::LLM::Primitives::Muon.orth(sess, t_orth, N, M)
# Gram matrix of the random arm's output: Y Yᵀ, ne=[M,M].
gram = TinyNN.tnn_matmul(sess, y_r, y_r)
TinyNN.tnn_set_output(y_r)
TinyNN.tnn_set_output(y_s)
TinyNN.tnn_set_output(y_o)
TinyNN.tnn_set_output(gram)
TinyNN.tnn_add_to_graph(sess, y_s)
TinyNN.tnn_add_to_graph(sess, y_o)
TinyNN.tnn_add_to_graph(sess, gram)
TinyNN.tnn_build_forward_only(sess, y_r)
TinyNN.tnn_compute(sess)

br = zeros(N * M); bs = zeros(N * M); bo = zeros(N * M); bg = zeros(M * M)
TinyNN.tnn_download_to_f64_array(sess, y_r,  br, N * M)
TinyNN.tnn_download_to_f64_array(sess, y_s,  bs, N * M)
TinyNN.tnn_download_to_f64_array(sess, y_o,  bo, N * M)
TinyNN.tnn_download_to_f64_array(sess, gram, bg, M * M)

fails = 0

# ---- 1. no rotation: orthonormal in -> same axes, band-scaled ----
diag_lo = 1.0e30
off_hi  = 0.0
i = 0
while i < M
  j = 0
  while j < N
    v = bo[i * N + j]
    av = v
    if av < 0.0
      av = 0.0 - av
    end
    if i == j
      if av < diag_lo
        diag_lo = av
      end
    else
      if av > off_hi
        off_hi = av
      end
    end
    j = j + 1
  end
  i = i + 1
end
if diag_lo > 0.5 && off_hi < 0.15
  puts "muon: NO-ROTATION ok — orthonormal input keeps its axes (min diag " +
       diag_lo.to_s + ", max off-diag " + off_hi.to_s + ")"
else
  puts "muon: NO-ROTATION FAIL — min diag " + diag_lo.to_s + " max off-diag " + off_hi.to_s
  fails = fails + 1
end

# ---- 2. band: Y Yᵀ ≈ I ----
bad = 0
i = 0
while i < M
  j = 0
  while j < M
    v = bg[j * M + i]
    if i == j
      if v < 0.5 || v > 1.5
        bad = bad + 1
      end
    else
      if v > 0.15 || v < -0.15
        bad = bad + 1
      end
    end
    j = j + 1
  end
  i = i + 1
end
if bad == 0
  puts "muon: BAND ok — Y Yt within the quintic band (diag in [0.5,1.5], off-diag |.|<=0.15)"
else
  puts "muon: BAND FAIL — " + bad.to_s + " Gram entries out of band"
  fails = fails + 1
end

# ---- 3. scale invariance: orth(100x) == orth(x) ----
worst = 0.0
i = 0
while i < N * M
  d = br[i] - bs[i]
  if d < 0.0
    d = 0.0 - d
  end
  if d > worst
    worst = d
  end
  i = i + 1
end
if worst < 1.0e-4
  puts "muon: SCALE-INVARIANCE ok — orth(100x) == orth(x) (max dev " + worst.to_s + ")"
else
  puts "muon: SCALE-INVARIANCE FAIL — max dev " + worst.to_s
  fails = fails + 1
end

# ---- 4. non-trivial: the output is not a rescaled input ----
# A degenerate iteration returning c*X would satisfy 2 and 3, so check
# that the ratio y/x is NOT constant across entries.
rmin = 1.0e30
rmax = -1.0e30
i = 0
while i < N * M
  if xv[i] > 0.05 || xv[i] < -0.05
    r = br[i] / xv[i]
    if r < rmin
      rmin = r
    end
    if r > rmax
      rmax = r
    end
  end
  i = i + 1
end
if rmax - rmin > 0.05
  puts "muon: NON-TRIVIAL ok — y/x ratio spans " + rmin.to_s + " .. " + rmax.to_s
else
  puts "muon: NON-TRIVIAL FAIL — output is a uniform rescale of the input"
  fails = fails + 1
end

if fails == 0
  puts "muon-ns: ok"
else
  puts "muon-ns: FAIL (" + fails.to_s + ")"
end
