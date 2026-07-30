# lib/toy/llm/primitives/muon.rb — L1 primitive: Muon's Newton–Schulz
# orthogonalization (toy#139 / K-series M9).
#
# Muon (Jordan et al.) replaces the per-coordinate magnitude scaling of
# Adam with an update GEOMETRY step: take the momentum matrix M, replace
# it with the nearest semi-orthogonal matrix (its UVᵀ from M = USVᵀ), and
# step along that. The orthogonalization is done with a quintic
# Newton–Schulz iteration — no SVD, just matmuls:
#
#   X ← M / ‖M‖_F
#   repeat 5×:  A = X Xᵀ ;  B = b·A + c·A² ;  X ← a·X + B X
#   (a, b, c) = (3.4445, −4.7750, 2.0315)
#
# The coefficients are tuned so five steps push every singular value into
# a band around 1 (NOT to machine-precision orthogonality — the quintic
# deliberately trades exactness for speed; the gate asserts the BAND, not
# equality).
#
# ── ggml index bookkeeping (the part that silently produces garbage) ──
# A tensor with ne = [n, m] is read here as the m×n matrix M where
# M[i, :] is the ne0-slice at ne1 = i. ggml_mul_mat(a, b) requires
# a.ne0 == b.ne0 and gives result[p, q] = Σ_k a[k,p]·b[k,q] with
# result.ne = [a.ne1, b.ne1]. From that:
#   mul_mat(X, X)            → ne=[m,m], value (M Mᵀ)[i,j]        = A
#   mul_mat(A, A)            → ne=[m,m], value (A²)[i,j]  (A is symmetric)
#   mul_mat(contᵀ(X), B)     → ne=[n,m], value (B M)[q,p]
# The last line is why the transpose is there at all: mul_mat contracts
# ne0, and B·M needs the contraction over M's ROW index, which only Xᵀ
# exposes as ne0.
#
# ── What this primitive deliberately does NOT do ──
# Jordan's implementation transposes when rows > cols so the iteration
# runs on the smaller Gram matrix. That is a cost optimization; the
# quintic converges to the same UVᵀ either way, so at toy widths we run
# it directly and skip the branch (noted rather than hidden).
#
# Every op has a ggml backward, but NOTHING here needs one: this runs on
# the momentum buffer inside the optimizer step, outside the loss graph.
#
# Pure module, `self.` methods only; no Cfg ctor / default args
# (landmine #4); no require_relative "tinynn" (the loader picks the
# backend; the mirror generator renames TinyNN. → TinyNN<Backend>.).

module Toy
  module LLM
    module Primitives
      module Muon
        NAME = :muon

        # Jordan's quintic coefficients + step count.
        NS_A = 3.4445
        NS_B = -4.7750
        NS_C = 2.0315
        NS_STEPS = 5
        # Momentum coefficient (the Muon default).
        MU = 0.95

        # Frobenius-normalize a 2d tensor: X / (‖X‖_F + eps). Composed
        # like GDN.l2_train's denominator (sum of squares → sqrt →
        # broadcast divide), because ggml_scale takes a CONSTANT, not a
        # tensor — the norm is data-dependent so it must ride a div.
        def self.fro_normalize(sess, x, n_elem)
          sq  = TinyNN.tnn_mul(sess, x, x)
          col = TinyNN.tnn_reshape_2d(sess, sq, n_elem, 1)     # ne=[n_elem,1]
          ss  = TinyNN.tnn_sum_rows(sess, col)                 # ne=[1,1]
          ssb = TinyNN.tnn_scale_bias(sess, ss, 1.0, 1.0e-14)
          den = TinyNN.tnn_sqrt(sess, ssb)
          TinyNN.tnn_div(sess, x, TinyNN.tnn_repeat(sess, den, x))
        end

        # ONE Newton–Schulz quintic step. x has ne=[n, m].
        def self.ns_step(sess, x, n, m)
          a_g = TinyNN.tnn_matmul(sess, x, x)                  # [m,m] = M Mᵀ
          a2  = TinyNN.tnn_matmul(sess, a_g, a_g)              # [m,m] = A²
          b_g = TinyNN.tnn_add(sess,
                  TinyNN.tnn_scale(sess, a_g, NS_B),
                  TinyNN.tnn_scale(sess, a2,  NS_C))
          xt  = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, x), m, n)
          bx  = TinyNN.tnn_matmul(sess, xt, b_g)               # [n,m] = B M
          TinyNN.tnn_add(sess, TinyNN.tnn_scale(sess, x, NS_A), bx)
        end

        # Full orthogonalization: normalize then iterate. Returns a
        # tensor shaped like x whose singular values sit in a band
        # around 1.
        #
        # SCALE INVARIANCE is the property that matters scientifically:
        # orth(c·X) == orth(X) for any c > 0, because the Frobenius
        # normalization divides the scale out. That is exactly what
        # makes Muon "geometry, not magnitude" — and it is what the
        # F9m experiment is probing against AdamW's per-coordinate
        # magnitude invariance.
        def self.orth(sess, x, n, m)
          xh = fro_normalize(sess, x, n * m)
          i = 0
          while i < NS_STEPS
            xh = ns_step(sess, xh, n, m)
            i = i + 1
          end
          xh
        end

        # The Muon update for one 2D weight, as graph ops:
        #   M ← mu·M + G          (in-place into the momentum buffer)
        #   step ← orth(M)        (Newton–Schulz)
        # Returns the orthogonalized step; the caller feeds it to an
        # SGD-shaped apply. The momentum write uses tnn_cpy, whose
        # dependency on the read keeps the order correct inside one
        # graph evaluation (read → combine → write).
        def self.update(sess, grad, mom, n, m)
          blend = TinyNN.tnn_add(sess, TinyNN.tnn_scale(sess, mom, MU), grad)
          m_new = TinyNN.tnn_cpy(sess, blend, mom)
          orth(sess, m_new, n, m)
        end
      end
    end
  end
end
