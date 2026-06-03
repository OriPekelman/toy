# M2 phase-1 smoke: exercise the MoE primitives on a tiny 2-expert
# toy. Confirms the binding shapes line up and the graph realizes +
# computes without aborting. Reference values picked so the result
# is hand-verifiable.
#
# Shapes:
#   experts: [d_in=4, d_out=4, n_experts=2]
#     expert 0: identity (4x4 I)
#     expert 1: 2× identity (2 * I)
#   input:    x ∈ R^{4} (one token)
#   ids:      [0]   → "route to expert 0 only"
#
# Expected result: expert 0 · x = x (identity).
# Then re-run with ids = [1] → 2x.
# Then top_k(router=[0.1, 0.9], k=1) → [1] (expert 1 wins).

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

sess = TinyNN.tnn_session_new(0)

D_IN  = 4
D_OUT = 4
N_EXP = 2
K     = 1
T     = 1

# Allocate persistent expert stack [d_in, d_out, n_experts] and
# a router weight + per-token input. ids is built in compute ctx.
t_experts = TinyNN.tnn_input_3d_f32_persistent(sess, D_IN, D_OUT, N_EXP)
t_router  = TinyNN.tnn_input_2d_f32_persistent(sess, N_EXP, D_IN)   # ne=[D_IN, N_EXP]
t_x       = TinyNN.tnn_input_2d_f32(sess, T, D_IN)                  # ne=[D_IN, T]
t_ids     = TinyNN.tnn_input_2d_f32(sess, T, K)                     # placeholder for shape

# Router output → top-K indices.
t_logits  = TinyNN.tnn_matmul(sess, t_router, t_x)    # ne=[N_EXP, T]
t_top_idx = TinyNN.tnn_top_k(sess, t_logits, K)       # ne=[K, T]

# Sparse expert matmul. With K=1 and one token, this picks one
# expert and runs its matmul on x.
t_out = TinyNN.tnn_mul_mat_id(sess, t_experts, t_x, t_top_idx)

TinyNN.tnn_finalize_weights(sess)

# Build the expert stack: expert 0 = I, expert 1 = 2*I. Stored as
# row-major-by-expert via the persistent f32 upload.
exps = [0.0]; exps.pop
ne = D_IN * D_OUT
e = 0
while e < N_EXP
  i = 0
  while i < D_IN
    j = 0
    while j < D_OUT
      v = (i == j ? 1.0 : 0.0) * (e == 0 ? 1.0 : 2.0)
      exps.push(v)
      j = j + 1
    end
    i = i + 1
  end
  e = e + 1
end
TinyNN.tnn_upload_from_float_array(sess, t_experts, exps, ne * N_EXP)

# Router: expert 0 gets logit 0.1; expert 1 gets logit 0.9 for any
# input. Use a simple bias-like router that ignores x's content:
# every column of router maps to a constant per expert.
# Layout ne=[D_IN, N_EXP]; data[i_expert * D_IN + i_in].
router = [0.0]; router.pop
exi = 0
while exi < N_EXP
  ini = 0
  while ini < D_IN
    router.push(exi == 0 ? 0.025 : 0.225)  # x sums to 4; 4*0.025=0.1, 4*0.225=0.9
    ini = ini + 1
  end
  exi = exi + 1
end
TinyNN.tnn_upload_from_float_array(sess, t_router, router, N_EXP * D_IN)

TinyNN.tnn_realize(sess, t_out)

# Run with x = [1, 1, 1, 1].
TinyNN.tnn_scratch_set(sess, 0, 1.0)
TinyNN.tnn_scratch_set(sess, 1, 1.0)
TinyNN.tnn_scratch_set(sess, 2, 1.0)
TinyNN.tnn_scratch_set(sess, 3, 1.0)
TinyNN.tnn_upload(sess, t_x)

TinyNN.tnn_compute(sess)

# Verify top_k picked expert 1 (logit 0.9 > 0.1).
TinyNN.tnn_download(sess, t_top_idx)
picked = TinyNN.tnn_scratch_get_i32(sess, 0)
puts "moe_smoke: top_k picked expert " + picked.to_s + " (expected 1)"

# Verify mul_mat_id ran. With expert 1 = 2*I and x = [1,1,1,1],
# expected output = [2,2,2,2]. Result ne=[D_OUT, K, T] = [4, 1, 1].
TinyNN.tnn_download(sess, t_out)
out = [0.0]; out.pop
i = 0
while i < D_OUT
  out.push(TinyNN.tnn_scratch_get(sess, i))
  i = i + 1
end
puts "moe_smoke: out = " + out.inspect + " (expected [2.0, 2.0, 2.0, 2.0])"

expected = 2.0
ok = true
i = 0
while i < D_OUT
  d = out[i] - expected
  if d < 0; d = -d; end
  if d > 1.0e-4; ok = false; end
  i = i + 1
end
puts "moe_smoke: " + (ok ? "PASS" : "FAIL") +
     " (picked=" + picked.to_s + ", match=" + ok.to_s + ")"

TinyNN.tnn_session_free(sess)
exit (ok && picked == 1 ? 0 : 1)
