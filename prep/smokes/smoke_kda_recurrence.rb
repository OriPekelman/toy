#!/usr/bin/env ruby
# prep/smokes/smoke_kda_recurrence.rb — toy#137 (K2a): the KDA
# recurrence + decay parameterization, proven by REDUCTION.
#
# THE NULL: KDA's channel-wise forget gate (S' = Diag(α)S) must
# reproduce GDN's per-head SCALAR decay (S' = S·exp(g)) exactly when
# every channel of α carries the same value. GDN's unrolled recurrence
# is itself gate-verified against the fused ggml gated_delta_net
# kernel, so this chains KDA to that reference: if the reduction is
# byte-exact, the KDA recurrence IS the delta rule, and the only thing
# the channel-wise form adds is what it is supposed to add.
#
# Legs (all in ONE graph — the one-graph-twins rule; the sched is
# process-global):
#   1. REDUCTION: uniform-α KDA == scalar-g GDN, byte-exact.
#   2. LIVE: non-uniform α differs (the mechanism is not a no-op).
#   3. BOUND: with the K3 parameterization every α ∈ (e^-5, 1) — the
#      lower-bounded decay (K3 eq 5) that the chunkwise form needs.
#   4. HEADS: the 2-head striding path reduces too (per-head bases).
#
# Spinel hygiene: while loops, popped-empty literals, no interpolation.

require_relative "../../lib/toy"
require_relative "../../lib/toy/ffi/tinynn"
require_relative "../../lib/toy/llm/primitives/rms_norm"
require_relative "../../lib/toy/llm/primitives/gdn"
require_relative "../../lib/toy/llm/primitives/kda"

S_V     = 4
T_TOK   = 3
EPS     = 1.0e-5

def fillv(n, seed)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(((((i + seed) * 1103515245 + 12345) % 1000) - 500).to_f * 0.001)
    i = i + 1
  end
  a
end

def zeros(n)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(0.0)
    i = i + 1
  end
  a
end

sess = TinyNN.tnn_session_new(0)
TinyNN.tnn_session_set_graph_capacity(sess, 262144)

# ---- one-head tensors (n_heads = 1) ----
# q/k/v/g_chan packed [S_V, 1, T]; g_scal [1, 1, T]; beta [1, 1, T].
t_q  = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V)
t_k  = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V)
t_v  = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V)
t_gs = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, 1)      # scalar decay
t_gc = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V)    # channel decay
t_gx = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V)    # NON-uniform
t_b  = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, 1)
t_s0 = TinyNN.tnn_input_2d_f32_persistent(sess, S_V, S_V)
# decay-parameterization leg: a raw z stream + a per-head log-scale
t_z    = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V)
t_alog = TinyNN.tnn_input_2d_f32_persistent(sess, 1, 1)
# ---- two-head tensors (n_heads = 2) ----
t_q2  = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V * 2)
t_k2  = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V * 2)
t_v2  = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V * 2)
t_gs2 = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, 2)
t_gc2 = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, S_V * 2)
t_b2  = TinyNN.tnn_input_2d_f32_persistent(sess, T_TOK, 2)
t_s02 = TinyNN.tnn_input_2d_f32_persistent(sess, S_V, S_V)

TinyNN.tnn_finalize_weights(sess)

# Values. The decay is NEGATIVE (log-space) so exp(g) < 1.
qv = fillv(S_V * T_TOK, 11)
kv = fillv(S_V * T_TOK, 23)
vv = fillv(S_V * T_TOK, 37)
bv = [0.4, 0.7, 0.55]
gs = [-0.3, -0.7, -0.2]
gc = zeros(S_V * T_TOK)
gx = zeros(S_V * T_TOK)
ti = 0
while ti < T_TOK
  ci = 0
  while ci < S_V
    # ggml [S_V, 1, T] packing: token-major stride S_V.
    gc[ti * S_V + ci] = gs[ti]                       # UNIFORM per token
    gx[ti * S_V + ci] = gs[ti] - 0.15 * ci.to_f      # channel-VARYING
    ci = ci + 1
  end
  ti = ti + 1
end
TinyNN.tnn_upload_from_float_array(sess, t_q, qv, S_V * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_k, kv, S_V * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_v, vv, S_V * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_gs, gs, T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_gc, gc, S_V * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_gx, gx, S_V * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_b, bv, T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_s0, fillv(S_V * S_V, 5), S_V * S_V)
TinyNN.tnn_upload_from_float_array(sess, t_z, fillv(S_V * T_TOK, 61), S_V * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_alog, zeros(1), 1)   # A_h init 0

q2v = fillv(S_V * 2 * T_TOK, 71)
k2v = fillv(S_V * 2 * T_TOK, 83)
v2v = fillv(S_V * 2 * T_TOK, 97)
gs2 = [-0.25, -0.5, -0.35, -0.6, -0.15, -0.45]   # [1,H,T] packing: H-major
gc2 = zeros(S_V * 2 * T_TOK)
ti = 0
while ti < T_TOK
  hh = 0
  while hh < 2
    ci = 0
    while ci < S_V
      # [S_V, H, T]: index = t*(S_V*H) + h*S_V + c
      gc2[ti * S_V * 2 + hh * S_V + ci] = gs2[ti * 2 + hh]
      ci = ci + 1
    end
    hh = hh + 1
  end
  ti = ti + 1
end
TinyNN.tnn_upload_from_float_array(sess, t_q2, q2v, S_V * 2 * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_k2, k2v, S_V * 2 * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_v2, v2v, S_V * 2 * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_gs2, gs2, 2 * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_gc2, gc2, S_V * 2 * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_b2, [0.5, 0.6, 0.45, 0.65, 0.55, 0.5], 2 * T_TOK)
TinyNN.tnn_upload_from_float_array(sess, t_s02, fillv(S_V * S_V, 9), S_V * S_V)

# ---- build: pack to 3d, then the two recurrences side by side ----
q3  = TinyNN.tnn_reshape_3d(sess, t_q,  S_V, 1, T_TOK)
k3  = TinyNN.tnn_reshape_3d(sess, t_k,  S_V, 1, T_TOK)
v3  = TinyNN.tnn_reshape_3d(sess, t_v,  S_V, 1, T_TOK)
gs3 = TinyNN.tnn_reshape_3d(sess, t_gs, 1,   1, T_TOK)
gc3 = TinyNN.tnn_reshape_3d(sess, t_gc, S_V, 1, T_TOK)
gx3 = TinyNN.tnn_reshape_3d(sess, t_gx, S_V, 1, T_TOK)
b3  = TinyNN.tnn_reshape_3d(sess, t_b,  1,   1, T_TOK)

o_gdn = Toy::LLM::Primitives::GDN.recur_unrolled(sess, q3, k3, v3, gs3, b3, t_s0, S_V, 1, 0, T_TOK)
o_kda = Toy::LLM::Primitives::KDA.recur_unrolled(sess, q3, k3, v3, gc3, b3, t_s0, S_V, 1, 0, T_TOK)
o_var = Toy::LLM::Primitives::KDA.recur_unrolled(sess, q3, k3, v3, gx3, b3, t_s0, S_V, 1, 0, T_TOK)
TinyNN.tnn_set_output(o_gdn)
TinyNN.tnn_set_output(o_kda)
TinyNN.tnn_set_output(o_var)

# decay parameterization: g = G_MIN*sigmoid(exp(A)*z), then alpha = exp(g)
z3   = TinyNN.tnn_reshape_3d(sess, t_z, S_V, 1, T_TOK)
al3  = TinyNN.tnn_reshape_3d(sess, t_alog, 1, 1, 1)
g_kd = Toy::LLM::Primitives::KDA.decay_logits(sess, z3, al3)
a_kd = TinyNN.tnn_exp(sess, g_kd)
TinyNN.tnn_set_output(g_kd)
TinyNN.tnn_set_output(a_kd)

q23  = TinyNN.tnn_reshape_3d(sess, t_q2,  S_V, 2, T_TOK)
k23  = TinyNN.tnn_reshape_3d(sess, t_k2,  S_V, 2, T_TOK)
v23  = TinyNN.tnn_reshape_3d(sess, t_v2,  S_V, 2, T_TOK)
gs23 = TinyNN.tnn_reshape_3d(sess, t_gs2, 1,   2, T_TOK)
gc23 = TinyNN.tnn_reshape_3d(sess, t_gc2, S_V, 2, T_TOK)
b23  = TinyNN.tnn_reshape_3d(sess, t_b2,  1,   2, T_TOK)
o_g_h1 = Toy::LLM::Primitives::GDN.recur_unrolled(sess, q23, k23, v23, gs23, b23, t_s02, S_V, 2, 1, T_TOK)
o_k_h1 = Toy::LLM::Primitives::KDA.recur_unrolled(sess, q23, k23, v23, gc23, b23, t_s02, S_V, 2, 1, T_TOK)
TinyNN.tnn_set_output(o_g_h1)
TinyNN.tnn_set_output(o_k_h1)

# LANDMINE: every extra root must be add_to_graph'd BEFORE
# build_forward_only (the toy#120 ordering rule) — otherwise its
# buffer is never allocated and the download aborts in
# ggml_backend_tensor_get.
TinyNN.tnn_add_to_graph(sess, o_gdn)
TinyNN.tnn_add_to_graph(sess, o_var)
TinyNN.tnn_add_to_graph(sess, a_kd)
TinyNN.tnn_add_to_graph(sess, g_kd)
TinyNN.tnn_add_to_graph(sess, o_g_h1)
TinyNN.tnn_add_to_graph(sess, o_k_h1)
TinyNN.tnn_build_forward_only(sess, o_kda)
TinyNN.tnn_compute(sess)

n_out = S_V * T_TOK
bg = zeros(n_out); bk = zeros(n_out); bv2 = zeros(n_out)
bgh = zeros(n_out); bkh = zeros(n_out)
ba = zeros(n_out); bgl = zeros(n_out)
TinyNN.tnn_download_to_f64_array(sess, o_gdn, bg, n_out)
TinyNN.tnn_download_to_f64_array(sess, o_kda, bk, n_out)
TinyNN.tnn_download_to_f64_array(sess, o_var, bv2, n_out)
TinyNN.tnn_download_to_f64_array(sess, a_kd, ba, n_out)
TinyNN.tnn_download_to_f64_array(sess, g_kd, bgl, n_out)
TinyNN.tnn_download_to_f64_array(sess, o_g_h1, bgh, n_out)
TinyNN.tnn_download_to_f64_array(sess, o_k_h1, bkh, n_out)

fails = 0

# ---- leg 1: the reduction null ----
i = 0
maxd = 0.0
while i < n_out
  d = bg[i] - bk[i]
  if d < 0.0
    d = 0.0 - d
  end
  if d > maxd
    maxd = d
  end
  i = i + 1
end
if maxd == 0.0
  puts "kda: REDUCTION ok — uniform-alpha KDA byte-equals scalar-g GDN (1 head)"
else
  puts "kda: REDUCTION FAIL — max |gdn - kda| = " + maxd.to_s
  fails = fails + 1
end

# ---- leg 2: channel-varying alpha is live ----
i = 0
diff_seen = 0
while i < n_out
  if bv2[i] != bk[i]
    diff_seen = 1
  end
  i = i + 1
end
if diff_seen == 1
  puts "kda: LIVE ok — channel-varying alpha changes the output"
else
  puts "kda: LIVE FAIL — varying alpha produced identical output"
  fails = fails + 1
end

# ---- leg 3: the lower bound ----
lo = Math.exp(-5.0)
i = 0
bad = 0
while i < n_out
  if ba[i] <= lo || ba[i] >= 1.0
    bad = bad + 1
  end
  if bgl[i] <= -5.0 || bgl[i] >= 0.0
    bad = bad + 1
  end
  i = i + 1
end
if bad == 0
  puts "kda: BOUND ok — every alpha in (e^-5, 1), every g in (-5, 0)"
else
  puts "kda: BOUND FAIL — " + bad.to_s + " out-of-range decay values"
  fails = fails + 1
end

# ---- leg 4: 2-head striding reduces too ----
i = 0
maxd = 0.0
while i < n_out
  d = bgh[i] - bkh[i]
  if d < 0.0
    d = 0.0 - d
  end
  if d > maxd
    maxd = d
  end
  i = i + 1
end
if maxd == 0.0
  puts "kda: HEADS ok — head-1 of a 2-head pack reduces byte-exactly"
else
  puts "kda: HEADS FAIL — max |gdn - kda| = " + maxd.to_s
  fails = fails + 1
end

if fails == 0
  puts "kda-recurrence: ok"
else
  puts "kda-recurrence: FAIL (" + fails.to_s + ")"
end
