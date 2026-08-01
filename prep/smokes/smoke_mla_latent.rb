#!/usr/bin/env ruby
# prep/smokes/smoke_mla_latent.rb — K-series M2: the Gated-MLA pieces,
# proven by the properties that make them MLA rather than "some
# matmuls that happen to train".
#
# There is no reference implementation to byte-match at this seam (the
# deepseek2 inference path is a different engine, different weights,
# different RoPE story), so the gate asserts what the MECHANISM means:
#
#   1. FACTORIZATION. The latent sandwich is a factorization of an
#      ordinary projection, not an approximation of one. With the
#      kv_a_norm OFF, r = inner and W_kv_a = I, the composition
#      W_k_b · (I · h) must equal W_k_b · h BYTE-FOR-BYTE. This is the
#      null that catches transposed index bookkeeping — the failure
#      mode that produces plausible-looking garbage.
#
#   2. ECONOMY. The whole point of the latent is parameter count:
#      r·d + 2·inner·r must beat 2·inner·d at the ranks we actually
#      use. Asserted as an inequality AND reported as numbers, because
#      at toy widths the crossover is close enough that a badly chosen
#      r makes the "efficient" path more expensive — worth knowing
#      rather than assuming.
#
#   3. CAUSALITY. Perturbing token t's value must leave outputs at
#      positions < t bit-identical. This is the leg that can actually
#      catch a mask failure: a duplicated-window null cannot (softmax
#      over duplicated logits is the same mixture either way — the
#      lesson the MoE batch-mask work paid for), but a FUTURE-TOKEN
#      PERTURBATION can, because a leaked future value changes the
#      past output or it does not.
#
#   4. THE GATE GATES. σ(W_g h) ⊙ RMSNorm(õ) with a large negative
#      gate stream must drive the output to ~0, and with a large
#      positive one must pass RMSNorm(õ) through. A gate that is wired
#      to the wrong operand still trains; this says it is a gate.
#
# Spinel hygiene: while loops, popped-empty literals, no interpolation.

require_relative "../../lib/toy"
require_relative "../../lib/toy/ffi/tinynn"
require_relative "../../lib/toy/llm/primitives/rms_norm"
require_relative "../../lib/toy/llm/primitives/gqa"
require_relative "../../lib/toy/llm/primitives/gdn"
require_relative "../../lib/toy/llm/primitives/kda"
require_relative "../../lib/toy/llm/primitives/mla"

D_MODEL = 8
S_V     = 4     # head width
T       = 5     # tokens
EPS     = 1.0e-5

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
    a.push(((((i + seed) * 1103515245 + 12345) % 1000) - 500).to_f * 0.003)
    i = i + 1
  end
  a
end

fails = 0

# ---------------------------------------------------------------- 1 --
# FACTORIZATION: identity latent, norm off  =>  W_k_b·(I·h) == W_k_b·h
sess = TinyNN.tnn_session_new(0)
TinyNN.tnn_session_set_graph_capacity(sess, 262144)

t_h   = TinyNN.tnn_input_2d_f32_persistent(sess, T, D_MODEL)   # ne=[D_MODEL, T]
t_eye = TinyNN.tnn_input_2d_f32_persistent(sess, D_MODEL, D_MODEL)
t_wkb = TinyNN.tnn_input_2d_f32_persistent(sess, S_V, D_MODEL)
TinyNN.tnn_finalize_weights(sess)

# input_2d_f32_persistent(a, b) -> ne=[b, a] (the block's "(out, in)"
# convention), so an activation of shape [d_model, T] — ne0 = d_model,
# ne1 = T — is allocated as (T, D_MODEL). Getting this backwards is
# what mul_mat aborts on, loudly, which is the good case.
hv = fillv(D_MODEL * T, 11)
TinyNN.tnn_upload_from_float_array(sess, t_h, hv, D_MODEL * T)
ev = zeros(D_MODEL * D_MODEL)
i = 0
while i < D_MODEL
  ev[i * D_MODEL + i] = 1.0
  i = i + 1
end
TinyNN.tnn_upload_from_float_array(sess, t_eye, ev, D_MODEL * D_MODEL)
TinyNN.tnn_upload_from_float_array(sess, t_wkb, fillv(S_V * D_MODEL, 23), S_V * D_MODEL)

# Latent path with norm OFF and W_kv_a = I, then the up-projection.
c_id  = Toy::LLM::Primitives::MLA.kv_latent(sess, t_h, t_eye,
                                            TinyNN.tnn_null_ptr, 0, EPS)
k_lat = TinyNN.tnn_matmul(sess, t_wkb, c_id)
# The ordinary projection it must reduce to.
k_pln = TinyNN.tnn_matmul(sess, t_wkb, t_h)
TinyNN.tnn_set_output(k_lat)
TinyNN.tnn_set_output(k_pln)
TinyNN.tnn_add_to_graph(sess, k_pln)
TinyNN.tnn_build_forward_only(sess, k_lat)
TinyNN.tnn_compute(sess)

bl = zeros(S_V * T); bp = zeros(S_V * T)
TinyNN.tnn_download_to_f64_array(sess, k_lat, bl, S_V * T)
TinyNN.tnn_download_to_f64_array(sess, k_pln, bp, S_V * T)
worst = 0.0
i = 0
while i < S_V * T
  d = bl[i] - bp[i]
  if d < 0.0
    d = 0.0 - d
  end
  if d > worst
    worst = d
  end
  i = i + 1
end
if worst == 0.0
  puts "mla: FACTORIZATION ok — identity latent (norm off) is BYTE-EXACT vs the plain projection"
else
  puts "mla: FACTORIZATION FAIL — max deviation " + worst.to_s
  fails = fails + 1
end
TinyNN.tnn_session_free(sess)

# ---------------------------------------------------------------- 2 --
# ECONOMY: the latent must actually cost less at the ranks we ship.
inner = S_V * 2
r_def = inner / 2
lat = Toy::LLM::Primitives::MLA.kv_params_latent(D_MODEL, inner, r_def)
pln = Toy::LLM::Primitives::MLA.kv_params_plain(D_MODEL, inner)
if lat < pln
  puts "mla: ECONOMY ok — latent KV params " + lat.to_s + " < plain " + pln.to_s +
       " at r=" + r_def.to_s + " (d=" + D_MODEL.to_s + ", inner=" + inner.to_s + ")"
else
  puts "mla: ECONOMY FAIL — latent " + lat.to_s + " >= plain " + pln.to_s +
       " at r=" + r_def.to_s + "; the sandwich is not buying anything at this shape"
  fails = fails + 1
end

# ---------------------------------------------------------------- 3 --
# CAUSALITY: perturbing the LAST token must not move earlier outputs.
# Two sessions, identical except v[last]; compare head outputs.
def attend_once(vpert)
  sess = TinyNN.tnn_session_new(0)
  TinyNN.tnn_session_set_graph_capacity(sess, 262144)
  t_q = TinyNN.tnn_input_2d_f32_persistent(sess, T, S_V)   # ne=[S_V, T]
  t_k = TinyNN.tnn_input_2d_f32_persistent(sess, T, S_V)
  t_v = TinyNN.tnn_input_2d_f32_persistent(sess, T, S_V)
  TinyNN.tnn_finalize_weights(sess)
  TinyNN.tnn_upload_from_float_array(sess, t_q, fillv(S_V * T, 31), S_V * T)
  TinyNN.tnn_upload_from_float_array(sess, t_k, fillv(S_V * T, 37), S_V * T)
  vv = fillv(S_V * T, 41)
  if vpert == 1
    # Perturb ONLY the final token's value vector (row T-1).
    j = 0
    while j < S_V
      vv[(T - 1) * S_V + j] = vv[(T - 1) * S_V + j] + 7.0
      j = j + 1
    end
  end
  TinyNN.tnn_upload_from_float_array(sess, t_v, vv, S_V * T)
  o = Toy::LLM::Primitives::MLA.head_attend(sess, t_q, t_k, t_v, S_V)
  TinyNN.tnn_set_output(o)
  TinyNN.tnn_build_forward_only(sess, o)
  TinyNN.tnn_compute(sess)
  b = zeros(S_V * T)
  TinyNN.tnn_download_to_f64_array(sess, o, b, S_V * T)
  TinyNN.tnn_session_free(sess)
  b
end

b0 = attend_once(0)
b1 = attend_once(1)
past_worst = 0.0
i = 0
while i < (T - 1) * S_V
  d = b0[i] - b1[i]
  if d < 0.0
    d = 0.0 - d
  end
  if d > past_worst
    past_worst = d
  end
  i = i + 1
end
last_moved = 0.0
i = (T - 1) * S_V
while i < T * S_V
  d = b0[i] - b1[i]
  if d < 0.0
    d = 0.0 - d
  end
  if d > last_moved
    last_moved = d
  end
  i = i + 1
end
# Two-sided: the past must be FROZEN and the perturbed position must
# actually move (otherwise a dead output would pass the first half).
if past_worst == 0.0 && last_moved > 0.1
  puts "mla: CAUSALITY ok — perturbing token " + (T - 1).to_s +
       " leaves all earlier outputs BYTE-IDENTICAL (and moves its own by " +
       last_moved.to_s + ")"
else
  puts "mla: CAUSALITY FAIL — past deviation " + past_worst.to_s +
       " (want 0), own deviation " + last_moved.to_s + " (want > 0.1)"
  fails = fails + 1
end

# ---------------------------------------------------------------- 4 --
# THE GATE GATES: sigma(large negative) kills the output; sigma(large
# positive) passes RMSNorm(o) through.
def gated_once(gate_fill)
  sess = TinyNN.tnn_session_new(0)
  TinyNN.tnn_session_set_graph_capacity(sess, 262144)
  t_o   = TinyNN.tnn_input_2d_f32_persistent(sess, T, S_V)  # ne=[S_V, T]
  t_wgx = TinyNN.tnn_input_2d_f32_persistent(sess, T, S_V)
  t_g   = TinyNN.tnn_input_1d_f32_persistent(sess, S_V)
  TinyNN.tnn_finalize_weights(sess)
  TinyNN.tnn_upload_from_float_array(sess, t_o, fillv(S_V * T, 53), S_V * T)
  gv = zeros(S_V * T)
  i = 0
  while i < S_V * T
    gv[i] = gate_fill
    i = i + 1
  end
  TinyNN.tnn_upload_from_float_array(sess, t_wgx, gv, S_V * T)
  ones = zeros(S_V)
  i = 0
  while i < S_V
    ones[i] = 1.0
    i = i + 1
  end
  TinyNN.tnn_upload_from_float_array(sess, t_g, ones, S_V)
  y = Toy::LLM::Primitives::KDA.gated_out(sess, t_o, t_wgx, t_g, EPS)
  TinyNN.tnn_set_output(y)
  TinyNN.tnn_build_forward_only(sess, y)
  TinyNN.tnn_compute(sess)
  b = zeros(S_V * T)
  TinyNN.tnn_download_to_f64_array(sess, y, b, S_V * T)
  TinyNN.tnn_session_free(sess)
  b
end

closed = gated_once(-20.0)
open_g = gated_once(20.0)
cmax = 0.0
omax = 0.0
i = 0
while i < S_V * T
  c = closed[i]
  if c < 0.0
    c = 0.0 - c
  end
  if c > cmax
    cmax = c
  end
  o = open_g[i]
  if o < 0.0
    o = 0.0 - o
  end
  if o > omax
    omax = o
  end
  i = i + 1
end
if cmax < 1.0e-5 && omax > 0.1
  puts "mla: GATE ok — sigma(-20) closes the output to " + cmax.to_s +
       ", sigma(+20) passes " + omax.to_s + " through"
else
  puts "mla: GATE FAIL — closed max " + cmax.to_s + " (want < 1e-5), open max " +
       omax.to_s + " (want > 0.1)"
  fails = fails + 1
end

if fails == 0
  puts "mla-latent: ok"
else
  puts "mla-latent: FAIL (" + fails.to_s + ")"
end
