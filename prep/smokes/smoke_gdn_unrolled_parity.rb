#!/usr/bin/env ruby
# prep/smokes/smoke_gdn_unrolled_parity.rb — Phase 4 (Dragon/GDN arc, Path B).
#
# Proves the UNROLLED, autograd-differentiable GDN recurrence
# (Toy::LLM::Primitives::GDN.recur_unrolled) reproduces the FUSED inference
# kernel (tnn_gated_delta_net) token outputs to eps — the numeric-parity gate
# that lets training use the composition (whose every op has a ggml backward)
# while inference keeps the fast fused kernel. See
# docs/roadmap/dragon-gdn-arch-2026-06-20.md (Phase 4).
#
# Two sessions, identical uploaded data:
#   FUSED   : tnn_gated_delta_net -> [S_v*H, T*B + K*S_v*B]; token cols = first T*B
#   UNROLLED: recur_unrolled       -> [S_v, T]
# Compare the first S_v*T token-output floats elementwise.

require_relative "../../lib/toy"
require_relative "../../lib/toy/ffi/tinynn"
require_relative "../../lib/toy/llm/primitives/gdn"

S_V = 2   # value/key/query head dim (kernel reads S_v elems of q/k)
H   = 1   # single head per recur_unrolled call
T   = 3   # tokens (>2 to exercise the carried state across multiple steps)
B   = 1   # single seq
K   = 1   # one state snapshot slot
EPS = 1.0e-6

# Distinct-ish per-element values so a transpose/orientation bug can't hide.
def fill(n)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(((i % 7) + 1).to_f * 0.13 - 0.2)
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

q_data     = fill(S_V * H * T * B)
k_data     = fill(S_V * H * T * B)
v_data     = fill(S_V * H * T * B)
g_data     = fill(H * T * B)
beta_data  = fill(H * T * B)
state_data = fill(S_V * S_V * H * K * B)

# ---------------- FUSED path ----------------
sess_f  = TinyNN.tnn_session_new(0)
fq      = TinyNN.tnn_input_4d_f32_persistent(sess_f, S_V, H, T, B)
fk      = TinyNN.tnn_input_4d_f32_persistent(sess_f, S_V, H, T, B)
fv      = TinyNN.tnn_input_4d_f32_persistent(sess_f, S_V, H, T, B)
fg      = TinyNN.tnn_input_4d_f32_persistent(sess_f, 1,   H, T, B)
fbeta   = TinyNN.tnn_input_4d_f32_persistent(sess_f, 1,   H, T, B)
fstate  = TinyNN.tnn_input_4d_f32_persistent(sess_f, S_V * S_V * H, K, B, 1)
# Finalize so the persistent input leaves get real backend buffers (data != NULL)
# before the compute graph is built — required once the graph takes VIEWS of them
# (a view's data = src->data + offset; an unfinalized src->data == NULL leaves the
# view unplaceable → galloc buffer_id == -1 abort).
TinyNN.tnn_finalize_weights(sess_f)

t_fused = Toy::LLM::Primitives::GDN.recur(sess_f, fq, fk, fv, fg, fbeta, fstate)
if t_fused == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: fused recur returned NULL"
  exit 1
end
TinyNN.tnn_set_output(t_fused)
TinyNN.tnn_realize(sess_f, t_fused)
TinyNN.tnn_upload_from_float_array(sess_f, fq,     q_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fk,     k_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fv,     v_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fg,     g_data,     H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fbeta,  beta_data,  H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fstate, state_data, S_V * S_V * H * K * B)
TinyNN.tnn_compute(sess_f)

fused_n   = S_V * H * ((T * B) + (K * S_V * B))   # full packed output element count
fused_buf = zeros(fused_n)
TinyNN.tnn_download_to_f64_array(sess_f, t_fused, fused_buf, fused_n)

# ---------------- UNROLLED path ----------------
sess_u  = TinyNN.tnn_session_new(0)
uq      = TinyNN.tnn_input_4d_f32_persistent(sess_u, S_V, H, T, B)
uk      = TinyNN.tnn_input_4d_f32_persistent(sess_u, S_V, H, T, B)
uv      = TinyNN.tnn_input_4d_f32_persistent(sess_u, S_V, H, T, B)
ug      = TinyNN.tnn_input_4d_f32_persistent(sess_u, 1,   H, T, B)
ubeta   = TinyNN.tnn_input_4d_f32_persistent(sess_u, 1,   H, T, B)
ustate0 = TinyNN.tnn_input_2d_f32_persistent(sess_u, S_V, S_V)   # [S_v,S_v] state[i,j]
TinyNN.tnn_finalize_weights(sess_u)   # real buffers before the view-heavy graph

t_unroll = Toy::LLM::Primitives::GDN.recur_unrolled(sess_u, uq, uk, uv, ug, ubeta, ustate0, S_V, 1, 0, T)
if t_unroll == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: recur_unrolled returned NULL"
  exit 1
end
if TinyNN.tnn_tensor_ne0(t_unroll) != S_V || TinyNN.tnn_tensor_ne1(t_unroll) != T
  STDERR.puts "FAIL: unrolled shape [" + TinyNN.tnn_tensor_ne0(t_unroll).to_s + "," +
              TinyNN.tnn_tensor_ne1(t_unroll).to_s + "] expected [" + S_V.to_s + "," + T.to_s + "]"
  exit 1
end
TinyNN.tnn_set_output(t_unroll)
TinyNN.tnn_realize(sess_u, t_unroll)
TinyNN.tnn_upload_from_float_array(sess_u, uq,    q_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, uk,    k_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, uv,    v_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, ug,    g_data,     H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, ubeta, beta_data,  H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, ustate0, state_data, S_V * S_V)
TinyNN.tnn_compute(sess_u)

token_n    = S_V * T
unroll_buf = zeros(token_n)
TinyNN.tnn_download_to_f64_array(sess_u, t_unroll, unroll_buf, token_n)

# ---------------- Compare token outputs ----------------
# Fused token outputs are the first T*B columns (column-major, ne0=S_v*H fast),
# i.e. the first S_v*T floats of fused_buf. Unrolled is [S_v, T] => same order.
ok = true
i = 0
while i < token_n
  diff = fused_buf[i] - unroll_buf[i]
  if diff < 0.0
    diff = -diff
  end
  if diff > EPS
    STDERR.puts "FAIL: token elem " + i.to_s + " fused=" + fused_buf[i].to_s +
                " unrolled=" + unroll_buf[i].to_s + " |diff|=" + diff.to_s
    ok = false
  end
  i = i + 1
end

if ok
  puts "GDN unrolled-parity smoke PASS: recur_unrolled matches fused tnn_gated_delta_net " +
       "token outputs within " + EPS.to_s + " over " + token_n.to_s + " elems (T=" + T.to_s + ")"
  exit 0
else
  exit 1
end
