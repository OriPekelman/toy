#!/usr/bin/env ruby
# prep/smokes/smoke_gdn_unrolled_parity_mh.rb — Phase 5 (Dragon/GDN arc).
#
# Multi-head parity: the per-head GDN.recur_unrolled, looped over H heads and
# concatenated along ne0, reproduces the FUSED tnn_gated_delta_net token outputs
# for a MULTI-HEAD shape (H=2) to eps. Proves the strided per-head slicing
# (token stride S_v·H, head base S_v·head) the trainable block will use matches
# the kernel's head packing. See docs/roadmap/dragon-gdn-arch-2026-06-20.md (P5).
#
#   FUSED   : tnn_gated_delta_net -> [S_v*H, T*B + K*S_v*B]; token cols = first T*B
#   UNROLLED: concat_h recur_unrolled(head=h) -> [S_v*H, T]

require_relative "../../lib/toy"
require_relative "../../lib/toy/ffi/tinynn"
require_relative "../../lib/toy/llm/primitives/gdn"

S_V = 2
H   = 2   # MULTI-HEAD
T   = 3
B   = 1
K   = 1
EPS = 1.0e-6

def fill(n)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(((i % 7) + 1).to_f * 0.11 - 0.17)
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
TinyNN.tnn_finalize_weights(sess_f)

t_fused = Toy::LLM::Primitives::GDN.recur(sess_f, fq, fk, fv, fg, fbeta, fstate)
TinyNN.tnn_set_output(t_fused)
TinyNN.tnn_realize(sess_f, t_fused)
TinyNN.tnn_upload_from_float_array(sess_f, fq,     q_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fk,     k_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fv,     v_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fg,     g_data,     H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fbeta,  beta_data,  H * T * B)
TinyNN.tnn_upload_from_float_array(sess_f, fstate, state_data, S_V * S_V * H * K * B)
TinyNN.tnn_compute(sess_f)

fused_n   = S_V * H * ((T * B) + (K * S_V * B))
fused_buf = zeros(fused_n)
TinyNN.tnn_download_to_f64_array(sess_f, t_fused, fused_buf, fused_n)

# ---------------- UNROLLED path (head loop) ----------------
sess_u  = TinyNN.tnn_session_new(0)
uq      = TinyNN.tnn_input_4d_f32_persistent(sess_u, S_V, H, T, B)
uk      = TinyNN.tnn_input_4d_f32_persistent(sess_u, S_V, H, T, B)
uv      = TinyNN.tnn_input_4d_f32_persistent(sess_u, S_V, H, T, B)
ug      = TinyNN.tnn_input_4d_f32_persistent(sess_u, 1,   H, T, B)
ubeta   = TinyNN.tnn_input_4d_f32_persistent(sess_u, 1,   H, T, B)
# State for all heads in one [S_v, S_v*H] tensor; head h is columns [h*S_v..).
ustate  = TinyNN.tnn_input_2d_f32_persistent(sess_u, S_V, S_V * H)
TinyNN.tnn_finalize_weights(sess_u)

fbytes = 4
t_mh = TinyNN.tnn_null_ptr
hh = 0
while hh < H
  # Head hh's [S_v,S_v] initial state — a contiguous column-block view.
  st_h = TinyNN.tnn_view_2d(sess_u, ustate, S_V, S_V, S_V * fbytes, hh * S_V * S_V * fbytes)
  o_h  = Toy::LLM::Primitives::GDN.recur_unrolled(sess_u, uq, uk, uv, ug, ubeta, st_h, S_V, H, hh, T)
  if hh == 0
    t_mh = o_h
  else
    t_mh = TinyNN.tnn_concat(sess_u, t_mh, o_h, 0)   # stack heads along ne0
  end
  hh = hh + 1
end

if TinyNN.tnn_tensor_ne0(t_mh) != S_V * H || TinyNN.tnn_tensor_ne1(t_mh) != T
  STDERR.puts "FAIL: mh shape [" + TinyNN.tnn_tensor_ne0(t_mh).to_s + "," +
              TinyNN.tnn_tensor_ne1(t_mh).to_s + "] expected [" + (S_V * H).to_s + "," + T.to_s + "]"
  exit 1
end
TinyNN.tnn_set_output(t_mh)
TinyNN.tnn_realize(sess_u, t_mh)
TinyNN.tnn_upload_from_float_array(sess_u, uq,    q_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, uk,    k_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, uv,    v_data,     S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, ug,    g_data,     H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, ubeta, beta_data,  H * T * B)
TinyNN.tnn_upload_from_float_array(sess_u, ustate, state_data, S_V * S_V * H)
TinyNN.tnn_compute(sess_u)

token_n  = S_V * H * T
mh_buf   = zeros(token_n)
TinyNN.tnn_download_to_f64_array(sess_u, t_mh, mh_buf, token_n)

# Compare: fused token outputs are the first T*B columns (col-major, ne0=S_v*H),
# i.e. the first S_v*H*T floats — same packing as the head-concatenated unrolled.
ok = true
i = 0
while i < token_n
  diff = fused_buf[i] - mh_buf[i]
  if diff < 0.0
    diff = -diff
  end
  if diff > EPS
    puts "FAIL: elem " + i.to_s + " fused=" + fused_buf[i].to_s + " mh=" + mh_buf[i].to_s + " |diff|=" + diff.to_s
    ok = false
  end
  i = i + 1
end

if ok
  puts "GDN unrolled-parity-mh smoke PASS: H=" + H.to_s + " head-looped recur_unrolled matches " +
       "fused kernel within " + EPS.to_s + " over " + token_n.to_s + " elems (T=" + T.to_s + ")"
  exit 0
else
  exit 1
end
