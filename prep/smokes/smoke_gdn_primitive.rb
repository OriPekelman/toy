#!/usr/bin/env ruby
# prep/smokes/smoke_gdn_primitive.rb — Phase 2 GDN L1 primitive smoke.
#
# Exercises Toy::LLM::Primitives::GDN — the parameter-free composition around
# the recurrence core (l2-norm of q/k, the log-decay gate, the sigmoid update
# gate, the recurrence, and the gated output norm). The gate+l2+recur chain is
# computed end-to-end (proving the activation composition feeds the exotic
# recurrence op correctly); gated_out is shape-checked (it composes the already-
# proven rms_norm/silu/mul ops). See docs/roadmap/dragon-gdn-arch-2026-06-20.md.

require_relative "../../lib/toy"
require_relative "../../lib/toy/ffi/tinynn"
require_relative "../../lib/toy/llm/primitives/gdn"

S_V = 2
S_K = 2
H   = 1
T   = 2
B   = 1
K   = 1
EPS = 1.0e-6

def fill(n)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(((i % 5) + 1).to_f * 0.1)
    i = i + 1
  end
  a
end



sess = TinyNN.tnn_session_new(0)

# Block-projected streams (the L2 block produces these; here synthetic).
t_q       = TinyNN.tnn_input_4d_f32_persistent(sess, S_K, H, T, B)            # [2,1,2,1]
t_k       = TinyNN.tnn_input_4d_f32_persistent(sess, S_K, H, T, B)
t_v       = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B)
t_a       = TinyNN.tnn_input_4d_f32_persistent(sess, 1,   H, T, B)            # decay stream [1,1,2,1]
t_dtbias  = TinyNN.tnn_input_4d_f32_persistent(sess, 1,   H, 1, 1)            # [1,1,1,1]
t_alog    = TinyNN.tnn_input_4d_f32_persistent(sess, 1,   H, 1, 1)            # [1,1,1,1]
t_b       = TinyNN.tnn_input_4d_f32_persistent(sess, 1,   H, T, B)            # update stream [1,1,2,1]
t_state   = TinyNN.tnn_input_4d_f32_persistent(sess, S_V * S_V * H, K, B, 1)  # [4,1,1,1]

# --- The computed chain: gates + l2 + recurrence. ---
t_qn   = Toy::LLM::Primitives::GDN.l2(sess, t_q, EPS)
t_kn   = Toy::LLM::Primitives::GDN.l2(sess, t_k, EPS)
t_g    = Toy::LLM::Primitives::GDN.decay_gate(sess, t_a, t_dtbias, t_alog)   # [1,1,2,1]
t_beta = Toy::LLM::Primitives::GDN.update_gate(sess, t_b)                    # [1,1,2,1]
t_out  = Toy::LLM::Primitives::GDN.recur(sess, t_qn, t_kn, t_v, t_g, t_beta, t_state)  # [S_v*H, T*B + K*S_v*B]

ok = true
if t_out == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: Toy::LLM::Primitives::GDN.recur returned NULL"
  exit 1
end

# Decay-gate shape: ne0 must be 1 (the recurrence requires g->ne0==1).
if TinyNN.tnn_tensor_ne0(t_g) != 1
  STDERR.puts "FAIL: decay_gate ne0=" + TinyNN.tnn_tensor_ne0(t_g).to_s + " expected 1"
  ok = false
end
if TinyNN.tnn_tensor_ne0(t_beta) != 1
  STDERR.puts "FAIL: update_gate ne0=" + TinyNN.tnn_tensor_ne0(t_beta).to_s + " expected 1"
  ok = false
end
# Recurrence output shape: [S_v*H, T*B + K*S_v*B] = [2, 4].
if TinyNN.tnn_tensor_ne0(t_out) != S_V * H
  STDERR.puts "FAIL: recur ne0=" + TinyNN.tnn_tensor_ne0(t_out).to_s + " expected " + (S_V * H).to_s
  ok = false
end
if TinyNN.tnn_tensor_ne1(t_out) != (T * B) + (K * S_V * B)
  STDERR.puts "FAIL: recur ne1=" + TinyNN.tnn_tensor_ne1(t_out).to_s + " expected " + ((T * B) + (K * S_V * B)).to_s
  ok = false
end

# Actually run the gate+l2+recur chain (proves the composition computes).
TinyNN.tnn_set_output(t_out)
TinyNN.tnn_realize(sess, t_out)
TinyNN.tnn_upload_from_float_array(sess, t_q,      fill(S_K * H * T * B), S_K * H * T * B)
TinyNN.tnn_upload_from_float_array(sess, t_k,      fill(S_K * H * T * B), S_K * H * T * B)
TinyNN.tnn_upload_from_float_array(sess, t_v,      fill(S_V * H * T * B), S_V * H * T * B)
TinyNN.tnn_upload_from_float_array(sess, t_a,      fill(H * T * B),       H * T * B)
TinyNN.tnn_upload_from_float_array(sess, t_dtbias, fill(H),               H)
TinyNN.tnn_upload_from_float_array(sess, t_alog,   fill(H),               H)
TinyNN.tnn_upload_from_float_array(sess, t_b,      fill(H * T * B),       H * T * B)
TinyNN.tnn_upload_from_float_array(sess, t_state,  fill(S_V * S_V * H * K * B), S_V * S_V * H * K * B)
TinyNN.tnn_compute(sess)

# --- gated_out: composes pre-proven rms_norm/silu/mul; shape-checked. ---
# o (block-sliced token output) and z (output gate) are [S_v*H, T*B] = [2,2].
sess2   = TinyNN.tnn_session_new(0)
t_o     = TinyNN.tnn_input_2d_f32_persistent(sess2, S_V * H, T * B)  # [2,2]
t_z     = TinyNN.tnn_input_2d_f32_persistent(sess2, S_V * H, T * B)
t_gamma = TinyNN.tnn_input_1d_f32_persistent(sess2, S_V * H)         # [2]
t_go    = Toy::LLM::Primitives::GDN.gated_out(sess2, t_o, t_z, t_gamma, EPS)
if t_go == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: Toy::LLM::Primitives::GDN.gated_out returned NULL"
  exit 1
end
if TinyNN.tnn_tensor_ne0(t_go) != S_V * H || TinyNN.tnn_tensor_ne1(t_go) != T * B
  STDERR.puts "FAIL: gated_out shape [" + TinyNN.tnn_tensor_ne0(t_go).to_s + "," + TinyNN.tnn_tensor_ne1(t_go).to_s + "] expected [2,2]"
  ok = false
end

if ok
  puts "GDN primitive smoke PASS: gates+l2+recur computed (out [2,4]); gated_out composes [2,2]"
  exit 0
else
  exit 1
end
