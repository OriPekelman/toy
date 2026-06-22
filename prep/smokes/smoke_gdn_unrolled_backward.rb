#!/usr/bin/env ruby
# prep/smokes/smoke_gdn_unrolled_backward.rb — Phase 4 (Dragon/GDN arc, Path B).
#
# The crux of Path B: proves the UNROLLED GDN recurrence
# (Toy::LLM::Primitives::GDN.recur_unrolled) is END-TO-END DIFFERENTIABLE —
# ggml builds + runs a backward graph through the whole gated delta rule and
# produces FINITE, NON-ZERO gradients on q/k/v, with NO hand-written
# fused-kernel backward (ggml has none for GATED_DELTA_NET). This is what makes
# Dragon's GDN layers trainable. See docs/roadmap/dragon-gdn-arch-2026-06-20.md.
#
# Recipe mirrors the engine's train step (alloc -> set_param -> finalize ->
# build fwd -> loss -> build_backward -> opt_step per param -> realize -> upload
# -> compute_backward). The opt_step nodes CONSUME the grads, which is what keeps
# galloc computing + retaining them (a grad with no consumer is never written —
# the engine reads grads via drift_grad only because opt_step makes them live).
#
#   loss = sum(recur_unrolled(q,k,v,g,beta,state0))   # scalar
#   backward -> dL/dq, dL/dk, dL/dv  (finite + at least one |g|>0)

require_relative "../../lib/toy"
require_relative "../../lib/toy/ffi/tinynn"
require_relative "../../lib/toy/llm/primitives/gdn"

S_V = 2
H   = 1
T   = 3
B   = 1
NG  = S_V * H * T * B   # elems per q/k/v param

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

# Grad health: every element finite (not NaN/Inf) and at least one |g| > tiny.
def grad_ok(buf, n, label)
  any_nz = false
  i = 0
  ok = true
  while i < n
    v = buf[i]
    if v != v
      puts "FAIL: " + label + " grad[" + i.to_s + "] is NaN"
      ok = false
    end
    av = v < 0.0 ? -v : v
    if av > 1.0e30
      puts "FAIL: " + label + " grad[" + i.to_s + "] is Inf-ish (" + v.to_s + ")"
      ok = false
    end
    if av > 1.0e-9
      any_nz = true
    end
    i = i + 1
  end
  if !any_nz
    puts "FAIL: " + label + " grad is all-zero (no gradient flowed)"
    ok = false
  end
  ok
end

q_data     = fill(NG)
k_data     = fill(NG)
v_data     = fill(NG)
g_data     = fill(H * T * B)
beta_data  = fill(H * T * B)
state_data = fill(S_V * S_V)
hp_data    = [0.001, 0.9, 0.95, 1.0e-8, 0.0, 0.9, 0.95]   # adam: lr b1 b2 eps wd b1h b2h

sess = TinyNN.tnn_session_new(0)
tq      = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B)
tk      = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B)
tv      = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B)
tg      = TinyNN.tnn_input_4d_f32_persistent(sess, 1,   H, T, B)
tbeta   = TinyNN.tnn_input_4d_f32_persistent(sess, 1,   H, T, B)
tstate0 = TinyNN.tnn_input_2d_f32_persistent(sess, S_V, S_V)
# Adam moments per param (opt_step_adamw requires m/v to MATCH the param shape,
# so 4d [S_v,H,T,B] like q/k/v, not flat 1d) + a shared hyper-param vector.
mq = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B); vq = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B)
mk = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B); vk = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B)
mv = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B); vv = TinyNN.tnn_input_4d_f32_persistent(sess, S_V, H, T, B)
hp = TinyNN.tnn_input_1d_f32_persistent(sess, 7)

# Load-bearing order (gpt2_seq_engine.rb:128): alloc -> set_param -> finalize.
TinyNN.tnn_set_param(tq)
TinyNN.tnn_set_param(tk)
TinyNN.tnn_set_param(tv)
TinyNN.tnn_finalize_weights(sess)

t_out = Toy::LLM::Primitives::GDN.recur_unrolled(sess, tq, tk, tv, tg, tbeta, tstate0, S_V, 1, 0, T)
if t_out == TinyNN.tnn_null_ptr
  puts "FAIL: recur_unrolled returned NULL"
  exit 0
end

t_loss = TinyNN.tnn_sum(sess, t_out)   # scalar reduction
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)

TinyNN.tnn_build_forward_only(sess, t_loss)
TinyNN.tnn_build_backward(sess)

# opt_step per param — the grad CONSUMER that forces galloc to compute + retain
# the grads (exactly the engine's pattern). We read the grads after, not the
# updated weights, so the Adam update is just the vehicle.
gq = TinyNN.tnn_tensor_grad(sess, tq)
gk = TinyNN.tnn_tensor_grad(sess, tk)
gv = TinyNN.tnn_tensor_grad(sess, tv)
if gq == TinyNN.tnn_null_ptr || gk == TinyNN.tnn_null_ptr || gv == TinyNN.tnn_null_ptr
  puts "FAIL: a tensor_grad handle is NULL (no grad node built)"
  exit 0
end
oq = TinyNN.tnn_opt_step_adamw(sess, tq, gq, mq, vq, hp); TinyNN.tnn_extend_backward_graph(sess, oq)
ok2 = TinyNN.tnn_opt_step_adamw(sess, tk, gk, mk, vk, hp); TinyNN.tnn_extend_backward_graph(sess, ok2)
ov = TinyNN.tnn_opt_step_adamw(sess, tv, gv, mv, vv, hp); TinyNN.tnn_extend_backward_graph(sess, ov)

# Keep the grads alive for the host read (set_output BEFORE realize).
TinyNN.tnn_set_output(gq)
TinyNN.tnn_set_output(gk)
TinyNN.tnn_set_output(gv)

TinyNN.tnn_pin_all_graph_b_nodes(sess)
TinyNN.tnn_realize_backward(sess)

# Seed the backward: ggml_graph_reset zeros the grad accumulators AND sets the
# loss tensor's grad to 1.0. Without it the loss seed is 0 → every grad is 0
# (this is the engine's step! reset, the step I was missing).
TinyNN.tnn_graph_reset(sess)

TinyNN.tnn_upload_from_float_array(sess, tq,      q_data,     NG)
TinyNN.tnn_upload_from_float_array(sess, tk,      k_data,     NG)
TinyNN.tnn_upload_from_float_array(sess, tv,      v_data,     NG)
TinyNN.tnn_upload_from_float_array(sess, tg,      g_data,     H * T * B)
TinyNN.tnn_upload_from_float_array(sess, tbeta,   beta_data,  H * T * B)
TinyNN.tnn_upload_from_float_array(sess, tstate0, state_data, S_V * S_V)
TinyNN.tnn_upload_from_float_array(sess, hp,      hp_data,    7)
TinyNN.tnn_zero_tensor(sess, mq); TinyNN.tnn_zero_tensor(sess, vq)
TinyNN.tnn_zero_tensor(sess, mk); TinyNN.tnn_zero_tensor(sess, vk)
TinyNN.tnn_zero_tensor(sess, mv); TinyNN.tnn_zero_tensor(sess, vv)

TinyNN.tnn_compute_backward(sess)


bq = zeros(NG); TinyNN.tnn_download_to_f64_array(sess, gq, bq, NG)
bk = zeros(NG); TinyNN.tnn_download_to_f64_array(sess, gk, bk, NG)
bv = zeros(NG); TinyNN.tnn_download_to_f64_array(sess, gv, bv, NG)
puts "dL/dv=" + bv[0].to_s + "," + bv[1].to_s + "," + bv[2].to_s +
     " dL/dq[0]=" + bq[0].to_s + " dL/dk[0]=" + bk[0].to_s

ok = true
ok = grad_ok(bq, NG, "dL/dq") && ok
ok = grad_ok(bk, NG, "dL/dk") && ok
ok = grad_ok(bv, NG, "dL/dv") && ok

if ok
  puts "GDN unrolled-backward smoke PASS: recur_unrolled is differentiable — " +
       "finite non-zero dL/dq,dL/dk,dL/dv over " + NG.to_s + " elems (T=" + T.to_s + ")"
  exit 0
else
  puts "GDN unrolled-backward smoke FAIL"
  exit 0
end
