#!/usr/bin/env ruby
# prep/smokes/smoke_dragon_attn_prims.rb — Phase 2 smoke for the Dragon
# attention-side L1 primitives: DiffAttention (lambda_scalar, combine, subln),
# ScalableSoftmax, and DepthScale. lambda_scalar+combine build (the differential
# lambda dot-reduce + the A1-lambda*A2 broadcast); scalable_softmax computes
# end-to-end; depth_scale + subln compose. All shape-checked against the
# documented contracts. See docs/roadmap/dragon-gdn-arch-2026-06-20.md.
#
# Shape note (cost me a detour): tnn_input_2d_f32_persistent(rows, cols) builds
# ne0=cols, ne1=rows, and rms_norm normalises over ne0 — so a per-head norm of
# feature width F over T tokens needs the tensor allocated as (rows=T, cols=F)
# and gamma=[F]; getting this backwards makes rms_norm's gamma-mul fail
# ggml_can_repeat. (There was NO diff-attention broadcast bug.)

require_relative "../../lib/toy"
require_relative "../../lib/toy/ffi/tinynn"
require_relative "../../lib/toy/llm/primitives/diff_attention"
require_relative "../../lib/toy/llm/primitives/scalable_softmax"
require_relative "../../lib/toy/llm/primitives/depth_scale"

HD  = 4
T   = 4
EPS = 1.0e-6
LAMBDA_INIT = 0.2

def fill(n)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(((i % 5) + 1).to_f * 0.1)
    i = i + 1
  end
  a
end

ok = true
sess = TinyNN.tnn_session_new(0)

# DiffAttention.lambda_scalar + combine: the differential lambda and A1-lambda*A2.
t_lq1 = TinyNN.tnn_input_1d_f32_persistent(sess, HD)
t_lk1 = TinyNN.tnn_input_1d_f32_persistent(sess, HD)
t_lq2 = TinyNN.tnn_input_1d_f32_persistent(sess, HD)
t_lk2 = TinyNN.tnn_input_1d_f32_persistent(sess, HD)
t_a1  = TinyNN.tnn_input_2d_f32_persistent(sess, T, T)
t_a2  = TinyNN.tnn_input_2d_f32_persistent(sess, T, T)
t_lambda = Toy::LLM::Primitives::DiffAttention.lambda_scalar(sess, t_lq1, t_lk1, t_lq2, t_lk2, LAMBDA_INIT)
t_comb   = Toy::LLM::Primitives::DiffAttention.combine(sess, t_a1, t_a2, t_lambda)
if t_lambda == TinyNN.tnn_null_ptr || t_comb == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: DiffAttention lambda/combine returned NULL"
  exit 1
end
if TinyNN.tnn_tensor_ne0(t_comb) != T || TinyNN.tnn_tensor_ne1(t_comb) != T
  STDERR.puts "FAIL: combine shape [" + TinyNN.tnn_tensor_ne0(t_comb).to_s + "," + TinyNN.tnn_tensor_ne1(t_comb).to_s + "]"; ok = false
end

# ScalableSoftmax: SSMax-scaled softmax over [T,T] scores.
t_scores = TinyNN.tnn_input_2d_f32_persistent(sess, T, T)
t_attn   = Toy::LLM::Primitives::ScalableSoftmax.attend(sess, t_scores, TinyNN.tnn_null_ptr, 0.5, 0.0)

# DepthScale: 1/sqrt(depth) on a [HD,T] sublayer input.
t_x  = TinyNN.tnn_input_2d_f32_persistent(sess, HD, T)
t_ds = Toy::LLM::Primitives::DepthScale.apply(sess, t_x, 0.7071)

# DiffAttention.subln: per-head rms_norm * (1-lambda_init) on a [2*HD,T] output.
# NOTE: tnn_input_2d_f32_persistent(rows, cols) builds ne0=cols, ne1=rows; the
# norm is over ne0, so the feature width 2*HD must be ne0 → pass (rows=T, cols=2*HD).
t_o     = TinyNN.tnn_input_2d_f32_persistent(sess, T, 2 * HD)
t_gamma = TinyNN.tnn_input_1d_f32_persistent(sess, 2 * HD)
t_subln = Toy::LLM::Primitives::DiffAttention.subln(sess, t_o, t_gamma, EPS, 1.0 - LAMBDA_INIT)

if t_attn == TinyNN.tnn_null_ptr || t_ds == TinyNN.tnn_null_ptr || t_subln == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: a Dragon attn primitive returned NULL"
  exit 1
end

# Shapes.
if TinyNN.tnn_tensor_ne0(t_attn) != T || TinyNN.tnn_tensor_ne1(t_attn) != T
  STDERR.puts "FAIL: scalable_softmax shape"; ok = false
end
if TinyNN.tnn_tensor_ne0(t_ds) != HD || TinyNN.tnn_tensor_ne1(t_ds) != T
  STDERR.puts "FAIL: depth_scale shape"; ok = false
end
if TinyNN.tnn_tensor_ne0(t_subln) != 2 * HD || TinyNN.tnn_tensor_ne1(t_subln) != T
  STDERR.puts "FAIL: subln shape [" + TinyNN.tnn_tensor_ne0(t_subln).to_s + "," + TinyNN.tnn_tensor_ne1(t_subln).to_s + "]"; ok = false
end

# Compute the scalable_softmax chain (proves SSMax computes end-to-end).
TinyNN.tnn_set_output(t_attn)
TinyNN.tnn_realize(sess, t_attn)
TinyNN.tnn_upload_from_float_array(sess, t_scores, fill(T * T), T * T)
TinyNN.tnn_compute(sess)

if ok
  puts "Dragon attn prims smoke PASS: diff lambda+combine build; ssmax computed; depth_scale+subln compose"
  exit 0
else
  exit 1
end
