# A/B smoke: manual silu+mul vs fused ggml_swiglu_split.
#
# Purpose: macro-op fusion candidate. ggml-cuda fuses (SILU, MUL) into
# a single kernel when called via ggml_swiglu_split. This smoke measures
# whether the fusion delivers a measurable wallclock win at FFN shape,
# to decide whether to thread tnn_swiglu_split through toy's FFN graph
# builders.
#
# Manual:  N_LAYERS × (silu(gate) → mul(silu, up))  = 2 N_LAYERS ops
# Fused:   N_LAYERS × swiglu_split(gate, up)        = 1 N_LAYERS ops
#
# Both consume the same gate and up tensors and produce the same logical
# output. The difference is whether ggml-cuda dispatches one fused
# kernel or two separate ones (silu then mul) per layer.

require_relative "../lib/transformer"
require_relative "../lib/tinynn_cuda"

# Llama-1.5B FFN shape: d_model=1536, d_ff=8960, T=256.
D_FF     = (ENV["D_FF"]     || "8960").to_i
T        = (ENV["T"]        || "256").to_i
N_LAYERS = (ENV["N_LAYERS"] || "28").to_i
ITERS    = (ENV["ITERS"]    || "100").to_i
WARMUP   = (ENV["WARMUP"]   || "10").to_i

# ============================================================
# Manual graph: N_LAYERS × (silu → mul)
# gate, up: ne=(D_FF, T) — call (T, D_FF) → ne0=D_FF, ne1=T.
# ============================================================
sess_m = TinyNNCuda.tnn_session_new(1)
t_gate_m = TinyNNCuda.tnn_input_2d_f32_persistent(sess_m, T, D_FF)
t_up_m   = TinyNNCuda.tnn_input_2d_f32_persistent(sess_m, T, D_FF)
last_m = TinyNNCuda.tnn_null_ptr
li = 0
while li < N_LAYERS
  t_silu = TinyNNCuda.tnn_silu(sess_m, t_gate_m)
  t_act  = TinyNNCuda.tnn_mul(sess_m, t_silu, t_up_m)
  if li == 0
    last_m = t_act
  else
    last_m = TinyNNCuda.tnn_add(sess_m, last_m, t_act)
  end
  li = li + 1
end
TinyNNCuda.tnn_set_output(last_m)
TinyNNCuda.tnn_realize(sess_m, last_m)

# ============================================================
# Fused graph: N_LAYERS × swiglu_split
# ============================================================
sess_f = TinyNNCuda.tnn_session_new(1)
t_gate_f = TinyNNCuda.tnn_input_2d_f32_persistent(sess_f, T, D_FF)
t_up_f   = TinyNNCuda.tnn_input_2d_f32_persistent(sess_f, T, D_FF)
last_f = TinyNNCuda.tnn_null_ptr
li = 0
while li < N_LAYERS
  t_act = TinyNNCuda.tnn_swiglu_split(sess_f, t_gate_f, t_up_f)
  if li == 0
    last_f = t_act
  else
    last_f = TinyNNCuda.tnn_add(sess_f, last_f, t_act)
  end
  li = li + 1
end
TinyNNCuda.tnn_set_output(last_f)
TinyNNCuda.tnn_realize(sess_f, last_f)

# ============================================================
# Zero inputs.
# ============================================================
m_gate = Mat.new(T, D_FF)
m_up   = Mat.new(T, D_FF)
i = 0
n = T * D_FF
while i < n; m_gate.flat[i] = 0.0; m_up.flat[i] = 0.0; i = i + 1; end

# ============================================================
# Time manual
# ============================================================
i = 0
while i < WARMUP
  TinyNNCuda.upload_row_major(sess_m, t_gate_m, m_gate)
  TinyNNCuda.upload_row_major(sess_m, t_up_m,   m_up)
  TinyNNCuda.tnn_compute(sess_m)
  i = i + 1
end
t0 = Time.now
i = 0
while i < ITERS
  TinyNNCuda.upload_row_major(sess_m, t_gate_m, m_gate)
  TinyNNCuda.upload_row_major(sess_m, t_up_m,   m_up)
  TinyNNCuda.tnn_compute(sess_m)
  i = i + 1
end
t_m_ms = (Time.now - t0) * 1000.0
per_iter_m = t_m_ms / ITERS.to_f

# ============================================================
# Time fused
# ============================================================
i = 0
while i < WARMUP
  TinyNNCuda.upload_row_major(sess_f, t_gate_f, m_gate)
  TinyNNCuda.upload_row_major(sess_f, t_up_f,   m_up)
  TinyNNCuda.tnn_compute(sess_f)
  i = i + 1
end
t0 = Time.now
i = 0
while i < ITERS
  TinyNNCuda.upload_row_major(sess_f, t_gate_f, m_gate)
  TinyNNCuda.upload_row_major(sess_f, t_up_f,   m_up)
  TinyNNCuda.tnn_compute(sess_f)
  i = i + 1
end
t_f_ms = (Time.now - t0) * 1000.0
per_iter_f = t_f_ms / ITERS.to_f

puts "=== ab_smoke_swiglu_split_cuda ==="
puts "shape: D_FF=" + D_FF.to_s + " T=" + T.to_s + " N_LAYERS=" + N_LAYERS.to_s +
     " ITERS=" + ITERS.to_s
puts ""
puts "manual (silu + mul): " + (N_LAYERS * 2).to_s + " ops + " + (N_LAYERS - 1).to_s + " add chain"
puts "  total ms = " + t_m_ms.to_s + "  per iter = " + per_iter_m.to_s + " ms"
puts ""
puts "fused (swiglu_split): " + N_LAYERS.to_s + " ops + " + (N_LAYERS - 1).to_s + " add chain"
puts "  total ms = " + t_f_ms.to_s + "  per iter = " + per_iter_f.to_s + " ms"
puts ""
speedup = per_iter_m / per_iter_f
saved   = per_iter_m - per_iter_f
puts "per-iter speedup (manual/fused): " + speedup.to_s + "x"
puts "per-iter saved:                  " + saved.to_s + " ms"
puts "per-step estimate (fwd only, × " + N_LAYERS.to_s + " layers's worth): " +
     saved.to_s + " ms"
