# A/B smoke: OUT_PROD vs MUL_MAT at gradient-update shape.
#
# Purpose: the attribution shows OUT_PROD at 30.8 % of step time
# (1598 calls, mean 26.5 µs). Is OUT_PROD specifically expensive on
# ggml-cuda, or is the cost in line with an equivalent MUL_MAT?
#
# Same logical work, different input layout:
#   MUL_MAT : a.ne[0] == b.ne[0] (K on ne0), result ne=(a.ne1, b.ne1)
#   OUT_PROD: a.ne[1] == b.ne[1] (K on ne1), result ne=(a.ne0, b.ne0)
#
# Both produce a [M, N] result computed by sum-over-K of two tensors;
# only the memory layout differs. We allocate the matching input
# shapes per op and compare per-iteration wallclock.
#
# Shape comes from real LoRA-Q backward: K = seq_len (the reduce
# axis), M = d_model or R (gradient output rows), N = R or d_head
# (gradient output cols).

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn_cuda"

K       = (ENV["K"]      || "256").to_i      # reduce axis (sequence length in real LoRA bwd)
M       = (ENV["M"]      || "1536").to_i     # output rows (d_model for LoRA-A grad)
N       = (ENV["N"]      || "8").to_i        # output cols (R for LoRA-A grad)
N_OPS   = (ENV["N_OPS"]  || "24").to_i       # ops per compute (mimics 12 heads × 2 grads)
ITERS   = (ENV["ITERS"]  || "100").to_i
WARMUP  = (ENV["WARMUP"] || "10").to_i

# ============================================================
# MUL_MAT graph: N_OPS independent matmuls.
#   a: ne=(K, M) — call (M, K). K on ne0 (reduce), M on ne1 (out rows).
#   b: ne=(K, N) — call (N, K). K on ne0, N on ne1.
#   out: ne=(M, N)
# ============================================================
sess_mm = TinyNNCuda.tnn_session_new(1)
t_a_mm = TinyNNCuda.tnn_input_2d_f32_persistent(sess_mm, M, K)
t_b_mm = TinyNNCuda.tnn_input_2d_f32_persistent(sess_mm, N, K)
last_mm = TinyNNCuda.tnn_null_ptr
oi = 0
while oi < N_OPS
  t = TinyNNCuda.tnn_matmul(sess_mm, t_a_mm, t_b_mm)
  if oi == 0
    last_mm = t
  else
    last_mm = TinyNNCuda.tnn_add(sess_mm, last_mm, t)
  end
  oi = oi + 1
end
TinyNNCuda.tnn_set_output(last_mm)
TinyNNCuda.tnn_realize(sess_mm, last_mm)

# ============================================================
# OUT_PROD graph: N_OPS independent out_prod calls. Different layout:
#   a: ne=(M, K) — call (K, M). M on ne0 (out rows), K on ne1 (reduce).
#   b: ne=(N, K) — call (K, N).
#   out: ne=(M, N)  -- same shape as the MUL_MAT result above.
# ============================================================
sess_op = TinyNNCuda.tnn_session_new(1)
t_a_op = TinyNNCuda.tnn_input_2d_f32_persistent(sess_op, K, M)
t_b_op = TinyNNCuda.tnn_input_2d_f32_persistent(sess_op, K, N)
last_op = TinyNNCuda.tnn_null_ptr
oi = 0
while oi < N_OPS
  t = TinyNNCuda.tnn_out_prod(sess_op, t_a_op, t_b_op)
  if oi == 0
    last_op = t
  else
    last_op = TinyNNCuda.tnn_add(sess_op, last_op, t)
  end
  oi = oi + 1
end
TinyNNCuda.tnn_set_output(last_op)
TinyNNCuda.tnn_realize(sess_op, last_op)

# ============================================================
# Zero inputs (timing launches+compute, not math). Both ops consume
# the same number of bytes per tensor, just with axes swapped — the
# total bytes uploaded per iter is identical between MUL_MAT and OUT_PROD.
# ============================================================
m_a_mm = Mat.new(K, M)   # ne=(K, M) → (rows=M, cols=K) → Mat (K, M) row-major
m_b_mm = Mat.new(K, N)
m_a_op = Mat.new(M, K)   # ne=(M, K) → Mat (M, K)
m_b_op = Mat.new(N, K)
[m_a_mm, m_b_mm, m_a_op, m_b_op].each do |m|
  n = m.nrows * m.ncols
  i = 0; while i < n; m.flat[i] = 0.0; i = i + 1; end
end

# ============================================================
# Time MUL_MAT
# ============================================================
i = 0
while i < WARMUP
  TinyNNCuda.upload_row_major(sess_mm, t_a_mm, m_a_mm)
  TinyNNCuda.upload_row_major(sess_mm, t_b_mm, m_b_mm)
  TinyNNCuda.tnn_compute(sess_mm)
  i = i + 1
end
t0 = Time.now
i = 0
while i < ITERS
  TinyNNCuda.upload_row_major(sess_mm, t_a_mm, m_a_mm)
  TinyNNCuda.upload_row_major(sess_mm, t_b_mm, m_b_mm)
  TinyNNCuda.tnn_compute(sess_mm)
  i = i + 1
end
t_mm = (Time.now - t0) * 1000.0
per_iter_mm = t_mm / ITERS.to_f
per_op_mm   = per_iter_mm / N_OPS.to_f * 1000.0   # µs

# ============================================================
# Time OUT_PROD
# ============================================================
i = 0
while i < WARMUP
  TinyNNCuda.upload_row_major(sess_op, t_a_op, m_a_op)
  TinyNNCuda.upload_row_major(sess_op, t_b_op, m_b_op)
  TinyNNCuda.tnn_compute(sess_op)
  i = i + 1
end
t0 = Time.now
i = 0
while i < ITERS
  TinyNNCuda.upload_row_major(sess_op, t_a_op, m_a_op)
  TinyNNCuda.upload_row_major(sess_op, t_b_op, m_b_op)
  TinyNNCuda.tnn_compute(sess_op)
  i = i + 1
end
t_op = (Time.now - t0) * 1000.0
per_iter_op = t_op / ITERS.to_f
per_op_op   = per_iter_op / N_OPS.to_f * 1000.0   # µs

puts "=== ab_smoke_out_prod_vs_matmul_cuda ==="
puts "shape (per op): a[K=" + K.to_s + ", M=" + M.to_s + "] × b[K=" + K.to_s + ", N=" + N.to_s + "] -> [M, N]"
puts "ops per compute: " + N_OPS.to_s + "  ITERS=" + ITERS.to_s + "  WARMUP=" + WARMUP.to_s
puts ""
puts "MUL_MAT:  total=" + t_mm.to_s + " ms   per-iter=" + per_iter_mm.to_s + " ms   per-op≈" + per_op_mm.to_s + " µs"
puts "OUT_PROD: total=" + t_op.to_s + " ms   per-iter=" + per_iter_op.to_s + " ms   per-op≈" + per_op_op.to_s + " µs"
puts ""
ratio = per_op_op / per_op_mm
puts "OUT_PROD / MUL_MAT per-op ratio: " + ratio.to_s + "x"
puts "  (>1 = OUT_PROD slower; <1 = OUT_PROD faster)"
