# A/B smoke: N_HEADS small matmuls vs 1 batched matmul, at LoRA-A shape.
#
# Purpose: isolate the "fewer kernel launches" mechanic from all toy-side
# plumbing. We're answering ONE question: at the heavy LoRA shape, does
# replacing N_HEADS independent matmuls (x @ A_h, one per head) with a
# single batched matmul (x @ A_fused with n_heads on ne2) reduce
# wallclock? The full HeadFuseLoRAQ refactor adds a second matmul for B,
# a sum-reduce across heads, etc. — those have identical kernel-launch
# profiles, so this isolated test answers the call-count question.
#
# Both variants do the same total compute (N_HEADS x [D_MODEL, R] worth
# of FLOPs). Only the launch pattern differs.
#
# ggml batched-matmul broadcast quirk: ggml_can_mul_mat requires
# `b.ne[2] % a.ne[2] == 0`, i.e. a's batch is a divisor of b's. For a
# per-head weight with batch=N_HEADS broadcasting across a shared x
# with batch=1, we must put x in the a-arg position and the weights in
# the b-arg position. Result ends up shape [a.ne1, b.ne1, b.ne2].
#
# ggml 2D layout note: tnn_input_2d_f32_persistent(sess, X, Y) creates a
# tensor with ne0=Y, ne1=X (args are "math (rows, cols)"; ggml stores
# them as column-major).

require_relative "../lib/transformer"
require_relative "../lib/tinynn_cuda"

# Heavy bench shape: Qwen2.5-1.5B at seq=256, r=8 (defaults).
# Override via env to sweep launch-overhead vs compute-bound regimes.
D_MODEL  = (ENV["D_MODEL"]  || "1536").to_i
N_HEADS  = (ENV["N_HEADS"]  || "12").to_i
D_HEAD   = (ENV["D_HEAD"]   || "128").to_i
R        = (ENV["R"]        || "8").to_i
T        = (ENV["T"]        || "256").to_i
N_LAYERS = (ENV["N_LAYERS"] || "28").to_i   # used to scale per-iter to per-step
ITERS    = (ENV["ITERS"]    || "100").to_i
WARMUP   = (ENV["WARMUP"]   || "10").to_i

# ============================================================
# Per-head graph: N_HEADS independent (x @ A_h) matmuls.
# ============================================================
sess_ph = TinyNNCuda.tnn_session_new(1)
# x: math [T, D_MODEL] → ggml ne0=D_MODEL, ne1=T. Call args = (rows, cols).
t_x_ph  = TinyNNCuda.tnn_input_2d_f32_persistent(sess_ph, T, D_MODEL)

# Sum the per-head outputs so the graph has a single sink.
heads_out = TinyNNCuda.tnn_null_ptr
hi = 0
while hi < N_HEADS
  # A_h: math [D_MODEL, R] → need ggml ne0=D_MODEL (K), ne1=R (M).
  # Call (R, D_MODEL) → ne0=D_MODEL, ne1=R.  ✓
  t_a = TinyNNCuda.tnn_input_2d_f32_persistent(sess_ph, R, D_MODEL)
  # mul_mat(a, b): a[K, M] × b[K, N] → [M, N].
  #   A_h (ne0=D_MODEL=K, ne1=R=M) × x (ne0=D_MODEL=K, ne1=T=N)
  #   → [R, T]
  t_xa = TinyNNCuda.tnn_matmul(sess_ph, t_a, t_x_ph)
  if hi == 0
    heads_out = t_xa
  else
    heads_out = TinyNNCuda.tnn_add(sess_ph, heads_out, t_xa)
  end
  hi = hi + 1
end
TinyNNCuda.tnn_set_output(heads_out)
TinyNNCuda.tnn_realize(sess_ph, heads_out)

# ============================================================
# Fused graph: 1 batched matmul (n_heads on ne2). x in a-arg position
# so its ne2=1 broadcasts up to the weights' ne2=N_HEADS — see header
# note on ggml_can_mul_mat's broadcast direction.
# ============================================================
sess_fu = TinyNNCuda.tnn_session_new(1)
# x: ne0=D_MODEL, ne1=T, ne2=1 — same shape as per-head x.
t_x_fu  = TinyNNCuda.tnn_input_2d_f32_persistent(sess_fu, T, D_MODEL)
# A_fused: ne0=D_MODEL=K, ne1=R=N, ne2=N_HEADS=batch.
t_a_fu = TinyNNCuda.tnn_input_3d_f32_persistent(sess_fu, D_MODEL, R, N_HEADS)

# mul_mat(x, A_fused): a=x [K=D_MODEL, M=T, batch=1],
#                     b=A [K=D_MODEL, N=R, batch=N_HEADS]
#                     → result [M=T, N=R, batch=N_HEADS]
# Broadcast: b.ne2=N_HEADS, a.ne2=1, N_HEADS % 1 = 0 ✓.
t_xa_fu = TinyNNCuda.tnn_matmul(sess_fu, t_x_fu, t_a_fu)        # [T, R, N_HEADS]
TinyNNCuda.tnn_set_output(t_xa_fu)
TinyNNCuda.tnn_realize(sess_fu, t_xa_fu)

# ============================================================
# Zero inputs (we measure launches, not math)
# ============================================================
m_x_ph = Mat.new(T, D_MODEL)
m_x_fu = Mat.new(T, D_MODEL)
i = 0
n = T * D_MODEL
while i < n; m_x_ph.flat[i] = 0.0; m_x_fu.flat[i] = 0.0; i = i + 1; end

# ============================================================
# Time per-head
# ============================================================
i = 0
while i < WARMUP
  TinyNNCuda.upload_row_major(sess_ph, t_x_ph, m_x_ph)
  TinyNNCuda.tnn_compute(sess_ph)
  i = i + 1
end
t0 = Time.now
i = 0
while i < ITERS
  TinyNNCuda.upload_row_major(sess_ph, t_x_ph, m_x_ph)
  TinyNNCuda.tnn_compute(sess_ph)
  i = i + 1
end
t_ph_ms = (Time.now - t0) * 1000.0
per_iter_ph = t_ph_ms / ITERS.to_f

# ============================================================
# Time fused
# ============================================================
i = 0
while i < WARMUP
  TinyNNCuda.upload_row_major(sess_fu, t_x_fu, m_x_fu)
  TinyNNCuda.tnn_compute(sess_fu)
  i = i + 1
end
t0 = Time.now
i = 0
while i < ITERS
  TinyNNCuda.upload_row_major(sess_fu, t_x_fu, m_x_fu)
  TinyNNCuda.tnn_compute(sess_fu)
  i = i + 1
end
t_fu_ms = (Time.now - t0) * 1000.0
per_iter_fu = t_fu_ms / ITERS.to_f

ph_ops = N_HEADS + (N_HEADS - 1)   # 12 matmul + 11 add
fu_ops = 1                          # 1 batched matmul

puts "=== ab_smoke_lora_fused_cuda ==="
puts "shape: D_MODEL=" + D_MODEL.to_s + " N_HEADS=" + N_HEADS.to_s +
     " R=" + R.to_s + " T=" + T.to_s + "  (one matmul level; ITERS=" + ITERS.to_s + ")"
puts ""
puts "per-head: " + ph_ops.to_s + " ops/compute (" + N_HEADS.to_s + " matmul + " + (N_HEADS - 1).to_s + " add)"
puts "  total ms = " + t_ph_ms.to_s
puts "  per iter = " + per_iter_ph.to_s + " ms"
puts ""
puts "fused:    " + fu_ops.to_s + " op/compute (1 batched matmul, n_heads on ne2)"
puts "  total ms = " + t_fu_ms.to_s
puts "  per iter = " + per_iter_fu.to_s + " ms"
puts ""
speedup = per_iter_ph / per_iter_fu
saved   = per_iter_ph - per_iter_fu
puts "per-matmul-level speedup: " + speedup.to_s + "x"
puts "per-matmul-level saved:   " + saved.to_s + " ms"
puts ""
puts "scaling to LoRA-Q on heavy bench (A and B levels, fwd + bwd):"
puts "  delta ≈ saved × 2 levels × " + N_LAYERS.to_s + " layers × 2 (fwd+bwd) = " +
     (saved * 2.0 * N_LAYERS.to_f * 2.0).to_s + " ms / step"
