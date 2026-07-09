# P5 smoke: Q8_0 KV cache quantization roundtrip.
#
# Allocates a Q8_0 persistent tensor and a same-shape F32 reference,
# writes the SAME f32 data via ggml_cpy (which quantizes on f32→Q8
# destination), then runs both through a matmul that reads them back
# (mul_mat dequantizes Q8 internally for the kernel). Measures the
# numerical error introduced by Q8 quantization.
#
# Layout: K-cache-like.
#   F32 K: ne=[d_head=64, max_T=64]
#   Q8 K:  same shape, type Q8_0
#
# Acceptance: max abs diff < 0.02 per element on a unit-scale gaussian
# input (Q8_0 has ~1/128 step at unit scale). RMS error < 0.005.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

GGML_TYPE_F32  = 0
GGML_TYPE_Q8_0 = 8

D_HEAD = 64        # Q8_0 block size = 32; 64 = 2 blocks → cleanly quantizable
MAX_T  = 64
TOTAL  = D_HEAD * MAX_T

sess = TinyNN.tnn_session_new(0)

# F32 reference and Q8 candidate, plus an f32 "source" buffer.
t_src_f32 = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T, D_HEAD)
t_dst_f32 = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T, D_HEAD)
t_dst_q8  = TinyNN.tnn_input_2d_persistent_typed(sess, MAX_T, D_HEAD, GGML_TYPE_Q8_0)
if t_dst_q8 == nil
  puts "FAIL: tnn_input_2d_persistent_typed returned nil for Q8_0"
  exit 1
end

# Copy F32 → F32 and F32 → Q8 in the same graph (cpy quantizes on
# type mismatch). Then read both back via a no-op identity matmul:
# matmul with a 1x1 "scalar = 1.0" tensor is the simplest way to
# materialize a Q8 source as an F32 output we can download.
#
# But mul_mat needs ne0 to match. Simpler: use a probe matmul that
# multiplies K^T by a known column to read it back without changes.
# For correctness, just download via tnn_cpy(Q8 → f32 scratch).
t_dst_f32_via_q8 = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T, D_HEAD)

t_cpy1 = TinyNN.tnn_cpy(sess, t_src_f32, t_dst_f32)   # f32 → f32 (identity)
t_cpy2 = TinyNN.tnn_cpy(sess, t_src_f32, t_dst_q8)    # f32 → Q8  (quantize)
t_cpy3 = TinyNN.tnn_cpy(sess, t_dst_q8, t_dst_f32_via_q8)  # Q8 → f32 (dequantize)

TinyNN.tnn_set_output(t_dst_f32)
TinyNN.tnn_set_output(t_dst_f32_via_q8)
TinyNN.tnn_finalize_weights(sess)
TinyNN.tnn_add_to_graph(sess, t_cpy1)
TinyNN.tnn_add_to_graph(sess, t_cpy2)
TinyNN.tnn_add_to_graph(sess, t_cpy3)
TinyNN.tnn_realize(sess, t_dst_f32_via_q8)

# Build a unit-scale "K-like" input. Use a deterministic LCG-ish
# pattern that yields bounded float values in roughly [-1, 1].
# Avoid Math.sin/Math.sqrt — those trigger Spinel polymorphic-dispatch
# widening that breaks unrelated code in transformer.rb. The pattern
# below is what we use elsewhere for deterministic smoke inputs.
src = [0.0]; src.pop
i = 0
while i < TOTAL
  # Varied K-like values in [-1, 1] via int-arithmetic only (no Math.*,
  # no float-comparison loops). The shifted xorshift gives ~uniform
  # coverage of the Q8 quantization grid.
  bit = (i * 2654435761) & 0xFFFF
  v   = bit.to_f / 32768.0 - 1.0
  src.push(v * 0.5)
  i = i + 1
end
TinyNN.tnn_upload_from_float_array(sess, t_src_f32, src, TOTAL)

TinyNN.tnn_compute(sess)

# Download both. Compare element-wise.
TinyNN.tnn_download(sess, t_dst_f32)
ref = [0.0]; ref.pop
i = 0
while i < TOTAL; ref.push(TinyNN.tnn_scratch_get(sess, i)); i = i + 1; end

TinyNN.tnn_download(sess, t_dst_f32_via_q8)
got = [0.0]; got.pop
i = 0
while i < TOTAL; got.push(TinyNN.tnn_scratch_get(sess, i)); i = i + 1; end

max_abs  = 0.0
sum_sq   = 0.0
i = 0
while i < TOTAL
  d  = ref[i] - got[i]
  ad = d < 0 ? 0.0 - d : d
  if ad > max_abs; max_abs = ad; end
  sum_sq = sum_sq + d * d
  i = i + 1
end
# NOTE: name collision with `rms` in transformer.rb's rms_norm_backward
# triggers a Spinel whole-program-inference codegen bug where the
# unrelated rms_norm_backward arm gets polymorphic widening and fails
# to compile. Use `rms_err` to keep the two locals distinct globally.
mean_sq = sum_sq / TOTAL.to_f
# Newton sqrt with initial guess that converges across a wide range.
# 30 iterations is overkill but cheap on 1 scalar.
rms_err = mean_sq > 1.0 ? mean_sq : 1.0
nri     = 0
while nri < 30
  rms_err = 0.5 * (rms_err + mean_sq / rms_err)
  nri = nri + 1
end

puts "Q8 roundtrip on " + TOTAL.to_s + " unit-scale elements"
puts "  max |Δ| = " + max_abs.to_s
puts "  rms |Δ| = " + rms_err.to_s
puts "  sample ref[0..3] = " + ref[0].to_s + ", " + ref[1].to_s + ", " + ref[2].to_s + ", " + ref[3].to_s
puts "  sample got[0..3] = " + got[0].to_s + ", " + got[1].to_s + ", " + got[2].to_s + ", " + got[3].to_s

if max_abs < 0.02 && rms_err < 0.005
  puts "OK: Q8_0 quantization within expected tolerance"
else
  puts "FAIL: Q8_0 quantization error exceeds tolerance"
  exit 1
end

TinyNN.tnn_session_free(sess)
