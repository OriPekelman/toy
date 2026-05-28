# Probe: does tnn_cast work end-to-end?
# Test plan:
#   1. Allocate an F32 input tensor with known values.
#   2. Cast to BF16 (dtype=30).
#   3. Cast back to F32.
#   4. Verify the round-trip preserves values (bf16 has ~3-decimal
#      precision so allow a small epsilon).
#
# If this passes, the ggml_cast op + tnn_cast FFI are wired
# correctly and ready for GH#9's master-copy mixed-precision pattern.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

# Pin Mat type so sp_Mat resolves.
m = Mat.new(2, 2); m.flat[0] = 1.0

sess = TinyNN.tnn_session_new(0)

# 4-element F32 input tensor.
t_in = TinyNN.tnn_input_2d_f32_persistent(sess, 2, 2)
TinyNN.tnn_finalize_weights(sess)

# Cast F32 -> BF16 -> F32 round trip.
t_bf  = TinyNN.tnn_cast(sess, t_in, 30)   # GGML_TYPE_BF16 = 30
t_out = TinyNN.tnn_cast(sess, t_bf, 0)    # back to F32
TinyNN.tnn_set_output(t_out)
TinyNN.tnn_realize(sess, t_out)

# Upload known values.
m_in = Mat.new(2, 2)
m_in.flat[0] = 1.5
m_in.flat[1] = -2.25
m_in.flat[2] = 0.125
m_in.flat[3] = 100.0
TinyNN.upload_row_major(sess, t_in, m_in)
TinyNN.tnn_compute(sess)

m_out = TinyNN.download_row_major(sess, t_out, 2, 2)
puts "round-trip F32 -> BF16 -> F32:"
puts "  in[0]=" + m_in.flat[0].to_s + " out[0]=" + m_out.flat[0].to_s
puts "  in[1]=" + m_in.flat[1].to_s + " out[1]=" + m_out.flat[1].to_s
puts "  in[2]=" + m_in.flat[2].to_s + " out[2]=" + m_out.flat[2].to_s
puts "  in[3]=" + m_in.flat[3].to_s + " out[3]=" + m_out.flat[3].to_s

# BF16 has ~3-4 sig figs. Allow 1% relative error.
max_rel_err = 0.0
i = 0
while i < 4
  d = (m_out.flat[i] - m_in.flat[i]).abs
  ref = m_in.flat[i].abs
  if ref > 0.0
    rel = d / ref
    if rel > max_rel_err; max_rel_err = rel; end
  end
  i = i + 1
end
puts "max rel err = " + max_rel_err.to_s

if max_rel_err < 0.01
  puts "PROBE-PASS: cast round-trip within BF16 precision"
else
  puts "FAIL: cast round-trip error > 1%"
end
