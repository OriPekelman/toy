# Smoke for rope_ext_back. ggml provides the math; numerical-gradient
# parity here is overkill (would require fixturing positions through
# the project's scratch + i32 plumbing). We verify:
#   - the FFI binding compiles
#   - tnn_rope_ext_back returns a non-null handle for sane inputs
#   - the realize+compute cycle completes without aborting
#
# Real parity validation lands when rope_back is wired into a forward
# graph via ggml_build_backward_expand (Phase F0.4).

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

D_HEAD = 4
ROPE_BASE = 10000.0

# dy: 1 × d_head Mat with arbitrary values
dyt = Mat.new(1, D_HEAD)
dyt.flat[0] = 1.0
dyt.flat[1] = -0.5
dyt.flat[2] = 0.7
dyt.flat[3] = 0.3

sess = TinyNN.tnn_session_new(0)
tdy  = TinyNN.tnn_input_2d_f32(sess, dyt.nrows, dyt.ncols)
tpos = TinyNN.tnn_input_1d_i32_ctx(sess, 1)
tdx  = TinyNN.tnn_rope_ext_back(sess, tdy, tpos, D_HEAD,
                                ROPE_BASE, 1.0, 0.0, 1.0, 32.0, 1.0,
                                TinyNN.tnn_null_ptr)

if tdx == nil
  puts "rope_back: FAIL — tnn_rope_ext_back returned NULL"
  exit 1
end

TinyNN.tnn_realize(sess, tdx)
TinyNN.stage_row_major_and_upload(sess, tdy, dyt)
TinyNN.tnn_scratch_set_i32(sess, 0, 0)  # pos=0; arbitrary
TinyNN.tnn_compute(sess)
TinyNN.tnn_download(sess, tdx)

out = Mat.new(dyt.nrows, dyt.ncols)
i = 0
finite = 0
while i < D_HEAD
  v = TinyNN.tnn_scratch_get(sess, i)
  out.flat[i] = v
  if v == v
    finite = finite + 1
  end
  i = i + 1
end
TinyNN.tnn_session_free(sess)

puts "rope_back smoke: completed without abort"
puts "  output: " + out.flat.inspect
puts "  finite elements: " + finite.to_s + "/" + D_HEAD.to_s
if finite == D_HEAD
  puts "rope_back: match=true (smoke only — no parity check)"
else
  puts "rope_back: match=false (NaN in output)"
end
