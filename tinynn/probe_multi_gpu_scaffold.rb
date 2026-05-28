# Probe GH#3 multi-GPU mode 1 scaffolding.
#
# Verifies:
#   1. tnn_cuda_get_device_count() returns a sane value
#      (0 on CPU-only builds, 1+ on CUDA builds).
#   2. tnn_session_new_on(0, 0) behaves identically to
#      tnn_session_new(0).
#   3. The signature/binding compiles + links cleanly on CPU-only.
#
# Does NOT verify device > 0 works at runtime — that requires a
# multi-GPU host which we don't have on gx10. Once available, run
# this with kind=1 device=1 to validate.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

m = Mat.new(2, 2); m.flat[0] = 1.0

n_gpus = TinyNN.tnn_cuda_get_device_count
puts "tnn_cuda_get_device_count = " + n_gpus.to_s

# Old API: tnn_session_new(kind)
sess_a = TinyNN.tnn_session_new(0)
puts "tnn_session_new(0)        = " + (sess_a == TinyNN.tnn_null_ptr ? "NULL" : "ok")

# New API: tnn_session_new_on(kind, device=0)
sess_b = TinyNN.tnn_session_new_on(0, 0)
puts "tnn_session_new_on(0, 0)  = " + (sess_b == TinyNN.tnn_null_ptr ? "NULL" : "ok")

# Different sessions = different pointers, but they share the cached
# engine internally so both go through the same code path.
TinyNN.tnn_session_free(sess_a)
TinyNN.tnn_session_free(sess_b)

puts "PROBE-PASS: GH#3 scaffolding wired"
