# Standalone smoke test of the FFI → ggml-CPU integration in this project.
# Reproduces the 4x3 matmul demo from ggml's examples/simple/simple-ctx.cpp.
#
# Expected output (4x3 result, row-major):
#   60  90  42
#   55  54  29
#   50  54  28
#  110 126  64
#
# Used to verify the Makefile + tinynn/ shim before touching lib/transformer.rb.

module TinyNN
  ffi_lib "tinynn_ggml"
  ffi_lib "ggml"
  ffi_lib "ggml-cpu"
  ffi_lib "ggml-base"
  ffi_lib "stdc++"
  ffi_lib "pthread"
  ffi_lib "m"

  # Library search paths injected by Makefile via SPINEL_FFI_CFLAGS.
  # Default fallback works for the in-tree build.
  ffi_cflags "-L. -Ltinynn -Lvendor/ggml/build/src"

  ffi_func :tnn_session_new,      [:int],                   :ptr
  ffi_func :tnn_session_free,     [:ptr],                   :void
  ffi_func :tnn_backend_name,     [:ptr],                   :str
  ffi_func :tnn_link_check,       [],                       :int

  ffi_func :tnn_input_2d_f32,     [:ptr, :int, :int],       :ptr
  ffi_func :tnn_matmul,           [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_matmul_axb,       [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_realize,          [:ptr, :ptr],             :int
  ffi_func :tnn_compute,          [:ptr],                   :int

  ffi_func :tnn_scratch_set,      [:ptr, :int, :double],    :void
  ffi_func :tnn_scratch_get,      [:ptr, :int],             :double
  ffi_func :tnn_upload,           [:ptr, :ptr],             :int
  ffi_func :tnn_download,         [:ptr, :ptr],             :int

  ffi_func :tnn_tensor_ne0,       [:ptr],                   :int
  ffi_func :tnn_tensor_ne1,       [:ptr],                   :int
end

def stage_and_upload(sess, tensor, flat)
  i = 0
  n = flat.length
  while i < n
    TinyNN.tnn_scratch_set(sess, i, flat[i])
    i = i + 1
  end
  TinyNN.tnn_upload(sess, tensor)
end

puts "link_check = " + TinyNN.tnn_link_check.to_s

sess = TinyNN.tnn_session_new(0)   # 0 = CPU
if sess == nil
  puts "session_new failed"
else
  puts "backend = " + TinyNN.tnn_backend_name(sess)
end

a = TinyNN.tnn_input_2d_f32(sess, 4, 2)
b = TinyNN.tnn_input_2d_f32(sess, 3, 2)
c = TinyNN.tnn_matmul(sess, a, b)

rc = TinyNN.tnn_realize(sess, c)
puts "realize rc = " + rc.to_s

stage_and_upload(sess, a, [2.0, 8.0,
                            5.0, 1.0,
                            4.0, 2.0,
                            8.0, 6.0])
stage_and_upload(sess, b, [10.0, 5.0,
                            9.0, 9.0,
                            5.0, 4.0])

rc = TinyNN.tnn_compute(sess)
puts "compute rc = " + rc.to_s

TinyNN.tnn_download(sess, c)

# ggml result has ne0=4 (m), ne1=3 (n). Reading the logical 4x3 row-major
# requires the j*m+i transpose.
m = TinyNN.tnn_tensor_ne0(c)
n = TinyNN.tnn_tensor_ne1(c)
puts "result: m=" + m.to_s + " n=" + n.to_s + " (logical 4x3)"
i = 0
while i < m
  s = "row " + i.to_s + ":"
  j = 0
  while j < n
    s = s + " " + TinyNN.tnn_scratch_get(sess, j * m + i).to_s
    j = j + 1
  end
  puts s
  i = i + 1
end

TinyNN.tnn_session_free(sess)
puts "done"
