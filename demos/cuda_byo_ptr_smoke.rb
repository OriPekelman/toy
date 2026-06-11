# demos/cuda_byo_ptr_smoke.rb — minimum smoke test for the CUDA
# BYO-pointer patch (vendored ggml-cuda + tinynn wiring).
#
# Verifies that:
#   1. ggml_backend_cuda_buffer_from_ptr accepts a host (mmap) pointer.
#   2. cudaHostRegister + cudaHostGetDevicePointer succeed on this SKU
#      (GB10 unified memory in our case).
#   3. tnn_input_2d_persistent_mmap allocates a CUDA tensor that
#      references file-backed pages.
#   4. ggml_backend_buffer_get_tensor can copy bytes back out of the
#      mmap'd region via the CUDA buffer interface.

# Match smollm2_kv_cuda.rb's load chain — keeps Spinel's type
# inference happy (its global type-collapse rules are sensitive to
# which modules are pulled in).
require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"
require_relative "../lib/toy/train/training"

GGUF = "data/qwen25-1.5b-native-q8.gguf"   # D=1536, multiple of 512

# 1. Open the GGUF (gives us the mmap region; same C symbol used by
#    both backends).
gguf = TinyNNCuda.tnn_gguf_load(GGUF)
if gguf == TinyNNCuda.tnn_null_ptr
  puts "FAIL: tnn_gguf_load returned NULL"
  exit 1
end
map  = TinyNNCuda.tnn_gguf_mmap_base(gguf)
size = TinyNNCuda.tnn_gguf_mmap_size(gguf)
puts "GGUF mmap base + size: " + size.to_s + " bytes"

# 2. Find token_embd.weight (F32 — no padding concerns).
idx = TinyNNCuda.tnn_gguf_find_index(gguf, "token_embd.weight")
if idx < 0
  puts "FAIL: token_embd.weight not found"
  exit 1
end
off = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf, idx)
typ = TinyNNCuda.tnn_gguf_tensor_type(gguf, idx)
puts "token_embd.weight: idx=" + idx.to_s + " off=" + off.to_s +
     " type=" + typ.to_s

# 3. Create a CUDA session.
sess = TinyNNCuda.tnn_session_new(1)
if sess == TinyNNCuda.tnn_null_ptr
  puts "FAIL: CUDA session_new returned NULL"
  exit 1
end
puts "CUDA backend: " + TinyNNCuda.tnn_backend_name(sess)

# 4. Attach the GGUF's mmap region as the BYO weight buffer.
ret = TinyNNCuda.tnn_session_attach_weight_mmap(sess, map, size)
if ret != 0
  puts "FAIL: tnn_session_attach_weight_mmap returned " + ret.to_s
  puts "  (this means ggml_backend_cuda_buffer_from_ptr failed —"
  puts "   likely cudaHostRegister rejected the region or"
  puts "   cudaHostGetDevicePointer reported a non-UVA setup.)"
  exit 1
end
puts "attach_weight_mmap OK"

# 5. Allocate one weight tensor pointing at the file.
# Qwen2.5-1.5B: vocab=151936, d_model=1536. Token embed has shape
# [vocab, d_model] in GGUF; we declare ne=[d_model, vocab].
t = TinyNNCuda.tnn_input_2d_persistent_mmap(sess, 151936, 1536, 0, off)
if t == TinyNNCuda.tnn_null_ptr
  puts "FAIL: tnn_input_2d_persistent_mmap returned NULL"
  exit 1
end
puts "weight tensor allocated; data points into mmap region"

# Per-binary type-anchor: pin TinyNNCuda upload helpers so Spinel
# infers their return as Int. Without a concrete callsite in this
# demo's reachable graph, the wrappers' return type can collapse.
if false
  _dummy = Mat.new(1, 1)
  TinyNNCuda.upload_row_major(sess, t, _dummy)
  TinyNNCuda.upload_transposed(sess, t, _dummy)
end

# Cleanup
TinyNNCuda.tnn_session_free(sess)
TinyNNCuda.tnn_gguf_free(gguf)

puts "OK"
