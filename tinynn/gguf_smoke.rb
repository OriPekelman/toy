# GGUF loader smoke. Without a path argument: opens an empty GGUF
# context and confirms n_tensors=0 (validates the binding).
# With a path: loads the file, lists tensor names + shapes.
#
# To test with a real model:
#   wget -O data/tinystories.gguf \
#     https://huggingface.co/ggml-org/models/resolve/main/...
#   ./tinynn/gguf_smoke data/tinystories.gguf

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

def ggml_type_name(t)
  if t == 0
    return "f32"
  end
  if t == 1
    return "f16"
  end
  if t == 24
    return "i32"
  end
  if t == 30
    return "bf16"
  end
  return "type=" + t.to_s
end

# Default to empty GGUF unless the binary was built/invoked with a path.
# Spinel doesn't expose ARGV reliably, so we just always test the empty path
# and additionally try a hardcoded data/tinystories.gguf if present.

handle = TinyNN.tnn_gguf_load_empty
if handle == nil
  puts "tnn_gguf_load_empty failed"
else
  n = TinyNN.tnn_gguf_n_tensors(handle)
  puts "empty GGUF: n_tensors=" + n.to_s + " (expected 0)"
  TinyNN.tnn_gguf_free(handle)
end

# Write a tiny demo GGUF then load it back. This exercises the full
# round-trip without needing an externally-downloaded model file.
demo_path = "/tmp/tinynn_demo.gguf"
wrc = TinyNN.tnn_gguf_write_demo_file(demo_path)
puts ""
puts "wrote demo GGUF to " + demo_path + " rc=" + wrc.to_s

handle = TinyNN.tnn_gguf_load(demo_path)
if handle == nil
  puts "tnn_gguf_load(" + demo_path + ") failed"
else
  n = TinyNN.tnn_gguf_n_tensors(handle)
  puts "loaded demo: n_tensors=" + n.to_s
  i = 0
  while i < n
    name = TinyNN.tnn_gguf_tensor_name(handle, i)
    ne0  = TinyNN.tnn_gguf_tensor_ne(handle, i, 0)
    ne1  = TinyNN.tnn_gguf_tensor_ne(handle, i, 1)
    typ  = TinyNN.tnn_gguf_tensor_type(handle, i)
    nb   = TinyNN.tnn_gguf_tensor_nbytes(handle, i)
    puts "  [" + i.to_s + "] " + name + "  shape ne0=" + ne0.to_s + " ne1=" + ne1.to_s + "  " + ggml_type_name(typ) + "  " + nb.to_s + " bytes"

    # Read it back into a Mat-style buffer.
    nel = ne0 * ne1
    buf = Array.new(nel, 0.0)
    rrc = TinyNN.tnn_gguf_read_f32_to_doubles(handle, i, buf, nel)
    puts "  read rc=" + rrc.to_s + " values=" + buf[0].to_s + "," + buf[1].to_s + "," + buf[2].to_s + "," + buf[3].to_s + "," + buf[4].to_s + "," + buf[5].to_s
    puts "  expected: 1.0,2.0,3.0,4.0,5.0,6.0"
    i = i + 1
  end
  TinyNN.tnn_gguf_free(handle)
end

# Optional: try a real file the user may have dropped.
real_path = "data/tinystories.gguf"
real_handle = TinyNN.tnn_gguf_load(real_path)
if real_handle != nil
  puts ""
  puts "also loaded " + real_path + ": n_tensors=" + TinyNN.tnn_gguf_n_tensors(real_handle).to_s
  TinyNN.tnn_gguf_free(real_handle)
end
