# GGUF inspect tool. Walks every tensor in a GGUF via the project's
# tnn_gguf_* FFI and prints name/shape/dtype/first-few-values. Used to
# confirm end-to-end read of large HF-converted files (distilgpt2 etc).
#
# Spinel doesn't expose ARGV reliably, so the path is hardcoded; flip
# GGUF_PATH below to point at whichever file you want to inspect.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

GGUF_PATH  = "data/distilgpt2-f32.gguf"
SHOW_VALS  = 4   # first N float values to print per tensor

def type_name(t)
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

handle = TinyNN.tnn_gguf_load(GGUF_PATH)
if handle == nil
  puts "FAILED to open " + GGUF_PATH
else
  n = TinyNN.tnn_gguf_n_tensors(handle)
  puts "opened " + GGUF_PATH + " (" + n.to_s + " tensors)"
  puts ""

  total_bytes = 0
  i = 0
  while i < n
    name = TinyNN.tnn_gguf_tensor_name(handle, i)
    ne0  = TinyNN.tnn_gguf_tensor_ne(handle, i, 0)
    ne1  = TinyNN.tnn_gguf_tensor_ne(handle, i, 1)
    ne2  = TinyNN.tnn_gguf_tensor_ne(handle, i, 2)
    ne3  = TinyNN.tnn_gguf_tensor_ne(handle, i, 3)
    typ  = TinyNN.tnn_gguf_tensor_type(handle, i)
    nb   = TinyNN.tnn_gguf_tensor_nbytes(handle, i)
    quant = TinyNN.tnn_gguf_tensor_is_quantized(handle, i)

    total_bytes = total_bytes + nb

    # 1D printed as just ne0; 2D as ne0xne1; higher dims included if non-1.
    shape_str = ne0.to_s
    if ne1 > 1 || ne2 > 1 || ne3 > 1
      shape_str = shape_str + "x" + ne1.to_s
    end
    if ne2 > 1 || ne3 > 1
      shape_str = shape_str + "x" + ne2.to_s
    end
    if ne3 > 1
      shape_str = shape_str + "x" + ne3.to_s
    end

    suffix = ""
    if quant != 0
      suffix = " quant"
    end

    line = "  [" + i.to_s + "] " + name +
           "  " + type_name(typ) + suffix +
           "  shape=" + shape_str +
           "  " + nb.to_s + "B"
    puts line

    # Read first SHOW_VALS f32 values. The C side caps n at the tensor's
    # element count, so we only need to allocate SHOW_VALS doubles even
    # for very large tensors (the embed matrix is 38M elements).
    # Quantized tensors do a full dequant inside the C side regardless
    # of n, but that's the C's malloc; doesn't reach the Ruby Array.
    if SHOW_VALS > 0
      nel = ne0 * ne1 * ne2 * ne3
      n_show = SHOW_VALS
      if n_show > nel
        n_show = nel
      end
      buf = Array.new(n_show, 0.0)
      rc = TinyNN.tnn_gguf_read_f32_to_doubles(handle, i, buf, n_show)
      if rc == 0
        out = "      first " + n_show.to_s + ": "
        k = 0
        while k < n_show
          out = out + buf[k].to_s
          if k < n_show - 1
            out = out + ", "
          end
          k = k + 1
        end
        puts out
      else
        puts "      read failed rc=" + rc.to_s
      end
    end

    i = i + 1
  end

  puts ""
  puts "total tensor bytes: " + total_bytes.to_s
  TinyNN.tnn_gguf_free(handle)
end
