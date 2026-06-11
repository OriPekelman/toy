require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

class NativeProbe
  def self.check(path)
    h = TinyNN.tnn_gguf_load(path)
    if h == nil
      puts "FAIL " + path
      return
    end
    is_native = TinyNN.tnn_gguf_get_bool(h, "toy.ggml_native")
    puts (is_native == 1 ? "native " : "legacy ") + path
    TinyNN.tnn_gguf_free(h)
  end
end

NativeProbe.check("data/qwen25-1.5b-native.gguf")
NativeProbe.check("data/tinyllama-1.1b-f32.gguf")
NativeProbe.check("data/llama-3.2-1b-f32.gguf")
NativeProbe.check("data/smollm2-135m-f32.gguf")
