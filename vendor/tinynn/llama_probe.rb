require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

class LlamaProbe
  def self.probe(path)
    h = TinyNN.tnn_gguf_load(path)
    if h == nil
      puts "open failed"
      return
    end
    keys = ["llama.rope.dimension_count",
            "llama.rope.scaling.type",
            "llama.rope.scaling.factor",
            "llama.rope.scaling.original_context_length",
            "llama.attention.key_length",
            "llama.attention.value_length",
            "llama.context_length",
            "llama.expert_count",
            "general.architecture"]
    i = 0
    while i < keys.length
      k = keys[i]
      u = TinyNN.tnn_gguf_get_u32(h, k)
      f = TinyNN.tnn_gguf_get_f32(h, k)
      s = TinyNN.tnn_gguf_get_str(h, k)
      line = "  " + k + ":"
      if u >= 0
        line = line + " u32=" + u.to_s
      end
      if f > 0.0
        line = line + " f32=" + f.to_s
      end
      if s != nil
        line = line + " str=" + s
      end
      puts line
      i = i + 1
    end
    TinyNN.tnn_gguf_free(h)
  end
end

LlamaProbe.probe("data/llama-3.2-1b-f32.gguf")
