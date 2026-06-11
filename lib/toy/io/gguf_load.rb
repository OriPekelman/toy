# Populate a GPT2LM's weights from a project-converted GGUF file (see
# prep/convert_distilgpt2_to_gguf.py for the producer side).
#
# Caller constructs the GPT2LM with the right hyperparams (we don't read
# them out of the GGUF metadata yet — tnn_gguf_kv_get_* is a TODO; for
# distilgpt2 they're known: vocab=50257, d=768, ff=3072, heads=12, ...).
#
# Reads go straight into the destination Mat.flat / Array<Float> via
# the existing tnn_gguf_read_f32_to_doubles FFI — no intermediate
# buffers, no copies after the FFI write. token_embd alone is 38.6 M
# elements (~300 MB f64); a copy step would be wasteful.

require_relative "../models/transformer"
require_relative "../models/gpt2"
require_relative "../ffi/tinynn"

# GPT-2 hyperparameters read from the GGUF's kv metadata. Same shape
# (50257 vocab, 1024 ctx) across all variants; only d_model / d_ff /
# n_heads / n_layers change. The converter writes these keys directly
# via gguf.GGUFWriter (see prep/convert_distilgpt2_to_gguf.py).
class GPT2Config
  attr_accessor :vocab_size, :d_model, :d_ff, :n_heads, :n_layers,
                :context_length, :ln_eps

  def initialize(vocab_size, d_model, d_ff, n_heads, n_layers,
                 context_length, ln_eps)
    @vocab_size     = vocab_size
    @d_model        = d_model
    @d_ff           = d_ff
    @n_heads        = n_heads
    @n_layers       = n_layers
    @context_length = context_length
    @ln_eps         = ln_eps
  end
end

module GPT2ConfigLoader
  # Read all hyperparams from a GGUF file's kv metadata. Returns a
  # populated GPT2Config; the caller passes it to GPT2LM.new and the
  # FFI cache realize_for methods.
  def self.read(path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "GPT2ConfigLoader: failed to open " + path
      return GPT2Config.new(0, 0, 0, 0, 0, 0, 1.0e-5)
    end
    vocab   = TinyNN.tnn_gguf_get_u32(handle, "gpt2.vocab_size")
    d_model = TinyNN.tnn_gguf_get_u32(handle, "gpt2.embedding_length")
    d_ff    = TinyNN.tnn_gguf_get_u32(handle, "gpt2.feed_forward_length")
    n_head  = TinyNN.tnn_gguf_get_u32(handle, "gpt2.attention.head_count")
    n_layer = TinyNN.tnn_gguf_get_u32(handle, "gpt2.block_count")
    ctx     = TinyNN.tnn_gguf_get_u32(handle, "gpt2.context_length")
    eps     = TinyNN.tnn_gguf_get_f32(handle, "gpt2.attention.layer_norm_epsilon")
    TinyNN.tnn_gguf_free(handle)
    GPT2Config.new(vocab, d_model, d_ff, n_head, n_layer, ctx, eps)
  end
end

module GGUFLoad
  # Linear-scan tensor lookup. 100 tensors × ~50 reads = 5000 string
  # compares — fine. A hash map would force Spinel into a polymorphic
  # value type; not worth it.
  def self.find_index(handle, name, n_tensors)
    i = 0
    while i < n_tensors
      if TinyNN.tnn_gguf_tensor_name(handle, i) == name
        return i
      end
      i = i + 1
    end
    -1
  end

  # Read a 1-D tensor straight into an existing Array<Float>.
  def self.read_array(handle, name, target, n_tensors)
    idx = find_index(handle, name, n_tensors)
    if idx < 0
      puts "missing: " + name
      return
    end
    nel = target.length
    rc = TinyNN.tnn_gguf_read_f32_to_doubles(handle, idx, target, nel)
    if rc != 0
      puts "read failed: " + name + " rc=" + rc.to_s
    end
  end

  # Read a 2-D tensor straight into an existing Mat (writes to mat.flat).
  def self.read_mat(handle, name, mat, n_tensors)
    idx = find_index(handle, name, n_tensors)
    if idx < 0
      puts "missing: " + name
      return
    end
    nel = mat.nrows * mat.ncols
    rc = TinyNN.tnn_gguf_read_f32_to_doubles(handle, idx, mat.flat, nel)
    if rc != 0
      puts "read failed: " + name + " rc=" + rc.to_s
    end
  end

  # Read a [d_model, d_model] concatenated-heads weight tensor into
  # an Array<Mat> of n_heads × (d_model, d_head). Column block
  # [h*d_head : (h+1)*d_head] of the source becomes head h's matrix.
  def self.read_split_heads_weight(handle, name, dst, n_heads, d_model, d_head, n_tensors)
    idx = find_index(handle, name, n_tensors)
    if idx < 0
      puts "missing: " + name
      return
    end
    nel = d_model * d_model
    # Stage via a temporary flat buffer (~2.4 MB for distilgpt2);
    # the strided per-head copy can't run while ggml writes to dst.
    tmp = Array.new(nel, 0.0)
    rc  = TinyNN.tnn_gguf_read_f32_to_doubles(handle, idx, tmp, nel)
    if rc != 0
      puts "read failed: " + name + " rc=" + rc.to_s
      return
    end
    h = 0
    while h < n_heads
      mat = dst[h]
      i = 0
      while i < d_model
        j = 0
        while j < d_head
          mat.flat[i * d_head + j] = tmp[i * d_model + h * d_head + j]
          j = j + 1
        end
        i = i + 1
      end
      h = h + 1
    end
  end

  # Read a [d_model] concatenated-heads bias into n_heads × Array<Float>(d_head).
  def self.read_split_heads_bias(handle, name, dst, n_heads, d_head, n_tensors)
    idx = find_index(handle, name, n_tensors)
    if idx < 0
      puts "missing: " + name
      return
    end
    d_model = n_heads * d_head
    tmp = Array.new(d_model, 0.0)
    rc  = TinyNN.tnn_gguf_read_f32_to_doubles(handle, idx, tmp, d_model)
    if rc != 0
      puts "read failed: " + name + " rc=" + rc.to_s
      return
    end
    h = 0
    while h < n_heads
      arr = dst[h]
      j = 0
      while j < d_head
        arr[j] = tmp[h * d_head + j]
        j = j + 1
      end
      h = h + 1
    end
  end

  # GQA variant of read_split_heads_bias for K/V: the source is a
  # 1-D bias of length n_kv * d_head, split into n_kv arrays of d_head.
  # Used for Qwen2.x attn_k.bias / attn_v.bias.
  def self.read_split_kv_bias(handle, name, dst, n_kv, d_head, n_tensors)
    idx = find_index(handle, name, n_tensors)
    if idx < 0
      puts "missing: " + name
      return
    end
    nel = n_kv * d_head
    tmp = Array.new(nel, 0.0)
    rc  = TinyNN.tnn_gguf_read_f32_to_doubles(handle, idx, tmp, nel)
    if rc != 0
      puts "read failed: " + name + " rc=" + rc.to_s
      return
    end
    h = 0
    while h < n_kv
      arr = dst[h]
      j = 0
      while j < d_head
        arr[j] = tmp[h * d_head + j]
        j = j + 1
      end
      h = h + 1
    end
  end

  # GQA variant of read_split_heads_weight: the source tensor is
  # [d_model, n_kv * d_head] (not square), and we want to split it into
  # n_kv per-head matrices of shape (d_model, d_head). Mirrors the
  # logic of read_split_heads_weight but with the narrower output dim.
  def self.read_split_kv_weight(handle, name, dst, n_kv, d_model, d_head, n_tensors)
    idx = find_index(handle, name, n_tensors)
    if idx < 0
      puts "missing: " + name
      return
    end
    nel = d_model * n_kv * d_head
    tmp = Array.new(nel, 0.0)
    rc  = TinyNN.tnn_gguf_read_f32_to_doubles(handle, idx, tmp, nel)
    if rc != 0
      puts "read failed: " + name + " rc=" + rc.to_s
      return
    end
    # Source row stride = n_kv * d_head; column block h is [h*d_head, (h+1)*d_head).
    src_cols = n_kv * d_head
    h = 0
    while h < n_kv
      mat = dst[h]
      i = 0
      while i < d_model
        j = 0
        while j < d_head
          mat.flat[i * d_head + j] = tmp[i * src_cols + h * d_head + j]
          j = j + 1
        end
        i = i + 1
      end
      h = h + 1
    end
  end

  # toy-checkpoint variant: each head is its own tensor named
  # blk.N.attn_<q|k|v>.head_H.weight, shape [d_head, d_model] in ggml
  # column-major (== row-major [d_model × d_head] in our Mat layout).
  # That is exactly what a per-head Mat expects, so each tensor reads
  # straight into its slot — no fan-out / strided extraction.
  #
  # Used by toy#gguf-checkpoint-reload (#153) to load from-scratch
  # toy GGUFs without going through the fused llama.cpp convention.
  def self.read_per_head_weight(handle, prefix_attn, dst, n_heads, d_model, d_head, n_tensors)
    h = 0
    while h < n_heads
      name = prefix_attn + ".head_" + h.to_s + ".weight"
      idx = find_index(handle, name, n_tensors)
      if idx < 0
        puts "missing: " + name
        return
      end
      mat = dst[h]
      nel = d_model * d_head
      rc = TinyNN.tnn_gguf_read_f32_to_doubles(handle, idx, mat.flat, nel)
      if rc != 0
        puts "read failed: " + name + " rc=" + rc.to_s
        return
      end
      h = h + 1
    end
  end

  # Per-head bias: blk.N.attn_<q|k|v>.head_H.bias, shape [d_head].
  def self.read_per_head_bias(handle, prefix_attn, dst, n_heads, d_head, n_tensors)
    h = 0
    while h < n_heads
      name = prefix_attn + ".head_" + h.to_s + ".bias"
      idx = find_index(handle, name, n_tensors)
      if idx < 0
        puts "missing: " + name
        return
      end
      rc = TinyNN.tnn_gguf_read_f32_to_doubles(handle, idx, dst[h], d_head)
      if rc != 0
        puts "read failed: " + name + " rc=" + rc.to_s
        return
      end
      h = h + 1
    end
  end

  # Load distilgpt2-shaped GGUF (also fits gpt2-small/medium/large) into
  # a caller-constructed GPT2LM. Returns true on success.
  def self.load_gpt2(model, path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "open failed: " + path
      return false
    end
    n_tensors = TinyNN.tnn_gguf_n_tensors(handle)
    puts "loading " + path + " (" + n_tensors.to_s + " tensors)"

    d_model = model.d_model
    d_head  = model.d_head
    n_heads = model.n_heads

    # Globals
    read_mat(handle,   "token_embd.weight",    model.token_embed, n_tensors)
    read_mat(handle,   "position_embd.weight", model.pos_embed,   n_tensors)
    read_array(handle, "output_norm.weight",   model.ln_f_gamma,  n_tensors)
    read_array(handle, "output_norm.bias",     model.ln_f_beta,   n_tensors)

    # Per-block
    li = 0
    while li < model.n_layers
      blk    = model.gpt2_blocks[li]
      prefix = "blk." + li.to_s

      read_array(handle, prefix + ".attn_norm.weight", blk.ln1_gamma, n_tensors)
      read_array(handle, prefix + ".attn_norm.bias",   blk.ln1_beta,  n_tensors)
      read_array(handle, prefix + ".ffn_norm.weight",  blk.ln2_gamma, n_tensors)
      read_array(handle, prefix + ".ffn_norm.bias",    blk.ln2_beta,  n_tensors)

      read_split_heads_weight(handle, prefix + ".attn_q.weight",
                               blk.w_q, n_heads, d_model, d_head, n_tensors)
      read_split_heads_weight(handle, prefix + ".attn_k.weight",
                               blk.w_k, n_heads, d_model, d_head, n_tensors)
      read_split_heads_weight(handle, prefix + ".attn_v.weight",
                               blk.w_v, n_heads, d_model, d_head, n_tensors)
      read_split_heads_bias(handle, prefix + ".attn_q.bias",
                             blk.b_q, n_heads, d_head, n_tensors)
      read_split_heads_bias(handle, prefix + ".attn_k.bias",
                             blk.b_k, n_heads, d_head, n_tensors)
      read_split_heads_bias(handle, prefix + ".attn_v.bias",
                             blk.b_v, n_heads, d_head, n_tensors)

      read_mat(handle,   prefix + ".attn_output.weight", blk.w_o, n_tensors)
      read_array(handle, prefix + ".attn_output.bias",   blk.b_o, n_tensors)

      read_mat(handle,   prefix + ".ffn_up.weight",   blk.w_ff1, n_tensors)
      read_array(handle, prefix + ".ffn_up.bias",     blk.b_ff1, n_tensors)
      read_mat(handle,   prefix + ".ffn_down.weight", blk.w_ff2, n_tensors)
      read_array(handle, prefix + ".ffn_down.bias",   blk.b_ff2, n_tensors)

      li = li + 1
    end

    TinyNN.tnn_gguf_free(handle)
    true
  end
end
