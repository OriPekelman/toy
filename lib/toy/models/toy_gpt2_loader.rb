# lib/toy_gpt2_loader.rb — GGUF weight load for Toy::GPT2.
#
# Kept separate from lib/gguf_load.rb so a demo that only uses GPT-2
# doesn't have to pull SmolLM2 types into Spinel's compile graph.

require_relative "../io/gguf_load"
require_relative "toy_gpt2"

module GGUFLoad
  # Same GGUF layout, loaded into a Toy::GPT2. The weights live under
  # sub-modules now (`blk.attn.w_q[h]`, `blk.ln1.gamma`, …), so this
  # mirrors load_gpt2 with the new path expressions.
  def self.load_toy_gpt2(model, path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "open failed: " + path
      return false
    end
    n_tensors = TinyNN.tnn_gguf_n_tensors(handle)
    puts "loading " + path + " (" + n_tensors.to_s + " tensors)"

    cfg     = model.cfg
    d_model = cfg.d_model
    n_heads = cfg.n_heads
    d_head  = d_model / n_heads

    read_mat(handle,   "token_embd.weight",    model.token_embed.weight, n_tensors)
    read_mat(handle,   "position_embd.weight", model.pos_embed.weight,   n_tensors)
    read_array(handle, "output_norm.weight",   model.final_norm.gamma,   n_tensors)
    read_array(handle, "output_norm.bias",     model.final_norm.beta,    n_tensors)

    li = 0
    while li < cfg.n_layers
      blk   = model.stack[li]
      prefix = "blk." + li.to_s

      read_array(handle, prefix + ".attn_norm.weight", blk.ln1.gamma, n_tensors)
      read_array(handle, prefix + ".attn_norm.bias",   blk.ln1.beta,  n_tensors)
      read_array(handle, prefix + ".ffn_norm.weight",  blk.ln2.gamma, n_tensors)
      read_array(handle, prefix + ".ffn_norm.bias",    blk.ln2.beta,  n_tensors)

      read_split_heads_weight(handle, prefix + ".attn_q.weight",
                               blk.attn.w_q, n_heads, d_model, d_head, n_tensors)
      read_split_heads_weight(handle, prefix + ".attn_k.weight",
                               blk.attn.w_k, n_heads, d_model, d_head, n_tensors)
      read_split_heads_weight(handle, prefix + ".attn_v.weight",
                               blk.attn.w_v, n_heads, d_model, d_head, n_tensors)
      read_split_heads_bias(handle, prefix + ".attn_q.bias",
                             blk.attn.b_q, n_heads, d_head, n_tensors)
      read_split_heads_bias(handle, prefix + ".attn_k.bias",
                             blk.attn.b_k, n_heads, d_head, n_tensors)
      read_split_heads_bias(handle, prefix + ".attn_v.bias",
                             blk.attn.b_v, n_heads, d_head, n_tensors)

      read_mat(handle,   prefix + ".attn_output.weight", blk.attn.w_o, n_tensors)
      read_array(handle, prefix + ".attn_output.bias",   blk.attn.b_o, n_tensors)

      read_mat(handle,   prefix + ".ffn_up.weight",   blk.ffn.w1, n_tensors)
      read_array(handle, prefix + ".ffn_up.bias",     blk.ffn.b1, n_tensors)
      read_mat(handle,   prefix + ".ffn_down.weight", blk.ffn.w2, n_tensors)
      read_array(handle, prefix + ".ffn_down.bias",   blk.ffn.b2, n_tensors)

      li = li + 1
    end

    TinyNN.tnn_gguf_free(handle)
    true
  end

end
