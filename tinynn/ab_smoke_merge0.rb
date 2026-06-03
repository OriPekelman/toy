# Dump the first 5 merges in each tokenizer's GGUF and check whether
# the LHS+RHS combination exists in the vocab. If merge[0] = "? Ċ"
# but vocab has no "?Ċ" entry, we have an inconsistent BPE
# serialization — that's the SmolLM2 bug.

require_relative "../lib/toy/io/gguf_kv"

def dump(path)
  h = GgufKV.tnn_gguf_load(path)
  if h == nil; return; end
  n_tok    = GgufKV.tnn_gguf_arr_n(h, "tokenizer.ggml.tokens")
  n_merges = GgufKV.tnn_gguf_arr_n(h, "tokenizer.ggml.merges")
  puts "==== " + path
  puts "  vocab=" + n_tok.to_s + "  merges=" + n_merges.to_s
  # First 5 merges + check both pieces and the concat in vocab
  i = 0
  while i < 5
    m = GgufKV.tnn_gguf_arr_str(h, "tokenizer.ggml.merges", i)
    if m != nil
      # Find space; split into A, B.
      sp = m.index(" ")
      if sp != nil && sp > 0
        a = m[0..(sp - 1)]
        b = m[(sp + 1)..(m.length - 1)]
        concat = a + b
        # Linear scan for vocab presence (slow but only 49k entries)
        idx = -1
        j = 0
        while j < n_tok
          s = GgufKV.tnn_gguf_arr_str(h, "tokenizer.ggml.tokens", j)
          if s == concat; idx = j; j = n_tok; end
          j = j + 1
        end
        puts "  merge[" + i.to_s + "] " + m.inspect + " → " + concat.inspect +
             " in_vocab=" + (idx >= 0 ? ("yes (id " + idx.to_s + ")") : "no")
      end
    end
    i = i + 1
  end
  GgufKV.tnn_gguf_free(h)
end

dump("data/smollm2-135m-tok-new.gguf")
dump("data/llama-3.2-1b-tok.gguf")
