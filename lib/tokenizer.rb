# Tokenizer — GGUF-embedded BPE decode side.
#
# Phase 0 ships **decode only** (token IDs → text). The encode side
# (text → IDs) stays external for now; clients tokenize via the Python
# scripts in `prep/` and pass IDs to inference.
#
# Construction is via Tokenizer.from_gguf(path). Returns a Tokenizer
# instance whose `vocab` array is populated when the GGUF embeds the
# tokenizer (HF-converted GGUFs typically do); otherwise an "absent"
# tokenizer that raises on #decode.
#
# Current state: most GGUFs converted by this repo's `prep/convert_*`
# scripts do NOT embed the tokenizer. Extend those scripts to call
# gguf.GGUFWriter.add_tokenizer_* if you need decode-side support.

require_relative "tinynn"

class Tokenizer
  attr_reader :vocab_size, :bos_id, :eos_id, :pad_id, :unk_id, :present

  def initialize(vocab, bos_id, eos_id, pad_id, unk_id)
    @vocab      = vocab
    @vocab_size = vocab.length
    @bos_id     = bos_id
    @eos_id     = eos_id
    @pad_id     = pad_id
    @unk_id     = unk_id
    @present    = (vocab.length > 0)
  end

  # Look up a single token. Returns the GGUF-stored string form (which
  # for byte-BPE tokenizers may include the bytechar marker — the
  # caller's decode loop handles whitespace and byte chars).
  def token_at(id)
    if id < 0 || id >= @vocab_size
      return ""
    end
    @vocab[id]
  end

  # Greedy decode: concatenate token strings, skipping BOS/EOS.
  # For Llama / Qwen tokenizers this gives a readable approximation
  # (the leading-space marker `Ġ` from byte-BPE shows up as-is for
  # now; a follow-up decoder handles the byte-char remap).
  def decode(ids)
    if !@present
      puts "Tokenizer.decode: tokenizer not embedded in GGUF (vocab is empty). " +
           "Use the Python tokenizer in prep/ for now, or extend the converter " +
           "to embed tokenizer.ggml.tokens / .merges."
      return ""
    end
    out = ""
    i = 0
    while i < ids.length
      tok_id = ids[i]
      if tok_id == @bos_id || tok_id == @eos_id || tok_id == @pad_id
        i = i + 1
        next
      end
      out = out + token_at(tok_id)
      i = i + 1
    end
    out
  end

  # Build from a GGUF file. If the GGUF has `tokenizer.ggml.tokens`
  # populated, the result decodes correctly; otherwise the Tokenizer
  # is constructed with an empty vocab (callers see #present? == false).
  def self.from_gguf(path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      return Tokenizer.new([], -1, -1, -1, -1)
    end

    bos = TinyNN.tnn_gguf_get_u32(handle, "tokenizer.ggml.bos_token_id")
    eos = TinyNN.tnn_gguf_get_u32(handle, "tokenizer.ggml.eos_token_id")
    pad = TinyNN.tnn_gguf_get_u32(handle, "tokenizer.ggml.padding_token_id")
    unk = TinyNN.tnn_gguf_get_u32(handle, "tokenizer.ggml.unknown_token_id")
    n   = TinyNN.tnn_gguf_arr_n(handle, "tokenizer.ggml.tokens")

    vocab = [""]
    vocab.pop
    if n > 0
      i = 0
      while i < n
        s = TinyNN.tnn_gguf_arr_str(handle, "tokenizer.ggml.tokens", i)
        if s == nil
          vocab.push("")
        else
          vocab.push(s)
        end
        i = i + 1
      end
    end

    TinyNN.tnn_gguf_free(handle)
    Tokenizer.new(vocab, bos, eos, pad, unk)
  end
end
