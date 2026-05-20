# Tokenizer — GGUF-embedded BPE decoder.
#
# Phase D1: prep/convert_smollm2_to_gguf.py --with-tokenizer embeds
# vocab + merges + special-token IDs in the GGUF.
#
# Phase D2 (this file): the Ruby decoder. Decouples from lib/tinynn.rb
# by going through lib/gguf_kv.rb instead — this dodges a Spinel
# cross-class type-inference issue where loading Tokenizer alongside
# lib/transformer.rb widened Mat#nrows from mrb_int to sp_RbVal.
#
# Status: filing 3 Spinel issues; see docs/design/tokenizer-status.md
# for the catalog. Full encoder + large-vocab decode are blocked on
# upstream fixes. This file ships a viable scaffold + a small-vocab
# smoke test.

require_relative "gguf_kv"

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

  def token_at(id)
    if id < 0 || id >= @vocab_size
      return ""
    end
    @vocab[id]
  end

  # Decode token IDs back to text. Returns the raw byte-char form
  # (GPT-2 mapping: "Ġ" for space, "Ċ" for newline). A follow-up
  # decoder lifts these back to UTF-8 — blocked on Spinel-side
  # constraints in the byte-char build path.
  def decode(ids)
    if !@present
      puts "Tokenizer.decode: vocab not loaded (re-convert with --with-tokenizer)"
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

  # Build from a GGUF file. The full-vocab populate path is currently
  # blocked by Spinel issues (see docs/design/tokenizer-status.md); we
  # populate first N tokens only as a placeholder until the upstream
  # fixes land.
  def self.from_gguf(path)
    empty = [""]
    empty.pop
    handle = GgufKV.tnn_gguf_load(path)
    if handle == nil
      return Tokenizer.new(empty, -1, -1, -1, -1)
    end

    bos = GgufKV.tnn_gguf_get_u32(handle, "tokenizer.ggml.bos_token_id")
    eos = GgufKV.tnn_gguf_get_u32(handle, "tokenizer.ggml.eos_token_id")
    pad = GgufKV.tnn_gguf_get_u32(handle, "tokenizer.ggml.padding_token_id")
    unk = GgufKV.tnn_gguf_get_u32(handle, "tokenizer.ggml.unknown_token_id")
    n   = GgufKV.tnn_gguf_arr_n(handle, "tokenizer.ggml.tokens")

    vocab = [""]
    vocab.pop
    # SPINEL-LIMITED: populate up to a small cap. Populating the full
    # vocab triggers a GC segv on Tokenizer.new — Spinel's :str FFI
    # return aliases to ggml-owned memory and even fresh-string-copy
    # workarounds don't currently produce a Ruby-owned string. Tracking
    # in docs/design/tokenizer-status.md.
    cap = n
    if cap > 256
      cap = 256
    end
    if cap > 0
      i = 0
      while i < cap
        s = GgufKV.tnn_gguf_arr_str(handle, "tokenizer.ggml.tokens", i)
        if s == nil
          vocab.push("")
        else
          # Walk chars to (try to) force a fresh string. Imperfect under
          # current Spinel; see the doc.
          copy = ""
          chars = s.chars
          j = 0
          while j < chars.length
            copy = copy + chars[j]
            j = j + 1
          end
          vocab.push(copy)
        end
        i = i + 1
      end
    end

    GgufKV.tnn_gguf_free(handle)
    Tokenizer.new(vocab, bos, eos, pad, unk)
  end
end
