# Tokenizer — GGUF-embedded BPE encoder + decoder.
#
# Standalone Ruby tokenizer for tiktoken-style byte-level BPE (Llama-3,
# Qwen2/3, SmolLM2). The GGUF must have been converted with
# prep/convert_smollm2_to_gguf.py --with-tokenizer.
#
# Caller API:
#   tok = Tokenizer.from_gguf("data/smollm2-135m-tok.gguf")
#   ids = tok.encode("Hello world.")
#   txt = tok.decode(ids)
#
# Pre-tokenizer regex is ASCII-tolerant — Spinel's regex engine doesn't
# expose \p{L}/\p{N} via Onigmo. English text encodes correctly;
# non-ASCII (CJK, accented letters) falls into the "other" branch and
# may diverge from HF for those code points.
#
# Decoupled from lib/tinynn.rb via lib/gguf_kv.rb to keep this loadable
# in tokenizer-only binaries (avoids a known Spinel cross-class
# type-inference issue with lib/transformer.rb's Mat).

require_relative "gguf_kv"

class Tokenizer
  attr_reader :vocab_size, :bos_id, :eos_id, :pad_id, :unk_id, :present

  def initialize(vocab, merges, bos_id, eos_id, pad_id, unk_id)
    @vocab      = vocab
    @vocab_size = vocab.length
    @merges     = merges
    @bos_id     = bos_id
    @eos_id     = eos_id
    @pad_id     = pad_id
    @unk_id     = unk_id
    @present    = (vocab.length > 0)

    # Inverse vocab: token-string → id.
    @vocab_inv = {}
    i = 0
    while i < vocab.length
      @vocab_inv[vocab[i]] = i
      i = i + 1
    end

    # Merge-rank hash: "a b" → rank. Lower = higher priority.
    @merge_rank = {}
    i = 0
    while i < merges.length
      @merge_rank[merges[i]] = i
      i = i + 1
    end

    # GPT-2 byte→char table built lazily on first access (initialize
    # used to segv on Spinel when both this big build and the large
    # vocab/merges hashes ran inside one ctor — moved out for safety).
    @byte_to_char = nil
    @char_to_byte = nil
  end

  def build_byte_tables
    return if @byte_to_char != nil
    btc = [""]
    btc.pop
    j = 0
    while j < 256
      btc.push("")
      j = j + 1
    end
    is_kept = [false]
    is_kept.pop
    j = 0
    while j < 256
      is_kept.push(false)
      j = j + 1
    end
    b = 0x21
    while b <= 0x7E
      is_kept[b] = true
      btc[b] = b.chr
      b = b + 1
    end
    b = 0xA1
    while b <= 0xAC
      is_kept[b] = true
      btc[b] = Tokenizer.cp_to_utf8(b)
      b = b + 1
    end
    b = 0xAE
    while b <= 0xFF
      is_kept[b] = true
      btc[b] = Tokenizer.cp_to_utf8(b)
      b = b + 1
    end
    n_mapped = 0
    b = 0
    while b < 256
      if !is_kept[b]
        btc[b] = Tokenizer.cp_to_utf8(256 + n_mapped)
        n_mapped = n_mapped + 1
      end
      b = b + 1
    end
    @byte_to_char = btc
    ctb = {}
    i = 0
    while i < btc.length
      ctb[btc[i]] = i
      i = i + 1
    end
    @char_to_byte = ctb
  end

  # Codepoint → UTF-8 string. Used only for codepoints < 0x800 (the
  # GPT-2 mapping maxes at 0x143). Spinel-friendly: no Encoding::UTF_8.
  def self.cp_to_utf8(c)
    if c < 0x80
      return c.chr
    end
    if c < 0x800
      b1 = (0xC0 | (c >> 6)).chr
      b2 = (0x80 | (c & 0x3F)).chr
      return b1 + b2
    end
    "?"
  end

  def token_at(id)
    if id < 0 || id >= @vocab_size
      return ""
    end
    @vocab[id]
  end

  # Decode IDs → text. Walks token byte-chars, maps each back to its
  # original byte, returns the concatenated UTF-8 string.
  def decode(ids)
    if !@present
      puts "Tokenizer.decode: vocab not loaded (re-convert with --with-tokenizer)"
      return ""
    end
    build_byte_tables
    chained = ""
    i = 0
    while i < ids.length
      tok_id = ids[i]
      if tok_id == @bos_id || tok_id == @eos_id || tok_id == @pad_id
        i = i + 1
        next
      end
      chained = chained + token_at(tok_id)
      i = i + 1
    end
    out = ""
    chars = chained.chars
    j = 0
    while j < chars.length
      c = chars[j]
      b = @char_to_byte[c]
      if b == nil
        out = out + "?"
      else
        out = out + b.chr
      end
      j = j + 1
    end
    out
  end

  # Encode text → IDs. Pre-tokenize via regex; for each chunk, run the
  # byte→char map then BPE merge loop; lookup pieces in vocab.
  def encode(text)
    if !@present
      puts "Tokenizer.encode: vocab not loaded (re-convert with --with-tokenizer)"
      return []
    end
    build_byte_tables
    ids = [0]
    ids.pop
    # Pre-tokenizer regex (Llama-3 / cl100k_base style, ASCII fallback).
    pre_re = /'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D|[^\r\na-zA-Z0-9]?[a-zA-Z]+|[0-9]{1,3}| ?[^\sa-zA-Z0-9]+[\r\n]*|\s+/
    chunks = text.scan(pre_re)
    ci = 0
    while ci < chunks.length
      chunk = chunks[ci]
      bytes = chunk.bytes
      # Lift bytes to GPT-2 byte-chars.
      bc = ""
      bi = 0
      while bi < bytes.length
        bc = bc + @byte_to_char[bytes[bi]]
        bi = bi + 1
      end
      # BPE merge loop: start with single-char pieces; iteratively apply
      # the lowest-rank merge until no merge applies.
      pieces = bc.chars
      while true
        best_rank = 999999999
        best_idx = -1
        k = 0
        while k < pieces.length - 1
          key = pieces[k] + " " + pieces[k + 1]
          r = @merge_rank[key]
          if r != nil && r < best_rank
            best_rank = r
            best_idx = k
          end
          k = k + 1
        end
        if best_idx < 0
          break
        end
        pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1]
        pieces.delete_at(best_idx + 1)
      end
      # Vocab lookup.
      pi = 0
      while pi < pieces.length
        tid = @vocab_inv[pieces[pi]]
        if tid == nil
          if @unk_id != nil && @unk_id >= 0
            ids.push(@unk_id)
          end
        else
          ids.push(tid)
        end
        pi = pi + 1
      end
      ci = ci + 1
    end
    ids
  end

  # Build from a GGUF file with embedded tokenizer metadata.
  def self.from_gguf(path)
    empty = [""]
    empty.pop
    handle = GgufKV.tnn_gguf_load(path)
    if handle == nil
      return Tokenizer.new(empty, empty, -1, -1, -1, -1)
    end

    bos = GgufKV.tnn_gguf_get_u32(handle, "tokenizer.ggml.bos_token_id")
    eos = GgufKV.tnn_gguf_get_u32(handle, "tokenizer.ggml.eos_token_id")
    pad = GgufKV.tnn_gguf_get_u32(handle, "tokenizer.ggml.padding_token_id")
    unk = GgufKV.tnn_gguf_get_u32(handle, "tokenizer.ggml.unknown_token_id")
    n_tok    = GgufKV.tnn_gguf_arr_n(handle, "tokenizer.ggml.tokens")
    n_merges = GgufKV.tnn_gguf_arr_n(handle, "tokenizer.ggml.merges")

    vocab = [""]
    vocab.pop
    if n_tok > 0
      i = 0
      while i < n_tok
        s = GgufKV.tnn_gguf_arr_str(handle, "tokenizer.ggml.tokens", i)
        if s == nil
          vocab.push("")
        else
          vocab.push(s)
        end
        i = i + 1
      end
    end

    merges = [""]
    merges.pop
    if n_merges > 0
      i = 0
      while i < n_merges
        s = GgufKV.tnn_gguf_arr_str(handle, "tokenizer.ggml.merges", i)
        if s == nil
          merges.push("")
        else
          merges.push(s)
        end
        i = i + 1
      end
    end

    GgufKV.tnn_gguf_free(handle)
    Tokenizer.new(vocab, merges, bos, eos, pad, unk)
  end
end
