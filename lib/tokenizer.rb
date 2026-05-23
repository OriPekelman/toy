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
  attr_reader :vocab_size, :bos_id, :eos_id, :pad_id, :unk_id, :present, :spm

  def initialize(vocab, merges, bos_id, eos_id, pad_id, unk_id)
    @vocab      = vocab
    @vocab_size = vocab.length
    @merges     = merges
    @bos_id     = bos_id
    @eos_id     = eos_id
    @pad_id     = pad_id
    @unk_id     = unk_id
    @present    = (vocab.length > 0)
    # T1.3: SentencePiece vs GPT-2 byte-level BPE. Detected by
    # checking vocab[3] == "<0x00>" (the first byte-fallback token).
    # SPM models (Llama-1/2, Mistral, TinyLlama) use U+2581 (▁) for
    # leading space and emit <0xHH> tokens for chars not in vocab.
    # GPT-2 models (SmolLM2, Llama-3, Qwen) use the Ġ byte-map.
    @spm = (vocab.length > 3 && vocab[3] == "<0x00>")

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

    # One-shot warn flag for UNK emissions. We *never* silently emit
    # UNK — see lib/tokenizer.rb's encode for the rationale. The first
    # piece that misses vocab prints to stderr with the piece value;
    # subsequent misses are quiet to avoid spamming long prompts.
    @warned_unk = false
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
    if @spm
      return decode_spm(ids)
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

  # T1.3: SentencePiece decode. Concatenate token strings; replace ▁
  # with space; collapse byte-fallback <0xHH> sequences into UTF-8
  # bytes. Llama-1/2 / Mistral / TinyLlama use this path.
  #
  # SPM tokenizers prepend a leading ▁ to encode the first word's
  # boundary (Llama-2 / Mistral convention — encoding "X" gives
  # ["▁X"]). On decode, we strip exactly one leading ▁ at the start
  # of the output so the round-trip is lossless. After the first
  # piece, ▁ in the middle of a token (e.g. "▁the") becomes a
  # regular space.
  def decode_spm(ids)
    out = ""
    first_emit = true
    i = 0
    while i < ids.length
      tid = ids[i]
      if tid == @bos_id || tid == @eos_id || tid == @pad_id
        i = i + 1
        next
      end
      piece = token_at(tid)
      # Byte-fallback token: "<0xHH>". Hex parse via byte indexing
      # because Spinel's String#[Range] can mis-slice on multi-char
      # ranges (memory feedback_spinel_type_inference_landmines).
      pb = piece.bytes
      if pb.length == 6 && pb[0] == 60 && pb[1] == 48 && pb[2] == 120 && pb[5] == 62
        out = out + ((hex_digit_value(pb[3]) << 4) | hex_digit_value(pb[4])).chr
        first_emit = false
      else
        # Walk UTF-8 bytes; collapse 0xE2 0x96 0x81 (▁) into ASCII
        # space, but skip the very first ▁ if it's a leading-space
        # encoding marker.
        bi = 0
        while bi < pb.length
          if bi + 2 < pb.length && pb[bi] == 226 && pb[bi + 1] == 150 && pb[bi + 2] == 129
            if first_emit
              # Drop the leading ▁
            else
              out = out + " "
            end
            first_emit = false
            bi = bi + 3
          else
            out = out + pb[bi].chr
            first_emit = false
            bi = bi + 1
          end
        end
      end
      i = i + 1
    end
    out
  end

  # ASCII hex char → 0..15. Caller has already verified it's a hex
  # digit (because the surrounding token matches <0x..>).
  def hex_digit_value(b)
    if b >= 48 && b <= 57; return b - 48; end           # '0'..'9'
    if b >= 65 && b <= 70; return b - 65 + 10; end      # 'A'..'F'
    if b >= 97 && b <= 102; return b - 97 + 10; end     # 'a'..'f'
    0
  end

  # Encode text → IDs. Pre-tokenize via regex; for each chunk, run the
  # byte→char map then BPE merge loop; lookup pieces in vocab.
  def encode(text)
    if !@present
      puts "Tokenizer.encode: vocab not loaded (re-convert with --with-tokenizer)"
      return []
    end
    if @spm
      return encode_spm(text)
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
          # IMPORTANT: in Spinel, `Hash#[missing_key]` returns the
          # integer 0, not nil. Without the has_key? guard, every
          # absent merge appears to have rank 0 (the highest
          # priority), which makes BPE apply spurious merges and
          # produce pieces that aren't in the vocab. The bug shows
          # up on SmolLM2 (where merges are sparser) but the same
          # broken control flow is there on every model.
          if @merge_rank.has_key?(key)
            r = @merge_rank[key]
            if r < best_rank
              best_rank = r
              best_idx = k
            end
          end
          k = k + 1
        end
        if best_idx < 0
          break
        end
        pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1]
        pieces.delete_at(best_idx + 1)
      end
      # Vocab lookup. Same has_key? rule as the merge loop above —
      # without it, missing vocab entries silently resolve to id 0
      # (whatever vocab[0] is, usually a special token like
      # <|endoftext|>), and the decode side strips it. End result:
      # text round-trips with silently-dropped characters.
      pi = 0
      while pi < pieces.length
        piece = pieces[pi]
        if @vocab_inv.has_key?(piece)
          ids.push(@vocab_inv[piece])
        else
          if !@warned_unk
            puts "WARN: tokenizer: piece " + piece.inspect +
                 " not in vocab — emitting UNK (this prompt may decode lossy)"
            @warned_unk = true
          end
          if @unk_id != nil && @unk_id >= 0
            ids.push(@unk_id)
          end
        end
        pi = pi + 1
      end
      ci = ci + 1
    end
    ids
  end

  # T1.3: SentencePiece encode. Llama-1/2 / Mistral / TinyLlama. Differs
  # from GPT-2 byte-level BPE in two ways:
  #   - leading space is encoded as ▁ (U+2581), not Ġ
  #   - chars not in vocab fall back to per-UTF-8-byte <0xHH> tokens
  #     instead of going through a fixed byte-to-char map
  # Algorithm: prepend ▁; replace each space with ▁; split into chars;
  # byte-fallback any char missing from vocab; then run the BPE merge
  # loop (identical to the GPT-2 path, same has_key? rule for Spinel).
  def encode_spm(text)
    ids = [0]; ids.pop
    # Prepend ▁ + replace spaces with ▁. Bytewise to dodge encoding
    # concerns under Spinel; ▁ = U+2581 = 0xE2 0x96 0x81 in UTF-8.
    sp = "\xE2\x96\x81"
    text_bytes = text.bytes
    pre = sp + ""   # leading ▁
    tb = 0
    while tb < text_bytes.length
      b = text_bytes[tb]
      if b == 0x20         # ASCII space → ▁
        pre = pre + sp
      else
        pre = pre + b.chr
      end
      tb = tb + 1
    end

    pieces = pre.chars
    # Byte-fallback for any char not in vocab. UTF-8 chars are
    # decomposed into per-byte <0xHH> piece strings; those ARE in
    # vocab (positions 3..258).
    pi = 0
    expanded = [""]; expanded.pop
    while pi < pieces.length
      ch = pieces[pi]
      if @vocab_inv.has_key?(ch)
        expanded.push(ch)
      else
        cbytes = ch.bytes
        cbi = 0
        while cbi < cbytes.length
          hex = cbytes[cbi].to_s(16).upcase
          if hex.length == 1; hex = "0" + hex; end
          expanded.push("<0x" + hex + ">")
          cbi = cbi + 1
        end
      end
      pi = pi + 1
    end
    pieces = expanded

    # BPE merge loop. Same form as the GPT-2 path; merges use a
    # space-delimited "a b" key. has_key? guards against Spinel's
    # hash-missing-returns-0 (memory feedback #9).
    while true
      best_rank = 999999999
      best_idx = -1
      k = 0
      while k < pieces.length - 1
        key = pieces[k] + " " + pieces[k + 1]
        if @merge_rank.has_key?(key)
          r = @merge_rank[key]
          if r < best_rank
            best_rank = r
            best_idx = k
          end
        end
        k = k + 1
      end
      if best_idx < 0; break; end
      pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1]
      pieces.delete_at(best_idx + 1)
    end

    # Vocab lookup with the never-mask rule from T1.2.
    pi = 0
    while pi < pieces.length
      piece = pieces[pi]
      if @vocab_inv.has_key?(piece)
        ids.push(@vocab_inv[piece])
      else
        if !@warned_unk
          puts "WARN: tokenizer(spm): piece " + piece.inspect +
               " not in vocab — emitting UNK"
          @warned_unk = true
        end
        if @unk_id != nil && @unk_id >= 0
          ids.push(@unk_id)
        end
      end
      pi = pi + 1
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
