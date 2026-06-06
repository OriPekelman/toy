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
  attr_reader :vocab_size, :bos_id, :eos_id, :pad_id, :unk_id, :present, :spm,
              :spm_unigram, :add_bos
  attr_accessor :add_bos

  def initialize(vocab, merges, bos_id, eos_id, pad_id, unk_id,
                 model_name = "", add_bos = false)
    @vocab      = vocab
    @vocab_size = vocab.length
    @merges     = merges
    @bos_id     = bos_id
    @eos_id     = eos_id
    @pad_id     = pad_id
    @unk_id     = unk_id
    @add_bos    = add_bos
    @present    = (vocab.length > 0)
    # SPM (SentencePiece, marker U+2581 ▁) vs GPT-2 byte-level BPE.
    # Detection: vocab heuristic OR model_name says "llama".
    #
    # The heuristic (vocab[3] == "<0x00>") is reliable for any
    # historic SPM model — every SPM tokenizer in the wild puts
    # <0x00> at index 3 (after <unk>/<s>/</s>). For Gemma 2 the
    # special-tokens band is longer and <0x00> sits further in;
    # the heuristic returns false for Gemma. The model_name "llama"
    # picks Gemma 2 (and any future SPM model) up.
    #
    # We deliberately do NOT trust model_name alone — older project
    # converters wrote "gpt2" for SPM models (Mistral's tokenizer
    # GGUF is the canonical example), so authoritatively trusting
    # model_name would flip Mistral to the wrong path. OR both
    # signals: either says SPM, treat as SPM.
    @spm = (vocab.length > 3 && vocab[3] == "<0x00>") || (model_name == "llama")
    # T-Gemma (#117): SPM split into two encoding paths:
    #   - BPE+scores (Mistral, Llama-1/2, TinyLlama): merges array
    #     populated; encode via the existing merge-loop algorithm.
    #   - Unigram (Gemma 2, newer SPM models): merges array empty;
    #     pieces are scored individually and tokenization is greedy
    #     longest-match (an approximation of the proper Viterbi
    #     decode). The vocab IS the unigram model.
    # Distinguish by the merges array being non-empty.
    @spm_unigram = @spm && (merges.length == 0)

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
    # GPT-2 bytes_to_unicode in ONE pass with an inline "kept" boolean.
    # Bytes in these three ranges map to their own codepoint; every other
    # byte maps to 256, 257, … in order (so the space 0x20 → U+0120 = Ġ).
    #
    # SPINEL LANDMINE (the #34 root cause): the previous version used an
    # `is_kept[]` Array<bool> seeded with `false` + a separate `if !is_kept[b]`
    # pass. Under Spinel that else-branch NEVER ran (n_mapped stayed 0), so the
    # mapped chars — including Ġ for 0x20 — were never built. `@byte_to_char[32]`
    # came out empty, encode dropped every leading space and selected the
    # space-less token (`upon`=25705 instead of `Ġupon`=1980), and decode had no
    # marker to restore. Inline the test as a plain boolean and branch on `k`
    # (no bool array, no `!`) — verified to produce btc[0x20]=Ġ.
    n_mapped = 0
    b = 0
    while b < 256
      k = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
      if k
        btc[b] = Tokenizer.cp_to_utf8(b)
      else
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
    # T-Gemma (#117): prepend BOS only on the SPM-Unigram path
    # (Gemma 2 needs it; without BOS at pos 0 the model produces
    # degenerate output). Other paths (byte-level BPE for SmolLM2/
    # Qwen3; BPE-SPM for Mistral/TinyLlama) preserve their existing
    # tokenization to maintain bit-identical regression behavior on
    # canonical prompts.
    if @spm_unigram
      ids = [0]; ids.pop
      if @add_bos && @bos_id != nil && @bos_id >= 0
        ids.push(@bos_id)
      end
      body = encode_spm_unigram(text)
      bi = 0
      while bi < body.length; ids.push(body[bi]); bi = bi + 1; end
      return ids
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

  # T-Gemma (#117): SPM Unigram encode (Gemma 2 and similar models
  # whose GGUF carries `tokenizer.ggml.tokens` + `tokenizer.ggml.scores`
  # but NO `tokenizer.ggml.merges`). The vocab itself IS the model —
  # no merge rules to apply.
  #
  # Algorithm: greedy longest-match over the prefixed string. For each
  # cursor position, try the longest substring (up to MAX_PIECE_LEN
  # bytes) that exists in vocab; emit its id; advance. Fall back to
  # per-UTF-8-byte <0xHH> tokens for characters not covered.
  #
  # This is an APPROXIMATION of the proper Unigram tokenizer (which
  # does Viterbi over piece scores to find the maximum-score
  # segmentation). For Gemma 2's vocab=256000, greedy longest-match
  # produces sensible tokenization for prose; rare-character cases
  # may differ from the canonical SentencePiece library output by a
  # token here or there.
  #
  # Spinel constraint: use Hash#has_key? not Hash#[]; the latter
  # returns 0 for missing keys (landmine #9). String byte/char
  # indexing via [i...j] is safe under Spinel — verified by the
  # existing decode_spm code.
  def encode_spm_unigram(text)
    ids = [0]; ids.pop
    # Prepend ▁ + replace each space with ▁. Same prefix shape as
    # the BPE-SPM path.
    sp = "\xE2\x96\x81"
    text_bytes = text.bytes
    prepared = sp + ""
    tb = 0
    while tb < text_bytes.length
      b = text_bytes[tb]
      if b == 0x20         # ASCII space → ▁
        prepared = prepared + sp
      else
        prepared = prepared + b.chr
      end
      tb = tb + 1
    end

    # Greedy longest-match. Walk character-by-character (NOT byte-by-
    # byte — multi-byte UTF-8 like ▁ must be intact for vocab lookup).
    # Max piece length cap: 64 chars handles the longest pieces in
    # known SPM vocabs comfortably.
    chars      = prepared.chars
    n          = chars.length
    max_piece  = 64
    pos        = 0
    while pos < n
      # Build the longest candidate substring (up to max_piece chars or
      # end of input), then shrink until we find a hit in vocab.
      jmax = pos + max_piece
      if jmax > n; jmax = n; end
      j      = jmax
      hit_id = -1
      hit_len = 0
      while j > pos
        piece = ""
        k = pos
        while k < j; piece = piece + chars[k]; k = k + 1; end
        if @vocab_inv.has_key?(piece)
          hit_id  = @vocab_inv[piece]
          hit_len = j - pos
          break
        end
        j = j - 1
      end
      if hit_id >= 0
        ids.push(hit_id)
        pos = pos + hit_len
      else
        # Byte-fallback: decompose the single character at `pos` into
        # per-byte <0xHH> tokens. SPM vocabs include all 256 byte
        # tokens for exactly this case.
        ch = chars[pos]
        cbytes = ch.bytes
        cbi = 0
        while cbi < cbytes.length
          hex = cbytes[cbi].to_s(16).upcase
          if hex.length == 1; hex = "0" + hex; end
          tag = "<0x" + hex + ">"
          if @vocab_inv.has_key?(tag)
            ids.push(@vocab_inv[tag])
          else
            if !@warned_unk
              puts "WARN: tokenizer(spm-unigram): byte-fallback " + tag +
                   " not in vocab — emitting UNK"
              @warned_unk = true
            end
            if @unk_id != nil && @unk_id >= 0
              ids.push(@unk_id)
            end
          end
          cbi = cbi + 1
        end
        pos = pos + 1
      end
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
    # T-Gemma (#117): tokenizer.ggml.model is the authoritative kind
    # ("llama" = SPM, "gpt2" = byte-level BPE). Older Llama-1/2
    # GGUFs may omit it; the Tokenizer ctor falls back to a vocab
    # heuristic when model_name is empty.
    model_name = GgufKV.tnn_gguf_get_str(handle, "tokenizer.ggml.model")
    if model_name == nil; model_name = ""; end
    # add_bos_token: per-arch flag (Gemma 2 sets this to true).
    # Returns -1 when the key is missing; treat as false.
    add_bos_v = GgufKV.tnn_gguf_get_bool(handle, "tokenizer.ggml.add_bos_token")
    add_bos   = (add_bos_v == 1)
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
    Tokenizer.new(vocab, merges, bos, eos, pad, unk, model_name, add_bos)
  end
end
