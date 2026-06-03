# GPT-2 byte-level BPE in Ruby (Spinel-compatible). Replaces the
# Python-based prep/tokens.py for runtime tokenization so the
# Spinel binary becomes self-contained.
#
# Three tables loaded from disk (produced by prep/dump_bpe.py):
#
#   data/gpt2-bpe-bytechars.tsv   byte → "visible char" mapping (256)
#   data/gpt2-bpe-vocab.tsv       <id>\t<token>  × 50257
#   data/gpt2-bpe-merges.tsv      <rank>\t<A>\t<B>  × ~50000
#
# Tokens are already in GPT-2's byte-encoded form (space → "Ġ" etc),
# matching what HF transformers' tokenizer emits.
#
# Pre-tokenization here is a simple char-class splitter (letters /
# digits / punctuation / whitespace) — covers ASCII English and most
# punctuation but does NOT reproduce GPT-2's full contraction regex
# (handles 's / 't etc by treating them as separate punct runs, which
# usually but not always tokenizes identically). For BPE-perfect
# parity with HF use prep/tokens.py.

# Load and hold the three lookup tables. Built once at startup.
class GPT2BPETables
  attr_accessor :byte_chars,    # Array<String>(256): byte → utf-8 char
                :char_bytes,    # Hash<String,Int>: reverse of byte_chars
                :vocab_id,      # Hash<String,Int>: token → id
                :vocab_tok,     # Array<String>(50257): id → token
                :merge_rank,    # Hash<String,Int>: "A\tB" → rank

                :punct_byte     # Array<Int>(256): 1 if punct (single-byte chars only)

  def initialize
    @byte_chars  = Array.new(256, "?")
    @char_bytes  = {}
    @vocab_id    = {}
    @vocab_tok   = Array.new(50257, "")
    @merge_rank  = {}
    @punct_byte  = Array.new(256, 0)
  end
end

module GPT2BPE
  # Build the punctuation mask: anything that's not a letter, digit,
  # or whitespace. Bytes 0..127 are the ones we make decisions on
  # (English + punctuation); bytes >= 128 are part of multi-byte UTF-8
  # sequences and are treated as "word chars" so they stay glued.
  def self.build_punct_mask(tables)
    b = 0
    while b < 256
      is_word = false
      if b == 0x20      # space — special-cased before this mask
        is_word = false
      elsif b >= 0x80   # high byte: part of UTF-8 multi-byte; treat as word
        is_word = true
      elsif (b >= 0x30 && b <= 0x39)   # 0-9
        is_word = true
      elsif (b >= 0x41 && b <= 0x5a)   # A-Z
        is_word = true
      elsif (b >= 0x61 && b <= 0x7a)   # a-z
        is_word = true
      end
      if !is_word && b != 0x20
        tables.punct_byte[b] = 1
      end
      b = b + 1
    end
  end

  def self.load(dir)
    tables = GPT2BPETables.new

    # byte → "visible char" table
    File.open(dir + "/gpt2-bpe-bytechars.tsv", "r") do |f|
      f.each_line do |line|
        parts = line.chomp.split("\t")
        bv = parts[0].to_i
        cs = parts[1]
        tables.byte_chars[bv] = cs
        tables.char_bytes[cs] = bv
      end
    end

    # vocab: <id>\t<token>
    File.open(dir + "/gpt2-bpe-vocab.tsv", "r") do |f|
      f.each_line do |line|
        parts = line.chomp.split("\t")
        idv = parts[0].to_i
        tok = parts[1]
        # Skip blank-token lines (none in GPT-2's vocab but defensive).
        if tok != nil
          tables.vocab_id[tok] = idv
          tables.vocab_tok[idv] = tok
        end
      end
    end

    # merges: <rank>\t<A>\t<B>. Store the actual rank; lookup uses
    # has_key? to distinguish a stored 0 (the rank-0 merge "Ġ t",
    # GPT-2's highest-priority merge) from missing keys.
    #
    # (Older versions of this file stored `rank + 1` to dodge a Spinel
    # quirk where `Int 0 != nil` evaluated to false — see matz/spinel#521
    # for the discussion. has_key? always worked correctly on the
    # typed hash, so this is the cleaner form matz recommended.)
    File.open(dir + "/gpt2-bpe-merges.tsv", "r") do |f|
      f.each_line do |line|
        parts = line.chomp.split("\t")
        rank = parts[0].to_i
        a    = parts[1]
        b    = parts[2]
        tables.merge_rank[a + "\t" + b] = rank
      end
    end

    build_punct_mask(tables)
    tables
  end

  # Apply BPE merges to one pre-token's char-string sequence.
  # Returns Array<String> of final sub-tokens.
  def self.bpe_merge(chars, tables)
    # Copy into a fresh Array<String>. Seeded with first element so
    # Spinel pins the element type to String (matches the existing
    # File.open/each_line seed pattern in lib/training.rb).
    if chars.length == 0
      return chars
    end
    word = [chars[0]]
    j = 1
    while j < chars.length
      word.push(chars[j])
      j = j + 1
    end

    loop_guard = 0
    while word.length >= 2 && loop_guard < 1000
      best_rank = 1_000_000_000
      best_i    = -1
      i = 0
      while i < word.length - 1
        key = word[i] + "\t" + word[i + 1]
        if tables.merge_rank.has_key?(key)
          rank = tables.merge_rank[key]
          if rank < best_rank
            best_rank = rank
            best_i    = i
          end
        end
        i = i + 1
      end

      if best_i < 0
        break
      end

      target_a = word[best_i]
      target_b = word[best_i + 1]

      new_word = [target_a + target_b]
      new_word.pop   # seed-and-pop on Array<String> (this one works)
      i = 0
      while i < word.length
        if i < word.length - 1 && word[i] == target_a && word[i + 1] == target_b
          new_word.push(target_a + target_b)
          i = i + 2
        else
          new_word.push(word[i])
          i = i + 1
        end
      end
      word = new_word
      loop_guard = loop_guard + 1
    end

    word
  end

  # Append one pre-token's BPE-encoded IDs to `out`.
  # `chars` is an Array<String> — visible-char sequence for the group.
  def self.bpe_one_group_into(out, chars, tables)
    sub = bpe_merge(chars, tables)
    k = 0
    while k < sub.length
      idv = tables.vocab_id[sub[k]]
      if idv == nil
        # Fallback: every char is in the byte-encoding table, so worst
        # case we emit single-char IDs.
        ci = 0
        while ci < sub[k].length
          one = tables.byte_chars[sub[k].getbyte(ci)]
          cid = tables.vocab_id[one]
          if cid == nil
            cid = 0
          end
          out.push(cid)
          ci = ci + 1
        end
      else
        out.push(idv)
      end
      k = k + 1
    end
  end

  # text → Array<Int> of GPT-2 token IDs.
  #
  # Inline pretokenize + BPE: rather than build an Array<Array<…>> of
  # groups (Spinel's pop on nested int-arrays mis-types as
  # int_array_ptr_array → silent no-op, leaves the seed in the output),
  # walk the byte stream once and run BPE per group as we go.
  def self.encode(text, tables)
    out = [0]
    out.pop

    chars = [tables.byte_chars[0]]
    chars.pop

    i = 0
    n = text.bytesize
    while i < n
      b = text.getbyte(i)

      # Drop any chars accumulated for the previous group (shouldn't
      # happen at the top of the loop, but defensive).
      while chars.length > 0
        chars.pop
      end

      # GPT-2 contractions: 's 't 're 've 'm 'll 'd live as their own
      # pre-tokens, regardless of the preceding context. Without this
      # we'd split "'s" into "'" + "s" and the model sees different
      # IDs than the HF tokenizer at every contraction.
      contraction_len = 0
      if b == 0x27 && i + 1 < n
        n2 = text.getbyte(i + 1) | 0x20   # lowercase the suffix byte
        if n2 == 0x73 || n2 == 0x74 || n2 == 0x6d || n2 == 0x64
          contraction_len = 2   # 's 't 'm 'd
        elsif (n2 == 0x72 || n2 == 0x76) && i + 2 < n &&
              (text.getbyte(i + 2) | 0x20) == 0x65
          contraction_len = 3   # 're 've
        elsif n2 == 0x6c && i + 2 < n &&
              (text.getbyte(i + 2) | 0x20) == 0x6c
          contraction_len = 3   # 'll
        end
      end
      if contraction_len > 0
        ci = 0
        while ci < contraction_len
          chars.push(tables.byte_chars[text.getbyte(i + ci)])
          ci = ci + 1
        end
        i = i + contraction_len
        bpe_one_group_into(out, chars, tables)
      else
        if b == 0x20
          # Leading-space group: glue space + following word/punct run.
          chars.push(tables.byte_chars[b])
          i = i + 1
          # eat extra spaces
          while i < n && text.getbyte(i) == 0x20
            chars.push(tables.byte_chars[text.getbyte(i)])
            i = i + 1
          end
          if i < n
            first_after = text.getbyte(i)
            if tables.punct_byte[first_after] == 1
              while i < n && tables.punct_byte[text.getbyte(i)] == 1 && text.getbyte(i) != 0x20
                chars.push(tables.byte_chars[text.getbyte(i)])
                i = i + 1
              end
            else
              while i < n && text.getbyte(i) != 0x20 && tables.punct_byte[text.getbyte(i)] != 1
                chars.push(tables.byte_chars[text.getbyte(i)])
                i = i + 1
              end
            end
          end
        elsif tables.punct_byte[b] == 1
          while i < n && tables.punct_byte[text.getbyte(i)] == 1 && text.getbyte(i) != 0x20
            chars.push(tables.byte_chars[text.getbyte(i)])
            i = i + 1
          end
        else
          while i < n && text.getbyte(i) != 0x20 && tables.punct_byte[text.getbyte(i)] != 1
            chars.push(tables.byte_chars[text.getbyte(i)])
            i = i + 1
          end
        end
        bpe_one_group_into(out, chars, tables)
      end
    end
    out
  end

  # Array<Int> → String. Concatenates each id's byte-encoded token
  # then maps the visible-char sequence back to raw bytes.
  def self.decode(ids, tables)
    enc = ""
    i = 0
    while i < ids.length
      tok = tables.vocab_tok[ids[i]]
      if tok != nil
        enc = enc + tok
      end
      i = i + 1
    end
    # Walk enc as UTF-8 chars, look up each in char_bytes.
    bytes = []
    j = 0
    while j < enc.length
      ch = enc[j]
      b = tables.char_bytes[ch]
      if b != nil
        bytes.push(b)
      end
      j = j + 1
    end
    out = ""
    k = 0
    while k < bytes.length
      out = out + bytes[k].chr
      k = k + 1
    end
    out
  end
end
