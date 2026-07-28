# E2.4 / GH#14 — streaming token-corpus loader.
#
# Reads packed i32 tokens from a binary file (produced by
# prep/pretokenize_corpus.py) in fixed-size sequences. Caller owns
# the byte-offset cursor.
#
# Wraparound: when a read would land past EOF, the loader wraps to
# offset 0 and re-reads from the start. That mirrors the standard
# "epoch" pattern — finite corpus, train_steps × T may exceed corpus
# token count → wrap and repeat.
#
# The C-side primitive (tnn_read_i32_file) does one fopen+fseek+fread
# per call. For a streaming training loop, this is one syscall per
# training step — negligible. If we ever need higher throughput, mmap
# the file once and slice (a follow-up; not urgent at TinyStories or
# 10M-token-shard scale).

module ToyCorpusLoader
  TOKEN_BYTES = 4   # int32

  # toy#129 item 1 — the self-describing pack header (written by
  # prep/pretokenize_pack.py):
  #   bytes 0-3   "TOYC" magic
  #   bytes 4-7   u32 LE version (1)
  #   bytes 8-11  u32 LE vocab
  #   bytes 12-15 u32 LE reserved
  # Headerless packs (ts_seqs*.bin, the toy#123 fixture) stay readable:
  # probe_vocab returns 0 for them and data_offset returns 0.
  TOYC_MAGIC  = 1129926484   # "TOYC" as little-endian i32
  HEADER_I32S = 4
  HEADER_BYTES = 16

  # Returns the pack's declared vocab (TOYC v1), or 0 for a headerless
  # legacy pack. FAILS LOUD on a TOYC pack with an unknown version —
  # silently misreading a future format as token data is the exact
  # never-mask failure this header exists to prevent.
  def self.probe_vocab(path)
    hd = Array.new(HEADER_I32S, 0)
    got = TinyNN.tnn_read_i32_file(path, 0, HEADER_I32S, hd)
    if got < HEADER_I32S
      return 0
    end
    if hd[0] != TOYC_MAGIC
      return 0
    end
    if hd[1] != 1
      raise "ToyCorpusLoader: TOYC version " + hd[1].to_s +
            " unsupported (this build reads v1) — " + path
    end
    hd[2]
  end

  # First token byte: 16 for a TOYC pack, 0 for a headerless one.
  def self.data_offset(path)
    if probe_vocab(path) > 0
      return HEADER_BYTES
    end
    0
  end

  # Read exactly n_tokens tokens starting at byte_offset. Returns the
  # tokens Array<Int>. If reading would short-cut at EOF, wraps to 0
  # and retries; if even that fails (corpus < n_tokens), pads with 0s
  # and emits a warning.
  def self.read_seq(path, byte_offset, n_tokens)
    buf = Array.new(n_tokens, 0)
    got = TinyNN.tnn_read_i32_file(path, byte_offset, n_tokens, buf)
    if got == n_tokens
      return buf
    end
    if got < 0
      puts "WARN: ToyCorpusLoader.read_seq rc=" + got.to_s + " path=" + path + " offset=" + byte_offset.to_s
      return buf
    end
    # Short read at EOF — wrap to 0 and try to fill the rest from the
    # start. If THAT also short-reads, the corpus is shorter than
    # n_tokens — pad with zeros and warn once.
    remainder = n_tokens - got
    wrap_buf = Array.new(remainder, 0)
    got2 = TinyNN.tnn_read_i32_file(path, 0, remainder, wrap_buf)
    if got2 < remainder
      puts "WARN: corpus shorter than n_tokens=" + n_tokens.to_s +
           " (got " + got.to_s + " + " + got2.to_s + "); padding with 0"
    end
    i = 0
    while i < remainder
      buf[got + i] = wrap_buf[i]
      i = i + 1
    end
    buf
  end
end
