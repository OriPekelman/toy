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

  # Advance the byte cursor. Wraps when (new_offset + n_tokens) would
  # land past EOF. Caller passes the corpus byte size (from a one-time
  # File.size or equivalent).
  def self.next_offset(byte_offset, n_tokens, corpus_bytes)
    new_offset = byte_offset + n_tokens * TOKEN_BYTES
    if new_offset + n_tokens * TOKEN_BYTES > corpus_bytes
      return 0
    end
    new_offset
  end
end
