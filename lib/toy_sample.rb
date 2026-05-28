# Sample event emission (toy#21). After training, decodes a handful
# of completions from fixed prompts and emits one `sample` event per
# generation. Toy/v1 schema:
#   {"kind":"sample","phase":"decode","step":N,"prompt":"...","text":"..."}
#
# Why this lives in a helper module rather than in
# 06_train_from_scratch.rb directly: the vocab reader returns an
# Array<String>. Exposing Array<String> in main scope poisons
# Spinel's poly-dispatch inference table for Toy_Embedding#lookup et
# al. (06_train_from_scratch.rb header documents the failure mode).
# Wrapping the vocab read in a helper module keeps Array<String>
# inference-table-local; main scope only sees the final String
# returned from `detokenize`.

require_relative "tinynn"

module ToySample
  # Greedy autoregressive decode using a realized forward graph.
  #
  # The caller supplies the realized cache's token-IDs / positions /
  # logits tensors. We run the forward graph once per decode step:
  # upload [prompt + decoded-so-far + zero-padding], compute, argmax
  # logits[cur_len - 1, :], append, repeat.
  #
  # Returns the full sequence as Array<Int> (length prompt_len +
  # n_new). Stops early if the context window fills up.
  def self.greedy_decode(sess, t_logits, t_tokens, t_positions,
                           prompt_ids, n_new, context, vocab_size)
    tokens    = [0]; tokens.pop
    positions = [0]; positions.pop
    # Seed [prompt + zero-pad to context]. Positions are 0..context-1
    # (RoPE encodes each slot's position independently).
    pk = 0
    while pk < prompt_ids.length
      tokens.push(prompt_ids[pk])
      positions.push(pk)
      pk = pk + 1
    end
    while tokens.length < context
      tokens.push(0)
      positions.push(tokens.length - 1)
    end

    n_done = 0
    while n_done < n_new
      cur_len = prompt_ids.length + n_done
      if cur_len >= context
        return tokens[0...cur_len]
      end
      TinyNN.upload_int_array(sess, t_tokens,    tokens)
      TinyNN.upload_int_array(sess, t_positions, positions)
      TinyNN.tnn_compute(sess)
      # logits ggml shape [vocab, T] → row-major Mat is [T, vocab].
      logits = TinyNN.download_row_major(sess, t_logits, context, vocab_size)
      base = (cur_len - 1) * vocab_size
      best_v = 0
      best_l = logits.flat[base]
      v = 1
      while v < vocab_size
        x = logits.flat[base + v]
        if x > best_l
          best_l = x
          best_v = v
        end
        v = v + 1
      end
      tokens[cur_len] = best_v
      n_done = n_done + 1
    end
    tokens[0...(prompt_ids.length + n_new)]
  end

  # Read a vocab file (one token per line; line index = token ID).
  # File.read is the natural "whole file as one string" primitive
  # here; the historical Spinel d59926a interaction with File.open-
  # with-block + FFI-init is fixed as of a03bb49. Pure while-loop
  # walk over the split lines (no .each) keeps Array<String>
  # dispatch monomorphic inside this module.
  def self._read_vocab(path)
    raw = File.read(path)
    parts = raw.split("\n")
    out = ["?"]; out.pop
    i = 0
    while i < parts.length
      out.push(parts[i])
      i = i + 1
    end
    out
  end

  # IDs → space-separated word string via vocab file. Out-of-range
  # IDs render as <UNK:N>; never crashes.
  def self.detokenize(ids, vocab_path)
    vocab = _read_vocab(vocab_path)
    out = ""
    i = 0
    while i < ids.length
      if i > 0; out = out + " "; end
      tid = ids[i]
      if tid >= 0 && tid < vocab.length
        out = out + vocab[tid]
      else
        out = out + "<UNK:" + tid.to_s + ">"
      end
      i = i + 1
    end
    out
  end

  # Minimal JSON-escape for the `prompt` / `text` fields. ts_vocab
  # words are plain ASCII so the common case is identity; the quotes
  # / backslash / newline branches harden against real-corpus tokens
  # that may carry punctuation in future use.
  def self.json_escape(s)
    out = ""
    i = 0
    while i < s.length
      c = s[i...i + 1]
      if c == "\""
        out = out + "\\\""
      elsif c == "\\"
        out = out + "\\\\"
      elsif c == "\n"
        out = out + "\\n"
      elsif c == "\t"
        out = out + "\\t"
      else
        out = out + c
      end
      i = i + 1
    end
    out
  end

  # Emit one toy/v1 sample event. Caller hands in already-detokenized
  # prompt + text strings, the macro step, and the wall-time tick.
  def self.emit_event(prompt_text, text, step, t_now)
    e = "{\"kind\":\"sample\",\"phase\":\"decode\""
    e = e + ",\"t\":"      + t_now.to_s
    e = e + ",\"step\":"   + step.to_s
    e = e + ",\"prompt\":\"" + json_escape(prompt_text) + "\""
    e = e + ",\"text\":\""   + json_escape(text)         + "\""
    e = e + "}"
    TinyNN.tnn_events_emit(e)
  end
end
