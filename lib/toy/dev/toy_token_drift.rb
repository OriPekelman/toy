# Per-token embedding drift + corpus frequency (toy#20). Emits one
# `drift` event per vocab row per tick, with the token's L2 + cosine
# vs the step-0 snapshot AND its training-corpus frequency. Tao
# renders this as the freq↔drift figure (the granite_transfer
# Pearson r = -0.835 headline finding).
#
# Caller-driven schedule (TOY_TOKEN_DRIFT=N in the example trainer).
# At vocab=627 (TinyStories) one tick = 627 events; fine. At
# Qwen-class vocab (152k) per-token events would be too noisy — a
# binned-arrays variant should be a separate follow-up.
#
# Spinel notes:
#   - Mat round-trip via tnn_download_to_f64_array. The full embed
#     is a single Mat held in main scope across ticks (sp_Mat_ptr_
#     array isn't generated; one Mat is fine).
#   - File.read + while-loop walk for the corpus. The historical
#     Spinel block-form-File.open + FFI-init crash is fixed
#     (a03bb49); the pattern here is just the natural primitive
#     for "whole file as one string" + monomorphic Array<String>
#     dispatch.

require_relative "../io/toy_json"

module ToyTokenDrift
  # One-time corpus frequency histogram. Returns Array<Int> of
  # length vocab_size where index = token_id, value = occurrence
  # count across all lines in the seqs file. IDs outside [0,vocab)
  # are silently skipped (corpus-vs-vocab drift is the caller's
  # concern).
  def self.corpus_freq(seqs_path, vocab_size)
    counts = [0]; counts.pop
    i = 0
    while i < vocab_size
      counts.push(0)
      i = i + 1
    end
    raw   = File.read(seqs_path)
    lines = raw.split("\n")
    li = 0
    while li < lines.length
      parts = lines[li].split(" ")
      pi = 0
      while pi < parts.length
        tid = parts[pi].to_i
        if tid >= 0 && tid < vocab_size
          counts[tid] = counts[tid] + 1
        end
        pi = pi + 1
      end
      li = li + 1
    end
    counts
  end

  # Snapshot the full embed table into a Mat (held in main scope as
  # the step-0 baseline). Returns Mat of length vocab_size *
  # d_model — the same row-major flat as the tensor's ggml column-
  # major data slot (ne=[d_model, vocab] → flat[row*d_model + d]).
  def self.snapshot(sess, t_embed)
    n = TinyNN.tnn_tensor_nelements(t_embed)
    m = Mat.new(1, n)
    TinyNN.tnn_download_to_f64_array(sess, t_embed, m.flat, n)
    m
  end

  # Emit one `drift` event per vocab row. Each event carries:
  #   param      = "token_embd.weight"
  #   token_id   = the row index
  #   cos_to_init, l2_to_init = per-row metric vs snap_mat
  #   freq       = corpus occurrence count
  def self.emit_per_token(sess, t_embed, snap_mat, freqs,
                            vocab_size, d_model, step, t_now)
    n = vocab_size * d_model
    cur = Mat.new(1, n)
    TinyNN.tnn_download_to_f64_array(sess, t_embed, cur.flat, n)

    row = 0
    while row < vocab_size
      base = row * d_model
      dot = 0.0
      sum_sq_s = 0.0
      sum_sq_c = 0.0
      sum_sq_diff = 0.0
      d = 0
      while d < d_model
        sv = snap_mat.flat[base + d]
        cv = cur.flat[base + d]
        dot = dot + sv * cv
        sum_sq_s = sum_sq_s + sv * sv
        sum_sq_c = sum_sq_c + cv * cv
        diff = sv - cv
        sum_sq_diff = sum_sq_diff + diff * diff
        d = d + 1
      end
      norm_s = sum_sq_s ** 0.5
      norm_c = sum_sq_c ** 0.5
      cos_to_init = 0.0
      if norm_s > 0.0 && norm_c > 0.0
        cos_to_init = dot / (norm_s * norm_c)
      end
      l2_to_init = sum_sq_diff ** 0.5
      freq = 0
      if row < freqs.length
        freq = freqs[row]
      end

      ev = Toy::Json.new
      ev.j_str("kind",  "drift")
      ev.j_str("phase", "train")
      ev.j_num("t",           t_now)
      ev.j_num("step",        step)
      ev.j_str("param",       "token_embd.weight")
      ev.j_num("token_id",    row)
      ev.j_num("cos_to_init", cos_to_init)
      ev.j_num("l2_to_init",  l2_to_init)
      ev.j_num("freq",        freq)
      TinyNN.tnn_events_emit(ev.j_dump)
      row = row + 1
    end
  end
end
