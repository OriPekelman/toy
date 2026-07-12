# tao#kv-tap-surface — named activation taps for observability.
#
# Usage from a training entry point:
#
#   ToyTap.emit(sess, "attn_q_post_rope", t_q, 0, -1, step, t_now)
#                  └─ region label       └─ tensor          └─ optional layer/head
#
# Per Tao spec, regions are user-named strings (suggested
# convention: `<sublayer>_<phase>`, e.g. attn_q_post_rope,
# ffn_gate_pre_silu, embed_post_lookup). Stable across runs so cross-
# run comparisons work.
#
# Cheap-when-off: caller-side env guard (TOY_TAP=1) gates the calls.
# When TOY_TAP isn't set, the example skips the emit entirely — zero
# cost beyond an env-read.
#
# Stats: l2, abs_mean, nan_count by default; per_head_l2 when n_heads
# is passed > 0 (slices the flat into n_heads chunks and L2's each).
# Schema fields match docs/events-schema.md §tap.
#
# Spinel-friendly: no symbols-as-args, no keyword args, no Array<Mat>.
# All stats are scalars; per_head_l2 is built as a JSON array string
# via concatenation in this module.

require_relative "../io/json_builder"

module ToyTap
  # Single tap event. Pass layer=-1 / head=-1 to omit those fields
  # (null in the JSON). n_heads>0 enables per_head_l2 (n must be
  # divisible by n_heads).
  def self.emit(sess, region, tensor, layer, head, step, t_now, n_heads)
    if tensor == nil || tensor == TinyNN.tnn_null_ptr
      return
    end
    n = TinyNN.tnn_tensor_nelements(tensor)
    TinyNN.tnn_download(sess, tensor)
    sum_sq  = TinyNN.tnn_scratch_sum_sq_f32(sess, n)
    sum_abs = TinyNN.tnn_scratch_sum_abs_f32(sess, n)
    nan_n   = TinyNN.tnn_scratch_nan_count_f32(sess, n)
    l2       = sum_sq ** 0.5
    abs_mean = sum_abs / n.to_f
    ne0 = TinyNN.tnn_tensor_ne0(tensor)
    ne1 = TinyNN.tnn_tensor_ne1(tensor)
    dtype = TinyNN.tnn_tensor_dtype(tensor)
    dn = "f32"
    if dtype == 1
      dn = "f16"
    elsif dtype == 8
      dn = "q8_0"
    elsif dtype == 26
      dn = "i32"
    elsif dtype == 30
      dn = "bf16"
    end

    ev = Toy::Json::Builder.new
    ev.add_str("kind",  "tap")
    ev.add_str("phase", "train")
    ev.add_num("t",      t_now)
    ev.add_num("step",   step)
    ev.add_str("region", region)
    if layer >= 0
      ev.add_num("layer", layer)
    else
      ev.add_raw("layer", "null")
    end
    if head >= 0
      ev.add_num("head", head)
    else
      ev.add_raw("head", "null")
    end
    ev.add_raw("shape",     "[" + ne0.to_s + "," + ne1.to_s + "]")
    ev.add_str("dtype",     dn)
    ev.add_num("l2",        l2)
    ev.add_num("abs_mean",  abs_mean)
    ev.add_num("nan_count", nan_n)

    # per_head_l2: download to f64 Mat, split into n_heads contiguous
    # chunks, L2 each. Costs n*8 bytes of host traffic + an O(n) loop
    # — comparable to the scratch-stat path; we avoid adding a
    # tnn_scratch_*_range C primitive to keep the FFI surface minimal.
    if n_heads > 0 && n_heads <= n && (n % n_heads) == 0
      per = n / n_heads
      buf = Mat.new(1, n)
      TinyNN.tnn_download_to_f64_array(sess, tensor, buf.flat, n)
      phl2 = "["
      h = 0
      while h < n_heads
        sq = 0.0
        i = 0
        while i < per
          v = buf.flat[h * per + i]
          sq = sq + v * v
          i = i + 1
        end
        l2h = sq ** 0.5
        phl2 = phl2 + (h == 0 ? "" : ",") + l2h.to_s
        h = h + 1
      end
      phl2 = phl2 + "]"
      ev.add_raw("per_head_l2", phl2)
    end

    TinyNN.tnn_events_emit(ev.dump)
  end

  # GH#15 — Activation-Gram tap for linear CKA. For activation A of
  # shape [d, T] in ggml column-major, computes G = Aᵀ·A (T×T) and
  # emits it as a `gram` field on the tap event alongside the standard
  # scalar stats. T×T fits easily in JSON for typical context lengths
  # (e.g. T=32 → 32×32 = 1024 floats = a few KB per event).
  #
  # Region naming convention is the issue's: attn_q_post_rope,
  # ffn_out, resid_post_block, etc. — user-chosen + stable across runs.
  def self.emit_cka(sess, region, tensor, layer, head, step, t_now)
    if tensor == nil || tensor == TinyNN.tnn_null_ptr
      return
    end
    d = TinyNN.tnn_tensor_ne0(tensor)
    t = TinyNN.tnn_tensor_ne1(tensor)
    if d <= 0 || t <= 0
      return
    end
    n = d * t
    TinyNN.tnn_download(sess, tensor)
    sum_sq  = TinyNN.tnn_scratch_sum_sq_f32(sess, n)
    sum_abs = TinyNN.tnn_scratch_sum_abs_f32(sess, n)
    nan_n   = TinyNN.tnn_scratch_nan_count_f32(sess, n)
    l2       = sum_sq ** 0.5
    abs_mean = sum_abs / n.to_f

    # Gram G = Aᵀ·A. A is column-major [d, T] → flat[i + j*d] = A[i,j].
    # G[i, j] = sum_k A[k, i] * A[k, j] = sum_k flat[k + i*d] * flat[k + j*d]
    # Cost: T*T*d ops. For T=32, d=64: 65k. For T=128, d=1024: 16M.
    buf = Mat.new(1, n)
    TinyNN.tnn_download_to_f64_array(sess, tensor, buf.flat, n)

    ev = Toy::Json::Builder.new
    ev.add_str("kind",  "tap")
    ev.add_str("phase", "train")
    ev.add_num("t",      t_now)
    ev.add_num("step",   step)
    ev.add_str("region", region)
    if layer >= 0
      ev.add_num("layer", layer)
    else
      ev.add_raw("layer", "null")
    end
    if head >= 0
      ev.add_num("head", head)
    else
      ev.add_raw("head", "null")
    end
    ev.add_raw("shape",     "[" + d.to_s + "," + t.to_s + "]")
    ev.add_str("dtype",     "f32")
    ev.add_num("l2",        l2)
    ev.add_num("abs_mean",  abs_mean)
    ev.add_num("nan_count", nan_n)

    gram = "["
    i = 0
    while i < t
      gram = gram + (i == 0 ? "[" : ",[")
      j = 0
      while j < t
        s = 0.0
        k = 0
        while k < d
          s = s + buf.flat[k + i * d] * buf.flat[k + j * d]
          k = k + 1
        end
        gram = gram + (j == 0 ? "" : ",") + s.to_s
        j = j + 1
      end
      gram = gram + "]"
      i = i + 1
    end
    gram = gram + "]"
    ev.add_raw("gram", gram)

    TinyNN.tnn_events_emit(ev.dump)
  end
end
