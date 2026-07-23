# tao#drift-grad-sentinels — opt-in producers for `drift` and `grad`
# events (schema v1, docs/events-schema.md).
#
# Two independent toggles, both driven from the caller-side loop:
#
#   ToyDriftGrad.params(sess) → Array<:ptr>
#     enumerate PARAM-flagged tensors in dispatch order
#
#   ToyDriftGrad.snapshot_one(sess, t) → Mat
#     download one PARAM's current values for use as a step-0 baseline
#
#   ToyDriftGrad.emit_drift_event(sess, t, snap_mat, step, t_now)
#     compute (cos_to_init, l2_to_init) against `snap_mat`, emit one
#     `drift` event
#
#   ToyDriftGrad.emit_grad_event(sess, t, step, t_now)
#     download t's grad, emit one `grad` event with (l2, abs_mean,
#     nan_count) computed via tnn_scratch_*
#
# Both are cheap-when-off — caller-side env guards (TOY_DRIFT_EVERY,
# TOY_GRAD_SENTINELS) gate the calls entirely, so the unconfigured
# path costs nothing beyond an env-read at boot.
#
# Spinel notes:
#   - sp_Mat_ptr_array isn't a generated type, so we keep Mat-typed
#     arrays in main scope (caller manages the snapshot[] array); the
#     module never accepts or returns Array<Mat>.
#   - Use `** 0.5` over Math.sqrt where the receiver could be poly-
#     typed via Spinel's inference cascade (per landmine #12).
#   - Mat round-trip uses tnn_download_to_f64_array → Mat.flat[i],
#     same primitive Mat-roundtrip uses elsewhere in toy.
#   - Param enumeration walks the graph leaves + nodes (same logic as
#     ToyDescribeFlow.build_index but inlined to avoid cross-module
#     dependency).

require_relative "../io/json_builder"

module ToyDriftGrad
  # Walk session graph + leaves, return ordered array of (ptr, name)
  # for every PARAM-flagged tensor (flags bit 4 set). The flag is
  # `ggml_set_param`-marked, so this exactly matches the params the
  # training graph differentiates over.
  def self.params(sess)
    ptrs = [TinyNN.tnn_null_ptr]; ptrs.pop
    seen = [TinyNN.tnn_null_ptr]; seen.pop

    # Collect every distinct tensor reachable from the graph: nodes +
    # their src dependencies. Filter to PARAM-flagged.
    n_nodes = TinyNN.tnn_graph_n_nodes(sess)
    i = 0
    while i < n_nodes
      node = TinyNN.tnn_graph_node(sess, i)
      # The node itself
      seen.push(node)
      # Its srcs (leaves usually carry the PARAM flag)
      si = 0
      while si < 10
        src = TinyNN.tnn_tensor_src(node, si)
        if src == nil || src == TinyNN.tnn_null_ptr
          si = 10
        else
          # Already seen?
          dup = false
          k = 0
          while k < seen.length
            if seen[k] == src
              dup = true
              k = seen.length
            else
              k = k + 1
            end
          end
          if !dup
            seen.push(src)
          end
          si = si + 1
        end
      end
      i = i + 1
    end

    # Filter by PARAM flag.
    i = 0
    while i < seen.length
      t = seen[i]
      flags = TinyNN.tnn_tensor_flags(t)
      if (flags & 4) != 0
        ptrs.push(t)
      end
      i = i + 1
    end
    ptrs
  end

  # Snapshot ONE param's current values into a fresh Mat. Caller
  # keeps these in a main-scope Array<Mat> (a separate per-step Array
  # would force Spinel to declare sp_Mat_ptr_array, which it can't).
  def self.snapshot_one(sess, t)
    n = TinyNN.tnn_tensor_nelements(t)
    m = Mat.new(1, n)
    TinyNN.tnn_download_to_f64_array(sess, t, m.flat, n)
    m
  end

  # Emit one `drift` event for tensor t, comparing against snap_mat.
  # Single-pass cos + L2 loop over the f64 flats.
  def self.emit_drift_event(sess, t, snap_mat, step, t_now)
    n = TinyNN.tnn_tensor_nelements(t)
    cur = Mat.new(1, n)
    TinyNN.tnn_download_to_f64_array(sess, t, cur.flat, n)

    dot = 0.0
    sum_sq_s = 0.0
    sum_sq_c = 0.0
    sum_sq_diff = 0.0
    i = 0
    while i < n
      sv = snap_mat.flat[i]
      cv = cur.flat[i]
      dot = dot + sv * cv
      sum_sq_s = sum_sq_s + sv * sv
      sum_sq_c = sum_sq_c + cv * cv
      d = sv - cv
      sum_sq_diff = sum_sq_diff + d * d
      i = i + 1
    end
    norm_s = sum_sq_s ** 0.5
    norm_c = sum_sq_c ** 0.5
    cos_to_init = 0.0
    if norm_s > 0.0 && norm_c > 0.0
      cos_to_init = dot / (norm_s * norm_c)
    end
    l2_to_init = sum_sq_diff ** 0.5

    name = TinyNN.tnn_tensor_name(t)
    ev = Toy::Json::Builder.new
    ev.add_str("kind",  "drift")
    ev.add_str("phase", "train")
    ev.add_num("t",           t_now)
    ev.add_num("step",        step)
    ev.add_str("param",       name)
    ev.add_num("cos_to_init", cos_to_init)
    ev.add_num("l2_to_init",  l2_to_init)
    TinyNN.tnn_events_emit(ev.dump)
  end

  # Emit one `grad` event for tensor t. Call after tnn_compute_backward
  # but before the next step's reset (so the grad tensor is still live).
  def self.emit_grad_event(sess, t, step, t_now)
    g = TinyNN.tnn_tensor_grad(sess, t)
    if g == nil || g == TinyNN.tnn_null_ptr
      return
    end
    n = TinyNN.tnn_tensor_nelements(g)
    TinyNN.tnn_download(sess, g)
    sum_sq   = TinyNN.tnn_scratch_sum_sq_f32(sess, n)
    sum_abs  = TinyNN.tnn_scratch_sum_abs_f32(sess, n)
    nan      = TinyNN.tnn_scratch_nan_count_f32(sess, n)
    l2       = sum_sq ** 0.5
    abs_mean = sum_abs / n.to_f
    ne0 = TinyNN.tnn_tensor_ne0(g)
    ne1 = TinyNN.tnn_tensor_ne1(g)
    name = TinyNN.tnn_tensor_name(t)
    ev = Toy::Json::Builder.new
    ev.add_str("kind",  "grad")
    ev.add_str("phase", "train")
    ev.add_num("t",         t_now)
    ev.add_num("step",      step)
    ev.add_str("param",     name)
    ev.add_raw("shape",     "[" + ne0.to_s + "," + ne1.to_s + "]")
    ev.add_num("l2",        l2)
    ev.add_num("abs_mean",  abs_mean)
    ev.add_num("nan_count", nan)
    TinyNN.tnn_events_emit(ev.dump)
  end
end
