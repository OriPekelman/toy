# tao#kv-describe-flow — DAG introspector over a built compute graph.
#
# Walks the session's primary graph (`graph_a` — populated by
# tnn_build_forward_only or tnn_realize), discovers leaves by scanning
# each node's src tensors for non-node references (= weights, inputs,
# constants), classifies every tensor by role, and emits three forms:
#
#   ToyDescribeFlow.text(sess)    → terminal-readable summary
#   ToyDescribeFlow.mermaid(sess) → mermaid flowchart for md reports
#   ToyDescribeFlow.json(sess)    → structured DAG for Tao's HTML
#
# Acceptance (Tao spec):
#   - round-trips through JSON.parse
#   - text labels weight tensors as `weight`, intermediates as `ephemeral`
#
# Spinel notes: avoids String interpolation, Hash<Ptr,…> (not supported),
# and Math.exp / Array<Hash> patterns. Uses the seed-and-pop idiom for
# typed-pointer arrays (`[TinyNN.tnn_null_ptr]; .pop` per
# docs/spinel-tech-debt-audit-2026-05-25.md landmine #1).

module ToyDescribeFlow
  # ggml type enum (subset toy uses) → human name. Keep in sync with
  # vendor/ggml/include/ggml.h GGML_TYPE_*.
  def self.dtype_name(t)
    if t == 0
      "f32"
    elsif t == 1
      "f16"
    elsif t == 8
      "q8_0"
    elsif t == 9
      "q8_1"
    elsif t == 15
      "q8_k"
    elsif t == 24
      "i8"
    elsif t == 25
      "i16"
    elsif t == 26
      "i32"
    elsif t == 27
      "i64"
    elsif t == 30
      "bf16"
    else
      "type" + t.to_s
    end
  end

  # ggml_tensor.flags bitmask → role string. PARAM > LOSS > OUTPUT >
  # INPUT, then COMPUTE/leaf distinguished by caller.
  def self.flag_role(flags, is_node)
    if (flags & 4) != 0
      "weight"           # PARAM
    elsif (flags & 8) != 0
      "loss"             # LOSS
    elsif (flags & 2) != 0
      "output"           # OUTPUT
    elsif (flags & 1) != 0
      "input"            # INPUT
    elsif is_node
      "ephemeral"
    else
      "constant"
    end
  end

  # Shape as "ne0×ne1×ne2×ne3" trimming trailing 1s.
  def self.shape_string(t)
    n0 = TinyNN.tnn_tensor_ne0(t)
    n1 = TinyNN.tnn_tensor_ne1(t)
    n2 = TinyNN.tnn_tensor_ne2(t)
    n3 = TinyNN.tnn_tensor_ne3(t)
    s = n0.to_s
    if n1 > 1 || n2 > 1 || n3 > 1
      s = s + "×" + n1.to_s
    end
    if n2 > 1 || n3 > 1
      s = s + "×" + n2.to_s
    end
    if n3 > 1
      s = s + "×" + n3.to_s
    end
    s
  end

  # Build the full ptr→index lookup. Returns parallel arrays
  # (ptrs, is_node) — leaves first (is_node=false), then compute nodes
  # (is_node=true). Pure Spinel-friendly arithmetic; no Hash<Ptr, Int>.
  def self.build_index(sess)
    nodes = [TinyNN.tnn_null_ptr]; nodes.pop
    leaves = [TinyNN.tnn_null_ptr]; leaves.pop

    n_nodes = TinyNN.tnn_graph_n_nodes(sess)
    i = 0
    while i < n_nodes
      nodes.push(TinyNN.tnn_graph_node(sess, i))
      i = i + 1
    end

    # For each node's src, check whether it's already known as a node
    # OR already added to leaves; if neither, it's a fresh leaf.
    i = 0
    while i < n_nodes
      node = nodes[i]
      si = 0
      while si < 10
        src = TinyNN.tnn_tensor_src(node, si)
        if src == nil || src == TinyNN.tnn_null_ptr
          si = 10
        else
          # Already a node? Linear scan over `nodes`.
          in_nodes = false
          k = 0
          while k < nodes.length
            if nodes[k] == src
              in_nodes = true
              k = nodes.length
            else
              k = k + 1
            end
          end
          if !in_nodes
            # Already in leaves?
            in_leaves = false
            k = 0
            while k < leaves.length
              if leaves[k] == src
                in_leaves = true
                k = leaves.length
              else
                k = k + 1
              end
            end
            if !in_leaves
              leaves.push(src)
            end
          end
          si = si + 1
        end
      end
      i = i + 1
    end

    # Return (ptrs, n_leafs) where ptrs = leaves + nodes; first n_leafs
    # entries are leaves, the rest are compute nodes.
    ptrs = [TinyNN.tnn_null_ptr]; ptrs.pop
    i = 0; while i < leaves.length; ptrs.push(leaves[i]); i = i + 1; end
    i = 0; while i < nodes.length;  ptrs.push(nodes[i]);  i = i + 1; end
    [ptrs, leaves.length]
  end

  # Pure-string emit for one tensor.
  def self.emit_one(t, is_node)
    name  = TinyNN.tnn_tensor_name(t)
    flags = TinyNN.tnn_tensor_flags(t)
    dtype = TinyNN.tnn_tensor_dtype(t)
    nel   = TinyNN.tnn_tensor_nelements(t)
    nb    = TinyNN.tnn_tensor_nbytes(t)
    op    = TinyNN.tnn_tensor_op_name(t)
    role  = flag_role(flags, is_node)
    shape = shape_string(t)

    j  = "{"
    j = j + "\"name\":\""  + name + "\""
    j = j + ",\"role\":\"" + role + "\""
    j = j + ",\"shape\":\""+ shape + "\""
    j = j + ",\"dtype\":\""+ dtype_name(dtype) + "\""
    j = j + ",\"nelements\":" + nel.to_s
    j = j + ",\"nbytes\":"    + nb.to_s
    if is_node
      j = j + ",\"op\":\""   + op + "\""
    end
    j = j + "}"
    j
  end

  # tao#flow-json-emit (issue #25): write the realized graph's flow.json into
  # `run_dir` (a no-op when run_dir is ""). Called by every training runner right
  # after realize! so the run bundle is self-describing — parallel to events.jsonl
  # — and Tao no longer needs a separate TOY_DESCRIBE realize-only pre-pass. One
  # graph walk, once. Backend-agnostic: ToyDescribeFlow reads the ggml graph
  # structure via CPU TinyNN (same seam the CUDA/Metal checkpoint writer uses).
  def self.emit_flow_json(run_dir, sess)
    if run_dir.length > 0
      File.open(run_dir + "/flow.json", "w") do |ff|
        ff.write(ToyDescribeFlow.json(sess))
      end
    end
  end

  def self.json(sess)
    pair    = build_index(sess)
    ptrs    = pair[0]
    n_leafs = pair[1]

    out = "{\"format\":\"toy/v1\",\"nodes\":["
    i = 0
    while i < ptrs.length
      out = out + (i == 0 ? "" : ",") + emit_one(ptrs[i], i >= n_leafs)
      i = i + 1
    end
    out = out + "],\"summary\":{"
    out = out + "\"n_leafs\":"  + n_leafs.to_s
    out = out + ",\"n_nodes\":" + (ptrs.length - n_leafs).to_s
    out = out + "}}"
    out
  end

  def self.text(sess)
    pair    = build_index(sess)
    ptrs    = pair[0]
    n_leafs = pair[1]
    n_nodes = ptrs.length - n_leafs

    out = ""
    out = out + "DAG over session graph (toy/v1)\n"
    out = out + "  leafs: " + n_leafs.to_s + "   nodes: " + n_nodes.to_s + "\n"
    out = out + "\n"
    out = out + "LEAFS (weights / inputs / constants):\n"
    out = out + "  " + "role".ljust(10) + "dtype".ljust(7) +
                "shape".ljust(20) + "name\n"
    i = 0
    while i < n_leafs
      t = ptrs[i]
      flags = TinyNN.tnn_tensor_flags(t)
      role  = flag_role(flags, false)
      out = out + "  " + role.ljust(10) +
            dtype_name(TinyNN.tnn_tensor_dtype(t)).ljust(7) +
            shape_string(t).ljust(20) +
            TinyNN.tnn_tensor_name(t) + "\n"
      i = i + 1
    end
    out = out + "\n"
    out = out + "NODES (compute ops, in dispatch order):\n"
    out = out + "  " + "idx".ljust(5) + "role".ljust(10) + "dtype".ljust(7) +
                "shape".ljust(20) + "op".ljust(14) + "name\n"
    i = 0
    while i < n_nodes
      t = ptrs[n_leafs + i]
      flags = TinyNN.tnn_tensor_flags(t)
      role  = flag_role(flags, true)
      out = out + "  " + i.to_s.ljust(5) + role.ljust(10) +
            dtype_name(TinyNN.tnn_tensor_dtype(t)).ljust(7) +
            shape_string(t).ljust(20) +
            TinyNN.tnn_tensor_op_name(t).ljust(14) +
            TinyNN.tnn_tensor_name(t) + "\n"
      i = i + 1
    end
    out
  end

  def self.mermaid(sess)
    pair    = build_index(sess)
    ptrs    = pair[0]
    n_leafs = pair[1]
    n_nodes = ptrs.length - n_leafs

    out = "flowchart LR\n"

    i = 0
    while i < ptrs.length
      t = ptrs[i]
      is_node = i >= n_leafs
      flags = TinyNN.tnn_tensor_flags(t)
      role  = flag_role(flags, is_node)
      name  = TinyNN.tnn_tensor_name(t)
      shape = shape_string(t)
      op    = is_node ? TinyNN.tnn_tensor_op_name(t) : ""
      label = name.length > 0 ? name : "(unnamed)"
      if is_node && op.length > 0
        label = label + " " + op
      end
      label = label + " [" + shape + "]"
      glyph = "[" + label + "]"                      # default: ephemeral / output
      if role == "weight"
        glyph = "[[" + label + "]]"                  # subroutine = weight
      elsif role == "input"
        glyph = "([" + label + "])"                  # stadium = input
      elsif role == "constant"
        glyph = "(" + label + ")"                    # round = constant
      end
      out = out + "  t" + i.to_s + glyph + "\n"
      i = i + 1
    end

    # Edges (src → dst) for each compute node.
    i = 0
    while i < n_nodes
      dst_idx = n_leafs + i
      dst = ptrs[dst_idx]
      si = 0
      while si < 10
        src = TinyNN.tnn_tensor_src(dst, si)
        if src == nil || src == TinyNN.tnn_null_ptr
          si = 10
        else
          src_idx = -1
          j = 0
          while j < ptrs.length
            if ptrs[j] == src
              src_idx = j
              j = ptrs.length
            else
              j = j + 1
            end
          end
          if src_idx >= 0
            out = out + "  t" + src_idx.to_s + " --> t" + dst_idx.to_s + "\n"
          end
          si = si + 1
        end
      end
      i = i + 1
    end
    out
  end

  # P1 — derive a Toy::Card from a realized session by walking its
  # compute graph. Structural-exhaustive view (every op, every shape,
  # every tensor name from the graph). Complements the textbook
  # `algorithm` methods on stdlib archs; doesn't replace them.
  #
  # Returns a Toy::Card with:
  # - PARAM leaves → add_param  (name, shape, "")
  # - INPUT leaves → add_input  (name, shape, "")
  # - each compute node → step_bind (var, "OP(src1, src2, …)", shape)
  # - the last OUTPUT-flagged node → step_return
  #
  # Anonymous tensors (empty names) are labeled "_t<index>" so the
  # rendered pseudocode stays unambiguous.
  def self.card(sess)
    c = Toy::Card.new("derived", "structural")

    # Direct-index access to avoid the Array<Array<mixed>> destructure
    # landmine. Same pattern as the existing text/json/mermaid emitters.
    pair    = build_index(sess)
    ptrs    = pair[0]
    n_leafs = pair[1]

    # P1 limitation: building per-node Step records via
    # Card#step_bind with FFI :str args triggers a Spinel poly-cascade
    # landmine that broadens Step's field types and crashes static
    # init. For now, we record counts as hypers so callers get a
    # useful Card (param count, input count, node count) without
    # full per-step detail. Restoring the full step list is a
    # follow-up — either restructure Card to accept loosely-typed
    # builders, or move Step construction to C-side primitives
    # (`tnn_card_push_step`).
    n_param_leafs = 0
    n_input_leafs = 0
    j = 0
    while j < n_leafs
      lflags = TinyNN.tnn_tensor_flags(ptrs[j])
      if (lflags & 4) != 0; n_param_leafs = n_param_leafs + 1; end
      if (lflags & 1) != 0; n_input_leafs = n_input_leafs + 1; end
      j = j + 1
    end
    c.add_hyper("leafs.total",  n_leafs.to_s)
    c.add_hyper("leafs.params", n_param_leafs.to_s)
    c.add_hyper("leafs.inputs", n_input_leafs.to_s)
    c.add_hyper("nodes.compute", (ptrs.length - n_leafs).to_s)

    # Identify the OUTPUT — last OUTPUT-flagged node wins; fallback
    # to the final node.
    out_name = ""
    i = n_leafs
    while i < ptrs.length
      t     = ptrs[i]
      flags = TinyNN.tnn_tensor_flags(t)
      if (flags & 2) != 0
        name = TinyNN.tnn_tensor_name(t)
        if name == ""
          name = "_t" + i.to_s
        end
        out_name = name
      end
      i = i + 1
    end
    if out_name == "" && ptrs.length > n_leafs
      t = ptrs[ptrs.length - 1]
      name = TinyNN.tnn_tensor_name(t)
      if name == ""
        name = "_t" + (ptrs.length - 1).to_s
      end
      out_name = name
    end
    if out_name != ""
      c.step_return(out_name)
    end

    c
  end

  # P1 — count the PARAM tensors in the session graph. Used by the
  # derivation smoke as a cross-check against ToyDriftGrad.params.
  def self.param_count(sess)
    n = 0
    pair    = build_index(sess)
    ptrs    = pair[0]
    n_leafs = pair[1]
    i = 0
    while i < n_leafs
      flags = TinyNN.tnn_tensor_flags(ptrs[i])
      if (flags & 4) != 0
        n = n + 1
      end
      i = i + 1
    end
    n
  end
end
