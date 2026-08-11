# lib/toy/io/toy_gnn_task.rb — the graph + node-classification task
# behind the toy#153 (DFA-arch T1) GNN lane.
#
# Holds ONE type for both graph sources — a seeded synthetic graph
# (`build_synthetic!`) and a real citation graph read off disk
# (`load_bundle!`, what prep/fetch_cora.rb writes). Two classes here
# would be two Spinel compilation-unit types for one shape; one class
# with two builders keeps the engine and the runner monomorphic.
#
# ── WHY A SYNTHETIC GRAPH IS THE DEFAULT (and Cora is still available) ──
#
# toy#153 names Cora/CiteSeer/PubMed. Those are reachable and the lane
# reads them (`GNN_GRAPH=data/gnn_cora`), but the DEFAULT is a seeded
# contextual stochastic block model, for the same reason toy#152's
# anchor is a random teacher rather than gaussian blobs
# ([[control-arm-must-be-able-to-lose]]): the success bar tao#19 made
# MANDATORY is
#
#     positive = all-DFA within the stated gap of all-BP
#                AND provably beating the FROZEN control
#
# and in a GNN the frozen control is unusually strong, because
# NEIGHBOURHOOD AGGREGATION IS ARCHITECTURE, NOT LEARNING. A frozen
# random hidden stack still smooths features over the graph, so on a
# task whose labels are recoverable by smoothing alone (a plain block
# model: label == community) the frozen arm ties and the bar is
# unfalsifiable by construction. That degenerate task is selectable
# (`GNN_TASK=community`) precisely so its degeneracy is a MEASURED fact
# and not an assumption.
#
# The default (`GNN_TASK=teacher`) draws labels from a fixed random
# 2-layer GNN teacher over the same graph — propagate, random weights,
# ReLU, propagate, random weights, argmax. Structure is still
# load-bearing (the teacher aggregates), but so are the hidden layers,
# so BP can beat the frozen control and the bar can be met or missed on
# the merits.
#
# ── THE ADJACENCY IS KEPT SPARSE ON THE HOST, DENSE ONLY IN THE GRAPH ──
#
# `adj_dense` materialises the GCN-normalised S-hat = D^-1/2 (A+I) D^-1/2
# as a dense [N, N] tensor because tinynn has no sparse matmul. But
# every HOST-side propagation (the teacher, and the pre-propagated input
# features) walks the EDGE LIST instead: on Cora dense host propagation
# of the 1433-dim features would be 2708^2 x 1433 = 10.5 G scalar ops in
# a Spinel loop, and the sparse walk is E x F = 18 M. Same numbers, four
# orders of magnitude apart.
#
# Spinel hygiene: plain class, no-arg ctor, no default args, no Struct,
# while loops, typed-empty array seeds, no #{} interpolation.

class GnnTask
  KIND_TEACHER   = 0
  KIND_COMMUNITY = 1

  SPLIT_TRAIN = 0
  SPLIT_VAL   = 1

  attr_accessor :gt_nodes, :gt_feat_dim, :gt_classes,
                :gt_edge_a, :gt_edge_b, :gt_deg,
                :gt_feat, :gt_label, :gt_split, :gt_comm,
                :gt_n_train, :gt_n_val, :gt_s

  def initialize
    @gt_nodes    = 0
    @gt_feat_dim = 0
    @gt_classes  = 0
    @gt_edge_a = [0]; @gt_edge_a.pop
    @gt_edge_b = [0]; @gt_edge_b.pop
    @gt_deg    = [0]; @gt_deg.pop
    @gt_feat   = [0.0]; @gt_feat.pop
    @gt_label  = [0]; @gt_label.pop
    @gt_split  = [0]; @gt_split.pop
    @gt_comm   = [0]; @gt_comm.pop
    @gt_n_train = 0
    @gt_n_val   = 0
    @gt_s       = [0]
  end

  # ---- builder 1: the seeded contextual stochastic block model ----
  #
  # `homophily` is the share of edges that stay inside a community and
  # `degree` the target average degree; p_in/p_out are DERIVED from them
  # so the two knobs a reader cares about are the two knobs that exist.
  def build_synthetic!(kind, n_nodes, feat_dim, n_classes, degree,
                       homophily, feat_signal, teacher_dim, task_seed)
    @gt_nodes    = n_nodes
    @gt_feat_dim = feat_dim
    @gt_classes  = n_classes
    @gt_s[0]     = lcg_seed_state(task_seed)

    # Communities: balanced by construction (i % C after a fixed
    # shuffle) rather than by a draw, so a small graph cannot land with
    # an empty community and silently change what the task is.
    @gt_comm = Array.new(n_nodes, 0)
    i = 0
    while i < n_nodes
      @gt_comm[i] = i % n_classes
      i = i + 1
    end
    shuffle_int!(@gt_comm, n_nodes)

    # p_in / p_out from (degree, homophily). Expected edge count is
    # n*degree/2; the intra-community pair count is ~ n^2/(2C) and the
    # inter one ~ n^2 (1 - 1/C) / 2.
    nf = n_nodes.to_f
    cf = n_classes.to_f
    p_in  = homophily * degree.to_f * cf / nf
    p_out = (1.0 - homophily) * degree.to_f / (nf * (1.0 - 1.0 / cf))
    if p_in > 1.0;  p_in  = 1.0; end
    if p_out > 1.0; p_out = 1.0; end

    @gt_deg = Array.new(n_nodes, 0)
    a = 0
    while a < n_nodes
      b = a + 1
      while b < n_nodes
        p = @gt_comm[a] == @gt_comm[b] ? p_in : p_out
        if next_u < p
          @gt_edge_a.push(a)
          @gt_edge_b.push(b)
          @gt_deg[a] = @gt_deg[a] + 1
          @gt_deg[b] = @gt_deg[b] + 1
        end
        b = b + 1
      end
      a = a + 1
    end

    # Features: a per-community mean at `feat_signal` scale plus unit
    # gaussian noise. At the default 0.5 a single node's own features
    # are weak evidence and the neighbourhood is what sharpens them —
    # which is the regime a GNN exists for.
    mu = [0.0]; mu.pop
    mi = 0
    while mi < n_classes * feat_dim
      mu.push(gauss)
      mi = mi + 1
    end
    @gt_feat = Array.new(n_nodes * feat_dim, 0.0)
    ni = 0
    while ni < n_nodes
      base = ni * feat_dim
      mbase = @gt_comm[ni] * feat_dim
      fj = 0
      while fj < feat_dim
        @gt_feat[base + fj] = feat_signal * mu[mbase + fj] + gauss
        fj = fj + 1
      end
      ni = ni + 1
    end

    if kind == KIND_COMMUNITY
      @gt_label = Array.new(n_nodes, 0)
      li = 0
      while li < n_nodes
        @gt_label[li] = @gt_comm[li]
        li = li + 1
      end
    else
      @gt_label = teacher_labels(teacher_dim)
    end
    nil
  end

  # ---- builder 2: a graph bundle on disk (prep/fetch_cora.rb) ----
  #
  # Five little-endian files under one prefix. Deliberately NOT a dense
  # adjacency on disk: the edge list is what the host propagation walks,
  # and a dense Cora adjacency would be a 29 MB regenerable artefact.
  #
  #   <prefix>.meta.i32   [n_nodes, feat_dim, n_classes, n_edges]
  #   <prefix>.feat.f32   n_nodes * feat_dim, node-major
  #   <prefix>.edge.i32   2 * n_edges (a, b) — undirected, listed once
  #   <prefix>.label.i32  n_nodes
  #   <prefix>.split.i32  n_nodes (0 = train, 1 = val)
  def load_bundle!(prefix)
    meta = Array.new(4, 0)
    got = TinyNN.tnn_read_i32_file(prefix + ".meta.i32", 0, 4, meta)
    if got != 4
      puts "gnn: could not read " + prefix + ".meta.i32 (rc=" + got.to_s + ")"
      return 1
    end
    @gt_nodes    = meta[0]
    @gt_feat_dim = meta[1]
    @gt_classes  = meta[2]
    n_edges      = meta[3]
    if @gt_nodes < 2 || @gt_feat_dim < 1 || @gt_classes < 2 || n_edges < 1
      puts "gnn: bundle meta is degenerate: nodes=" + @gt_nodes.to_s +
           " feat=" + @gt_feat_dim.to_s + " classes=" + @gt_classes.to_s +
           " edges=" + n_edges.to_s
      return 1
    end

    nf = @gt_nodes * @gt_feat_dim
    @gt_feat = Array.new(nf, 0.0)
    gf = TinyNN.tnn_read_f32_file(prefix + ".feat.f32", 0, nf, @gt_feat)
    if gf != nf
      puts "gnn: " + prefix + ".feat.f32 short: got " + gf.to_s +
           " want " + nf.to_s
      return 1
    end

    ebuf = Array.new(n_edges * 2, 0)
    ge = TinyNN.tnn_read_i32_file(prefix + ".edge.i32", 0, n_edges * 2, ebuf)
    if ge != n_edges * 2
      puts "gnn: " + prefix + ".edge.i32 short: got " + ge.to_s +
           " want " + (n_edges * 2).to_s
      return 1
    end
    @gt_deg = Array.new(@gt_nodes, 0)
    ei = 0
    while ei < n_edges
      a = ebuf[ei * 2]
      b = ebuf[ei * 2 + 1]
      if a < 0 || a >= @gt_nodes || b < 0 || b >= @gt_nodes
        puts "gnn: edge " + ei.to_s + " out of range: " + a.to_s + "," + b.to_s
        return 1
      end
      @gt_edge_a.push(a)
      @gt_edge_b.push(b)
      @gt_deg[a] = @gt_deg[a] + 1
      @gt_deg[b] = @gt_deg[b] + 1
      ei = ei + 1
    end

    @gt_label = Array.new(@gt_nodes, 0)
    gl = TinyNN.tnn_read_i32_file(prefix + ".label.i32", 0, @gt_nodes, @gt_label)
    if gl != @gt_nodes
      puts "gnn: " + prefix + ".label.i32 short: got " + gl.to_s
      return 1
    end
    @gt_split = Array.new(@gt_nodes, 0)
    gs = TinyNN.tnn_read_i32_file(prefix + ".split.i32", 0, @gt_nodes, @gt_split)
    if gs != @gt_nodes
      puts "gnn: " + prefix + ".split.i32 short: got " + gs.to_s
      return 1
    end
    @gt_comm = Array.new(@gt_nodes, 0)
    ci = 0
    while ci < @gt_nodes
      @gt_comm[ci] = @gt_label[ci]
      ci = ci + 1
    end
    count_split!
    0
  end

  # ---- the transductive split: N per class train, the rest val ----
  #
  # Planetoid's shape (a handful of labels per class, everything else
  # held out) and the reason this lane is a semi-supervised one: with
  # 20 labels per class most of the graph carries NO error term at all,
  # which is exactly the gap `--dfa-feedback structure` exists to fill.
  def build_split!(train_per_class)
    @gt_split = Array.new(@gt_nodes, SPLIT_VAL)
    seen = Array.new(@gt_classes, 0)
    i = 0
    while i < @gt_nodes
      c = @gt_label[i]
      if seen[c] < train_per_class
        @gt_split[i] = SPLIT_TRAIN
        seen[c] = seen[c] + 1
      end
      i = i + 1
    end
    # never-mask: a class that could not fill its quota means the arms
    # are being compared on a training set that is not what was asked
    # for. Say so and stop.
    ci = 0
    while ci < @gt_classes
      if seen[ci] < train_per_class
        puts "gnn: class " + ci.to_s + " has only " + seen[ci].to_s +
             " nodes but GNN_TRAIN_PER_CLASS=" + train_per_class.to_s +
             " — the label distribution cannot support this split"
        return 1
      end
      ci = ci + 1
    end
    count_split!
    if @gt_n_val < 1
      puts "gnn: the split left no validation nodes"
      return 1
    end
    0
  end

  def count_split!
    @gt_n_train = 0
    @gt_n_val   = 0
    i = 0
    while i < @gt_nodes
      if @gt_split[i] == SPLIT_TRAIN
        @gt_n_train = @gt_n_train + 1
      else
        @gt_n_val = @gt_n_val + 1
      end
      i = i + 1
    end
    nil
  end

  def train_indices
    out = [0]; out.pop
    i = 0
    while i < @gt_nodes
      if @gt_split[i] == SPLIT_TRAIN
        out.push(i)
      end
      i = i + 1
    end
    out
  end

  # ---- derived tensors ----

  # S-hat = D^-1/2 (A + I) D^-1/2, dense and row-major: flat[i*N + j] is
  # S-hat[i][j]. Symmetric, so the same buffer serves both the forward
  # propagation and the structure-aware feedback.
  def adj_dense
    n = @gt_nodes
    s = Array.new(n * n, 0.0)
    k = 0
    while k < n
      s[k * n + k] = 1.0 / (@gt_deg[k] + 1).to_f
      k = k + 1
    end
    e = 0
    while e < @gt_edge_a.length
      a = @gt_edge_a[e]
      b = @gt_edge_b[e]
      w = 1.0 / Math.sqrt(((@gt_deg[a] + 1) * (@gt_deg[b] + 1)).to_f)
      s[a * n + b] = w
      s[b * n + a] = w
      e = e + 1
    end
    s
  end

  # One host-side propagation over the EDGE LIST: out = S-hat * src,
  # with src laid out node-major (node i's `dim` values at i*dim).
  def propagate(src, dim)
    n = @gt_nodes
    out = Array.new(n * dim, 0.0)
    i = 0
    while i < n
      w = 1.0 / (@gt_deg[i] + 1).to_f
      b = i * dim
      j = 0
      while j < dim
        out[b + j] = w * src[b + j]
        j = j + 1
      end
      i = i + 1
    end
    e = 0
    while e < @gt_edge_a.length
      ea = @gt_edge_a[e]
      eb = @gt_edge_b[e]
      w2 = 1.0 / Math.sqrt(((@gt_deg[ea] + 1) * (@gt_deg[eb] + 1)).to_f)
      ba = ea * dim
      bb = eb * dim
      j2 = 0
      while j2 < dim
        out[ba + j2] = out[ba + j2] + w2 * src[bb + j2]
        out[bb + j2] = out[bb + j2] + w2 * src[ba + j2]
        j2 = j2 + 1
      end
      e = e + 1
    end
    out
  end

  # The graph's INPUT tensor. The first propagation is folded into the
  # host preprocessing because S-hat and X are both constant, so
  # S-hat(XW) == (S-hat X)W is an exact identity, and doing it once here
  # keeps an N^2 x feat_dim matmul out of every single training step.
  # Every LATER propagation is in the compute graph, at hidden width.
  def propagated_features
    propagate(@gt_feat, @gt_feat_dim)
  end

  # ---- the random 2-layer GNN teacher ----
  #
  # propagate -> W1 -> ReLU -> propagate -> W2 -> argmax, i.e. the same
  # architecture family the student is, which is what makes the hidden
  # layers load-bearing. Logits are CENTRED per class across nodes
  # before the argmax: a raw random teacher's argmax is badly
  # unbalanced, and an unbalanced label set silently turns "accuracy"
  # into "how often did you say the majority class".
  def teacher_labels(teacher_dim)
    n = @gt_nodes
    f = @gt_feat_dim
    c = @gt_classes
    w1 = [0.0]; w1.pop
    sc1 = 1.0 / Math.sqrt(f.to_f)
    i = 0
    while i < f * teacher_dim
      w1.push(gauss * sc1)
      i = i + 1
    end
    w2 = [0.0]; w2.pop
    sc2 = 1.0 / Math.sqrt(teacher_dim.to_f)
    j = 0
    while j < teacher_dim * c
      w2.push(gauss * sc2)
      j = j + 1
    end

    p1 = propagate(@gt_feat, f)
    hid = Array.new(n * teacher_dim, 0.0)
    ni = 0
    while ni < n
      pb = ni * f
      hb = ni * teacher_dim
      t = 0
      while t < teacher_dim
        acc = 0.0
        k = 0
        while k < f
          acc = acc + p1[pb + k] * w1[k * teacher_dim + t]
          k = k + 1
        end
        hid[hb + t] = acc > 0.0 ? acc : 0.0
        t = t + 1
      end
      ni = ni + 1
    end
    p2 = propagate(hid, teacher_dim)

    logits = Array.new(n * c, 0.0)
    mi = 0
    while mi < n
      pb2 = mi * teacher_dim
      lb  = mi * c
      cj = 0
      while cj < c
        acc2 = 0.0
        t2 = 0
        while t2 < teacher_dim
          acc2 = acc2 + p2[pb2 + t2] * w2[t2 * c + cj]
          t2 = t2 + 1
        end
        logits[lb + cj] = acc2
        cj = cj + 1
      end
      mi = mi + 1
    end
    # per-class centring
    cc = 0
    while cc < c
      sum = 0.0
      si = 0
      while si < n
        sum = sum + logits[si * c + cc]
        si = si + 1
      end
      mean = sum / n.to_f
      sj = 0
      while sj < n
        logits[sj * c + cc] = logits[sj * c + cc] - mean
        sj = sj + 1
      end
      cc = cc + 1
    end

    out = Array.new(n, 0)
    oi = 0
    while oi < n
      base = oi * c
      best = 0
      bv = logits[base]
      ck = 1
      while ck < c
        if logits[base + ck] > bv
          bv = logits[base + ck]
          best = ck
        end
        ck = ck + 1
      end
      out[oi] = best
      oi = oi + 1
    end
    out
  end

  # ---- deterministic stream (the tree-wide 31-bit LCG; toy#114) ----

  def shuffle_int!(arr, n)
    i = n - 1
    while i > 0
      j = next_int(i + 1)
      t = arr[i]
      arr[i] = arr[j]
      arr[j] = t
      i = i - 1
    end
    nil
  end

  def lcg_seed_state(seed)
    s = ((seed + 104729) * 2654435761) % 2147483647
    if s <= 0
      s = seed + 104729
    end
    w = 0
    while w < 8
      s = (s * 1103515245 + 12345) & 0x7FFFFFFF
      w = w + 1
    end
    s
  end

  def next_u
    s = @gt_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @gt_s[0] = s
    (s.to_f + 1.0) / 2147483648.0
  end

  def next_int(n)
    s = @gt_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @gt_s[0] = s
    (s >> 8) % n
  end

  def gauss
    u1 = next_u
    u2 = next_u
    if u1 < 1.0e-12; u1 = 1.0e-12; end
    Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
  end
end
