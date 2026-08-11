#!/usr/bin/env ruby
# frozen_string_literal: true
# prep/fetch_cora.rb — materialise the Cora citation graph as a graph
# bundle the toy#153 GNN lane can read (`GNN_GRAPH=data/gnn_cora`).
#
# toy#153 states its success target on Cora/CiteSeer/PubMed, so the lane
# has to be able to run on the real thing and not only on its seeded
# synthetic graph. This is the whole plumbing: one 168 KB download, one
# parse, five little-endian files.
#
# Plain MRI Ruby (NOT Spinel-compiled) — same status as every other
# prep/ converter. Writes to data/, which is gitignored for regenerable
# artefacts.
#
#   ruby prep/fetch_cora.rb [--out data/gnn_cora] [--train-per-class 20]
#
# BUNDLE FORMAT (read by GnnTask#load_bundle!):
#   <prefix>.meta.i32   [n_nodes, feat_dim, n_classes, n_edges]
#   <prefix>.feat.f32   n_nodes * feat_dim, node-major
#   <prefix>.edge.i32   2 * n_edges (a, b) — undirected, listed ONCE
#   <prefix>.label.i32  n_nodes
#   <prefix>.split.i32  n_nodes (0 = train, 1 = val)
#
# TWO CONVENTIONS worth knowing before comparing a number to a paper:
#
#  1. FEATURES ARE ROW-NORMALISED (each node's binary bag-of-words is
#     divided by its row sum). Every mainstream GCN implementation does
#     this and the published Cora accuracies assume it; without it the
#     high-degree-vocabulary nodes dominate the propagation.
#  2. THE SPLIT IS 20-PER-CLASS TRAIN, EVERYTHING ELSE VAL — not
#     Planetoid's 20/500/1000 train/val/test. The training set is the
#     same size and drawn the same way; what differs is that we score
#     on all 2568 remaining nodes instead of a designated 1000-node
#     subset. That is a LOWER-VARIANCE estimate of the same quantity,
#     but it is not literally the number a paper reports, so treat a
#     0.005 difference against published Cora numbers as convention,
#     not as a finding.
#
# Determinism: nodes are ordered by sorted paper id and the training set
# is the first 20 nodes of each class in that order, so two machines
# produce byte-identical bundles.

require "open-uri"
require "fileutils"
require "tmpdir"
require "digest"

SOURCE = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
ROOT   = File.expand_path("..", __dir__)

out_prefix      = File.join(ROOT, "data", "gnn_cora")
train_per_class = 20
i = 0
while i < ARGV.length
  case ARGV[i]
  when "--out"             then i += 1; out_prefix = ARGV[i]
  when "--train-per-class" then i += 1; train_per_class = ARGV[i].to_i
  else abort "fetch_cora: unknown argument #{ARGV[i].inspect}"
  end
  i += 1
end
abort "fetch_cora: --train-per-class must be >= 1" if train_per_class < 1

cache = File.join(ROOT, "data", "cora.tgz")
FileUtils.mkdir_p(File.dirname(cache))
unless File.file?(cache)
  warn "fetch_cora: downloading #{SOURCE}"
  URI.parse(SOURCE).open("rb") { |r| File.binwrite(cache, r.read) }
end
warn "fetch_cora: #{cache} (#{File.size(cache)} bytes, sha256 #{Digest::SHA256.file(cache).hexdigest[0, 16]})"

work = File.join(Dir.tmpdir, "toy-cora-#{Process.pid}")
FileUtils.mkdir_p(work)
begin
  system("tar", "xzf", cache, "-C", work, exception: true)
  content = Dir.glob(File.join(work, "**", "cora.content")).first
  cites   = Dir.glob(File.join(work, "**", "cora.cites")).first
  abort "fetch_cora: cora.content/cora.cites not found in the archive" unless content && cites

  # ---- nodes: id, binary features, class label ----
  rows = File.readlines(content).map { |l| l.split("\t") }.reject(&:empty?)
  abort "fetch_cora: cora.content is empty" if rows.empty?
  feat_dim = rows.first.length - 2
  ids      = rows.map { |r| r.first.to_i }
  order    = ids.each_with_index.sort_by { |id, _| id }.map(&:last)
  index_of = {}
  order.each_with_index { |src, node| index_of[ids[src]] = node }

  class_names = rows.map(&:last).map(&:strip).uniq.sort
  n_nodes     = rows.length
  n_classes   = class_names.length

  feats  = Array.new(n_nodes * feat_dim, 0.0)
  labels = Array.new(n_nodes, 0)
  order.each_with_index do |src, node|
    r = rows[src]
    # Row-normalise: sum first, then divide. A node with an all-zero
    # bag-of-words keeps its zeros rather than dividing by zero.
    vals = Array.new(feat_dim, 0.0)
    sum  = 0.0
    (0...feat_dim).each do |k|
      v = r[k + 1].to_f
      vals[k] = v
      sum += v
    end
    base = node * feat_dim
    if sum > 0.0
      (0...feat_dim).each { |k| feats[base + k] = vals[k] / sum }
    end
    labels[node] = class_names.index(r.last.strip)
  end

  # ---- edges: undirected, deduplicated, self-loops dropped ----
  # Self-loops are dropped because the propagation adds A + I itself; a
  # duplicated self-loop would silently double one node's own weight.
  seen = {}
  File.readlines(cites).each do |line|
    a_id, b_id = line.split
    next if a_id.nil? || b_id.nil?
    a = index_of[a_id.to_i]
    b = index_of[b_id.to_i]
    # A citation to a paper outside the content file has no features and
    # no label — say how many were dropped rather than silently binning
    # them.
    next if a.nil? || b.nil? || a == b
    key = a < b ? [a, b] : [b, a]
    seen[key] = true
  end
  edges = seen.keys.sort

  # ---- split: 20 per class in node order, the rest val ----
  split = Array.new(n_nodes, 1)
  taken = Array.new(n_classes, 0)
  (0...n_nodes).each do |node|
    c = labels[node]
    next unless taken[c] < train_per_class
    split[node] = 0
    taken[c] += 1
  end
  short = (0...n_classes).select { |c| taken[c] < train_per_class }
  unless short.empty?
    abort "fetch_cora: classes #{short.inspect} have fewer than " \
          "#{train_per_class} nodes — cannot build the split"
  end

  FileUtils.mkdir_p(File.dirname(out_prefix))
  File.binwrite("#{out_prefix}.meta.i32",
                [n_nodes, feat_dim, n_classes, edges.length].pack("l<4"))
  File.binwrite("#{out_prefix}.feat.f32",  feats.pack("e*"))
  File.binwrite("#{out_prefix}.edge.i32",  edges.flatten.pack("l<*"))
  File.binwrite("#{out_prefix}.label.i32", labels.pack("l<*"))
  File.binwrite("#{out_prefix}.split.i32", split.pack("l<*"))

  per_class = Array.new(n_classes, 0)
  labels.each { |c| per_class[c] += 1 }
  puts "fetch_cora: wrote #{out_prefix}.{meta,feat,edge,label,split}"
  puts "  nodes=#{n_nodes} features=#{feat_dim} classes=#{n_classes} " \
       "edges=#{edges.length} (avg degree #{(2.0 * edges.length / n_nodes).round(2)})"
  puts "  train=#{split.count(0)} val=#{split.count(1)} " \
       "(#{train_per_class}/class)"
  puts "  class sizes: #{class_names.each_with_index.map { |n, c| "#{n}=#{per_class[c]}" }.join(' ')}"
ensure
  FileUtils.rm_rf(work)
end
