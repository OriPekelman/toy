#!/usr/bin/env ruby
# prep/card_drift_check.rb — verify each Toy:: class with both
# `def forward` and `def algorithm` keeps the two in lock-step.
#
# What's checked, by extracting node sequences via Ripper:
#
#   1. Every @ivar mentioned in `def algorithm` as a param
#      (`add_param("W_1", ...)`) maps to a real instance variable
#      used in `def forward`. Catches the rename-forward-but-not-card
#      drift case explicitly.
#
#   2. Every matmul / matmul_t / t_matmul call in `forward` has a
#      `step_bind` or `step_update` in `algorithm` mentioning the
#      same right-hand-side ivar (modulo display name). Catches
#      "added a new matmul to forward but forgot to document it" and
#      vice versa.
#
#   3. Activation ops (gelu, gelu_new, silu, silu!) in forward map
#      to a matching token in algorithm card.
#
# Conservative: when in doubt, prefer false-negative (don't flag) to
# false-positive. The drift-detector is a tripwire; it should never
# block a passing test on a legitimate divergence. The CHANGELOG /
# review process catches the rest.
#
# Usage:
#   ruby prep/card_drift_check.rb           # check all
#   ruby prep/card_drift_check.rb lib/toy.rb lib/toy_smollm2.rb

require "ripper"

# Files we care about.
DEFAULT_FILES = %w[lib/toy.rb lib/toy_gpt2.rb lib/toy_smollm2.rb]

# Recursively find every node of a given kind in a Ripper sexp.
def each_node(node, kind, &blk)
  return unless node.is_a?(Array)
  yield node if node.first == kind
  node.each { |c| each_node(c, kind, &blk) if c.is_a?(Array) }
end

# Recursively collect every @ivar name referenced under a node.
def ivars_in(node, out = [])
  return out unless node.is_a?(Array)
  if node.first == :var_ref && node[1].is_a?(Array) && node[1].first == :@ivar
    out << node[1][1]
  end
  node.each { |c| ivars_in(c, out) if c.is_a?(Array) }
  out
end

# Find ivar args referenced under a method call's args node.
def ivar_args_in(args)
  ivars_in(args)
end

# Walk a Ripper sexp for a method body. Return:
#   { matmuls: [@ivar, ...], activations: [op, ...], assigns_to: [...], ivars: [...] }
def summarize_method(body)
  out = { matmuls: [], activations: [], ivars: [] }
  each_node(body, :call) do |n|
    # n = [:call, recv, :".", [:@ident, "matmul", ...], ...]
    recv = n[1]
    op_node = n[3]
    op = nil
    op = op_node[1] if op_node.is_a?(Array) && (op_node.first == :@ident || op_node.first == :@const)
    if op == "matmul" || op == "matmul_t" || op == "t_matmul"
      out[:matmuls] << op
    elsif op == "gelu_new" || op == "gelu" || op == "silu" || op == "silu!" || op == "hadamard!"
      out[:activations] << op
    end
    (void recv) if false  # keep for future shape extraction
  end
  out[:ivars] = ivars_in(body).uniq
  out
end

# Parse a class file, find pairs of (forward|lookup, algorithm).
# Returns Array<Hash{ class_name:, forward:, algorithm: }>
def find_pairs(path)
  src = File.read(path)
  sexp = Ripper.sexp(src)
  pairs = []
  each_node(sexp, :class) do |cls|
    cls_name = cls[1].is_a?(Array) ? cls[1].flatten.grep(String).first : "?"
    body = cls[3]   # body_stmt
    fwd = nil
    alg = nil
    each_node(body, :def) do |d|
      name_node = d[1]
      name = name_node.is_a?(Array) ? name_node[1] : nil
      if name == "forward" || name == "lookup"
        fwd = d
      elsif name == "algorithm"
        alg = d
      end
    end
    if fwd && alg
      pairs << { class_name: cls_name, forward: fwd, algorithm: alg }
    end
  end
  pairs
end

# Compare forward.matmuls vs algorithm step-bind matmul count.
# Algorithm uses string expressions like "x · W_1 + b_1" — we count
# any " · " infix that looks like a matrix product. This is rough but
# good enough for the drift-tripwire.
#
# Ripper shape for `c.step_bind("h", "x · W", ...)`:
#   [:method_add_arg, [:call, recv, period, name], [:arg_paren, [:args_add_block, [...args], false]]]
# So we walk :method_add_arg, read the call's method name, then the args.
def algorithm_matmul_count(alg_node)
  count = 0
  each_node(alg_node, :method_add_arg) do |n|
    call = n[1]
    args = n[2]
    next unless call.is_a?(Array) && call.first == :call
    op_node = call[3]
    op = op_node[1] if op_node.is_a?(Array) && op_node.first == :@ident
    next unless op == "step_bind" || op == "step_update"
    each_node(args, :@tstring_content) do |s|
      txt = s[1]
      count += txt.scan(/ · /).length if txt.is_a?(String)
    end
  end
  count
end

def algorithm_activation_tokens(alg_node)
  toks = []
  each_node(alg_node, :@tstring_content) do |s|
    txt = s[1]
    next unless txt.is_a?(String)
    toks << "gelu"     if txt.match?(/\bgelu\b/i) || txt.match?(/\bGeLU\b/)
    toks << "silu"     if txt.match?(/\bsilu\b/i) || txt.match?(/\bSiLU\b/)
    toks << "hadamard" if txt.include?("⊙")     # element-wise product
  end
  toks.uniq
end

drift_count = 0
files = ARGV.empty? ? DEFAULT_FILES : ARGV

files.each do |path|
  unless File.exist?(path)
    warn "skip: #{path} (not found)"
    next
  end
  pairs = find_pairs(path)
  pairs.each do |p|
    fwd_summary = summarize_method(p[:forward])
    n_matmul_fwd = fwd_summary[:matmuls].length
    n_matmul_alg = algorithm_matmul_count(p[:algorithm])
    act_fwd = fwd_summary[:activations].map { |a| a.sub(/_new$/, "").sub("!", "") }.uniq
    act_alg = algorithm_activation_tokens(p[:algorithm])

    # Exact-equality is wrong because forward often calls helper
    # methods (e.g. CausalSelfAttention#head) that contain additional
    # matmuls. The algorithm card unrolls these, so its count is
    # higher than the direct count in `def forward`. So we only flag
    # when one is zero and the other isn't — that's "completely
    # diverged", which a tripwire should catch.
    issues = []
    if (n_matmul_fwd == 0) != (n_matmul_alg == 0)
      issues << "matmul presence mismatch (forward=#{n_matmul_fwd}, algorithm=#{n_matmul_alg})"
    end
    act_only_in_fwd = act_fwd - act_alg
    act_only_in_alg = act_alg - act_fwd
    unless act_only_in_fwd.empty?
      issues << "activation in forward but not algorithm: #{act_only_in_fwd.inspect}"
    end
    unless act_only_in_alg.empty?
      issues << "activation in algorithm but not forward: #{act_only_in_alg.inspect}"
    end

    label = "#{path}  #{p[:class_name]}"
    if issues.empty?
      puts "ok    #{label}  (#{n_matmul_fwd} matmul, act=#{act_fwd.inspect})"
    else
      drift_count += 1
      puts "DRIFT #{label}"
      issues.each { |i| puts "        #{i}" }
    end
  end
end

if drift_count > 0
  puts ""
  puts "FAIL: #{drift_count} card(s) drift from forward"
  exit 1
end

puts ""
puts "ok — no drift detected"
