#!/usr/bin/env ruby
# prep/eval_gate.rb — deterministic functional gate for `toy eval`.
#
# Proves the CRuby→runner COMPUTE BRIDGE reproduces a RECORDED BASELINE
# byte-for-byte. `toy eval <model> --top-k K` builds the lib-side Spinel
# runner libexec/toy-eval (source lib/toy/run/eval.rb) and shells out to it.
# The eval point is frozen (prefill ids=[1,100,200], logprobs at pos=3) and
# the compute is a pure CPU f32 forward + log_softmax + manual top-K → no
# sampler, no seed, deterministic. We assert the gathered "logprob:" block
# equals the committed baseline in prep/fixtures/eval_baseline.txt.
#
#   ruby prep/eval_gate.rb       # exit 0 on byte-for-byte match, 1 otherwise
#
# `toy eval` builds libexec/toy-eval itself (ToyRoot.ensure_built), so no
# separate `make` is needed first — only bin/toy must be present.
#
# FIXTURE: data/smollm2-135m-f32.gguf (no tokenizer needed; logprobs work on
# raw IDs). The baseline file is keyed by fixture basename; the stored value
# is the full ordered TOP_K-line block, \n-escaped. We require the fixture +
# its baseline record (fail loud, never silently skip).

require "open3"

ROOT     = File.expand_path("..", __dir__)
TOY      = File.join(ROOT, "bin", "toy")
BASELINE = File.join(ROOT, "prep", "fixtures", "eval_baseline.txt")
TOP_K    = 5

unless File.executable?(TOY)
  warn "eval_gate: bin/toy not executable: #{TOY}"
  exit 2
end
unless File.file?(BASELINE)
  warn "eval_gate: missing recorded baseline: #{BASELINE}"
  exit 2
end

# Load the recorded baseline: fixture-basename → expected block (decoded).
expected = {}
File.foreach(BASELINE) do |line|
  next if line.strip.empty? || line.start_with?("#")
  key, want = line.chomp.split("\t", 2)
  expected[key] = want.gsub("\\n", "\n") if key && want
end
if expected.empty?
  warn "eval_gate: baseline file has no records: #{BASELINE}"
  exit 2
end

F32 = File.join(ROOT, "data", "smollm2-135m-f32.gguf")

# Run `toy eval` (stdout only — build chatter is on stderr). Returns the
# ordered "logprob:" block joined with "\n", or nil on failure.
def run_eval(model, device: nil)
  argv = [TOY, "eval", model, "--top-k", TOP_K.to_s]
  argv += ["--device", device] if device
  got, st = Open3.capture2e(*argv)
  return nil unless st.success?
  lines = got.lines.map(&:chomp).select { |l| l.start_with?("logprob:") }
  return nil if lines.empty?
  lines.join("\n")
end

# gx10 (aarch64-linux) is the CANONICAL gate platform: the recorded FLOAT
# baseline is byte-exact-reproducible there (deterministic). On other platforms
# (e.g. macOS) CPU libm differs at ~1e-6, so the float text won't byte-match.
# There we gate the DISCRETE invariant — the top-k id ORDER — byte-exact, plus
# the logprob floats within FLOAT_TOL, and note that the float baseline is
# gx10-pinned. The canonical (Linux) path is UNCHANGED (strict byte-exact).
# TOY_GATE_FORCE_PORTABLE=1 forces the non-canonical arm (for testing it here).
CANONICAL = RUBY_PLATFORM.include?("linux") && ENV["TOY_GATE_FORCE_PORTABLE"] != "1"
FLOAT_TOL = 1.0e-3  # ~400x the observed cross-libm drift, ~100x below any real-model-bug shift

# Compare two "logprob:" blocks. Canonical: byte-exact. Non-canonical: top-k id
# ORDER byte-exact + floats within FLOAT_TOL. Returns [ok, note_or_nil].
def cmp_logprobs(got, want)
  return [true, nil] if got == want
  g = got.to_s.lines.map(&:split)
  w = want.to_s.lines.map(&:split)
  return [false, "line-count differs (#{g.length} vs #{w.length})"] unless !g.empty? && g.length == w.length
  gids = g.map { |t| t[1] }; wids = w.map { |t| t[1] }
  return [false, "top-k id ORDER differs: #{gids.inspect} vs #{wids.inspect}"] unless gids == wids
  maxabs = 0.0
  g.each_index { |i| d = (g[i][2].to_f - w[i][2].to_f).abs; maxabs = d if d > maxabs }
  return [false, "logprob float drift #{maxabs} exceeds tol #{FLOAT_TOL}"] if maxabs > FLOAT_TOL
  [true, "non-canonical #{RUBY_PLATFORM}: top-k id order identical; logprob floats within #{FLOAT_TOL} (max #{maxabs}); float baseline is gx10-canonical"]
end

ran = 0
failures = []
[F32].each do |m|
  next unless File.file?(m)
  base = File.basename(m)
  want = expected[base]
  unless want
    warn "eval_gate: fixture #{base} present but no baseline record; skipping"
    next
  end
  got = run_eval(m)
  ran += 1
  ok, note = CANONICAL ? [got == want, nil] : cmp_logprobs(got, want)
  puts "fixture : #{base} (logprob path, top-#{TOP_K})"
  puts "expected: #{want.inspect}"
  puts "actual  : #{got.inspect}"
  puts "note    : #{note}" if note
  failures << base unless ok
end

if ran == 0
  warn "eval_gate: no usable fixture (need data/smollm2-135m-f32.gguf with a baseline record)"
  exit 2
end

# --- CUDA parity arm (additive, env-gated) ---------------------------------
# Default `ruby prep/eval_gate.rb` stays cpu-only/portable. When TOY_GATE_CUDA=1
# (on gx10: `make gate-cuda`), ALSO run `--device cuda` and assert the top-K
# token-id ORDERING is IDENTICAL to cpu IN THE SAME RUN. The INVARIANT is the
# discrete id sequence (descending logprob order) — NOT the float text: CUDA
# accumulates in F32, so logprobs can differ from the CPU f64 path at ~F32
# precision (~1e-6 relative). We compare the float text too: literally equal →
# "BIT-IDENTICAL"; ids equal but floats differ → record explicitly
# "top-k-order-identical, logprobs differ in float text on N of K ranks". An
# id-ORDER divergence (incl. an F32-induced swap across a near-tie) → FAIL, both
# sequences reported. We do NOT loosen to a tolerance.
def parse_ids_and_vals(block)
  ids  = []
  vals = []
  block.each_line do |l|
    l = l.chomp
    next unless l.start_with?("logprob:")
    body = l[("logprob:".length)..].strip
    id_s, val_s = body.split(" ", 2)
    ids  << id_s.to_i
    vals << val_s
  end
  [ids, vals]
end

if ENV["TOY_GATE_CUDA"] == "1"
  puts
  puts "--- CUDA parity arm (TOY_GATE_CUDA=1) ---"
  cuda_failures = []
  cuda_ran = 0
  [F32].each do |m|
    next unless File.file?(m)
    base = File.basename(m)
    cpu_block  = run_eval(m)
    cuda_block = run_eval(m, device: "cuda")
    cuda_ran += 1
    if cpu_block.nil? || cuda_block.nil?
      puts "fixture : #{base}"
      puts "verdict : eval cuda FAILED to run"
      cuda_failures << base
      next
    end
    cpu_ids,  cpu_vals  = parse_ids_and_vals(cpu_block)
    cuda_ids, cuda_vals = parse_ids_and_vals(cuda_block)
    puts "fixture : #{base} (top-#{TOP_K})"
    puts "cpu  ids: #{cpu_ids.inspect}"
    puts "cuda ids: #{cuda_ids.inspect}"
    if cpu_ids == cuda_ids
      if cpu_block == cuda_block
        puts "verdict : eval cuda BIT-IDENTICAL (#{base})"
      else
        n_diff = 0
        cpu_vals.each_index { |i| n_diff += 1 if cpu_vals[i] != cuda_vals[i] }
        puts "verdict : eval cuda top-k-order-identical, logprobs differ in float text on #{n_diff}/#{TOP_K} ranks (CUDA F32 vs CPU f64) (#{base})"
      end
    else
      puts "verdict : eval cuda top-k id ORDER DIVERGED (#{base})"
      puts "cpu  vals: #{cpu_vals.inspect}"
      puts "cuda vals: #{cuda_vals.inspect}"
      cuda_failures << base
    end
  end
  if cuda_ran == 0
    warn "eval_gate[cuda]: no usable fixture for CUDA parity arm"
    exit 2
  end
  unless cuda_failures.empty?
    warn "GATE FAIL (cuda): toy eval --device cuda top-k id order diverged on: #{cuda_failures.join(', ')}"
    exit 1
  end
  puts "GATE PASS (cuda): toy eval --device cuda top-k id ordering identical to cpu (#{cuda_ran} fixture(s))"
end

if failures.empty?
  puts "GATE PASS: toy eval reproduces recorded baseline byte-for-byte (#{ran} fixture(s))"
  exit 0
else
  warn "GATE FAIL: toy eval diverged from recorded baseline on: #{failures.join(', ')}"
  exit 1
end
