#!/usr/bin/env ruby
# frozen_string_literal: true

# prep/decode_fused_gate.rb — toy#206 gate: the fused multi-head decode
# step and batched prompt evaluation.
#
# The KV engine has two decode graphs. The per-head one builds a small node
# chain per attention head; the fused one carries every head of a layer in
# one tensor and can evaluate many prompt tokens in one step. The fused
# graph is the default wherever the model is eligible. This gate holds:
#
#   same      fused ids == per-head ids (KV_FUSED=0)
#   batch     ids do not depend on how the prompt is cut into batches
#             (KV_BATCH=1, 7, default)
#   graph     the fused graph really is the small one, and KV_FUSED=0 really
#             is the per-head one (nodes per step, from TNN_STEP_PROFILE);
#             the prompt really went in batches (step counts)
#   chain     greedy decoding is self-consistent: generating k tokens from P
#             equals generating one token at a time from the growing prompt.
#             (toy#206 also fixed `generate` feeding the last prompt token
#             twice; that bug breaks this leg.)
#   fallback  configurations the fused step does not cover still run, on the
#             per-head graph: KV_Q8=1, and a legacy-layout GGUF (weights
#             copied, not mapped); toy-infer-trace still runs (cpu)
#
# Run on data/smollm2-135m-tok.gguf (GQA 9/3, no bias) and, when present,
# data/qwen25-0.5b-native-q8.gguf (GQA, QKV bias, Q8 weights).
#
#   ruby prep/decode_fused_gate.rb [--device cuda]   # exit 0 pass, 1 fail, 2 unusable

require "open3"

ROOT   = File.expand_path("..", __dir__)
DEVICE = (i = ARGV.index("--device")) ? ARGV[i + 1].to_s : "cpu"
TARGET = DEVICE == "cuda" ? "libexec/toy-infer-cuda" : "libexec/toy-infer"
RUNNER = ENV["TOY_GATE_RUNNER"] || File.join(ROOT, TARGET)
TRACE  = File.join(ROOT, "libexec", "toy-infer-trace")
LABEL  = DEVICE == "cuda" ? "decode-fused-cuda" : "decode-fused"
SMOL   = File.join(ROOT, "data", "smollm2-135m-tok.gguf")   # native (mmap) layout
LEGACY = File.join(ROOT, "data", "smollm2-135m-f32.gguf")   # legacy copy-load layout
QWEN   = File.join(ROOT, "data", "qwen25-0.5b-native-q8.gguf")

unless %w[cpu cuda].include?(DEVICE)
  warn "decode_fused_gate: --device must be cpu or cuda"
  exit 2
end
[SMOL, LEGACY].each do |f|
  next if File.file?(f)
  warn "decode_fused_gate: missing fixture #{f} (never skipped silently)"
  exit 2
end
targets = []
targets << TARGET unless ENV["TOY_GATE_RUNNER"]
targets << "libexec/toy-infer-trace" if DEVICE == "cpu"
targets.each do |t|
  out, st = Open3.capture2e("make", "-C", ROOT, t)
  next if st.success?
  warn "decode_fused_gate: `make #{t}` failed:\n#{out.lines.last(15).join}"
  exit 2
end

N_PROMPT = 48
N_NEW    = 24
PROMPT   = (1..N_PROMPT).map { |i| 200 + (i * 37) % 4000 }.freeze

Run = Struct.new(:ids, :steps, :nodes)

def infer(gguf, prompt, n_new, env = {})
  e = { "GGUF" => gguf, "PROMPT_IDS" => prompt.join(" "), "N_NEW" => n_new.to_s,
        "CTX" => "256", "TNN_STEP_PROFILE" => "1" }.merge(env)
  out, st = Open3.capture2e(e, RUNNER, chdir: ROOT)
  raise "runner exited #{st.exitstatus} (#{env.inspect}): #{out.lines.last.to_s.strip}" unless st.success?
  ids = out[/^ids: (.*)$/, 1]
  raise "no ids line (#{env.inspect})" if ids.nil?
  prof = out[/\[tnn-step-profile\] steps=(\d+) nodes\/step=(\d+)/]
  raise "no step profile line (#{env.inspect})" if prof.nil?
  Run.new(ids.split.map(&:to_i), Regexp.last_match(1).to_i, Regexp.last_match(2).to_i)
end

failures = []
ok = ->(leg, what) { puts "  ok: #{leg} — #{what}" }

def check_model(name, gguf, failures, ok)
  fused   = infer(gguf, PROMPT, N_NEW)
  perhead = infer(gguf, PROMPT, N_NEW, "KV_FUSED" => "0")
  b1      = infer(gguf, PROMPT, N_NEW, "KV_BATCH" => "1")
  b7      = infer(gguf, PROMPT, N_NEW, "KV_BATCH" => "7")
  gen     = fused.ids.length - N_PROMPT
  if gen < 2
    failures << "#{name}: only #{gen} token(s) generated; the legs need at least 2"
    return
  end

  # same
  if fused.ids == perhead.ids
    ok.call("#{name} same", "fused and per-head graphs give the same #{gen} tokens")
  else
    at = (0...fused.ids.length).find { |i| fused.ids[i] != perhead.ids[i] }
    failures << "#{name} same: fused and per-head ids differ from index #{at} " \
                "(#{fused.ids[at].inspect} vs #{perhead.ids[at].inspect})"
  end

  # batch
  if b1.ids == fused.ids && b7.ids == fused.ids
    ok.call("#{name} batch", "KV_BATCH=1, 7 and the default give the same ids")
  else
    failures << "#{name} batch: ids depend on the batch size " \
                "(KV_BATCH=1 #{b1.ids == fused.ids ? 'same' : 'differs'}, " \
                "KV_BATCH=7 #{b7.ids == fused.ids ? 'same' : 'differs'})"
  end

  # graph
  g = []
  g << "fused graph has #{fused.nodes} nodes/step, per-head #{perhead.nodes}; expected under a third" unless fused.nodes * 3 < perhead.nodes
  want = { "default" => [fused, 1 + gen - 1], "KV_BATCH=7" => [b7, (N_PROMPT / 7.0).ceil + gen - 1],
           "KV_BATCH=1" => [b1, N_PROMPT + gen - 1], "KV_FUSED=0" => [perhead, N_PROMPT + gen - 1] }
  want.each do |k, (run, steps)|
    g << "#{k}: #{run.steps} decode steps, expected #{steps}" unless run.steps == steps
  end
  if g.empty?
    ok.call("#{name} graph", "#{fused.nodes} nodes/step fused vs #{perhead.nodes} per-head; " \
                             "#{N_PROMPT}-token prompt in 1 step (per-head: #{N_PROMPT})")
  else
    g.each { |m| failures << "#{name} graph: #{m}" }
  end

  # chain
  k = [gen, 4].min
  whole = infer(gguf, PROMPT, k).ids
  grown = PROMPT.dup
  k.times { grown = infer(gguf, grown, 1).ids }
  if whole == grown
    ok.call("#{name} chain", "#{k} tokens at once == one token at a time from the growing prompt")
  else
    failures << "#{name} chain: #{k} tokens at once #{whole.last(k).inspect} != " \
                "one at a time #{grown.last(k).inspect}"
  end
end

begin
  check_model("smollm2", SMOL, failures, ok)
  if File.file?(QWEN)
    check_model("qwen", QWEN, failures, ok)
  else
    puts "  not run: qwen — #{QWEN} is absent (GQA + QKV-bias + Q8 leg)"
  end

  # fallback
  fb = []
  q8 = infer(SMOL, PROMPT, 4, "KV_Q8" => "1")
  ph = infer(SMOL, PROMPT, 4, "KV_FUSED" => "0")
  fb << "KV_Q8=1 ran a #{q8.nodes}-node graph; expected a per-head-sized one" unless q8.nodes * 2 > ph.nodes
  fb << "KV_Q8=1 produced #{q8.ids.length - N_PROMPT} tokens, expected 4" unless q8.ids.length == N_PROMPT + 4
  lg = infer(LEGACY, PROMPT, 4)
  fb << "the legacy-layout file ran a #{lg.nodes}-node graph; expected a per-head-sized one" unless lg.nodes * 2 > ph.nodes
  fb << "the legacy-layout file gives different ids than the native one" unless lg.ids == ph.ids
  if DEVICE == "cpu"
    out, st = Open3.capture2e({ "GGUF" => SMOL, "PROMPT_IDS" => PROMPT.first(3).join(" ") }, TRACE, chdir: ROOT)
    fb << "toy-infer-trace exited #{st.exitstatus}: #{out.lines.last.to_s.strip}" unless st.success?
    fb << "toy-infer-trace printed no trace taps" unless out.include?("rn1")
  end
  if fb.empty?
    ok.call("fallback", "KV_Q8=1 and a legacy-layout file run on the per-head graph" + (DEVICE == "cpu" ? "; toy-infer-trace runs" : ""))
  else
    fb.each { |m| failures << "fallback: #{m}" }
  end
rescue StandardError => e
  failures << "gate aborted: #{e.message.lines.first.to_s.strip}"
end

if failures.empty?
  puts "GATE PASS [#{LABEL}]: fused decode and batched prompt evaluation match the per-head graph, on #{DEVICE}"
  exit 0
end
failures.each { |m| puts "  FAIL: #{m}" }
puts "GATE FAIL [#{LABEL}]: #{failures.length} failure(s)"
exit 1
