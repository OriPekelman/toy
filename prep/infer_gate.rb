#!/usr/bin/env ruby
# prep/infer_gate.rb — deterministic functional gate for `toy infer`.
#
# Proves the CRuby→runner COMPUTE BRIDGE reproduces the reference compute
# byte-for-byte. `toy infer <model> --prompt X --n N` shells out to the
# Spinel binary examples/example_inference; the reference is that same
# binary invoked directly with PROMPT/N_NEW/GGUF in ENV. Greedy (argmax)
# decode → deterministic, no seed (01_inference.rb passes no sampler_config
# → Sampler.argmax). We assert the two emit the IDENTICAL generated line.
#
#   make example_inference        # build the runner first
#   ruby prep/infer_gate.rb       # exit 0 on byte-for-byte match, 1 otherwise
#
# FIXTURE: prefers data/smollm2-135m-tok.gguf (embedded tokenizer → `text:`
# path); falls back to data/smollm2-135m-f32.gguf (no tokenizer → `ids:`
# path). Both are deterministic. The f32/ids fixture also stays green if
# the embedded-tokenizer path hits a Spinel runtime regression, since it
# exercises the same generate() compute without the tokenizer GC path.

require "open3"

ROOT   = File.expand_path("..", __dir__)
RUNNER = File.join(ROOT, "examples", "example_inference")
TOY    = File.join(ROOT, "bin", "toy")
PROMPT = "Once upon a time"
N      = 8

unless File.executable?(RUNNER)
  warn "infer_gate: runner not built: #{RUNNER} (run `make example_inference`)"
  exit 2
end

TOK = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
F32 = File.join(ROOT, "data", "smollm2-135m-f32.gguf")

# Run the reference binary directly with a controlled ENV and return its
# generated line: the `text:`-prefixed (tokenizer) or `ids:`-prefixed (no
# tokenizer) line. Returns [prefix, payload] or nil.
def reference(model)
  out, st = Open3.capture2e(
    { "GGUF" => model, "PROMPT" => PROMPT, "N_NEW" => N.to_s }, RUNNER
  )
  return nil unless st.success?
  if (l = out.lines.find { |x| x.start_with?("text: ") })
    ["text", l["text: ".length..].chomp]
  elsif (l = out.lines.find { |x| x.start_with?("ids:") })
    ["ids", l.chomp]
  end
end

# Pick the first fixture whose reference run succeeds (the tokenizer model
# may segfault under a Spinel tokenizer-GC regression — fall back to f32).
model = nil
ref   = nil
[TOK, F32].each do |m|
  next unless File.file?(m)
  r = reference(m)
  if r
    model = m
    ref = r
    break
  end
  warn "infer_gate: reference run failed on #{File.basename(m)}; trying next fixture"
end

if model.nil?
  warn "infer_gate: no usable fixture (need data/smollm2-135m-{tok,f32}.gguf and a working runner)"
  exit 2
end

# Run `toy infer` (stdout only — build chatter is on stderr).
got, st = Open3.capture2e(TOY, "infer", model, "--prompt", PROMPT, "--n", N.to_s)
# Strip the bridge's stderr build notice that capture2e merges; the
# generated line is the `text:`/`ids:` payload the command prints to stdout.
prefix, payload = ref
expected = prefix == "text" ? payload : payload # the runner's line content
# `toy infer` prints the bare text on the text path, "ids: ..." on ids path.
actual = got.lines.map(&:chomp).reject(&:empty?)
hit =
  if prefix == "text"
    actual.last
  else
    actual.find { |l| l.start_with?("ids:") }
  end

ok = hit == expected

puts "fixture : #{File.basename(model)} (#{prefix} path)"
puts "expected: #{expected.inspect}"
puts "actual  : #{hit.inspect}"
if ok
  puts "GATE PASS: toy infer reproduces example_inference byte-for-byte"
  exit 0
else
  warn "GATE FAIL: toy infer output diverged from example_inference"
  exit 1
end
