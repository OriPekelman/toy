#!/usr/bin/env ruby
# prep/metal_gate.rb — Metal RUNTIME parity gate (macOS ONLY).
#
# THIS is the gate that actually validates `--device metal`. gx10 (Linux, no
# Apple frameworks) source-wires metal and verifies syntax/codegen/structural-
# parity/clean-error/no-regression, but CANNOT build or run the metal binaries.
# This harness MUST be run ON THE MAC, on the metal-source-wiring branch, after
# `make setup-ggml-metal` + the three libexec/toy-*-metal builds.
#
# Three parity arms (each mirrors its cuda gate):
#   1. INFER  — cpu vs --device metal, generated token-id line BYTE-EQUAL
#               (discrete greedy argmax → byte-equal IS bit-identical). NEVER
#               compared against the committed float fixture.
#   2. EVAL   — cpu vs --device metal, top-K id ORDER equality (PRIMARY). If
#               float text also byte-equal → BIT-IDENTICAL; if ids equal but
#               floats differ → top-k-order-identical (Metal F32 vs CPU f64),
#               still PASS. id-ORDER divergence → FAIL. No tolerance loosening.
#   3. TRAIN  — from-scratch --device metal: (a) run-twice byte-determinism
#               (or a Mac-pinned fixture if present), (b) loss decreases,
#               (c) checkpoint round-trips through CPU `toy infer` vs the
#               SHARED ckpt_roundtrip_baseline.txt, (d) events.jsonl
#               run_start-first / run_end-last.
#
#   ruby prep/metal_gate.rb     # exit 0 all-pass; non-zero → do NOT merge
#
# On non-macOS this SKIPS GREEN (exit 0) BEFORE any bin/toy call, so gx10 /
# Linux CI do not false-fail. (A gate that can't run skips green; the metal
# BUILD targets, by contrast, echo macOS-only + exit 1.)

if RUBY_PLATFORM !~ /darwin/
  puts "SKIP [metal-gate]: Metal is macOS-only; platform #{RUBY_PLATFORM} cannot build/run libexec/toy-*-metal. (gx10 source-wires metal but cannot runtime-gate it — run on the Mac.)"
  exit 0
end

require "open3"
require "json"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")

unless File.executable?(TOY)
  warn "metal_gate: bin/toy not executable: #{TOY}"
  exit 2
end

TOK = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
F32 = File.join(ROOT, "data", "smollm2-135m-f32.gguf")
N   = 8

# --- shared helpers (mirror prep/infer_gate.rb + prep/eval_gate.rb) ---------

# Run `toy infer`. Returns the generated id/text line, re-prefixed for the
# text path exactly like prep/infer_gate.rb#run_infer.
def run_infer(model, device: nil)
  argv = [TOY, "infer", model, "--n", N.to_s]
  if File.basename(model) == "smollm2-135m-f32.gguf"
    argv += ["--prompt-ids", "6403 1980 253 655 28"]
  else
    argv += ["--prompt", "Once upon a time"]
  end
  argv += ["--device", device] if device
  got, st = Open3.capture2e(*argv, chdir: ROOT)
  return nil unless st.success?
  lines = got.lines.map(&:chomp).reject(&:empty?)
  ids = lines.find { |l| l.start_with?("ids:") }
  return ids if ids
  last = lines.last
  last ? "text: #{last}" : nil
end

def run_eval(model, device: nil)
  argv = [TOY, "eval", model, "--top-k", "5"]
  argv += ["--device", device] if device
  got, st = Open3.capture2e(*argv, chdir: ROOT)
  return nil unless st.success?
  got
end

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

# ============================================================================
# ARM 1 — INFER (cpu vs metal, byte-equal ids line, same run)
# ============================================================================
puts "--- ARM 1: infer (cpu vs metal) ---"
infer_ran = 0
[TOK, F32].each do |m|
  next unless File.file?(m)
  base = File.basename(m)
  cpu_line   = run_infer(m)
  metal_line = run_infer(m, device: "metal")
  infer_ran += 1
  puts "fixture : #{base}"
  puts "cpu     : #{cpu_line.inspect}"
  puts "metal   : #{metal_line.inspect}"
  if cpu_line.nil? || metal_line.nil?
    warn "GATE FAIL [metal-infer]: a run failed to produce output (#{base})"
    exit 1
  end
  # The ids/text line carries no floats → byte-equal here IS bit-identical.
  unless cpu_line == metal_line
    warn "GATE FAIL [metal-infer]: metal token IDs diverged from cpu (#{base})"
    exit 1
  end
  puts "verdict : infer metal BIT-IDENTICAL (#{base})"
end
if infer_ran == 0
  warn "metal_gate: no infer fixture present (need data/smollm2-135m-{tok,f32}.gguf)"
  exit 2
end
puts "GATE PASS [metal-infer]"

# ============================================================================
# ARM 2 — EVAL (cpu vs metal, top-K id ORDER equality)
# ============================================================================
puts
puts "--- ARM 2: eval (cpu vs metal) ---"
unless File.file?(F32)
  warn "metal_gate: eval fixture missing: #{F32}"
  exit 2
end
cpu_block   = run_eval(F32)
metal_block = run_eval(F32, device: "metal")
if cpu_block.nil? || metal_block.nil?
  warn "GATE FAIL [metal-eval]: a run failed to produce output"
  exit 1
end
cpu_ids,  cpu_vals  = parse_ids_and_vals(cpu_block)
metal_ids, metal_vals = parse_ids_and_vals(metal_block)
puts "cpu   ids: #{cpu_ids.inspect}"
puts "metal ids: #{metal_ids.inspect}"
if cpu_ids != metal_ids
  warn "GATE FAIL [metal-eval]: top-k id ORDER diverged"
  warn "  cpu   vals: #{cpu_vals.inspect}"
  warn "  metal vals: #{metal_vals.inspect}"
  exit 1
end
if cpu_block == metal_block
  puts "verdict : eval metal BIT-IDENTICAL"
else
  n_diff = 0
  cpu_vals.each_index { |i| n_diff += 1 if cpu_vals[i] != metal_vals[i] }
  puts "verdict : eval metal top-k-order-identical, logprobs differ in float text on #{n_diff}/#{cpu_vals.length} ranks (Metal F32 vs CPU f64)"
end
puts "GATE PASS [metal-eval]"

# ============================================================================
# ARM 3 — TRAIN-FROM-SCRATCH (byte-determinism, loss-decrease, ckpt RT, events)
# ============================================================================
puts
puts "--- ARM 3: train from-scratch (metal) ---"

def train_metal_curve
  out, status = Open3.capture2e(TOY, "train", "from-scratch", "--device", "metal",
                                "--steps", "5", "--seed", "0", chdir: ROOT)
  [out, status]
end

# (a) byte-determinism: prefer a Mac-pinned fixture; else run-twice self-consistency.
PINNED = File.join(ROOT, "prep", "fixtures", "train_metal_baseline.txt")
out1, st1 = train_metal_curve
unless st1.success?
  warn "GATE FAIL [metal-train]: `toy train --device metal` exited #{st1.exitstatus}:"
  warn out1.lines.last(20).join
  exit 1
end
curve1 = out1.lines.map(&:chomp).select { |l| l.start_with?("step ") }
if curve1.empty?
  warn "GATE FAIL [metal-train]: no `step ` lines in metal train output"
  exit 1
end

if File.file?(PINNED)
  expected = []
  File.foreach(PINNED) do |line|
    next if line.strip.empty? || line.start_with?("#")
    expected << line.chomp if line.start_with?("step ")
  end
  puts "determinism: comparing against PINNED fixture #{PINNED}"
  if curve1 != expected
    warn "GATE FAIL [metal-train]: loss curve diverged from pinned Mac baseline"
    warn "  expected:"; expected.each { |l| warn "    #{l}" }
    warn "  actual  :"; curve1.each  { |l| warn "    #{l}" }
    exit 1
  end
  puts "verdict : metal train byte-equals pinned baseline"
else
  puts "determinism: NO pinned fixture — using RUN-TWICE self-consistency"
  out2, st2 = train_metal_curve
  unless st2.success?
    warn "GATE FAIL [metal-train]: second `toy train --device metal` exited #{st2.exitstatus}"
    exit 1
  end
  curve2 = out2.lines.map(&:chomp).select { |l| l.start_with?("step ") }
  if curve1 != curve2
    warn "GATE FAIL [metal-train]: metal train NOT byte-deterministic run-to-run"
    warn "  run1:"; curve1.each { |l| warn "    #{l}" }
    warn "  run2:"; curve2.each { |l| warn "    #{l}" }
    exit 1
  end
  puts "verdict : metal train byte-deterministic run-to-run (pin train_metal_baseline.txt on this Mac to lock it)"
end

# (b) loss-decrease
first_loss = (curve1.first =~ /loss=(.+)\z/) ? $1.to_f : nil
last_loss  = (curve1.last  =~ /loss=(.+)\z/) ? $1.to_f : nil
if first_loss.nil? || last_loss.nil?
  warn "GATE FAIL [metal-train]: could not parse loss from step lines"
  exit 1
end
unless last_loss < first_loss
  warn "GATE FAIL [metal-train]: loss did not decrease (initial=#{first_loss}, final=#{last_loss})"
  exit 1
end
puts "verdict : loss decreases (#{first_loss} -> #{last_loss})"

# (c) checkpoint round-trip through CPU `toy infer` vs the SHARED fixture.
puts
puts "WARN: train-from-scratch metal checkpoint round-trip exercises the"
puts "WARN: CPU-TinyNN-reads-Metal-buffer seam — if Metal buffers are not"
puts "WARN: host-addressable through the CPU FFI handle this WILL fail or"
puts "WARN: produce garbage; this is the assertion most likely to fail on Metal."

run_line = out1.lines.find { |l| l.start_with?("run ") }
unless run_line && run_line =~ /\Arun (\S+) .* (\/\S+)\s*\z/
  warn "GATE FAIL [metal-train]: could not parse run dir: #{run_line.inspect}"
  exit 1
end
run_dir = $2.strip
ckpt    = File.join(run_dir, "weights", "step_5.gguf")
unless File.file?(ckpt)
  warn "GATE FAIL [metal-train]: checkpoint missing: #{ckpt}"
  exit 1
end

IDS_FIXTURE = File.join(ROOT, "prep", "fixtures", "ckpt_roundtrip_baseline.txt")
unless File.file?(IDS_FIXTURE)
  warn "metal_gate: missing shared ids fixture: #{IDS_FIXTURE}"
  exit 2
end
expected_ids = nil
File.foreach(IDS_FIXTURE) do |line|
  next if line.strip.empty? || line.start_with?("#")
  expected_ids = line.chomp if line.start_with?("ids:")
end
if expected_ids.nil?
  warn "metal_gate: ids fixture has no `ids:` body line: #{IDS_FIXTURE}"
  exit 2
end

out2, st2 = Open3.capture2e(TOY, "infer", ckpt,
                            "--prompt-ids", "1 2 3", "--n", "5", chdir: ROOT)
unless st2.success?
  warn "GATE FAIL [metal-train]: `toy infer` (ckpt RT) exited #{st2.exitstatus}:"
  warn out2.lines.last(20).join
  exit 1
end
ids_line = out2.lines.map(&:chomp).find { |l| l.start_with?("ids:") }
if ids_line.nil?
  warn "GATE FAIL [metal-train]: `toy infer` produced no `ids:` line"
  exit 1
end
unless ids_line == expected_ids
  warn "GATE FAIL [metal-train]: checkpoint round-trip ids diverged from shared fixture"
  warn "  expected: #{expected_ids.inspect}"
  warn "  actual  : #{ids_line.inspect}"
  exit 1
end
puts "verdict : checkpoint round-trips through CPU `toy infer` (shared fixture)"

# (d) events.jsonl structural: run_start first, run_end last.
events = File.join(run_dir, "events.jsonl")
unless File.file?(events)
  warn "GATE FAIL [metal-train]: events.jsonl missing: #{events}"
  exit 1
end
parsed = []
File.foreach(events) do |line|
  next if line.strip.empty?
  begin
    parsed << JSON.parse(line)
  rescue JSON::ParserError => e
    warn "GATE FAIL [metal-train]: events.jsonl unparseable line: #{e.message}"
    exit 1
  end
end
if parsed.empty?
  warn "GATE FAIL [metal-train]: events.jsonl has no events"
  exit 1
end
unless parsed.first["kind"] == "run_start"
  warn "GATE FAIL [metal-train]: first event is #{parsed.first['kind'].inspect}, expected run_start"
  exit 1
end
unless parsed.last["kind"] == "run_end"
  warn "GATE FAIL [metal-train]: last event is #{parsed.last['kind'].inspect}, expected run_end"
  exit 1
end
puts "verdict : events.jsonl run_start-first / run_end-last"
puts "GATE PASS [metal-train]"

puts
puts "ALL METAL ARMS PASS"
exit 0
