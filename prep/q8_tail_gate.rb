#!/usr/bin/env ruby
# prep/q8_tail_gate.rb — toy#205 regression gate (CUDA).
#
# ggml-cuda sizes a quantized tensor whose row width (ne0) is not a
# multiple of 512 as its bytes plus up to one padding row segment, because
# its matmul kernels may read that far past the last row. toy maps weights
# straight from the GGUF, where nothing follows the LAST tensor — so a file
# ending in such a tensor aborted at load in ggml_backend_tensor_alloc's
# range check (Qwen2.5-0.5B Q8, d_ff 4864; a pruned student, d_ff 9472),
# while the same model in F32, or on CPU, ran. The fix maps every GGUF with
# a zero tail (TNN_GGUF_MMAP_TAIL) and registers it with the CUDA buffer.
#
# The fixture is manufactured: prep/gguf_relabel.rb --q8-last moves one
# F32 tensor of data/smollm2-135m-native.gguf (ne0 = 576, 576 % 512 != 0)
# to the end of the file and re-encodes it as Q8_0.
#
# Legs:
#   tail   the manufactured file loads on CUDA and decodes the ids the CPU
#          decodes from the same file
#   qwen   (when data/qwen25-0.5b-native-q8.gguf is present) the issue's
#          own file: CUDA ids == CPU ids
#
#   ruby prep/q8_tail_gate.rb     # exit 0 pass, 1 fail, 2 unusable

require "open3"
require "tmpdir"

ROOT    = File.expand_path("..", __dir__)
CPU     = ENV["TOY_GATE_RUNNER"] || File.join(ROOT, "libexec", "toy-infer")
CUDA    = ENV["TOY_GATE_RUNNER_CUDA"] || File.join(ROOT, "libexec", "toy-infer-cuda")
SRC     = File.join(ROOT, "data", "smollm2-135m-native.gguf")
QWEN    = File.join(ROOT, "data", "qwen25-0.5b-native-q8.gguf")
RELABEL = File.join(ROOT, "prep", "gguf_relabel.rb")
TENSOR  = "blk.29.ffn_up.weight"

unless File.file?(SRC)
  warn "q8_tail_gate: missing fixture #{SRC} (never skipped silently)"
  exit 2
end
[["libexec/toy-infer", "TOY_GATE_RUNNER"], ["libexec/toy-infer-cuda", "TOY_GATE_RUNNER_CUDA"]].each do |target, env|
  next if ENV[env]
  _o, st = Open3.capture2e("make", "-C", ROOT, target)
  unless st.success?
    warn "q8_tail_gate: could not build #{target}"
    exit 2
  end
end

def infer(runner, path, ids)
  # KV_BATCH=1: one token per step on both devices. The qwen prompt has a
  # near-tie at its third generated token (the F32 file picks 448 there, Q8
  # picks 323, and Q8 on CPU with a batched prompt picks 11), so CPU and CUDA
  # are compared on the same per-token kernels (toy#206, toy#212).
  out, st = Open3.capture2e({ "GGUF" => path, "PROMPT_IDS" => ids, "N_NEW" => "8", "KV_BATCH" => "1" }, runner)
  [out[/^ids: (.*)$/, 1].to_s.strip, st, out]
end

# Why a run failed, in one line: the ggml assert if there was one.
def why(out, st)
  a = out.lines.find { |l| l.include?("GGML_ASSERT") }
  a ? a.strip[0, 160] : "exit #{st.exitstatus.inspect}: #{out.lines.last.to_s.strip[0, 120]}"
end

failures = []

def compare(failures, leg, path, ids)
  c_ids, c_st, c_out = infer(CPU, path, ids)
  g_ids, g_st, g_out = infer(CUDA, path, ids)
  if !c_st.success? || c_ids.empty?
    failures << "#{leg}: CPU run failed — #{why(c_out, c_st)}"
  elsif !g_st.success? || g_ids.empty?
    failures << "#{leg}: CUDA run failed — #{why(g_out, g_st)}"
  elsif !g_out.include?("ggml_cuda_init: found")
    failures << "#{leg}: the CUDA runner did not initialise a CUDA device"
  elsif g_ids != c_ids
    failures << "#{leg}: CUDA ids differ from CPU ids\n      cpu:  #{c_ids}\n      cuda: #{g_ids}"
  else
    puts "  ok: #{leg} — CUDA == CPU (#{g_ids})"
  end
end

# ---- tail ----
Dir.mktmpdir("toy205") do |dir|
  tail = File.join(dir, "q8-tail.gguf")
  o, st = Open3.capture2e("ruby", RELABEL, SRC, tail, "--prefix", "llama", "--q8-last", TENSOR)
  if !st.success?
    failures << "tail: could not manufacture the fixture: #{o.lines.last.to_s.strip}"
  else
    compare(failures, "tail", tail, "6403 1980 253 655 28")
  end
end

# ---- qwen ----
if File.file?(QWEN)
  compare(failures, "qwen", QWEN, "9707 11 1879 0 358 1079 264")
else
  puts "  not run: qwen — #{QWEN} is not present"
end

if failures.empty?
  puts "GATE PASS [q8-tail-cuda]: a GGUF ending in a quantized tensor with ne0 % 512 != 0 loads on CUDA and decodes the CPU's ids"
  exit 0
end
failures.each { |m| puts "  FAIL: #{m}" }
puts "GATE FAIL [q8-tail-cuda]: #{failures.length} failure(s)"
exit 1
