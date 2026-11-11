#!/usr/bin/env ruby
# prep/llamacpp_gguf_gate.rb — toy#203 regression gate.
#
# A GGUF written by llama.cpp's convert_hf_to_gguf.py for a Qwen2.5 has
# general.architecture=qwen2, "qwen2.*" scalar keys, ggml-native weight
# layout and NO toy.ggml_native key. Two things were wrong with it:
#
#   1. Arch.from_gguf and the loader each probed a fixed prefix list that
#      lacked qwen2, so every dimension read back -1 and the run died in a
#      ggml assert that never mentioned metadata.
#   2. Past that, layout was decided by the toy.ggml_native FLAG alone, so
#      the file went down the legacy transposing loader and decoded
#      plausible-looking WRONG token ids with exit 0.
#
# (2) is the dangerous one, and it is why this gate compares IDS rather
# than checking that the file merely loads: a fix for (1) alone passes any
# "it loaded" assertion while generating garbage.
#
# The fixture is manufactured, not committed: prep/gguf_relabel.rb rewrites
# the metadata of data/smollm2-135m-native.gguf into the llama.cpp shape
# and copies the tensor bytes verbatim. Same weights, so the relabelled
# file must decode EXACTLY the ids of its source.
#
# Legs:
#   resolve   relabelled file reports the source's dimensions (not -1)
#   decode    relabelled ids == source ids, byte for byte
#   layout    with the layout flag dropped but the prefix left as "llama",
#             ids still match (isolates shape detection from the prefix fix)
#   foreign   an unknown architecture prefix exits non-zero and says
#             "unsupported architecture" — and builds no graph
#
#   ruby prep/llamacpp_gguf_gate.rb     # exit 0 pass, 1 fail, 2 unusable

require "open3"
require "tmpdir"
require "fileutils"

ROOT   = File.expand_path("..", __dir__)
RUNNER = ENV["TOY_GATE_RUNNER"] || File.join(ROOT, "libexec", "toy-infer")
SRC    = ENV["TOY_GATE_SRC"] || File.join(ROOT, "data", "smollm2-135m-native.gguf")
RELABEL = File.join(ROOT, "prep", "gguf_relabel.rb")
IDS    = "6403 1980 253 655 28"
N_NEW  = "8"

unless File.file?(SRC)
  warn "llamacpp_gguf_gate: missing fixture #{SRC} (never skipped silently)"
  exit 2
end
unless ENV["TOY_GATE_RUNNER"]
  _o, st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-infer")
  unless st.success?
    warn "llamacpp_gguf_gate: could not build libexec/toy-infer"
    exit 2
  end
end

def infer(path)
  Open3.capture2e({ "GGUF" => path, "PROMPT_IDS" => IDS, "N_NEW" => N_NEW }, RUNNER)
end

def ids_line(out)
  out.lines.map(&:strip).reverse.find { |l| l.start_with?("ids:") }.to_s
end

def arch_line(out)
  out.lines.map(&:strip).find { |l| l.start_with?("Arch(") }.to_s
end

def relabel(src, dst, *args)
  out, st = Open3.capture2e("ruby", RELABEL, src, dst, *args)
  abort "llamacpp_gguf_gate: relabel failed: #{out}" unless st.success?
end

failures = []
Dir.mktmpdir("toy203-") do |dir|
  ref_out, ref_st = infer(SRC)
  ref_ids  = ids_line(ref_out)
  ref_arch = arch_line(ref_out)
  if !ref_st.success? || ref_ids.empty?
    warn "llamacpp_gguf_gate: the SOURCE fixture does not decode:\n#{ref_out}"
    exit 2
  end
  # The reference must be a real decode, or "equal to it" proves nothing.
  if ref_ids.split.length != 1 + IDS.split.length + N_NEW.to_i
    warn "llamacpp_gguf_gate: reference line has the wrong token count: #{ref_ids}"
    exit 2
  end
  ref_dims = ref_arch[/vocab=.*?d_ff=\d+/].to_s

  # ---- resolve + decode: the full llama.cpp shape ----
  q = File.join(dir, "as-llamacpp-qwen2.gguf")
  relabel(SRC, q, "--prefix", "qwen2", "--drop", "toy.ggml_native")
  out, st = infer(q)
  dims = arch_line(out)[/vocab=.*?d_ff=\d+/].to_s
  if dims.empty? || dims != ref_dims
    failures << "resolve: relabelled dims #{dims.inspect} != source #{ref_dims.inspect}"
  else
    puts "  ok: resolve — qwen2.* keys read as #{dims}"
  end
  if !st.success?
    failures << "decode: runner exited #{st.exitstatus} on the qwen2-labelled file"
  elsif ids_line(out) != ref_ids
    failures << "decode: ids differ from the source (same weights)\n      source   : #{ref_ids}\n      relabelled: #{ids_line(out)}"
  else
    puts "  ok: decode — #{ref_ids}"
  end

  # ---- layout: flag dropped, prefix untouched ----
  l = File.join(dir, "native-no-flag.gguf")
  relabel(SRC, l, "--prefix", "llama", "--drop", "toy.ggml_native")
  out, st = infer(l)
  if !st.success? || ids_line(out) != ref_ids
    failures << "layout: a native file without toy.ggml_native decoded #{ids_line(out).inspect}, want #{ref_ids.inspect}"
  else
    puts "  ok: layout — native weights recognised by shape, without the flag"
  end

  # ---- foreign: an architecture the engine does not model ----
  f = File.join(dir, "foreign.gguf")
  relabel(SRC, f, "--prefix", "notanarch", "--drop", "toy.ggml_native")
  out, st = infer(f)
  if st.success?
    failures << "foreign: an unknown architecture was ACCEPTED (#{ids_line(out)})"
  elsif !out.include?("unsupported architecture") || !out.include?("notanarch")
    failures << "foreign: failed, but without naming the architecture:\n      #{out.lines.first.to_s.strip}"
  elsif out.include?("GGML_ASSERT") || out.include?("loading ")
    failures << "foreign: got as far as loading tensors before failing"
  else
    puts "  ok: foreign — refused before any graph is built, architecture named"
  end
end

if failures.empty?
  puts "GATE PASS [llamacpp-gguf]: llama.cpp-labelled GGUF resolves, decodes byte-identically to its source, and an unknown architecture fails loud"
  exit 0
end
failures.each { |m| puts "  FAIL: #{m}" }
puts "GATE FAIL [llamacpp-gguf]: #{failures.length} leg(s)"
exit 1
