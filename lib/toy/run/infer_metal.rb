# lib/toy/run/infer_metal.rb — Spinel-compiled Metal inference COMPUTE runner.
#
# Metal sibling of lib/toy/run/infer.rb (libexec/toy-infer). The CRuby CLI shell
# (lib/toy/core/cli/infer.rb) selects this binary when invoked with
# `--device metal` on macOS: it builds libexec/toy-infer-metal and shells out
# with the SAME controlled ENV (GGUF/PROMPT/N_NEW). DEVICE is irrelevant in here
# — this binary IS the metal path.
#
# HAND-WRITTEN, NOT mechanically mirrored: ToyLMMetal's ctor arity is 1
# (ToyLMMetal.new(arch)), unlike ToyLM.new(arch, :cpu), so gen_cuda_mirror.rb
# cannot derive this file from the CPU runner. It is therefore DELIBERATELY
# ABSENT from MIRRORABLE in prep/gen_cuda_mirror.rb — exactly like the CPU
# runner is, for the inverse reason.
#
# SINGLE-TYPE FILE: only ToyLMMetal is referenced (no ToyLM), so the Spinel
# polymorphic-lm-var miscompile (landmine #16) cannot fire. Each device has its
# own self-contained runner; the CLI does the dispatch by picking the binary.
#
# OUTPUT: byte-identical prefixes to the CPU runner ("text: …" / "ids: …") so
# the CLI's scan_line and prep/infer_gate.rb parse both branches identically.
# Hand-built output only — no #{} interpolation in any emitted line.
#
# DETERMINISM: generate(..) with no sampler_config → Sampler.argmax (greedy,
# first-max-wins, no rand/seed). The Metal F32 forward is expected to match the
# CPU runner's generated token IDs; prep/metal_gate.rb gates that discrete
# invariant metal-vs-cpu in the same run. RUNTIME-UNVERIFIED on gx10 (Linux,
# no Apple frameworks) — the parity gate runs on the Mac.

require_relative "../../arch"
require_relative "../../transformer_lm_metal"
require_relative "../../tokenizer"

GGUF  = ENV["GGUF"] || "data/smollm2-135m-f32.gguf"
PROMPT = ENV["PROMPT"] || "Once upon a time"
N_NEW  = (ENV["N_NEW"] || "16").to_i

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "toy-infer: could not load " + GGUF +
       " — set GGUF= to a valid file (see `toy list`)."
  exit 1
end
puts arch.summary

lm = ToyLMMetal.new(arch)
lm.load(GGUF)

tok = Tokenizer.from_gguf(GGUF)

if tok.present
  in_ids = tok.encode(PROMPT)
  puts "prompt: " + PROMPT.inspect + " → " + in_ids.length.to_s + " tokens"
  out_ids = lm.generate(in_ids, N_NEW)
  puts "text: " + tok.decode(out_ids)
else
  puts "no tokenizer in GGUF; re-convert with --with-tokenizer for text I/O"
  ids = lm.generate([6403, 1980, 253, 655, 28], N_NEW)
  print "ids:"
  k = 0
  while k < ids.length
    print " " + ids[k].to_s
    k = k + 1
  end
  puts ""
end
