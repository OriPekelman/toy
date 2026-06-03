# Phase 0.7 acceptance gates for Qwen2.5 inference via TransformerLM.
#
# Loads each Qwen2.5 GGUF, runs greedy decode on a fixed token-ID prompt,
# and asserts the generated IDs match a recorded golden array. Outputs
# were captured 2026-05-21 on gx10 CPU with Spinel @ 7beeb54 + tinynn @
# the F1.1 fix commit (472cf86).
#
# Why token-ID gates rather than text: the project's TransformerLM speaks
# token IDs (server-side tokenizer is opt-in via the Phase D2/D3 BPE
# encoder, not used here). IDs are bit-stable across runs; text comparison
# would require a working encoder for every model family.
#
# Why CPU only: bit-identical reproducibility. The CUDA path uses the
# same graph and produces matching tokens on gx10 (see the CUDA demos);
# adding it here would double runtime without changing what the gate
# detects. CUDA-side acceptance lives in the *_cuda demos.
#
# Why no 7B: at ~1s/token on CPU, gating it would add ~15s for minimal
# extra signal (the 0.5B-Q8 vs 0.5B-f32 byte-for-byte match already
# proves the dequant path is sound, and 3B exercises the largest f32
# graph). Run it manually before tagging a release:
#   GGUF=data/qwen25-7b-q8_0.gguf N_NEW=8 ./demos/qwen25_transformer_lm
#
# How to run:
#   make qwen25_acceptance && ./demos/qwen25_acceptance
# Or by hand:
#   spinel demos/qwen25_acceptance.rb -o demos/qwen25_acceptance
#   ./demos/qwen25_acceptance
#
# Exit status: 0 if all gates pass, non-zero otherwise.

require_relative "../lib/toy/models/arch"
require_relative "../lib/transformer_lm"

PROMPT = [9707, 11, 847, 829, 374]   # "Hello, my name is" in Qwen2 tokens

def run_gate(name, gguf, n_new, gold)
  print "[gate] " + name + " ... "

  arch = Arch.from_gguf(gguf)
  lm = ToyLM.new(arch, :cpu)
  lm.load(gguf)
  ids = lm.generate(PROMPT, n_new)

  ok = (ids.length == gold.length)
  if ok
    j = 0
    while j < ids.length
      if ids[j] != gold[j]
        ok = false
      end
      j = j + 1
    end
  end

  if ok
    puts "PASS"
    return 1
  else
    puts "FAIL"
    print "  expected: ["
    j = 0
    while j < gold.length
      print " " + gold[j].to_s
      j = j + 1
    end
    puts " ]"
    print "  actual:   ["
    j = 0
    while j < ids.length
      print " " + ids[j].to_s
      j = j + 1
    end
    puts " ]"
    return 0
  end
end

# Golden token-ID sequences captured 2026-05-21 on gx10 CPU. Each is
# prompt (5 tokens) + N_NEW generated tokens. Update via the recipe in
# docs/design/phase-07-acceptance.md when graph code intentionally
# changes.

n_pass = 0
n_pass = n_pass + run_gate(
  "qwen25-0.5b-native",
  "data/qwen25-0.5b-native.gguf",
  12,
  [9707, 11, 847, 829, 374, 264, 220, 16, 15, 1042, 2310, 8171,
   13, 358, 614, 264, 3405])

n_pass = n_pass + run_gate(
  "qwen25-0.5b-native-q8",
  "data/qwen25-0.5b-native-q8.gguf",
  12,
  [9707, 11, 847, 829, 374, 264, 220, 16, 15, 1042, 2310, 8171,
   13, 358, 614, 264, 3405])

n_pass = n_pass + run_gate(
  "qwen25-1.5b-native",
  "data/qwen25-1.5b-native.gguf",
  12,
  [9707, 11, 847, 829, 374, 71, 6255, 323, 358, 1079, 264, 220,
   16, 17, 339, 11972, 5458])

n_pass = n_pass + run_gate(
  "qwen25-3b-native",
  "data/qwen25-3b-native.gguf",
  8,
  [9707, 11, 847, 829, 374, 323, 358, 1079, 264, 220, 16, 15, 339])

puts ""
total = 4
n_fail = total - n_pass
puts "[acceptance] pass=" + n_pass.to_s + " fail=" + n_fail.to_s + " total=" + total.to_s
if n_fail > 0
  raise "acceptance gates failed"
end
