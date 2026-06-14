# lib/toy/run/train_gpt2_metal.rb — Metal twin of lib/toy/run/train_gpt2.rb.
#
# Compute for `toy train from-scratch --arch gpt2 --device metal`. A SEPARATE
# single-type binary (libexec/toy-train-gpt2-metal; landmine #16) that links the
# GENERATED CUDA engine mirror (gpt2_seq_engine_metal.rb → GPT2SeqEngineMetal,
# TinyNNMetal, tnn_session_new(2)) + the Metal TinyNN shim. Same training math as
# the CPU runner; the GELU/LayerNorm backward ops (GGML_OP_GELU_BACK/NORM_BACK)
# have no CUDA kernel, so the backend scheduler runs THEM on the CPU fallback
# backend (correct, slower) while the rest runs on Metal.
#
# REQUIRE-PATH CAUTION (see train_gpt2.rb): toplevel libs are "../../toy" /
# "../../tinynn_metal" (two levels up from lib/toy/run/); a wrong path is silently
# ignored → TinyNNMetal/Mat unloaded → emit-0 → CE=0.

require_relative "../../toy"
require_relative "../ffi/tinynn_metal"
require_relative "../llm/engine/gpt2_seq_engine_metal"

STEPS    = (ENV["STEPS"]    || "5").to_i
SEED     = (ENV["SEED"]     || "0").to_i
LR       = (ENV["LR"]       || "0.001").to_f
# From-scratch gate shape (literal, matches the CPU runner so the curve compares).
VOCAB    = 627
D_MODEL  = 64
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

engine = Toy::LLM::Engine::GPT2SeqEngineMetal.new
engine.realize!(VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS, CONTEXT, SEED)

# FAIL LOUD on a missing corpus (spinel-dev#17: silent "" then nil SEGV).
if !File.exist?("data/ts_seqs.txt")
  puts "toy-train-gpt2-metal: corpus not found: data/ts_seqs.txt (cwd-relative)"
  puts "  `toy new` seeds a project copy; in a toy checkout run prep/prep_tinystories.rb."
  exit 1
end
raw     = File.read("data/ts_seqs.txt")
lines   = raw.split("\n")
parts   = lines[0].split(" ")
seq_ids = [0]; seq_ids.pop
k = 0
while k < CONTEXT
  if k < parts.length
    seq_ids.push(parts[k].to_i)
  else
    seq_ids.push(0)
  end
  k = k + 1
end

positions = [0]; positions.pop
p = 0
while p < CONTEXT
  positions.push(p)
  p = p + 1
end

m_labels = Mat.new(CONTEXT, VOCAB)
zj = 0
while zj < CONTEXT * VOCAB; m_labels.flat[zj] = 0.0; zj = zj + 1; end
lk = 0
while lk < CONTEXT
  tgt = (lk + 1 < CONTEXT) ? seq_ids[lk + 1].to_i : seq_ids[lk].to_i
  m_labels.flat[lk * VOCAB + tgt] = 1.0
  lk = lk + 1
end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0

step = 0
while step < STEPS
  sp1 = (step + 1).to_f
  m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** sp1))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** sp1))
  loss = engine.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  step = step + 1
end

# toy#90 — Metal teardown drain. The GPT-2 Metal training session is never
# explicitly freed; without this the ggml-metal device-free residency assert
# (ggml-metal-device.m:618) aborts the process (exit 134) AFTER a correct
# run. Spinel has no at_exit (lib/toy/run/serve.rb:123) so drain explicitly.
# METAL-ONLY no-op for non-Metal. RUNTIME-UNVERIFIED on gx10 — Mac proves it.
TinyNNMetal.tnn_shutdown_engines
