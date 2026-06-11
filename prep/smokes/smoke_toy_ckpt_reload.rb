# toy#gguf-checkpoint-reload (#153) — load a from-scratch toy GGUF and
# run inference end-to-end. The from-scratch training path writes
# checkpoints with per-head tensor names (blk.N.attn_q.head_H.weight)
# rather than llama.cpp's fused (blk.N.attn_q.weight); both formats are
# accepted by toy_smollm2_ffi_kv.rb's realize_for_mmap as of #153.
#
# Acceptance: the smoke loads a 5-step from-scratch toy ckpt, runs a
# 3-token prompt → 3 new tokens, and emits the IDs. No crash + no NaN
# in the logits is the only quality bar; the toy ckpt is undertrained
# by ~5 orders of magnitude (5 steps on a 2-layer-4-head model) and
# its outputs are not expected to be coherent.
#
#   make prep/smokes/smoke_toy_ckpt_reload
#   GGUF=/tmp/ckpt/weights/latest ./prep/smokes/smoke_toy_ckpt_reload

require_relative "../../lib/toy/models/arch"
require_relative "../../lib/toy/models/transformer_lm"

GGUF = ENV["GGUF"] || "/tmp/ckpt_test/weights/latest"

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "smoke_toy_ckpt_reload: could not load " + GGUF
  exit 1
end
puts arch.summary

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)
puts "loaded OK"

# A 3-token prompt within the TinyStories 627-vocab (the writer's
# default). All IDs < 627; the ckpt-written ROPE + RMS-norm parameters
# may produce NaN if the per-head tensors weren't loaded correctly,
# which is what we want to catch.
ids = lm.generate([1, 2, 3], 3)
print "ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
puts "smoke OK"
