# F2 step 1 — LoRA-Q forward-parity smoke on CUDA.
#
# CUDA mirror of demos/smollm2_lora_forward.rb. Loads SmolLM2-135M
# twice on CUDA, once baseline and once with LoRA r=16 + B=0 init,
# decodes 12 tokens, asserts the two ID sequences are bit-identical.
#
# Acceptance gate against this is two-sided:
#   1. CUDA baseline IDs == CPU baseline IDs (already covered by
#      the existing CUDA decode demos for parity vs CPU at HEAD).
#   2. CUDA LoRA-B=0 IDs == CUDA baseline IDs (this smoke).
#
# If 2 fails, the CUDA LoRA splice in
# lib/toy_smollm2_ffi_kv_cuda.rb#build_attention_qhead_step is wrong.
#
# Run: GGUF=data/smollm2-135m-native.gguf ./demos/smollm2_lora_forward_cuda

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/models/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"

GGUF  = ENV["GGUF"]  || "data/smollm2-135m-native.gguf"
MAX_T = (ENV["MAX_T"] || "128").to_i
N_NEW = (ENV["N_NEW"] || "12").to_i
RANK  = (ENV["RANK"]  || "16").to_i
SEED  = (ENV["SEED"]  || "42").to_i

def decode_n(kv, prompt_ids, n_new, vocab_size)
  ids = []
  i = 0
  while i < prompt_ids.length
    ids.push(prompt_ids[i])
    i = i + 1
  end
  i = 0
  while i < prompt_ids.length
    SmolLM2KVCuda.decode_step(kv, prompt_ids[i], i)
    i = i + 1
  end
  n = 0
  while n < n_new
    pos = ids.length
    last_id = ids[pos - 1]
    logits = SmolLM2KVCuda.decode_step(kv, last_id, pos)
    best_i = 0
    best_v = logits.flat[0]
    j = 1
    while j < vocab_size
      v = logits.flat[j]
      if v > best_v; best_v = v; best_i = j; end
      j = j + 1
    end
    ids.push(best_i)
    n = n + 1
  end
  ids
end

PROMPT = [12092, 4845, 253, 1429]   # "Once upon a time"

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s

puts ""
puts "=== Pass 1: baseline (CUDA) ==="
gguf_base = TinyNNCuda.tnn_gguf_load(GGUF)
kv_base = SmolLM2KVFFICacheCuda.new
kv_base.realize_for_mmap(gguf_base, cfg, MAX_T, flags.untied, flags.qkv_bias)
puts "  backend: " + TinyNNCuda.tnn_backend_name(kv_base.sess)
ids_base = decode_n(kv_base, PROMPT, N_NEW, cfg.vocab)
print "  ids:"
k = 0
while k < ids_base.length
  print " " + ids_base[k].to_s
  k = k + 1
end
puts ""

puts ""
puts "=== Pass 2: LoRA r=" + RANK.to_s + " B=0 (CUDA) ==="
gguf_lora = TinyNNCuda.tnn_gguf_load(GGUF)
kv_lora = SmolLM2KVFFICacheCuda.new
kv_lora.enable_lora_q!(RANK)
kv_lora.realize_for_mmap(gguf_lora, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv_lora.upload_lora_q_init!(SEED, 0.01)
ids_lora = decode_n(kv_lora, PROMPT, N_NEW, cfg.vocab)
print "  ids:"
k = 0
while k < ids_lora.length
  print " " + ids_lora[k].to_s
  k = k + 1
end
puts ""

puts ""
ok = (ids_base.length == ids_lora.length)
if ok
  j = 0
  while j < ids_base.length
    if ids_base[j] != ids_lora[j]; ok = false; end
    j = j + 1
  end
end

if ok
  puts "VERDICT: PASS (CUDA LoRA-B=0 matches CUDA baseline bit-for-bit)"
else
  puts "VERDICT: FAIL"
  exit 1
end
