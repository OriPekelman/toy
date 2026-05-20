# F1.2 step 2 — LoRA on SmolLM2-135M Q projection, forward parity at B=0.
#
# Validates the SmolLM2KVFFICache#enable_lora_q! integration:
#   1. Load SmolLM2-135M once WITHOUT LoRA, decode N tokens, capture IDs.
#   2. Load it again WITH LoRA r=16 + standard init (A small Gaussian,
#      B zero), decode N tokens, capture IDs.
#   3. Assert the two ID sequences are bit-identical.
#
# At step 0 of LoRA training, B=0 makes the adapter contribute exactly
# 0 to the Q projection — forward output MUST match the base model.
# If this gate goes red, the splice in build_attention_qhead_step is
# wrong (wrong sign, wrong tensor binding, or graph-order issue).
#
# This is forward-only; training (backward + opt_step on the adapters)
# is F1.2 step 3.
#
# Run: GGUF=data/smollm2-135m-native.gguf ./demos/smollm2_lora_forward

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"

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
  # Prefill
  i = 0
  while i < prompt_ids.length
    SmolLM2KV.decode_step(kv, prompt_ids[i], i)
    i = i + 1
  end
  # Greedy decode
  n = 0
  while n < n_new
    pos = ids.length
    last_id = ids[pos - 1]
    logits = SmolLM2KV.decode_step(kv, last_id, pos)
    best_i = 0
    best_v = logits.flat[0]
    j = 1
    while j < vocab_size
      v = logits.flat[j]
      if v > best_v
        best_v = v
        best_i = j
      end
      j = j + 1
    end
    ids.push(best_i)
    n = n + 1
  end
  ids
end

# Prompt: "Once upon a time" in SmolLM2 tokens. Same prompt the
# handoff doc records for SmolLM2-135M ("Once upon a time, there
# was a little girl named Lily...").
PROMPT = [12092, 4845, 253, 1429]

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s + " d_head=" + (cfg.d_model / cfg.n_heads).to_s

# ---- Pass 1: baseline (no LoRA) ----
puts ""
puts "=== Pass 1: baseline ==="
gguf_base = TinyNN.tnn_gguf_load(GGUF)
kv_base = SmolLM2KVFFICache.new
kv_base.realize_for_mmap(gguf_base, cfg, MAX_T, flags.untied, flags.qkv_bias)
ids_base = decode_n(kv_base, PROMPT, N_NEW, cfg.vocab)
print "  ids:"
k = 0
while k < ids_base.length
  print " " + ids_base[k].to_s
  k = k + 1
end
puts ""

# ---- Pass 2: with LoRA r=RANK, B=0 init ----
puts ""
puts "=== Pass 2: LoRA r=" + RANK.to_s + " B=0 ==="
gguf_lora = TinyNN.tnn_gguf_load(GGUF)
kv_lora = SmolLM2KVFFICache.new
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

# ---- Compare ----
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
  puts "VERDICT: PASS (LoRA-B=0 matches baseline bit-for-bit)"
else
  puts "VERDICT: FAIL (LoRA-B=0 diverged from baseline)"
  exit 1
end
