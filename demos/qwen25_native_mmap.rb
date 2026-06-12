# Phase 2 BYO-pointer mmap inference (CPU). Weights are NOT copied
# into a destination buffer — persistent ggml tensors point directly
# at the GGUF's mmap'd pages.
#
# Pick model + precision via env:
#   GGUF=data/qwen25-0.5b-native.gguf     ./demos/qwen25_native_mmap   # F32
#   GGUF=data/qwen25-1.5b-native.gguf     ./demos/qwen25_native_mmap   # F32
#   GGUF=data/qwen25-7b-native-q8.gguf    ./demos/qwen25_native_mmap   # Q8
#   GGUF=data/qwen25-7b-native.gguf       ./demos/qwen25_native_mmap   # F32
#
# Defaults to 0.5B F32 for quick smoke checks.

require_relative "../lib/toy/llm/engine/llama_kv_engine"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"

GGUF  = ENV["GGUF"]  || "data/qwen25-0.5b-native.gguf"
MAX_T = (ENV["MAX_T"] || "256").to_i
N_NEW = (ENV["N_NEW"] || "8").to_i

ids = [9707, 11, 847, 829, 374]

cfg = SmolLM2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " L=" + cfg.n_layers.to_s

flags = GGUFLoad.detect_smollm2_flags(GGUF)
wtype = GGUFLoad.detect_weight_type(GGUF)
# toy#77: ternaries, not Bool#to_s — Spinel emits Bool#to_s as a raw
# unguarded C literal and GC-roots it; two in one concat chain can
# segfault sp_gc_mark depending on link layout (see qwen25_bench_cuda).
puts "flags: untied=" + (flags.untied ? "true" : "false") +
     " qkv_bias=" + (flags.qkv_bias ? "true" : "false") +
     " weight_type=" + wtype.to_s

gguf = TinyNN.tnn_gguf_load(GGUF)

kv = SmolLM2KVFFICache.new
kv.set_weight_type(wtype)

t0 = Time.now
# toy#77: realize_for_mmap is 6-arg now (qk_norm); Spinel zero-fills
# under-arity calls with NO diagnostic. Pass qk_norm explicitly.
kv.qk_norm_kind = flags.qk_norm_kind
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias, flags.qk_norm)
puts "  realized + mmap'd in " + ((Time.now - t0) * 1000.0).to_s + " ms"

puts "prefilling " + ids.length.to_s + " prompt tokens..."
i = 0
while i < ids.length
  SmolLM2KV.decode_step(kv, ids[i], i)
  i = i + 1
end

puts "generating " + N_NEW.to_s + " tokens..."
n = 0
while n < N_NEW
  pos = ids.length
  last_id = ids[pos - 1]
  logits = SmolLM2KV.decode_step(kv, last_id, pos)
  best_i = 0
  best_v = logits.flat[0]
  j = 1
  while j < cfg.vocab
    v = logits.flat[j]
    if v > best_v
      best_v = v
      best_i = j
    end
    j = j + 1
  end
  if n == 0
    puts "  step 0: top index=" + best_i.to_s + " val=" + best_v.to_s
  end
  ids.push(best_i)
  n = n + 1
end

print "generated ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
