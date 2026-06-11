# CPU mirror of demos/qwen25_bench_cuda.rb. Same shape, swaps the CUDA
# FFI for the CPU FFI. Use for side-by-side CPU/CUDA timing.
#
# Run:
#   GGUF=data/qwen25-0.5b-native.gguf PREFILL_T=128 N_NEW=64 ./demos/qwen25_bench_cpu

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_kv_engine"

GGUF      = ENV["GGUF"]       || "data/qwen25-0.5b-native.gguf"
MAX_T     = (ENV["MAX_T"]     || "512").to_i
N_WARMUP  = (ENV["N_WARMUP"]  || "8").to_i
PREFILL_T = (ENV["PREFILL_T"] || "128").to_i
N_NEW     = (ENV["N_NEW"]     || "64").to_i

puts "=== Bench (CPU): " + GGUF + " ==="
puts "  warmup=" + N_WARMUP.to_s + " prefill_T=" + PREFILL_T.to_s +
     " n_new=" + N_NEW.to_s + " max_T=" + MAX_T.to_s

cfg = SmolLM2ConfigLoader.read(GGUF)
puts "  cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s

flags = GGUFLoad.detect_smollm2_flags(GGUF)

t_load_0 = Time.now
gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
t_load_ms = (Time.now - t_load_0) * 1000.0
puts "  realize+mmap: " + t_load_ms.to_s + " ms"
puts "  backend: " + TinyNN.tnn_backend_name(kv.sess)

i = 0
while i < N_WARMUP
  SmolLM2KV.decode_step(kv, 9707, i)
  i = i + 1
end
warmup_end_pos = N_WARMUP

t_prefill_0 = Time.now
i = 0
while i < PREFILL_T
  tok = (i * 1103515245 + 12345) & 0x3FFF
  SmolLM2KV.decode_step(kv, tok, warmup_end_pos + i)
  i = i + 1
end
t_prefill_ms = (Time.now - t_prefill_0) * 1000.0
puts "  prefill " + PREFILL_T.to_s + " tok: total=" + t_prefill_ms.to_s +
     " ms  per_tok=" + (t_prefill_ms / PREFILL_T.to_f).to_s + " ms"

per_tok_ms = []
decode_start_pos = warmup_end_pos + PREFILL_T
n = 0
while n < N_NEW
  pos = decode_start_pos + n
  tok = ((pos * 1664525) + 1013904223) & 0x3FFF
  t0 = Time.now
  SmolLM2KV.decode_step(kv, tok, pos)
  t1 = Time.now
  per_tok_ms.push((t1 - t0) * 1000.0)
  n = n + 1
end

sorted = []
k = 0
while k < per_tok_ms.length
  sorted.push(per_tok_ms[k])
  k = k + 1
end
i = 1
while i < sorted.length
  v = sorted[i]
  j = i - 1
  while j >= 0 && sorted[j] > v
    sorted[j + 1] = sorted[j]
    j = j - 1
  end
  sorted[j + 1] = v
  i = i + 1
end

n_dec = sorted.length
min_v = sorted[0]
max_v = sorted[n_dec - 1]
med_v = sorted[n_dec / 2]
p95_i = ((n_dec - 1).to_f * 0.95).to_i
p95_v = sorted[p95_i]
sum   = 0.0
i = 0
while i < n_dec
  sum = sum + sorted[i]
  i = i + 1
end
mean_v = sum / n_dec.to_f
toks_per_sec = 1000.0 / mean_v

puts "  decode " + N_NEW.to_s + " tok per-token (ms):"
puts "    mean   = " + mean_v.to_s
puts "    median = " + med_v.to_s
puts "    min    = " + min_v.to_s
puts "    p95    = " + p95_v.to_s
puts "    max    = " + max_v.to_s
puts "  throughput: " + toks_per_sec.to_s + " tok/s (from mean)"
