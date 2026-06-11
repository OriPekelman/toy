# demos/tinyllama_kv_cuda.rb — TinyLlama-1.1B inference via the FFI KV-cache (CUDA).
#
# Uses Toy::SmolLM2 / SmolLM2KVFFICacheCuda — they're already the
# llama-family architecture, just configured differently here:
#   d_model=2048, n_layers=22, n_heads=32, n_kv=4, d_ff=5632,
#   vocab=32000, ctx=2048, rope_base=10000, untied embeddings.
#
# Tokenize host-side first:
#   ./prep/llama_tokens.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --ids data/tinyllama_prompt_ids.txt encode "Once upon a time"

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"
require_relative "../lib/toy/train/training"   # parse_ids

GGUF     = "data/tinyllama-1.1b-f32.gguf"
IDS_PATH = "data/tinyllama_prompt_ids.txt"
MAX_T    = 256
N_NEW    = 16

cfg = SmolLM2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " L=" + cfg.n_layers.to_s +
     " d_ff=" + cfg.d_ff.to_s +
     " rope_base=" + cfg.rope_base.to_s

model = Toy::SmolLM2.new(cfg)
GGUFLoad.load_toy_smollm2(model, GGUF)
puts ""
puts model.algorithm_card

puts "realizing KV cache (MAX_T=" + MAX_T.to_s + ")..."
kv = SmolLM2KVFFICacheCuda.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                model.has_untied_output, model.stack[0].attn.has_qkv_bias)
t0 = Time.now
SmolLM2KVCuda.upload_from(kv, model)
puts "  uploaded weights in " + ((Time.now - t0) * 1000.0).to_s + " ms"

raw = ["?"]
raw.pop
File.open(IDS_PATH, "r") do |f|
  f.each_line { |line| raw.push(line.chomp) }
end
ids = parse_ids(raw[0])
puts ""
puts "prefilling " + ids.length.to_s + " prompt tokens..."

t0 = Time.now
i = 0
while i < ids.length
  SmolLM2KVCuda.decode_step(kv, ids[i], i)
  i = i + 1
end
prefill_ms = (Time.now - t0) * 1000.0
puts "  prefill: " + prefill_ms.to_s + " ms (" +
     (prefill_ms / ids.length.to_f).to_s + " ms/token)"

puts "generating " + N_NEW.to_s + " tokens..."
t0 = Time.now
n = 0
while n < N_NEW
  pos = ids.length
  last_id = ids[pos - 1]
  logits = SmolLM2KVCuda.decode_step(kv, last_id, pos)
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
  ids.push(best_i)
  n = n + 1
end
gen_ms = (Time.now - t0) * 1000.0
puts "  generation total: " + gen_ms.to_s + " ms  (" +
     (gen_ms / N_NEW.to_f).to_s + " ms/token)"

File.open(IDS_PATH, "w") do |out|
  n = ids.length
  k = 0
  while k < n
    out.write(ids[k].to_s)
    if k < n - 1
      out.write(" ")
    end
    k = k + 1
  end
  out.write("\n")
end
puts "wrote " + IDS_PATH
