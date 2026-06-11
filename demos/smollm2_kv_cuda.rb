# demos/smollm2_kv_cuda.rb — SmolLM2-135M inference via the FFI KV-cache.
#
# Mirror of demos/distilgpt2_demo_kv.rb but for the llama-family path
# (Toy::SmolLM2 + Toy::SmolLM2KVFFI). Per-step compute is constant in
# prompt length thanks to the KV cache.
#
# Tokenize first:
#   ./prep/smollm2_tokens.py encode "Once upon a time"

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_kv_engine_cuda"
require_relative "../lib/toy/train/training"   # parse_ids

GGUF     = "data/smollm2-135m-f32.gguf"
IDS_PATH = "data/smollm2_prompt_ids.txt"
MAX_T    = 256
N_NEW    = 16

# --- config from GGUF ---
cfg = SmolLM2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " L=" + cfg.n_layers.to_s +
     " rope_base=" + cfg.rope_base.to_s

# --- build native model + load GGUF weights ---
model = Toy::SmolLM2.new(cfg)
GGUFLoad.load_toy_smollm2(model, GGUF)
puts ""
puts model.algorithm_card
puts ""

# --- realize KV cache + upload weights ---
puts "realizing KV cache (MAX_T=" + MAX_T.to_s + ")..."
kv = SmolLM2KVFFICacheCuda.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                model.has_untied_output, model.stack[0].attn.has_qkv_bias)
t0 = Time.now
SmolLM2KVCuda.upload_from(kv, model)
puts "  uploaded weights in " + ((Time.now - t0) * 1000.0).to_s + " ms"

# --- read prompt ---
raw = ["?"]
raw.pop
File.open(IDS_PATH, "r") do |f|
  f.each_line { |line| raw.push(line.chomp) }
end
ids = parse_ids(raw[0])
puts ""
puts "prefilling " + ids.length.to_s + " prompt tokens..."

# --- prefill: one decode_step per prompt token ---
t0 = Time.now
i = 0
while i < ids.length
  SmolLM2KVCuda.decode_step(kv, ids[i], i)
  i = i + 1
end
prefill_ms = (Time.now - t0) * 1000.0
puts "  prefill: " + prefill_ms.to_s + " ms (" +
     (prefill_ms / ids.length.to_f).to_s + " ms/token)"

# --- generation: argmax decode of N_NEW more tokens ---
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

# --- write back ---
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
