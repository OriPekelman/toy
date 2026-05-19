# Qwen2.5 inference via the Mat-mediated FFI KV-cache path. The slow,
# correct, gold-standard reference. Use the mmap demos
# (qwen25_native_mmap{,_cuda}) for actual performance.
#
# Pick model via env:
#   GGUF=data/qwen25-0.5b-f32.gguf  ./demos/qwen25_kv   # default
#   GGUF=data/qwen25-1.5b-f32.gguf  ./demos/qwen25_kv
#   GGUF=data/qwen25-3b-f32.gguf    ./demos/qwen25_kv
#   GGUF=data/qwen25-7b-q8_0.gguf   ./demos/qwen25_kv   # Q8 via Mat-mediated
#
# Tokenize first:
#   ./prep/qwen25_tokens.py encode "Hello, my name is"

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/training"

GGUF     = ENV["GGUF"]     || "data/qwen25-0.5b-f32.gguf"
IDS_PATH = ENV["IDS_PATH"] || "data/qwen25_prompt_ids.txt"
MAX_T    = (ENV["MAX_T"]   || "256").to_i
N_NEW    = (ENV["N_NEW"]   || "16").to_i

# --- config from GGUF ---
cfg = SmolLM2ConfigLoader.read(GGUF)
# DEBUG: truncate stack to bisect the NaN-at-deep-stack issue.
forced_layers = ENV["FORCE_N_LAYERS"]
if forced_layers != nil
  cfg.n_layers = forced_layers.to_i
  puts "  [debug] forced n_layers=" + cfg.n_layers.to_s
end
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
kv = SmolLM2KVFFICache.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                model.has_untied_output, model.stack[0].attn.has_qkv_bias)
t0 = Time.now
SmolLM2KV.upload_from(kv, model)
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

# --- prefill ---
t0 = Time.now
i = 0
while i < ids.length
  SmolLM2KV.decode_step(kv, ids[i], i)
  i = i + 1
end
prefill_ms = (Time.now - t0) * 1000.0
puts "  prefill: " + prefill_ms.to_s + " ms (" +
     (prefill_ms / ids.length.to_f).to_s + " ms/token)"

# --- generation ---
puts "generating " + N_NEW.to_s + " tokens..."
t0 = Time.now
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
  # First step diagnostics: how does argmax compare to the next few candidates?
  if n == 0
    puts "  step 0 logits: top index=" + best_i.to_s + " val=" + best_v.to_s +
         " (logits[0]=" + logits.flat[0].to_s + ", logits[100]=" +
         logits.flat[100].to_s + ", logits[1000]=" + logits.flat[1000].to_s + ")"
    finite_count = 0
    nan_count    = 0
    k = 0
    while k < cfg.vocab
      v = logits.flat[k]
      if v == v && v.abs < 1.0e30  # finite-ish, not NaN
        finite_count = finite_count + 1
      else
        nan_count = nan_count + 1
      end
      k = k + 1
    end
    puts "  finite=" + finite_count.to_s + " nan_or_inf=" + nan_count.to_s
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
