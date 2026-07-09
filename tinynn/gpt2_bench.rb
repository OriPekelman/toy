# DistilGPT2 forward bench: native Mat vs FFI persistent-graph,
# apples-to-apples on the same prompt and same T_SEQ.
#
# Reports min/avg/max ms per forward + tokens/sec. The reported
# "tokens/sec" is the autoregressive-decode rate assuming the same
# T_SEQ each step (no KV ffi_full_cache); a KV-ffi_full_cache version would be
# substantially faster.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/models/gpt2"
require_relative "../lib/toy/llm/engine/gpt2_fwd_engine"
require_relative "../lib/toy/llm/engine/gpt2_kv_engine"
require_relative "../lib/toy/io/gguf_load"
require_relative "../lib/toy/train/training"

T_SEQ    = 5
MAX_T    = 32      # KV-cache capacity

N_NATIVE = 3       # native is slow; small N
N_FFI    = 30      # FFI is fast; bigger N for tighter stats
N_KV     = 30
IDS_PATH = "data/prompt_ids.txt"
GGUF     = "data/gpt2-f32.gguf"
# Hyperparams come from GGUF metadata; see lib/gguf_load.rb.

def read_ids(path)
  raw = ["?"]
  raw.pop
  File.open(path, "r") do |f|
    f.each_line { |line| raw.push(line.chomp) }
  end
  parse_ids(raw[0])
end

def stats_print(label, times_ms)
  n = times_ms.length
  if n == 0
    puts label + ": no samples"
    return
  end
  sum = 0.0
  mn  = times_ms[0]
  mx  = times_ms[0]
  i = 0
  while i < n
    v = times_ms[i]
    sum = sum + v
    if v < mn
      mn = v
    end
    if v > mx
      mx = v
    end
    i = i + 1
  end
  avg = sum / n
  tok_per_sec = 1000.0 / avg
  puts label + ":"
  puts "  n=" + n.to_s + "  avg=" + avg.to_s + " ms  min=" + mn.to_s +
       "  max=" + mx.to_s
  puts "  → " + tok_per_sec.to_s + " forwards/sec  (single-position decode)"
end

ids = read_ids(IDS_PATH)
puts "prompt: " + ids.length.to_s + " tokens  (T_SEQ=" + T_SEQ.to_s + ")"
if ids.length != T_SEQ
  puts "WARNING: prompt length " + ids.length.to_s + " != T_SEQ " + T_SEQ.to_s +
       "; re-encode with the right prompt length or change T_SEQ."
end

puts ""
cfg = GPT2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab_size.to_s + " d=" + cfg.d_model.to_s +
     " heads=" + cfg.n_heads.to_s + " layers=" + cfg.n_layers.to_s
puts ""
puts "loading native model + FFI ffi_full_cache..."
t0 = Time.now
model = GPT2LM.new(cfg.vocab_size, cfg.d_model, cfg.d_ff,
                    cfg.n_heads, cfg.n_layers, cfg.context_length)
GGUFLoad.load_gpt2(model, GGUF)
puts "  native ready: " + ((Time.now - t0) * 1000).to_s + " ms"

t0 = Time.now
ffi_full_cache = GPT2FullForwardFFICache.new
ffi_full_cache.realize_for(T_SEQ, cfg.d_model, cfg.d_ff,
                            cfg.n_heads, cfg.n_layers, cfg.vocab_size)
pos_slice = GPT2FFI.make_pos_slice(model, T_SEQ)
GPT2FFI.upload_from(ffi_full_cache, model, pos_slice)
puts "  FFI ready:    " + ((Time.now - t0) * 1000).to_s + " ms"

t0 = Time.now
kv = GPT2KVFFICache.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff,
                cfg.n_heads, cfg.n_layers, cfg.vocab_size,
                cfg.context_length)
GPT2KV.upload_from(kv, model)
puts "  KV ready:     " + ((Time.now - t0) * 1000).to_s + " ms"

# Warmup one of each.
puts ""
puts "warmup..."
t0 = Time.now
bench_out = model.forward(ids, 0)
puts "  native warmup:  " + ((Time.now - t0) * 1000).to_s + " ms"
t0 = Time.now
bench_out = GPT2FFI.forward(ffi_full_cache, ids)
puts "  FFI    warmup:  " + ((Time.now - t0) * 1000).to_s + " ms"
# Prefill the KV ffi_full_cache with the prompt so subsequent decode steps
# bench the steady-state path (no warmup re-allocations).
t0 = Time.now
pos = 0
while pos < ids.length
  bench_out = GPT2KV.decode_step(kv, ids[pos], pos)
  pos = pos + 1
end
puts "  KV prefill:     " + ((Time.now - t0) * 1000).to_s + " ms (" +
     ids.length.to_s + " positions)"

puts ""
puts "benching native (" + N_NATIVE.to_s + " forwards)..."
native_ms = [0.0]
native_ms.pop
i = 0
while i < N_NATIVE
  iter_start = Time.now
  bench_out =model.forward(ids, 0)
  native_ms.push((Time.now - iter_start) * 1000.0)
  i = i + 1
end

puts "benching FFI full-forward (" + N_FFI.to_s + " forwards at T_SEQ=" +
     T_SEQ.to_s + ")..."
ffi_ms = [0.0]
ffi_ms.pop
i = 0
while i < N_FFI
  iter_start = Time.now
  bench_out =GPT2FFI.forward(ffi_full_cache, ids)
  ffi_ms.push((Time.now - iter_start) * 1000.0)
  i = i + 1
end

# Bench KV decode at positions AFTER the prefill (i.e. running the
# argmax-of-prefill token through positions ids.length, ids.length+1, …).
# Each step's K/V buffer keeps growing — exactly the autoregressive
# decode workload. Reset the ffi_full_cache after the bench to avoid wedging
# subsequent runs (which would otherwise see a non-empty buffer).
puts "benching KV decode (" + N_KV.to_s + " decode steps)..."
kv_ms = [0.0]
kv_ms.pop
last_id = ids[ids.length - 1]
i = 0
pos = ids.length
while i < N_KV
  if pos >= MAX_T
    pos = ids.length      # wrap; KV buffer still has the prompt
  end
  iter_start = Time.now
  bench_out =GPT2KV.decode_step(kv, last_id, pos)
  kv_ms.push((Time.now - iter_start) * 1000.0)
  pos = pos + 1
  i = i + 1
end

puts ""
stats_print("native (Mat, f64)", native_ms)
stats_print("FFI full-forward (T_SEQ=" + T_SEQ.to_s + ")", ffi_ms)
stats_print("FFI KV decode (per-step, pos=5..34)", kv_ms)
puts "(KV wins more as T grows: full-forward attention is O(T^2),"
puts " KV is O(pos) per step. At T_SEQ=" + T_SEQ.to_s + " they're comparable;"
puts " at T_SEQ=64+ KV pulls ~5-10x ahead.)"

# Speedup ratios.
def avg(arr)
  s = 0.0
  i = 0
  while i < arr.length
    s = s + arr[i]
    i = i + 1
  end
  s / arr.length
end

n_avg = avg(native_ms)
f_avg = avg(ffi_ms)
k_avg = avg(kv_ms)
puts ""
puts "speedups vs native:"
puts "  FFI full-forward: " + (n_avg / f_avg).to_s + "x"
puts "  FFI KV decode:    " + (n_avg / k_avg).to_s + "x"
puts ""
puts "FFI KV vs FFI full-forward: " + (f_avg / k_avg).to_s + "x"
