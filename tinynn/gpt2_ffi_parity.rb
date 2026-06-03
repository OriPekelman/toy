# FFI parity probe: build a persistent GPT2 ggml graph at the same
# T_SEQ as the prompt, upload distilgpt2 weights, run one forward,
# dump the last-row logits to data/ours_ffi_logits.txt.
#
# Pair with prep/parity.py compare --ours data/ours_ffi_logits.txt to
# check that the FFI path matches HF transformers within F32 noise.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/models/gpt2"
require_relative "../lib/gpt2_ffi"
require_relative "../lib/toy/io/gguf_load"
require_relative "../lib/toy/train/training"   # parse_ids

VOCAB    = 50257
D_MODEL  = 768
D_FF     = 3072
N_HEADS  = 12
N_LAYERS = 6

# T_SEQ pinned at prompt length. The padded-prompt path (T_SEQ > prompt
# length) needs the position-embedding slice to match the runtime
# positions; that's a generation-time concern, not a parity-test one.
T_SEQ    = 5

IDS_PATH = "data/prompt_ids.txt"
GGUF     = "data/distilgpt2-f32.gguf"
OUT      = "data/ours_ffi_logits.txt"

def read_ids(path)
  raw = ["?"]
  raw.pop
  File.open(path, "r") do |f|
    f.each_line { |line| raw.push(line.chomp) }
  end
  parse_ids(raw[0])
end

ids = read_ids(IDS_PATH)
puts "prompt: " + ids.length.to_s + " tokens"
if ids.length != T_SEQ
  puts "WARNING: T_SEQ=" + T_SEQ.to_s + " but prompt length=" + ids.length.to_s +
       "; re-encode or change T_SEQ."
end

# Build native GPT2LM as the weight source.
puts "constructing GPT2LM (native, for weight source)..."
model = GPT2LM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, 1024)
ok = GGUFLoad.load_gpt2(model, GGUF)
if !ok
  puts "load failed"
else
  puts "loaded native."

  puts ""
  puts "realizing FFI graph at T_SEQ=" + T_SEQ.to_s + "..."
  t0 = Time.now
  cache = GPT2FullForwardFFICache.new
  cache.realize_for(T_SEQ, D_MODEL, D_FF, N_HEADS, N_LAYERS, VOCAB)
  puts "  realized in " + ((Time.now - t0) * 1000).to_s + " ms"

  puts "uploading weights (this is the slow scratch-buffer transpose)..."
  t0 = Time.now
  pos_slice = GPT2FFI.make_pos_slice(model, T_SEQ)
  GPT2FFI.upload_from(cache, model, pos_slice)
  puts "  uploaded in " + ((Time.now - t0) * 1000).to_s + " ms"

  puts ""
  puts "FFI forward..."
  t0 = Time.now
  logits = GPT2FFI.forward(cache, ids)
  puts "  forward: " + ((Time.now - t0) * 1000).to_s + " ms  (shape " +
       logits.nrows.to_s + " x " + logits.ncols.to_s + ")"

  last = logits.nrows - 1
  File.open(OUT, "w") do |f|
    v = 0
    while v < VOCAB
      f.write(logits.flat[last * VOCAB + v].to_s)
      if v < VOCAB - 1
        f.write(" ")
      end
      v = v + 1
    end
    f.write("\n")
  end
  puts "wrote " + VOCAB.to_s + " logits to " + OUT
  puts "compare with: ./prep/parity.py compare --ours " + OUT

  best   = 0
  best_v = logits.flat[last * VOCAB]
  v = 1
  while v < VOCAB
    val = logits.flat[last * VOCAB + v]
    if val > best_v
      best_v = val
      best   = v
    end
    v = v + 1
  end
  puts "FFI argmax: id=" + best.to_s + "  logit=" + best_v.to_s
end
