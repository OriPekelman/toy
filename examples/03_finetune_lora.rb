# LoRA fine-tune via the sequence-mode forward graph.
#
# Trains rank-8 LoRA-Q adapters across the model's attention heads
# while the base weights stay frozen (mmap'd from the GGUF, no
# duplication). One forward + backward + opt_step per training
# step, T positions at a time. AdamW state lives in ctx_w next to
# the adapters and survives across step rebuilds.
#
#   make example_finetune
#   ./examples/example_finetune                                    # F32 base
#   GGUF=data/qwen25-0.5b-native-q8.gguf ./examples/example_finetune  # QLoRA (Q8 base)
#
# CUDA variant: make example_finetune_cuda. CUDA path supports F32
# base today; Q8 base on CUDA hits a BYO-pointer buffer padding
# limitation (vendor-patches/0002 caveat) — pending a follow-up
# patch. F4 QLoRA is fully supported on the CPU path.
#
# Memory:
#   SmolLM2-135M F32 + r=8 LoRA + Adam ≈ 540 MB.
#   Qwen2.5-0.5B Q8 + r=8 LoRA + Adam ≈ 480 MB.
#   Larger Q8 bases (1.5B/3B/7B) need proportionally more.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/llama_seq_forward_ffi"

GGUF      = ENV["GGUF"]    || "data/smollm2-135m-native.gguf"
RANK      = (ENV["RANK"]   || "8").to_i
STEPS     = (ENV["STEPS"]  || "20").to_i
LR        = (ENV["LR"]     || "0.001").to_f
TRACE     = ENV["TRACE"]   || ""    # path to Chrome Trace JSON; empty disables
GRAD_DUMP = ENV["GRAD_DUMP"] || ""  # path to CSV; empty disables. Writes per-(layer,head,param) grad stats after each step.

# T=4 prompt; CE objective pushes every position's argmax toward
# TARGET_ID. Real SFT (shifted next-token labels + a position mask)
# is the natural extension; the wiring is the same.
TOKENS    = [12092, 4845, 253, 1429]
TARGET_ID = (ENV["TARGET_ID"] || "99").to_i

if !File.exist?(GGUF)
  puts "example_finetune: cannot find " + GGUF
  puts ""
  puts "LoRA fine-tune needs a native-layout GGUF — the base weights"
  puts "are mmap'd in place, which only works when the converter wrote"
  puts "them in HF [out, in] layout (toy.ggml_native=true)."
  puts ""
  puts "Build one from the HuggingFace checkpoint:"
  puts ""
  puts "  ./prep/convert_smollm2_to_gguf.py --ggml-native \\"
  puts "      --out " + GGUF
  puts ""
  puts "Then re-run this example."
  exit 1
end

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s
puts "training: GGUF=" + GGUF + " RANK=" + RANK.to_s + " STEPS=" + STEPS.to_s

gguf = TinyNN.tnn_gguf_load(GGUF)
seq = LlamaSeqForwardFFICache.new
seq.enable_lora_q!(RANK)
seq.enable_lora_q_adamw!
seq.realize_for_mmap(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)
seq.upload_lora_q_init!(42, 0.01)

result   = seq.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

# Vocab × T one-hot label matrix in Mat(T, vocab) layout.
m_labels = Mat.new(TOKENS.length, cfg.vocab)
i = 0
while i < TOKENS.length * cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
ti = 0
while ti < TOKENS.length
  m_labels.flat[ti * cfg.vocab + TARGET_ID] = 1.0
  ti = ti + 1
end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0

positions = [0, 1, 2, 3]

if TRACE.length > 0
  rc = TinyNN.tnn_trace_open(TRACE)
  if rc != 0
    puts "trace_open failed: rc=" + rc.to_s + " (TRACE=" + TRACE + ")"
  else
    puts "tracing to " + TRACE
  end
end

# Helper: compute stats over the scratch buffer's first n f32 elements
# (Mat-roundtrip is overkill — we just want min/max/mean/L2/nan-count
# per tensor for the CPU/CUDA bisection). Caller must have just done
# tnn_download(sess, tensor). Prints one CSV row to stdout (or whoever
# captures it). Backend name is part of the row so a CSV merge is
# trivial.
def dump_grad_row(sess, backend, step, layer, head, param_name, grad_tensor)
  n = TinyNN.tnn_tensor_nelements(grad_tensor)
  TinyNN.tnn_download(sess, grad_tensor)
  mn  = TinyNN.tnn_scratch_min_f32(sess, n)
  mx  = TinyNN.tnn_scratch_max_f32(sess, n)
  sm  = TinyNN.tnn_scratch_sum_f32(sess, n)
  sq  = TinyNN.tnn_scratch_sum_sq_f32(sess, n)
  nan = TinyNN.tnn_scratch_nan_count_f32(sess, n)
  mean = sm / n.to_f
  l2   = Math.sqrt(sq)
  puts "GRAD," + backend + "," + step.to_s + "," + layer.to_s + "," +
       head.to_s + "," + param_name + "," + n.to_s + "," +
       mn.to_s + "," + mx.to_s + "," + mean.to_s + "," +
       l2.to_s + "," + nan.to_s
end

if GRAD_DUMP.length > 0
  puts "GRAD_HEADER,backend,step,layer,head,param,n,min,max,mean,l2,nan_count"
end

step = 1
while step <= STEPS
  _t_step = TinyNN.tnn_trace_begin("step")
  m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  if step == 1
    TinyNN.tnn_graph_reset(seq.sess)
  else
    TinyNN.tnn_graph_reset_grads_only(seq.sess)
  end
  TinyNN.upload_int_array(seq.sess, seq.t_seq_token_ids, TOKENS)
  TinyNN.upload_int_array(seq.sess, seq.t_seq_positions, positions)
  TinyNN.upload_row_major(seq.sess, t_labels, m_labels)
  TinyNN.upload_row_major(seq.sess, t_hp,     m_hp)
  TinyNN.tnn_compute_backward(seq.sess)
  TinyNN.tnn_download(seq.sess, t_loss)
  puts "step " + step.to_s.rjust(3) + ": CE=" + TinyNN.tnn_scratch_get(seq.sess, 0).to_s

  if GRAD_DUMP.length > 0
    li = 0
    while li < cfg.n_layers
      blk = seq.seq_blocks_ffi[li]
      hq = 0
      while hq < cfg.n_heads
        ga = TinyNN.tnn_tensor_grad(seq.sess, blk.t_seq_w_lora_a_q[hq])
        gb = TinyNN.tnn_tensor_grad(seq.sess, blk.t_seq_w_lora_b_q[hq])
        if ga != nil
          dump_grad_row(seq.sess, "cpu", step, li, hq, "A", ga)
        end
        if gb != nil
          dump_grad_row(seq.sess, "cpu", step, li, hq, "B", gb)
        end
        hq = hq + 1
      end
      li = li + 1
    end
  end

  TinyNN.tnn_trace_end("step", _t_step)
  step = step + 1
end

if TRACE.length > 0
  TinyNN.tnn_trace_close
  puts "trace closed: " + TRACE
end
