# CUDA mirror of examples/03_finetune_lora.rb.
# Same shape: sequence-mode forward + LoRA-Q + persistent AdamW.
#
#   make example_finetune_cuda
#   ./examples/example_finetune_cuda
#
# F32 base on CUDA — fully supported.
# Q8 base on CUDA (QLoRA) — set Q8=1 to switch to the Q8-stays-Q8
# path that bypasses the BYO-pointer padding issue. Slightly more
# load time (cudaMemcpy instead of mmap) but full GPU training works.

require_relative "../../lib/toy"
require_relative "../../lib/toy/models/toy_smollm2"
require_relative "../../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../../lib/toy/llm/engine/llama_seq_engine_cuda"

GGUF      = ENV["GGUF"]    || "data/smollm2-135m-native.gguf"
RANK      = (ENV["RANK"]   || "8").to_i
STEPS     = (ENV["STEPS"]  || "20").to_i
LR        = (ENV["LR"]     || "0.001").to_f
TRACE     = ENV["TRACE"]   || ""
TRACE_OPS = ENV["TRACE_OPS"] || ""
GRAD_DUMP = ENV["GRAD_DUMP"] || ""

TOKENS    = [12092, 4845, 253, 1429]
TARGET_ID = (ENV["TARGET_ID"] || "99").to_i

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s
puts "training (CUDA): GGUF=" + GGUF + " RANK=" + RANK.to_s + " STEPS=" + STEPS.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
seq = Toy::LLM::Engine::LlamaSeqEngineCuda.new
seq.enable_lora_q!(RANK)
seq.enable_lora_q_adamw!
if (ENV["Q8"] || "0").to_i == 1
  seq.realize_for_q8_copy(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)
else
  seq.realize_for_mmap(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)
end
seq.upload_lora_q_init!(42, 0.01)

result   = seq.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

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
  rc = TinyNNCuda.tnn_trace_open(TRACE)
  if rc != 0
    puts "trace_open failed: rc=" + rc.to_s
  else
    puts "tracing to " + TRACE
    if TRACE_OPS == "1"
      TinyNNCuda.tnn_trace_set_op_capture(1)
      puts "per-op trace enabled (TRACE_OPS=1, CUDA timings reflect enqueue latency)"
    end
  end
end

def dump_grad_row_cuda(sess, backend, step, layer, head, param_name, grad_tensor)
  n = TinyNNCuda.tnn_tensor_nelements(grad_tensor)
  TinyNNCuda.tnn_download(sess, grad_tensor)
  mn  = TinyNNCuda.tnn_scratch_min_f32(sess, n)
  mx  = TinyNNCuda.tnn_scratch_max_f32(sess, n)
  sm  = TinyNNCuda.tnn_scratch_sum_f32(sess, n)
  sq  = TinyNNCuda.tnn_scratch_sum_sq_f32(sess, n)
  nan = TinyNNCuda.tnn_scratch_nan_count_f32(sess, n)
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
  _t_step = TinyNNCuda.tnn_trace_begin("step")
  m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  if step == 1
    TinyNNCuda.tnn_graph_reset(seq.sess)
  else
    TinyNNCuda.tnn_graph_reset_grads_only(seq.sess)
  end
  TinyNNCuda.upload_int_array(seq.sess, seq.t_seq_token_ids, TOKENS)
  TinyNNCuda.upload_int_array(seq.sess, seq.t_seq_positions, positions)
  TinyNNCuda.upload_row_major(seq.sess, t_labels, m_labels)
  TinyNNCuda.upload_row_major(seq.sess, t_hp,     m_hp)
  TinyNNCuda.tnn_compute_backward(seq.sess)
  TinyNNCuda.tnn_download(seq.sess, t_loss)
  puts "step " + step.to_s.rjust(3) + ": CE=" + TinyNNCuda.tnn_scratch_get(seq.sess, 0).to_s

  if GRAD_DUMP.length > 0
    li = 0
    while li < cfg.n_layers
      blk = seq.seq_blocks_ffi[li]
      hq = 0
      while hq < cfg.n_heads
        ga = TinyNNCuda.tnn_tensor_grad(seq.sess, blk.t_seq_w_lora_a_q[hq])
        gb = TinyNNCuda.tnn_tensor_grad(seq.sess, blk.t_seq_w_lora_b_q[hq])
        if ga != nil
          dump_grad_row_cuda(seq.sess, "cuda", step, li, hq, "A", ga)
        end
        if gb != nil
          dump_grad_row_cuda(seq.sess, "cuda", step, li, hq, "B", gb)
        end
        hq = hq + 1
      end
      li = li + 1
    end
  end

  TinyNNCuda.tnn_trace_end("step", _t_step)
  step = step + 1
end

if TRACE.length > 0
  TinyNNCuda.tnn_trace_close
  puts "trace closed: " + TRACE
end
