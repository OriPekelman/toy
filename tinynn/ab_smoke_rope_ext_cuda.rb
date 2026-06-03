# CUDA parity for tnn_rope_ext (GGML_ROPE_TYPE_NEOX / rotate-half).
#
# RoPE is one of the three candidate buggy ops in the
# SmolLM2KVCuda.decode_step path (the others being view+cpy and
# soft_max_ext). The basic ab_smoke_*_cuda parity tests cover
# matmul / add / gelu / softmax / rms_norm / etc., but never
# exercised RoPE. Wrong-logits full-inference output through CUDA
# could be from any of those three; this test isolates RoPE.
#
# Strategy: build the SAME small input on both backends, apply
# tnn_rope_ext, compare element-wise. (Both go through ggml — CPU's
# ggml-cpu kernel vs ggml-cuda's kernel.) CPU's rope_ext has been
# validated by working full-inference output on CPU, so any
# mismatch implicates the CUDA kernel.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"
require_relative "../lib/tinynn_cuda"

D_HEAD    = 64       # Llama-family typical
N_HEADS   = 1        # one head — easier to inspect
N_TOK     = 1        # one position at a time (decode-step shape)
ROPE_BASE = 10000.0
POS       = 7

# Build Q-shaped input on CPU and CUDA. Per-head Q tensor in
# SmolLM2KV is ne=[d_head, 1] (one position). Use synthetic data
# (deterministic, easy to inspect).
src = Mat.new(1, D_HEAD)
i = 0
while i < D_HEAD
  src.flat[i] = (i.to_f / D_HEAD.to_f) - 0.5
  i = i + 1
end

# --- CPU path ---
cpu_sess = TinyNN.tnn_session_new(0)
t_q_cpu  = TinyNN.tnn_input_2d_f32(cpu_sess, 1, D_HEAD)    # ne=[d_head, 1]
t_pos_c  = TinyNN.tnn_input_1d_i32_ctx(cpu_sess, 1)
t_out_c  = TinyNN.tnn_rope_ext(cpu_sess, t_q_cpu, t_pos_c, D_HEAD, ROPE_BASE)
TinyNN.tnn_set_output(t_out_c)
TinyNN.tnn_realize(cpu_sess, t_out_c)
TinyNN.upload_row_major(cpu_sess, t_q_cpu, src)
TinyNN.upload_int_array(cpu_sess, t_pos_c, [POS])
TinyNN.tnn_compute(cpu_sess)
cpu_out = TinyNN.download_row_major(cpu_sess, t_out_c, 1, D_HEAD)

# --- CUDA path ---
gpu_sess = TinyNNCuda.tnn_session_new(1)
puts "CUDA backend: " + TinyNNCuda.tnn_backend_name(gpu_sess)
t_q_gpu  = TinyNNCuda.tnn_input_2d_f32(gpu_sess, 1, D_HEAD)
t_pos_g  = TinyNNCuda.tnn_input_1d_i32_ctx(gpu_sess, 1)
t_out_g  = TinyNNCuda.tnn_rope_ext(gpu_sess, t_q_gpu, t_pos_g, D_HEAD, ROPE_BASE)
TinyNNCuda.tnn_set_output(t_out_g)
TinyNNCuda.tnn_realize(gpu_sess, t_out_g)
TinyNNCuda.upload_row_major(gpu_sess, t_q_gpu, src)
TinyNNCuda.upload_int_array(gpu_sess, t_pos_g, [POS])
TinyNNCuda.tnn_compute(gpu_sess)
gpu_out = TinyNNCuda.download_row_major(gpu_sess, t_out_g, 1, D_HEAD)

# --- Compare ---
puts "input[0..3]: " + src.flat[0].to_s + " " + src.flat[1].to_s +
     " " + src.flat[2].to_s + " " + src.flat[3].to_s
puts "cpu  [0..3]: " + cpu_out.flat[0].to_s + " " + cpu_out.flat[1].to_s +
     " " + cpu_out.flat[2].to_s + " " + cpu_out.flat[3].to_s
puts "cuda [0..3]: " + gpu_out.flat[0].to_s + " " + gpu_out.flat[1].to_s +
     " " + gpu_out.flat[2].to_s + " " + gpu_out.flat[3].to_s

# Per-element diff statistics.
max_diff = 0.0
sum_diff = 0.0
n_off    = 0
i = 0
while i < D_HEAD
  d = cpu_out.flat[i] - gpu_out.flat[i]
  if d < 0
    d = -d
  end
  if d > max_diff
    max_diff = d
  end
  sum_diff = sum_diff + d
  if d > 1.0e-4
    n_off = n_off + 1
  end
  i = i + 1
end

avg_diff = sum_diff / D_HEAD.to_f
puts "max_abs_diff: " + max_diff.to_s
puts "avg_abs_diff: " + avg_diff.to_s
puts "elements off by >1e-4: " + n_off.to_s + " / " + D_HEAD.to_s
puts "match: " + (max_diff < 1.0e-4 ? "true" : "false")

TinyNN.tnn_session_free(cpu_sess)
TinyNNCuda.tnn_session_free(gpu_sess)
