# CUDA parity for tnn_soft_max_ext.
#
# Attention softmax in SmolLM2KV.decode_step uses
# `soft_max_ext(scores, mask_or_nil, scale, max_bias=0.0)`. This
# isolates that op: build a small score matrix on CPU and CUDA,
# apply soft_max_ext with a scale, compare.
#
# Pattern mirror from lib/toy_smollm2_ffi_kv.rb:
#   t_attn = TinyNN.tnn_soft_max_ext(@sess, t_scores, t_mask_or_null, scale, 0.0)

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"
require_relative "../lib/toy/ffi/tinynn_cuda"

ROWS  = 4
COLS  = 16
SCALE = 0.125   # 1/sqrt(64) — typical attention scale

# Synthetic score matrix.
src = Mat.new(ROWS, COLS)
i = 0
while i < ROWS * COLS
  src.flat[i] = (i.to_f - (ROWS * COLS / 2).to_f) * 0.1
  i = i + 1
end

# --- CPU ---
cpu_sess = TinyNN.tnn_session_new(0)
t_x_c = TinyNN.tnn_input_2d_f32(cpu_sess, ROWS, COLS)
t_out_c = TinyNN.tnn_soft_max_ext(cpu_sess, t_x_c, TinyNN.tnn_null_ptr, SCALE, 0.0)
TinyNN.tnn_set_output(t_out_c)
TinyNN.tnn_realize(cpu_sess, t_out_c)
TinyNN.upload_row_major(cpu_sess, t_x_c, src)
TinyNN.tnn_compute(cpu_sess)
cpu_out = TinyNN.download_row_major(cpu_sess, t_out_c, ROWS, COLS)

# --- CUDA ---
gpu_sess = TinyNNCuda.tnn_session_new(1)
puts "CUDA backend: " + TinyNNCuda.tnn_backend_name(gpu_sess)
t_x_g = TinyNNCuda.tnn_input_2d_f32(gpu_sess, ROWS, COLS)
t_out_g = TinyNNCuda.tnn_soft_max_ext(gpu_sess, t_x_g, TinyNNCuda.tnn_null_ptr, SCALE, 0.0)
TinyNNCuda.tnn_set_output(t_out_g)
TinyNNCuda.tnn_realize(gpu_sess, t_out_g)
TinyNNCuda.upload_row_major(gpu_sess, t_x_g, src)
TinyNNCuda.tnn_compute(gpu_sess)
gpu_out = TinyNNCuda.download_row_major(gpu_sess, t_out_g, ROWS, COLS)

# Compare per-element.
max_diff = 0.0
n_off    = 0
i = 0
n_total = ROWS * COLS
while i < n_total
  d = cpu_out.flat[i] - gpu_out.flat[i]
  if d < 0
    d = -d
  end
  if d > max_diff
    max_diff = d
  end
  if d > 1.0e-5
    n_off = n_off + 1
  end
  i = i + 1
end

puts "max_abs_diff: " + max_diff.to_s
puts "elements off by >1e-5: " + n_off.to_s + " / " + n_total.to_s
print "cpu  row 0: "
i = 0
while i < COLS
  print " " + cpu_out.flat[i].to_s
  i = i + 1
end
puts ""
print "cuda row 0: "
i = 0
while i < COLS
  print " " + gpu_out.flat[i].to_s
  i = i + 1
end
puts ""
puts "match: " + (max_diff < 1.0e-5 ? "true" : "false")

TinyNN.tnn_session_free(cpu_sess)
TinyNNCuda.tnn_session_free(gpu_sess)
