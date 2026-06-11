# P4 smoke: ggml_flash_attn_ext binding correctness.
#
# Builds a tiny attention block two ways in the same session and
# compares outputs at d_head=4, T_q=1, T_k=4, n_head=1, n_head_kv=1,
# scale=1/sqrt(4)=0.5, no mask:
#
#   Path A (reference): manual scale → softmax → matmul triplet over
#                       per-head 2D tensors (the path our SmolLM2KV
#                       decode currently uses).
#   Path B (flash):     tnn_flash_attn_ext over 3D persistent tensors
#                       (ne[3]=1 batch).
#
# Acceptance: max abs diff(A, B) < 1e-5 per output element.
#
# Limitation: flash_attn_ext input layouts are 4D (d_head, T, n_head,
# batch) so we can't share tensors between paths — we upload identical
# data to two separate tensor families. The reference path squeezes
# n_head=1 so its 2D tensors match the flash path's 3D after squeezing
# n_head=1.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

D_HEAD = 4
T_Q    = 1
T_K    = 4
N_HEAD = 1
N_KV   = 1
SCALE  = 0.5     # = 1/sqrt(d_head=4)

# Hand-picked Q/K/V values so the result is hand-verifiable.
Q_VALS  = [1.0, 2.0, 3.0, 4.0]                         # [d_head]
K_VALS  = [1.0, 0.0, 0.0, 0.0,                          # K[0] = e1
           0.0, 1.0, 0.0, 0.0,                          # K[1] = e2
           0.0, 0.0, 1.0, 0.0,                          # K[2] = e3
           0.0, 0.0, 0.0, 1.0]                          # K[3] = e4
V_VALS  = [10.0, 11.0, 12.0, 13.0,                      # V[0]
           20.0, 21.0, 22.0, 23.0,                      # V[1]
           30.0, 31.0, 32.0, 33.0,                      # V[2]
           40.0, 41.0, 42.0, 43.0]                      # V[3]

sess = TinyNN.tnn_session_new(0)

# Path A (reference) tensors: 2D persistent for d_head × T.
t_q_a = TinyNN.tnn_input_2d_f32_persistent(sess, T_Q, D_HEAD)   # ne=[D_HEAD, T_Q]
t_k_a = TinyNN.tnn_input_2d_f32_persistent(sess, T_K, D_HEAD)   # ne=[D_HEAD, T_K]
t_v_a = TinyNN.tnn_input_2d_f32_persistent(sess, D_HEAD, T_K)   # ne=[T_K, D_HEAD] — V stored transposed

# Path B (flash) tensors: 3D persistent (n_head=1 means ne[2]=1).
t_q_b = TinyNN.tnn_input_3d_f32_persistent(sess, D_HEAD, T_Q, N_HEAD)
t_k_b = TinyNN.tnn_input_3d_f32_persistent(sess, D_HEAD, T_K, N_KV)
t_v_b = TinyNN.tnn_input_3d_f32_persistent(sess, D_HEAD, T_K, N_KV)

# Reference attention: scale → softmax → matmul.
t_scores_a = TinyNN.tnn_matmul(sess, t_k_a, t_q_a)   # ne=[T_K, T_Q]
t_scaled_a = TinyNN.tnn_scale(sess, t_scores_a, SCALE)
t_attn_a   = TinyNN.tnn_softmax(sess, t_scaled_a)    # ne=[T_K, T_Q]
t_out_a    = TinyNN.tnn_matmul(sess, t_v_a, t_attn_a) # ne=[D_HEAD, T_Q]

# Flash attention: one fused op.
t_out_b = TinyNN.tnn_flash_attn_ext(sess, t_q_b, t_k_b, t_v_b, nil,
                                    SCALE, 0.0, 0.0)

# Both outputs need a buffer post-compute. Mark both, add path A's
# subtree, then realize on path B (which appends B's subtree).
TinyNN.tnn_set_output(t_out_a)
TinyNN.tnn_set_output(t_out_b)
TinyNN.tnn_finalize_weights(sess)
TinyNN.tnn_add_to_graph(sess, t_out_a)

# Upload reference path inputs. V_VALS is per-row [V[0]...V[T_K-1]],
# each row d_head wide. For the reference path's V tensor with
# ne=[T_K, D_HEAD] (row-major d_head rows of T_K cols), we transpose:
# v_a[i_d, i_t] = V_VALS[i_t * D_HEAD + i_d].
TinyNN.tnn_upload_from_float_array(sess, t_q_a, Q_VALS, D_HEAD * T_Q)
TinyNN.tnn_upload_from_float_array(sess, t_k_a, K_VALS, D_HEAD * T_K)
v_a_buf = [0.0]; v_a_buf.pop
i_d = 0
while i_d < D_HEAD
  i_t = 0
  while i_t < T_K
    v_a_buf.push(V_VALS[i_t * D_HEAD + i_d])
    i_t = i_t + 1
  end
  i_d = i_d + 1
end
TinyNN.tnn_upload_from_float_array(sess, t_v_a, v_a_buf, T_K * D_HEAD)

# Flash path: same row-major data, just with ne[2]=1 for n_head/n_kv.
TinyNN.tnn_upload_from_float_array(sess, t_q_b, Q_VALS, D_HEAD * T_Q * N_HEAD)
TinyNN.tnn_upload_from_float_array(sess, t_k_b, K_VALS, D_HEAD * T_K * N_KV)
TinyNN.tnn_upload_from_float_array(sess, t_v_b, V_VALS, D_HEAD * T_K * N_KV)

TinyNN.tnn_realize(sess, t_out_b)
TinyNN.tnn_compute(sess)

# Reference output: ne=[D_HEAD, T_Q].
TinyNN.tnn_download(sess, t_out_a)
out_a = [0.0]; out_a.pop
i = 0
while i < D_HEAD * T_Q
  out_a.push(TinyNN.tnn_scratch_get(sess, i))
  i = i + 1
end

# Flash output: ne=[D_HEAD, N_HEAD, T_Q, 1]. With N_HEAD=T_Q=1, total
# size is D_HEAD elements (= 4).
TinyNN.tnn_download(sess, t_out_b)
out_b = [0.0]; out_b.pop
i = 0
while i < D_HEAD * N_HEAD * T_Q
  out_b.push(TinyNN.tnn_scratch_get(sess, i))
  i = i + 1
end

puts "reference path output: " + out_a.to_s
puts "flash attn   output:  " + out_b.to_s

max_diff = 0.0
i = 0
while i < D_HEAD
  d = out_a[i] - out_b[i]
  ad = d < 0 ? 0.0 - d : d
  max_diff = ad > max_diff ? ad : max_diff
  i = i + 1
end
puts "max |A - B| = " + max_diff.to_s

if max_diff < 1.0e-5
  puts "OK: flash_attn_ext output matches reference path within 1e-5"
else
  puts "FAIL: outputs diverge"
  exit 1
end
