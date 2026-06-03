# Multi-step KV-cache smoke using cpy-into-view (the working M2-proof
# pattern). The earlier set_rows/set_2d multi-step smoke fails because
# ggml_set_2d / ggml_set_rows return a *copy* of the destination (via
# ggml_dup_tensor) — the write doesn't land in the persistent buffer
# across reset_for_rebuild cycles.
#
# cpy(src, view_of_K) uses ggml_view_tensor on the destination, so the
# write goes straight to K's underlying buffer; subsequent decode
# steps see the accumulated history. This smoke validates that.
#
# Shapes match what GPT2KVFFICache will use:
#   K: ne=[d_head, max_T]   each "row" along ne1 is one position
#   V: ne=[max_T, d_head]   transposed; ne0=max_T is the k_dim for
#                            head_out = matmul(V_view, attn)

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

MAX_T  = 8
D_HEAD = 6

sess = TinyNN.tnn_session_new(0)

# Persistent K (d_head, max_T) and V (max_T, d_head). Both zero-init'd
# by ggml_backend_alloc_ctx_tensors.
t_K = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T,  D_HEAD)
t_V = TinyNN.tnn_input_2d_f32_persistent(sess, D_HEAD, MAX_T)
TinyNN.tnn_finalize_weights(sess)

# Mirror the K/V history on the host so we can compute a native
# reference at each step.
K_history = Mat.new(MAX_T, D_HEAD)
V_history = Mat.new(MAX_T, D_HEAD)

def fill_seq(rows, cols, base, stride)
  out = Mat.new(rows, cols)
  n = rows * cols
  i = 0
  while i < n
    out.flat[i] = base + i.to_f * stride
    i = i + 1
  end
  out
end

def overlay_row(mat, row, src, cols)
  i = 0
  while i < cols
    mat.flat[row * cols + i] = src.flat[i]
    i = i + 1
  end
end

def native_head_out(q, k_hist, v_hist, n_valid, d_head)
  inv_sqrt = 1.0 / Math.sqrt(d_head.to_f)
  scores = Array.new(n_valid, 0.0)
  p = 0
  while p < n_valid
    s = 0.0
    f = 0
    while f < d_head
      s = s + q.flat[f] * k_hist.flat[p * d_head + f]
      f = f + 1
    end
    scores[p] = s * inv_sqrt
    p = p + 1
  end
  mx = scores[0]
  i = 1
  while i < n_valid
    if scores[i] > mx
      mx = scores[i]
    end
    i = i + 1
  end
  expsum = 0.0
  i = 0
  while i < n_valid
    scores[i] = Math.exp(scores[i] - mx)
    expsum = expsum + scores[i]
    i = i + 1
  end
  i = 0
  while i < n_valid
    scores[i] = scores[i] / expsum
    i = i + 1
  end
  out = Mat.new(1, d_head)
  f = 0
  while f < d_head
    s = 0.0
    p = 0
    while p < n_valid
      s = s + scores[p] * v_hist.flat[p * d_head + f]
      p = p + 1
    end
    out.flat[f] = s
    f = f + 1
  end
  out
end

scale = 1.0 / Math.sqrt(D_HEAD.to_f)
all_ok = true
max_observed = 0.0

pos = 0
while pos < MAX_T
  TinyNN.tnn_reset_for_rebuild(sess)

  # Fresh compute tensors for THIS step.
  t_q       = TinyNN.tnn_input_2d_f32(sess, 1,      D_HEAD)  # ne=[d_head, 1]
  t_k_new   = TinyNN.tnn_input_2d_f32(sess, 1,      D_HEAD)  # ne=[d_head, 1]
  t_v_new   = TinyNN.tnn_input_2d_f32(sess, D_HEAD, 1)       # ne=[1, d_head]

  # Write k_new into K[pos] = first d_head elements at offset pos*d_head*4.
  # K view: ne=[d_head, 1], nb1=d_head*4 (row stride), offset=pos*d_head*4
  t_K_slot = TinyNN.tnn_view_2d(sess, t_K, D_HEAD, 1, D_HEAD * 4, pos * D_HEAD * 4)
  t_cpy_k  = TinyNN.tnn_cpy(sess, t_k_new, t_K_slot)

  # Write v_new into V[:, pos] — strided write. V has ne=[max_T, d_head].
  # Column pos of V is at positions stride max_T from offset pos*4.
  # View: ne=[1, d_head], nb1=max_T*4, offset=pos*4
  t_V_slot = TinyNN.tnn_view_2d(sess, t_V, 1, D_HEAD, MAX_T * 4, pos * 4)
  t_cpy_v  = TinyNN.tnn_cpy(sess, t_v_new, t_V_slot)

  # The cpy ops aren't reachable from t_head_out (their result tensors
  # are 1-row views, not used downstream); add them to the graph
  # explicitly so the scheduler runs them before the matmul reads K/V.
  TinyNN.tnn_add_to_graph(sess, t_cpy_k)
  TinyNN.tnn_add_to_graph(sess, t_cpy_v)

  # History views of size pos+1.
  t_K_hist = TinyNN.tnn_view_2d(sess, t_K, D_HEAD, pos + 1, D_HEAD * 4, 0)
  t_V_hist = TinyNN.tnn_view_2d(sess, t_V, pos + 1, D_HEAD, MAX_T  * 4, 0)

  # Attention math.
  t_scores  = TinyNN.tnn_matmul(sess, t_K_hist, t_q)     # ne=[pos+1, 1]
  t_scaled  = TinyNN.tnn_scale(sess, t_scores, scale)
  t_attn    = TinyNN.tnn_softmax(sess, t_scaled)         # softmax along ne0
  t_head_out = TinyNN.tnn_matmul(sess, t_V_hist, t_attn) # ne=[d_head, 1]
  TinyNN.tnn_set_output(t_head_out)

  TinyNN.tnn_realize(sess, t_head_out)

  q_host     = fill_seq(1, D_HEAD, 0.1 + pos.to_f * 0.05, 0.01)
  k_new_host = fill_seq(1, D_HEAD, 1.0 + pos.to_f * 0.1,  0.02)
  v_new_host = fill_seq(1, D_HEAD, -0.5 + pos.to_f * 0.07, 0.03)

  TinyNN.upload_row_major(sess, t_q,     q_host)
  TinyNN.upload_row_major(sess, t_k_new, k_new_host)
  TinyNN.upload_row_major(sess, t_v_new, v_new_host)

  TinyNN.tnn_compute(sess)
  ffi_out = TinyNN.download_row_major(sess, t_head_out, 1, D_HEAD)

  overlay_row(K_history, pos, k_new_host, D_HEAD)
  overlay_row(V_history, pos, v_new_host, D_HEAD)
  nat_out = native_head_out(q_host, K_history, V_history, pos + 1, D_HEAD)

  max_d = 0.0
  i = 0
  while i < D_HEAD
    d = nat_out.flat[i] - ffi_out.flat[i]
    if d < 0
      d = -d
    end
    if d > max_d
      max_d = d
    end
    i = i + 1
  end
  if max_d > max_observed
    max_observed = max_d
  end
  ok = max_d < 1.0e-4
  if !ok
    all_ok = false
  end
  puts "pos=" + pos.to_s + ":  max_abs_diff=" + max_d.to_s + " " + (ok ? "OK" : "FAIL")
  pos = pos + 1
end

puts ""
puts "max across all positions: " + max_observed.to_s
puts "cpy+view KV multi-step: " + (all_ok ? "OK" : "FAIL")
TinyNN.tnn_session_free(sess)
