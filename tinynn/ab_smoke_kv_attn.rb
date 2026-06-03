# Single-head, single-step KV-cache attention via FFI. Validates the
# math + plumbing for M2's "decode one new token using cached K/V"
# pattern.
#
# Layout:
#   K storage:  ne=[d_head, max_T]  -- "row r" along ne1 is key r
#   V storage:  ne=[max_T, d_head]  -- transposed; ne0=max_T is the
#                                      k_dim for the head_out matmul
#
# Per step at position pos:
#   - K_after  = set_rows(K, k_new, [pos])             -- row write
#   - V_after  = set_2d(V, v_new_t, max_T*4, pos*4)    -- column write
#   - scores   = mul_mat(K_after, q)                    -- ne=[max_T, 1]
#   - attn     = soft_max_ext(scores, mask, 1/sqrt(d_head), 0)
#   - head_out = mul_mat(V_after, attn)                 -- ne=[d_head, 1]

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

MAX_T  = 4
D_HEAD = 8
POS    = 2

sess = TinyNN.tnn_session_new(0)

# Persistent K (d_head, max_T) and V (max_T, d_head).
t_K = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T,  D_HEAD)  # ne=[d_head, max_T]
t_V = TinyNN.tnn_input_2d_f32_persistent(sess, D_HEAD, MAX_T)   # ne=[max_T, d_head]
TinyNN.tnn_finalize_weights(sess)

# Compute inputs.
t_q      = TinyNN.tnn_input_2d_f32(sess, 1,      D_HEAD)  # ne=[d_head, 1]
t_k_new  = TinyNN.tnn_input_2d_f32(sess, 1,      D_HEAD)  # ne=[d_head, 1] -- one new key row
# v_new shaped so it can ggml_set_2d into V[:, pos]. The slot is a
# (1, d_head) shape in ggml-terms (ne=[1, d_head]). v_new ditto.
t_v_new  = TinyNN.tnn_input_2d_f32(sess, D_HEAD, 1)       # ne=[1, d_head]
t_idx    = TinyNN.tnn_input_1d_i32(sess, 1)               # current pos (runtime)
t_mask   = TinyNN.tnn_input_2d_f32(sess, 1,      MAX_T)   # ne=[max_T, 1]

# === Build graph ===
# K[idx[0]] <- k_new. set_rows returns a view of K with K's shape.
t_K_after = TinyNN.tnn_set_rows(sess, t_K, t_k_new, t_idx)
# V[:, POS] <- v_new via a column write. set_2d returns a view of V
# with V's shape, so the matmul downstream reads the modified V.
# POS is baked here as a literal offset; multi-step decode would
# rebuild this part of the graph per step.
t_V_after = TinyNN.tnn_set_2d(sess, t_V, t_v_new, MAX_T * 4, POS * 4)

scale = 1.0 / Math.sqrt(D_HEAD.to_f)
t_scores   = TinyNN.tnn_matmul(sess, t_K_after, t_q)             # ne=[max_T, 1]
t_attn     = TinyNN.tnn_soft_max_ext(sess, t_scores, t_mask, scale, 0.0)
t_head_out = TinyNN.tnn_matmul(sess, t_V_after, t_attn)          # ne=[d_head, 1]

TinyNN.tnn_set_output(t_head_out)
TinyNN.tnn_realize(sess, t_head_out)

# === Pre-fill K and V for positions 0..POS-1 ===
def fill_seq(rows, cols, base)
  out = Mat.new(rows, cols)
  n = rows * cols
  i = 0
  while i < n
    out.flat[i] = base + i.to_f * 0.01
    i = i + 1
  end
  out
end

K_host = fill_seq(MAX_T, D_HEAD, 0.1)   # K[pos][feature]
# Put noise into positions > POS so we can verify the mask zeroes
# their contribution.
i = (POS + 1) * D_HEAD
while i < MAX_T * D_HEAD
  K_host.flat[i] = 1000.0
  i = i + 1
end
V_host = fill_seq(MAX_T, D_HEAD, -0.2)
i = (POS + 1) * D_HEAD
while i < MAX_T * D_HEAD
  V_host.flat[i] = -1000.0
  i = i + 1
end

# Upload K row-major. K.ne=[d_head, max_T] -> data[d + p*d_head] =
# K_host[p][d] which matches K_host.flat[p*d_head + d].
TinyNN.upload_row_major(sess, t_K, K_host)

# Upload V transposed (its storage layout). V.ne=[max_T, d_head] ->
# data[p + f*max_T] = V_host[p][f].
v_buf = Mat.new(D_HEAD, MAX_T)
p = 0
while p < MAX_T
  f = 0
  while f < D_HEAD
    v_buf.flat[f * MAX_T + p] = V_host.flat[p * D_HEAD + f]
    f = f + 1
  end
  p = p + 1
end
TinyNN.upload_row_major(sess, t_V, v_buf)

# === Upload step inputs ===
q_host     = fill_seq(1, D_HEAD, 0.5)
k_new_host = fill_seq(1, D_HEAD, 2.0)
v_new_host = fill_seq(1, D_HEAD, 3.0)
TinyNN.upload_row_major(sess, t_q,     q_host)
TinyNN.upload_row_major(sess, t_k_new, k_new_host)
# v_new (1, d_head) row-major. The receiving tensor t_v_new has
# ne=[1, d_head] so data[i0 + i1*1] for i0 in [0,1), i1 in [0,d_head).
# Row-major data is [v_new[0][0], v_new[0][1], ...] which equals
# data[0 + f*1] = v_new[0][f]. Match.
TinyNN.upload_row_major(sess, t_v_new, v_new_host)
TinyNN.upload_int_array(sess, t_idx, [POS])

# Mask: 0.0 for positions 0..POS, -inf-ish for POS+1..MAX_T-1.
mask = Mat.new(1, MAX_T)
i = 0
while i < MAX_T
  if i <= POS
    mask.flat[i] = 0.0
  else
    mask.flat[i] = -1.0e30
  end
  i = i + 1
end
TinyNN.upload_row_major(sess, t_mask, mask)

# === Compute ===
TinyNN.tnn_compute(sess)
ffi_head_out = TinyNN.download_row_major(sess, t_head_out, 1, D_HEAD)

# === Native reference ===
def overlay_row(mat, row, new_row, cols)
  i = 0
  while i < cols
    mat.flat[row * cols + i] = new_row.flat[i]
    i = i + 1
  end
end
overlay_row(K_host, POS, k_new_host, D_HEAD)
overlay_row(V_host, POS, v_new_host, D_HEAD)

n_valid = POS + 1
scores = Array.new(n_valid, 0.0)
p = 0
while p < n_valid
  s = 0.0
  f = 0
  while f < D_HEAD
    s = s + q_host.flat[f] * K_host.flat[p * D_HEAD + f]
    f = f + 1
  end
  scores[p] = s / Math.sqrt(D_HEAD.to_f)
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

native_head_out = Mat.new(1, D_HEAD)
f = 0
while f < D_HEAD
  s = 0.0
  p = 0
  while p < n_valid
    s = s + scores[p] * V_host.flat[p * D_HEAD + f]
    p = p + 1
  end
  native_head_out.flat[f] = s
  f = f + 1
end

max_d = 0.0
i = 0
while i < D_HEAD
  d = native_head_out.flat[i] - ffi_head_out.flat[i]
  if d < 0
    d = -d
  end
  if d > max_d
    max_d = d
  end
  i = i + 1
end
puts "POS=" + POS.to_s + " d_head=" + D_HEAD.to_s
puts "native head_out[0..3]=" + native_head_out.flat[0].to_s + " " +
     native_head_out.flat[1].to_s + " " + native_head_out.flat[2].to_s + " " +
     native_head_out.flat[3].to_s
puts "ffi    head_out[0..3]=" + ffi_head_out.flat[0].to_s + " " +
     ffi_head_out.flat[1].to_s + " " + ffi_head_out.flat[2].to_s + " " +
     ffi_head_out.flat[3].to_s
puts "max_abs_diff=" + max_d.to_s
puts "M2 single-step decode parity: " + (max_d < 1.0e-4 ? "OK" : "FAIL")
TinyNN.tnn_session_free(sess)
