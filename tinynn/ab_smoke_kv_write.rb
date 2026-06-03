# Validates the KV-cache write primitive: a persistent (max_T, d_head)
# buffer, into which a one-row k_new is written at position p via
# ggml_view_2d + ggml_cpy as part of the compute graph.
#
# Sequence:
#   1. Allocate persistent t_K of shape (max_T, d_head). Initial contents
#      are zero (ggml_backend_alloc_ctx_tensors zeros the buffer).
#   2. Build a graph: cpy(t_k_new, view_2d(t_K, d_head, 1, ..., pos=2)).
#   3. Upload t_k_new with known values. Compute.
#   4. Download t_K and check row 2 holds k_new's values; rows 0/1/3
#      remain zero.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

MAX_T  = 4
D_HEAD = 3
POS    = 2

sess = TinyNN.tnn_session_new(0)

# Persistent K buffer.
t_K = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T, D_HEAD)
TinyNN.tnn_finalize_weights(sess)

# Compute-side: the new k row.
t_k_new = TinyNN.tnn_input_2d_f32(sess, 1, D_HEAD)

# View into K at row POS. K's ne=[D_HEAD, MAX_T] (input_2d_f32(MAX_T,
# D_HEAD) gives ne=[cols=D_HEAD, rows=MAX_T]). Layout: row-major, so
# row r starts at offset r * D_HEAD * sizeof(float). nb1 = same.
row_bytes = D_HEAD * 4
t_K_slot  = TinyNN.tnn_view_2d(sess, t_K, D_HEAD, 1, row_bytes, POS * row_bytes)

# Compute: copy t_k_new into K_slot. The result is a view; mark output
# so the scheduler keeps it alive and we can verify by reading t_K
# afterward.
t_cpy = TinyNN.tnn_cpy(sess, t_k_new, t_K_slot)
TinyNN.tnn_set_output(t_cpy)
TinyNN.tnn_set_output(t_K)
TinyNN.tnn_realize(sess, t_cpy)

# Upload k_new = [1.0, 2.0, 3.0].
k_new = Mat.new(1, D_HEAD)
k_new.flat[0] = 1.0
k_new.flat[1] = 2.0
k_new.flat[2] = 3.0
TinyNN.upload_row_major(sess, t_k_new, k_new)

TinyNN.tnn_compute(sess)

# Download t_K (the whole buffer). Read row-major: data[r*D_HEAD + c]
# = K[r][c].
TinyNN.tnn_download(sess, t_K)
puts "K after write at row " + POS.to_s + ":"
r = 0
while r < MAX_T
  row = "  K[" + r.to_s + "] = ["
  c = 0
  while c < D_HEAD
    val = TinyNN.tnn_scratch_get(sess, r * D_HEAD + c)
    row = row + val.to_s
    if c < D_HEAD - 1
      row = row + ", "
    end
    c = c + 1
  end
  row = row + "]"
  puts row
  r = r + 1
end

# Verify: row POS = [1, 2, 3], others = [0, 0, 0].
ok = true
r = 0
while r < MAX_T
  c = 0
  while c < D_HEAD
    val = TinyNN.tnn_scratch_get(sess, r * D_HEAD + c)
    if r == POS
      expected = (c + 1).to_f
      if (val - expected).abs > 1.0e-5
        ok = false
      end
    else
      if val.abs > 1.0e-5
        ok = false
      end
    end
    c = c + 1
  end
  r = r + 1
end
puts "KV write at runtime position: " + (ok ? "OK" : "FAIL")
TinyNN.tnn_session_free(sess)
