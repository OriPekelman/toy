# prep/gpt2_train_min.rb — minimal inline GPT-2 training proof (CPU).
#
# "Record-from-inline-first" foundation for toy#12 / the gpt2-train arch:
# a self-contained forward + CE + backward + AdamW loop that exercises the
# two vendored ggml backward kernels (ggml_gelu_back, ggml_norm_back —
# vendor-patches/0007) end-to-end through the toy FFI, on the GPT-2-
# distinctive structure: learned positional embeddings (wte + wpe), GELU
# FFN, LayerNorm (= ggml_norm + mul γ + add β, the composite tnn_layer_norm),
# multi-head causal self-attention (per-head weights + concat, qkv biases),
# and a tied output embedding. The pre-LN block is
# ln1→attn→residual→ln2→GELU-FFN→residual. norm_back fires on every LayerNorm;
# gelu_back on the FFN. Attention uses the engine's per-head-loop pattern (each
# head has its own w_q/w_k/w_v of shape [d_head, d_model]; outputs concatenated
# then out-projected) — simpler + segfault-free vs reshape/permute, and
# N_HEADS=1 reproduces the single-head curve byte-for-byte.
#
# Attention V-bias gotcha (carry to the full arch): transpose(v)'s backward
# yields a non-contiguous gradient that ggml's repeat_back (bias broadcast)
# rejects (ops.cpp: GGML_ASSERT(nb00 == sizeof(float))). Fix: drop the V bias
# before the transpose and add it to the attention OUTPUT — exact because
# softmax rows sum to 1 (Σ_k probs·(v+b) = Σ_k probs·v + b). Q/K biases are
# fine (their grads come from matmul, already contiguous).
#
# Realize ordering (mirrors the engine's realize_for_random_init):
#   alloc all weights (ctx_w) → set_param → tnn_finalize_weights (alloc the
#   weight buffer) → upload weight inits + zero Adam m/v → build forward + CE
#   + backward + opt_step_adamw → tnn_realize_backward → train loop.
#
# Acceptance: CE loss decreases on a memorizable synthetic sequence. We do
# NOT decode tokens. This is the inline reference the byte-exact
# prep/gpt2_train_gate.rb will later record from.
#
# Spinel hygiene: explicit while loops; weights in flat Array<:ptr>; init
# Mats kept in a main-scope Array<Mat> (never passed across functions —
# Spinel has no sp_Mat_ptr_array). No Struct, no default-arg ctor, no FFI
# :str at class load. Random init via a seeded Ruby LCG.

require_relative "../lib/toy"
require_relative "../lib/toy/ffi/tinynn"

VOCAB    = (ENV["VOCAB"]    || "32").to_i
D_MODEL  = (ENV["D_MODEL"]  || "16").to_i
D_FF     = (ENV["D_FF"]     || "32").to_i
N_HEADS  = (ENV["N_HEADS"]  || "4").to_i
N_LAYERS = (ENV["N_LAYERS"] || "2").to_i
CONTEXT  = (ENV["CONTEXT"]  || "8").to_i
STEPS    = (ENV["STEPS"]    || "80").to_i
LR       = (ENV["LR"]       || "0.01").to_f
SEED     = (ENV["SEED"]     || "1234").to_i
LN_EPS   = 1.0e-5
D_HEAD   = D_MODEL / N_HEADS   # multi-head attention (per-head weights + concat)

$rng_state = SEED
def next_rand_unit
  $rng_state = (($rng_state * 1103515245) + 12345) & 0x7fffffff
  (($rng_state >> 8).to_f / 8388608.0) - 1.0   # ~[-1,1)
end

def random_mat(rows, cols, scale)
  m = Mat.new(rows, cols)
  n = rows * cols
  i = 0
  while i < n
    m.flat[i] = next_rand_unit * scale
    i = i + 1
  end
  m
end

def const_mat(rows, cols, value)
  m = Mat.new(rows, cols)
  n = rows * cols
  i = 0
  while i < n
    m.flat[i] = value
    i = i + 1
  end
  m
end

sess = TinyNN.tnn_session_new(0)

# Trainable weights in parallel ptr arrays; init Mats parallel to weights.
weights = [TinyNN.tnn_null_ptr]; weights.pop
opt_m   = [TinyNN.tnn_null_ptr]; opt_m.pop
opt_v   = [TinyNN.tnn_null_ptr]; opt_v.pop
inits   = [Mat.new(1, 1)];       inits.pop

# alloc-only (no upload yet — buffers don't exist until tnn_finalize_weights).
def alloc_w2(sess, weights, opt_m, opt_v, inits, rows, cols, init_mat)
  w = TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols)
  weights.push(w)
  opt_m.push(TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols))
  opt_v.push(TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols))
  inits.push(init_mat)
  w
end

def alloc_w1(sess, weights, opt_m, opt_v, inits, n, init_mat)
  w = TinyNN.tnn_input_1d_f32_persistent(sess, n)
  weights.push(w)
  opt_m.push(TinyNN.tnn_input_1d_f32_persistent(sess, n))
  opt_v.push(TinyNN.tnn_input_1d_f32_persistent(sess, n))
  inits.push(init_mat)
  w
end

# Global embeddings: wte[vocab,d] (ggml ne=[d,vocab]), wpe[context,d].
wte = alloc_w2(sess, weights, opt_m, opt_v, inits, VOCAB,   D_MODEL, random_mat(VOCAB,   D_MODEL, 0.02))
wpe = alloc_w2(sess, weights, opt_m, opt_v, inits, CONTEXT, D_MODEL, random_mat(CONTEXT, D_MODEL, 0.02))

# Per-layer block weights. GPT-2 pre-LN block:
#   x = x + attn(ln1(x));  x = x + ffn(ln2(x))
# Single-head attention (d_head == D_MODEL) to avoid the head-reshape; the
# qkv/out projections carry learned biases (GPT-2 convention).
ln1_g = [TinyNN.tnn_null_ptr]; ln1_g.pop
ln1_b = [TinyNN.tnn_null_ptr]; ln1_b.pop
w_q   = [TinyNN.tnn_null_ptr]; w_q.pop
b_q   = [TinyNN.tnn_null_ptr]; b_q.pop
w_k   = [TinyNN.tnn_null_ptr]; w_k.pop
b_k   = [TinyNN.tnn_null_ptr]; b_k.pop
w_v   = [TinyNN.tnn_null_ptr]; w_v.pop
b_v   = [TinyNN.tnn_null_ptr]; b_v.pop
w_o   = [TinyNN.tnn_null_ptr]; w_o.pop
b_o   = [TinyNN.tnn_null_ptr]; b_o.pop
ln2_g = [TinyNN.tnn_null_ptr]; ln2_g.pop
ln2_b = [TinyNN.tnn_null_ptr]; ln2_b.pop
fc_W  = [TinyNN.tnn_null_ptr]; fc_W.pop
fc_b  = [TinyNN.tnn_null_ptr]; fc_b.pop
pr_W  = [TinyNN.tnn_null_ptr]; pr_W.pop
pr_b  = [TinyNN.tnn_null_ptr]; pr_b.pop
# q/k/v weights are per-(layer, head), flat-indexed [li*N_HEADS + h]; each is
# [d_head, d_model]. w_o is per-layer [d_model, n_heads*d_head == d_model].
li = 0
while li < N_LAYERS
  ln1_g.push(alloc_w1(sess, weights, opt_m, opt_v, inits, D_MODEL, const_mat(1, D_MODEL, 1.0)))
  ln1_b.push(alloc_w1(sess, weights, opt_m, opt_v, inits, D_MODEL, const_mat(1, D_MODEL, 0.0)))
  hh = 0
  while hh < N_HEADS
    w_q.push(alloc_w2(sess, weights, opt_m, opt_v, inits, D_HEAD, D_MODEL, random_mat(D_HEAD, D_MODEL, 0.02)))
    b_q.push(alloc_w1(sess, weights, opt_m, opt_v, inits, D_HEAD, const_mat(1, D_HEAD, 0.0)))
    w_k.push(alloc_w2(sess, weights, opt_m, opt_v, inits, D_HEAD, D_MODEL, random_mat(D_HEAD, D_MODEL, 0.02)))
    b_k.push(alloc_w1(sess, weights, opt_m, opt_v, inits, D_HEAD, const_mat(1, D_HEAD, 0.0)))
    w_v.push(alloc_w2(sess, weights, opt_m, opt_v, inits, D_HEAD, D_MODEL, random_mat(D_HEAD, D_MODEL, 0.02)))
    b_v.push(alloc_w1(sess, weights, opt_m, opt_v, inits, D_HEAD, const_mat(1, D_HEAD, 0.0)))
    hh = hh + 1
  end
  w_o.push(  alloc_w2(sess, weights, opt_m, opt_v, inits, D_MODEL, D_MODEL, random_mat(D_MODEL, D_MODEL, 0.02)))
  b_o.push(  alloc_w1(sess, weights, opt_m, opt_v, inits, D_MODEL, const_mat(1, D_MODEL, 0.0)))
  ln2_g.push(alloc_w1(sess, weights, opt_m, opt_v, inits, D_MODEL, const_mat(1, D_MODEL, 1.0)))
  ln2_b.push(alloc_w1(sess, weights, opt_m, opt_v, inits, D_MODEL, const_mat(1, D_MODEL, 0.0)))
  fc_W.push( alloc_w2(sess, weights, opt_m, opt_v, inits, D_FF, D_MODEL, random_mat(D_FF, D_MODEL, 0.02)))
  fc_b.push( alloc_w1(sess, weights, opt_m, opt_v, inits, D_FF, const_mat(1, D_FF, 0.0)))
  pr_W.push( alloc_w2(sess, weights, opt_m, opt_v, inits, D_MODEL, D_FF, random_mat(D_MODEL, D_FF, 0.02)))
  pr_b.push( alloc_w1(sess, weights, opt_m, opt_v, inits, D_MODEL, const_mat(1, D_MODEL, 0.0)))
  li = li + 1
end

# Final LayerNorm.
lnf_g = alloc_w1(sess, weights, opt_m, opt_v, inits, D_MODEL, const_mat(1, D_MODEL, 1.0))
lnf_b = alloc_w1(sess, weights, opt_m, opt_v, inits, D_MODEL, const_mat(1, D_MODEL, 0.0))

# Mark every weight trainable, then finalize the weight buffer.
gi = 0
while gi < weights.length
  TinyNN.tnn_set_param(weights[gi])
  gi = gi + 1
end
TinyNN.tnn_finalize_weights(sess)

# Buffers now exist: upload weight inits, zero Adam m/v.
gk = 0
while gk < weights.length
  TinyNN.upload_row_major(sess, weights[gk], inits[gk])
  TinyNN.tnn_zero_tensor(sess, opt_m[gk])
  TinyNN.tnn_zero_tensor(sess, opt_v[gk])
  gk = gk + 1
end

# --- forward graph -----------------------------------------------------------
t_tok = TinyNN.tnn_input_1d_i32(sess, CONTEXT)
t_pos = TinyNN.tnn_input_1d_i32(sess, CONTEXT)

x_tok = TinyNN.tnn_get_rows(sess, wte, t_tok)   # [d, T]
x_pos = TinyNN.tnn_get_rows(sess, wpe, t_pos)   # [d, T]
x = TinyNN.tnn_add(sess, x_tok, x_pos)
TinyNN.tnn_set_output(x)

att_scale = 1.0 / Math.sqrt(D_HEAD.to_f)
li2 = 0
while li2 < N_LAYERS
  # --- attention sub-block: x = x + attn(ln1(x)) --- per-head loop + concat.
  h1 = TinyNN.tnn_layer_norm(sess, x, ln1_g[li2], ln1_b[li2], LN_EPS)
  head_out = TinyNN.tnn_null_ptr
  hh2 = 0
  while hh2 < N_HEADS
    hi = li2 * N_HEADS + hh2
    q  = TinyNN.tnn_add(sess, TinyNN.tnn_matmul(sess, w_q[hi], h1), b_q[hi])  # [d_head,T]
    k  = TinyNN.tnn_add(sess, TinyNN.tnn_matmul(sess, w_k[hi], h1), b_k[hi])  # [d_head,T]
    # V carries NO bias before the transpose: transpose(v)'s backward yields a
    # non-contiguous grad that ggml's repeat_back (bias broadcast) rejects.
    # Since softmax rows sum to 1, Σ_k probs·(v+b_v) = (Σ_k probs·v) + b_v, so
    # b_v is added to the per-head OUTPUT (exact, keeps the grad contiguous).
    v  = TinyNN.tnn_matmul(sess, w_v[hi], h1)                                 # [d_head,T]
    scores = TinyNN.tnn_scale(sess, TinyNN.tnn_matmul(sess, k, q), att_scale) # [T_k,T_q]
    scores = TinyNN.tnn_diag_mask_inf(sess, scores, 0)
    probs  = TinyNN.tnn_softmax(sess, scores)
    v_t    = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, v), CONTEXT, D_HEAD) # [T_k,d_head]
    head   = TinyNN.tnn_add(sess, TinyNN.tnn_matmul(sess, v_t, probs), b_v[hi])       # [d_head,T]
    if hh2 == 0
      head_out = head
    else
      head_out = TinyNN.tnn_concat(sess, head_out, head, 0)   # along d → [d_model,T]
    end
    hh2 = hh2 + 1
  end
  ao = TinyNN.tnn_add(sess, TinyNN.tnn_matmul(sess, w_o[li2], head_out), b_o[li2]) # [d,T]
  x  = TinyNN.tnn_add(sess, x, ao)   # residual
  TinyNN.tnn_set_output(x)

  # --- FFN sub-block: x = x + ffn(ln2(x)) ---
  h    = TinyNN.tnn_layer_norm(sess, x, ln2_g[li2], ln2_b[li2], LN_EPS)
  pre  = TinyNN.tnn_add(sess, TinyNN.tnn_matmul(sess, fc_W[li2], h), fc_b[li2])   # [d_ff,T]
  act  = TinyNN.tnn_gelu(sess, pre)
  mlp  = TinyNN.tnn_add(sess, TinyNN.tnn_matmul(sess, pr_W[li2], act), pr_b[li2]) # [d,T]
  x    = TinyNN.tnn_add(sess, x, mlp)   # residual
  TinyNN.tnn_set_output(x)
  li2 = li2 + 1
end

x_final = TinyNN.tnn_layer_norm(sess, x, lnf_g, lnf_b, LN_EPS)
TinyNN.tnn_set_output(x_final)
logits  = TinyNN.tnn_matmul(sess, wte, x_final)   # tied unembed → [vocab, T]
TinyNN.tnn_set_output(logits)

# --- CE loss + backward + AdamW ---------------------------------------------
t_labels = TinyNN.tnn_input_2d_f32(sess, CONTEXT, VOCAB)   # ggml ne=[vocab,T]
t_hp     = TinyNN.tnn_input_1d_f32(sess, 7)
t_loss   = TinyNN.tnn_cross_entropy_loss(sess, logits, t_labels)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)

TinyNN.tnn_build_forward_only(sess, t_loss)
TinyNN.tnn_build_backward(sess)

gj = 0
while gj < weights.length
  tw = weights[gj]
  tg = TinyNN.tnn_tensor_grad(sess, tw)
  to = TinyNN.tnn_opt_step_adamw(sess, tw, tg, opt_m[gj], opt_v[gj], t_hp)
  TinyNN.tnn_extend_backward_graph(sess, to)
  gj = gj + 1
end
TinyNN.tnn_realize_backward(sess)

# --- synthetic memorizable data ---------------------------------------------
seq_ids = [0]; seq_ids.pop
positions = [0]; positions.pop
ti = 0
while ti < CONTEXT
  seq_ids.push((ti * 7 + 3) % VOCAB)
  positions.push(ti)
  ti = ti + 1
end

# Shift-by-one next-token one-hot labels (ne=[vocab,T]).
m_labels = const_mat(CONTEXT, VOCAB, 0.0)
ti2 = 0
while ti2 < CONTEXT
  nxt = ti2 + 1 < CONTEXT ? ti2 + 1 : ti2
  tgt = seq_ids[nxt]
  m_labels.flat[ti2 * VOCAB + tgt] = 1.0
  ti2 = ti2 + 1
end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0

losses = [0.0]; losses.pop
step = 1
while step <= STEPS
  m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  if step == 1
    TinyNN.tnn_graph_reset(sess)
  else
    TinyNN.tnn_graph_reset_grads_only(sess)
  end
  TinyNN.upload_int_array(sess, t_tok, seq_ids)
  TinyNN.upload_int_array(sess, t_pos, positions)
  TinyNN.upload_row_major(sess, t_labels, m_labels)
  TinyNN.upload_row_major(sess, t_hp, m_hp)
  TinyNN.tnn_compute_backward(sess)
  TinyNN.tnn_download(sess, t_loss)
  loss = TinyNN.tnn_scratch_get(sess, 0)
  losses.push(loss)
  if step <= 5 || step % 10 == 0 || step == STEPS
    puts "step " + step.to_s.rjust(4) + ": CE=" + loss.to_s
  end
  step = step + 1
end

initial = losses[0]
final   = losses[losses.length - 1]
ratio   = final / initial
puts "initial CE = " + initial.to_s
puts "final   CE = " + final.to_s
puts "ratio      = " + ratio.to_s
if ratio >= 0.9
  puts "VERDICT: NOT learning (gelu_back / norm_back path FAILED)"
  exit 1
else
  puts "VERDICT: learning — gelu_back + norm_back train end-to-end"
  exit 0
end
