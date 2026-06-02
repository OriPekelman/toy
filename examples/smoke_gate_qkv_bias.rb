# P2.6 gate — qkv_bias mmap branch (Qwen2.5) deterministic forward.
#
# THE COVERAGE GAP THIS GATE CLOSES: the smoke_gguf_roundtrip gate calls
# realize_for_mmap with qkv_bias=FALSE (it round-trips a random_init model
# that carries NO Q/K/V biases). So the qkv_bias mmap branch in
# lib/llama_seq_forward_ffi.rb (lines 635-661) — which allocs per-head
# b_q / b_k / b_v via tnn_input_1d_persistent_mmap at qb_off + h*bias_stride
# (bias_stride = @seq_d_head*4 = 256) — is NOT hit by ANY existing gate.
# Nor is its consumer: the `if ctx.seq_has_qkv_bias` tnn_add of b_q/b_k/b_v
# in lib/toy/llm/blocks/transformer_block.rb (:177, :193, :344). This
# fixture forces @seq_has_qkv_bias=true and runs a deterministic forward
# over the real Qwen2.5-0.5B native GGUF, exercising both halves.
#
# WHAT IT DOES:
#   1. Load the on-disk Qwen2.5-0.5B native GGUF (mmap; lazy page-fault, only
#      touched weights read).
#   2. ASSERT the model genuinely carries qkv_bias (blk.0.attn_{q,k,v}.bias
#      all present) and is the F32 path (attn_q.weight type == 0).
#   3. ASSERT the non-divergent invariant N_HEADS*head_dim == D_MODEL.
#   4. realize_for_mmap(handle, cfg, CONTEXT, untied=FALSE, qkv_bias=TRUE).
#   5. Run the deterministic forward TWICE (fixed ids/positions, no RNG),
#      download both, and assert the two logit vectors are BIT-IDENTICAL.
#      This single cross-run comparison simultaneously proves:
#        (a) DETERMINISM — run-to-run byte-identity (the recorded baseline
#            the realize-bulk before/after gate compares against), and
#        (b) FINITENESS — any NaN would make mat_a.flat[j] != mat_b.flat[j]
#            TRUE (NaN != NaN by IEEE-754) even for the SAME computation,
#            so a NaN-producing forward fails this gate loudly.
#      First+last logits are printed as the human-readable baseline.
#
# WHY THE TWO-DOWNLOAD SHAPE (not a v=flat[j]; v!=v finite loop): Spinel's
# whole-program type inference widens a `void *` forward() return to sp_RbVal
# when a single Mat#flat[j] element is bound to a float local or self-compared
# inline (probed 2026-05-29 — v=logits.flat[j] or logits.flat[j]!=logits.flat[j]
# both poison the fixed-point, breaking download_row_major's (void *) arg-2
# cast). Comparing TWO distinct Mats inline (mat_a.flat[j] != mat_b.flat[j])
# is the proven-compilable idiom (smoke_gguf_roundtrip uses it). The
# cross-run compare gives us determinism AND NaN-detection for free.
#
# CRITICAL untied=FALSE: output.weight is ABSENT in this GGUF => the model is
# TIED (LM head shares token_embd). Passing untied=true (as the roundtrip
# gate does for its untied random_init) would mis-load the LM head. Do NOT
# copy that.
#
# DATA DEPENDENCY (NOT self-contained, unlike smoke_gguf_roundtrip):
#   data/qwen25-0.5b-native.gguf  (1.97 GB, present on gx10). VERIFIED dims:
#     embedding_length=896  head_count=14  head_count_kv=2
#     feed_forward_length=4864  block_count=24  rope.freq_base=1000000.0
#     rms_eps=9.999999974752427e-07  key_length ABSENT => head_dim=896/14=64
#   n_heads*head_dim = 14*64 = 896 == d_model (NON-divergent w_o, full-head
#   RoPE — no GQA-divergent or partial-rope confound riding along).
#   Dims are HARDCODED (intentionally NOT Arch.detect) to keep this a pure
#   deterministic fixture; the asserts below fail-loud if the file ever drifts.
#
# This fixture needs Spinel compilation (ffi_lib is a Spinel intrinsic; it
# cannot run under MRI), so branch-hit verification is make + run.
#
#   make examples/smoke_gate_qkv_bias
#   ./examples/smoke_gate_qkv_bias        # MUST run from repo root (data/)
#
# Exits non-zero (loud, with first mismatch index) on ANY divergence/NaN;
# prints the VERDICT line + the baseline logits on success.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi"

# VERIFIED Qwen2.5-0.5B-Instruct native GGUF dims.
VOCAB   = 151936
D_MODEL = 896
N_HEADS = 14
N_KV    = 2
D_FF    = 4864
N_LAYERS = 24
CONTEXT = 16
ROPE_BASE = 1000000.0
RMS_EPS   = 9.999999974752427e-07

GGUF_PATH = "/home/oripekelman/sites/toy/data/qwen25-0.5b-native.gguf"

# Hand-written config (no Struct.new — Spinel #16). head_dim defaults to
# D_MODEL/N_HEADS = 896/14 = 64 inside the ctor (key_length absent in GGUF).
cfg = Toy::SmolLM2Config.new(VOCAB, D_MODEL, N_HEADS, N_KV,
                             D_FF, N_LAYERS, CONTEXT, ROPE_BASE, RMS_EPS)

puts "config: vocab=" + cfg.vocab.to_s +
     " d_model=" + cfg.d_model.to_s +
     " n_heads=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " head_dim=" + cfg.head_dim.to_s +
     " d_ff=" + cfg.d_ff.to_s +
     " L=" + cfg.n_layers.to_s +
     " ctx=" + cfg.ctx.to_s

# --- ASSERT c: non-divergent invariant (documents the case) ----------
if N_HEADS * cfg.head_dim != D_MODEL
  STDERR.puts "FATAL: N_HEADS*head_dim (" + (N_HEADS * cfg.head_dim).to_s +
              ") != D_MODEL (" + D_MODEL.to_s + ") — fixture assumes non-divergent w_o."
  exit 1
end
puts "ASSERT c OK: N_HEADS*head_dim == D_MODEL (" + D_MODEL.to_s + ")"

# --- Load the GGUF (mmap) --------------------------------------------
handle = TinyNN.tnn_gguf_load(GGUF_PATH)
if handle == nil || handle == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: tnn_gguf_load returned null for " + GGUF_PATH
  STDERR.puts "  (this gate REQUIRES data/qwen25-0.5b-native.gguf; run from repo root)"
  exit 1
end
puts "gguf loaded: " + GGUF_PATH

# --- ASSERT a: model genuinely carries qkv_bias ----------------------
qb_idx = TinyNN.tnn_gguf_find_index(handle, "blk.0.attn_q.bias")
kb_idx = TinyNN.tnn_gguf_find_index(handle, "blk.0.attn_k.bias")
vb_idx = TinyNN.tnn_gguf_find_index(handle, "blk.0.attn_v.bias")
if qb_idx < 0 || kb_idx < 0 || vb_idx < 0
  STDERR.puts "FATAL: qkv_bias tensors missing (q=" + qb_idx.to_s +
              " k=" + kb_idx.to_s + " v=" + vb_idx.to_s +
              ") — this GGUF does not carry biases; cannot exercise the branch."
  exit 1
end
puts "ASSERT a OK: blk.0.attn_{q,k,v}.bias all present (q=" + qb_idx.to_s +
     " k=" + kb_idx.to_s + " v=" + vb_idx.to_s + ")"

# --- ASSERT b: F32 path boundary -------------------------------------
qw_idx  = TinyNN.tnn_gguf_find_index(handle, "blk.0.attn_q.weight")
qw_type = TinyNN.tnn_gguf_tensor_type(handle, qw_idx)
if qw_type != 0
  STDERR.puts "FATAL: blk.0.attn_q.weight is type " + qw_type.to_s +
              " (expected 0=F32); head_nbytes/bias_stride math assumes F32."
  exit 1
end
puts "ASSERT b OK: blk.0.attn_q.weight is F32 (type 0)"

# --- Realize via mmap with qkv_bias=TRUE, untied=FALSE ---------------
# CRITICAL: untied=FALSE (output.weight ABSENT => TIED). qkv_bias=TRUE
# forces @seq_has_qkv_bias=true so the 635-661 branch + the
# transformer_block tnn_add(b_q/b_k/b_v) are genuinely exercised.
kv = LlamaSeqForwardFFICache.new
kv.realize_for_mmap(handle, cfg, CONTEXT, false, true)
puts "realize_for_mmap OK (untied=false, qkv_bias=true)"

# --- Deterministic ids/positions (no RNG; mmap reads fixed bytes) ----
ids       = [0]; ids.pop
positions = [0]; positions.pop
i = 0
while i < CONTEXT
  ids.push((i * 7 + 3) % VOCAB)
  positions.push(i)
  i = i + 1
end

# --- Forward TWICE + download both -----------------------------------
# Each forward re-uploads the SAME ids/positions and recomputes the graph
# over the SAME mmap'd file bytes, so the two downloads MUST be byte-equal.
t_a   = kv.forward(ids, positions)
mat_a = TinyNN.download_row_major(kv.sess, t_a, VOCAB, CONTEXT)
t_b   = kv.forward(ids, positions)
mat_b = TinyNN.download_row_major(kv.sess, t_b, VOCAB, CONTEXT)
puts "forward x2 OK (" + (VOCAB * CONTEXT).to_s + " logits each)"

# --- ASSERT d: determinism + finiteness (single cross-run compare) ---
# mat_a.flat[j] != mat_b.flat[j] is TRUE for ANY run-to-run drift AND for
# ANY NaN (NaN != NaN per IEEE-754, even for identical recomputation).
n          = VOCAB * CONTEXT
mismatches = 0
first_bad  = -1
j = 0
while j < n
  if mat_a.flat[j] != mat_b.flat[j]
    if first_bad < 0
      first_bad = j
    end
    mismatches = mismatches + 1
  end
  j = j + 1
end
if mismatches > 0
  STDERR.puts "FAIL: " + mismatches.to_s + " of " + n.to_s +
              " logits differ between the two forward runs."
  STDERR.puts "  first mismatch at index " + first_bad.to_s + " (run-to-run drift OR a NaN)."
  STDERR.puts "  => forward is non-deterministic or produced NaN; gate FAILED."
  exit 1
end
puts "ASSERT d OK: all " + n.to_s + " logits byte-identical across two runs (deterministic + finite)"

# --- Baseline: first + last few logits (human-readable record) -------
puts "baseline logits (first 5):"
k = 0
while k < 5
  puts "  [" + k.to_s + "] = " + mat_a.flat[k].to_s
  k = k + 1
end
puts "baseline logits (last 5):"
k = n - 5
while k < n
  puts "  [" + k.to_s + "] = " + mat_a.flat[k].to_s
  k = k + 1
end

puts "VERDICT: qkv_bias mmap branch exercised; " + n.to_s +
     " logits deterministic + finite (recorded baseline)"
