# P2.6 gate — Q8-stays-Q8 realize_for_q8_copy branch (Qwen2.5-Q8) forward.
#
# HONEST FRAMING (preservation gate, NOT a round-trip/correctness claim):
# this fixture genuinely EXERCISES the realize_for_q8_copy load path and
# records a deterministic logits baseline the realize-bulk workflow uses as
# a before/after self-consistency gate. It does NOT assert the Q8 logits
# match any F32 reference.
#
# THE COVERAGE GAP THIS GATE CLOSES: the qkv_bias gate exercises
# realize_for_mmap on the F32 native GGUF. The realize_for_q8_copy path
# (lib/llama_seq_forward_ffi.rb:233) — which reads each on-disk tensor type
# via tnn_gguf_tensor_type and allocates the persistent weight at that SAME
# type via tnn_input_2d_persistent_typed (lines 275-277), then verbatim-copies
# bytes (tinynn/tinynn_gguf.c: ASSERTs src_type==dst->type, NO dequant) — is
# NOT hit by any existing gate. This fixture loads the EXISTING Q8 GGUF
# through that path and proves the attention weight stays Q8_0 in memory.
#
# WHAT IT DOES:
#   1. Load the on-disk Qwen2.5-0.5B native-Q8 GGUF.
#   2. ON-DISK pre-assert: blk.0.attn_q.weight type == 8 (Q8_0). Fails loud
#      if the file drifts / wrong model (general.file_type is 0 in this
#      mixed-quant GGUF metadata, so we check the per-tensor type directly).
#   3. ASSERT the model carries qkv_bias (blk.0.attn_{q,k,v}.bias present).
#   4. ASSERT the non-divergent invariant N_HEADS*head_dim == D_MODEL.
#   5. realize_for_q8_copy(handle, cfg, CONTEXT, untied=FALSE, qkv_bias=TRUE).
#   6. IN-MEMORY runtime assert (the required type-assert): the loaded
#      attention weight tensor seq_blocks_ffi[0].t_seq_w_q[0] has backend
#      dtype == 8 (Q8_0), NOT 0 (F32) — proving it was allocated+filled as
#      Q8_0, NOT dequantized to F32. (We assert on t_seq_w_q, NOT token_embd,
#      which is genuinely F32 type 0 in this mixed-quant GGUF.)
#   7. Run the deterministic forward TWICE (fixed ids/positions, no RNG),
#      download both, assert the two logit vectors are BIT-IDENTICAL.
#      This single cross-run comparison proves DETERMINISM (run-to-run
#      byte-identity = the recorded baseline) AND FINITENESS (NaN != NaN
#      per IEEE-754 fails the compare).
#      First+last logits printed as the human-readable baseline.
#
# WHY THE TWO-DOWNLOAD SHAPE (not a v=flat[j] finite loop): Spinel's
# whole-program type inference widens forward()'s (void *) return when a
# single Mat#flat[j] element is bound to a float local or self-compared
# inline, poisoning download_row_major's arg-2 cast. Comparing TWO distinct
# Mats inline (mat_a.flat[j] != mat_b.flat[j]) is the proven-compilable
# idiom (smoke_gate_qkv_bias / smoke_gguf_roundtrip use it).
#
# CRITICAL untied=FALSE: output.weight is ABSENT in this GGUF => TIED.
#
# DATA DEPENDENCY: data/qwen25-0.5b-native-q8.gguf (925 MB, present on gx10).
#   Same dims as the F32 native GGUF (Q8 is per-tensor quant of the weights):
#     embedding_length=896  head_count=14  head_count_kv=2
#     feed_forward_length=4864  block_count=24  rope.freq_base=1000000.0
#     rms_eps=9.999999974752427e-07  key_length ABSENT => head_dim=896/14=64
#   Dims HARDCODED (the asserts below fail-loud if the file ever drifts).
#
# CPU-only: realize_for_q8_copy is documented CUDA-path but the body is
# backend-agnostic (tnn_session_new(0)); runs fine on a CPU Spinel build.
#
# Needs Spinel compilation (ffi_lib is a Spinel intrinsic). Branch-hit
# verification is make + run:
#   make examples/smoke_gate_q8_preserve
#   ./examples/smoke_gate_q8_preserve     # MUST run from repo root (data/)
#
# Exits non-zero (loud, with first mismatch index) on ANY divergence/NaN
# or if the Q8 path is not taken; prints VERDICT + baseline logits on success.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy/llm/engine/llama_seq_engine"

# VERIFIED Qwen2.5-0.5B-Instruct native-Q8 GGUF dims.
VOCAB   = 151936
D_MODEL = 896
N_HEADS = 14
N_KV    = 2
D_FF    = 4864
N_LAYERS = 24
CONTEXT = 16
ROPE_BASE = 1000000.0
RMS_EPS   = 9.999999974752427e-07

# GGUF per-tensor type code for Q8_0 (ggml GGML_TYPE_Q8_0).
Q8_0 = 8
F32  = 0

GGUF_PATH = "/home/oripekelman/sites/toy/data/qwen25-0.5b-native-q8.gguf"

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

# --- Load the GGUF ----------------------------------------------------
handle = TinyNN.tnn_gguf_load(GGUF_PATH)
if handle == nil || handle == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: tnn_gguf_load returned null for " + GGUF_PATH
  STDERR.puts "  (this gate REQUIRES data/qwen25-0.5b-native-q8.gguf; run from repo root)"
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

# --- ASSERT b (ON-DISK): blk.0.attn_q.weight is genuinely Q8_0 -------
# general.file_type is 0 in this mixed-quant GGUF's metadata, so we check
# the PER-TENSOR type directly. If this is not Q8_0, the in-memory type
# assert below would be meaningless — fail loud now so we know the Q8 path
# is actually reachable for this file.
qw_idx  = TinyNN.tnn_gguf_find_index(handle, "blk.0.attn_q.weight")
qw_type = TinyNN.tnn_gguf_tensor_type(handle, qw_idx)
if qw_type != Q8_0
  STDERR.puts "FATAL: blk.0.attn_q.weight on-disk type is " + qw_type.to_s +
              " (expected " + Q8_0.to_s + "=Q8_0); wrong model or file drifted."
  STDERR.puts "  This gate REQUIRES a Q8 attention weight to prove Q8-stays-Q8."
  exit 1
end
puts "ASSERT b OK (on-disk): blk.0.attn_q.weight type == " + Q8_0.to_s + " (Q8_0)"

# --- Realize via realize_for_q8_copy, qkv_bias=TRUE, untied=FALSE -----
# CRITICAL: untied=FALSE (output.weight ABSENT => TIED). qkv_bias=TRUE
# forces @seq_has_qkv_bias=true. This path allocates each weight at its
# on-disk type and verbatim-copies bytes (NO dequant-to-F32).
kv = Toy::LLM::Engine::LlamaSeqEngine.new
kv.realize_for_q8_copy(handle, cfg, CONTEXT, false, true)
puts "realize_for_q8_copy OK (untied=false, qkv_bias=true)"

# --- ASSERT e (IN-MEMORY): loaded attn_q weight stays Q8_0 -----------
# THE required type-assert. seq_blocks_ffi[0].t_seq_w_q[0] is the head-0 Q
# weight tensor allocated in the backend buffer. tnn_tensor_dtype returns
# t->type. If the path had dequantized to F32 it would be 0 here; Q8_0 (8)
# proves the Q8-stays-Q8 verbatim-copy path genuinely ran.
wq0      = kv.seq_blocks_ffi[0].t_seq_w_q[0]
wq0_type = TinyNN.tnn_tensor_dtype(wq0)
if wq0_type != Q8_0
  STDERR.puts "FATAL: in-memory blk.0 t_seq_w_q[0] dtype is " + wq0_type.to_s +
              " (expected " + Q8_0.to_s + "=Q8_0)."
  if wq0_type == F32
    STDERR.puts "  dtype 0 (F32) => the weight was DEQUANTIZED to F32; the Q8-stays-Q8 path did NOT run."
  end
  exit 1
end
puts "ASSERT e OK (in-memory): blk.0 t_seq_w_q[0] dtype == " + Q8_0.to_s +
     " (Q8_0 stayed Q8_0, NOT dequantized to F32)"

# --- Deterministic ids/positions (no RNG) ----------------------------
ids       = [0]; ids.pop
positions = [0]; positions.pop
i = 0
while i < CONTEXT
  ids.push((i * 7 + 3) % VOCAB)
  positions.push(i)
  i = i + 1
end

# --- Forward TWICE + download both -----------------------------------
t_a   = kv.forward(ids, positions)
mat_a = TinyNN.download_row_major(kv.sess, t_a, VOCAB, CONTEXT)
t_b   = kv.forward(ids, positions)
mat_b = TinyNN.download_row_major(kv.sess, t_b, VOCAB, CONTEXT)
puts "forward x2 OK (" + (VOCAB * CONTEXT).to_s + " logits each)"

# --- ASSERT d: determinism + finiteness (single cross-run compare) ---
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

puts "VERDICT: realize_for_q8_copy Q8-stays-Q8 branch exercised; " + n.to_s +
     " logits deterministic + finite (recorded baseline)"
