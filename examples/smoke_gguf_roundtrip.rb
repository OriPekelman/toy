# P2.6 gate — GGUF F32 mmap round-trip parity.
#
# THE COVERAGE GAP THIS GATE CLOSES: the smoke_projection_lens dual
# gate only exercises realize_for_random_init, so the GGUF-loading
# realize paths (realize_for_mmap, realize_for_q8_copy, the GGUF half
# of realize_for_full_finetune) are NOT behaviorally gated. This
# fixture gates the F32 realize_for_mmap path end-to-end.
#
# WHAT IT DOES:
#   1. Build a random_init model (F32, untied, no bias). Its attention
#      weights are named PER-HEAD ("blk.N.attn_q.head_H.weight").
#   2. Run a forward, record the in-memory logits.
#   3. Fuse the per-head weights into the FUSED llama.cpp naming
#      ("blk.N.attn_q.weight" = head-major concat) via ToyGGUFFuser and
#      write a GGUF.
#   4. Reload that GGUF via realize_for_mmap (which slices the fused
#      tensor back into per-head views at the byte offsets the fuser
#      placed them).
#   5. Run the reloaded forward and assert the logits are BIT-IDENTICAL
#      to the in-memory ones.
#
# WHY BIT-IDENTITY IS THE RIGHT BAR: the reload feeds the SAME f32 bytes
# through the SAME forward graph. The f32->f64->f32 round-trip is
# lossless, and head-major concat is the identity layout for the mmap
# slice math. So a faithful round-trip MUST reproduce the logits exactly.
# No tolerance — strict ==. A mismatch means the round-trip is unfaithful.
#
#   make examples/smoke_gguf_roundtrip
#   SEED=0 ./examples/smoke_gguf_roundtrip
#
# Exits non-zero (with the first mismatching index + both values on
# STDERR) on ANY logit divergence; prints the VERDICT line on full match.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi"
require_relative "../lib/toy_gguf_fuse"
require_relative "../lib/toy_gguf_writer"

SEED     = (ENV["SEED"] || "0").to_i
VOCAB    = 627
D_MODEL  = 64
N_HEADS  = 4
N_KV     = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

# head_dim = d_model / n_heads = 16, so n_heads*d_head == d_model == 64.
# This keeps attn_output's mmap [d_model,d_model] equal to random_init's
# [d_model, n_heads*d_head] (no key_length divergence). Documented
# invariant — do NOT change to a shape where n_heads*d_head != d_model
# (the gate cannot catch a latent-GQA mismatch there).
cfg = Toy::SmolLM2Config.new(VOCAB, D_MODEL, N_HEADS, N_KV,
                             D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
# donor_d_in stays 0 (standard random_init; realize_for_mmap has no
# donor/lens path).

puts "config: vocab=" + cfg.vocab.to_s +
     " d_model=" + cfg.d_model.to_s +
     " n_heads=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " head_dim=" + cfg.head_dim.to_s +
     " d_ff=" + cfg.d_ff.to_s +
     " L=" + cfg.n_layers.to_s +
     " ctx=" + cfg.ctx.to_s

# --- Step 1: random_init source model -------------------------------
src = LlamaSeqForwardFFICache.new
src.realize_for_random_init(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)
puts "src realize_for_random_init OK (SEED=" + SEED.to_s + ")"

# Deterministic ids/positions (fixed, no file dependency).
ids       = [0]; ids.pop
positions = [0]; positions.pop
i = 0
while i < CONTEXT
  ids.push((i * 7 + 3) % VOCAB)
  positions.push(i)
  i = i + 1
end

# --- Step 2: in-memory baseline logits ------------------------------
t_logits_src   = src.forward(ids, positions)
baseline_logits = TinyNN.download_row_major(src.sess, t_logits_src, VOCAB, CONTEXT)
puts "baseline forward OK (" + (VOCAB * CONTEXT).to_s + " logits)"

# --- Step 3: fuse per-head weights + write GGUF ---------------------
write_sess = TinyNN.tnn_session_new(0)
plist = ToyGGUFFuser.build_fused_into_write_session(src, write_sess, true)
puts "fused plist built (" + plist.length.to_s + " tensors)"

# write_step is the public convenience API (mkdir + write + `latest`
# symlink). We use it instead of bare write() because it pins the
# weights_dir String type for Spinel's whole-program inference; the
# underlying serialization (ToyGGUFWriter.write -> tnn_gguf_w_*) is the
# same code path either way. step 0 => "step_0.gguf".
weights_dir = "/tmp/toy_gguf_roundtrip_seed" + SEED.to_s
rc = ToyGGUFWriter.write_step(cfg, plist, weights_dir, "roundtrip", 0)
if rc != 0
  STDERR.puts "FATAL: ToyGGUFWriter.write_step returned " + rc.to_s
  exit 1
end
path = weights_dir + "/step_0.gguf"
puts "GGUF written: " + path

# --- Step 4: reload via realize_for_mmap ----------------------------
handle = TinyNN.tnn_gguf_load(path)
if handle == nil || handle == TinyNN.tnn_null_ptr
  STDERR.puts "FATAL: tnn_gguf_load returned null for " + path
  exit 1
end
rl = LlamaSeqForwardFFICache.new
rl.realize_for_mmap(handle, cfg, CONTEXT, true, false)
puts "reload realize_for_mmap OK"

# F32-only boundary: assert the reloaded Q tensor is F32 (type 0). A
# non-F32 source would make the head_nbytes/stride math wrong.
q0_idx  = TinyNN.tnn_gguf_find_index(handle, "blk.0.attn_q.weight")
q0_type = TinyNN.tnn_gguf_tensor_type(handle, q0_idx)
if q0_type != 0
  STDERR.puts "FATAL: blk.0.attn_q.weight is type " + q0_type.to_s + " (expected 0=F32)"
  exit 1
end

# --- Step 5: reloaded forward + bit-identity assertion --------------
t_logits_rl  = rl.forward(ids, positions)
reload_logits = TinyNN.download_row_major(rl.sess, t_logits_rl, VOCAB, CONTEXT)
puts "reload forward OK"

n = VOCAB * CONTEXT
mismatches = 0
first_bad  = -1
j = 0
while j < n
  if baseline_logits.flat[j] != reload_logits.flat[j]
    if first_bad < 0
      first_bad = j
    end
    mismatches = mismatches + 1
  end
  j = j + 1
end

if mismatches > 0
  STDERR.puts "FAIL: " + mismatches.to_s + " of " + n.to_s + " logits differ."
  STDERR.puts "  first mismatch at index " + first_bad.to_s +
              ": baseline=" + baseline_logits.flat[first_bad].to_s +
              " reload=" + reload_logits.flat[first_bad].to_s
  STDERR.puts "  => round-trip is NOT faithful; gate FAILED (no tolerance applied)."
  exit 1
end

puts "all " + n.to_s + " logits bit-identical"
puts "VERDICT: GGUF F32 mmap round-trip bit-identical"
