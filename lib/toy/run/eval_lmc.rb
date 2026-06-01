# lib/toy/run/eval_lmc.rb — Spinel-compiled LMC (Linear Mode Connectivity)
# COMPUTE runner. Given two toy checkpoints A + B, evaluates loss along the
# interpolation θ_α = (1-α)·θ_A + α·θ_B on a fixed sequence, emits one `eval`
# event per α (kind="eval", name="lmc", extra.alpha=α), and prints a
# deterministic α→loss curve to STDOUT.
#
# This is the lib-side home of `toy eval lmc`'s compute, mirroring the
# CRuby→runner COMPUTE BRIDGE of the other eval/infer/train runners. The CRuby
# CLI shell (lib/toy/core/cli/eval.rb) cannot compute in-process — every
# ffi_lib-bearing lib crashes under MRI — so it locates the toy root, builds
# this runner (`make libexec/toy-eval-lmc`), and shells out to it via Open3
# with a CONTROLLED ENV.
#
# It is a FUSED-FORMAT-AWARE port of examples/08_lmc.rb. 08_lmc predates the
# #3 fused-llama checkpoint format; `toy train` now writes a STANDARD FUSED
# llama GGUF (projection-lens folded + per-head attention FUSED to
# blk.N.attn_q.weight at write time). The session realized here is the
# random-init per-HEAD layout (blk.N.attn_q.head_H.weight), so attention is
# blended by SLICING the H-th block out of the fused GGUF tensor. Non-attention
# tensors match the GGUF name directly.
#
# CONTRACT (read from ENV only — lib-vs-example scope, no experiment config
# baked in):
#   LMC_A, LMC_B   — required .gguf paths (the two checkpoints to interpolate)
#   LMC_ALPHAS     — comma-separated α grid (default "0,0.25,0.5,0.75,1.0")
#   CONTEXT        — SEQ_LEN (default "32"; MUST equal train.rb CONTEXT so the
#                    endpoints match each ckpt's trained loss)
#   SEED           — random-init seed for the eval session (default "0")
#   TAO_RUN_DIR    — when set, emit events.jsonl HERE (FILE only); when empty,
#                    stdout-only.
#   TOY_RUN_ID     — run_id string for events (default "lmc")
#
# Backend: CPU only. A cuda LMC twin is a later slice. This file is deliberately
# ABSENT from MIRRORABLE in prep/gen_cuda_mirror.rb (exactly like the other
# CPU runners). `make verify-mirrors` stays green.
#
# DETERMINISM: pinned ckpts + fixed seq + lr=0 forward + ggml-internal CE (no
# Ruby libm) → byte-exact cross-platform, like the train loss curves. This is
# what prep/lmc_gate.rb gates against prep/fixtures/lmc_baseline.txt.
#
# OUTPUT (byte-exact prefix line the CLI + gate parse): one flat line per α, in
# α-grid order:
#   "lmc: <alpha>=<loss>"
#
# Spinel hygiene (landmine #16): hand-built String-concat JSON (no #{}
# interpolation, no Math.*); no Struct.new; .split(",")+.to_f into a
# popped-empty array literal.

require_relative "../../toy"
require_relative "../../toy_smollm2"
require_relative "../../llama_seq_forward_ffi"
require_relative "../../toy_drift_grad"

LMC_A    = ENV["LMC_A"]      || ""
LMC_B    = ENV["LMC_B"]      || ""
ALPHAS_S = ENV["LMC_ALPHAS"] || "0,0.25,0.5,0.75,1.0"
SEQ_LEN  = (ENV["CONTEXT"] || "32").to_i
SEED     = (ENV["SEED"]    || "0").to_i
RUN_DIR  = ENV["TAO_RUN_DIR"] || ""
RUN_ID   = ENV["TOY_RUN_ID"]  || "lmc"

if LMC_A == "" || LMC_B == ""
  STDERR.puts "toy-eval-lmc: LMC_A and LMC_B are required"
  exit 1
end

EVENTS = RUN_DIR.length > 0 ? (RUN_DIR + "/events.jsonl") : ""
if EVENTS != ""
  TinyNN.tnn_events_open(EVENTS)
end

# Parse α-grid (Spinel-safe: popped-empty literal + .split + .to_f).
alphas_arr = [0.0]; alphas_arr.pop
ALPHAS_S.split(",").each do |a|
  alphas_arr.push(a.to_f)
end

# Load both checkpoints (config read from A; the two must share architecture).
ggA = TinyNN.tnn_gguf_load(LMC_A)
ggB = TinyNN.tnn_gguf_load(LMC_B)
if ggA == nil || ggA == TinyNN.tnn_null_ptr
  STDERR.puts "toy-eval-lmc: cannot open LMC_A=" + LMC_A; exit 1
end
if ggB == nil || ggB == TinyNN.tnn_null_ptr
  STDERR.puts "toy-eval-lmc: cannot open LMC_B=" + LMC_B; exit 1
end

vocab     = TinyNN.tnn_gguf_get_u32(ggA, "llama.vocab_size")
d_model   = TinyNN.tnn_gguf_get_u32(ggA, "llama.embedding_length")
d_ff      = TinyNN.tnn_gguf_get_u32(ggA, "llama.feed_forward_length")
n_heads   = TinyNN.tnn_gguf_get_u32(ggA, "llama.attention.head_count")
n_kv      = TinyNN.tnn_gguf_get_u32(ggA, "llama.attention.head_count_kv")
n_layers  = TinyNN.tnn_gguf_get_u32(ggA, "llama.block_count")
ctx_len   = TinyNN.tnn_gguf_get_u32(ggA, "llama.context_length")
rope_base = TinyNN.tnn_gguf_get_f32(ggA, "llama.rope.freq_base")
rms_eps   = TinyNN.tnn_gguf_get_f32(ggA, "llama.attention.layer_norm_rms_epsilon")

cfg = Toy::SmolLM2Config.new(vocab, d_model, n_heads, n_kv, d_ff,
                             n_layers, ctx_len, rope_base, rms_eps)

# D_HEAD = d_model / n_heads. Per-head fused-tensor slice geometry.
D_HEAD = d_model / n_heads

# Fixed eval sequence — first TinyStories line, truncated/padded to SEQ_LEN.
# Byte-identical to train.rb:282-291 so endpoints match each ckpt's loss.
raw        = File.read("data/ts_seqs.txt")
first_line = raw.split("\n")[0]
parts      = first_line.split(" ")
seq_ids    = [0]; seq_ids.pop
k = 0
while k < parts.length && k < SEQ_LEN
  seq_ids.push(parts[k].to_i)
  k = k + 1
end
while seq_ids.length < SEQ_LEN; seq_ids.push(0); end

# Emit run_start (FILE only).
if EVENTS != ""
  t_open = TinyNN.tnn_events_now_seconds
  rs  = "{\"kind\":\"run_start\",\"phase\":\"eval\""
  rs  = rs + ",\"t\":"            + t_open.to_s
  rs  = rs + ",\"started_at\":\"" + TinyNN.tnn_events_iso8601_now + "\""
  rs  = rs + ",\"run_id\":\""     + RUN_ID + "\""
  rs  = rs + ",\"name\":\"lmc\""
  rs  = rs + ",\"host\":{\"name\":\"" + TinyNN.tnn_provenance_host_name + "\""
  rs  = rs + ",\"os\":\""             + TinyNN.tnn_provenance_host_os   + "\""
  rs  = rs + ",\"arch\":\""           + TinyNN.tnn_provenance_host_arch + "\"}"
  rs  = rs + ",\"backend\":{\"kind\":\"cpu\"}"
  rs  = rs + "}"
  TinyNN.tnn_events_emit(rs)
end

# Cache for the FULL fused Q/K/V buffers — read each fused GGUF tensor exactly
# ONCE (not n_heads times). Keyed by fused_name; α-invariant so it can live
# across all α (the GGUF buffers never change). Two parallel arrays
# (Spinel-safe; no Hash<String,Mat>): names + their Mat buffers, for A and B.
fused_names = [""]; fused_names.pop
fused_a     = [Mat.new(1, 1)]; fused_a.pop
fused_b     = [Mat.new(1, 1)]; fused_b.pop

# Look up (or read+cache) the full fused tensor for `fused_name` from ckpt A.
def fused_lookup_a(fused_names, fused_a, fused_b, ggA, ggB, fused_name, nel_full)
  fi = 0
  while fi < fused_names.length
    if fused_names[fi] == fused_name
      return fi
    end
    fi = fi + 1
  end
  # Miss — read both A and B once.
  idx_a = TinyNN.tnn_gguf_find_index(ggA, fused_name)
  idx_b = TinyNN.tnn_gguf_find_index(ggB, fused_name)
  if idx_a < 0 || idx_b < 0
    STDERR.puts "toy-eval-lmc: missing fused " + fused_name +
                " in A=" + idx_a.to_s + " B=" + idx_b.to_s
    return -1
  end
  ma = Mat.new(1, nel_full)
  mb = Mat.new(1, nel_full)
  TinyNN.tnn_gguf_read_f32_to_doubles(ggA, idx_a, ma.flat, nel_full)
  TinyNN.tnn_gguf_read_f32_to_doubles(ggB, idx_b, mb.flat, nel_full)
  fused_names.push(fused_name)
  fused_a.push(ma)
  fused_b.push(mb)
  return fused_names.length - 1
end

# For each α, build a fresh session, blend weights, run forward+loss.
ai = 0
while ai < alphas_arr.length
  alpha = alphas_arr[ai]

  fcache = LlamaSeqForwardFFICache.new
  # untied=TRUE (arg5) so the session allocates output.weight (the fused
  # ckpt carries output.weight; train.rb folds with untied=true). We do NOT
  # set cfg.donor_d_in (it stays 0) so the session embed is the STANDARD
  # [vocab, d_model] tensor matching the folded fused ckpt.
  fcache.realize_for_random_init(cfg, SEQ_LEN, 1, 0, true, false, SEED, 1.0)

  result   = fcache.build_training_step
  t_loss   = result[0]
  t_labels = result[1]
  t_hp     = result[2]

  # Blend every PARAM tensor: θ = (1-α)·θ_A + α·θ_B.
  plist = ToyDriftGrad.params(fcache.sess)
  pi = 0
  while pi < plist.length
    t    = plist[pi]
    name = TinyNN.tnn_tensor_name(t)
    nel  = TinyNN.tnn_tensor_nelements(t)

    if name.include?(".head_")
      # Per-head attention weight: blk.<N>.attn_<q|k|v>.head_<H>.weight.
      # The fused GGUF carries blk.<N>.attn_<q|k|v>.weight (full matrix);
      # slice the H-th block out of it. Parse via .split (no regex captures).
      pre_parts = name.split(".head_")          # ["blk.N.attn_X", "H.weight"]
      fused_name = pre_parts[0] + ".weight"
      tail_parts = pre_parts[1].split(".")        # ["H", "weight"]
      head_h = tail_parts[0].to_i

      # nel_full = (q? n_heads : n_kv) * D_HEAD * d_model. Detect q vs k/v by
      # the fused_name's attn_ component (q has n_heads rows; k/v have n_kv).
      is_q = fused_name.include?(".attn_q.")
      rows = is_q ? n_heads : n_kv
      nel_full = rows * D_HEAD * d_model

      fci = fused_lookup_a(fused_names, fused_a, fused_b, ggA, ggB,
                           fused_name, nel_full)
      if fci < 0
        pi = pi + 1
      else
        fa = fused_a[fci]
        fb = fused_b[fci]
        off = head_h * D_HEAD * d_model
        blended = Mat.new(1, nel)
        j = 0
        while j < nel
          blended.flat[j] = (1.0 - alpha) * fa.flat[off + j] +
                            alpha * fb.flat[off + j]
          j = j + 1
        end
        TinyNN.tnn_upload_from_float_array(fcache.sess, t, blended.flat, nel)
        pi = pi + 1
      end
    else
      # Non-attention tensor: DIRECT GGUF name match (08_lmc unchanged).
      idx_a = TinyNN.tnn_gguf_find_index(ggA, name)
      idx_b = TinyNN.tnn_gguf_find_index(ggB, name)
      if idx_a < 0 || idx_b < 0
        STDERR.puts "toy-eval-lmc: missing " + name +
                    " in A=" + idx_a.to_s + " B=" + idx_b.to_s
        pi = pi + 1
      else
        ta = Mat.new(1, nel); tb = Mat.new(1, nel)
        TinyNN.tnn_gguf_read_f32_to_doubles(ggA, idx_a, ta.flat, nel)
        TinyNN.tnn_gguf_read_f32_to_doubles(ggB, idx_b, tb.flat, nel)
        blended = Mat.new(1, nel)
        j = 0
        while j < nel
          blended.flat[j] = (1.0 - alpha) * ta.flat[j] + alpha * tb.flat[j]
          j = j + 1
        end
        TinyNN.tnn_upload_from_float_array(fcache.sess, t, blended.flat, nel)
        pi = pi + 1
      end
    end
  end

  # Eval: forward + CE. build_training_step with lr=0 → AdamW no-op (pure
  # forward; the Adam moment update mutates m/v but the session is fresh).
  positions = [0]; positions.pop
  p = 0; while p < SEQ_LEN; positions.push(p); p = p + 1; end
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_token_ids, seq_ids)
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_positions, positions)

  # One-hot label rows (shift-by-one targets).
  m_labels = Mat.new(SEQ_LEN, vocab)
  j = 0; while j < SEQ_LEN * vocab; m_labels.flat[j] = 0.0; j = j + 1; end
  k = 0
  while k < SEQ_LEN
    target = (k + 1 < SEQ_LEN) ? seq_ids[k + 1] : seq_ids[k]
    m_labels.flat[k * vocab + target] = 1.0
    k = k + 1
  end
  TinyNN.upload_row_major(fcache.sess, t_labels, m_labels)

  # hp = [lr=0, β1, β2, eps, wd, β1h, β2h]. lr=0 → no weight update.
  m_hp = Mat.new(1, 7)
  m_hp.flat[0] = 0.0
  m_hp.flat[1] = 0.9
  m_hp.flat[2] = 0.95
  m_hp.flat[3] = 1.0e-8
  m_hp.flat[4] = 0.0
  m_hp.flat[5] = 0.9
  m_hp.flat[6] = 0.95
  TinyNN.upload_row_major(fcache.sess, t_hp, m_hp)

  TinyNN.tnn_compute_backward(fcache.sess)
  loss_mat = TinyNN.download_row_major(fcache.sess, t_loss, 1, 1)
  loss = loss_mat.flat[0]

  # The byte-gated line (plain ASCII prefix, NOT 08_lmc's non-ASCII α).
  puts "lmc: " + alpha.to_s + "=" + loss.to_s

  # Emit eval event (FILE only).
  if EVENTS != ""
    t_now = TinyNN.tnn_events_now_seconds
    ev = "{\"kind\":\"eval\",\"phase\":\"eval\""
    ev = ev + ",\"t\":"     + t_now.to_s
    ev = ev + ",\"name\":\"lmc\""
    ev = ev + ",\"loss\":"  + loss.to_s
    ev = ev + ",\"extra\":{\"alpha\":" + alpha.to_s + "}"
    ev = ev + "}"
    TinyNN.tnn_events_emit(ev)
  end

  ai = ai + 1
end

# Run-end (FILE only).
if EVENTS != ""
  t_close = TinyNN.tnn_events_now_seconds
  re = "{\"kind\":\"run_end\",\"phase\":\"eval\""
  re = re + ",\"t\":"        + t_close.to_s
  re = re + ",\"reason\":\"completed\""
  re = re + ",\"n_alphas\":" + alphas_arr.length.to_s
  re = re + "}"
  TinyNN.tnn_events_emit(re)
  TinyNN.tnn_events_close
end

TinyNN.tnn_gguf_free(ggA)
TinyNN.tnn_gguf_free(ggB)
