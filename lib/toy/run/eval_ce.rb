# lib/toy/run/eval_ce.rb — Spinel-compiled CE-over-pack COMPUTE runner
# (toy#130, the consuming half of the toy#129 item-3 eval seam).
#
# Loads ONE fused-llama checkpoint GGUF (the `toy train franken
# --ckpt-every` artifacts qualify — lens folded, heads fused), streams a
# corpus pack in disjoint CONTEXT-token windows, and reports the mean
# next-token cross-entropy over EVAL_TOKENS tokens. Quality-per-FLOP at
# F9 joins this against run_start.cost and the training wall_us.
#
# MACHINERY (eval_lmc.rb VERBATIM where possible — the gated path):
# random-init session at [CONTEXT] → build_training_step → overwrite
# every param from the GGUF (per-head attention slices out of the fused
# tensors; direct name match elsewhere) → per window: upload ids +
# guarded one-hot labels + lr=0 AdamW hp → tnn_compute_backward →
# download the ggml-internal CE (no Ruby libm — byte-exact, the lmc
# determinism discipline). lr=0 → AdamW is a weight no-op, so windows
# are independent; the mutated Adam moments belong to THIS throwaway
# session (the toy#122 pollution objection was about the TRAINING
# session — offline it is moot).
#
# LABELS AT REAL VOCAB (the toy#130 perf note): the one-hot Mat is
# built ONCE and per window only the previous T scatter positions are
# cleared + T new ones set — no ctx*vocab zero-fill per window. The
# upload is still ctx*vocab floats (the CE node's contract); at
# vocab 50257 / ctx 256 that is ~50 MB per window, dominated by the
# forward itself.
#
# CONTRACT (read from ENV only):
#   GGUF          — checkpoint path (required)
#   PACK          — corpus pack path (required; TOYC v1 or headerless)
#   CONTEXT       — window length (default 32)
#   EVAL_TOKENS   — token budget (default 8192; windows = budget/CONTEXT,
#                   truncated at EOF — NO rotation: eval never repeats)
#   EVAL_OFFSET   — starting TOKEN offset into the pack (default 0; the
#                   held-out discipline — pointing this past the train
#                   slice — is the CALLER's contract)
#   SEED          — session random-init seed (default 0; every param is
#                   overwritten, the seed only names the throwaway init)
#   TAO_RUN_DIR   — when set, emit events.jsonl HERE (FILE only)
#   TOY_RUN_ID    — run_id for events (default "eval-ce")
#
# VOCAB: the model's llama.vocab_size is the authority. A TOYC pack
# whose header disagrees FAILS LOUD (the toy#129 header exists exactly
# so this mismatch cannot be silent); a headerless pack is scanned
# per-window — any token id outside [0, vocab) fails loud (an OOB id
# would read past the embed table).
#
# OUTPUT (byte-exact prefix line the CLI + gate parse):
#   "eval_ce: windows=<n> tokens=<n> ce=<mean>"
# Perplexity is exp(ce) — derived, deliberately NOT emitted (no Math.*
# in the gated line; the lmc hygiene).
#
# Backend: CPU only (the eval/infer precedent). ABSENT from MIRRORABLE.
# Spinel hygiene (landmine #16): no #{}, no Struct.new, popped-empty
# literals, monomorphic drive.

require_relative "../../toy"
require_relative "../io/json_builder"
require_relative "../io/toy_corpus_loader"
require_relative "../models/toy_smollm2"
require_relative "../llm/engine/llama_seq_engine"
require_relative "../llm/adamw"
require_relative "../train/toy_drift_grad"

GGUF_P   = ENV["GGUF"] || ""
PACK     = ENV["PACK"] || ""
SEQ_LEN  = (ENV["CONTEXT"] || "32").to_i
EVAL_TOKENS = (ENV["EVAL_TOKENS"] || "8192").to_i
EVAL_OFFSET = (ENV["EVAL_OFFSET"] || "0").to_i
SEED     = (ENV["SEED"] || "0").to_i
RUN_DIR  = ENV["TAO_RUN_DIR"] || ""
RUN_ID   = ENV["TOY_RUN_ID"]  || "eval-ce"

if GGUF_P == "" || PACK == ""
  puts "toy-eval-ce: GGUF and PACK are required"
  exit 1
end
if !File.exist?(GGUF_P)
  puts "toy-eval-ce: no such file: GGUF=" + GGUF_P +
       " (pass a fused-llama checkpoint, e.g. runs/<id>/weights/latest)"
  exit 1
end
if !File.exist?(PACK)
  puts "toy-eval-ce: no such file: PACK=" + PACK +
       " (packed-i32 tokens; prep/pretokenize_pack.py writes TOYC packs)"
  exit 1
end
if SEQ_LEN < 2
  puts "toy-eval-ce: CONTEXT must be >= 2, got " + SEQ_LEN.to_s
  exit 1
end

gg = TinyNN.tnn_gguf_load(GGUF_P)
if gg == nil || gg == TinyNN.tnn_null_ptr
  puts "toy-eval-ce: cannot open GGUF=" + GGUF_P
  exit 1
end

vocab     = TinyNN.tnn_gguf_get_u32(gg, "llama.vocab_size")
d_model   = TinyNN.tnn_gguf_get_u32(gg, "llama.embedding_length")
d_ff      = TinyNN.tnn_gguf_get_u32(gg, "llama.feed_forward_length")
n_heads   = TinyNN.tnn_gguf_get_u32(gg, "llama.attention.head_count")
n_kv      = TinyNN.tnn_gguf_get_u32(gg, "llama.attention.head_count_kv")
n_layers  = TinyNN.tnn_gguf_get_u32(gg, "llama.block_count")
ctx_len   = TinyNN.tnn_gguf_get_u32(gg, "llama.context_length")
rope_base = TinyNN.tnn_gguf_get_f32(gg, "llama.rope.freq_base")
rms_eps   = TinyNN.tnn_gguf_get_f32(gg, "llama.attention.layer_norm_rms_epsilon")

cfg = Toy::SmolLM2Config.new(vocab, d_model, n_heads, n_kv, d_ff,
                             n_layers, ctx_len, rope_base, rms_eps)
D_HEAD = d_model / n_heads

# Pack vocab discipline (toy#129 header): TOYC disagreement fails loud.
pack_vocab = ToyCorpusLoader.probe_vocab(PACK)
if pack_vocab > 0 && pack_vocab != vocab
  puts "toy-eval-ce: pack vocab " + pack_vocab.to_s +
       " (TOYC header) != model vocab " + vocab.to_s + " — wrong pack or wrong model"
  exit 1
end
pack_base  = ToyCorpusLoader.data_offset(PACK)
pack_bytes = File.size(PACK)

EVENTS = RUN_DIR.length > 0 ? (RUN_DIR + "/events.jsonl") : ""
if EVENTS != ""
  TinyNN.tnn_events_open(EVENTS)
end
if EVENTS != ""
  rs = Toy::Json::Builder.new
  rs.add_str("kind",  "run_start")
  rs.add_str("phase", "eval")
  rs.add_num("t",          TinyNN.tnn_events_now_seconds)
  rs.add_str("started_at", TinyNN.tnn_events_iso8601_now)
  rs.add_str("run_id",     RUN_ID)
  rs.add_str("name",       "eval-ce")
  rs.add_str("gguf",       GGUF_P)
  rs.add_str("pack",       PACK)
  cfgo = Toy::Json::Builder.new
  cfgo.add_num("context",     SEQ_LEN)
  cfgo.add_num("vocab",       vocab)
  cfgo.add_num("eval_tokens", EVAL_TOKENS)
  cfgo.add_num("eval_offset", EVAL_OFFSET)
  rs.add_obj("config", cfgo)
  host = Toy::Json::Builder.new
  host.add_str("name", TinyNN.tnn_provenance_host_name)
  host.add_str("os",   TinyNN.tnn_provenance_host_os)
  host.add_str("arch", TinyNN.tnn_provenance_host_arch)
  rs.add_obj("host", host)
  backend = Toy::Json::Builder.new
  backend.add_str("kind", "cpu")
  rs.add_obj("backend", backend)
  TinyNN.tnn_events_emit(rs.dump)
end

# Session + training graph (the lmc path); every param overwritten below.
fcache = Toy::LLM::Engine::LlamaSeqEngine.new
fcache.realize_for_random_init(cfg, SEQ_LEN, 1, 0, true, false, SEED, 1.0)
result   = fcache.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

# Overwrite every param from the ONE checkpoint (eval_lmc's loader,
# single-gguf form): per-head attention slices from the fused tensor
# (read each fused tensor once, cached), direct name match elsewhere.
fused_names = [""]; fused_names.pop
fused_bufs  = [Mat.new(1, 1)]; fused_bufs.pop

def fused_lookup(fused_names, fused_bufs, gg, fused_name, nel_full)
  fi = 0
  while fi < fused_names.length
    if fused_names[fi] == fused_name
      return fi
    end
    fi = fi + 1
  end
  idx = TinyNN.tnn_gguf_find_index(gg, fused_name)
  if idx < 0
    puts "toy-eval-ce: missing fused " + fused_name + " in checkpoint"
    return -1
  end
  mf = Mat.new(1, nel_full)
  TinyNN.tnn_gguf_read_f32_to_doubles(gg, idx, mf.flat, nel_full)
  fused_names.push(fused_name)
  fused_bufs.push(mf)
  return fused_names.length - 1
end

missing = 0
plist = ToyDriftGrad.params(fcache.sess)
pi = 0
while pi < plist.length
  t    = plist[pi]
  name = TinyNN.tnn_tensor_name(t)
  nel  = TinyNN.tnn_tensor_nelements(t)
  if name.include?(".head_")
    pre_parts  = name.split(".head_")
    fused_name = pre_parts[0] + ".weight"
    tail_parts = pre_parts[1].split(".")
    head_h = tail_parts[0].to_i
    is_q = fused_name.include?(".attn_q.")
    rows = is_q ? n_heads : n_kv
    nel_full = rows * D_HEAD * d_model
    fci = fused_lookup(fused_names, fused_bufs, gg, fused_name, nel_full)
    if fci < 0
      missing = missing + 1
    else
      mf = fused_bufs[fci]
      off = head_h * D_HEAD * d_model
      sl = Mat.new(1, nel)
      j = 0
      while j < nel
        sl.flat[j] = mf.flat[off + j]
        j = j + 1
      end
      TinyNN.tnn_upload_from_float_array(fcache.sess, t, sl.flat, nel)
    end
  else
    idx = TinyNN.tnn_gguf_find_index(gg, name)
    if idx < 0
      puts "toy-eval-ce: missing " + name + " in checkpoint"
      missing = missing + 1
    else
      mt = Mat.new(1, nel)
      TinyNN.tnn_gguf_read_f32_to_doubles(gg, idx, mt.flat, nel)
      TinyNN.tnn_upload_from_float_array(fcache.sess, t, mt.flat, nel)
    end
  end
  pi = pi + 1
end
if missing > 0
  puts "toy-eval-ce: " + missing.to_s + " params missing from the checkpoint — refusing a partial eval"
  exit 1
end

positions = [0]; positions.pop
p = 0; while p < SEQ_LEN; positions.push(p); p = p + 1; end

# lr=0 named AdamW: weights never move; windows are independent.
adamw_eval = Toy::AdamW.for_from_scratch
adamw_eval.lr = 0.0
m_hp = adamw_eval.hp(0)

# One-hot labels: allocate once; per window clear ONLY the previous
# scatter positions (T entries), never a ctx*vocab zero-fill.
m_labels = Mat.new(SEQ_LEN, vocab)
j = 0; while j < SEQ_LEN * vocab; m_labels.flat[j] = 0.0; j = j + 1; end
prev_targets = [0]; prev_targets.pop
k = 0; while k < SEQ_LEN; prev_targets.push(-1); k = k + 1; end

n_windows_want = EVAL_TOKENS / SEQ_LEN
if n_windows_want < 1
  puts "toy-eval-ce: EVAL_TOKENS " + EVAL_TOKENS.to_s + " < CONTEXT " + SEQ_LEN.to_s
  exit 1
end

sum_ce = 0.0
n_done = 0
off = pack_base + EVAL_OFFSET * 4
wi = 0
while wi < n_windows_want
  if off + SEQ_LEN * 4 > pack_bytes
    # NO rotation: eval never repeats data. Stop at EOF.
    break
  end
  seq_ids = ToyCorpusLoader.read_seq(PACK, off, SEQ_LEN)
  off = off + SEQ_LEN * 4

  # In-vocab guard on the INPUT ids (an OOB id reads past the embed).
  k = 0
  while k < SEQ_LEN
    if seq_ids[k] < 0 || seq_ids[k] >= vocab
      puts "toy-eval-ce: token id " + seq_ids[k].to_s + " outside [0, " +
           vocab.to_s + ") at window " + wi.to_s + " — pack/model mismatch"
      exit 1
    end
    k = k + 1
  end

  # Guarded shift-by-one one-hot, incremental form.
  k = 0
  while k < SEQ_LEN
    if prev_targets[k] >= 0
      m_labels.flat[k * vocab + prev_targets[k]] = 0.0
    end
    k = k + 1
  end
  k = 0
  while k < SEQ_LEN
    target = (k + 1 < SEQ_LEN) ? seq_ids[k + 1] : seq_ids[k]
    m_labels.flat[k * vocab + target] = 1.0
    prev_targets[k] = target
    k = k + 1
  end

  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_token_ids, seq_ids)
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_positions, positions)
  TinyNN.upload_row_major(fcache.sess, t_labels, m_labels)
  TinyNN.upload_row_major(fcache.sess, t_hp, m_hp)
  TinyNN.tnn_compute_backward(fcache.sess)
  loss_mat = TinyNN.download_row_major(fcache.sess, t_loss, 1, 1)
  sum_ce = sum_ce + loss_mat.flat[0]
  n_done = n_done + 1
  wi = wi + 1
end

if n_done == 0
  puts "toy-eval-ce: zero windows evaluated (EVAL_OFFSET " + EVAL_OFFSET.to_s +
       " + CONTEXT " + SEQ_LEN.to_s + " runs past the pack end)"
  exit 1
end
if n_done < n_windows_want
  puts "toy-eval-ce: EOF after " + n_done.to_s + " of " + n_windows_want.to_s +
       " windows (no rotation — eval never repeats data)"
end

mean_ce = sum_ce / n_done.to_f

# The byte-gated line.
puts "eval_ce: windows=" + n_done.to_s +
     " tokens=" + (n_done * SEQ_LEN).to_s +
     " ce=" + mean_ce.to_s

if EVENTS != ""
  ev = Toy::Json::Builder.new
  ev.add_str("kind",  "eval")
  ev.add_str("phase", "eval")
  ev.add_num("t",       TinyNN.tnn_events_now_seconds)
  ev.add_str("name",    "eval-ce")
  ev.add_num("loss",    mean_ce)
  ev.add_num("windows", n_done)
  ev.add_num("tokens",  n_done * SEQ_LEN)
  TinyNN.tnn_events_emit(ev.dump)
  re = Toy::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",        TinyNN.tnn_events_now_seconds)
  re.add_str("ended_at", TinyNN.tnn_events_iso8601_now)
  re.add_str("reason",   "completed")
  re.add_raw("exit_code", "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
