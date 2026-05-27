# GH#18 — Linear Mode Connectivity (LMC) runner. Given two toy
# checkpoints A + B, evaluates loss along θ_α = (1-α)·θ_A + α·θ_B on
# a fixed sequence. Emits one `eval` event per α (kind="eval",
# name="lmc", extra.alpha=α). Tao's Analyze.lmc reads these to plot
# the α→loss curve and decide same-basin / disconnected.
#
#   LMC_A=runA/weights/latest LMC_B=runB/weights/latest \
#     LMC_ALPHAS=0,0.25,0.5,0.75,1.0 TAO_RUN_DIR=/tmp/lmc \
#     ./examples/08_lmc
#
# Two endpoints (α=0 and α=1) match their training final loss when
# the eval sequence is the same. Midpoint bump tells the basin story.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi"
require_relative "../lib/toy_drift_grad"

LMC_A    = ENV["LMC_A"]    || ""
LMC_B    = ENV["LMC_B"]    || ""
ALPHAS_S = ENV["LMC_ALPHAS"] || "0,0.25,0.5,0.75,1.0"
SEQ_LEN  = (ENV["CONTEXT"] || "32").to_i
SEED     = (ENV["SEED"]    || "0").to_i
RUN_DIR  = ENV["TAO_RUN_DIR"] || ""
RUN_ID   = ENV["TOY_RUN_ID"]  || "lmc"

if LMC_A == "" || LMC_B == ""
  STDERR.puts "08_lmc: LMC_A and LMC_B are required (paths to two toy checkpoints)"
  exit 1
end
EVENTS = RUN_DIR.length > 0 ? (RUN_DIR + "/events.jsonl") : ""
if EVENTS != ""
  TinyNN.tnn_events_open(EVENTS)
end

# Parse α-grid.
alphas_arr = [0.0]; alphas_arr.pop
ALPHAS_S.split(",").each do |a|
  alphas_arr.push(a.to_f)
end
puts "LMC: " + alphas_arr.length.to_s + " alphas: " + ALPHAS_S

# Read config from ckpt A. The two ckpts must share architecture.
ggA = TinyNN.tnn_gguf_load(LMC_A)
ggB = TinyNN.tnn_gguf_load(LMC_B)
if ggA == nil || ggA == TinyNN.tnn_null_ptr
  STDERR.puts "08_lmc: cannot open LMC_A=" + LMC_A; exit 1
end
if ggB == nil || ggB == TinyNN.tnn_null_ptr
  STDERR.puts "08_lmc: cannot open LMC_B=" + LMC_B; exit 1
end

vocab    = TinyNN.tnn_gguf_get_u32(ggA, "llama.vocab_size")
d_model  = TinyNN.tnn_gguf_get_u32(ggA, "llama.embedding_length")
d_ff     = TinyNN.tnn_gguf_get_u32(ggA, "llama.feed_forward_length")
n_heads  = TinyNN.tnn_gguf_get_u32(ggA, "llama.attention.head_count")
n_kv     = TinyNN.tnn_gguf_get_u32(ggA, "llama.attention.head_count_kv")
n_layers = TinyNN.tnn_gguf_get_u32(ggA, "llama.block_count")
ctx_len  = TinyNN.tnn_gguf_get_u32(ggA, "llama.context_length")
rope_base= TinyNN.tnn_gguf_get_f32(ggA, "llama.rope.freq_base")
rms_eps  = TinyNN.tnn_gguf_get_f32(ggA, "llama.attention.layer_norm_rms_epsilon")

cfg = Toy::SmolLM2Config.new(vocab, d_model, n_heads, n_kv, d_ff,
                              n_layers, ctx_len, rope_base, rms_eps)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " heads=" + cfg.n_heads.to_s +
     " d_ff=" + cfg.d_ff.to_s +
     " L=" + cfg.n_layers.to_s

# Fixed eval sequence — first TinyStories line, truncated/padded to SEQ_LEN.
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

# Emit run_start.
if EVENTS != ""
  t_open = TinyNN.tnn_events_now_seconds
  rs  = "{\"kind\":\"run_start\",\"phase\":\"eval\""
  rs  = rs + ",\"t\":"          + t_open.to_s
  rs  = rs + ",\"started_at\":\"" + TinyNN.tnn_events_iso8601_now + "\""
  rs  = rs + ",\"run_id\":\""   + RUN_ID + "\""
  rs  = rs + ",\"name\":\"lmc\""
  rs  = rs + ",\"host\":{\"name\":\""   + TinyNN.tnn_provenance_host_name + "\""
  rs  = rs + ",\"os\":\""               + TinyNN.tnn_provenance_host_os   + "\""
  rs  = rs + ",\"arch\":\""             + TinyNN.tnn_provenance_host_arch + "\"}"
  rs  = rs + ",\"backend\":{\"kind\":\"cpu\"}"
  rs  = rs + "}"
  TinyNN.tnn_events_emit(rs)
end

# For each α, build a fresh session, blend weights, run forward+loss.
ai = 0
while ai < alphas_arr.length
  alpha = alphas_arr[ai]
  puts "α=" + alpha.to_s + " realizing …"

  fcache = LlamaSeqForwardFFICache.new
  fcache.realize_for_random_init(cfg, SEQ_LEN, false, false, SEED, 1.0)

  result   = fcache.build_training_step
  t_loss   = result[0]
  t_labels = result[1]
  t_hp     = result[2]

  # Blend every PARAM tensor: θ = (1-α)·θ_A + α·θ_B. Look up by name
  # in each GGUF (toy from-scratch ckpts carry semantic names per #11).
  plist = ToyDriftGrad.params(fcache.sess)
  pi = 0
  while pi < plist.length
    t      = plist[pi]
    name   = TinyNN.tnn_tensor_name(t)
    nel    = TinyNN.tnn_tensor_nelements(t)
    idx_a  = TinyNN.tnn_gguf_find_index(ggA, name)
    idx_b  = TinyNN.tnn_gguf_find_index(ggB, name)
    if idx_a < 0 || idx_b < 0
      STDERR.puts "08_lmc: missing " + name + " in A=" + idx_a.to_s + " B=" + idx_b.to_s
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

  # Eval: forward + CE. We run build_training_step with lr=0 in hp so
  # AdamW is a no-op (θ - 0·m/(…) = θ). The Adam moment update still
  # mutates m/v but the session is fresh so we don't care.
  positions = [0]; positions.pop
  p = 0; while p < SEQ_LEN; positions.push(p); p = p + 1; end
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_token_ids, seq_ids)
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_positions, positions)

  # Build one-hot label rows (shift-by-one targets).
  m_labels = Mat.new(SEQ_LEN, cfg.vocab)
  j = 0; while j < SEQ_LEN * cfg.vocab; m_labels.flat[j] = 0.0; j = j + 1; end
  k = 0
  while k < SEQ_LEN
    target = (k + 1 < SEQ_LEN) ? seq_ids[k + 1] : seq_ids[k]
    m_labels.flat[k * cfg.vocab + target] = 1.0
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
  puts "α=" + alpha.to_s + "  loss=" + loss.to_s

  # Emit eval event.
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

# Run-end.
if EVENTS != ""
  t_close = TinyNN.tnn_events_now_seconds
  re = "{\"kind\":\"run_end\",\"phase\":\"eval\""
  re = re + ",\"t\":"      + t_close.to_s
  re = re + ",\"reason\":\"completed\""
  re = re + ",\"n_alphas\":" + alphas_arr.length.to_s
  re = re + "}"
  TinyNN.tnn_events_emit(re)
  TinyNN.tnn_events_close
end

TinyNN.tnn_gguf_free(ggA)
TinyNN.tnn_gguf_free(ggB)
puts "08_lmc done"
