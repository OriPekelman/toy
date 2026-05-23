# Train a tiny Llama-shape model from scratch on TinyStories via the
# FFI training graph (forward + backward + AdamW in one ggml call).
# Llama-arch (RMSNorm + GQA + RoPE + SwiGLU); the GPT-2-style
# example_train (lib/transformer.rb) stays as the pure-Ruby teaching
# path.
#
#   make example_train_from_scratch
#   ./examples/example_train_from_scratch
#   D_MODEL=128 N_LAYERS=4 STEPS=200 ./examples/example_train_from_scratch
#
# Implementation notes:
#   - Reads sequences via File.read + split. The File.open-with-block
#     form interacts badly with FFI session-init under Spinel d59926a
#     on gx10 (segfaults during realize_for_random_init). File.read
#     sidesteps it.
#   - VOCAB_SIZE is hardcoded at 627 (TinyStories). Reading
#     data/ts_vocab.txt would pull Array<String> into main scope and
#     poison Spinel's polymorphic Array<T> dispatch table inside
#     Toy_Embedding#lookup / RMSNorm#forward, breaking C codegen.
#     Acceptance: loss decreases monotonically; we don't decode
#     generated tokens.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi"

VOCAB_SIZE = 627
D_MODEL    = (ENV["D_MODEL"]  || "64").to_i
D_FF       = (ENV["D_FF"]     || "128").to_i
N_HEADS    = (ENV["N_HEADS"]  || "4").to_i
N_LAYERS   = (ENV["N_LAYERS"] || "2").to_i
CONTEXT    = (ENV["CONTEXT"]  || "32").to_i
STEPS      = (ENV["STEPS"]    || "60").to_i
LR         = (ENV["LR"]       || "0.001").to_f
SEED       = (ENV["SEED"]     || "42").to_i

cfg = Toy::SmolLM2Config.new(VOCAB_SIZE, D_MODEL, N_HEADS, N_HEADS,
                              D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " heads=" + cfg.n_heads.to_s +
     " d_ff=" + cfg.d_ff.to_s +
     " L=" + cfg.n_layers.to_s +
     " head_dim=" + cfg.head_dim.to_s

t_realize = Time.now
fcache = LlamaSeqForwardFFICache.new
fcache.realize_for_random_init(cfg, CONTEXT, false, false, SEED, 1.0)
puts "realize_for_random_init: " + ((Time.now - t_realize).to_f * 1000.0).to_s + " ms"

# Read first TinyStories sequence (Array<Integer>).
raw        = File.read("data/ts_seqs.txt")
first_line = raw.split("\n")[0]
parts      = first_line.split(" ")
seq_ids    = [0]; seq_ids.pop
k = 0
while k < parts.length && k < CONTEXT
  seq_ids.push(parts[k].to_i)
  k = k + 1
end
while seq_ids.length < CONTEXT; seq_ids.push(0); end

result   = fcache.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

# Shift-by-one next-token targets, one-hot rows.
m_labels = Mat.new(CONTEXT, VOCAB_SIZE)
li = 0
while li < CONTEXT * VOCAB_SIZE; m_labels.flat[li] = 0.0; li = li + 1; end
ti = 0
while ti < CONTEXT
  tgt = ti + 1 < CONTEXT ? seq_ids[ti + 1] : seq_ids[ti]
  m_labels.flat[ti * VOCAB_SIZE + tgt] = 1.0
  ti = ti + 1
end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0

positions = [0]; positions.pop
pi = 0
while pi < CONTEXT; positions.push(pi); pi = pi + 1; end

losses = [0.0]; losses.pop

t_start = Time.now
step = 1
while step <= STEPS
  m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  if step == 1
    TinyNN.tnn_graph_reset(fcache.sess)
  else
    TinyNN.tnn_graph_reset_grads_only(fcache.sess)
  end
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_token_ids, seq_ids)
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_positions, positions)
  TinyNN.upload_row_major(fcache.sess, t_labels, m_labels)
  TinyNN.upload_row_major(fcache.sess, t_hp,     m_hp)
  TinyNN.tnn_compute_backward(fcache.sess)
  TinyNN.tnn_download(fcache.sess, t_loss)
  loss = TinyNN.tnn_scratch_get(fcache.sess, 0)
  losses.push(loss)
  if step <= 5 || step % 10 == 0 || step == STEPS
    puts "step " + step.to_s.rjust(4) + ": CE=" + loss.to_s
  end
  step = step + 1
end
total = (Time.now - t_start).to_f
puts "trained " + STEPS.to_s + " steps in " + total.to_s + "s " +
     "(" + (total / STEPS.to_f * 1000.0).to_s + " ms/step)"
puts "initial loss = " + losses[0].to_s
puts "final   loss = " + losses[losses.length - 1].to_s
ratio = losses[losses.length - 1] / losses[0]
if ratio < 0.9
  puts "VERDICT: training is learning (final/initial = " + ratio.to_s + ")"
else
  puts "VERDICT: training NOT learning (final/initial = " + ratio.to_s + ")"
  exit 1
end
