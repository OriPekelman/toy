# E2.3 (towards GH#14) — projection-lens smoke. Builds a target model
# with a "wide" donor embed (donor_d_in > d_model) and a trainable
# Linear(donor_d_in, d_model) between get_rows and the first block.
# Verifies the forward graph compiles, the backward propagates
# through W_proj only (token_embd stays frozen), and loss decreases.
#
#   make examples/smoke_projection_lens
#   STEPS=5 ./examples/smoke_projection_lens
#
# Tied output is incompatible with donor_d_in > 0 because LM-head
# matmul(token_embd, x_final) would need ne[0] = d_model on
# token_embd, but token_embd has ne[0] = donor_d_in. So we always
# realize_for_random_init with untied=true.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi"
require_relative "../lib/toy_drift_grad"

STEPS    = (ENV["STEPS"]    || "5").to_i
D_MODEL  = (ENV["D_MODEL"]  || "64").to_i
DONOR_D  = (ENV["DONOR_D"]  || "128").to_i
SEED     = (ENV["SEED"]     || "0").to_i
VOCAB    = 627
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

cfg = Toy::SmolLM2Config.new(VOCAB, D_MODEL, N_HEADS, N_HEADS,
                              D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = DONOR_D
puts "config: vocab=" + cfg.vocab.to_s +
     " donor_d_in=" + cfg.donor_d_in.to_s +
     " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s

fcache = LlamaSeqForwardFFICache.new
fcache.realize_for_random_init(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)
puts "realize OK"

# Walk PARAM tensors, log names — confirm token_embd is NOT a PARAM
# and lens.proj.weight IS a PARAM.
plist = ToyDriftGrad.params(fcache.sess)
puts "params (" + plist.length.to_s + "):"
i = 0
seen_proj = false; seen_embed_param = false
while i < plist.length
  name = TinyNN.tnn_tensor_name(plist[i])
  if name == "lens.proj.weight"; seen_proj = true; end
  if name == "token_embd.weight"; seen_embed_param = true; end
  if i < 6 || name == "lens.proj.weight" || name == "token_embd.weight"
    puts "  " + name
  end
  i = i + 1
end
puts "lens.proj.weight is a PARAM: " + (seen_proj    ? "YES" : "NO")
puts "token_embd.weight is a PARAM: " + (seen_embed_param ? "YES (BUG)" : "NO (correct — frozen)")

# Train a few steps on a fixed sequence.
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

positions = [0]; positions.pop
p = 0; while p < CONTEXT; positions.push(p); p = p + 1; end

# Build labels: shift-by-one one-hot.
m_labels = Mat.new(CONTEXT, VOCAB)
j = 0; while j < CONTEXT * VOCAB; m_labels.flat[j] = 0.0; j = j + 1; end
k = 0
while k < CONTEXT
  target = (k + 1 < CONTEXT) ? seq_ids[k + 1] : seq_ids[k]
  m_labels.flat[k * VOCAB + target] = 1.0
  k = k + 1
end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = 0.001; m_hp.flat[1] = 0.9; m_hp.flat[2] = 0.95
m_hp.flat[3] = 1.0e-8; m_hp.flat[4] = 0.0
m_hp.flat[5] = 0.9; m_hp.flat[6] = 0.95

losses = [0.0]; losses.pop
step = 0
while step < STEPS
  if step == 0
    TinyNN.tnn_graph_reset(fcache.sess)
  else
    TinyNN.tnn_graph_reset_grads_only(fcache.sess)
  end
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_token_ids, seq_ids)
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_positions, positions)
  TinyNN.upload_row_major(fcache.sess, t_labels, m_labels)
  TinyNN.upload_row_major(fcache.sess, t_hp,     m_hp)
  TinyNN.tnn_compute_backward(fcache.sess)
  loss_mat = TinyNN.download_row_major(fcache.sess, t_loss, 1, 1)
  losses.push(loss_mat.flat[0])
  puts "step " + (step + 1).to_s + ": loss=" + loss_mat.flat[0].to_s
  step = step + 1
end

initial = losses[0]
final   = losses[losses.length - 1]
ratio   = final / initial
puts "initial=" + initial.to_s + " final=" + final.to_s + " ratio=" + ratio.to_s
if ratio < 0.95
  puts "VERDICT: projection-lens training is learning"
else
  puts "VERDICT: training NOT learning (final/initial = " + ratio.to_s + ")"
end
