# Loader smoke. Constructs a distilgpt2-shaped GPT2LM, loads weights
# from data/distilgpt2-f32.gguf, prints a few sentinel values from
# each weight type. Used to confirm name mapping + per-head split are
# correct before doing the full forward.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/models/gpt2"
require_relative "../lib/toy/io/gguf_load"

# distilgpt2 hyperparams. Hardcoded — eventually read from GGUF metadata.
VOCAB    = 50257
D_MODEL  = 768
D_FF     = 3072
N_HEADS  = 12
N_LAYERS = 6
CONTEXT  = 1024

puts "constructing GPT2LM..."
model = GPT2LM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
puts "  built (vocab=" + VOCAB.to_s + " d=" + D_MODEL.to_s +
     " heads=" + N_HEADS.to_s + " layers=" + N_LAYERS.to_s + ")"

ok = GGUFLoad.load_gpt2(model, "data/distilgpt2-f32.gguf")
if !ok
  puts "load failed"
else
  puts "loaded."

  # Sentinel checks — these values come from prep/convert_distilgpt2_to_gguf.py
  # and gguf_inspect output. If the loader scrambles anything they'll mismatch.
  puts ""
  puts "token_embed[0, 0..3]: " +
       model.token_embed.flat[0].to_s + ", " +
       model.token_embed.flat[1].to_s + ", " +
       model.token_embed.flat[2].to_s + ", " +
       model.token_embed.flat[3].to_s
  puts "pos_embed[0, 0..3]: " +
       model.pos_embed.flat[0].to_s + ", " +
       model.pos_embed.flat[1].to_s + ", " +
       model.pos_embed.flat[2].to_s + ", " +
       model.pos_embed.flat[3].to_s

  blk0 = model.gpt2_blocks[0]
  puts ""
  puts "blk0.ln1_gamma[0..3]: " +
       blk0.ln1_gamma[0].to_s + ", " +
       blk0.ln1_gamma[1].to_s + ", " +
       blk0.ln1_gamma[2].to_s + ", " +
       blk0.ln1_gamma[3].to_s
  puts "blk0.ln1_beta[0..3]: " +
       blk0.ln1_beta[0].to_s + ", " +
       blk0.ln1_beta[1].to_s + ", " +
       blk0.ln1_beta[2].to_s + ", " +
       blk0.ln1_beta[3].to_s

  # First row of head-0 Q weight = first row of concatenated attn_q.weight,
  # columns [0..d_head). So blk0.w_q[0].flat[0..3] == attn_q.weight[0, 0..3].
  puts ""
  puts "blk0.w_q[0][0, 0..3]: " +
       blk0.w_q[0].flat[0].to_s + ", " +
       blk0.w_q[0].flat[1].to_s + ", " +
       blk0.w_q[0].flat[2].to_s + ", " +
       blk0.w_q[0].flat[3].to_s
  # Head 1's first row, first column = concatenated row 0, col d_head.
  puts "blk0.w_q[1][0, 0]: " + blk0.w_q[1].flat[0].to_s +
       "   (should differ from w_q[0][0,0])"

  puts "blk0.b_q[0][0..3]: " +
       blk0.b_q[0][0].to_s + ", " +
       blk0.b_q[0][1].to_s + ", " +
       blk0.b_q[0][2].to_s + ", " +
       blk0.b_q[0][3].to_s

  puts ""
  puts "blk0.w_ff1[0, 0..3]: " +
       blk0.w_ff1.flat[0].to_s + ", " +
       blk0.w_ff1.flat[1].to_s + ", " +
       blk0.w_ff1.flat[2].to_s + ", " +
       blk0.w_ff1.flat[3].to_s
  puts "blk0.b_ff1[0..3]: " +
       blk0.b_ff1[0].to_s + ", " +
       blk0.b_ff1[1].to_s + ", " +
       blk0.b_ff1[2].to_s + ", " +
       blk0.b_ff1[3].to_s

  puts ""
  puts "ln_f_gamma[0..3]: " +
       model.ln_f_gamma[0].to_s + ", " +
       model.ln_f_gamma[1].to_s + ", " +
       model.ln_f_gamma[2].to_s + ", " +
       model.ln_f_gamma[3].to_s
end
