# demos/train.rb — TinyStories from-scratch training via Toy::Trainer.
#
# Configure → loop → step → log. The Trainer (lib/toy_trainer.rb)
# absorbs the (grads.fill_zero → forward → backward → optimizer.step)
# per-step boilerplate into one verb, so the outer epoch/sequence
# loop stays visible — what you're reading IS the training algorithm.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/train/training"
require_relative "../lib/toy/train/toy_trainer"

# ---- hyperparams ----------------------------------------------------------
D_MODEL  = 32
D_FF     = 64
N_HEADS  = 4
N_LAYERS = 2
CONTEXT  = 64
EPOCHS   = 1     # smoke; bump for real training
BATCH    = 32

# ---- data + model ---------------------------------------------------------
vocab      = read_vocab("data/ts_vocab.txt")
sequences  = read_sequences("data/ts_seqs.txt")
prompt_ids = read_prompt("data/ts_prompt.txt")
puts "vocab=" + vocab.length.to_s + " seqs=" + sequences.length.to_s

model = TransformerLM.new(vocab.length, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
model.vocabulary = vocab

# ---- trainer --------------------------------------------------------------
trainer             = Toy::Trainer.new(model)
trainer.lr_max      = 0.001
trainer.lr_min      = 0.00001
trainer.warmup      = 200
trainer.total_steps = EPOCHS * sequences.length
trainer.schedule    = LRSchedule.new(trainer.warmup, trainer.total_steps,
                                     trainer.lr_max, trainer.lr_min)

# Anchor Spinel's type inference end-to-end on the prompt before
# training, then reset the optimizer so the warm-up step doesn't count.
puts "warm-up to anchor types..."
trainer.step!(prompt_ids)
trainer.reset_optimizer!
trainer.step_idx = 0

# ---- train ---------------------------------------------------------------
# The whole training loop. The Trainer absorbs the four-line per-step
# body — what's left is the *shape* of training: epoch → batch → step.
puts "training " + EPOCHS.to_s + " epochs over " + sequences.length.to_s + " sequences..."
t_start = Time.now
epoch   = 0
while epoch < EPOCHS
  total = 0.0
  n     = 0
  i = 0
  while i < sequences.length
    seq = sequences[i]
    sl  = seq.length
    if sl >= 2 && sl <= CONTEXT
      total += trainer.step!(seq)
      n     += 1
    end
    i += 1
  end
  if n > 0 && epoch % 4 == 0
    puts "epoch " + (epoch + 1).to_s + "/" + EPOCHS.to_s +
         "  loss=" + (total / n).to_s +
         "  lr="   + trainer.lr.to_s
  end
  epoch += 1
end
puts "trained in " + (Time.now - t_start).to_s + "s"

# ---- generate ------------------------------------------------------------
ids = model.generate_from_ids(prompt_ids, 60, 0.7)
out = vocab[ids[0]]
i = 1
while i < ids.length
  out = out + " " + vocab[ids[i]]
  i += 1
end
puts ""
puts out
