# Train a tiny GPT from scratch on TinyStories. The training loop
# IS what you read — Toy::Trainer wraps the per-step plumbing
# (zero-grads, forward, backward, optimizer.step) into one verb.
#
#   ./prep/prep_tinystories.rb        # one-time: build vocab/sequences
#   make example_train
#   ./examples/example_train
#
# Bumping EPOCHS produces a model that writes English-shaped stories.
# Defaults stay small so the example finishes in seconds.

require_relative "../../lib/toy/models/transformer"
require_relative "../../lib/toy/train/training"
require_relative "../../lib/toy/train/toy_trainer"

D_MODEL  = 32
D_FF     = 64
N_HEADS  = 4
N_LAYERS = 2
CONTEXT  = 64
EPOCHS   = (ENV["EPOCHS"] || "1").to_i
LR       = 0.001
TRACE     = ENV["TRACE"] || ""
TRACE_OPS = ENV["TRACE_OPS"] || ""

vocab      = read_vocab("data/ts_vocab.txt")
sequences  = read_sequences("data/ts_seqs.txt")
prompt_ids = read_prompt("data/ts_prompt.txt")
puts "vocab=" + vocab.length.to_s + " sequences=" + sequences.length.to_s

model = TransformerLM.new(vocab.length, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
model.vocabulary = vocab

trainer             = Toy::Trainer.new(model)
trainer.lr_max      = LR
trainer.lr_min      = LR / 100.0
trainer.warmup      = 200
trainer.total_steps = EPOCHS * sequences.length
trainer.schedule    = LRSchedule.new(trainer.warmup, trainer.total_steps,
                                      trainer.lr_max, trainer.lr_min)

# Warm-up step so Spinel sees every type before timed training.
trainer.step!(prompt_ids); trainer.reset_optimizer!; trainer.step_idx = 0

if TRACE.length > 0
  rc = TinyNN.tnn_trace_open(TRACE)
  if rc == 0
    puts "tracing to " + TRACE
    if TRACE_OPS == "1"
      TinyNN.tnn_trace_set_op_capture(1)
      puts "per-op trace enabled (TRACE_OPS=1)"
    end
  else
    puts "trace_open rc=" + rc.to_s
  end
end

t0 = Time.now
e = 0
while e < EPOCHS
  total = 0.0; n = 0; i = 0
  while i < sequences.length
    sl = sequences[i].length
    if sl >= 2 && sl <= CONTEXT
      _t = TinyNN.tnn_trace_begin("train_step")
      total += trainer.step!(sequences[i])
      TinyNN.tnn_trace_end("train_step", _t)
      n += 1
    end
    i += 1
  end
  puts "epoch " + (e + 1).to_s + ": loss=" + (total / n.to_f).to_s
  e += 1
end
puts "trained in " + (Time.now - t0).to_s + "s"

if TRACE.length > 0
  TinyNN.tnn_trace_close
  puts "trace closed: " + TRACE
end

ids = model.generate_from_ids(prompt_ids, 60, 0.7)
print "generated: "
i = 0
while i < ids.length
  print vocab[ids[i]] + " "
  i += 1
end
puts ""
