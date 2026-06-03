# lib/toy_trainer.rb — Toy::Trainer: training-loop wrapper.
#
# Wraps the existing Gradients + Adam + LRSchedule machinery so a training
# script can read like a Sinatra app: configure a handful of knobs, then
# run a short loop that calls `trainer.step!(seq)` and inspects the loss.
#
# The full epoch/batch/seq pseudocode stays visible in the demo — no
# magic hidden loop. The Trainer's job is to absorb the
# (grads.fill_zero → forward → backward → optimizer.step) repetition
# into one verb.
#
# Spinel notes:
#   • No stored Proc / block hooks (Spinel can't infer through them).
#     "Every N steps do X" becomes an inline `if step % N == 0` in the
#     demo's own loop, which is fine — the loop body is what the reader
#     wants to see anyway.
#   • Field names chosen to avoid collisions with TransformerLM ivars
#     (e.g. `lr_max`, not `max_lr`).
#
# Used like:
#
#   trainer = Toy::Trainer.new(model)
#   trainer.lr_max = 1e-3
#   trainer.warmup = 200
#   trainer.total_steps = 60 * data.length
#
#   step = 0
#   while step < total_steps
#     loss = trainer.step!(data[step % data.length])
#     puts "#{step}: #{loss}" if step % 100 == 0
#     step += 1
#   end

require_relative "../models/transformer"
require_relative "training"

module Toy
  class Trainer
    attr_accessor :model, :grads, :optimizer, :schedule,
                  :lr_max, :lr_min, :warmup, :total_steps,
                  :beta1, :beta2, :eps, :step_idx

    # Defaults match the train_tinystories.rb constants — sensible
    # starting points for a small transformer LM.
    def initialize(model)
      @model       = model
      @beta1       = 0.9
      @beta2       = 0.999
      @eps         = 0.00000001
      @lr_max      = 0.001
      @lr_min      = 0.00001
      @warmup      = 200
      @total_steps = 1000
      @step_idx    = 0

      @grads     = Gradients.new(model.vocab_size, model.d_model, model.d_ff,
                                  model.n_heads, model.d_head, model.n_layers,
                                  model.context_length)
      @optimizer = Adam.new(model, @beta1, @beta2, @eps)
      @schedule  = LRSchedule.new(@warmup, @total_steps, @lr_max, @lr_min)
    end

    # One optimizer step on a single sequence. Returns the loss.
    # The four-line body is the whole point: this is what training is.
    def step!(seq)
      @grads.fill_zero
      @model.forward(seq)
      @model.backward(seq, @grads)
      @optimizer.step(@grads, @schedule.at(@step_idx))
      @step_idx += 1
      @grads.loss
    end

    # Reset optimizer state (e.g. after a warm-up step that you don't
    # want to count). Step counter stays where it is — change @step_idx
    # by hand if you want that too.
    def reset_optimizer!
      @optimizer.reset
    end

    # Convenience: current learning rate.
    def lr
      @schedule.at(@step_idx)
    end
  end
end
