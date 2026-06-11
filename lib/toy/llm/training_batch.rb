# lib/toy/llm/training_batch.rb — Toy::LLM::TrainingBatch: a VALIDATING
# wrapper for the per-step training quartet (toy#64 item 3).
#
# Every recipe step! takes the same four inputs — seq_ids, positions,
# labels, hp — and before this class each fixture hand-built them with
# zero validation: a seq_ids of the wrong length silently trained on
# garbage, an out-of-vocab id silently scattered a one-hot off the end
# of the row. TrainingBatch fails LOUD instead:
#
#   batch = Toy::LLM::TrainingBatch.new(VOCAB, CONTEXT, 1)
#   batch.fill!(seq_ids)              # validates + rebuilds labels
#   batch.hp = Toy::AdamW.new.hp(0)   # caller-owned (optimizer knob)
#   recipe.step!(batch.seq_ids, batch.positions, batch.labels,
#                batch.hp, step == 0)
#
# The ctor builds the validated positions vector (0..context-1, the
# only shape every current graph accepts); fill! validates the sequence
# (length == context, every id in 0...vocab) and rebuilds the labels
# Mat via Toy::Labels.next_token — byte-identical to the hand loops it
# replaces. fill_fixed_target! (toy#73 item 3) is the second objective:
# same sequence validation, labels target ONE fixed id at every
# position (the lora-smoke shape). hp stays a plain accessor: which
# AdamW convention applies is the caller's choice (see the slot-5/6
# finding in adamw.rb). The image-classification sibling is
# Toy::LLM::ClassifyBatch (classify_batch.rb).
#
# batch_size must be 1: batched training is deferred (the Labels
# builders do not multiply batch into the row count — see labels.rb);
# a silent batch>1 here would train on a wrongly-shaped one-hot.
#
# Pure-Ruby: NO TinyNN / FFI calls — only Mat (via Toy::Labels) — so it
# is SHARED (not mirrored) across CPU / CUDA / Metal, like AdamW /
# Labels / RecipeOptions.
#
# Spinel hygiene: PLAIN class (NO Struct.new — landmine #16), explicit
# ctor with NO default args (landmine #4), while-loops, NO #{}
# interpolation (raise messages use String concat — raise verified on
# spinel a699cf9: compiles, aborts nonzero with the message). Plain
# accessor names (monomorphic Int / Array / Mat members).

require_relative "labels"

module Toy; module LLM
  class TrainingBatch
    attr_accessor :vocab, :context, :batch, :seq_ids, :positions,
                  :labels, :hp

    def initialize(vocab, context, batch_size)
      if vocab <= 0
        raise "TrainingBatch: vocab must be positive, got " + vocab.to_s
      end
      if context <= 0
        raise "TrainingBatch: context must be positive, got " + context.to_s
      end
      if batch_size != 1
        raise "TrainingBatch: batch_size " + batch_size.to_s +
              " unsupported — batched training deferred (labels are " +
              "context x vocab, batch is not multiplied into the row count)"
      end
      @vocab   = vocab
      @context = context
      @batch   = batch_size

      # The validated positions vector: 0..context-1, the only shape
      # every current training graph accepts (RoPE reads positions[k]).
      @positions = [0]; @positions.pop
      p = 0
      while p < context
        @positions.push(p)
        p = p + 1
      end

      # Typed-empty until fill! (type-pin Array[Int] — literal-seed +
      # pop, the codebase's standard pattern).
      @seq_ids = [0]; @seq_ids.pop

      # Zero labels Mat at the final shape until fill! rebuilds it; hp
      # is caller-owned (Mat(1,7), see Toy::AdamW#hp) — zero Mat(1,7)
      # placeholder so the member type is concrete from construction.
      @labels = Mat.new(context, vocab)
      @hp     = Mat.new(1, 7)
    end

    # Validate the sequence and rebuild the labels Mat. Fails LOUD on a
    # length mismatch or an out-of-vocab id (the unguarded one-hot
    # scatter would silently write outside the row otherwise). Labels
    # are rebuilt via Toy::Labels.next_token — byte-identical to the
    # hand-built shift-by-one one-hot it replaces. Returns nil.
    def fill!(new_seq_ids)
      if new_seq_ids.length != @context
        raise "TrainingBatch#fill!: seq_ids length " +
              new_seq_ids.length.to_s + " != context " + @context.to_s
      end
      k = 0
      while k < @context
        id = new_seq_ids[k]
        if id < 0 || id >= @vocab
          raise "TrainingBatch#fill!: seq_ids[" + k.to_s + "] = " +
                id.to_s + " out of vocab 0..." + @vocab.to_s
        end
        k = k + 1
      end
      @seq_ids = new_seq_ids
      @labels  = Toy::Labels.next_token(new_seq_ids, @vocab, @context, @batch)
      nil
    end

    # FIXED-TARGET objective (toy#73 item 3): validate the sequence the
    # same way fill! does, but build labels where EVERY position
    # targets `target_id` (the lora-smoke objective — push the whole
    # prompt toward one token). Same loud guarantees: length mismatch,
    # out-of-vocab id, and out-of-vocab target all raise (via
    # Toy::Labels.fixed_target for the target). Returns nil.
    def fill_fixed_target!(new_seq_ids, target_id)
      if new_seq_ids.length != @context
        raise "TrainingBatch#fill_fixed_target!: seq_ids length " +
              new_seq_ids.length.to_s + " != context " + @context.to_s
      end
      k = 0
      while k < @context
        id = new_seq_ids[k]
        if id < 0 || id >= @vocab
          raise "TrainingBatch#fill_fixed_target!: seq_ids[" + k.to_s +
                "] = " + id.to_s + " out of vocab 0..." + @vocab.to_s
        end
        k = k + 1
      end
      @seq_ids = new_seq_ids
      @labels  = Toy::Labels.fixed_target(@vocab, @context, target_id)
      nil
    end
  end
end; end
