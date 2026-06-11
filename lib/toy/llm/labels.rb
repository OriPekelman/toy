# lib/toy/llm/labels.rb — Toy::Labels: shift-by-one one-hot label Mats.
#
# Pure-Ruby builders for the next-token (shift-by-one) one-hot label
# matrix that the LM training recipes hand to step!. They build the
# SAME Mat the runners used to fill with a hand-written triple loop,
# byte-identically. NO TinyNN / FFI calls — only Mat (already loaded
# transitively by every caller via require_relative "../../toy"), so
# this file adds NO require and is SHARED (not mirrored) across the
# CPU / CUDA / Metal runners.
#
# Spinel hygiene: a module with `self.` methods (no ivar layout to
# collide), NO #{} interpolation, NO kwargs / default args.
#
# TWO named methods (preferred over a `guard:` boolean — recipes are
# read by users, and two named methods read cleaner; both byte-safe):
#   next_token          — UNGUARDED  (from-scratch: known-good seq_ids)
#   next_token_guarded  — in-vocab guarded (warm-start: streamed corpus)
#
# OUT OF SCOPE (deliberately NOT routed through here, they are NOT
# shift-by-one): lora's constant-TARGET_ID one-hot (every row targets
# the same id) and vit's single-class one-hot stay as inline builds in
# their runners.

module Toy
  module Labels
    # UNGUARDED shift-by-one one-hot. Reproduces the from-scratch hand
    # loop (train.rb:297-304, smoke_recipe_from_scratch.rb:71-78)
    # VERBATIM. target = next token, or self at the last position.
    #
    # `batch` is INCLUDED per the requested signature but is NOT
    # multiplied into the row count: all current callers are
    # single-sequence context×vocab (batch implicitly 1). Multiplying
    # it in would change the Mat shape and break the byte gate. It is a
    # forward-looking param for a future batched caller — and because a
    # batch != 1 would otherwise be SILENTLY IGNORED (training on a
    # wrongly-shaped one-hot), it now FAILS LOUD (toy#64 item 5).
    def self.next_token(seq_ids, vocab, context, batch)
      if batch != 1
        raise "Toy::Labels.next_token: batch " + batch.to_s +
              " unsupported — batched training deferred (the one-hot " +
              "is context x vocab; batch is not multiplied into the " +
              "row count)"
      end
      m = Mat.new(context, vocab)
      j = 0
      while j < context * vocab
        m.flat[j] = 0.0
        j = j + 1
      end
      k = 0
      while k < context
        target = (k + 1 < context) ? seq_ids[k + 1] : seq_ids[k]
        m.flat[k * vocab + target] = 1.0
        k = k + 1
      end
      m
    end

    # IN-VOCAB-GUARDED shift-by-one one-hot. Reproduces the warm-start
    # hand loop (train.rb:202-214, smoke_recipe_warm_start.rb:111-124)
    # VERBATIM. Identical to next_token EXCEPT the scatter is guarded by
    # `target >= 0 && target < vocab` (with `&&` exactly) — warm-start
    # streams arbitrary corpus, so it guards against out-of-vocab ids.
    # Rebuilt every step (seq_ids streams from the corpus).
    #
    # `batch` is INCLUDED per the requested signature, NOT multiplied
    # into the row count (same rationale as next_token) — and FAILS
    # LOUD on batch != 1 (toy#64 item 5, same trap).
    def self.next_token_guarded(seq_ids, vocab, context, batch)
      if batch != 1
        raise "Toy::Labels.next_token_guarded: batch " + batch.to_s +
              " unsupported — batched training deferred (the one-hot " +
              "is context x vocab; batch is not multiplied into the " +
              "row count)"
      end
      m = Mat.new(context, vocab)
      j = 0
      while j < context * vocab
        m.flat[j] = 0.0
        j = j + 1
      end
      k = 0
      while k < context
        target = (k + 1 < context) ? seq_ids[k + 1] : seq_ids[k]
        if target >= 0 && target < vocab
          m.flat[k * vocab + target] = 1.0
        end
        k = k + 1
      end
      m
    end
  end
end
