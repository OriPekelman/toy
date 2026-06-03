# Training utilities — small classes that live around the model:
# learning-rate schedule, batch loader, optimizer wrapper, and the
# tokenized-corpus file readers. Designed to be Spinel-compatible
# (no kwargs, no blocks-as-iterators on user types).

# ---------------------------------------------------------------------------
#   Corpus readers — load vocab, sequences, and prompt from data/ts_*.txt
# ---------------------------------------------------------------------------

def read_vocab(path)
  vocab = ["?"]
  vocab.pop
  File.open(path, "r") do |f|
    f.each_line { |line| vocab.push(line.chomp) }
  end
  vocab
end

def parse_ids(line)
  parts = line.split(" ")
  ids   = [parts[0].to_i]
  k = 1
  while k < parts.length
    ids.push(parts[k].to_i)
    k += 1
  end
  ids
end

def read_sequences(path)
  raw = ["?"]
  raw.pop
  File.open(path, "r") do |f|
    f.each_line { |line| raw.push(line.chomp) }
  end
  seqs = [parse_ids(raw[0])]
  i = 1
  while i < raw.length
    seqs.push(parse_ids(raw[i]))
    i += 1
  end
  seqs
end

def read_prompt(path)
  raw = ["?"]
  raw.pop
  File.open(path, "r") do |f|
    f.each_line { |line| raw.push(line.chomp) }
  end
  parse_ids(raw[0])
end


# ---------------------------------------------------------------------------
#   LRSchedule — linear warmup → cosine decay
# ---------------------------------------------------------------------------
# at(step) returns the LR for the given (0-indexed) optimizer step.
# During warmup the LR ramps linearly from 0 to max_lr; after warmup
# a half-period cosine decays from max_lr to min_lr over the
# remaining (total_steps − warmup_steps) steps.
class LRSchedule
  attr_accessor :warmup_steps, :total_steps, :max_lr, :min_lr

  def initialize(warmup_steps, total_steps, max_lr, min_lr)
    @warmup_steps = warmup_steps
    @total_steps  = total_steps
    @max_lr       = max_lr
    @min_lr       = min_lr
  end

  def at(step)
    if step < @warmup_steps
      return @max_lr * (step.to_f + 1.0) / @warmup_steps.to_f
    end
    if step >= @total_steps
      return @min_lr
    end
    progress = (step - @warmup_steps).to_f / (@total_steps - @warmup_steps).to_f
    cos_v    = 0.5 * (1.0 + Math.cos(Math::PI * progress))
    @min_lr + (@max_lr - @min_lr) * cos_v
  end
end


# ---------------------------------------------------------------------------
#   DataLoader — manual batch iteration over an Array of IntArrays
# ---------------------------------------------------------------------------
# Spinel doesn't let us hand a class instance to the each_slice / each_with_index
# style, and Spinel compiles every class method (with poly fallback for
# uncalled-typed params) — uncalled `at(i)` would force @sequences's element
# type to PolyArray and poison IntArray inference everywhere. So we keep
# DataLoader to batch-bounds only, and the training loop reads sequences
# directly: `seq = sequences[inner]`.
class DataLoader
  attr_accessor :sequences, :batch_size

  def initialize(sequences, batch_size)
    @sequences  = sequences
    @batch_size = batch_size
  end

  def length
    @sequences.length
  end

  def batch_count
    n  = @sequences.length
    bc = n / @batch_size
    if n % @batch_size != 0
      bc += 1
    end
    bc
  end

  def batch_start(bi)
    bi * @batch_size
  end

  def batch_end(bi)
    e = (bi + 1) * @batch_size
    if e > @sequences.length
      e = @sequences.length
    end
    e
  end
end


# ---------------------------------------------------------------------------
#   Adam — optimizer wrapper that owns the m/v state and steps the model
# ---------------------------------------------------------------------------
# Constructed from the model so the AdamState shapes itself to the model's
# parameter inventory. step(grads, lr) does one Adam update; reset zeroes
# the moments + bias-correction products (used right after the warm-up
# call so the warm-up step doesn't leak into real training).
class Adam
  attr_accessor :model, :state, :beta1, :beta2, :eps

  def initialize(model, beta1, beta2, eps)
    @model = model
    @state = AdamState.new(model.vocab_size, model.d_model, model.d_ff,
                           model.n_heads, model.d_head, model.n_layers,
                           model.context_length)
    @beta1 = beta1
    @beta2 = beta2
    @eps   = eps
  end

  def step(grads, lr)
    @model.apply_gradients_adam(grads, @state, lr, @beta1, @beta2, @eps)
  end

  def reset
    @state.bc1 = 1.0
    @state.bc2 = 1.0
    @state.m.fill_zero
    @state.v.fill_zero
  end
end
