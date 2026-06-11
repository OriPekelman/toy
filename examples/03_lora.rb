# examples/03_lora.rb — LoRA adapters over a frozen, mmap'd base model.
#
# WHAT YOU'LL SEE: a REAL pretrained model (SmolLM2-135M by default) is
# mmap'd read-only as the frozen base; rank-8 LoRA adapter pairs (~5 MB
# of trainable f32) are attached to the attention projections; training
# pushes every position of a 4-token prompt toward one target token.
# Cross-entropy collapses from ~9.2 within 20 steps — the adapters are
# learning while the base stays byte-untouched on disk.
#
# HOW LONG: ~10 s for 20 steps (CPU). The base GGUF is a dev artifact —
# convert one with ./prep/convert_smollm2_to_gguf.py --ggml-native, or
# point GGUF= at any *native-layout* llama-family GGUF you have
# (`toy list`). Q8_0 bases work too: that is QLoRA (Q8 base + f32
# adapters), same command.
#
#   make example_03
#   ./examples/example_03_lora
#   GGUF=data/qwen25-0.5b-native-q8.gguf ./examples/example_03_lora   # QLoRA
#
# WHAT TO TWEAK (env, no recompile):
#   RANK=16        bigger adapters (more capacity, more params)
#   STEPS=50 LR=0.01 TARGET_ID=42   the toy objective
#
# THE API — Toy::LLM::Recipes::LoRA:
#   realize!(gguf, cfg, rank, opts)  — mmap base + attach rank-R adapters
#   step!(ids, pos, labels, hp, is_first) — same contract as the siblings
# NOTE: LoRA is required DIRECTLY (not part of the one-require
# toy/compute surface yet — Spinel constructor-slot inference, toy#52).
# The CLI form of this example is `toy train lora`.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_seq_engine"
require_relative "../lib/toy/llm/adamw"
require_relative "../lib/toy/llm/recipes/lora"

GGUF      = ENV["GGUF"]  || "data/smollm2-135m-native.gguf"
RANK      = (ENV["RANK"]  || "8").to_i
STEPS     = (ENV["STEPS"] || "20").to_i
LR        = (ENV["LR"]    || "0.001").to_f
TARGET_ID = (ENV["TARGET_ID"] || "99").to_i

# The toy objective: a fixed 4-token prompt; every position's next-token
# target is TARGET_ID. Dumb on purpose — the visible point is the CE
# collapse, not the task.
TOKENS = [12092, 4845, 253, 1429]

# Fail LOUD before compute (spinel-dev#17: silent File.read of a
# missing path returns "").
if !File.exist?(GGUF)
  puts "03_lora: base GGUF not found: " + GGUF
  puts "  LoRA needs a NATIVE-layout llama-family GGUF (mmap'd in place)."
  puts "  convert one:  ./prep/convert_smollm2_to_gguf.py --ggml-native --out " + GGUF
  puts "  or point GGUF= at one you have (see `toy list`)."
  exit 1
end

# Shape + layout flags come from the GGUF itself.
cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "base: " + GGUF
puts "model: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s +
     "  adapters: rank=" + RANK.to_s

gguf   = TinyNN.tnn_gguf_load(GGUF)
recipe = Toy::LLM::Recipes::LoRA.new

# Named realize options. The `? true : false` pins poly Bools from the
# flags reader to concrete Bools (Spinel landmine — see
# prep/smokes/smoke_recipe_lora.rb for the full story).
opts = Toy::LLM::RecipeOptions.new
opts.t_seq      = TOKENS.length
opts.untied     = flags.untied ? true : false
opts.qkv_bias   = flags.qkv_bias ? true : false
opts.seed       = 42
opts.init_scale = 0.01
recipe.realize!(gguf, cfg, RANK, opts)
puts "realize OK (base mmap'd, adapters attached)"

# One-hot labels: every position targets TARGET_ID. (A custom objective
# like this is hand-built — Toy::Labels/TrainingBatch model the
# next-token objective only.)
m_labels = Mat.new(TOKENS.length, cfg.vocab)
i = 0
while i < TOKENS.length * cfg.vocab
  m_labels.flat[i] = 0.0
  i = i + 1
end
ti = 0
while ti < TOKENS.length
  m_labels.flat[ti * cfg.vocab + TARGET_ID] = 1.0
  ti = ti + 1
end

# LoRA's AdamW convention differs from from-scratch: beta2=0.999 and
# PER-STEP bias correction (slots 5/6 of hp carry the correction
# denominators, not the betas) — hence hp is rebuilt every step.
adamw = Toy::AdamW.for_lora
adamw.lr = LR

positions = [0, 1, 2, 3]

first_loss = 0.0
final_loss = 0.0
step = 1
while step <= STEPS
  m_hp = adamw.hp(step)                  # 1-indexed for bias correction
  loss = recipe.step!(TOKENS, positions, m_labels, m_hp, step == 1)
  if step == 1
    first_loss = loss
  end
  final_loss = loss
  puts "step " + step.to_s + ": loss=" + loss.to_s
  step = step + 1
end

puts ""
puts "lora: " + STEPS.to_s + " steps, loss " + first_loss.to_s +
     " -> " + final_loss.to_s
if final_loss < first_loss
  puts "VERDICT: adapters learning (base weights untouched)"
else
  puts "VERDICT: NOT learning (loss did not fall — try more STEPS or higher LR)"
end
