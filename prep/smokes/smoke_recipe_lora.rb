# L4 LoRA recipe gate. Drives the SAME LoRA fine-tune config as the
# frozen reference examples/03_finetune_lora.rb (at the fixed config
# below) THROUGH Toy::LLM::Recipes::LoRA and prints "step N: loss="
# lines whose loss VALUES MUST byte-equal the reference's per-step CE.
#
#   make prep/smokes/smoke_recipe_lora
#   STEPS=5 RANK=8 ./prep/smokes/smoke_recipe_lora | grep '^step'
#
# Reference (examples/03_finetune_lora.rb, STEPS=5 RANK=8 LR=0.001
# GGUF=data/smollm2-135m-native.gguf, seeded upload_lora_q_init!(42,0.01)):
#   step 1: loss=9.236274719238281
#   step 2: loss=9.213868141174316
#   step 3: loss=9.153078079223633
#   step 4: loss=9.050341606140137
#   step 5: loss=8.904613494873047
#
# The ONLY difference from the inlined reference is that realize/step go
# THROUGH the recipe. Every numeric input (GGUF base, config, untied/
# qkv_bias flags, RANK, seeded init, TOKENS, positions, one-hot labels,
# per-step bias-corrected hp) is identical to 03_finetune_lora.rb so the
# loss is bit-identical. Experiment config + Mat construction stay in
# this FIXTURE per lib-vs-example scope. CPU-only, like FromScratch.
#
# Load order is verbatim so the backend TinyNN loads (via
# llama_seq_forward_ffi) before the recipe is required.

require_relative "../../lib/toy"
require_relative "../../lib/toy/models/toy_smollm2"
require_relative "../../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../../lib/toy/llm/engine/llama_seq_engine"
require_relative "../../lib/toy/llm/adamw"
require_relative "../../lib/toy/llm/recipes/lora"

GGUF  = ENV["GGUF"]  || "data/smollm2-135m-native.gguf"
RANK  = (ENV["RANK"]  || "8").to_i
STEPS = (ENV["STEPS"] || "5").to_i
LR    = (ENV["LR"]    || "0.001").to_f

# T=4 prompt; CE objective pushes every position's argmax toward
# TARGET_ID. Identical to the reference (03_finetune_lora.rb:41-42).
TOKENS    = [12092, 4845, 253, 1429]
TARGET_ID = (ENV["TARGET_ID"] || "99").to_i

if !File.exist?(GGUF)
  puts "smoke_recipe_lora: cannot find " + GGUF
  exit 1
end

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s
puts "training: GGUF=" + GGUF + " RANK=" + RANK.to_s + " STEPS=" + STEPS.to_s

gguf   = TinyNN.tnn_gguf_load(GGUF)
recipe = Toy::LLM::Recipes::LoRA.new
# Named realize options (toy#64): same values as the reference call
# sites (T positions, untied/qkv_bias flags, seed=42, init_scale=0.01).
# RANK stays a leading positional (lora-specific).
#
# TYPE-PIN (`? true : false`): SmolLM2Flags has a default-arg ctor
# (landmine #4), so flags.untied/.qkv_bias read back as poly. Storing
# poly into RecipeOptions ivars silently poisons the options object and
# SHIFTS THE LOSS CURVE from step 2 (verified on spinel a699cf9:
# step 2 9.213868141174316 -> 9.213866233825684 without the pin — no
# analyzer warning fires). The ternary forces a concrete Bool. Keep it.
opts = Toy::LLM::RecipeOptions.new
opts.t_seq      = TOKENS.length
opts.untied     = flags.untied ? true : false
opts.qkv_bias   = flags.qkv_bias ? true : false
opts.seed       = 42
opts.init_scale = 0.01
recipe.realize!(gguf, cfg, RANK, opts)
puts "realize OK"

# Vocab × T one-hot label matrix in Mat(T, vocab) layout — identical
# construction to 03_finetune_lora.rb:79-86.
m_labels = Mat.new(TOKENS.length, cfg.vocab)
i = 0
while i < TOKENS.length * cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
ti = 0
while ti < TOKENS.length
  m_labels.flat[ti * cfg.vocab + TARGET_ID] = 1.0
  ti = ti + 1
end

# NAMED AdamW. lora differs from the from-scratch defaults: beta2=0.999
# and bias_correct=true, so slots 5/6 carry the PER-STEP bias-correction
# denominators 1/(1-beta^t) — NOT constant betas (see the loud finding in
# lib/toy/llm/adamw.rb: the lora FFI graph reads slots 5/6 DIFFERENTLY
# from the from-scratch/warm/vit graphs). lr comes from ENV (default
# 0.001). m_hp is rebuilt per step below.
adamw = Toy::AdamW.for_lora   # beta2=0.999 + per-step bias correction
adamw.lr = LR

positions = [0, 1, 2, 3]

losses = [0.0]; losses.pop
step = 1
while step <= STEPS
  # 1-indexed step; bias_correct=true → slots5/6 = 1/(1-0.9^t),
  # 1/(1-0.999^t). Byte-identical to 03_finetune_lora.rb:177-178.
  m_hp = adamw.hp(step)
  loss = recipe.step!(TOKENS, positions, m_labels, m_hp, step == 1)
  losses.push(loss)
  puts "step " + step.to_s + ": loss=" + loss.to_s
  step = step + 1
end

initial = losses[0]
final   = losses[losses.length - 1]
ratio   = final / initial
puts "initial=" + initial.to_s + " final=" + final.to_s + " ratio=" + ratio.to_s
if ratio < 1.0
  puts "VERDICT: lora recipe training is learning"
else
  puts "VERDICT: training NOT learning (final/initial = " + ratio.to_s + ")"
end
