# lib/toy/run/train_gpt2.rb — Spinel-compiled GPT-2 from-scratch TRAINING runner.
#
# Trains a GPT-2-shape model from scratch (loss 6.44 → decreasing on the
# from-scratch corpus). REQUIRE-PATH CAUTION (this bit us hard, 2026-06-04): the
# toplevel lib files are `require_relative "../../toy"` / `"../../tinynn"` (two
# levels up from lib/toy/run/), NOT `"../toy"`. A wrong path resolves to a
# nonexistent file; Spinel *ignores* the require (warning only), so TinyNN/Mat
# are never loaded and every `TinyNN.*`/`Mat.new` call emits 0 → zero
# weights/labels → CE=0, no crash. `spinel <file>.rb --emit-types` surfaces the
# "require could not be resolved … ignored" line that fingerprints this.
#
# The lib-side compute for `toy train from-scratch --arch gpt2`. A SEPARATE
# binary (libexec/toy-train-gpt2) from the llama runner (lib/toy/run/train.rb):
# compiling the GPT-2 realize path alongside the llama one would make Spinel
# merge the engine/cfg receiver types (landmine #16) — the same reason the LoRA
# recipe is its own binary. This keeps both realize paths monomorphic and
# protects the llama byte-exact gates from GPT-2 churn.
#
# Env (mirrors the llama from-scratch runner):
#   STEPS   — training steps (default "5")
#   SEED    — random-init seed (default "0")
#   plus the shape knobs below (VOCAB/D_MODEL/N_HEADS/D_FF/N_LAYERS/CONTEXT).
#
# Output: "step <N>: loss=<float>" per step on STDOUT. Trains on the first line
# of data/ts_seqs.txt (the from-scratch ground-truth corpus). Backward of the
# LayerNorm + GELU rides the two vendored kernels (vendor-patches/0007). CPU
# only (this slice); the CUDA/Metal mirrors come after the CPU gate.

require_relative "../../toy"
require_relative "../ffi/tinynn"
require_relative "../llm/engine/gpt2_seq_engine"
require_relative "../llm/adamw"

STEPS    = (ENV["STEPS"]    || "5").to_i
SEED     = (ENV["SEED"]     || "0").to_i
LR       = (ENV["LR"]       || "0.001").to_f
# From-scratch gate shape (literal, like the llama from-scratch runner). Env-driven
# shapes are a follow-up; not blocked by anything (the earlier "runtime dims
# degrade" claim was a misdiagnosis of the require-path bug — see the header).
VOCAB    = 627
D_MODEL  = 64
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

engine = Toy::LLM::Engine::GPT2SeqEngine.new
engine.realize!(VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS, CONTEXT, SEED)

# First corpus line → CONTEXT token ids, zero-padded (from-scratch ground truth).
raw     = File.read("data/ts_seqs.txt")
lines   = raw.split("\n")
parts   = lines[0].split(" ")
seq_ids = [0]; seq_ids.pop
k = 0
while k < CONTEXT
  if k < parts.length
    seq_ids.push(parts[k].to_i)
  else
    seq_ids.push(0)
  end
  k = k + 1
end

positions = [0]; positions.pop
p = 0
while p < CONTEXT
  positions.push(p)
  p = p + 1
end

# Shift-by-one next-token one-hot labels (target = next token, self at last pos).
# (Toy::Labels.next_token would also work now that the requires are correct; kept
# inline as the minimal self-contained form.)
m_labels = Mat.new(CONTEXT, VOCAB)
zj = 0
while zj < CONTEXT * VOCAB; m_labels.flat[zj] = 0.0; zj = zj + 1; end
lk = 0
while lk < CONTEXT
  tgt = (lk + 1 < CONTEXT) ? seq_ids[lk + 1].to_i : seq_ids[lk].to_i
  m_labels.flat[lk * VOCAB + tgt] = 1.0
  lk = lk + 1
end

# NAMED AdamW (byte-identical to the old hand-filled m_hp). slots 5/6 =
# 1/(1-beta^t) bias correction, matching the gated inline GPT-2 trainer.
adamw = Toy::AdamW.for_lora   # gpt2 graph reads the lora hp convention
adamw.lr = LR

step = 0
while step < STEPS
  m_hp = adamw.hp(step + 1)
  loss = engine.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  step = step + 1
end
