# prep/smokes/smoke_franken_parity.rb — toy#109 P2: the engine-parity proof.
#
# THREE sequential runs of the SAME tiny config in one process (sequential
# sessions are safe on the global sched — only interleaved compute breaks;
# the P1 finding):
#
#   A. Recipes::FromScratch            — the existing byte-pinned path
#   B. Recipes::FrankenFromScratch     — EMPTY policy (all-chain)
#   C. Recipes::FrankenFromScratch     — policy all-:dfa (attention q/k/v)
#
# Asserts: (1) A and B loss curves are IDENTICAL strings per step (the
# all-chain null hypothesis: build_training_step_franken emits the same
# graph); (2) C differs from A and still DECREASES (the dfa arm trains
# through the real llama engine, per-head granularity).
#
#   make prep/smokes/smoke_franken_parity && ./prep/smokes/smoke_franken_parity
#
# Expected last line: "franken-parity: ok".

require_relative "../../lib/toy/compute"
require_relative "../../lib/toy/llm/recipes/franken_from_scratch"

VOCAB   = 627
CONTEXT = 16
STEPS   = 12

def build_batch
  seq_ids = [0]
  seq_ids.pop
  i = 0
  while i < CONTEXT
    seq_ids.push(i % VOCAB)
    i = i + 1
  end
  batch = Toy::LLM::TrainingBatch.new(VOCAB, CONTEXT, 1)
  batch.fill!(seq_ids)
  batch.hp = Toy::AdamW.for_from_scratch.hp(0)
  batch
end

def build_opts(policy)
  opts = Toy::LLM::RecipeOptions.new
  opts.t_seq = CONTEXT
  pi = 0
  while pi < policy.length
    opts.credit_assignment.push(policy[pi])
    pi = pi + 1
  end
  opts.dfa_b_seed = 42

  opts
end

# MONOMORPHIC drivers — one concrete recipe class per function (poly
# receiver in one local slot garbles whole-program arg typing; found
# the hard way: opts reached cfg.head_dim).
def run_from_scratch_curve(policy)
  cfg = Toy::SmolLM2Config.new(VOCAB, 64, 4, 4, 128, 2, CONTEXT, 10000.0, 1.0e-5)
  opts = build_opts(policy)
  batch = build_batch
  curve = [""]
  curve.pop
  recipe = Toy::LLM::Recipes::FromScratch.new
  recipe.realize!(cfg, opts)
  step = 0
  while step < STEPS
    loss = recipe.step!(batch.seq_ids, batch.positions, batch.labels,
                        batch.hp, step == 0)
    curve.push(loss.to_s)
    step = step + 1
  end
  curve
end

def run_franken_curve(policy, mix_alpha)
  cfg = Toy::SmolLM2Config.new(VOCAB, 64, 4, 4, 128, 2, CONTEXT, 10000.0, 1.0e-5)
  opts = build_opts(policy)
  opts.dfa_mix_alpha = mix_alpha
  batch = build_batch
  curve = [""]
  curve.pop
  recipe = Toy::LLM::Recipes::FrankenFromScratch.new
  recipe.realize!(cfg, opts)
  step = 0
  while step < STEPS
    loss = recipe.step!(batch.seq_ids, batch.positions, batch.labels,
                        batch.hp, step == 0)
    curve.push(loss.to_s)
    step = step + 1
  end
  curve
end

empty_pol = [0]; empty_pol.pop
dfa_pol   = [1, 1, 1, 1]
mix1_pol  = [2, 2, 2, 2]   # with MIX_ALPHA=1.0 must byte-equal chain

curve_a = run_from_scratch_curve(empty_pol)
curve_b = run_franken_curve(empty_pol, 0.5)
curve_c = run_franken_curve(dfa_pol, 0.5)
curve_d = run_franken_curve(mix1_pol, 1.0)

ok = true

i = 0
while i < STEPS
  puts "step " + i.to_s + ": from_scratch=" + curve_a[i] +
       " franken_chain=" + curve_b[i] + " franken_dfa=" + curve_c[i]
  if curve_a[i] != curve_b[i]
    puts "PARITY FAIL at step " + i.to_s + ": " + curve_a[i] + " != " + curve_b[i]
    ok = false
  end
  i = i + 1
end

# mix(1.0) vs chain: BYTE equality. The long-standing "near-null drift"
# (tolerance 1e-5, then 2e-2) was a PROBE BUG, resolved 2026-07-26: this
# smoke set a $mix_alpha_override global that NOTHING read, so the mix
# arm ran at opts.dfa_mix_alpha's default 0.5 — the "drift" was real
# half-DFA signal (init-dependent, glacial under Adam sign-scale),
# never sched noise. With alpha actually 1.0 the combiner graph
# [add(scale(acc,1.0), scale(gd,0.0))] is BYTE-inert: 16-step value
# dump showed g == acc bitwise on all 24 mix weights, zero weight
# divergence, identical loss strings.
i = 0
while i < STEPS
  if curve_d[i] != curve_a[i]
    puts "MIX(1.0) BYTE-PARITY FAIL at step " + i.to_s + ": " +
         curve_d[i] + " != " + curve_a[i]
    ok = false
  end
  i = i + 1
end

if curve_c[0] != curve_a[0]
  # step 0 loss is pre-update forward on identical init — MUST match too
  puts "DFA-ARM FAIL: step-0 forward differs (" + curve_c[0] + " vs " + curve_a[0] + ")"
  ok = false
end
last_c  = curve_c[STEPS - 1].to_f
first_c = curve_c[0].to_f
if !(last_c < first_c)
  puts "DFA-ARM FAIL: loss did not decrease (" + first_c.to_s + " -> " + last_c.to_s + ")"
  ok = false
end
if curve_c[STEPS - 1] == curve_a[STEPS - 1]
  puts "DFA-ARM FAIL: dfa curve identical to chain (policy had no effect)"
  ok = false
end

if ok
  puts "franken-parity: engine all-chain byte-parity + per-head dfa arm trains + mix(1.0) byte-null"
  puts "franken-parity: ok"
else
  puts "franken-parity: FAIL"
end
