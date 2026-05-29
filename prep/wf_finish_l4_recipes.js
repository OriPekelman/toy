export const meta = {
  name: 'finish-l4-from-scratch-recipe',
  description: 'Extract the FromScratch training recipe (L4) — the smallest viable Toy::LLM::Recipes::FromScratch + Trainer/Stage surface that encapsulates the existing random_init training loop — behind a bit-identical loss-curve gate (a recipe-driven fixture must reproduce smoke_projection_lens losses). Defer LoRA/WarmStart/Curriculum + the full DataSpec/Eval taxonomy. Halt if too design-open or the gate fails.',
  whenToUse: 'P2 L4 recipes, pass 1: only the FromScratch recipe, minimal API, gated by the from_scratch loss curve.',
  phases: [
    { title: 'Baseline', detail: 'record the from_scratch training loss curve (the gate target) + HEAD' },
    { title: 'Recon', detail: 'parallel read-only: existing training-loop code, what abstractions exist, the gate fixture seam, mirror gen' },
    { title: 'Plan', detail: 'design the SMALLEST viable FromScratch recipe API + the recipe-driven gate fixture; feasible flag' },
    { title: 'Build', detail: 'implement recipe + gate fixture; verify loss curve bit-identical; commit; halt on failure' },
  ],
}

// ---- Constants -------------------------------------------------------------

const MONOLITH = 'lib/llama_seq_forward_ffi.rb'
const MIRRORGEN = 'prep/gen_cuda_mirror.rb'
const RECIPE_DIR = 'lib/toy/llm/recipes/'

// The from_scratch loss curve is the gate target. smoke_projection_lens
// IS a from_scratch training loop; its losses are the reference.
const REF_BUILD = 'make examples/smoke_projection_lens'
const REF_FP = 'SEED=0 STEPS=5 ./examples/smoke_projection_lens 2>/dev/null | grep -E "^step|^initial="'  // baseline 4eaf019b…

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29f.md  (checkpoint; L4 is greenfield API design — read the L4 section)',
  'lib/toy/llm/recipes/README.md               (L4 contract SKETCH — none of it exists yet)',
  'lib/toy/llm/archs/llama_arch.rb             (the Arch a recipe composes)',
  'examples/smoke_projection_lens.rb           (an inlined from_scratch training loop = the gate reference; do NOT modify it)',
  'examples/06_train_from_scratch.rb           (the fuller from_scratch driver — the loop to encapsulate)',
  'lib/toy_trainer.rb / lib/training.rb / lib/toy_lr_schedule.rb  (existing training-loop machinery to reuse, not reinvent)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_lib_vs_example_scope.md  (lib is a library; experiment configs belong in examples/)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (#16 Struct.new — matz/spinel#1043)',
].join('\n  - ')

const FRAMING = `CRITICAL FRAMING.
(1) GREENFIELD, SO BE MINIMAL. Unlike L1-L3 (mechanical relocation of existing code), the L4 Recipe/Trainer/Stage/DataSpec/Eval classes do NOT exist — only a README sketch. Build ONLY the smallest viable surface to encapsulate the EXISTING from_scratch random_init training loop: Toy::LLM::Recipes::FromScratch plus the minimum Trainer/Stage needed. DEFER LoRA, WarmStart, Curriculum, and the full DataSpecs/Evals taxonomy — do not speculatively build them. Reuse existing machinery (toy_trainer / training / lr_schedule / the cache's build_training_step) rather than reinventing. Match the established module style (no require_relative "tinynn"). Surface the exact API you commit so it can be reviewed.
(2) GATE = BIT-IDENTICAL LOSS CURVE. The recipe must reproduce the from_scratch training behavior EXACTLY. Build a NEW fixture (e.g. examples/smoke_recipe_from_scratch.rb) that drives the SAME random_init config (vocab=627 d=64 donor=128 n_heads=4 d_ff=128 L=2 ctx=32, SEED=0, 5 steps) THROUGH Toy::LLM::Recipes::FromScratch and prints the same "step N: loss=" lines. Its loss curve MUST byte-equal the reference (4eaf019b… loss/initial lines from smoke_projection_lens). Do NOT modify smoke_projection_lens itself (that's the frozen reference — rewiring it through the recipe would be circular). If bit-identity can't be achieved, report infeasible with the blocker; do NOT loosen to a tolerance.
(3) SPINEL hygiene: no Struct.new (landmine #16), positional ctors, no default args, uniquely-prefixed members, no FFI :str into seeded builders. If a Spinel-compiled lib used on the GPU path changes, regenerate mirrors + verify-mirrors.
(4) ONE focused commit.`

// ---- Schemas ---------------------------------------------------------------

const BASE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['loss_curve', 'deterministic', 'notes'],
  properties: {
    loss_curve: { type: 'string', description: 'the exact "step N: loss=" + "initial=…" lines from the reference (the bit-identical target)' },
    deterministic: { type: 'boolean', description: 'two runs byte-identical' },
    notes: { type: 'string' },
  },
}

const RECON_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['facet', 'findings', 'line_refs', 'risks'],
  properties: {
    facet: { type: 'string' },
    findings: { type: 'string' },
    line_refs: { type: 'array', items: { type: 'string' } },
    reuse: { type: 'array', items: { type: 'string' }, description: 'existing classes/methods the recipe should REUSE (toy_trainer, build_training_step, lr_schedule, etc.) rather than reinvent' },
    risks: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['feasible', 'recipe_api', 'what_it_reuses', 'gate_fixture', 'files_touched', 'deferred', 'risks'],
  properties: {
    feasible: { type: 'boolean', description: 'true only if a minimal FromScratch recipe with a bit-identical loss-curve gate is achievable without speculative design' },
    recipe_api: { type: 'string', description: 'the EXACT minimal Toy::LLM::Recipes::FromScratch (+ minimum Trainer/Stage) definition to create. Smallest surface that works. Flag every public name introduced.' },
    what_it_reuses: { type: 'array', items: { type: 'string' } },
    gate_fixture: { type: 'string', description: 'the new recipe-driven fixture: name, config, how it prints the loss curve, expected to byte-equal the reference' },
    files_touched: { type: 'array', items: { type: 'string' } },
    deferred: { type: 'array', items: { type: 'string' }, description: 'LoRA/WarmStart/Curriculum/DataSpecs/Evals + anything else intentionally NOT built' },
    mirror_changes: { type: 'string' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['recipe_built', 'loss_curve_bit_identical', 'deterministic', 'committed', 'notes'],
  properties: {
    recipe_built: { type: 'boolean', description: 'true ONLY if loss_curve_bit_identical && deterministic && committed' },
    loss_curve_bit_identical: { type: 'boolean', description: 'recipe-driven fixture loss curve == reference, byte-for-byte' },
    deterministic: { type: 'boolean' },
    verify_mirrors_clean: { type: 'boolean', description: 'if any Spinel lib changed; state N/A otherwise' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    api_committed: { type: 'string', description: 'the exact public API shape committed (for user review)' },
    deferred: { type: 'array', items: { type: 'string' } },
    infeasible_blocker: { type: 'string' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: Baseline -----------------------------------------------------

phase('Baseline')

const baseline = await agent(`Record the from_scratch training LOSS CURVE — the bit-identical target for the FromScratch recipe. ${FRAMING}

Build \`${REF_BUILD}\`, then run \`${REF_FP}\` TWICE; confirm byte-identical (deterministic); record the exact loss-curve lines. This is what a recipe-driven training run must reproduce. Do NOT edit any file.`, { label: 'loss-curve-target', phase: 'Baseline', schema: BASE_SCHEMA })

const shaRaw = await agent(`Run \`git rev-parse HEAD\` then \`git status --porcelain --untracked-files=no\`. Report the 40-char sha. End with "CLEAN <sha>" if no tracked changes, else "DIRTY <sha>".`, { label: 'base-sha', phase: 'Baseline' })
const baseSha = ((shaRaw || '').match(/\b([0-9a-f]{40})\b/) || [])[1] || ''
const treeDirty = /\bDIRTY\b/.test(shaRaw || '')

log(`HEAD ${baseSha || '(unparsed)'} | tree ${treeDirty ? 'DIRTY' : 'clean'} | target det=${baseline?.deterministic}`)
if (!baseline || !baseline.deterministic) { log('ABORT: no deterministic loss-curve target.'); return { aborted: true, reason: 'no deterministic target', baseline } }
if (!baseSha || treeDirty) { log('ABORT: dirty tree / bad HEAD.'); return { aborted: true, reason: 'dirty tree', baseSha } }

// ---- Phase 1: Recon --------------------------------------------------------

phase('Recon')

const FACETS = [
  { key: 'training-loop', prompt: `Map the EXISTING from_scratch training loop: how examples/06_train_from_scratch.rb and examples/smoke_projection_lens.rb drive training (realize_for_random_init, build_training_step, the graph_reset/upload/compute_backward/download-loss loop, opt_step/AdamW, LR). Identify the loop body that the FromScratch recipe should encapsulate. What's in lib/toy_trainer.rb / lib/training.rb / lib/toy_lr_schedule.rb?` },
  { key: 'abstractions', prompt: `Determine what of the recipe contract (Recipe/Trainer/Stage/DataSpec/Eval) ALREADY exists vs is greenfield. grep lib/ for any Trainer/Stage/Recipe/optim classes. The README says none exist — confirm. List the MINIMUM new surface needed for FromScratch and what existing code it should wrap.` },
  { key: 'gate-fixture', prompt: `Design the gate seam: how to build examples/smoke_recipe_from_scratch.rb that drives the SAME config as smoke_projection_lens THROUGH a FromScratch recipe and prints identical "step N: loss=" lines. What does smoke_projection_lens do that the recipe must reproduce exactly (the donor projection-lens setup, label construction, hp vector, step loop)? Confirm smoke_projection_lens stays UNMODIFIED.` },
  { key: 'mirror-scope', prompt: `Will the recipe be Spinel-compiled / used on the GPU path (needs mirroring), or is it CPU-only orchestration? Check whether recipe code calls TinyNN.* directly (would need mirror) or only drives the cache. Note Spinel hazards (#16) for any new value object.` },
]

const reconArr = await parallel(FACETS.map((f) => () =>
  agent(`READ-ONLY recon for the L4 FromScratch recipe extraction. ${FRAMING}\n\nFacet: ${f.prompt}\n\nReport current file:line refs (grep). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: `recon:${f.key}`, phase: 'Recon', schema: RECON_SCHEMA })))
const recon = {}; FACETS.forEach((f, i) => { recon[f.key] = reconArr[i] })
for (const f of FACETS) { const r = recon[f.key]; log(`recon ${f.key}: ${r ? 'ok (' + (r.risks?.length || 0) + ' risks)' : 'FAILED'}`) }

// ---- Phase 2: Plan ---------------------------------------------------------

phase('Plan')

const plan = await agent(`Architect the L4 FromScratch recipe — the SMALLEST viable surface. ${FRAMING}\n\n== Recon ==\n${JSON.stringify(recon, null, 2)}\n\nDecide recipe_api (exact minimal Toy::LLM::Recipes::FromScratch + minimum Trainer/Stage; flag every public name), what_it_reuses, the gate_fixture design (recipe-driven, reproduces the loss curve), files_touched, and deferred (LoRA/WarmStart/Curriculum/DataSpecs/Evals). Set feasible=false if FromScratch can't be extracted without speculative design or a bit-identical gate isn't achievable. No Struct.new (#16). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: 'l4-plan', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan || !plan.feasible) {
  log(`STOP: L4 FromScratch not safely extractable as a minimal gated unit — ${plan ? (plan.risks || []).join('; ') : 'planner failed'}. No edits. Surface the design question to the user.`)
  return { aborted: true, reason: 'infeasible / too design-open', baseSha, recon, plan }
}
log(`Plan ready. API: ${plan.recipe_api?.slice(0, 120)}. Gate fixture: ${plan.gate_fixture?.slice(0, 80)}`)

// ---- Phase 3: Build --------------------------------------------------------

phase('Build')

const result = await agent(`Build the L4 FromScratch recipe per the plan. ${FRAMING}\n\n== Plan ==\n${JSON.stringify(plan, null, 2)}\n\n== Loss-curve target (must reproduce byte-for-byte) ==\n${baseline.loss_curve}\n\n== Procedure ==\n1. Implement the minimal Toy::LLM::Recipes::FromScratch (+ minimum Trainer/Stage) per recipe_api, reusing existing machinery.\n2. Build the recipe-driven gate fixture; run it; its loss curve MUST byte-equal the target above. Run TWICE for determinism. smoke_projection_lens MUST remain unmodified and still produce 4eaf019b… (re-run it to confirm you didn't disturb shared code).\n3. If bit-identity FAILS: do NOT loosen / do NOT alter behavior to force it. Diagnose; if it's a real blocker, recipe_built=false + infeasible_blocker, revert (git checkout -- / rm new files; clean vs ${baseSha}).\n4. If a Spinel lib changed: ruby ${MIRRORGEN} && make verify-mirrors (exit 0). State N/A if recipe is CPU-only orchestration.\n5. COMMIT (only if loss_curve_bit_identical && deterministic): git add ONLY new/changed source (NEVER -A, no binaries). Title "P2.6: L4 FromScratch recipe". Capture sha + the exact api_committed for review.\n\nrecipe_built=true ONLY if loss_curve_bit_identical && deterministic && committed. Reference docs:\n  - ${REFS}`, { label: 'build-recipe', phase: 'Build', schema: BUILD_SCHEMA })

if (result && result.recipe_built && result.committed) {
  log(`OK: L4 FromScratch recipe ${result.commit_sha || ''} | loss curve bit-identical | API: ${result.api_committed?.slice(0, 100)}`)
} else {
  log(`HALT: recipe not built — ${result ? (result.infeasible_blocker || result.notes) : 'agent returned nothing'}. Tree clean vs ${baseSha}.`)
}

return {
  baseSha,
  target: baseline.loss_curve,
  feasible: plan.feasible,
  result,
  followup: result && result.recipe_built
    ? 'FromScratch recipe landed (gated). Review the committed API. Deferred: LoRA/WarmStart/Curriculum recipes + DataSpecs/Evals taxonomy — each its own pass. P2 nears complete; then P3 core+CLI.'
    : 'FromScratch recipe not built — surface the design question; consider locking the recipe API with the user before retrying.',
}
