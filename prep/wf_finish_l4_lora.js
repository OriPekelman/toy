export const meta = {
  name: 'finish-l4-lora-recipe',
  description: 'Extract the LoRA fine-tune recipe (L4) — minimal Toy::LLM::Recipes::LoRA wrapping the existing realize_for_mmap + upload_lora_q_init! + training loop — behind a bit-identical loss-curve gate (a recipe-driven fixture must reproduce the 03_finetune_lora loss curve). Sibling to FromScratch; same realize!/step! shape, no new taxonomy. Halt if the gate fails or determinism breaks.',
  whenToUse: 'P2 L4 recipes, pass 2: the LoRA recipe (single-stage, fits the FromScratch flat pattern). WarmStart/Curriculum (multi-stage → taxonomy fork) come after, with a user check.',
  phases: [
    { title: 'Baseline', detail: 'record a deterministic LoRA-finetune loss curve (the gate target) + HEAD' },
    { title: 'Recon', detail: 'parallel read-only: the LoRA training loop vs FromScratch, the realize_for_mmap+lora-init path, gate fixture seam, mirror scope' },
    { title: 'Plan', detail: 'design the minimal Toy::LLM::Recipes::LoRA (sibling of FromScratch) + the recipe-driven gate fixture' },
    { title: 'Build', detail: 'implement recipe + gate fixture; verify loss curve bit-identical + deterministic; commit; halt on failure' },
  ],
}

// ---- Constants -------------------------------------------------------------

const MIRRORGEN = 'prep/gen_cuda_mirror.rb'

// LoRA gate: 03_finetune_lora loads data/smollm2-135m-native.gguf (present,
// 540MB) via realize_for_mmap, seeds LoRA with upload_lora_q_init!(42,0.01),
// trains. The baseline agent picks a fast deterministic config (small STEPS).
const REF_BUILD = 'make examples/example_finetune'
const REF_RUN = './examples/example_finetune'  // env: GGUF (default data/smollm2-135m-native.gguf), RANK, STEPS, LR

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29g.md  (checkpoint; L4 FromScratch done, LoRA next)',
  'lib/toy/llm/recipes/from_scratch.rb          (THE PRECEDENT — sibling shape: minimal realize!/step!, hand-written class, fs_-prefixed members, no Struct.new, no Trainer/Stage)',
  'lib/toy/llm/recipes/README.md                (L4 contract sketch)',
  'examples/03_finetune_lora.rb                 (the inlined LoRA loop = gate reference; realize_for_mmap + upload_lora_q_init! + build_training_step + step loop; do NOT modify it)',
  'examples/smoke_recipe_from_scratch.rb        (the FromScratch gate-fixture pattern to mirror)',
  'lib/llama_seq_forward_ffi.rb                 (realize_for_mmap, upload_lora_q_init!, build_training_step)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_lib_vs_example_scope.md',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (#16 Struct.new — matz/spinel#1043)',
].join('\n  - ')

const FRAMING = `CRITICAL FRAMING.
(1) MIRROR THE FROMSCRATCH PRECEDENT. lib/toy/llm/recipes/from_scratch.rb already established the L4 shape: a minimal hand-written class with realize!/step!, uniquely-prefixed members (fs_*), NO Struct.new, NO speculative Trainer/Stage (AdamW is in the ggml graph). LoRA is its SIBLING: same step! loop (the train loop is identical — graph_reset/uploads/compute_backward/download-loss), DIFFERENT realize! (realize_for_mmap on a GGUF + upload_lora_q_init! + build_training_step, instead of realize_for_random_init). Use lora_*-prefixed members. Consider whether step! is literally identical to FromScratch's — if so, the plan may share it via a tiny common helper/module, but DO NOT over-abstract; a sibling class duplicating the ~8-line step! is acceptable and simpler. NO new Trainer/Stage/DataSpec/Eval. Keep experiment config (GGUF path, RANK, tokens, labels, hp) in the FIXTURE, not the recipe (lib-vs-example scope).
(2) GATE = BIT-IDENTICAL LOSS CURVE. Build a NEW fixture (examples/smoke_recipe_lora.rb) that drives the SAME LoRA config THROUGH Toy::LLM::Recipes::LoRA and prints "step N: loss=" lines byte-equal to the reference (the inlined 03_finetune_lora at the chosen fixed config). Use a FAST deterministic config (small fixed STEPS, fixed RANK, the present data/smollm2-135m-native.gguf, seeded upload_lora_q_init!). Do NOT modify 03_finetune_lora (frozen reference — rewiring it through the recipe is circular). If bit-identity can't be hit, report infeasible with the blocker; never loosen to a tolerance.
(3) SPINEL hygiene: no Struct.new (#16), positional/no-default ctor, lora_*-prefixed members. LoRA realize touches realize_for_mmap (a GGUF path) — but you are NOT refactoring realize here, just CALLING it from the recipe, so no realize-bulk coverage rules apply. If a Spinel-compiled lib changes, regen mirrors + verify-mirrors. CPU-only is fine (mirror deferred, like FromScratch).
(4) ONE focused commit.`

// ---- Schemas ---------------------------------------------------------------

const BASE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['loss_curve', 'deterministic', 'config', 'notes'],
  properties: {
    loss_curve: { type: 'string', description: 'the exact "step N: loss=" lines from the reference at the chosen fixed config (the bit-identical target)' },
    deterministic: { type: 'boolean', description: 'two runs byte-identical' },
    config: { type: 'string', description: 'the exact env/config used (GGUF, RANK, STEPS, LR, seed) — must be reused by the recipe fixture' },
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
    step_identical_to_fromscratch: { type: 'boolean', description: 'for the loop facet: is the per-step train loop byte-for-byte the same as FromScratch#step!?' },
    risks: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['feasible', 'recipe_api', 'shares_step_how', 'gate_fixture', 'files_touched', 'risks'],
  properties: {
    feasible: { type: 'boolean' },
    recipe_api: { type: 'string', description: 'the exact minimal Toy::LLM::Recipes::LoRA definition (realize! signature for the mmap+lora-init path, step!). Flag public names.' },
    shares_step_how: { type: 'string', description: 'how step! relates to FromScratch#step!: duplicate (simple), or shared via a tiny module/helper. Pick the simpler; justify.' },
    gate_fixture: { type: 'string', description: 'examples/smoke_recipe_lora.rb design: config, how it prints the loss curve, expected byte-equal to reference' },
    files_touched: { type: 'array', items: { type: 'string' } },
    mirror_changes: { type: 'string' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['recipe_built', 'loss_curve_bit_identical', 'deterministic', 'committed', 'notes'],
  properties: {
    recipe_built: { type: 'boolean', description: 'true ONLY if loss_curve_bit_identical && deterministic && committed' },
    loss_curve_bit_identical: { type: 'boolean' },
    deterministic: { type: 'boolean' },
    verify_mirrors_clean: { type: 'boolean', description: 'if a Spinel lib changed; N/A otherwise' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    api_committed: { type: 'string', description: 'exact public API shape committed (for review)' },
    infeasible_blocker: { type: 'string' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: Baseline -----------------------------------------------------

phase('Baseline')

const baseline = await agent(`Record a deterministic LoRA fine-tune LOSS CURVE — the bit-identical target for the LoRA recipe. ${FRAMING}

Build \`${REF_BUILD}\`. Run \`${REF_RUN}\` with a FAST deterministic config: the present model data/smollm2-135m-native.gguf, a small fixed STEPS (try STEPS=5), fixed RANK (default 8), default LR, and the seeded upload_lora_q_init!(42,0.01) the example already uses. Run TWICE; confirm the "step N: loss=" lines are byte-identical (deterministic). Record the loss_curve and the exact config string. If it's NOT deterministic (LoRA init RNG or token-order nondeterminism), report deterministic=false with what varies. Do NOT edit any file.`, { label: 'lora-loss-target', phase: 'Baseline', schema: BASE_SCHEMA })

const shaRaw = await agent(`Run \`git rev-parse HEAD\` then \`git status --porcelain --untracked-files=no\`. Report the 40-char sha. End with "CLEAN <sha>" if no tracked changes, else "DIRTY <sha>".`, { label: 'base-sha', phase: 'Baseline' })
const baseSha = ((shaRaw || '').match(/\b([0-9a-f]{40})\b/) || [])[1] || ''
const treeDirty = /\bDIRTY\b/.test(shaRaw || '')

log(`HEAD ${baseSha || '(unparsed)'} | tree ${treeDirty ? 'DIRTY' : 'clean'} | target det=${baseline?.deterministic}`)
if (!baseline || !baseline.deterministic) { log('ABORT: no deterministic LoRA loss-curve target.'); return { aborted: true, reason: 'no deterministic target', baseline } }
if (!baseSha || treeDirty) { log('ABORT: dirty tree / bad HEAD.'); return { aborted: true, reason: 'dirty tree', baseSha } }

// ---- Phase 1: Recon --------------------------------------------------------

phase('Recon')

const FACETS = [
  { key: 'lora-loop', prompt: `Map the LoRA training loop in examples/03_finetune_lora.rb: realize_for_mmap(gguf,cfg,...), upload_lora_q_init!(42,0.01), build_training_step, the per-step loop (graph_reset/uploads/compute_backward/download-loss). Compare the per-step loop to lib/toy/llm/recipes/from_scratch.rb#step! — set step_identical_to_fromscratch. Identify the realize! body the LoRA recipe needs (mmap + lora init).` },
  { key: 'fromscratch-precedent', prompt: `Study lib/toy/llm/recipes/from_scratch.rb closely: its class shape, realize!/step! signatures, member prefixes, no-Struct.new/no-Trainer rationale, the require/backend-coupling note. The LoRA recipe must be a consistent SIBLING. Decide if step! can be shared (module/mixin) or should be duplicated (simpler).` },
  { key: 'gate-fixture', prompt: `Design examples/smoke_recipe_lora.rb after examples/smoke_recipe_from_scratch.rb: drive the SAME LoRA config (the baseline's config) THROUGH Toy::LLM::Recipes::LoRA, print identical "step N: loss=" lines. What experiment setup (tokens, labels, hp, positions, GGUF load) must the fixture own (lib-vs-example scope)? Confirm 03_finetune_lora stays unmodified.` },
  { key: 'mirror-scope', prompt: `Will the LoRA recipe be CPU-only orchestration (no mirror) like FromScratch, or call TinyNN.* (needs mirror)? Check. Note Spinel hazards (#16) for any new identifier.` },
]

const reconArr = await parallel(FACETS.map((f) => () =>
  agent(`READ-ONLY recon for the L4 LoRA recipe. ${FRAMING}\n\nFacet: ${f.prompt}\n\nReport current file:line refs (grep). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: `recon:${f.key}`, phase: 'Recon', schema: RECON_SCHEMA })))
const recon = {}; FACETS.forEach((f, i) => { recon[f.key] = reconArr[i] })
for (const f of FACETS) { const r = recon[f.key]; log(`recon ${f.key}: ${r ? 'ok (' + (r.risks?.length || 0) + ' risks)' : 'FAILED'}`) }

// ---- Phase 2: Plan ---------------------------------------------------------

phase('Plan')

const plan = await agent(`Architect the L4 LoRA recipe — minimal, sibling of FromScratch. ${FRAMING}\n\n== Recon ==\n${JSON.stringify(recon, null, 2)}\n\nDecide recipe_api (exact minimal Toy::LLM::Recipes::LoRA), shares_step_how (duplicate vs tiny shared helper — pick simpler), gate_fixture design, files_touched. Set feasible=false if it can't be done minimally with a bit-identical gate. No Struct.new (#16). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: 'lora-plan', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan || !plan.feasible) {
  log(`STOP: LoRA recipe not safely extractable — ${plan ? (plan.risks || []).join('; ') : 'planner failed'}. No edits.`)
  return { aborted: true, reason: 'infeasible', baseSha, recon, plan }
}
log(`Plan ready. API: ${plan.recipe_api?.slice(0, 110)}. step: ${plan.shares_step_how?.slice(0, 60)}`)

// ---- Phase 3: Build --------------------------------------------------------

phase('Build')

const result = await agent(`Build the L4 LoRA recipe per the plan. ${FRAMING}\n\n== Plan ==\n${JSON.stringify(plan, null, 2)}\n\n== Loss-curve target (must reproduce byte-for-byte) ==\nconfig: ${baseline.config}\n${baseline.loss_curve}\n\n== Procedure ==\n1. Implement Toy::LLM::Recipes::LoRA (sibling of FromScratch) per recipe_api, reusing build_training_step / realize_for_mmap / upload_lora_q_init!.\n2. Build the recipe-driven gate fixture (examples/smoke_recipe_lora.rb); run it; its loss curve MUST byte-equal the target. Run TWICE for determinism. Confirm 03_finetune_lora unmodified.\n3. If bit-identity FAILS: do NOT loosen / alter behavior. Diagnose; if a real blocker, recipe_built=false + infeasible_blocker, revert (git checkout -- / rm new files; clean vs ${baseSha}).\n4. If a Spinel lib changed: ruby ${MIRRORGEN} && make verify-mirrors (exit 0). Else N/A.\n5. COMMIT (only if loss_curve_bit_identical && deterministic): git add ONLY new/changed source (NEVER -A, no binaries, no .gguf). Title "P2.6: L4 LoRA recipe". Capture sha + api_committed.\n\nrecipe_built=true ONLY if loss_curve_bit_identical && deterministic && committed. Reference docs:\n  - ${REFS}`, { label: 'build-lora', phase: 'Build', schema: BUILD_SCHEMA })

if (result && result.recipe_built && result.committed) {
  log(`OK: L4 LoRA recipe ${result.commit_sha || ''} | loss curve bit-identical | API: ${result.api_committed?.slice(0, 100)}`)
} else {
  log(`HALT: LoRA recipe not built — ${result ? (result.infeasible_blocker || result.notes) : 'agent returned nothing'}. Tree clean vs ${baseSha}.`)
}

return {
  baseSha,
  target_config: baseline.config,
  feasible: plan.feasible,
  result,
  followup: result && result.recipe_built
    ? 'LoRA recipe landed (gated). Remaining L4: WarmStart + Curriculum — both MULTI-STAGE, which forces the deferred Stage/each_stage taxonomy decision (a user API call). After L4: realize-bulk fixture-cascade, then P3.'
    : 'LoRA recipe not built — surface the blocker.',
}
