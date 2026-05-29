export const meta = {
  name: 'finish-l4-warmstart-curriculum',
  description: 'Finish L4 recipes: WarmStart (2-stage, extracted from 09_warm_start_train, gated bit-identical to its loss curve) and Curriculum (greenfield thin multi-stage template, gated by determinism + stage-1 == FromScratch). MINIMAL flat shape per user decision — NO Stage/Trainer/DataSpec/Eval taxonomy. Serial gated loop; halt on failure or infeasibility.',
  whenToUse: 'P2 L4 recipes, final pass: the two multi-stage recipes, minimal flat shape (user-confirmed).',
  phases: [
    { title: 'Baseline', detail: 'capture WarmStart (09) loss curve + FromScratch curve (Curriculum stage-1 ref) + HEAD; confirm deterministic' },
    { title: 'Recon', detail: 'parallel read-only: 09 structure, the precedent recipes, Curriculum design, gate fixtures' },
    { title: 'Plan', detail: 'design both minimal flat recipes + their gate fixtures; per-recipe feasible flags' },
    { title: 'Build', detail: 'serial gated loop: WarmStart then Curriculum; one commit each; halt on failure' },
  ],
}

// ---- Constants -------------------------------------------------------------

const MIRRORGEN = 'prep/gen_cuda_mirror.rb'

// WarmStart reference: 09_warm_start_train (random_init/projection-lens realize
// + INIT knob + streaming corpus + LR schedule). Fixed small config for a fast
// deterministic loss curve. FromScratch curve (4eaf019b loss lines) is the
// Curriculum stage-1 reference.
const WS_BUILD = 'make examples/example_warm_start_train'
const WS_RUN = 'SEED=0 STEPS=5 INIT=scratch ./examples/example_warm_start_train'  // baseline agent finalizes exact knobs
const FS_CURVE_REF = '4eaf019b… (SEED=0 STEPS=5 smoke_projection_lens loss/initial lines)'

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29g.md',
  'lib/toy/llm/recipes/from_scratch.rb + lib/toy/llm/recipes/lora.rb  (THE PRECEDENTS — minimal hand-written class, realize!/step!, no Struct.new, no Trainer/Stage, *_-prefixed members)',
  'examples/smoke_recipe_from_scratch.rb + examples/smoke_recipe_lora.rb  (gate-fixture pattern)',
  'examples/09_warm_start_train.rb  (WarmStart reference: INIT={scratch,warm}, toy_corpus_loader, toy_lr_schedule; gate reference — do NOT modify)',
  'lib/toy_corpus_loader.rb + lib/toy_lr_schedule.rb  (streaming data + LR schedule the FIXTURE drives, not the recipe)',
  'lib/toy/llm/recipes/README.md  (Curriculum = template showing the multi-stage shape)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_lib_vs_example_scope.md',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (#16 Struct.new)',
].join('\n  - ')

const FRAMING = `CRITICAL FRAMING.
(1) MINIMAL FLAT SHAPE — USER-CONFIRMED. Do NOT build the README's Stage/each_stage/Trainer/DataSpec/Eval taxonomy. Follow the FromScratch/LoRA precedent: a hand-written class with explicit methods + step!, uniquely-prefixed members (ws_* / cur_*), NO Struct.new (#16), no-default ctor (#4), no speculative abstractions. WarmStart gets explicit per-stage methods (e.g. realize_scratch! / realize_warm!) NOT an each_stage/Stage object. Experiment concerns — corpus loading, LR schedule, INIT choice, tokens/labels/hp — live in the FIXTURE (lib-vs-example scope), passed INTO step! (which may take a per-step batch + lr). The recipe stays thin.
(2) GATES.
 - WarmStart: a NEW fixture (examples/smoke_recipe_warm_start.rb) drives the SAME config as 09_warm_start_train (fixed SEED/STEPS/INIT/corpus) THROUGH Toy::LLM::Recipes::WarmStart and reproduces 09's loss curve BYTE-FOR-BYTE (compare the numeric loss values; 09 may print "CE=" while you print "loss=" — compare NUMBERS). Do NOT modify 09 (frozen reference). If you can't hit bit-identity, report infeasible — never loosen.
 - Curriculum: GREENFIELD — no inlined reference exists, so a bit-identical-to-reference gate is impossible. The honest gate is (a) DETERMINISM (run the recipe-driven fixture twice → byte-identical) AND (b) FAITHFULNESS: configure its FIRST stage identically to FromScratch and assert that stage's loss curve == ${FS_CURVE_REF} bit-for-bit (proves the template composes the real training path, not a toy mock). If Curriculum can't meet BOTH honestly as a thin template, set its feasible=false and DEFER it (report) rather than ship speculative illustrative code.
(3) SPINEL hygiene: no Struct.new (#16), prefixed members, no FFI :str at class load. CPU-only orchestration like the precedents (no mirror) is expected; if a Spinel lib changes, regen mirrors + verify-mirrors.
(4) ONE commit per recipe.`

// ---- Schemas ---------------------------------------------------------------

const BASE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['ws_loss_curve', 'ws_deterministic', 'ws_config', 'fs_curve', 'notes'],
  properties: {
    ws_loss_curve: { type: 'string', description: 'WarmStart reference: the numeric loss values from 09_warm_start_train at the chosen fixed config' },
    ws_deterministic: { type: 'boolean' },
    ws_config: { type: 'string', description: 'exact env/config for 09 (must be reused by the recipe fixture)' },
    fs_curve: { type: 'string', description: 'the FromScratch loss curve (smoke_projection_lens loss/initial lines) for the Curriculum stage-1 faithfulness check' },
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
    notes: { type: 'string' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['warmstart', 'curriculum'],
  properties: {
    warmstart: {
      type: 'object', additionalProperties: false,
      required: ['feasible', 'recipe_api', 'gate_fixture', 'files_touched', 'risks'],
      properties: {
        feasible: { type: 'boolean' },
        recipe_api: { type: 'string', description: 'minimal Toy::LLM::Recipes::WarmStart (realize_scratch!/realize_warm!, step!(batch,lr,...)); flag public names' },
        gate_fixture: { type: 'string', description: 'smoke_recipe_warm_start.rb design; reproduces 09 numeric loss curve' },
        files_touched: { type: 'array', items: { type: 'string' } },
        risks: { type: 'array', items: { type: 'string' } },
      },
    },
    curriculum: {
      type: 'object', additionalProperties: false,
      required: ['feasible', 'recipe_api', 'gate_design', 'files_touched', 'risks'],
      properties: {
        feasible: { type: 'boolean', description: 'true only if a thin template meets determinism + stage-1==FromScratch honestly; else false → defer' },
        recipe_api: { type: 'string' },
        gate_design: { type: 'string', description: 'how determinism + stage-1==FromScratch are checked' },
        files_touched: { type: 'array', items: { type: 'string' } },
        risks: { type: 'array', items: { type: 'string' } },
      },
    },
  },
}

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['recipe', 'built', 'gate_passed', 'committed', 'notes'],
  properties: {
    recipe: { type: 'string' },
    built: { type: 'boolean' },
    gate_passed: { type: 'boolean', description: 'WarmStart: bit-identical to 09. Curriculum: deterministic AND stage-1==FromScratch.' },
    bit_identical_or_faithful: { type: 'string', description: 'the exact comparison result' },
    deterministic: { type: 'boolean' },
    verify_mirrors_clean: { type: 'boolean', description: 'N/A if no Spinel lib changed' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    api_committed: { type: 'string' },
    failure_reason: { type: 'string' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: Baseline -----------------------------------------------------

phase('Baseline')

const baseline = await agent(`Record the gate targets for the two multi-stage recipes. ${FRAMING}

WarmStart: build \`${WS_BUILD}\`, run \`${WS_RUN}\` (pick a FAST deterministic config — small fixed STEPS, INIT=scratch, default corpus data/ts_seqs.bin, SEED=0) TWICE; confirm the numeric loss values are byte-identical; record ws_loss_curve + the exact ws_config. If non-deterministic, report ws_deterministic=false with what varies.
FromScratch curve (for Curriculum): run \`SEED=0 STEPS=5 ./examples/smoke_projection_lens | grep -E "^step|^initial="\` and record fs_curve (should be the 4eaf019b… lines).
Do NOT edit any file.`, { label: 'multistage-targets', phase: 'Baseline', schema: BASE_SCHEMA })

const shaRaw = await agent(`Run \`git rev-parse HEAD\` then \`git status --porcelain --untracked-files=no\`. Report the 40-char sha. End with "CLEAN <sha>" if no tracked changes, else "DIRTY <sha>".`, { label: 'base-sha', phase: 'Baseline' })
const baseSha = ((shaRaw || '').match(/\b([0-9a-f]{40})\b/) || [])[1] || ''
const treeDirty = /\bDIRTY\b/.test(shaRaw || '')

log(`HEAD ${baseSha || '(unparsed)'} | tree ${treeDirty ? 'DIRTY' : 'clean'} | WS det=${baseline?.ws_deterministic}`)
if (!baseline) { log('ABORT: baseline failed.'); return { aborted: true, reason: 'baseline failed' } }
if (!baseSha || treeDirty) { log('ABORT: dirty tree / bad HEAD.'); return { aborted: true, reason: 'dirty tree', baseSha } }
if (!baseline.ws_deterministic) { log('NOTE: WarmStart reference not deterministic — WarmStart will be deferred; Curriculum may still proceed.') }

// ---- Phase 1: Recon --------------------------------------------------------

phase('Recon')

const FACETS = [
  { key: 'warmstart-loop', prompt: `Map 09_warm_start_train.rb: the INIT={scratch,warm} branch (what each does at realize), the realize path, the streaming corpus loader (toy_corpus_loader) + LR schedule (toy_lr_schedule) usage, and the per-step loop. Identify what the WarmStart recipe should encapsulate (the realize + per-step train) vs what stays in the fixture (corpus iteration, LR schedule, INIT choice). Note 09's loss print format.` },
  { key: 'precedents', prompt: `Study lib/toy/llm/recipes/from_scratch.rb + lora.rb closely: class shape, realize!/step! signatures, member prefixes, the no-taxonomy rationale. WarmStart + Curriculum must be consistent SIBLINGS (ws_*/cur_* members). Determine how step! should look when it must accept a per-step batch + lr (WarmStart streams a corpus + schedules LR) — propose the minimal step! signature.` },
  { key: 'curriculum-design', prompt: `Curriculum is GREENFIELD (no example exists). Design the THINNEST honest multi-stage template: a recipe that runs N stages, each a realize+train with its own config (e.g. context/LR), reusing the FromScratch-style path. How would it be gated: (a) deterministic across runs, (b) its FIRST stage configured like FromScratch reproduces the 4eaf019b curve. If a thin faithful template isn't achievable without speculative machinery, say so (it should be DEFERRED).` },
  { key: 'gate-fixtures', prompt: `Design examples/smoke_recipe_warm_start.rb (drives WarmStart, reproduces 09's numeric loss curve at the baseline config) and the Curriculum gate fixture (determinism + stage-1==FromScratch). What does each fixture own (corpus loader, schedule, configs) per lib-vs-example scope? Confirm 09 + smoke_projection_lens stay unmodified.` },
]

const reconArr = await parallel(FACETS.map((f) => () =>
  agent(`READ-ONLY recon for the final L4 recipes (WarmStart + Curriculum), MINIMAL flat shape. ${FRAMING}\n\nFacet: ${f.prompt}\n\nReport current file:line refs (grep). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: `recon:${f.key}`, phase: 'Recon', schema: RECON_SCHEMA })))
const recon = {}; FACETS.forEach((f, i) => { recon[f.key] = reconArr[i] })
for (const f of FACETS) { const r = recon[f.key]; log(`recon ${f.key}: ${r ? 'ok' : 'FAILED'}`) }

// ---- Phase 2: Plan ---------------------------------------------------------

phase('Plan')

const plan = await agent(`Architect the two final L4 recipes, MINIMAL flat shape (user-confirmed: NO Stage/Trainer taxonomy). ${FRAMING}\n\n== Recon ==\n${JSON.stringify(recon, null, 2)}\n\n== WarmStart target config ==\n${baseline.ws_config} (deterministic=${baseline.ws_deterministic})\n\nDesign warmstart{recipe_api, gate_fixture} and curriculum{recipe_api, gate_design}. Set each feasible independently: WarmStart.feasible=false if ws not deterministic or bit-identity unreachable; Curriculum.feasible=false if a thin honest template isn't achievable (then it's deferred — acceptable). No Struct.new (#16). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: 'multistage-plan', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan) { log('ABORT: planner failed.'); return { aborted: true, reason: 'planner failed', baseSha, recon } }
log(`Plan: WarmStart feasible=${plan.warmstart?.feasible}, Curriculum feasible=${plan.curriculum?.feasible}`)

// ---- Phase 3: Serial gated build loop --------------------------------------

phase('Build')

const UNITS = [
  { key: 'WarmStart', spec: plan.warmstart, gate: `Reproduce 09_warm_start_train's numeric loss curve BYTE-FOR-BYTE at config: ${baseline.ws_config}\nReference loss values:\n${baseline.ws_loss_curve}\nCompare NUMBERS (09 prints CE=, you may print loss=). Run the recipe fixture TWICE for determinism. Do NOT modify 09.` },
  { key: 'Curriculum', spec: plan.curriculum, gate: `GREENFIELD — no bit-identical reference. Gate = (a) run the recipe fixture TWICE → byte-identical (deterministic) AND (b) configure stage 1 like FromScratch and assert its loss curve == this FromScratch reference byte-for-byte:\n${baseline.fs_curve}\nIf both can't be met honestly as a THIN template, do NOT ship — report gate_passed=false (defer).` },
]

const results = []
for (let i = 0; i < UNITS.length; i++) {
  const u = UNITS[i]
  if (!u.spec || !u.spec.feasible) {
    log(`SKIP ${u.key}: plan marked infeasible — deferred. ${u.spec ? (u.spec.risks || []).join('; ') : ''}`)
    results.push({ recipe: u.key, built: false, committed: false, deferred: true, notes: u.spec ? 'plan infeasible' : 'no plan' })
    continue  // independent recipes — defer one, still try the other
  }
  const r = await agent(`Build the L4 ${u.key} recipe per the plan. MINIMAL flat shape. ${FRAMING}\n\n== Plan (${u.key}) ==\n${JSON.stringify(u.spec, null, 2)}\n\n== Gate ==\n${u.gate}\n\n== Procedure ==\n1. Implement Toy::LLM::Recipes::${u.key} (minimal, sibling of FromScratch/LoRA), reusing existing realize_* / build_training_step / corpus / schedule machinery; experiment concerns in the fixture.\n2. Build the recipe-driven gate fixture; run it; meet the gate exactly (bit-identical numbers for WarmStart; determinism + stage-1==FromScratch for Curriculum). Confirm 09 + smoke_projection_lens unmodified + still green.\n3. If the gate FAILS: do NOT loosen / alter behavior. Diagnose; if a real blocker, built=false + failure_reason, revert (git checkout -- / rm new files; clean vs the unit's starting HEAD).\n4. If a Spinel lib changed: ruby ${MIRRORGEN} && make verify-mirrors. Else N/A.\n5. COMMIT (only if gate_passed): git add ONLY new/changed source (NEVER -A, no binaries/gguf). Title "P2.6: L4 ${u.key} recipe". Capture sha + api_committed.\n\ngate_passed=true ONLY if the gate is met && committed. Reference docs:\n  - ${REFS}`, { label: `build:${u.key}`, phase: 'Build', schema: BUILD_SCHEMA })
  results.push(r)
  if (r && r.gate_passed && r.committed) {
    log(`OK ${u.key}: ${r.commit_sha || ''} | ${r.bit_identical_or_faithful?.slice(0, 60)}`)
  } else {
    log(`HALT ${u.key}: ${r ? (r.failure_reason || r.notes) : 'no result'}. Reverted. (Other recipes independent — continuing to report.)`)
    // recipes are independent; do not break — let the loop try/record the next, but a failed build already reverted.
  }
}

const committed = results.filter((r) => r && r.gate_passed && r.committed)
return {
  baseSha,
  ws_config: baseline.ws_config,
  completed: committed.map((r) => ({ recipe: r.recipe, sha: r.commit_sha, api: r.api_committed })),
  deferred: results.filter((r) => r && (r.deferred || !r.gate_passed)).map((r) => ({ recipe: r.recipe, why: r.failure_reason || r.notes })),
  results,
  followup: 'L4 recipes: FromScratch + LoRA done previously; this pass = WarmStart + Curriculum (minimal flat, user-confirmed). After L4: realize-bulk fixture-cascade (optional), then P3 core+CLI.',
}
