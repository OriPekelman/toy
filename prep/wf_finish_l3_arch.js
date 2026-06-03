export const meta = {
  name: 'finish-l3-llama-arch',
  description: 'Lift the arch-level forward orchestration (build_forward_in_current_ctx: embed → block stack → final RMSNorm → LM head) + arch-owned tensors out of the monolith into the L3 Toy::LLM::Archs::LlamaArch, behind a bit-identical-logits gate, as ONE commit; minimal orchestration lift only (NOT the realize-path bulk); reset and halt on gate failure',
  whenToUse: 'Continuing the toy-framework P2 refactor after L1+L2: extract the L3 LlamaArch as a single gated unit',
  phases: [
    { title: 'Baseline', detail: 'capture HEAD + the deterministic gate fingerprint' },
    { title: 'Recon', detail: 'parallel read-only agents map the arch forward, arch-owned weights, the realize boundary, mirror gen' },
    { title: 'Plan', detail: 'one agent synthesizes recon into an exact, ordered extraction plan + the arch API' },
    { title: 'Extract', detail: 'one gated agent executes the plan; one commit; reset+halt on gate failure' },
  ],
}

// ---- Constants -------------------------------------------------------------

const MONOLITH = 'lib/llama_seq_forward_ffi.rb'
const MIRRORGEN = 'prep/gen_cuda_mirror.rb'
const ARCH_FILE = 'lib/toy/llm/archs/llama_arch.rb'

const BUILD_CMD = 'make examples/smoke_projection_lens'
const FIXTURE_CMD = 'SEED=0 STEPS=5 ./examples/smoke_projection_lens'
const FINGERPRINT_CMD = `${FIXTURE_CMD} 2>/dev/null | grep -E "^step|^initial="`

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29c.md  (authoritative checkpoint; L1+L2 done, L3 is next)',
  'lib/toy/llm/archs/README.md                 (L3 contract sketch — see CAVEAT below)',
  'lib/toy/llm/blocks/transformer_block.rb     (the L2 block this arch stacks; note its Ctx-class pattern)',
  'lib/toy/llm/primitives/rms_norm.rb          (final norm uses this primitive already)',
  'docs/roadmap/toy-framework-design-2026-05-28.md',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (incl. #16 Struct.new — matz/spinel#1043)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/project_step_bind_landmine_2026_05_28.md',
].join('\n  - ')

// Two framings every agent must hold: sketch-vs-reality, and SCOPE.
const SKETCH_CAVEAT = `CRITICAL FRAMING 1 — sketch vs reality. The archs/README contract is FORWARD-LOOKING. It shows build_forward(sess, ids, positions, cfg), per_layer overrides, with_hyper, learned position embeddings, build_initial_state. The seq-forward monolith has NONE of that yet: the orchestration lives in Toy::LLM::Engine::LlamaSeqEngine#build_forward_in_current_ctx (lines ~1532-1582), there are no per-layer overrides (every block shares one read-only ctx), positions/RoPE are on the block, and there is no learned position embedding. DO NOT implement per_layer / with_hyper / build_initial_state to match the sketch — that is later work. Do the MINIMAL faithful lift of the existing orchestration, preserving behavior BIT-IDENTICALLY. Flag the divergence in the file header.

CRITICAL FRAMING 2 — SCOPE. This is P2.5, the orchestration lift ONLY. The four realize paths (realize_for_q8_copy/_mmap/_full_finetune/_random_init) are the ~1750-line allocation bulk; the roadmap EXPLICITLY defers their restructuring to P2.6/P2.7. So: the arch OWNS the arch-level tensor handles (token_embed, final_norm_gamma, output, w_proj, the blocks array), and the realize paths still ALLOCATE and ASSIGN them — exactly the pattern L2 used (the block owns weights; the cache's realize paths allocate+assign them). Move the ORCHESTRATION method and the OWNERSHIP, not the allocation logic. Touch the realize paths only minimally (retarget assignments onto the arch object, like L2 did). If a faithful lift can't avoid restructuring the realize bulk, set feasible=false and explain — do not expand scope.`

// ---- Schemas ---------------------------------------------------------------

const GATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['deterministic', 'baseline_value', 'notes'],
  properties: {
    deterministic: { type: 'boolean', description: 'true only if two consecutive runs produced byte-identical fingerprints' },
    baseline_value: { type: 'string', description: 'sha256 of the fingerprint output at current HEAD' },
    notes: { type: 'string' },
  },
}

const RECON_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['facet', 'findings', 'line_refs', 'risks'],
  properties: {
    facet: { type: 'string' },
    findings: { type: 'string', description: 'detailed prose: what this facet contains and how it connects' },
    line_refs: { type: 'array', items: { type: 'string' }, description: 'current file:line refs (grep — do not trust doc numbers)' },
    ivars_read: { type: 'array', items: { type: 'string' }, description: 'cache ivars the orchestration reads (become arch-owned or passed-in context)' },
    arch_owned: { type: 'array', items: { type: 'string' }, description: 'tensor handles / fields that should move to the arch object' },
    cache_side_effects: { type: 'array', items: { type: 'string' }, description: 'ivars the orchestration WRITES that are read elsewhere (taps, training, decode) — these constrain the arch API (what it must expose back)' },
    risks: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['feasible', 'arch_api', 'what_moves', 'what_stays', 'realize_path_changes', 'mirror_changes', 'ordered_steps', 'parity_risks'],
  properties: {
    feasible: { type: 'boolean', description: 'true if a bit-identical single-commit orchestration lift looks achievable WITHOUT restructuring the realize bulk; false halts before any edit' },
    arch_api: { type: 'string', description: 'the exact Toy::LLM::Archs::LlamaArch definition: attr_accessors for owned tensors + blocks array + the build_forward signature, adapted to seq reality (no per_layer/with_hyper). Specify how the cache-written ivars (x_embed/x_final/logits/token_ids/positions) are exposed back so taps/training still see them. Reuse the L2 Ctx-CLASS pattern if any value object is needed — NEVER Struct.new (landmine #16 / matz/spinel#1043).' },
    what_moves: { type: 'array', items: { type: 'string' } },
    what_stays: { type: 'array', items: { type: 'string' }, description: 'esp. the realize-path allocation logic, which STAYS' },
    realize_path_changes: { type: 'array', items: { type: 'string' }, description: 'the MINIMAL retargeting of each of the 4 realize paths (assign arch-owned handles onto the arch object), nothing more' },
    mirror_changes: { type: 'string', description: 'gen_cuda_mirror.rb: archs/*.rb regex + MIRRORABLE entry + subs_for case (watch class-name renames; LlamaArch module shouldn\'t collide, verify)' },
    ordered_steps: { type: 'array', items: { type: 'string' } },
    parity_risks: { type: 'array', items: { type: 'string' } },
  },
}

const EXTRACT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gate_passed', 'bit_identical', 'verify_mirrors_clean', 'committed', 'notes'],
  properties: {
    gate_passed: { type: 'boolean', description: 'true ONLY if bit_identical && verify_mirrors_clean && committed' },
    bit_identical: { type: 'boolean' },
    verify_mirrors_clean: { type: 'boolean' },
    cuda_checked: { type: 'boolean' },
    cuda_result: { type: 'string', description: 'CUDA mirror smoke outcome, or why it could not run (never silently skip)' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    logit_diff_summary: { type: 'string' },
    scope_held: { type: 'boolean', description: 'true if the realize-path bulk was NOT restructured (only minimal retargeting)' },
    failure_reason: { type: 'string' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: Baseline -----------------------------------------------------

phase('Baseline')

const baseline = await agent(`Set up the parity gate for the L3 LlamaArch extraction. Repo root on host gx10. Gate fixture is proven deterministic + parity-sensitive (used for L1 + L2):

  Build:        ${BUILD_CMD}
  Fingerprint:  ${FINGERPRINT_CMD}  | sha256sum

Build, then run the fingerprint pipe TWICE, confirm the two sha256 values are byte-identical. Record the sha256 as baseline_value. Do NOT edit any file.`, { label: 'gate-fixture', phase: 'Baseline', schema: GATE_SCHEMA })

const shaRaw = await agent(`Run \`git rev-parse HEAD\`, then \`git status --porcelain --untracked-files=no\`. Report the 40-char HEAD sha. End your message with "CLEAN <sha>" if there are no tracked changes (untracked build artifacts are fine), else "DIRTY <sha>".`, { label: 'base-sha', phase: 'Baseline' })

const shaMatch = (shaRaw || '').match(/\b([0-9a-f]{40})\b/)
const baseSha = shaMatch ? shaMatch[1] : ''
const treeDirty = /\bDIRTY\b/.test(shaRaw || '')

log(`Baseline HEAD: ${baseSha || '(unparsed)'} | tracked-tree ${treeDirty ? 'DIRTY' : 'clean'}`)
log(`Gate baseline: ${baseline?.baseline_value || '(none)'} | deterministic=${baseline?.deterministic}`)

if (!baseline || !baseline.deterministic) {
  log('ABORT: gate fixture not deterministic — cannot guarantee bit-identical parity.')
  return { aborted: true, reason: 'no deterministic gate fixture', baseline }
}
if (!baseSha || treeDirty) {
  log('ABORT: dirty tracked tree or unreadable HEAD. Commit/stash first.')
  return { aborted: true, reason: 'dirty tree or bad HEAD', baseSha, shaRaw }
}

// ---- Phase 1: Recon (parallel, read-only) ----------------------------------

phase('Recon')

const FACETS = [
  {
    key: 'arch-forward',
    prompt: `Map the ORCHESTRATION to be lifted: Toy::LLM::Engine::LlamaSeqEngine#build_forward_in_current_ctx (around lines 1532-1582) in ${MONOLITH}. Detail every step: token_ids/positions input build, the per-block TransformerBlockCtx construction, the get_rows embed, the optional projection-lens branch (donor_d_in), the block-stacking while-loop calling blk.build_forward, the final RMSNorm.build, the tied/untied LM-head matmul, and every tnn_set_output tap. List in ivars_read the config/handle ivars it reads, and in cache_side_effects every @t_seq_* ivar it WRITES (x_embed, x_final, logits, token_ids, positions, ...) — find where each of those is read elsewhere (grep), because the arch API must keep them visible.`,
  },
  {
    key: 'arch-weights',
    prompt: `Map the ARCH-OWNED tensors: @t_seq_token_embed, @t_seq_final_norm_gamma, @t_seq_output, @t_seq_w_proj (and any other whole-network, non-block handle). For each: where declared (the attr_reader list ~line 31 + nil-init ~107), and where ASSIGNED in the 4 realize paths (q8_copy ~219, mmap ~519, full_finetune ~824, random_init). Populate arch_owned. Also: is there ANY per-layer override or learned position embedding in the seq path? (Expected: no — confirm by grep so the plan can safely omit per_layer/with_hyper.)`,
  },
  {
    key: 'realize-boundary',
    prompt: `Map the REALIZE BOUNDARY — the scope line for P2.5. The 4 realize paths are the ~1750-line allocation bulk whose RESTRUCTURING is deferred to P2.6/P2.7. Determine the MINIMAL change each realize path needs so that the arch object (not the cache) owns the arch-level handles, WITHOUT restructuring the allocation logic — i.e. the L2 pattern (cache allocates, then assigns onto the owning object). Identify exactly which assignment lines retarget. Flag anything that would FORCE deeper realize surgery (that would make the lift infeasible at this scope).`,
  },
  {
    key: 'mirror-gen',
    prompt: `Map the MIRROR GENERATOR changes for a new ${ARCH_FILE}. Read ${MIRRORGEN}: does derive_mirror_path's regex already accept lib/toy/llm/archs/*.rb, or only primitives/ + blocks/? What MIRRORABLE entry + subs_for case is needed? Verify the new module name Toy::LLM::Archs::LlamaArch does NOT collide with any TinyNN/class rename the generator performs; report whether the arch file needs only common_module_tail or also class-name renames + the require-rewrite in the monolith mirror.`,
  },
]

const reconArr = await parallel(FACETS.map((f) => () =>
  agent(`READ-ONLY recon for the L3 LlamaArch extraction (toy-framework P2.5). ${SKETCH_CAVEAT}

Your facet: ${f.prompt}

Report current file:line refs (grep — do not trust doc line numbers). Do NOT edit any file.

Reference docs:
  - ${REFS}`, { label: `recon:${f.key}`, phase: 'Recon', schema: RECON_SCHEMA })))

const recon = {}
FACETS.forEach((f, i) => { recon[f.key] = reconArr[i] })
for (const f of FACETS) {
  const r = recon[f.key]
  log(`recon ${f.key}: ${r ? 'ok (' + (r.line_refs?.length || 0) + ' refs, ' + (r.risks?.length || 0) + ' risks)' : 'FAILED'}`)
}

// ---- Phase 2: Plan ---------------------------------------------------------

phase('Plan')

const plan = await agent(`You are the architect for the L3 LlamaArch extraction (toy-framework P2.5). Synthesize the four recon reports into ONE precise, ordered plan to lift the forward orchestration into ${ARCH_FILE} as a single bit-identical, single-commit change. ${SKETCH_CAVEAT}

== Recon ==
${JSON.stringify(recon, null, 2)}

Decide and specify exactly:
- The arch API (Toy::LLM::Archs::LlamaArch): attr_accessors for owned tensors + the blocks array, and the build_forward signature adapted to seq reality (NO per_layer/with_hyper/build_initial_state). CRUCIAL: the orchestration currently WRITES several @t_seq_* ivars that other code reads (taps, training, decode). Specify how build_forward exposes those back (return struct? arch attr_accessors the cache reads? thin cache delegators?) so nothing downstream breaks. If you need a value object, use a hand-written CLASS with positional initialize and uniquely-prefixed members — NEVER Struct.new (landmine #16 / matz/spinel#1043; the L2 TransformerBlockCtx is the model).
- what_moves vs what_stays (the realize-path ALLOCATION logic STAYS).
- The MINIMAL retargeting of each of the 4 realize paths.
- The mirror-generator changes.
- ordered_steps for the executor.
- parity_risks (set_output tap order, tied/untied head, projection-lens branch, B>1 path, Spinel name-collision hazards on any new identifiers).

Set feasible=false if a bit-identical single-commit orchestration lift is not achievable WITHOUT restructuring the realize bulk (explain). Do NOT edit any file.

Reference docs:
  - ${REFS}`, { label: 'l3-plan', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan || !plan.feasible) {
  log(`ABORT: plan infeasible — ${plan ? plan.parity_risks?.join('; ') : 'planner failed'}. No edits made.`)
  return { aborted: true, reason: 'plan infeasible', baseSha, recon, plan }
}
log(`Plan ready: ${plan.ordered_steps?.length || 0} steps. Proceeding to gated extraction.`)

// ---- Phase 3: Single gated extraction (one commit) -------------------------

phase('Extract')

const result = await agent(`Execute the L3 LlamaArch extraction (toy-framework P2.5). STRICTLY GATED, SINGLE COMMIT. The ONLY acceptable outcome is byte-identical forward output; "close" is failure. Do it as ONE unit, one commit. ${SKETCH_CAVEAT}

== Gate fixture (proven deterministic + parity-sensitive) ==
Build:        ${BUILD_CMD}
Fingerprint:  ${FINGERPRINT_CMD}  | sha256sum
Baseline:     ${baseline.baseline_value}

== The plan to execute ==
${JSON.stringify(plan, null, 2)}

== Procedure ==
1. CAPTURE BASELINE: ${BUILD_CMD} && fingerprint | sha256sum, confirm == ${baseline.baseline_value}. If it doesn't match pre-edit, STOP (gate_passed=false, failure_reason="baseline mismatch pre-edit").
2. WRITE ${ARCH_FILE} per plan.arch_api. Match the L1/L2 module style (no require_relative "tinynn"; the monolith's backend require governs which TinyNN is in scope). Spinel hygiene: NO Struct.new (landmine #16 / matz/spinel#1043) — hand-written class, positional ctor, no default args, uniquely-prefixed members; no nil->RbVal ivar widening; no Array destructure widening; no FFI :str into step_bind.
3. LIFT the orchestration (build_forward_in_current_ctx body) into the arch + retarget the 4 realize paths MINIMALLY so the arch owns its handles. DO NOT restructure realize allocation (scope = P2.5). Keep the cache's call site working (it should construct/hold the arch and call arch.build_forward; expose the written ivars back per the plan). After this, the old orchestration body should be gone from the cache (or a thin delegator only if the plan says so).
4. MIRRORS: apply plan.mirror_changes to ${MIRRORGEN}, then \`ruby ${MIRRORGEN}\`, then \`make verify-mirrors\` — MUST exit 0.
5. CPU PARITY GATE: ${BUILD_CMD} && fingerprint | sha256sum. MUST byte-equal ${baseline.baseline_value}. If not, HARD FAIL.
6. CUDA SMOKE (best effort, never silent): attempt the CUDA mirror path; if the full CUDA toolchain isn't available (checked-in libtinynn_ggml_cuda.a may be the AB-smoke stub; full needs \`make setup-ggml-cuda\`), say so explicitly. verify-mirrors already guarantees mirror-source consistency.
7. COMMIT (only if bit_identical && verify_mirrors_clean): stage ONLY \`git add ${ARCH_FILE} ${MONOLITH} ${MIRRORGEN}\` + regenerated *_cuda.rb/*_metal.rb mirrors (incl. the new arch mirrors). NEVER \`git add -A\`. Commit title "P2.5: extract L3 LlamaArch". Capture the sha. Set scope_held=true only if you did NOT restructure the realize bulk.

== On ANY gate failure ==
Do NOT patch around it. Revert cleanly: \`git checkout -- ${MONOLITH} ${MIRRORGEN}\` and the mirror files, \`rm -f ${ARCH_FILE}\` and any new mirror files, confirm \`git status\` is clean vs ${baseSha}. Report gate_passed=false + precise failure_reason. A clean halt is the correct outcome.

gate_passed=true ONLY if bit_identical && verify_mirrors_clean && committed.

Reference docs:
  - ${REFS}`, { label: 'extract:llama_arch', phase: 'Extract', schema: EXTRACT_SCHEMA })

if (result && result.gate_passed && result.committed) {
  log(`OK: L3 LlamaArch committed ${result.commit_sha || '(sha?)'} | bit-identical | verify-mirrors clean | scope_held=${result.scope_held} | cuda=${result.cuda_checked ? result.cuda_result : 'not run'}`)
} else {
  log(`HALT: L3 extraction did not pass the gate — ${result ? (result.failure_reason || result.notes) : 'agent returned nothing'}. Tree should be clean vs ${baseSha}.`)
}

return {
  baseSha,
  gate_baseline: baseline.baseline_value,
  feasible: plan.feasible,
  result,
  followup: 'Pushed nothing (land on main, push when ready). If gate passed: P2 five-layer refactor reaches L3; remaining is the realize-path bulk (P2.6/P2.7) + L4 recipes. If halted: tree is clean — review failure_reason and re-plan the arch boundary.',
}
