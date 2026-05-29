export const meta = {
  name: 'finish-l2-transformer-block',
  description: 'Lift the seq-forward block body (build_seq_block + build_seq_qhead) out of the monolith into the L2 Toy::LLM::Blocks::TransformerBlock, composing the four L1 primitives, behind a bit-identical-logits gate, as ONE commit; reset and halt on gate failure',
  whenToUse: 'Continuing the toy-framework P2 refactor after L1: extract the L2 TransformerBlock as a single gated unit',
  phases: [
    { title: 'Baseline', detail: 'capture HEAD + the deterministic gate fingerprint' },
    { title: 'Recon', detail: 'parallel read-only agents map the block body, weight ownership, cache context, mirror gen' },
    { title: 'Plan', detail: 'one agent synthesizes recon into an exact, ordered extraction plan + the block API' },
    { title: 'Extract', detail: 'one gated agent executes the plan; one commit; reset+halt on gate failure' },
  ],
}

// ---- Constants -------------------------------------------------------------

const MONOLITH = 'lib/llama_seq_forward_ffi.rb'
const MIRRORGEN = 'prep/gen_cuda_mirror.rb'
const BLOCK_FILE = 'lib/toy/llm/blocks/transformer_block.rb'

// Known-good, proven-deterministic gate fixture (established in the L1 run
// and re-verified bit-identical pre/post-refactor).
const BUILD_CMD = 'make examples/smoke_projection_lens'
const FIXTURE_CMD = 'SEED=0 STEPS=5 ./examples/smoke_projection_lens'
const FINGERPRINT_CMD = `${FIXTURE_CMD} 2>/dev/null | grep -E "^step|^initial="`

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29b.md  (authoritative checkpoint; L1 done, L2 is next)',
  'lib/toy/llm/blocks/README.md                (L2 contract sketch — see CAVEAT below)',
  'lib/toy/llm/primitives/README.md            (L1 contract the block composes)',
  'lib/toy/llm/primitives/{rope,swiglu,rms_norm,gqa}.rb  (the four primitives, already landed)',
  'docs/roadmap/toy-framework-design-2026-05-28.md',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/project_step_bind_landmine_2026_05_28.md',
].join('\n  - ')

// The single most important framing for every agent: seq-mode reality vs the
// forward-looking README sketch.
const SKETCH_CAVEAT = `CRITICAL FRAMING — the README block contract is a FORWARD-LOOKING SKETCH, not the seq-mode reality. The sketch shows build_forward(sess, x, state, cfg) returning (out, state_out) with a KV-cache "state" threaded through each block. The seq-forward monolith (${MONOLITH}) does FULL-SEQUENCE forward with NO incremental per-block KV state threading — KV-cache decode lives in a SEPARATE file (lib/toy_smollm2_ffi_kv.rb), OUT OF SCOPE here. DO NOT invent a state-threading API to match the sketch. The L2 job is the minimal faithful lift: move the existing forward body verbatim into a block class, with the block owning the weight handles it already owns (LlamaSeqBlockFFI), preserving behavior BIT-IDENTICALLY. Adapt the sketch to seq reality; flag the divergence rather than forcing the sketch.`

// ---- Schemas ---------------------------------------------------------------

const GATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['deterministic', 'baseline_value', 'notes'],
  properties: {
    deterministic: { type: 'boolean', description: 'true only if two consecutive runs of the fixture produced byte-identical fingerprints' },
    baseline_value: { type: 'string', description: 'sha256 of the fingerprint output at current HEAD' },
    notes: { type: 'string' },
  },
}

const RECON_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['facet', 'findings', 'line_refs', 'risks'],
  properties: {
    facet: { type: 'string' },
    findings: { type: 'string', description: 'detailed prose: exactly what this facet contains and how it connects to the rest' },
    line_refs: { type: 'array', items: { type: 'string' }, description: 'current file:line references that matter for the extraction' },
    ivars_read: { type: 'array', items: { type: 'string' }, description: 'cache ivars the block body reads (for the body facet) — i.e. the per-forward context the block needs handed in' },
    block_owned: { type: 'array', items: { type: 'string' }, description: 'tensor handles / fields that belong to the block instance' },
    risks: { type: 'array', items: { type: 'string' }, description: 'parity or Spinel risks specific to this facet' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['feasible', 'block_api', 'what_moves', 'what_stays', 'realize_path_changes', 'mirror_changes', 'ordered_steps', 'parity_risks'],
  properties: {
    feasible: { type: 'boolean', description: 'true if a bit-identical single-commit lift looks achievable; false halts before any edit' },
    block_api: { type: 'string', description: 'the exact Toy::LLM::Blocks::TransformerBlock definition: attr_accessors for owned weights + the build_forward method signature, adapted to seq-mode reality (NO invented KV state)' },
    what_moves: { type: 'array', items: { type: 'string' }, description: 'what moves from the monolith into the block (method bodies, fields)' },
    what_stays: { type: 'array', items: { type: 'string' }, description: 'what stays on the cache/arch (per-forward context, block-stacking loop, realize allocation)' },
    realize_path_changes: { type: 'array', items: { type: 'string' }, description: 'edits to each of the 4 realize paths (q8_copy/mmap/full_finetune/random_init) to hand the block its weights + context' },
    mirror_changes: { type: 'string', description: 'gen_cuda_mirror.rb changes: blocks/*.rb regex + MIRRORABLE entry + subs_for case (watch class-name renames; TransformerBlock module shouldn\'t collide, but verify)' },
    ordered_steps: { type: 'array', items: { type: 'string' }, description: 'the exact ordered steps the extraction agent should execute' },
    parity_risks: { type: 'array', items: { type: 'string' } },
  },
}

const EXTRACT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gate_passed', 'bit_identical', 'verify_mirrors_clean', 'committed', 'notes'],
  properties: {
    gate_passed: { type: 'boolean', description: 'true ONLY if bit_identical && verify_mirrors_clean && committed' },
    bit_identical: { type: 'boolean', description: 'fixture fingerprint byte-identical to baseline after the lift' },
    verify_mirrors_clean: { type: 'boolean', description: '`make verify-mirrors` exited 0' },
    cuda_checked: { type: 'boolean' },
    cuda_result: { type: 'string', description: 'CUDA mirror smoke outcome, or why it could not run (never silently skip)' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    logit_diff_summary: { type: 'string', description: 'baseline fingerprint vs post fingerprint' },
    failure_reason: { type: 'string', description: 'if gate_passed=false: exactly what failed and what was reverted' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: Baseline -----------------------------------------------------

phase('Baseline')

const baseline = await agent(`Set up the parity gate for the L2 TransformerBlock extraction. Repo root on host gx10. The gate fixture is already known and proven deterministic + parity-sensitive from the L1 work:

  Build:        ${BUILD_CMD}
  Fingerprint:  ${FINGERPRINT_CMD}  | sha256sum

Steps: build, then run the fingerprint pipe TWICE, confirm the two sha256 values are byte-identical (deterministic). Record the sha256 as baseline_value. Do NOT edit any file.`, { label: 'gate-fixture', phase: 'Baseline', schema: GATE_SCHEMA })

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
    key: 'forward-body',
    prompt: `Map the FORWARD BODY to be lifted: the methods build_seq_block (around line 1776) and build_seq_qhead (around line 1860) in ${MONOLITH}. For each: every cache ivar it reads (these become the per-forward CONTEXT the block must be handed — e.g. @sess, @seq_t, @seq_b, @t_seq_positions, @seq_rope_cfg, scale, eps), every call into the four L1 primitives (RoPE.apply_2d, SwiGLU.gate, RMSNorm.build, GQA.attention) with exact args, the control flow (B>1 vs B=1 branches), any tnn_set_output taps, and the residual adds. Populate ivars_read with the full context list.`,
  },
  {
    key: 'weights-ownership',
    prompt: `Map WEIGHT OWNERSHIP: the LlamaSeqBlockFFI class (around line 30) — every field/tensor handle it holds (rn1/rn2 gamma, q/k/v/o weights, gate/up/down, biases, LoRA, head splits). Then how each of the 4 realize paths (realize_for_q8_copy ~236, realize_for_mmap ~533, realize_for_full_finetune ~838, realize_for_random_init ~1081) allocates and assigns those fields. Populate block_owned with the fields that should live on the new block.`,
  },
  {
    key: 'cache-context',
    prompt: `Map the CACHE/ARCH context that must STAY on LlamaSeqForwardFFICache: the block-stacking loop (where build_seq_block is called per layer — grep build_seq_block call sites), the final norm + LM head, token embedding, and the per-forward scalars/tensors (positions, rope_cfg, scale, eps, seq dims) that the block body reads but does NOT own. Distinguish "stays on cache, handed to block per forward" from "moves into block".`,
  },
  {
    key: 'mirror-gen',
    prompt: `Map the MIRROR GENERATOR changes for a new ${BLOCK_FILE}. Read ${MIRRORGEN}: does derive_mirror_path's regex already accept lib/toy/llm/blocks/*.rb (it was extended for primitives/)? What MIRRORABLE entry + subs_for case is needed? CRITICAL: the existing monolith mirror renames the block class LlamaSeqBlockFFI -> LlamaSeqBlockFFICuda/Metal. The new Toy::LLM::Blocks::TransformerBlock module name must NOT collide with TinyNN renames — verify, and report whether the block file needs only common_module_tail or also a class-name rename.`,
  },
]

const reconArr = await parallel(FACETS.map((f) => () =>
  agent(`READ-ONLY recon for the L2 TransformerBlock extraction (toy-framework P2.4). ${SKETCH_CAVEAT}

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

const plan = await agent(`You are the architect for the L2 TransformerBlock extraction (toy-framework P2.4). Synthesize the four recon reports below into ONE precise, ordered extraction plan that a single executor agent will follow to lift the forward body into ${BLOCK_FILE} as a single bit-identical, single-commit change.

${SKETCH_CAVEAT}

== Recon ==
${JSON.stringify(recon, null, 2)}

Decide and specify exactly:
- The block API (Toy::LLM::Blocks::TransformerBlock): attr_accessors for the weights it OWNS, and the build_forward method signature — adapted to seq-mode reality (the per-forward context from recon.forward-body.ivars_read is passed IN as args, NOT stored as state). Pure-ish: the block owns weights, the cache hands it the session + context per forward. NO invented KV-state return.
- what_moves vs what_stays (be concrete: method bodies, fields).
- The edit to each of the 4 realize paths.
- The mirror-generator changes.
- The ordered_steps the executor runs (write block file -> rewire block-stacking call sites -> rewire realize paths -> require -> regen mirrors -> verify-mirrors -> gate -> commit).
- parity_risks (Spinel landmines: no default-arg ctors, no ivar widening, Array destructure; and behavior-preservation risks like the B>1/B=1 GQA branch, set_output taps, residual order).

Set feasible=false ONLY if a bit-identical single-commit lift looks infeasible (explain). Do NOT edit any file — you only plan.

Reference docs:
  - ${REFS}`, { label: 'l2-plan', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan || !plan.feasible) {
  log(`ABORT: plan infeasible — ${plan ? plan.parity_risks?.join('; ') : 'planner failed'}. No edits made.`)
  return { aborted: true, reason: 'plan infeasible', baseSha, recon, plan }
}
log(`Plan ready: ${plan.ordered_steps?.length || 0} steps. Proceeding to gated extraction.`)

// ---- Phase 3: Single gated extraction (one commit) -------------------------

phase('Extract')

const result = await agent(`Execute the L2 TransformerBlock extraction (toy-framework P2.4). STRICTLY GATED, SINGLE COMMIT. The ONLY acceptable outcome is byte-identical forward output; "close" is a failure. Do the block as ONE unit (not field-by-field), one commit.

${SKETCH_CAVEAT}

== Gate fixture (proven deterministic + parity-sensitive) ==
Build:        ${BUILD_CMD}
Fingerprint:  ${FINGERPRINT_CMD}  | sha256sum
Baseline:     ${baseline.baseline_value}

== The plan to execute ==
${JSON.stringify(plan, null, 2)}

== Procedure ==
1. CAPTURE BASELINE: ${BUILD_CMD} && fingerprint | sha256sum, confirm == ${baseline.baseline_value}. If it doesn't match pre-edit, STOP (gate_passed=false, failure_reason="baseline mismatch pre-edit").
2. WRITE ${BLOCK_FILE} per plan.block_api. Spinel hygiene: no default-arg ctors (landmine #4), no nil->RbVal ivar widening (seed concrete values), no Array destructure widening, no FFI :str into step_bind. Match the L1 primitives' module style (no require_relative "tinynn" — the monolith's backend require governs which TinyNN is in scope).
3. REWIRE the block-stacking call sites + the 4 realize paths per the plan, so the cache constructs TransformerBlock instances, hands them their weights, and calls build_forward with the per-forward context. Remove the old build_seq_block/build_seq_qhead bodies from the monolith (or reduce them to thin delegators only if the plan says so).
4. MIRRORS: apply plan.mirror_changes to ${MIRRORGEN}, then \`ruby ${MIRRORGEN}\`, then \`make verify-mirrors\` — MUST exit 0.
5. CPU PARITY GATE: ${BUILD_CMD} && fingerprint | sha256sum. MUST byte-equal ${baseline.baseline_value}. If not, HARD FAIL.
6. CUDA SMOKE (best effort, never silent): attempt the CUDA mirror path; if the full CUDA toolchain isn't available (the checked-in libtinynn_ggml_cuda.a may be the AB-smoke stub; full needs \`make setup-ggml-cuda\`), say so explicitly in cuda_result. verify-mirrors already guarantees mirror-source consistency.
7. COMMIT (only if bit_identical && verify_mirrors_clean): stage ONLY \`git add ${BLOCK_FILE} ${MONOLITH} ${MIRRORGEN}\` + the regenerated *_cuda.rb/*_metal.rb mirrors (incl. the new block mirrors). NEVER \`git add -A\`. Commit title "P2.4: extract L2 TransformerBlock". Capture the sha.

== On ANY gate failure ==
Do NOT patch around it. Revert cleanly: \`git checkout -- ${MONOLITH} ${MIRRORGEN}\` and the mirror files, \`rm -f ${BLOCK_FILE}\` and any new mirror files, confirm \`git status\` is clean vs ${baseSha}. Report gate_passed=false + precise failure_reason. A clean halt is the correct outcome.

gate_passed=true ONLY if bit_identical && verify_mirrors_clean && committed.

Reference docs:
  - ${REFS}`, { label: 'extract:transformer_block', phase: 'Extract', schema: EXTRACT_SCHEMA })

if (result && result.gate_passed && result.committed) {
  log(`OK: L2 TransformerBlock committed ${result.commit_sha || '(sha?)'} | bit-identical | verify-mirrors clean | cuda=${result.cuda_checked ? result.cuda_result : 'not run'}`)
} else {
  log(`HALT: L2 extraction did not pass the gate — ${result ? (result.failure_reason || result.notes) : 'agent returned nothing'}. Tree should be clean vs ${baseSha}.`)
}

return {
  baseSha,
  gate_baseline: baseline.baseline_value,
  feasible: plan.feasible,
  result,
  followup: 'Pushed nothing (land on main, push when ready). If gate passed: next is L3 LlamaArch (P2.5), then the realize-path bulk (P2.6/P2.7). If halted: tree is clean — review the failure_reason and re-plan the block boundary.',
}
