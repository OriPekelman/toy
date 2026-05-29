export const meta = {
  name: 'finish-l1-primitives',
  description: 'Extract RoPE→SwiGLU→RMSNorm→GQA from the seq-forward monolith into L1 primitives, each behind a bit-identical-logits gate, one commit per primitive, stop on first gate failure',
  whenToUse: 'Continuing the toy-framework P2 five-layer refactor: extract the four L1 primitives with strict parity gating',
  phases: [
    { title: 'Baseline', detail: 'capture HEAD + confirm a deterministic gate fixture' },
    { title: 'Recon', detail: 'one read-only agent per primitive produces an exact extraction spec' },
    { title: 'Extract', detail: 'serial gated extraction; one commit per primitive; stop on first failure' },
  ],
}

// ---- Constants -------------------------------------------------------------

const MONOLITH = 'lib/llama_seq_forward_ffi.rb'
const MIRRORGEN = 'prep/gen_cuda_mirror.rb'

// Authoritative plan + landmine references the agents must consult.
const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29.md  (authoritative P2.3 plan; RoPE API + exact steps)',
  'lib/toy/llm/primitives/README.md           (L1 contract sketch)',
  'docs/roadmap/toy-framework-design-2026-05-28.md',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/project_step_bind_landmine_2026_05_28.md',
].join('\n  - ')

// RoPE is fully specified in the session-resume doc; the others get the
// generic contract + their grep-able anchor so the recon agent locates
// current line numbers itself (the doc's numbers drift as the file changes).
const PRIMITIVES = [
  {
    key: 'rope',
    title: 'RoPE',
    module: 'Toy::LLM::Primitives::RoPE',
    file: 'lib/toy/llm/primitives/rope.rb',
    anchor: 'tnn_rope_ext',
    notes: 'FULLY SPECIFIED in SESSION-RESUME-2026-05-29.md (the apply_2d signature, the Cfg value object with 7 ctor args, the K-path + Q-path call sites, and the 4 realize-path ivar builds). Follow that doc exactly. Reconcile the README "build" sketch against the doc — the doc (apply_2d with the shape-lift baked in) wins because the two call sites both do reshape_3d -> rope -> reshape_2d identically.',
  },
  {
    key: 'swiglu',
    title: 'SwiGLU',
    module: 'Toy::LLM::Primitives::SwiGLU',
    file: 'lib/toy/llm/primitives/swiglu.rb',
    anchor: 'tnn_silu',
    notes: 'SiLU-gated FFN. The FFN block (gate proj -> silu -> up proj -> elementwise mul -> down proj) lives around the tnn_silu call. Extract the SiLU-gate composition as a pure function; the weight tensors stay owned by the block/cache (passed in as args), per the README "What lives on the BLOCK, not here".',
  },
  {
    key: 'rms_norm',
    title: 'RMSNorm',
    module: 'Toy::LLM::Primitives::RMSNorm',
    file: 'lib/toy/llm/primitives/rms_norm.rb',
    anchor: 'tnn_rms_norm',
    notes: 'THREE call sites (final norm + rn1 + rn2). It is thin (one tnn call + eps), so the win is consistency, not LOC. Pure function build(sess, x, gamma, eps). Gamma tensors stay owned by the caller. Make sure all three sites collapse to the primitive.',
  },
  {
    key: 'gqa',
    title: 'GQA',
    module: 'Toy::LLM::Primitives::GQA',
    file: 'lib/toy/llm/primitives/gqa.rb',
    anchor: 'soft_max',
    notes: 'THE HARD ONE. The doc calls it "too thick — the head loop is half the block." Extract conservatively: keep weight ownership + KV cache on the block; the primitive should encapsulate only the attention math (scores -> scaled softmax -> weighted V), parameterised by group_size so MHA is group_size=1. If a clean pure-function extraction that passes the bit-identical gate is NOT achievable without dragging block state in, DO NOT force it: report gate_passed=false with a clear failure_reason describing what block state blocks the extraction, and let the loop halt for human re-planning. A halt here is a SUCCESS of the gate, not a workflow failure.',
  },
]

// ---- Schemas ---------------------------------------------------------------

const GATE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['build_cmd', 'fixture_cmd', 'deterministic', 'baseline_value', 'notes'],
  properties: {
    build_cmd: { type: 'string', description: 'exact make command that builds the chosen fixture example' },
    fixture_cmd: { type: 'string', description: 'exact command that runs the fixture and emits stable logits/output to stdout' },
    deterministic: { type: 'boolean', description: 'true only if two consecutive runs produced byte-identical output' },
    baseline_value: { type: 'string', description: 'a short, stable fingerprint of the baseline output (e.g. sha256 of the logits section, or the first N logit floats) that later agents compare against' },
    fingerprint_cmd: { type: 'string', description: 'the exact pipe that reduces fixture output to baseline_value (e.g. "... | sha256sum"); empty if none' },
    built_new_fixture: { type: 'boolean', description: 'true if you had to author a new deterministic harness because no existing example was usable' },
    notes: { type: 'string' },
  },
}

const RECON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['primitive', 'ready', 'api_signature', 'call_sites', 'cfg_fields', 'realize_touch', 'mirror_changes', 'risks'],
  properties: {
    primitive: { type: 'string' },
    ready: { type: 'boolean', description: 'true if a clean pure-function extraction looks achievable behind a bit-identical gate' },
    api_signature: { type: 'string', description: 'the exact module + method signature to create' },
    call_sites: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false, required: ['line', 'what'],
        properties: { line: { type: 'integer' }, what: { type: 'string' } },
      },
      description: 'CURRENT line numbers in the monolith (you must grep, do not trust the doc numbers)',
    },
    cfg_fields: { type: 'array', items: { type: 'string' }, description: 'fields of the Cfg value object, or empty if the primitive takes plain args' },
    realize_touch: { type: 'array', items: { type: 'string' }, description: 'realize-path edits needed (e.g. cfg-ivar build at lines X/Y/Z), or empty' },
    mirror_changes: { type: 'string', description: 'what to add to prep/gen_cuda_mirror.rb (MIRRORABLE entry + subs_for case)' },
    risks: { type: 'array', items: { type: 'string' }, description: 'Spinel landmines or parity risks specific to this primitive' },
  },
}

const EXTRACT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['primitive', 'gate_passed', 'bit_identical', 'verify_mirrors_clean', 'committed', 'notes'],
  properties: {
    primitive: { type: 'string' },
    gate_passed: { type: 'boolean', description: 'true ONLY if bit_identical && verify_mirrors_clean && committed' },
    bit_identical: { type: 'boolean', description: 'fixture output byte-identical to baseline after the extraction' },
    verify_mirrors_clean: { type: 'boolean', description: '`make verify-mirrors` exited 0' },
    cuda_checked: { type: 'boolean', description: 'CUDA mirror runtime smoke was run' },
    cuda_result: { type: 'string', description: 'CUDA smoke outcome, or why it could not be run (never silently skip)' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    logit_diff_summary: { type: 'string', description: 'baseline fingerprint vs post fingerprint' },
    failure_reason: { type: 'string', description: 'if gate_passed=false, exactly what failed and what was reverted' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: Baseline -----------------------------------------------------

phase('Baseline')

const baseline = await agent(`You are setting up the parity gate for a delicate refactor of the toy-ruby-neural-network seq-forward monolith. Working dir is the repo root on host gx10.

Goal of this phase: pick (or build) ONE fast, fully-deterministic example that exercises the sequence-mode forward pass in ${MONOLITH}, build it, and capture a baseline output fingerprint. Every later step re-runs this exact fixture and must reproduce byte-identical output.

Steps:
1. Read ${MONOLITH} header + ${REFS.split('\\n')[0]} to understand the seq-forward entry points.
2. Candidate fixtures that depend on ${MONOLITH} (from the Makefile): examples/example_lmc (08_lmc.rb), examples/smoke_projection_lens, examples/example_warm_start_train. Prefer the FASTEST one whose output is a deterministic function of the forward pass (no sampling/RNG, no wall-clock). smoke_projection_lens (logit-lens over fixed input) is the leading candidate. Avoid anything that does sampling or trains with randomness unless you seed it.
3. Build it via make (this builds tinynn/libtinynn_ggml.a + the example). Run it TWICE and confirm the relevant output (logits / lens values) is byte-identical across runs. If it is not deterministic, seed it or pick another; if none work, author a minimal deterministic harness under examples/ that runs the seq forward on a fixed tiny input and prints logits, and use that.
4. Define a fingerprint command (e.g. extract the logits block and pipe to sha256sum) and record the baseline value.

Do NOT modify ${MONOLITH} or any lib/ file. Only build + run, and optionally add a new example fixture file. Report the exact build_cmd, fixture_cmd, fingerprint_cmd, baseline_value, and whether you built a new fixture.

Reference docs:
  - ${REFS}`, { label: 'gate-fixture', phase: 'Baseline', schema: GATE_SCHEMA })

const shaRaw = await agent(`Run \`git rev-parse HEAD\` in the repo root. Then run \`git status --porcelain --untracked-files=no\` to list ONLY tracked changes (untracked build artifacts are expected and fine). Report two things: the 40-char HEAD sha, and whether there are any tracked staged/unstaged changes. If the tracked-changes list is empty, end your message with the line "CLEAN <sha>"; otherwise end with "DIRTY <sha>".`, { label: 'base-sha', phase: 'Baseline' })

const shaMatch = (shaRaw || '').match(/\b([0-9a-f]{40})\b/)
const baseSha = shaMatch ? shaMatch[1] : ''
const treeDirty = /\bDIRTY\b/.test(shaRaw || '')

log(`Baseline HEAD: ${baseSha || '(unparsed)'} | tracked-tree ${treeDirty ? 'DIRTY' : 'clean'}`)
log(`Gate fixture: ${baseline?.fixture_cmd || '(none — baseline agent failed)'} | deterministic=${baseline?.deterministic}`)

if (!baseline || !baseline.deterministic) {
  log('ABORT: no deterministic gate fixture — cannot guarantee bit-identical parity. Stopping before any edits.')
  return { aborted: true, reason: 'no deterministic gate fixture', baseline }
}
if (!baseSha || treeDirty) {
  log('ABORT: working tree has non-artifact tracked changes, or HEAD unreadable. Commit/stash first.')
  return { aborted: true, reason: 'dirty tree or bad HEAD', baseSha, shaRaw }
}

// ---- Phase 1: Recon (parallel, read-only) ----------------------------------

phase('Recon')

const reconPrompt = (p) => `You are doing READ-ONLY recon for extracting the ${p.title} primitive (${p.module}) out of ${MONOLITH} into ${p.file}, as part of the toy-framework P2 five-layer refactor.

Anchor: grep the monolith for \`${p.anchor}\` to find the current call site(s). Line numbers in the docs have DRIFTED — report the CURRENT ones.

Primitive-specific notes: ${p.notes}

Produce an exact extraction spec:
- The module + method signature to create (pure function, self. methods only — no ivars on the module; per the README this avoids Spinel ivar-layout landmines).
- Every current call site (line + what it does).
- The Cfg value-object fields if the param list is long (RoPE-style), else empty if plain args suffice.
- Realize-path edits needed (e.g. building a cfg ivar once; the doc lists ~4 realize paths around lines 222/512/810/1046 — verify current locations).
- What to add to ${MIRRORGEN}: the MIRRORABLE entry (the new primitive file must be mirrored too) and the subs_for case (likely just common_module_tail since the module name ${p.module} does not collide with TinyNN; plus the require-relative rewrite).
- Spinel landmines / parity risks specific to this primitive (default-arg poisoning on Cfg ctors, Array destructure widening, step_bind+:str, etc — see the landmine memory files).

Set ready=false ONLY if a clean pure-function extraction behind a bit-identical gate looks infeasible (explain why in risks). Do NOT edit any file.

Reference docs:
  - ${REFS}`

const specsArr = await parallel(PRIMITIVES.map((p) => () =>
  agent(reconPrompt(p), { label: `recon:${p.key}`, phase: 'Recon', schema: RECON_SCHEMA })))

const specs = {}
PRIMITIVES.forEach((p, i) => { specs[p.key] = specsArr[i] })

for (const p of PRIMITIVES) {
  const s = specs[p.key]
  log(`recon ${p.key}: ${s ? (s.ready ? 'READY' : 'NOT READY — ' + (s.risks || []).join('; ')) : 'FAILED'} | ${s?.call_sites?.length || 0} call site(s)`)
}

// ---- Phase 2: Serial gated extraction (one commit per primitive) -----------

phase('Extract')

const extractPrompt = (p, spec) => `You are extracting the ${p.title} primitive (${p.module}) from ${MONOLITH} into ${p.file}. This is a STRICTLY GATED, single-commit refactor. The ONLY acceptable outcome is byte-identical forward-pass output; "close" is a failure.

== The gate fixture (already verified deterministic) ==
Build:        ${baseline.build_cmd}
Run:          ${baseline.fixture_cmd}
Fingerprint:  ${baseline.fingerprint_cmd || '(compare full output)'}
Baseline:     ${baseline.baseline_value}

== Your recon spec for this primitive ==
${JSON.stringify(spec, null, 2)}

== Procedure (do exactly this, in order) ==
1. CAPTURE BASELINE: build + run the fixture, compute the fingerprint, confirm it equals the baseline above. If it does NOT match the recorded baseline even before you edit anything, STOP — something is wrong with the tree; report gate_passed=false, failure_reason="baseline mismatch pre-edit".
2. WRITE ${p.file}: the module per the spec's api_signature. Pure functions only (self.), no module ivars. ${p.key === 'rope' ? 'Follow SESSION-RESUME-2026-05-29.md exactly (apply_2d + the 7-arg Cfg ctor; NO optional defaults on the ctor — Spinel default-arg poisoning).' : 'Keep weight/KV-cache ownership on the caller; the primitive takes tensors + config as args.'} Do NOT require it from anywhere except step 3.
3. REWRITE CALL SITES: replace every call site listed in the spec with a call to the new primitive. Add the require_relative. ${spec.realize_touch && spec.realize_touch.length ? 'Build the cfg ivar once in each realize path as the spec describes.' : ''} After this, grep for \`${p.anchor}\` — it must match ONLY ${p.key === 'rope' ? 'vendor/ and your new primitive' : 'your new primitive (and vendor/ if applicable)'}; no bare call sites should remain in the monolith.
4. MIRRORS: apply ${spec.mirror_changes} to ${MIRRORGEN}, then regenerate: \`ruby ${MIRRORGEN}\`. Then \`make verify-mirrors\` — it MUST exit 0.
5. CPU PARITY GATE: rebuild + re-run the fixture, recompute the fingerprint. It MUST byte-equal the baseline. If not, the extraction changed behaviour — this is a HARD FAIL.
6. CUDA SMOKE (best effort, never silent): try to build/run the CUDA mirror path for the fixture. If the CUDA toolchain isn't readily available here, say so explicitly in cuda_result — do not pretend it passed. verify-mirrors already protects mirror-source consistency; the runtime smoke is extra assurance.
7. COMMIT (only if bit_identical AND verify-mirrors clean): stage ONLY the relevant paths (\`git add ${p.file} ${MONOLITH} ${MIRRORGEN}\` plus the regenerated *_cuda.rb / *_metal.rb mirror files and any new fixture file — NEVER \`git add -A\`). Commit with title "P2.x: extract ${p.title} as L1 primitive". Capture the sha.

== On ANY gate failure ==
Do NOT patch around it. Revert your edits cleanly: \`git checkout -- ${MONOLITH} ${MIRRORGEN}\` and the mirror files, and \`rm -f ${p.file}\` (and any new file you added). Confirm \`git status\` shows no tracked changes vs ${baseSha}. Report gate_passed=false with a precise failure_reason. The loop will halt for human re-planning — that is the correct, safe outcome.

Set gate_passed=true ONLY if bit_identical && verify_mirrors_clean && committed are all true.

Reference docs:
  - ${REFS}`

const results = []
for (const p of PRIMITIVES) {
  const spec = specs[p.key]
  if (!spec) {
    log(`STOP before ${p.key}: recon failed (no spec). Halting.`)
    results.push({ primitive: p.key, gate_passed: false, committed: false, notes: 'recon produced no spec', halted: true })
    break
  }
  if (!spec.ready) {
    log(`STOP before ${p.key}: recon flagged NOT READY. Halting for human re-planning.`)
    results.push({ primitive: p.key, gate_passed: false, committed: false, notes: 'recon not ready: ' + (spec.risks || []).join('; '), halted: true })
    break
  }

  const r = await agent(extractPrompt(p, spec), { label: `extract:${p.key}`, phase: 'Extract', schema: EXTRACT_SCHEMA })
  results.push(r)

  if (!r) {
    log(`STOP: ${p.key} extraction agent returned nothing. Halting.`)
    break
  }
  if (r.gate_passed && r.committed) {
    log(`OK ${p.key}: committed ${r.commit_sha || '(sha?)'} | bit-identical | verify-mirrors clean | cuda=${r.cuda_checked ? r.cuda_result : 'not run'}`)
  } else {
    log(`STOP: ${p.key} gate FAILED — ${r.failure_reason || r.notes}. Reverted; halting before remaining primitives.`)
    break
  }
}

const committed = results.filter((r) => r && r.gate_passed && r.committed)
const order = PRIMITIVES.map((p) => p.key)

return {
  baseSha,
  gate_fixture: baseline.fixture_cmd,
  planned_order: order,
  completed: committed.map((r) => ({ primitive: r.primitive, sha: r.commit_sha })),
  halted_at: results.length && !results[results.length - 1].gate_passed ? results[results.length - 1].primitive : null,
  results,
  followup: 'Pushed nothing (per session-resume doc: land on main, push when ready). L2 blocks / L3 arch (P2.4/P2.5) are the next workflow once L1 is clean.',
}
