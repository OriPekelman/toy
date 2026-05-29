export const meta = {
  name: 'finish-realize-bulk',
  description: 'Decompose the ~1750-line realize-path bulk (realize_for_q8_copy/_mmap/_full_finetune/_random_init) onto the L3 arch / L2 block now that they are the allocation units, behind a DUAL bit-identical gate (CPU + CUDA self-consistency), as a sequence of small gated commits; never behaviorally restructure an un-gated realize path; stop on first gate failure',
  whenToUse: 'Continuing the toy-framework P2 refactor after L1+L2+L3: the realize-path bulk (P2.6/P2.7), now with a real CUDA gate',
  phases: [
    { title: 'Baseline', detail: 'capture HEAD + BOTH (CPU + CUDA) deterministic gate fingerprints' },
    { title: 'Recon', detail: 'parallel read-only agents map the 4 realize paths, the arch/block seam, fixture coverage per path, mirror gen' },
    { title: 'Plan', detail: 'one architect proposes a conservative, incremental, dual-gated decomposition + handles the GGUF coverage gap' },
    { title: 'Extract', detail: 'serial gated loop over the planned steps; one commit each; dual gate; halt on first failure or any un-gated-path restructure' },
  ],
}

// ---- Constants -------------------------------------------------------------

const MONOLITH = 'lib/llama_seq_forward_ffi.rb'
const MIRRORGEN = 'prep/gen_cuda_mirror.rb'
const ARCH_FILE = 'lib/toy/llm/archs/llama_arch.rb'
const BLOCK_FILE = 'lib/toy/llm/blocks/transformer_block.rb'

// Dual gate. Both proven deterministic; CPU and CUDA fingerprints differ
// (GPU floats != CPU) — each backend is gated against ITS OWN baseline.
const CPU_BUILD = 'make examples/smoke_projection_lens'
const CPU_FP = 'SEED=0 STEPS=5 ./examples/smoke_projection_lens 2>/dev/null | grep -E "^step|^initial=" | sha256sum'
const CUDA_BUILD = 'make examples/smoke_projection_lens_cuda'
const CUDA_FP = 'SEED=0 STEPS=5 ./examples/smoke_projection_lens_cuda 2>/dev/null | grep -E "^step|^initial=" | sha256sum'

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29d.md  (authoritative checkpoint; L1+L2+L3 done, realize bulk is next)',
  'lib/toy/llm/archs/llama_arch.rb             (L3 arch — owns whole-graph handles; realize logic should land here/on the block)',
  'lib/toy/llm/blocks/transformer_block.rb     (L2 block — owns per-block weights; per-block weight LOADING could land here)',
  'docs/roadmap/toy-framework-design-2026-05-28.md  (design §6: arch owns whole-graph alloc; block owns its weights)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (incl. #16 Struct.new — matz/spinel#1043; NO Struct.new for value objects)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/project_step_bind_landmine_2026_05_28.md',
].join('\n  - ')

// The framing that makes this safe.
const FRAMING = `CRITICAL FRAMING.
(1) SIZE + INCREMENT. The four realize_for_* methods are ~1750 lines and intertwined. Do NOT attempt to lift them all in one change. Decompose into the SMALLEST safely-gated units (e.g. extract a shared weight-load helper; move per-block weight loading onto the block; move whole-graph ctx allocation onto the arch) and land each as its own gated commit. A PARTIAL but safe result (a few clean commits, the rest deferred) is the CORRECT outcome — do not over-reach.
(2) DUAL GATE. Every commit must reproduce BOTH the CPU and the CUDA fixture fingerprints byte-identically. CUDA floats != CPU floats, so each backend is compared against its OWN recorded baseline (self-consistency before/after), never CPU-vs-CUDA.
(3) COVERAGE GAP — the load-bearing constraint. BOTH gate fixtures (smoke_projection_lens CPU+CUDA) only exercise realize_for_random_init. The other three paths — realize_for_q8_copy, realize_for_mmap, realize_for_full_finetune — load GGUF weights and are NOT behaviorally gated by these fixtures. Therefore: you may move/extract code SHARED across all paths only if the shared unit is exercised by random_init (so the gate actually covers it). You may NOT behaviorally restructure logic that is SPECIFIC to a GGUF path (q8/mmap/full_finetune) unless you first build a deterministic gate that exercises it. If a clean extraction would require touching un-gated GGUF-specific logic, either (a) the recon/plan finds & wires a fast deterministic GGUF fixture for it, or (b) that part is DEFERRED and reported — never refactor it blind. Set touched_ungated_path=true and HALT if a step would cross this line without a gate.`

// ---- Schemas ---------------------------------------------------------------

const GATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['cpu_deterministic', 'cpu_baseline', 'cuda_deterministic', 'cuda_baseline', 'notes'],
  properties: {
    cpu_deterministic: { type: 'boolean', description: 'two consecutive CPU runs byte-identical' },
    cpu_baseline: { type: 'string', description: 'sha256 of the CPU fingerprint at HEAD' },
    cuda_deterministic: { type: 'boolean', description: 'two consecutive CUDA runs byte-identical (GPU reductions can be non-deterministic — verify!)' },
    cuda_baseline: { type: 'string', description: 'sha256 of the CUDA fingerprint at HEAD' },
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
    shared_vs_specific: { type: 'string', description: 'for the 4-paths facet: what logic is SHARED across all 4 realize paths (gate-covered via random_init) vs SPECIFIC to a GGUF path (NOT gated)' },
    fixture_coverage: { type: 'string', description: 'for the coverage facet: which realize path each available deterministic fixture exercises; whether a fast deterministic GGUF fixture (or a random_init->write_gguf->reload round-trip) exists or can be cheaply built for q8/mmap/full_finetune' },
    risks: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['feasible', 'decomposition', 'gguf_coverage_handling', 'deferred', 'mirror_changes', 'parity_risks'],
  properties: {
    feasible: { type: 'boolean', description: 'true if at least one safely-gated extraction step exists; false halts before any edit' },
    decomposition: {
      type: 'array',
      description: 'ordered, SMALL, independently-gated extraction steps (1..N). Empty if nothing can be safely extracted.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['step_title', 'what_moves', 'paths_touched', 'gate_covered', 'risk'],
        properties: {
          step_title: { type: 'string' },
          what_moves: { type: 'string', description: 'precise: what code moves where (onto arch / block / shared helper)' },
          paths_touched: { type: 'array', items: { type: 'string' }, description: 'which realize_for_* methods this step edits' },
          gate_covered: { type: 'boolean', description: 'true if random_init (the gated path) actually exercises everything this step changes' },
          new_gate: { type: 'string', description: 'if this step touches an un-gated GGUF path, the deterministic fixture that must be built/wired first; empty otherwise' },
          risk: { type: 'string' },
        },
      },
    },
    gguf_coverage_handling: { type: 'string', description: 'how the GGUF-path coverage gap is handled: build a fixture (which?) or scope those paths out' },
    deferred: { type: 'array', items: { type: 'string' }, description: 'realize logic intentionally NOT touched this pass + why' },
    mirror_changes: { type: 'string', description: 'any gen_cuda_mirror.rb changes needed (likely none — arch/block/monolith already mirrored)' },
    parity_risks: { type: 'array', items: { type: 'string' } },
  },
}

const STEP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['step_title', 'gate_passed', 'cpu_bit_identical', 'cuda_bit_identical', 'verify_mirrors_clean', 'committed', 'touched_ungated_path', 'notes'],
  properties: {
    step_title: { type: 'string' },
    gate_passed: { type: 'boolean', description: 'true ONLY if cpu_bit_identical && cuda_bit_identical && verify_mirrors_clean && committed && !touched_ungated_path' },
    cpu_bit_identical: { type: 'boolean' },
    cuda_bit_identical: { type: 'boolean' },
    verify_mirrors_clean: { type: 'boolean' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    touched_ungated_path: { type: 'boolean', description: 'true if the step had to change GGUF-path-specific logic without a gate — if so it must HALT, not commit' },
    fingerprints: { type: 'string', description: 'cpu baseline vs post + cuda baseline vs post' },
    failure_reason: { type: 'string' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: Baseline (dual) ----------------------------------------------

phase('Baseline')

const baseline = await agent(`Set up the DUAL parity gate for the realize-path refactor. Repo root, host gx10 (GB10 GPU present). Two fixtures, both must be confirmed deterministic:

  CPU:   build \`${CPU_BUILD}\`, then run \`${CPU_FP}\` TWICE — confirm byte-identical, record cpu_baseline.
  CUDA:  build \`${CUDA_BUILD}\` (slow ~712MB static link, be patient), then run \`${CUDA_FP}\` TWICE — confirm byte-identical, record cuda_baseline. NOTE: GPU reductions CAN be non-deterministic; if the two CUDA runs differ, set cuda_deterministic=false (that would block the CUDA gate — report it clearly).

The CPU and CUDA fingerprints WILL differ from each other (GPU floats != CPU) — that is expected; each backend is gated against its own baseline. Do NOT edit any file.`, { label: 'dual-gate', phase: 'Baseline', schema: GATE_SCHEMA })

const shaRaw = await agent(`Run \`git rev-parse HEAD\`, then \`git status --porcelain --untracked-files=no\`. Report the 40-char HEAD sha. End your message with "CLEAN <sha>" if there are no tracked changes (untracked build artifacts incl. the 712MB cuda binary are fine), else "DIRTY <sha>".`, { label: 'base-sha', phase: 'Baseline' })

const shaMatch = (shaRaw || '').match(/\b([0-9a-f]{40})\b/)
const baseSha = shaMatch ? shaMatch[1] : ''
const treeDirty = /\bDIRTY\b/.test(shaRaw || '')

log(`Baseline HEAD: ${baseSha || '(unparsed)'} | tracked-tree ${treeDirty ? 'DIRTY' : 'clean'}`)
log(`CPU gate:  ${baseline?.cpu_baseline || '(none)'} det=${baseline?.cpu_deterministic}`)
log(`CUDA gate: ${baseline?.cuda_baseline || '(none)'} det=${baseline?.cuda_deterministic}`)

if (!baseline || !baseline.cpu_deterministic || !baseline.cuda_deterministic) {
  log('ABORT: one or both gates not deterministic — cannot guarantee bit-identical parity on that backend.')
  return { aborted: true, reason: 'gate not deterministic', baseline }
}
if (!baseSha || treeDirty) {
  log('ABORT: dirty tracked tree or unreadable HEAD.')
  return { aborted: true, reason: 'dirty tree or bad HEAD', baseSha, shaRaw }
}

// ---- Phase 1: Recon (parallel, read-only) ----------------------------------

phase('Recon')

const FACETS = [
  {
    key: 'four-paths',
    prompt: `Map the FOUR realize paths in ${MONOLITH}: realize_for_q8_copy (~236), realize_for_mmap (~533), realize_for_full_finetune (~838), realize_for_random_init (~1081). For each: what it allocates (arch-level handles vs per-block weights), how it loads weights (random seed / gguf copy / mmap / finetune), and the block-stacking/assignment loop. KEY DELIVERABLE (shared_vs_specific): precisely what logic is SHARED/identical across all 4 (candidate for a gate-covered extraction since random_init exercises it) vs what is SPECIFIC to a GGUF path (q8/mmap/full_finetune — NOT exercised by the random_init gate).`,
  },
  {
    key: 'arch-block-seam',
    prompt: `Map the SEAM for moving realize logic onto the L3 arch / L2 block. The arch (${ARCH_FILE}) now owns whole-graph handles (token_embed/final_norm/output/w_proj/blocks); the block (${BLOCK_FILE}) owns per-block weights. Per design §6: arch owns whole-graph allocation, block owns its weight loading. Identify which realize sub-steps could move onto arch.realize_*/block.load_* methods WITHOUT restructuring the allocation primitives — i.e. the same "lift orchestration, keep primitives" pattern as L2/L3. Note every cache ivar each sub-step reads/writes (so the moved method can take them as args / set them on the owning object).`,
  },
  {
    key: 'fixture-coverage',
    prompt: `Map FIXTURE COVERAGE per realize path. The two gates (smoke_projection_lens CPU + CUDA) exercise realize_for_random_init only. Question 1: is there ANY existing deterministic example/demo that exercises realize_for_mmap / _q8_copy / _full_finetune reproducibly (e.g. demos/smollm2_seq_parity*, or a small GGUF in data/)? List candidates + whether they're deterministic + fast. Question 2: could a CHEAP deterministic GGUF gate be built — e.g. random_init -> toy_gguf_writer -> reload via realize_for_mmap/q8 (round-trip), giving the GGUF paths gate coverage without a real model download? Assess feasibility concretely (does toy_gguf_writer exist? is there a reload smoke?).`,
  },
  {
    key: 'mirror-gen',
    prompt: `Confirm the MIRROR situation. ${MONOLITH}, ${ARCH_FILE}, ${BLOCK_FILE} are all in MIRRORABLE already. If realize logic moves onto the arch/block, the existing mirrors regenerate automatically. Verify no gen_cuda_mirror.rb change is needed unless a NEW file is introduced (e.g. a shared realize helper). Note any class/identifier that could trip a TinyNN/class rename or a Spinel name-collision (landmine #16).`,
  },
]

const reconArr = await parallel(FACETS.map((f) => () =>
  agent(`READ-ONLY recon for the realize-path bulk refactor (toy-framework P2.6/P2.7). ${FRAMING}

Your facet: ${f.prompt}

Report current file:line refs (grep — do not trust doc numbers). Do NOT edit any file.

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

const plan = await agent(`You are the architect for the realize-path bulk refactor (toy-framework P2.6/P2.7). Synthesize the recon into a CONSERVATIVE, INCREMENTAL, DUAL-GATED decomposition. ${FRAMING}

== Recon ==
${JSON.stringify(recon, null, 2)}

Produce decomposition: an ORDERED list of the SMALLEST safely-gated extraction steps. For EACH step set gate_covered honestly (does random_init exercise everything it changes?) and, if it touches a GGUF-specific path, name the new_gate fixture that must exist first. Prefer steps that move SHARED logic (gate-covered) onto the arch/block. Put GGUF-path-specific restructuring into 'deferred' UNLESS gguf_coverage_handling describes a cheap deterministic GGUF gate you're confident in (then make building that gate an early step).

Bias strongly toward SAFETY over completeness: a 2-3 step plan that's all gate-covered beats an ambitious plan that touches un-gated paths. Set feasible=false (empty decomposition) if nothing can be safely extracted under the dual gate. Use hand-written classes (NO Struct.new — landmine #16) for any value object. Do NOT edit any file.

Reference docs:
  - ${REFS}`, { label: 'realize-plan', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan || !plan.feasible || !plan.decomposition || plan.decomposition.length === 0) {
  log(`STOP: no safely-gated extraction available — ${plan ? (plan.parity_risks || []).join('; ') : 'planner failed'}. No edits made. This is a valid outcome: the realize bulk may need gates built first.`)
  return { aborted: true, reason: 'no safely-gated steps', baseSha, recon, plan }
}
log(`Plan: ${plan.decomposition.length} step(s). Deferred: ${(plan.deferred || []).length}. GGUF handling: ${plan.gguf_coverage_handling?.slice(0, 120)}`)

// ---- Phase 3: Serial dual-gated extraction loop ----------------------------

phase('Extract')

const stepPrompt = (step, idx) => `Execute realize-bulk extraction STEP ${idx + 1}/${plan.decomposition.length}: "${step.step_title}". STRICTLY DUAL-GATED, SINGLE COMMIT. The ONLY acceptable outcome is byte-identical CPU AND CUDA fixture output. ${FRAMING}

== This step ==
${JSON.stringify(step, null, 2)}

== Plan context (for consistency across steps) ==
gguf_coverage_handling: ${plan.gguf_coverage_handling}
mirror_changes: ${plan.mirror_changes}

== Dual gate ==
CPU:  build \`${CPU_BUILD}\`; fingerprint \`${CPU_FP}\`; baseline ${baseline.cpu_baseline}
CUDA: build \`${CUDA_BUILD}\`; fingerprint \`${CUDA_FP}\`; baseline ${baseline.cuda_baseline}

== Procedure ==
1. GUARD: if this step would change logic SPECIFIC to a GGUF path (q8/mmap/full_finetune) and step.new_gate is empty (no gate built/wired), STOP immediately: set touched_ungated_path=true, gate_passed=false, do nothing, explain. (Unless the plan's gguf_coverage_handling has you BUILD step.new_gate first as part of this step — then build + verify that fixture is deterministic before proceeding.)
2. CAPTURE BASELINES: build + fingerprint BOTH backends, confirm == the recorded baselines. If either mismatches pre-edit, STOP (gate_passed=false, failure_reason="baseline mismatch pre-edit"). (CUDA build is slow; that's fine.)
3. APPLY the step's change per what_moves. Spinel hygiene: NO Struct.new (landmine #16 / matz/spinel#1043 — hand-written class, positional ctor, no default args, uniquely-prefixed members); no nil->RbVal ivar widening; no Array destructure widening. Match L1/L2/L3 module style.
4. MIRRORS: if mirror_changes are needed apply them to ${MIRRORGEN}; run \`ruby ${MIRRORGEN}\`; then \`make verify-mirrors\` — MUST exit 0.
5. CPU GATE: rebuild + fingerprint, MUST byte-equal ${baseline.cpu_baseline}.
6. CUDA GATE: rebuild + fingerprint, MUST byte-equal ${baseline.cuda_baseline}.
7. COMMIT (only if BOTH gates pass && verify-mirrors clean && !touched_ungated_path): \`git add\` ONLY the specific changed source + regenerated mirrors (NEVER \`git add -A\`; never the 712MB binaries). Commit title "P2.6: <step_title>". Capture sha.

== On ANY gate failure / un-gated-path crossing ==
Do NOT patch around it. Revert: \`git checkout -- <changed files>\`, \`rm -f\` any new files, confirm \`git status\` clean vs the step's starting HEAD. Report gate_passed=false + precise failure_reason. The loop halts — correct outcome.

gate_passed=true ONLY if cpu_bit_identical && cuda_bit_identical && verify_mirrors_clean && committed && !touched_ungated_path.

Reference docs:
  - ${REFS}`

const results = []
for (let i = 0; i < plan.decomposition.length; i++) {
  const step = plan.decomposition[i]
  const r = await agent(stepPrompt(step, i), { label: `extract:${step.step_title?.slice(0, 28) || ('step' + i)}`, phase: 'Extract', schema: STEP_SCHEMA })
  results.push(r)

  if (!r) { log(`STOP: step ${i + 1} agent returned nothing. Halting.`); break }
  if (r.gate_passed && r.committed) {
    log(`OK step ${i + 1}: "${r.step_title}" committed ${r.commit_sha || '(sha?)'} | CPU+CUDA bit-identical | mirrors clean`)
  } else {
    log(`STOP at step ${i + 1}: "${r.step_title}" — ${r.touched_ungated_path ? 'would touch UN-GATED GGUF path; ' : ''}${r.failure_reason || r.notes}. Reverted; halting.`)
    break
  }
}

const committed = results.filter((r) => r && r.gate_passed && r.committed)

return {
  baseSha,
  cpu_baseline: baseline.cpu_baseline,
  cuda_baseline: baseline.cuda_baseline,
  planned_steps: plan.decomposition.length,
  completed: committed.map((r) => ({ step: r.step_title, sha: r.commit_sha })),
  halted_at: results.length && !results[results.length - 1].gate_passed ? results[results.length - 1].step_title : null,
  deferred: plan.deferred,
  gguf_coverage_handling: plan.gguf_coverage_handling,
  results,
  followup: 'Pushed nothing (land on main, push when ready). Realize-path work is incremental: re-run after building GGUF-path gates to unlock the deferred items. Remaining P2: L4 recipes; then P3 core+CLI.',
}
