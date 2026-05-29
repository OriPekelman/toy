export const meta = {
  name: 'finish-realize-bulk',
  description: 'Decompose the ~1750-line realize-path bulk onto the L3 arch / L2 block, behind a TRIPLE bit-identical gate (CPU + CUDA random_init self-consistency, PLUS the GGUF F32 mmap round-trip), as a sequence of small gated commits; never behaviorally restructure a path no gate covers; stop on first gate failure',
  whenToUse: 'Realize-bulk pass 2: now that the GGUF F32 mmap round-trip gate exists (smoke_gguf_roundtrip), the F32 mmap path is gate-covered and its deferred extractions can land. q8_copy / GQA-divergent / llama3 / qkv_bias / B>1 remain un-gated.',
  phases: [
    { title: 'Baseline', detail: 'capture HEAD + THREE deterministic gate fingerprints (CPU, CUDA, GGUF round-trip)' },
    { title: 'Recon', detail: 'parallel read-only agents map the 4 realize paths, the arch/block seam, per-path gate coverage, mirror gen' },
    { title: 'Plan', detail: 'one architect proposes a conservative incremental triple-gated decomposition' },
    { title: 'Extract', detail: 'serial gated loop; one commit each; triple gate; halt on first failure or any un-gated-path restructure' },
  ],
}

// ---- Constants -------------------------------------------------------------

const MONOLITH = 'lib/llama_seq_forward_ffi.rb'
const MIRRORGEN = 'prep/gen_cuda_mirror.rb'
const ARCH_FILE = 'lib/toy/llm/archs/llama_arch.rb'
const BLOCK_FILE = 'lib/toy/llm/blocks/transformer_block.rb'

// TRIPLE gate. Each fixture is deterministic; each is gated against ITS OWN
// recorded baseline (self-consistency before/after). CPU & CUDA exercise
// realize_for_random_init; the GGUF round-trip exercises realize_for_mmap
// (F32, n_heads*d_head==d_model — no GQA divergence).
const CPU_BUILD = 'make examples/smoke_projection_lens'
const CPU_FP = 'SEED=0 STEPS=5 ./examples/smoke_projection_lens 2>/dev/null | grep -E "^step|^initial=" | sha256sum'
const CUDA_BUILD = 'make examples/smoke_projection_lens_cuda'
const CUDA_FP = 'SEED=0 STEPS=5 ./examples/smoke_projection_lens_cuda 2>/dev/null | grep -E "^step|^initial=" | sha256sum'
const GGUF_BUILD = 'make examples/smoke_gguf_roundtrip'
const GGUF_FP = 'SEED=0 ./examples/smoke_gguf_roundtrip 2>/dev/null | sha256sum'  // expect EXIT=0 + "VERDICT: GGUF F32 mmap round-trip bit-identical"

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29e.md  (checkpoint; the deferred GGUF list + the gate that just unlocked mmap)',
  'lib/toy/llm/archs/llama_arch.rb / blocks/transformer_block.rb  (the owning layers)',
  'lib/toy_gguf_fuse.rb + examples/smoke_gguf_roundtrip.rb  (the new GGUF F32 mmap round-trip gate)',
  'docs/roadmap/toy-framework-design-2026-05-28.md  (design §6)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (#16 Struct.new — matz/spinel#1043)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/project_step_bind_landmine_2026_05_28.md',
].join('\n  - ')

const FRAMING = `CRITICAL FRAMING.
(1) SIZE + INCREMENT. The four realize_for_* methods are ~1750 lines, intertwined. Decompose into the SMALLEST safely-gated units; land each as its own gated commit. A PARTIAL but safe result is the CORRECT outcome — do not over-reach.
(2) TRIPLE GATE. Every commit must reproduce ALL THREE fixture fingerprints byte-identically: CPU + CUDA (smoke_projection_lens, random_init) AND the GGUF round-trip (smoke_gguf_roundtrip, realize_for_mmap F32). Each backend/fixture is compared against its OWN recorded baseline.
(3) COVERAGE — the load-bearing constraint, UPDATED. Now GATED: realize_for_random_init (CPU+CUDA) AND realize_for_mmap on the F32 path with n_heads*d_head==d_model (the GGUF round-trip). So you MAY now extract shared logic that the mmap-F32 round-trip exercises end-to-end. STILL UN-GATED (do NOT behaviorally restructure without first building a gate): realize_for_q8_copy (no Q8 gate exists), GQA-DIVERGENT attn_output shapes ([d_model,d_model] vs [n_heads*d_head] — the round-trip fixture pins them equal, so the gate is BLIND to the divergence), llama3 rope_freq_factors branches, qkv_bias branches (gate has qkv_bias=false), the B>1 attn-mask BODY (gates are B=1), and realize_for_full_finetune's GGUF-LOAD half (the round-trip gates mmap reload, NOT the ft GGUF-load path). If a step would touch any of those, set touched_ungated_path=true and HALT.`

// ---- Schemas ---------------------------------------------------------------

const GATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['cpu_deterministic', 'cpu_baseline', 'cuda_deterministic', 'cuda_baseline', 'gguf_deterministic', 'gguf_baseline', 'notes'],
  properties: {
    cpu_deterministic: { type: 'boolean' },
    cpu_baseline: { type: 'string' },
    cuda_deterministic: { type: 'boolean' },
    cuda_baseline: { type: 'string' },
    gguf_deterministic: { type: 'boolean', description: 'two runs of smoke_gguf_roundtrip byte-identical AND exit 0 with the bit-identical VERDICT' },
    gguf_baseline: { type: 'string', description: 'sha256 of smoke_gguf_roundtrip stdout (expect c89fd3eb…)' },
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
    shared_vs_specific: { type: 'string', description: 'what logic is SHARED across paths and which gate covers it (random_init via CPU/CUDA, mmap-F32 via GGUF round-trip) vs SPECIFIC to an un-gated path' },
    fixture_coverage: { type: 'string', description: 'which realize path each gate exercises now; what remains un-gated (q8, GQA-divergent, llama3, qkv_bias, B>1, ft-GGUF-load)' },
    risks: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['feasible', 'decomposition', 'deferred', 'mirror_changes', 'parity_risks'],
  properties: {
    feasible: { type: 'boolean' },
    decomposition: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['step_title', 'what_moves', 'paths_touched', 'gate_covered_by', 'risk'],
        properties: {
          step_title: { type: 'string' },
          what_moves: { type: 'string' },
          paths_touched: { type: 'array', items: { type: 'string' } },
          gate_covered_by: { type: 'array', items: { type: 'string' }, description: 'which gate(s) actually exercise everything this step changes: any of cpu/cuda/gguf. Empty = NOT gate-covered → must not run.' },
          risk: { type: 'string' },
        },
      },
    },
    deferred: { type: 'array', items: { type: 'string' } },
    mirror_changes: { type: 'string' },
    parity_risks: { type: 'array', items: { type: 'string' } },
  },
}

const STEP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['step_title', 'gate_passed', 'cpu_bit_identical', 'cuda_bit_identical', 'gguf_bit_identical', 'verify_mirrors_clean', 'committed', 'touched_ungated_path', 'notes'],
  properties: {
    step_title: { type: 'string' },
    gate_passed: { type: 'boolean', description: 'true ONLY if cpu+cuda+gguf bit-identical && verify_mirrors_clean && committed && !touched_ungated_path' },
    cpu_bit_identical: { type: 'boolean' },
    cuda_bit_identical: { type: 'boolean' },
    gguf_bit_identical: { type: 'boolean' },
    verify_mirrors_clean: { type: 'boolean' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    touched_ungated_path: { type: 'boolean' },
    fingerprints: { type: 'string' },
    failure_reason: { type: 'string' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: Baseline (triple) --------------------------------------------

phase('Baseline')

const baseline = await agent(`Set up the TRIPLE parity gate. Repo root, host gx10 (GB10 GPU). Three fixtures, all must be confirmed deterministic:

  CPU:  \`${CPU_BUILD}\`; run \`${CPU_FP}\` TWICE → byte-identical; record cpu_baseline.
  CUDA: \`${CUDA_BUILD}\` (slow ~712MB link); run \`${CUDA_FP}\` TWICE → byte-identical; record cuda_baseline.
  GGUF: \`${GGUF_BUILD}\`; run \`${GGUF_FP}\` TWICE → byte-identical AND exit 0 with final line "VERDICT: GGUF F32 mmap round-trip bit-identical"; record gguf_baseline (expect c89fd3eb…).

CPU/CUDA/GGUF fingerprints differ from each other — expected; each is gated against its own baseline. Do NOT edit any file.`, { label: 'triple-gate', phase: 'Baseline', schema: GATE_SCHEMA })

const shaRaw = await agent(`Run \`git rev-parse HEAD\`, then \`git status --porcelain --untracked-files=no\`. Report the 40-char HEAD sha. End with "CLEAN <sha>" if no tracked changes (untracked binaries fine), else "DIRTY <sha>".`, { label: 'base-sha', phase: 'Baseline' })
const baseSha = ((shaRaw || '').match(/\b([0-9a-f]{40})\b/) || [])[1] || ''
const treeDirty = /\bDIRTY\b/.test(shaRaw || '')

log(`HEAD ${baseSha || '(unparsed)'} | tree ${treeDirty ? 'DIRTY' : 'clean'}`)
log(`CPU ${baseline?.cpu_baseline} det=${baseline?.cpu_deterministic} | CUDA ${baseline?.cuda_baseline} det=${baseline?.cuda_deterministic} | GGUF ${baseline?.gguf_baseline} det=${baseline?.gguf_deterministic}`)

if (!baseline || !baseline.cpu_deterministic || !baseline.cuda_deterministic || !baseline.gguf_deterministic) {
  log('ABORT: one of the three gates not deterministic.')
  return { aborted: true, reason: 'gate not deterministic', baseline }
}
if (!baseSha || treeDirty) { log('ABORT: dirty tree / bad HEAD.'); return { aborted: true, reason: 'dirty tree', baseSha } }

// ---- Phase 1: Recon --------------------------------------------------------

phase('Recon')

const FACETS = [
  { key: 'four-paths', prompt: `Map the four realize paths in ${MONOLITH} (q8_copy ~236, mmap ~533, full_finetune ~838, random_init ~1081), accounting for the 4 already-landed extractions (apply_seq_cfg!, arch.seed_blocks!, finalize_and_realize!, block.alloc_trainable_f32_weights!). KEY (shared_vs_specific): what remains SHARED and which gate covers it — random_init (CPU/CUDA) and now mmap-F32 (GGUF round-trip) — vs what is SPECIFIC to an un-gated path.` },
  { key: 'arch-block-seam', prompt: `Map remaining realize logic that could move onto the L3 arch / L2 block (design §6: arch owns whole-graph alloc, block owns its weight loading). Focus on what the GGUF mmap-F32 round-trip now lets us safely move (e.g. mmap per-block weight LOAD onto block.load_from_gguf_mmap!). Note ivars read/written.` },
  { key: 'coverage', prompt: `Confirm per-path gate coverage now: random_init=CPU+CUDA, mmap-F32(n_heads*d_head==d_model)=GGUF round-trip. EXPLICITLY list what stays un-gated: q8_copy, GQA-divergent attn_output shape (round-trip pins it equal → BLIND), llama3 rope branches, qkv_bias branches, B>1 mask body, full_finetune GGUF-load half. For each candidate extraction, which gate(s) exercise it end-to-end?` },
  { key: 'mirror-gen', prompt: `Confirm mirror situation. If logic moves onto arch/block (already in MIRRORABLE) mirrors regenerate automatically. A NEW file (e.g. a shared mmap-load helper, or moving toy_gguf_fuse-style logic) may need a MIRRORABLE entry IF it's used on the GPU path. Note Spinel name-collision hazards (landmine #16).` },
]

const reconArr = await parallel(FACETS.map((f) => () =>
  agent(`READ-ONLY recon for realize-bulk pass 2 (triple-gated). ${FRAMING}\n\nFacet: ${f.prompt}\n\nReport current file:line refs (grep). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: `recon:${f.key}`, phase: 'Recon', schema: RECON_SCHEMA })))
const recon = {}; FACETS.forEach((f, i) => { recon[f.key] = reconArr[i] })
for (const f of FACETS) { const r = recon[f.key]; log(`recon ${f.key}: ${r ? 'ok (' + (r.risks?.length || 0) + ' risks)' : 'FAILED'}`) }

// ---- Phase 2: Plan ---------------------------------------------------------

phase('Plan')

const plan = await agent(`Architect realize-bulk pass 2: a CONSERVATIVE, INCREMENTAL, TRIPLE-GATED decomposition. ${FRAMING}\n\n== Recon ==\n${JSON.stringify(recon, null, 2)}\n\nProduce decomposition: ordered SMALL steps. For each set gate_covered_by (which of cpu/cuda/gguf actually exercise everything it changes); a step with empty gate_covered_by MUST NOT be planned (defer it). Prefer mmap-F32 logic now unlocked by the GGUF gate + any remaining random_init-covered shared logic. Keep q8/GQA-divergent/llama3/qkv_bias/B>1/ft-GGUF-load in 'deferred'. Bias to SAFETY: feasible=false (empty decomposition) if nothing new is safely extractable. No Struct.new (#16). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: 'realize-plan-v2', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan || !plan.feasible || !plan.decomposition || plan.decomposition.length === 0) {
  log(`STOP: no newly-gated extraction available — ${plan ? (plan.parity_risks || []).join('; ') : 'planner failed'}. Valid outcome.`)
  return { aborted: true, reason: 'no safely-gated steps', baseSha, recon, plan }
}
log(`Plan: ${plan.decomposition.length} step(s). Deferred: ${(plan.deferred || []).length}.`)

// ---- Phase 3: Serial triple-gated extraction loop --------------------------

phase('Extract')

const stepPrompt = (step, idx) => `Execute realize-bulk pass-2 STEP ${idx + 1}/${plan.decomposition.length}: "${step.step_title}". STRICTLY TRIPLE-GATED, SINGLE COMMIT. The ONLY acceptable outcome is byte-identical CPU, CUDA, AND GGUF round-trip output. ${FRAMING}

== This step ==
${JSON.stringify(step, null, 2)}

== Triple gate ==
CPU:  \`${CPU_BUILD}\`; \`${CPU_FP}\`; baseline ${baseline.cpu_baseline}
CUDA: \`${CUDA_BUILD}\`; \`${CUDA_FP}\`; baseline ${baseline.cuda_baseline}
GGUF: \`${GGUF_BUILD}\`; \`${GGUF_FP}\`; baseline ${baseline.gguf_baseline} (must also exit 0 + bit-identical VERDICT)

== Procedure ==
1. GUARD: if this step would change logic an existing gate does NOT cover (q8/GQA-divergent/llama3/qkv_bias/B>1-body/ft-GGUF-load — see FRAMING), STOP: set touched_ungated_path=true, gate_passed=false, do nothing, explain.
2. CAPTURE BASELINES: build + fingerprint ALL THREE, confirm == recorded baselines. If any mismatches pre-edit, STOP (failure_reason="baseline mismatch pre-edit").
3. APPLY the change per what_moves. No Struct.new (#16); hand-written class, positional ctor, no default args, uniquely-prefixed members; no ivar widening; match L1/L2/L3 style.
4. MIRRORS: if needed apply mirror_changes to ${MIRRORGEN}; \`ruby ${MIRRORGEN}\`; \`make verify-mirrors\` (exit 0).
5. CPU GATE → MUST == ${baseline.cpu_baseline}.
6. CUDA GATE → MUST == ${baseline.cuda_baseline}.
7. GGUF GATE → MUST == ${baseline.gguf_baseline} (and exit 0 + VERDICT).
8. COMMIT (only if all three pass && verify-mirrors clean && !touched_ungated_path): \`git add\` ONLY the changed source + regenerated mirrors (NEVER -A; never binaries). Title "P2.6: <step_title>". Capture sha.

== On ANY gate failure / un-gated crossing ==
Revert cleanly (\`git checkout --\`, \`rm -f\` new files; clean vs the step's starting HEAD). gate_passed=false + failure_reason. The loop halts — correct.

gate_passed=true ONLY if cpu_bit_identical && cuda_bit_identical && gguf_bit_identical && verify_mirrors_clean && committed && !touched_ungated_path.

Reference docs:
  - ${REFS}`

const results = []
for (let i = 0; i < plan.decomposition.length; i++) {
  const step = plan.decomposition[i]
  if (!step.gate_covered_by || step.gate_covered_by.length === 0) {
    log(`SKIP step ${i + 1} "${step.step_title}": plan marked it not gate-covered. Halting (plan inconsistency).`)
    results.push({ step_title: step.step_title, gate_passed: false, committed: false, touched_ungated_path: true, notes: 'planned with empty gate_covered_by' })
    break
  }
  const r = await agent(stepPrompt(step, i), { label: `extract:${step.step_title?.slice(0, 26) || ('step' + i)}`, phase: 'Extract', schema: STEP_SCHEMA })
  results.push(r)
  if (!r) { log(`STOP: step ${i + 1} returned nothing.`); break }
  if (r.gate_passed && r.committed) {
    log(`OK step ${i + 1}: "${r.step_title}" ${r.commit_sha || ''} | CPU+CUDA+GGUF bit-identical | mirrors clean`)
  } else {
    log(`STOP at step ${i + 1}: "${r.step_title}" — ${r.touched_ungated_path ? 'un-gated path; ' : ''}${r.failure_reason || r.notes}. Reverted; halting.`)
    break
  }
}

const committed = results.filter((r) => r && r.gate_passed && r.committed)
return {
  baseSha,
  gates: { cpu: baseline.cpu_baseline, cuda: baseline.cuda_baseline, gguf: baseline.gguf_baseline },
  planned_steps: plan.decomposition.length,
  completed: committed.map((r) => ({ step: r.step_title, sha: r.commit_sha })),
  halted_at: results.length && !results[results.length - 1].gate_passed ? results[results.length - 1].step_title : null,
  deferred: plan.deferred,
  results,
  followup: 'Pushed nothing (land on main, push when ready). Still deferred: q8_copy (needs Q8 round-trip gate), GQA-divergent shapes (need a GQA fixture), llama3/qkv_bias/B>1 (need those fixtures), ft-GGUF-load. Then L4 recipes.',
}
