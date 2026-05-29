export const meta = {
  name: 'finish-realize-bulk',
  description: 'Decompose the realize-path bulk onto the L3 arch / L2 block behind an N-WAY bit-identical gate (cpu, cuda, gguf-mmap, gqa-divergent, B>1, qkv_bias, q8 — every realize path now has a deterministic fixture), as small gated commits; never behaviorally restructure a path no gate covers; stop on first gate failure.',
  whenToUse: 'Realize-bulk pass 3 (full-cascade): all config-variant + q8 gates exist; unlock the previously-deferred slices.',
  phases: [
    { title: 'Baseline', detail: 'capture HEAD + all N gate fingerprints; confirm each deterministic' },
    { title: 'Recon', detail: 'parallel read-only: 4 realize paths, arch/block seam, per-path gate coverage NOW, mirror gen' },
    { title: 'Plan', detail: 'conservative incremental N-gated decomposition; gate_covered_by per step' },
    { title: 'Extract', detail: 'serial gated loop; one commit each; N-way gate; halt on first failure or un-gated restructure' },
  ],
}

const MONOLITH = 'lib/llama_seq_forward_ffi.rb'
const MIRRORGEN = 'prep/gen_cuda_mirror.rb'
const ARCH_FILE = 'lib/toy/llm/archs/llama_arch.rb'
const BLOCK_FILE = 'lib/toy/llm/blocks/transformer_block.rb'

// The full gate set. Each is deterministic + gated against its OWN baseline
// (self-consistency before/after). `exercises` documents which realize
// path/branch each one actually covers (drives the coverage rules).
const GATES = [
  { key: 'cpu',           build: 'make examples/smoke_projection_lens',          fp: 'SEED=0 STEPS=5 ./examples/smoke_projection_lens 2>/dev/null | grep -E "^step|^initial=" | sha256sum',          exercises: 'realize_for_random_init forward+train, CPU' },
  { key: 'cuda',          build: 'make examples/smoke_projection_lens_cuda',     fp: 'SEED=0 STEPS=5 ./examples/smoke_projection_lens_cuda 2>/dev/null | grep -E "^step|^initial=" | sha256sum',     exercises: 'realize_for_random_init forward+train, GB10 GPU (CUDA mirror)' },
  { key: 'gguf',          build: 'make examples/smoke_gguf_roundtrip',           fp: 'SEED=0 ./examples/smoke_gguf_roundtrip 2>/dev/null | sha256sum',                                              exercises: 'realize_for_mmap F32 reload, n_heads*head_dim==d_model (NON-divergent)' },
  { key: 'gqa_divergent', build: 'make examples/smoke_gate_gqa_divergent',       fp: 'STEPS=5 SEED=0 ./examples/smoke_gate_gqa_divergent 2>/dev/null | sha256sum',                                  exercises: 'realize_for_random_init + block w_o with n_heads*head_dim != d_model (DIVERGENT shape)' },
  { key: 'b_gt_1',        build: 'make examples/smoke_gate_b_gt_1',              fp: 'STEPS=5 SEED=0 ./examples/smoke_gate_b_gt_1 2>/dev/null | sha256sum',                                         exercises: 'realize_for_random_init with seq_b>1 → B>1 attn-mask body (soft_max_ext + upload_block_causal_mask!)' },
  { key: 'qkv_bias',      build: 'make examples/smoke_gate_qkv_bias',           fp: './examples/smoke_gate_qkv_bias 2>/dev/null | sha256sum',                                                     exercises: 'realize_for_mmap on Qwen2.5-0.5B with qkv_bias=true → qkv_bias slice branches' },
  { key: 'q8',            build: 'make examples/smoke_gate_q8_preserve',        fp: './examples/smoke_gate_q8_preserve 2>/dev/null | sha256sum',                                                  exercises: 'realize_for_q8_copy on Qwen2.5-0.5B-Q8 → Q8-stays-Q8 typed-copy path' },
]

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29h.md  (deferred realize slices)',
  'lib/toy/llm/archs/llama_arch.rb / blocks/transformer_block.rb',
  'docs/roadmap/toy-framework-design-2026-05-28.md (design §6)',
  'examples/smoke_gate_*.rb  (the new gate fixtures + their assertions — what each proves)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (#16)',
].join('\n  - ')

const STILL_UNGATED = `STILL UN-GATED even now (do NOT behaviorally restructure these without a new gate):
 - GQA-divergent on the MMAP path specifically (gqa_divergent gate is random_init; the gguf round-trip PINS n_heads*head_dim==d_model — so realize_for_mmap's w_o under divergence is still blind).
 - llama3 rope_freq_factors realize branches (alloc/upload in realize_for_*): the toy forward is rope-ANGLE-insensitive at the logit level (established), and the llama3_tensor gate only covers the standalone RoPE primitive — NOT the realize wiring. So these branches have NO end-to-end gate.
 - tied-output (untied=false) branches NOT covered by a gate where untied=true (note: qkv_bias + q8 gates load Qwen TIED, so tied IS now partially covered there — confirm per-branch).
 - full_finetune GGUF-load half (ft_load_from_gguf dequant-slice path) — distinct from mmap reload; the gguf gate does NOT cover it.`

const FRAMING = `CRITICAL FRAMING.
(1) INCREMENT. Smallest safely-gated units; one gated commit each; partial-but-safe is correct.
(2) N-WAY GATE. Every commit must reproduce ALL gate fingerprints below byte-identically (each vs its OWN baseline). Building/running all gates per step is slow (CUDA 712MB link, Qwen 0.9-2GB loads) — that's expected; be patient.
(3) COVERAGE — now MUCH wider. Gated paths: random_init (cpu+cuda, incl. GQA-divergent + B>1 variants), mmap-F32-nondivergent (gguf), mmap-Qwen-qkv_bias (qkv_bias), q8_copy (q8). You MAY now extract the previously-deferred slices these cover: realize_for_q8_copy body, qkv_bias branches, B>1 mask body, GQA-divergent w_o (random_init/block side), per-block mmap LOAD (non-divergent). ${STILL_UNGATED}
If a step would touch a STILL-UN-GATED item, set touched_ungated_path=true and HALT.`

const GATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['baselines', 'all_deterministic', 'notes'],
  properties: {
    baselines: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['key', 'fingerprint', 'deterministic'],
      properties: { key: { type: 'string' }, fingerprint: { type: 'string' }, deterministic: { type: 'boolean' } },
    } },
    all_deterministic: { type: 'boolean' },
    notes: { type: 'string' },
  },
}
const RECON_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['facet', 'findings', 'line_refs', 'risks'],
  properties: { facet: { type: 'string' }, findings: { type: 'string' }, line_refs: { type: 'array', items: { type: 'string' } },
    shared_vs_specific: { type: 'string' }, coverage_now: { type: 'string' }, risks: { type: 'array', items: { type: 'string' } } },
}
const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['feasible', 'decomposition', 'deferred', 'parity_risks'],
  properties: {
    feasible: { type: 'boolean' },
    decomposition: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['step_title', 'what_moves', 'paths_touched', 'gate_covered_by', 'risk'],
      properties: { step_title: { type: 'string' }, what_moves: { type: 'string' }, paths_touched: { type: 'array', items: { type: 'string' } },
        gate_covered_by: { type: 'array', items: { type: 'string' }, description: 'which gate keys exercise everything this step changes; empty = not covered → must not run' }, risk: { type: 'string' } } } },
    deferred: { type: 'array', items: { type: 'string' } },
    mirror_changes: { type: 'string' },
    parity_risks: { type: 'array', items: { type: 'string' } },
  },
}
const STEP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['step_title', 'gate_passed', 'all_gates_bit_identical', 'verify_mirrors_clean', 'committed', 'touched_ungated_path', 'notes'],
  properties: { step_title: { type: 'string' }, gate_passed: { type: 'boolean' },
    all_gates_bit_identical: { type: 'boolean', description: 'every gate fingerprint matched its baseline' },
    per_gate: { type: 'string', description: 'gate key → match/mismatch summary' },
    verify_mirrors_clean: { type: 'boolean' }, committed: { type: 'boolean' }, commit_sha: { type: 'string' },
    touched_ungated_path: { type: 'boolean' }, failure_reason: { type: 'string' }, notes: { type: 'string' } },
}

// ---- Phase 0: Baseline (N-way) ---------------------------------------------

phase('Baseline')

const gateList = GATES.map((g) => `  ${g.key}: build \`${g.build}\`; fingerprint \`${g.fp}\`  [${g.exercises}]`).join('\n')

const baseline = await agent(`Set up the N-WAY parity gate for the realize refactor. Repo root, host gx10 (GB10 GPU). For EACH gate: build it, run its fingerprint pipe TWICE, confirm byte-identical (deterministic), record the sha. CUDA + Qwen/q8 builds/loads are slow — be patient. Gates:\n${gateList}\n\nReport baselines [{key, fingerprint, deterministic}] for all ${GATES.length}. all_deterministic=true only if every gate is deterministic. Do NOT edit any file.`, { label: 'nway-gate', phase: 'Baseline', schema: GATE_SCHEMA })

const shaRaw = await agent(`Run \`git rev-parse HEAD\` then \`git status --porcelain --untracked-files=no\`. Report the 40-char sha. End with "CLEAN <sha>" or "DIRTY <sha>".`, { label: 'base-sha', phase: 'Baseline' })
const baseSha = ((shaRaw || '').match(/\b([0-9a-f]{40})\b/) || [])[1] || ''
if (!baseSha || /\bDIRTY\b/.test(shaRaw || '')) { log('ABORT: dirty tree / bad HEAD.'); return { aborted: true, reason: 'dirty tree', baseSha } }
if (!baseline || !baseline.all_deterministic) { log('ABORT: not all gates deterministic.'); return { aborted: true, reason: 'gate not deterministic', baseline } }
const baselineStr = (baseline.baselines || []).map((b) => `  ${b.key} = ${b.fingerprint}`).join('\n')
log(`HEAD ${baseSha} | ${(baseline.baselines || []).length} gates baselined, all deterministic`)

// ---- Phase 1: Recon --------------------------------------------------------

phase('Recon')

const FACETS = [
  { key: 'four-paths', prompt: `Map the 4 realize paths in ${MONOLITH} (q8_copy ~236, mmap ~533, full_finetune ~838, random_init ~1081) accounting for the 5 already-landed extractions. shared_vs_specific: what's shared vs path-specific. coverage_now: with the full gate set (cpu/cuda/gguf/gqa_divergent/b_gt_1/qkv_bias/q8), which previously-deferred slices are NOW gate-covered end-to-end?` },
  { key: 'arch-block-seam', prompt: `Map remaining realize logic that can move onto the L3 arch / L2 block (design §6). With q8/qkv_bias/B>1/GQA-divergent now gated, what's the next safe extraction (e.g. per-block mmap LOAD onto block.load_from_gguf_mmap!, q8 typed-copy load, qkv_bias bias load)? Note ivars read/written.` },
  { key: 'coverage-map', prompt: `Produce the precise per-slice coverage map: for EACH still-deferred slice (q8_copy body, mmap LOAD loop, qkv_bias branches, B>1 body, GQA-divergent w_o, llama3 rope, ft-GGUF-load, tied-output), state which gate(s) NOW cover it end-to-end vs which remain un-gated. Be rigorous — this drives what's safe to touch. Recall: gqa_divergent is random_init (NOT mmap); gguf pins non-divergent; llama3 realize wiring is un-gated (rope-insensitive forward).` },
  { key: 'mirror-gen', prompt: `If logic moves onto arch/block (in MIRRORABLE) mirrors regenerate. A NEW shared helper used on the GPU path needs a MIRRORABLE entry. Note Spinel name-collision hazards (#16).` },
]
const reconArr = await parallel(FACETS.map((f) => () =>
  agent(`READ-ONLY recon for realize-bulk pass 3 (N-way gated). ${FRAMING}\n\nFacet: ${f.prompt}\n\nReport current file:line refs (grep). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: `recon:${f.key}`, phase: 'Recon', schema: RECON_SCHEMA })))
const recon = {}; FACETS.forEach((f, i) => { recon[f.key] = reconArr[i] })
for (const f of FACETS) { const r = recon[f.key]; log(`recon ${f.key}: ${r ? 'ok' : 'FAILED'}`) }

// ---- Phase 2: Plan ---------------------------------------------------------

phase('Plan')

const plan = await agent(`Architect realize-bulk pass 3: a CONSERVATIVE, INCREMENTAL, N-gated decomposition. ${FRAMING}\n\n== Recon ==\n${JSON.stringify(recon, null, 2)}\n\nOrdered SMALL steps; gate_covered_by per step lists the gate keys that actually exercise everything it changes (empty → defer). Prefer the slices newly unlocked by q8/qkv_bias/B>1/GQA-divergent/mmap. Keep STILL-UN-GATED items in 'deferred'. feasible=false if nothing new is safely extractable. No Struct.new (#16). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: 'realize-plan-v3', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan || !plan.feasible || !plan.decomposition || plan.decomposition.length === 0) {
  log(`STOP: no newly-gated extraction — ${plan ? (plan.parity_risks || []).join('; ') : 'planner failed'}. Valid outcome.`)
  return { aborted: true, reason: 'no safely-gated steps', baseSha, recon, plan }
}
log(`Plan: ${plan.decomposition.length} step(s). Deferred: ${(plan.deferred || []).length}.`)

// ---- Phase 3: Serial N-gated extraction loop -------------------------------

phase('Extract')

const stepPrompt = (step, idx) => `Execute realize-bulk pass-3 STEP ${idx + 1}/${plan.decomposition.length}: "${step.step_title}". STRICTLY N-WAY GATED, SINGLE COMMIT. ${FRAMING}\n\n== This step ==\n${JSON.stringify(step, null, 2)}\n\n== Gates (rebuild + fingerprint EACH; compare to baseline) ==\n${GATES.map((g) => `  ${g.key}: \`${g.build}\` then \`${g.fp}\``).join('\n')}\n== Baselines ==\n${baselineStr}\n\n== Procedure ==\n1. GUARD: if this step would change a STILL-UN-GATED item (see FRAMING), STOP: touched_ungated_path=true, gate_passed=false, do nothing, explain.\n2. CAPTURE BASELINES: build + fingerprint ALL ${GATES.length} gates, confirm == the baselines above. Any pre-edit mismatch → STOP (failure_reason="baseline mismatch pre-edit").\n3. APPLY the change per what_moves. No Struct.new (#16); hand-written class, positional ctor, prefixed members; match L1/L2/L3 style.\n4. MIRRORS: if needed apply mirror_changes to ${MIRRORGEN}; \`ruby ${MIRRORGEN}\`; \`make verify-mirrors\` (exit 0).\n5. RE-RUN ALL ${GATES.length} GATES → each MUST byte-equal its baseline. Record per_gate.\n6. COMMIT (only if all gates pass && verify-mirrors clean && !touched_ungated_path): \`git add\` ONLY changed source + regenerated mirrors (NEVER -A; no binaries/gguf). Title "P2.7: <step_title>". Capture sha.\n\nON ANY FAILURE: revert (\`git checkout --\`, \`rm -f\` new files; clean vs the step's starting HEAD). gate_passed=false + failure_reason. Loop halts.\n\ngate_passed=true ONLY if all_gates_bit_identical && verify_mirrors_clean && committed && !touched_ungated_path. Reference docs:\n  - ${REFS}`

const results = []
for (let i = 0; i < plan.decomposition.length; i++) {
  const step = plan.decomposition[i]
  if (!step.gate_covered_by || step.gate_covered_by.length === 0) {
    log(`SKIP step ${i + 1} "${step.step_title}": not gate-covered (plan). Halting.`)
    results.push({ step_title: step.step_title, gate_passed: false, committed: false, touched_ungated_path: true, notes: 'empty gate_covered_by' }); break
  }
  const r = await agent(stepPrompt(step, i), { label: `extract:${step.step_title?.slice(0, 24) || ('step' + i)}`, phase: 'Extract', schema: STEP_SCHEMA })
  results.push(r)
  if (!r) { log(`STOP: step ${i + 1} returned nothing.`); break }
  if (r.gate_passed && r.committed) log(`OK step ${i + 1}: "${r.step_title}" ${r.commit_sha || ''} | all ${GATES.length} gates bit-identical`)
  else { log(`STOP at step ${i + 1}: "${r.step_title}" — ${r.touched_ungated_path ? 'un-gated path; ' : ''}${r.failure_reason || r.notes}. Reverted; halting.`); break }
}

const committed = results.filter((r) => r && r.gate_passed && r.committed)
return {
  baseSha,
  gates: (baseline.baselines || []).map((b) => b.key),
  planned_steps: plan.decomposition.length,
  completed: committed.map((r) => ({ step: r.step_title, sha: r.commit_sha })),
  halted_at: results.length && !results[results.length - 1].gate_passed ? results[results.length - 1].step_title : null,
  deferred: plan.deferred,
  results,
  followup: 'Realize-bulk pass 3 (full gate set). Remaining truly-un-gated: GQA-divergent MMAP, llama3 realize wiring (rope-insensitive forward), ft-GGUF-load. Then P3 core+CLI.',
}
