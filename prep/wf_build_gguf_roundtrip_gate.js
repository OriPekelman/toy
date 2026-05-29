export const meta = {
  name: 'build-gguf-roundtrip-gate',
  description: 'Build a deterministic GGUF round-trip gate (random_init -> head-fusing GGUF write -> realize_for_mmap reload -> forward, logits reproduced bit-identically) so the GGUF-loading realize paths (mmap, full_finetune GGUF-load half) become parity-gateable. Halt if infeasible; do not fake a gate.',
  whenToUse: 'Before the next realize-bulk pass: the only existing gate exercises realize_for_random_init; the GGUF paths are un-gated. This builds the missing gate.',
  phases: [
    { title: 'Baseline', detail: 'record the in-memory random_init forward logits (the round-trip target) + HEAD' },
    { title: 'Recon', detail: 'parallel read-only: writer caps, the per-head vs fused-name mismatch, mmap reload expectations, existing round-trip smokes' },
    { title: 'Plan', detail: 'decide the head-fusing approach + fixture design; feasible flag' },
    { title: 'Build', detail: 'implement + verify deterministic round-trip parity; commit; halt if infeasible or parity fails' },
  ],
}

// ---- Constants -------------------------------------------------------------

const MONOLITH = 'lib/llama_seq_forward_ffi.rb'
const WRITER = 'lib/toy_gguf_writer.rb'
const MIRRORGEN = 'prep/gen_cuda_mirror.rb'

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29e.md  (authoritative; the GGUF coverage gap + this gate is the named unlock)',
  'lib/toy_gguf_writer.rb                       (ToyGGUFWriter — appears F32-only; confirm)',
  'lib/llama_seq_forward_ffi.rb                 (realize_for_random_init names per-head; realize_for_mmap/_q8_copy look up FUSED names + head-slice)',
  'examples/smoke_projection_lens.rb            (the random_init fixture shape to model the round-trip after)',
  'lib/toy/llm/{archs,blocks,primitives}/       (post-refactor layers — the forward path the reload must reproduce)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (#16 Struct.new -> matz/spinel#1043; no Struct.new)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_direct_loader_api_design.md',
].join('\n  - ')

const PROBLEM = `THE PROBLEM THIS GATE SOLVES. The dual gate (smoke_projection_lens CPU+CUDA) only exercises realize_for_random_init, so the GGUF-loading realize paths (realize_for_mmap, realize_for_q8_copy, the GGUF-load half of realize_for_full_finetune) are NOT behaviorally gated — the realize-bulk refactor had to defer all of them. The blocker (established by the realize-bulk recon): realize_for_random_init names attention weights PER-HEAD ('blk.N.attn_q.head_H.weight') while realize_for_mmap looks up FUSED names ('blk.N.attn_q.weight') and head-slices. So a naive random_init -> ToyGGUFWriter -> realize_for_mmap round-trip FAILS at tnn_gguf_find_index. GOAL: build a deterministic round-trip where a random_init model is written to a GGUF with FUSED attention-weight names (head-fusing on write) and reloaded via realize_for_mmap, and the reloaded forward reproduces the in-memory random_init forward's logits. Because reload uses the SAME f32 weights through the SAME forward graph, the reload logits MUST be bit-identical to the in-memory ones if the round-trip is faithful — THAT bit-identity (deterministic, repeatable) is what makes this a valid gate. SCOPE: F32 mmap path first (ToyGGUFWriter looks F32-only). Q8 stays out (needs quantize-on-write) and is explicitly NOT in scope.`

const FRAMING = `RULES. (1) The deliverable is a GATE — it must be HONEST. If the round-trip can't be made to reproduce the in-memory logits bit-identically (e.g. head-fusing changes weight order, mmap stride math differs, dtype mismatch), DO NOT paper over it with tolerances or by changing the forward — report infeasible with the precise blocker. A fake/loose gate is worse than none. (2) Don't paint the loader into an inference-only corner (see feedback_direct_loader_api_design): the head-fusing writer/helper should stay Mat-roundtrip-friendly and Ruby-like. (3) Spinel hygiene: no Struct.new (landmine #16), positional ctors, no default args, uniquely-prefixed members. (4) If you add/modify a Spinel-compiled lib used on the GPU path, regenerate mirrors + verify-mirrors. (5) Small, one focused commit.`

// ---- Schemas ---------------------------------------------------------------

const BASE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['logits_fingerprint', 'deterministic', 'how', 'notes'],
  properties: {
    logits_fingerprint: { type: 'string', description: 'a stable fingerprint of the in-memory random_init forward output at a fixed small config+seed (the round-trip target). Reuse smoke_projection_lens losses if they suffice, or a direct logits dump.' },
    deterministic: { type: 'boolean', description: 'two runs of the in-memory forward produced byte-identical fingerprints' },
    how: { type: 'string', description: 'exact command(s) used to produce the fingerprint' },
    notes: { type: 'string' },
  },
}

const RECON_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['facet', 'findings', 'line_refs', 'blockers'],
  properties: {
    facet: { type: 'string' },
    findings: { type: 'string' },
    line_refs: { type: 'array', items: { type: 'string' } },
    blockers: { type: 'array', items: { type: 'string' }, description: 'concrete obstacles to a bit-identical round-trip discovered in this facet' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['feasible', 'fusing_approach', 'fixture_design', 'files_touched', 'parity_argument', 'risks'],
  properties: {
    feasible: { type: 'boolean', description: 'true only if a BIT-IDENTICAL deterministic round-trip looks achievable on the F32 mmap path' },
    fusing_approach: { type: 'string', description: 'how per-head attention weights become fused GGUF tensors on write (and how the reloaded fused tensor reproduces the per-head forward math exactly)' },
    fixture_design: { type: 'string', description: 'the new fixture: build random_init small model -> forward (record logits) -> write GGUF (fused names) -> realize_for_mmap -> forward -> assert bit-identical. Name + where it lives + its env knobs (fixed SEED).' },
    files_touched: { type: 'array', items: { type: 'string' } },
    parity_argument: { type: 'string', description: 'WHY the reloaded forward should be bit-identical (same f32 bytes, same graph). Identify any step that could perturb bits (fusing reorder, padding, stride).' },
    new_gate_cmd: { type: 'string', description: 'the command that will BE the new gate once built' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gate_built', 'roundtrip_bit_identical', 'deterministic', 'committed', 'notes'],
  properties: {
    gate_built: { type: 'boolean', description: 'true ONLY if the round-trip reproduces the in-memory logits bit-identically, repeatably, AND is committed' },
    roundtrip_bit_identical: { type: 'boolean', description: 'reloaded-via-mmap forward == in-memory random_init forward, byte-for-byte' },
    deterministic: { type: 'boolean', description: 'the round-trip fixture is reproducible run-to-run' },
    verify_mirrors_clean: { type: 'boolean', description: 'if any Spinel lib changed; N/A otherwise (state which)' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    new_gate_cmd: { type: 'string', description: 'the exact command + expected fingerprint that downstream workflows should use as the GGUF-mmap gate' },
    infeasible_blocker: { type: 'string', description: 'if gate_built=false: the precise reason a bit-identical round-trip could not be achieved' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: Baseline -----------------------------------------------------

phase('Baseline')

const baseline = await agent(`Record the ROUND-TRIP TARGET: the in-memory realize_for_random_init forward output at a small fixed config + seed. ${PROBLEM}

Use smoke_projection_lens (SEED=0 STEPS=5) OR a more direct tiny random_init forward if that's cleaner — whatever gives a stable, small logits/loss fingerprint. Build, run TWICE, confirm byte-identical (deterministic). This is the value the reloaded-from-GGUF forward must reproduce. Do NOT edit any file.`, { label: 'roundtrip-target', phase: 'Baseline', schema: BASE_SCHEMA })

const shaRaw = await agent(`Run \`git rev-parse HEAD\` then \`git status --porcelain --untracked-files=no\`. Report the 40-char sha. End with "CLEAN <sha>" if no tracked changes, else "DIRTY <sha>".`, { label: 'base-sha', phase: 'Baseline' })
const baseSha = ((shaRaw || '').match(/\b([0-9a-f]{40})\b/) || [])[1] || ''
const treeDirty = /\bDIRTY\b/.test(shaRaw || '')

log(`HEAD ${baseSha || '(unparsed)'} | tree ${treeDirty ? 'DIRTY' : 'clean'} | target det=${baseline?.deterministic}`)
if (!baseline || !baseline.deterministic) { log('ABORT: no deterministic round-trip target.'); return { aborted: true, reason: 'no deterministic target', baseline } }
if (!baseSha || treeDirty) { log('ABORT: dirty tree / bad HEAD.'); return { aborted: true, reason: 'dirty tree', baseSha } }

// ---- Phase 1: Recon --------------------------------------------------------

phase('Recon')

const FACETS = [
  { key: 'writer-caps', prompt: `Map ${WRITER} (ToyGGUFWriter): what tensor dtypes it can write (F32 only?), the API to add named tensors, how it lays out / pads data, and whether it can write a fused attention weight. Confirm or refute "F32-only".` },
  { key: 'naming-mismatch', prompt: `Pin the EXACT per-head vs fused naming mismatch in ${MONOLITH}: where realize_for_random_init creates per-head attention weights ('blk.N.attn_q.head_H.weight', ~line 1168 pre-refactor — find current) and their shapes/order; where realize_for_mmap looks up fused names ('blk.N.attn_q.weight', ~542) and head-slices. Detail the head-slice stride math so a head-FUSING write can produce the byte layout mmap expects.` },
  { key: 'mmap-reload', prompt: `Map realize_for_mmap's expectations: which GGUF keys/metadata it reads (config, rope, dtypes), tensor file-offset + stride logic (head_nbytes), what it requires to reload an F32 model written by ToyGGUFWriter. What metadata must the writer emit for realize_for_mmap to succeed?` },
  { key: 'existing-roundtrips', prompt: `Find existing round-trip / reload smokes or converters (e.g. native-GGUF converter, smoke_toy_ckpt_reload, mmap parity demos, the Mat-roundtrip primitive tnn_download_to_f64_array / read_persistent_mat) to model the new fixture after and reuse machinery instead of reinventing.` },
]

const reconArr = await parallel(FACETS.map((f) => () =>
  agent(`READ-ONLY recon to build a GGUF round-trip gate. ${PROBLEM}\n\nFacet: ${f.prompt}\n\nReport current file:line refs (grep). Do NOT edit. Reference docs:\n  - ${REFS}`, { label: `recon:${f.key}`, phase: 'Recon', schema: RECON_SCHEMA })))
const recon = {}; FACETS.forEach((f, i) => { recon[f.key] = reconArr[i] })
for (const f of FACETS) { const r = recon[f.key]; log(`recon ${f.key}: ${r ? 'ok (' + (r.blockers?.length || 0) + ' blockers)' : 'FAILED'}`) }

// ---- Phase 2: Plan ---------------------------------------------------------

phase('Plan')

const plan = await agent(`Architect the GGUF round-trip gate. ${PROBLEM}\n${FRAMING}\n\n== Recon ==\n${JSON.stringify(recon, null, 2)}\n\nDecide: the head-fusing approach (must reproduce the per-head forward math EXACTLY after reload), the fixture design (small fixed-seed random_init -> forward -> write fused GGUF -> realize_for_mmap -> forward -> assert bit-identical), files touched, and the parity_argument (why bits are preserved; flag any reorder/pad/stride that could perturb them). Set feasible=false with the precise blocker if a bit-identical F32 round-trip isn't achievable. Do NOT edit any file. Reference docs:\n  - ${REFS}`, { label: 'gate-plan', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan || !plan.feasible) {
  log(`STOP: GGUF round-trip gate infeasible — ${plan ? (plan.risks || []).join('; ') : 'planner failed'}. No edits. This is a valid outcome; report the blocker so the realize GGUF paths stay deferred.`)
  return { aborted: true, reason: 'gate infeasible', baseSha, recon, plan }
}
log(`Plan ready. Approach: ${plan.fusing_approach?.slice(0, 100)}. New gate: ${plan.new_gate_cmd}`)

// ---- Phase 3: Build + verify ----------------------------------------------

phase('Build')

const result = await agent(`Build the GGUF round-trip gate per the plan. ${PROBLEM}\n${FRAMING}\n\n== Plan ==\n${JSON.stringify(plan, null, 2)}\n\n== Round-trip target (must reproduce) ==\n${baseline.logits_fingerprint}\n(produced by: ${baseline.how})\n\n== Procedure ==\n1. Implement the head-fusing write + the round-trip fixture per the plan.\n2. Build + run the fixture. The reloaded-via-mmap forward MUST be byte-identical to the in-memory random_init forward (the target above). Run it TWICE to confirm determinism.\n3. If bit-identity FAILS: do NOT loosen to a tolerance and do NOT alter the forward to force a match. Diagnose; if it's a faithful-round-trip blocker, set gate_built=false + infeasible_blocker and revert (git checkout -- / rm new files; clean vs ${baseSha}).\n4. If any Spinel-compiled lib changed: ruby ${MIRRORGEN} && make verify-mirrors (exit 0).\n5. COMMIT (only if roundtrip_bit_identical && deterministic): git add ONLY the new/changed source (NEVER -A, no binaries). Title "P2.6 gate: GGUF F32 round-trip parity gate (unlocks realize_for_mmap)". Capture sha + the exact new_gate_cmd (command + expected fingerprint) for downstream use.\n\ngate_built=true ONLY if roundtrip_bit_identical && deterministic && committed. Reference docs:\n  - ${REFS}`, { label: 'build-gate', phase: 'Build', schema: BUILD_SCHEMA })

if (result && result.gate_built && result.committed) {
  log(`OK: GGUF round-trip gate built ${result.commit_sha || ''} | bit-identical reload | new gate: ${result.new_gate_cmd}`)
} else {
  log(`HALT: gate not built — ${result ? (result.infeasible_blocker || result.notes) : 'agent returned nothing'}. Tree clean vs ${baseSha}. GGUF realize paths stay deferred.`)
}

return {
  baseSha,
  target: baseline.logits_fingerprint,
  feasible: plan.feasible,
  result,
  followup: result && result.gate_built
    ? `GGUF-mmap gate now exists (${result.new_gate_cmd}). Next: re-run prep/wf_finish_realize_bulk.js with this gate added so the deferred mmap/full_finetune-load steps unlock. Then L4 recipes.`
    : 'Gate infeasible — GGUF realize paths remain deferred. Proceed to L4 recipes (unblocked) instead.',
}
