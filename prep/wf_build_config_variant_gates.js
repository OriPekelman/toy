export const meta = {
  name: 'build-config-variant-gates',
  description: 'Build the config-variant parity gates that unlock the deferred realize-bulk slices: GQA-divergent (n_heads*head_dim != d_model), B>1 (micro-batching), qkv_bias (Qwen), llama3 (rope_freq_factors). Each = a NEW deterministic fixture that genuinely exercises its branch + a recorded baseline. Serial loop; defer any that is not honestly constructible. Fixtures only — no lib behavior change.',
  whenToUse: 'Realize-bulk fixture-cascade, batch 1: the 4 config-variant gates (Q8 round-trip is a separate workflow).',
  phases: [
    { title: 'Recon', detail: 'parallel read-only: constructibility + the right config/model/realize-path for each of the 4 variants' },
    { title: 'Plan', detail: 'per-gate: exact fixture design + feasibility (defer if not honestly constructible)' },
    { title: 'Build', detail: 'serial loop: build each feasible gate fixture, confirm it exercises the branch + is deterministic, record baseline, commit' },
  ],
}

const MIRRORGEN = 'prep/gen_cuda_mirror.rb'

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29h.md  (the deferred realize slices + why each needs a gate)',
  'examples/smoke_projection_lens.rb  (random_init CPU fixture pattern — clone for GQA-divergent + B>1)',
  'examples/smoke_gguf_roundtrip.rb + lib/toy_gguf_fuse.rb  (GGUF round-trip pattern — basis for qkv_bias/llama3 mmap gates)',
  'lib/toy_smollm2.rb  (SmolLM2Config: head_dim is OVERRIDABLE — default d_model/n_heads; set cfg.head_dim for GQA-divergence)',
  'lib/toy/llm/blocks/transformer_block.rb:311  (random_init w_o = [d_model, n_heads*head_dim]) vs lib/llama_seq_forward_ffi.rb mmap w_o = [d_model,d_model]',
  'data/ models present: qwen25-0.5b-native.gguf (qkv_bias=true), llama-3.2-1b-native.gguf (llama3 rope scaling), smollm2-135m-native.gguf',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (#16)',
].join('\n  - ')

const FRAMING = `CRITICAL FRAMING.
(1) DELIVERABLE = HONEST GATES. Each gate is a NEW deterministic fixture (examples/smoke_gate_<name>.rb + a Makefile target) that GENUINELY EXERCISES the target branch and whose output is reproducible run-to-run. It records a baseline the realize-bulk workflow will later use as an additional before/after self-consistency gate. A fixture that does NOT actually hit the branch is worthless — verify it does (e.g. for GQA-divergent, assert n_heads*head_dim != d_model AND the forward runs; for qkv_bias, assert the loaded model has qkv_bias=true; for llama3, assert rope_freq_factors are non-trivial; for B>1, assert seq_b>1).
(2) CONSTRUCTIBILITY / FEASIBILITY. If a variant cannot be honestly constructed (e.g. the toy forward doesn't actually support n_heads*head_dim != d_model and would crash or mis-shape, or no suitable model exists), set that gate feasible=false and DEFER it with the precise reason — do NOT force a broken or fake fixture. A discovered "this divergence can't occur / is provably safe" is itself a valuable result (it means that deferred realize slice can be unified safely later).
(3) FIXTURES ONLY. These are NEW example files + Makefile targets + (if needed) a small extension to toy_gguf_fuse for the GGUF variants. Do NOT change lib/ forward/realize BEHAVIOR. If you extend a Spinel-compiled lib, regen mirrors + verify-mirrors; pure-Ruby fixtures need none.
(4) Spinel hygiene: no Struct.new (#16). One commit per gate (or one commit for the batch of CPU gates — your call, keep it clean).`

const GATES = [
  { key: 'gqa_divergent', prompt: `GQA-DIVERGENT gate. Clone smoke_projection_lens (random_init CPU) but set cfg.head_dim so n_heads*head_dim != d_model (head_dim is overridable in SmolLM2Config; default d_model/n_heads). Goal: exercise the w_o shape that random_init allocates as [d_model, n_heads*head_dim] (transformer_block.rb:311) when it DIFFERS from [d_model,d_model]. FIRST verify the toy FORWARD actually supports this (does attention concat + w_o matmul work when n_heads*head_dim != d_model, or does it assume equality somewhere?). If the forward can't run with the divergence, feasible=false + report (means the divergence is unreachable → that realize slice is safe-by-construction).` },
  { key: 'b_gt_1', prompt: `B>1 gate. The merged toy#7 micro-batching added B>1 training. Clone smoke_projection_lens / 06 with t_batch>1 (e.g. 2) so seq_b>1 and the B>1 attn-mask BODY (soft_max_ext mask path + upload_block_causal_mask!) is exercised (not the B=1 diag_mask path). Confirm deterministic + that seq_b>1 actually routes the B>1 branch. Record baseline.` },
  { key: 'qkv_bias', prompt: `QKV-BIAS gate. random_init allocates NO qkv biases, so this needs the GGUF/mmap path on a model WITH qkv_bias=true — data/qwen25-0.5b-native.gguf (Qwen2.5 has qkv_bias). Build a deterministic forward (or short train) that loads it via realize_for_mmap with qkv_bias=true so the qkv_bias slice branches are exercised. Determinism + assert qkv_bias=true. (May reuse the GGUF round-trip machinery; or a fixed-input forward whose logits are deterministic.) If Qwen-0.5B is too big/slow for a gate, note it.` },
  { key: 'llama3', prompt: `LLAMA3 gate. Needs llama3 rope_freq_factors (Llama-3.2 rope scaling) — data/llama-3.2-1b-native.gguf, or a random_init config with llama3 rope params set. Build a deterministic forward that exercises the rope_freq_factors branches (assert they're non-trivial / the llama3 rope path is taken). Determinism + assert. If the 1B model is too heavy for a gate, consider a tiny random_init config with llama3 rope params instead.` },
]

const RECON_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gate', 'constructible', 'how', 'risks'],
  properties: {
    gate: { type: 'string' },
    constructible: { type: 'boolean', description: 'can this variant be HONESTLY constructed (forward runs, branch genuinely exercised)?' },
    how: { type: 'string', description: 'the exact config/model/realize-path + which fixture to clone + how to assert the branch is hit' },
    line_refs: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gates'],
  properties: {
    gates: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['key', 'feasible', 'fixture_design', 'assert_branch', 'files'],
        properties: {
          key: { type: 'string' },
          feasible: { type: 'boolean' },
          fixture_design: { type: 'string', description: 'exact new fixture (name, config, model, deterministic knobs)' },
          assert_branch: { type: 'string', description: 'how the fixture proves it hits the target branch' },
          files: { type: 'array', items: { type: 'string' } },
          defer_reason: { type: 'string', description: 'if feasible=false' },
        },
      },
    },
  },
}

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gate', 'built', 'exercises_branch', 'deterministic', 'committed', 'notes'],
  properties: {
    gate: { type: 'string' },
    built: { type: 'boolean' },
    exercises_branch: { type: 'boolean', description: 'verified the fixture actually hits the target branch' },
    deterministic: { type: 'boolean' },
    committed: { type: 'boolean' },
    commit_sha: { type: 'string' },
    gate_cmd: { type: 'string', description: 'the command + recorded baseline fingerprint for realize-bulk to use' },
    deferred_reason: { type: 'string' },
    notes: { type: 'string' },
  },
}

// ---- Phase 0: base sha -----------------------------------------------------

const shaRaw = await agent(`Run \`git rev-parse HEAD\` then \`git status --porcelain --untracked-files=no\`. Report the 40-char sha. End with "CLEAN <sha>" if no tracked changes, else "DIRTY <sha>".`, { label: 'base-sha', phase: 'Recon' })
const baseSha = ((shaRaw || '').match(/\b([0-9a-f]{40})\b/) || [])[1] || ''
if (!baseSha || /\bDIRTY\b/.test(shaRaw || '')) { log('ABORT: dirty tree / bad HEAD.'); return { aborted: true, reason: 'dirty tree', baseSha } }
log(`HEAD ${baseSha}`)

// ---- Phase 1: Recon --------------------------------------------------------

phase('Recon')

const reconArr = await parallel(GATES.map((g) => () =>
  agent(`READ-ONLY recon for a realize-cascade gate. ${FRAMING}\n\nGate: ${g.prompt}\n\nDetermine constructible + how (exact config/model/fixture). grep for current line refs. Do NOT edit. Reference docs:\n  - ${REFS}`, { label: `recon:${g.key}`, phase: 'Recon', schema: RECON_SCHEMA })))
const recon = {}; GATES.forEach((g, i) => { recon[g.key] = reconArr[i] })
for (const g of GATES) { const r = recon[g.key]; log(`recon ${g.key}: ${r ? (r.constructible ? 'constructible' : 'NOT constructible — ' + (r.risks || []).join('; ')) : 'FAILED'}`) }

// ---- Phase 2: Plan ---------------------------------------------------------

phase('Plan')

const plan = await agent(`Plan the config-variant gates from recon. ${FRAMING}\n\n== Recon ==\n${JSON.stringify(recon, null, 2)}\n\nFor each of the 4 gates produce {key, feasible, fixture_design, assert_branch, files}. feasible=false (with defer_reason) for any not honestly constructible. Do NOT edit. Reference docs:\n  - ${REFS}`, { label: 'gates-plan', phase: 'Plan', schema: PLAN_SCHEMA })

if (!plan || !plan.gates || plan.gates.length === 0) { log('ABORT: planner produced no gates.'); return { aborted: true, reason: 'no plan', baseSha, recon } }
log(`Plan: ${plan.gates.filter((g) => g.feasible).length}/${plan.gates.length} gates feasible`)

// ---- Phase 3: Build (serial; independent gates) ----------------------------

phase('Build')

const results = []
for (let i = 0; i < plan.gates.length; i++) {
  const g = plan.gates[i]
  if (!g.feasible) {
    log(`SKIP ${g.key}: deferred — ${g.defer_reason}`)
    results.push({ gate: g.key, built: false, committed: false, deferred_reason: g.defer_reason })
    continue
  }
  const r = await agent(`Build the "${g.key}" config-variant gate fixture. ${FRAMING}\n\n== Spec ==\n${JSON.stringify(g, null, 2)}\n\n== Procedure ==\n1. Create the fixture (examples/smoke_gate_${g.key}.rb) + Makefile target per fixture_design. It must (a) actually exercise the target branch — prove via assert_branch (print/assert the condition, e.g. n_heads*head_dim!=d_model, seq_b>1, qkv_bias=true, rope_freq_factors non-trivial), and (b) be deterministic.\n2. Build + run TWICE; confirm byte-identical output (deterministic) AND the branch-assertion passes. If the branch isn't actually hit, or the forward crashes (e.g. divergence unsupported), set built=false + exercises_branch=false + deferred_reason — do NOT ship a fixture that doesn't hit the branch.\n3. If you extended a Spinel lib (e.g. toy_gguf_fuse for GGUF variants): ruby ${MIRRORGEN} && make verify-mirrors. Pure-Ruby fixture → N/A.\n4. COMMIT (only if exercises_branch && deterministic): git add ONLY the new fixture(s) + Makefile (+ any helper). NEVER -A, no binaries/gguf. Title "P2.6 gate: ${g.key} parity fixture". Capture sha + gate_cmd (command + baseline fingerprint).\n\nbuilt=true ONLY if exercises_branch && deterministic && committed. Reference docs:\n  - ${REFS}`, { label: `build:${g.key}`, phase: 'Build', schema: BUILD_SCHEMA })
  results.push(r)
  if (r && r.built && r.committed) log(`OK ${g.key}: ${r.commit_sha || ''} | ${r.gate_cmd?.slice(0, 80)}`)
  else log(`DEFER ${g.key}: ${r ? (r.deferred_reason || r.notes) : 'no result'}`)
  // gates are independent — continue regardless
}

const built = results.filter((r) => r && r.built && r.committed)
return {
  baseSha,
  built: built.map((r) => ({ gate: r.gate, sha: r.commit_sha, gate_cmd: r.gate_cmd })),
  deferred: results.filter((r) => r && !r.built).map((r) => ({ gate: r.gate, why: r.deferred_reason || r.notes })),
  results,
  followup: 'Config-variant gates done. Next: Q8 round-trip gate (separate workflow), then re-run realize-bulk with ALL gates wired in to unlock the deferred slices. Then P3.',
}
