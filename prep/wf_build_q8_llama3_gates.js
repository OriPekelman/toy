export const meta = {
  name: 'build-q8-llama3-gates',
  description: 'Build the last two realize-cascade gates: Q8 (load an existing Q8 GGUF via realize_for_q8_copy, deterministic forward — preservation gate, NOT quantize-on-write) and llama3 (TENSOR-LEVEL: assert the post-rope K/Q tensor under llama3 rope_freq_factors is deterministic — logit-level was proven rope-angle-insensitive). Fixtures only. Defer any not honestly constructible.',
  whenToUse: 'Realize-bulk fixture-cascade, batch 2: Q8 + llama3 (tensor-level). After this: re-run realize-bulk with ALL gates, then P3.',
  phases: [
    { title: 'Recon', detail: 'parallel read-only: realize_for_q8_copy path + Q8 model; the post-rope tensor-tap seam for a llama3 tensor-level gate' },
    { title: 'Plan', detail: 'per-gate fixture design + feasibility' },
    { title: 'Build', detail: 'serial loop: build each gate fixture, verify branch + determinism, record baseline, commit' },
  ],
}

const MIRRORGEN = 'prep/gen_cuda_mirror.rb'

const REFS = [
  'docs/roadmap/SESSION-RESUME-2026-05-29h.md',
  'examples/smoke_gate_qkv_bias.rb  (just-built pattern: load a GGUF via realize_for_mmap, deterministic forward, assert branch — clone for Q8 but use realize_for_q8_copy)',
  'examples/smoke_gate_gqa_divergent.rb  (random_init fixture pattern; tnn_set_output + download for tensor taps)',
  'lib/llama_seq_forward_ffi.rb  (realize_for_q8_copy ~236; rope_freq_factors / llama3 branches; tnn_set_output usage)',
  'lib/toy/llm/primitives/rope.rb  (RoPE.apply_2d — the post-rope output to tap for the llama3 tensor gate)',
  'data/ present: qwen25-0.5b-native-q8.gguf (Q8 model for realize_for_q8_copy), llama-3.2-1b-native.gguf (llama3 rope scaling)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md  (#16)',
  '~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/project_q8_0_verified_2026_05_17.md',
].join('\n  - ')

const FRAMING = `CRITICAL FRAMING.
(1) HONEST PRESERVATION GATES (not round-trips). Each gate is a NEW deterministic fixture that genuinely exercises its branch + records a baseline the realize-bulk workflow uses as a before/after self-consistency gate. NOT a correctness/round-trip claim.
 - Q8: load the EXISTING Q8 GGUF (data/qwen25-0.5b-native-q8.gguf) via realize_for_q8_copy, run a FIXED-INPUT deterministic forward, record the logits baseline. ASSERT the q8_copy path ran (e.g. a loaded attn weight tensor type == Q8_0, not F32) so we know it's the q8 path, not a dequant-to-f32 path. NO quantize-on-write needed.
 - llama3: logit-level is proven rope-angle-INSENSITIVE (out of scope to fix), so gate at the TENSOR level. Build a forward under llama3 rope (rope_freq_factors non-trivial — either llama-3.2-1b-native.gguf via mmap, or a random_init config with llama3 rope params), tnn_set_output the POST-ROPE K (or Q) tensor, download it, record its baseline. ASSERT (a) rope_freq_factors are non-trivial/llama3 path taken, (b) the post-rope tensor is deterministic run-to-run. This gates the llama3 rope realize branches even though logits don't move.
(2) FEASIBILITY. If a gate can't be honestly built (Q8 path dequants so the type-assert can't prove q8; the post-rope tensor can't be tapped without a lib change; a model is too heavy), set feasible=false + DEFER with the precise reason. Do NOT fake.
(3) FIXTURES ONLY. New examples + Makefile targets (+ tiny pure-Ruby helper if needed). NO lib/ behavior change. If a Spinel lib changes, regen mirrors + verify-mirrors; pure-Ruby fixtures need none.
(4) Spinel hygiene: no Struct.new (#16). One commit per gate.`

const GATES = [
  { key: 'q8', prompt: `Q8 preservation gate. Load data/qwen25-0.5b-native-q8.gguf via realize_for_q8_copy (the typed-copy path, NOT mmap), run a FIXED-INPUT deterministic forward (small fixed token list, fixed positions), record the logits baseline. ASSERT the q8 path actually ran: read a loaded attention weight tensor's type and confirm it's Q8_0 (type 8), proving weights stayed Q8 in memory (per Phase-3 Q8-stays-Q8), not dequantized to F32. Determinism: run twice, byte-identical. Model is 925MB — forward should be seconds. If realize_for_q8_copy actually dequantizes to F32 (so the type-assert fails), report that (still a valid gate via logits, but note it).` },
  { key: 'llama3_tensor', prompt: `llama3 TENSOR-LEVEL gate. Logit-level is rope-angle-insensitive (established), so gate the post-rope tensor. Build a forward under llama3 rope (use llama-3.2-1b-native.gguf via mmap if feasible+fast, ELSE a random_init small config with rope_scaling=:llama3 params so rope_freq_factors are non-trivial). tnn_set_output the post-rope K tensor (RoPE.apply_2d output for the K path — find where it's built), run the forward, download that tensor, record its values baseline. ASSERT (a) rope_freq_factors non-trivial (llama3 path), (b) post-rope tensor deterministic run-to-run. If the post-rope tensor can't be tapped without modifying lib/ (set_output already present? or needs adding?), and adding it would change behavior, report feasible=false — but a set_output tap is usually non-behavioral (just marks for readback). Prefer the smallest deterministic config.` },
]

const RECON_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gate', 'constructible', 'how', 'risks'],
  properties: {
    gate: { type: 'string' }, constructible: { type: 'boolean' },
    how: { type: 'string' }, line_refs: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
  },
}
const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['gates'],
  properties: { gates: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    required: ['key', 'feasible', 'fixture_design', 'assert_branch', 'files'],
    properties: {
      key: { type: 'string' }, feasible: { type: 'boolean' },
      fixture_design: { type: 'string' }, assert_branch: { type: 'string' },
      files: { type: 'array', items: { type: 'string' } }, defer_reason: { type: 'string' },
    },
  } } },
}
const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gate', 'built', 'exercises_branch', 'deterministic', 'committed', 'notes'],
  properties: {
    gate: { type: 'string' }, built: { type: 'boolean' },
    exercises_branch: { type: 'boolean' }, deterministic: { type: 'boolean' },
    committed: { type: 'boolean' }, commit_sha: { type: 'string' },
    gate_cmd: { type: 'string' }, deferred_reason: { type: 'string' }, notes: { type: 'string' },
  },
}

const shaRaw = await agent(`Run \`git rev-parse HEAD\` then \`git status --porcelain --untracked-files=no\`. Report the 40-char sha. End with "CLEAN <sha>" or "DIRTY <sha>".`, { label: 'base-sha', phase: 'Recon' })
const baseSha = ((shaRaw || '').match(/\b([0-9a-f]{40})\b/) || [])[1] || ''
if (!baseSha || /\bDIRTY\b/.test(shaRaw || '')) { log('ABORT: dirty tree / bad HEAD.'); return { aborted: true, reason: 'dirty tree', baseSha } }
log(`HEAD ${baseSha}`)

phase('Recon')
const reconArr = await parallel(GATES.map((g) => () =>
  agent(`READ-ONLY recon for a realize-cascade gate. ${FRAMING}\n\nGate: ${g.prompt}\n\nDetermine constructible + how. grep for line refs. Do NOT edit. Reference docs:\n  - ${REFS}`, { label: `recon:${g.key}`, phase: 'Recon', schema: RECON_SCHEMA })))
const recon = {}; GATES.forEach((g, i) => { recon[g.key] = reconArr[i] })
for (const g of GATES) { const r = recon[g.key]; log(`recon ${g.key}: ${r ? (r.constructible ? 'constructible' : 'NOT — ' + (r.risks || []).join('; ')) : 'FAILED'}`) }

phase('Plan')
const plan = await agent(`Plan the Q8 + llama3 tensor-level gates. ${FRAMING}\n\n== Recon ==\n${JSON.stringify(recon, null, 2)}\n\nPer gate: {key, feasible, fixture_design, assert_branch, files}. feasible=false (+defer_reason) if not honestly constructible. Do NOT edit. Reference docs:\n  - ${REFS}`, { label: 'q8-llama3-plan', phase: 'Plan', schema: PLAN_SCHEMA })
if (!plan || !plan.gates || plan.gates.length === 0) { log('ABORT: no plan.'); return { aborted: true, reason: 'no plan', baseSha, recon } }
log(`Plan: ${plan.gates.filter((g) => g.feasible).length}/${plan.gates.length} feasible`)

phase('Build')
const results = []
for (let i = 0; i < plan.gates.length; i++) {
  const g = plan.gates[i]
  if (!g.feasible) { log(`SKIP ${g.key}: ${g.defer_reason}`); results.push({ gate: g.key, built: false, committed: false, deferred_reason: g.defer_reason }); continue }
  const r = await agent(`Build the "${g.key}" realize-cascade gate fixture. ${FRAMING}\n\n== Spec ==\n${JSON.stringify(g, null, 2)}\n\n== Procedure ==\n1. Create examples/smoke_gate_${g.key}.rb + Makefile target per fixture_design. It MUST genuinely exercise the branch (prove via assert_branch — print/assert the condition) AND be deterministic (run twice byte-identical).\n2. If the branch isn't hit or it's not deterministic, built=false + exercises_branch=false + deferred_reason. Do NOT fake.\n3. If a Spinel lib changed: ruby ${MIRRORGEN} && make verify-mirrors. Pure-Ruby → N/A.\n4. COMMIT (only if exercises_branch && deterministic): git add ONLY the new fixture(s) + Makefile (+ helper). NEVER -A, no binaries/gguf. Title "P2.6 gate: ${g.key} parity fixture". Capture sha + gate_cmd (command + baseline fingerprint).\n\nbuilt=true ONLY if exercises_branch && deterministic && committed. Reference docs:\n  - ${REFS}`, { label: `build:${g.key}`, phase: 'Build', schema: BUILD_SCHEMA })
  results.push(r)
  if (r && r.built && r.committed) log(`OK ${g.key}: ${r.commit_sha || ''} | ${r.gate_cmd?.slice(0, 80)}`)
  else log(`DEFER ${g.key}: ${r ? (r.deferred_reason || r.notes) : 'no result'}`)
}

const built = results.filter((r) => r && r.built && r.committed)
return {
  baseSha,
  built: built.map((r) => ({ gate: r.gate, sha: r.commit_sha, gate_cmd: r.gate_cmd })),
  deferred: results.filter((r) => r && !r.built).map((r) => ({ gate: r.gate, why: r.deferred_reason || r.notes })),
  results,
  followup: 'All cascade gates attempted. Next: re-run realize-bulk with every available gate (cpu, cuda, gguf, gqa_divergent, b_gt_1, qkv_bias, q8, llama3_tensor) wired in to unlock the deferred slices. Then P3.',
}
