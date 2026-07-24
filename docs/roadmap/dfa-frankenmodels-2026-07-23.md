# DFA + FrankenModels: the credit-assignment seam (design assessment, 2026-07-23)

**Verdict up front: GO, bounded.** The abstraction that covers every asked-for
experiment is small — a per-segment *gradient source* — and toy's training
machinery already has the two seams it needs. The monster shows up only if we
add a detach primitive too early or try to fold this into the shared engine
unit; both are avoidable by construction. Drop-triggers are listed and cheap to
evaluate at each phase.

## 1. The ask (Tao-side experiments)

Models partly/jointly trained with backprop (BP) vs Direct Feedback Alignment
(DFA): MoE with some experts DFA-trained; BOTH signals blended per layer; some
layers DFA / some BP; other mixes. Constraint: a general mechanism, or nothing.

## 2. DFA in one paragraph

BP computes δ_l by chaining Wᵀ through every layer above. DFA (Nøkland 2016)
hands every trainable unit its error directly from the top: δ_l = (B_l·e) ⊙
f'(a_l), with e = ∂L/∂logits and B_l a FIXED random matrix (never trained).
Updates are local: ΔW_l = δ_l·h_{l-1}ᵀ. No backward chain ⇒ layer-local,
parallelizable updates. Known limits: plain DFA lags BP on deep transformers at
scale (LightOn scaling-law results); it shines in MLP-ish blocks and as a
*component* — exactly the FrankenModel frame.

## 3. What the engine recon established (facts, with refs)

- Backward is ONE autodiff sweep: `tnn_build_backward` →
  `ggml_build_backward_expand` (tinynn_ggml.c:1744). No stop-gradient/detach
  primitive exists anywhere in the shim.
- Gradient selectivity today = param-set membership only: `tnn_set_param`
  (GGML_TENSOR_FLAG_PARAM). LoRA's "frozen base" is simply *not in the param
  set* (llama_seq_engine.rb:297-298). Autodiff only builds grad paths toward
  PARAM-flagged tensors.
- **Seam 1 — the backward graph is explicitly extensible**:
  `tnn_extend_backward_graph` (tinynn_ggml.c:1752) appends arbitrary forward
  nodes onto graph_b; it's how opt_step nodes are attached today.
- **Seam 2 — the optimizer takes ANY grad tensor**: `tnn_opt_step_adamw(sess,
  w, grad, m, v, hp)` (llama_seq_engine.rb:1131-1180 loop) is fed
  `tnn_tensor_grad(w)` by convention, but accepts any tensor of matching shape.
- Heterogeneous-stack precedent: the Dragon/GDN hybrid trains mixed layer kinds
  in ONE run via a flat per-layer INT kind + uniform flattened param arrays
  (train_hybrid.rb:87-160), as a DEDICATED compiled runner. The reason for the
  dedicated unit (union-pin `grads==NULL` miscompile) is likely obsolete on the
  plain-master pin, but the pattern is proven and cheap.
- MoE is INFER-ONLY today: `mul_mat_id` has no backward
  (docs/coverage.md:68) — no grads flow through expert dispatch.
- No fixed-random-matrix machinery exists; B_l tensors are net-new (persistent,
  xorshift-seeded, excluded from the param set).

## 4. The abstraction: per-segment gradient source

A **segment** = any weight-bearing subgraph with a designated output boundary
(a layer, a block, an expert, an adapter). Per segment, ONE dial:

| source        | boundary error δ_seg                | covers                     |
|---------------|-------------------------------------|----------------------------|
| `:chain`      | true backprop from above (default)  | today's behavior           |
| `:dfa`        | B_seg · e (B fixed, seeded)         | DFA layers / DFA experts   |
| `:mix(α)`     | α·chain + (1−α)·B_seg·e             | "both at once per layer"   |
| (not in set)  | —                                   | frozen (exists today)      |

Within a segment, δ_seg reaches the segment's params by the LOCAL product
(δ·hᵀ for its matmuls) — for single-layer segments that's the DFA update rule
verbatim; for a multi-layer BP-inside segment it's chain-within-the-segment.

**Why this maps cleanly onto the two seams:**

- A `:dfa` segment needs NO autodiff at all. δ_seg = B_seg·e and ΔW = δ·hᵀ are
  plain forward ops — appended onto graph_b via `tnn_extend_backward_graph`,
  with the product tensor passed to `tnn_opt_step_adamw` AS the grad (Seam 2).
  The autodiff sweep never knows the segment exists (its weights just aren't
  PARAM-flagged).
- A `:chain` segment is untouched — exactly today's path.
- `:mix(α)`: `ggml_add(scale(chain_grad,α), scale(dfa_grad,1−α))` → opt_step.
  Requires the weight to be PARAM-flagged (for chain_grad) plus the DFA nodes.
  Strictly additive; no new op classes.
- e = ∂L/∂logits: for CE this is softmax(logits)−onehot(labels) — cheap forward
  ops on tensors the graph already has. (ggml's own CE-backward computes
  exactly this; we materialize it once per step for all DFA segments.)

**The cut question (where the monster would live) — mostly dissolves.**
Classic DFA also *severs* the chain at segment boundaries. Observation: the
autodiff sweep only builds paths toward PARAM tensors. Two common Franken
configs need no cut at all:
- *DFA segments below BP segments* (e.g. DFA lower layers, BP head): nothing
  below is PARAM-flagged → no chain paths are built below the boundary. The
  cut emerges from param-set membership, free.
- *DFA experts + BP router*: router grads flow through the gate-weight ×
  expert-OUTPUT multiply — never through expert internals. Cut at the expert
  input is free for the same reason.
The one config that genuinely needs a detach (BP segment strictly BELOW a
DFA/transparent segment whose weights must not get chain grads... while chain
still must pass THROUGH) can be served in v1 by `:mix(α)` with α=0 on the DFA
segment — its weights get pure-DFA updates while the chain flows through
undisturbed to the BP segment below ("transparent DFA"). That is a legitimate
experimental arm in its own right (and the literature's hybrid variants do
exactly this). A true opaque-cut primitive (a vendored zero-backward op,
CONCAT-backward precedent: 15 LOC) is deferred until an experiment
specifically demands opacity-with-passthrough-below — T3 guards this.

**The MoE bonus (this strengthens GO).** DFA-trained experts sidestep the
missing `mul_mat_id` backward entirely: expert-weight updates are local
products over the tokens routed to that expert (a get_rows gather + δ·hᵀ),
all forward ops. "Franken-MoE with BP-router + DFA-experts" is trainable
WITHOUT implementing mul_mat_id autodiff — DFA isn't just an experiment here,
it's a workaround for a real coverage gap. (Full-BP MoE training remains a
separate, orthogonal leg: mul_mat_id backward, own issue.)

## 4b. The B network + the comparison protocol (2026-07-24 clarification)

**B_seg spec.** Fixed at init, never trained, shape `d_seg × d_out`. The
family and scale are EXPERIMENT PARAMETERS (the scale is a hidden
per-layer learning rate — it matters more than the family):

    dfa_b: { dist: :gaussian | :uniform | :rademacher(p),
             scale: :fixed(σ) | :inv_sqrt_fan | :glorot,
             seed:  <run seed> }

Defaults: `:gaussian`, `:inv_sqrt_fan` (σ = 1/√d_out). Sampling is
xorshift (+ Box–Muller for gaussian); each segment's B is seeded from
`(run_seed, stable_segment_id)` so re-policying one segment never
reshuffles another's B. Byte-reproducible by construction (gate T4).

**Two paired-comparison protocols** (they compose):

1. **Shadow gradients** — one model, ONE forward pass, both credit
   signals: apply the policy's update, LOG the other. Yields the FA
   literature's core diagnostic, per-layer/per-step alignment
   `cos∠(g_dfa, g_bp)` — the direct measurement of where DFA is a
   sufficient credit signal (alignment is the mechanism by which
   FA/DFA learns at all). Caveat: shadow-BP requires building the full
   autodiff chain on that lane, so the membership CUT is absent in
   diagnostics builds (chain grads computed, used only for telemetry;
   applied updates provably unaffected — gated byte-identical
   shadow-on vs shadow-off).
2. **Twin lanes** — two weight sets, bit-identical seeded W₀, same
   batch stream + hp schedule, stepped in lockstep in one binary.
   After step 1 the forwards necessarily diverge (the weights do);
   what stays paired is everything else — per-step loss deltas,
   per-layer weight-divergence (L2 / cosine), and the end states.

The instrumented runner does both: lane A `:chain` (doubles as the
null-parity gate), lane B policy-driven with optional shadow.

**Combiner backlog (noted 2026-07-24, for later experiments):** beyond
`:mix(α)`, a MASKED combiner family — use one signal as a per-weight
GATE for the other: `update = g_dfa ⊙ 1[|g_bp| > τ]` (or the
transpose; or top-k instead of threshold) — an "activation function on
the update". Cheap in this representation: both grad tensors already
exist per-weight in shadow builds; the combiner is three elementwise
ops at the same point where `:mix(α)` sits. The
alignment-by-depth / by-expert curves are the intended generator of
mixing intuitions (expert-wise, layer-wise, bottom-BP/top-DFA — the
literature's prior is that alignment weakens toward the bottom of deep
stacks — interleaved groups). Head-wise segmentation is structurally
expressible (a head's output slice is a boundary) but stays out of v1
per the non-goals.

## 4c. The parallelism reading + the v1 DFA variant (2026-07-24)

**Update-parallelism is represented, latently.** BP's backward is a
serial dependency spine (grad_l needs grad_{l+1} — the backward lock).
Each DFA update subgraph shares only {e, its own forward activations}
with the rest of the graph and has NO edges to other layers' update
subgraphs: width, not depth. ggml-cpu still executes the independent
branches in topo order (threading is op-internal), so v1's win is
graph-structural, not wall-clock — but it is the precondition for
layer-parallel dispatch, pipeline-style update unlocking, and the
scaling-story's loose-coupling unit (a layer shard needs broadcast-e +
local state only; no backward sync). The runner logs
backward-wallclock DFA-all vs BP-all to size the latent win.

**v1 variant choice — per-matmul DFA (identity-boundary segments).**
Block-boundary DFA (δ at block output, standard BP *within* the block —
the literature-canonical transformer form) requires per-segment
surrogate losses, and those contaminate the segments below unless the
chain is CUT at each segment input: it genuinely needs the detach
primitive. The v1 form instead gives EVERY weight matrix its own
B_W (projecting e into that weight's output space); the update
(B_W·e)·h_inᵀ is pure forward ops — P0 mechanics verbatim, no
autodiff, no cuts — and composes with ANY policy layout (interleaved,
expert-wise, head-wise later) rather than only contiguous stacks.
Explicitly a modeling choice: finer credit granularity than canonical
DFA; the shadow-alignment telemetry is the instrument that says what
it costs. Block-boundary DFA joins the roadmap as the cut-primitive
refinement (P3+).

## 5. Shape of the implementation (bounded)

- **One dedicated runner** (`toy-train-franken`), hybrid-runner pattern: flat
  per-segment INT `credit_mode` + parallel arrays (B tensors, α), single
  uniform opt_step loop. No changes to the existing five trainers. Fold-in to
  the shared engine is a later, separate decision (same as GDN reintegration).
- **Shim additions**: none required for P0/P1 beyond possibly
  `tnn_softmax_minus_onehot` if composing it from existing ops is awkward —
  everything else is existing binds (`mul_mat`, `scale`, `add`, `get_rows`,
  `tnn_extend_backward_graph`, `tnn_opt_step_adamw`). CUDA/Metal mirrors:
  only if a new FFI symbol is added (the mirror landmine is why we prefer
  composing from existing binds).
- **B matrices**: persistent tensors via existing `tnn_input_2d_f32_persistent`,
  xorshift-seeded from a recorded seed → byte-reproducible (gate-able).
- **hp slots**: the runner adopts the lora convention for slots 5/6
  (bias-correction denominators) — new graph, no legacy to honor.
- **Consumer surface (Tao)**: RecipeOptions grows
  `credit_assignment: [{segments:, source:, alpha:, seed:}]`; flat-INT
  encoding at the engine boundary (Spinel-safe; hybrid precedent).

## 6. Drop-triggers (evaluated per phase)

- **T1**: shim/API delta beyond ~300 LOC + the one runner, or any change
  required inside the five existing trainers → stop, reassess.
- **T2**: per-experiment recompiles for policy changes (policy must be data —
  flat arrays — not graph-shape variants per config).
- **T3**: a third policy axis (independent opacity control, per-step
  schedules) gets requested → resist; redesign only with evidence from P1/P2
  results.
- **T4**: DFA runs can't be made byte-reproducible → it doesn't fit toy's
  gate culture; stop.

## 7. Phases

- **P0 — ✅ DONE 2026-07-23** (`~/tmp/dfa_p0_poc.c`, deterministic
  ALL-PASS): 2-layer MLP, layer-1 `:dfa` / layer-2 `:chain` on one
  backend_sched graph_b (21 nodes, single alloc). Cut-by-membership
  confirmed (autodiff builds NO W1 grad); chain grad vs finite-diff
  1.98e-5; in-graph DFA grad vs hand reference 1.6e-6 over all 128
  entries; mixed AdamW training CE 1.208 → 0.056 in 300 steps,
  byte-identical across runs. The sched abort trigger did NOT fire.
  Two load-bearing idioms surfaced for the P1 runner:
  1. **Late-param flag**: `ggml_opt_step_adamw` asserts
     `GGML_TENSOR_FLAG_PARAM` on the stepped weight — flag the `:dfa`
     weight AFTER `build_backward_expand` (autodiff never sees it, the
     assert is satisfied). Ordering is load-bearing.
  2. **Pin all read-backs**: nodes appended after the backward (the DFA
     chain + opt steps) reuse grad-acc/loss buffer slots under sched —
     `ggml_set_output` everything read back (the engine already does
     this via `tnn_pin_all_graph_b_nodes`).
- **P1 — ✅ DONE 2026-07-24** (`lib/toy/run/train_franken.rb` →
  `libexec/toy-train-franken`, `gate-franken` PASS on all four legs).
  Self-contained twin-lane runner (hybrid-runner precedent; the
  engine-integrated byte-parity-vs-toy-train gate moves to P2 with
  RecipeOptions): two attention towers, bit-identical seeded init, ONE
  graph/session, per-layer FRANKEN_POLICY (chain|dfa), per-matmul DFA
  (§4c), shadow alignment telemetry. Gates: twin-parity (chain,chain ⇒
  lanes byte-identical every step), dfa-decreases (CE 2.90→0.61/60
  steps; BP lane 2.90→0.012), alignment well-formed (480 lines), byte-
  repro. Zero shim changes — all composed from existing FFI binds; T1
  margin huge. Findings:
  1. **One-graph twins are mandatory**: tinynn's engine+sched is a
     process-global singleton (`g_engine_cpu`) — two sessions
     alternating realize/compute desynchronize (F1.1 stale-alloc
     class; manifested as identical weights computing different
     losses). Both towers in one graph_b = one sched alloc = lockstep
     by construction. (A per-session sched is the eventual shim fix if
     multi-session is ever needed; not required for this program.)
  2. **Graph-root ordering**: `tnn_add_to_graph` silently refuses (-2)
     once `tnn_build_forward_only` has set `realized` — extra loss
     roots go in FIRST. Manifested as "tensor buffer not set" on the
     second tower's loss.
  3. **Early data**: bottom-BP/top-DFA interpolates between the pure
     arms (0.52 vs 0.17/0.97 at 40 steps); matmul-granularity
     alignment through attention is weak (|cos| mostly <0.3, some
     upward drift, e.g. layer-0 k → 0.57) yet training proceeds —
     the granularity-vs-alignment trade is now measurable.
- **P2**: per-segment policy table via RecipeOptions + the Franken-MoE arm
  (BP router + DFA experts on a small MoE; needs the routed-token gather).
  Tao gets the surface here; tao-side issue filed at P2 start.
- **P3**: `:mix(α)` blending arm + (only if experiments demand) the opaque-cut
  vendored op; CUDA leg LAST (mirror cost paid once the shape is stable).

## 8. Explicit non-goals (v1)

Full-BP MoE training (separate mul_mat_id-backward leg); attention-internal
DFA (segment granularity stops at block/expert/layer boundaries); serve/infer
changes (inference is untouched by construction); Metal until an experiment
needs it.
