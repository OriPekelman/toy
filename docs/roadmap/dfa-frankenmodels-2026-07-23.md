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
- **P1 (~days)**: `toy-train-franken` on the tiny-llama shape: last-block-BP /
  rest-DFA and all-chain configs. Gates: all-chain == existing trainer
  byte-exact (the null-hypothesis gate); DFA arm CE decreases; byte-repro
  across reruns. CPU only.
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
