# DFA-arch program (toy#152–#158) — resolved decisions + build order

> **SUPERSEDED for anything after toy#162. Read this as a record of #152–#162
> and nothing later.**
>
> It says "all shipped" and is accurate about the lanes it covers. It
> predates the whole 2026-08-30/09-03 arc (toy#176, #179–#186), which roughly
> doubled the DFA surface and — more importantly — **reversed or retired
> several conclusions a reader would otherwise take from here**:
>
> * **Faithful feedback is not what DFA runs on.** An oracle `B` routing
>   `p_energy = 0.99999` trains the body no better than never training it,
>   while a random `B` carrying 1.6% beats both (#176/D1b). DFA is not
>   approximate backpropagation in any useful sense.
> * **The representational-diversity explanation for that is dead.** Oracle
>   and random arms have statistically identical stable rank (t = 0.00) while
>   sitting 0.208 bpb apart (#183, measured with `GTX_ACTRANK`).
> * **The cost claim has nothing to account for on this implementation.**
>   `frozen` and `chain` build identical backward graphs and retain identical
>   activations; `dfa` is *larger* on both (#182, the `ncost:` line). A saving
>   would have to come from not BUILDING those nodes, and nothing here does.
> * **Mismatched stale feedback is fatal at k=1** — past the frozen floor,
>   +0.414, t = +5.59 at n=5 (#186). Pipelined delay is a different and
>   unmeasured problem.
>
> The current knob surface, with semantics and the trap attached to each, is
> the header block of `lib/toy/run/train_gtx.rb`. **That is the reference;
> this file is history.**
>
> Read those traps seriously if you are arriving cold. Several of these knobs
> fail **silently in the direction of the hoped-for answer** when misused,
> and this arc paid for it three times: #181 adapters identical to their own
> control, #185 a flip firing once at init while the cadence sweep measured
> nothing, #186 an unwired ring reading as perfect staleness tolerance.

Tao filed six new architecture lanes plus F15. This records what is
SETTLED (tao#18, tao#19) so each lane can be built without re-litigating
it, and flags the one thing I think is still ambiguous.

Build order: **#152 → #158 → #154 → #153 / #155 / #156 / #157**, then
the instrument (#159), the disambiguation lane (#160) and the fair BPTT
control (#162). Status: **all shipped** — #152, #158, #154, #153, #155,
#156, #157, #159, #160 and #162 (see their sections below).

## Settled (tao#18, tao#19 — Ori)

1. **`--policy-scope` is NOT accepted on the MLP recipe.** Reject it in
   the flag×recipe matrix like any other lane-specific flag. Per-layer
   `--policy chain|dfa` alone is what F16 needs. The `attn|ffn|all`
   meaning stays stable across lanes — a self-describing bundle is
   worth more than a convenience flag. If a head-vs-hidden split is
   ever wanted it gets a DIFFERENT name (`--policy-tensors`), never an
   overload of `--policy-scope`. (toy#152's mention of it was a
   copy-paste from the franken surface.)

2. **CPU-only for T0–T3 and F15's small case.** No CUDA twins. The
   anchors are small by construction; this halves the work per lane and
   removes twin-drift, which has silently bitten twice recently
   (toy#150, toy#151 — both caught by gates, not review).
   - **Exception, deferred:** F19 (SSM). The small-output *positive* is
     CPU-fine; the *value prop* is killing BPTT's seq-length memory,
     which wants long sequences. Add a CUDA twin only when that
     measurement happens, not before.
   - F18 (CTR): Criteo SUBSET or the synthetic generator. Do not pull
     full Criteo.

3. **`align` events off the transformer lane:** keep `li` as the layer
   index, make `wi` LANE-LOCAL, and add a **`wname` string**
   (`"w1"`, `"head"`, `"gate_proj"`, …). Tao's ingest keys on `wname`,
   not on any per-lane `wi` table. Adding `wname` to transformer
   bundles is additive and safe (unknown keys are ignored today).

4. **The success bar is MANDATORY in every lane's gate leg**, not
   optional:

   > a POSITIVE = all-DFA within the stated gap of all-BP
   > **AND** provably beating the frozen control,
   > at matched init and matched seed.

   The per-ticket gaps (#152 "small", #154 "~0.01 AUC") are only the
   BP-gap half. The frozen-beat is the other half and is
   non-negotiable — without it "near-BP" cannot distinguish "DFA
   learned" from "this task is trivially easy", which is the trap that
   inflated our own early MoE numbers. Bake it into the gate so it
   cannot be skipped.

## The one thing I think is still ambiguous — and what I am doing

tao#19 spells the control "**random B, never updated**". Taken
literally that IS standard fixed-DFA, so the arm would be identical to
the thing it is meant to control for.

Read against the discipline it cites (F9d / toy#141), the control that
actually does the job is **frozen WEIGHTS**: the DFA-policied layers
stay at init and only the head trains. That is what separates "DFA
taught the hidden layers something" from "the head alone could do this
task" — and it is exactly the shape of toy#141, where the frozen arm
BEAT both dfa and chain (frozen 7.928 < dfa 7.970 < chain 8.068), i.e.
training the experts at all was worse than leaving them alone.

**Decision: the frozen control is FROZEN WEIGHTS** (hidden layers at
init, head trains), matched init + seed. If Tao meant the literal
random-B reading, that arm is definitionally the DFA arm and the gate
would be vacuous — worth one round-trip to confirm, but not worth
blocking on, because the frozen-weights arm is strictly the stronger
control and subsumes the intent.

## toy#152 (T0 anchor) — what it needs

The anchor matters more than its size suggests: the F4–F14 body of work
is entirely NEGATIVE results for DFA. If the harness cannot reproduce a
KNOWN DFA positive at small output dim, those negatives are not
findings, they are potentially harness artifacts. This ticket is the
control for everything already shipped.

- New recipe `mlp` + runner (CPU only): N-layer MLP, `n_classes` head,
  cross-entropy.
- Data: synthetic gaussian blobs, deterministic from seed. No data
  plumbing, no fixture to pin, reproducible across machines.
- Per-layer `--policy chain|dfa`. NO `--policy-scope`.
- DFA form is the canonical one and the reason this lane is the
  cheapest: `g_l = a_inᵀ (B_l e)ᵀ` with `B_l` shaped
  `[n_classes, d_out]` — the SMALL output dim is the whole point, and
  `Toy::Train::DfaB` (fill / sigma_for) is already shared, so no new
  feedback machinery is needed.
- Events: `align` (with `wname`), plus loss AND **val accuracy** —
  accuracy is the metric the success bar is stated in.
- Arms: all-BP, all-DFA, frozen-hidden. Matched init + seed.
- Sweep `--classes {2, 10, 100, 1000}` — the output-dim degradation is
  the measured claim, not just the anchor.
- Gate: the mandatory bar above, plus determinism, plus `chain`
  byte-null vs absent.

## toy#152 — SHIPPED, and what it measured

`toy train mlp` (runner `libexec/toy-train-mlp`, engine
`lib/toy/llm/engine/mlp_engine.rb`, gate `prep/mlp_gate.rb`). Arms are
per-hidden-layer policy tokens `chain | dfa | frozen`; the head always
trains by BP (at the output layer DFA and BP coincide). CPU-only, own
compilation unit, `MLP_*` env namespace.

**The anchor holds.** At 3×64 hidden / 32 features / batch 64 / 1000
steps / lr 0.003, matched init + seed, val accuracy on 2048 held-out
samples (seed 0; the numbers the gate asserts):

| classes | all-BP | all-DFA | frozen | recovery |
|---------|--------|---------|--------|----------|
| 2       | 0.940  | **0.942** | 0.772 | 1.01 |
| 10      | 0.773  | 0.742   | 0.516  | 0.88 |
| 100     | 0.559  | 0.412   | 0.273  | 0.49 |
| 1000    | 0.411  | 0.265   | 0.162  | 0.42 |

*recovery* = (DFA − frozen) / (BP − frozen): the share of BP's
over-the-frozen-control gain that DFA recovers. Raw accuracy is not
comparable across output dims (chance is 1/C); this is.

Three results, all first-of-their-kind for this program:

1. **A POSITIVE.** At 2 classes DFA matches BP (and edges past it); at
   10 it is 3 points behind BP and 23 ahead of the frozen control. The
   harness CAN reproduce a known DFA success, so the F4–F14 negatives
   are findings about transformer LMs at vocab 50257, not harness
   artifacts.
2. **The output-dim lens, MEASURED.** Recovery falls monotonically
   1.01 → 0.88 → 0.49 → 0.42 across {2, 10, 100, 1000} classes, on
   every seed tried (0/1/2). The lens was a failure *explanation*; it
   is now a *prediction that was tested*, on the same feedback
   machinery (`Toy::Train::DfaB`) the franken lanes use — only the
   output dim differs.
3. **THE MECHANISM IS VISIBLE, not just the outcome.** With
   `--align-events`, cos(g_DFA, g_BP) on the policied weights starts at
   ~0 and climbs — seed 0, 1000 steps: w1 −0.002 → 0.67, w2 −0.089 →
   0.60, w3 −0.04 → 0.31. That is Refinetti's "align, then memorise"
   reproduced in our own telemetry, and it is what the gate asserts:
   a broken DFA wiring can still move a loss curve, but it cannot make
   the shadow gradient rotate towards a fixed random matrix. Note the
   ordering — the layer nearest the head (w3) aligns least, which is
   the same depth signature the F-series saw.

The BP gap at 10 classes is ~3 points and stable across seeds (.031 /
.029 / .029 for seeds 0/1/2); the gate's threshold is set ~3× above
that spread rather than at seed 0's value.

**One thing worth knowing before you read a big-#classes row: the val
set has to be big enough.** At the runner default of 512 held-out
samples the 1000-class recovery is noise-dominated — its numerator and
denominator are both differences of ~0.1-accuracy estimates — and the
100 → 1000 comparison flipped sign between two honest val sets. At
2048 samples the ordering is stable on every seed tried. Sweeps at
`--classes 100+` should pass `--val-batches 32`.

Also measured, so nobody re-runs it: under AdamW the **B scale
axis barely moves this lane** (inv_sqrt_fan / glorot / fixed:1.0 land
within a point) — Adam normalises per-parameter magnitude, so only B's
direction carries information. And DFA tolerates less LR than BP: at
lr 0.03 the DFA val loss blows up to 78 while BP is unbothered.

### Two deviations from the ticket text, both deliberate

- **The task is a random-teacher network, not gaussian blobs**
  (`--task blobs` still exists, and its degeneracy is MEASURED, not
  assumed: at the anchor cell blobs give BP 1.000 / DFA 0.9995 /
  **frozen 0.992**). Isotropic blobs are LINEARLY SEPARABLE, so the
  frozen control — a random hidden stack plus a trained linear head —
  scores as well as anything else, and the mandatory frozen-beat half
  of the success bar could never be met by *anything*, DFA or BP. A
  blob anchor would have been vacuous by construction, which is
  exactly the failure mode tao#19 wrote the bar to catch. Gaussian
  inputs with labels from a fixed random ReLU teacher is Refinetti et
  al.'s own setup — the paper toy#152 cites — keeps every property the
  ticket asked for (synthetic, seeded, no plumbing), and makes the
  hidden layers load-bearing.
- **The DFA rule includes the ⊙ f'(a) factor** the franken lane omits.
  Franken's per-matmul surrogate projects the error straight onto the
  weight; the anchor uses the literature rule (`ggml_silu_back` gives
  the exact derivative). If the anchor had failed with the surrogate
  we could not have told "DFA does not work at small output dim" from
  "our surrogate is not DFA" — which is the one question this ticket
  exists to answer.

### For Tao (F16)

Each cell is one process, ~0.5 s at 1000 steps:

```
toy train mlp --steps 1000 --seed 0 --classes 10 --val-batches 32 --policy chain,chain,chain
toy train mlp --steps 1000 --seed 0 --classes 10 --val-batches 32 --policy dfa,dfa,dfa --align-events
toy train mlp --steps 1000 --seed 0 --classes 10 --val-batches 32 --policy frozen,frozen,frozen
```

The three arms differ ONLY in `--policy`; init, task, and the held-out
set are identical by construction (the val set is materialised from the
head of one sample stream and training continues from what follows, so
train/val disjointness does not depend on two seeds happening to miss
each other in the same LCG cycle).

`val: acc=… loss=… n=…` rides stdout; events carry per-step
`train_acc`, the `align` events (with `wname`), and one end-of-run
`eval` event. run_start carries a `dfa` object (NOT `franken` — a
consumer keying on `franken` would read an MLP run as a transformer
one) with the policy, the B axes, and the realised `dfa_wired` /
`frozen` counts.

## toy#158 (F15) — SHIPPED: macro-DFA + RAdam on the dense franken

Recipe hygiene, not a new mechanism. Every transformer-LM negative we
have (F4–F14) used **micro** DFA — per-weight `--policy dfa`, lr 1e-3,
AdamW. LightOn's *working* recipe (arXiv:2006.12878) is **macro** DFA
(random feedback injected only at block outputs, full BP inside the
block) at lr 5e-5 with a RAdam-class optimizer. Until that is run, our
negatives are not clean.

`toy train franken --policy dfa,dfa --dfa-granularity block`
(`FRANKEN_DFA_GRANULARITY=block`), plus `--optimizer radam`.

**How the cut is built.** `llama_arch.rb` detaches the residual stream
at every block boundary and records each block's undetached output;
`llama_seq_engine.rb` attaches one surrogate root per block,
`L_l = sum(tap_l ⊙ B_l·e)` with `e` detached — whose gradient *at the
tap* is exactly the random-projected output error, so autodiff then
does ordinary BP inside the block and stops at the cut. The CE root
stays live and trains the final norm + lm_head, which is how LightOn
trains them too. The embedding boundary is deliberately **not** cut:
block 0's surrogate propagates into the embedding, matching a tinydfa
`DFALayer` stack.

**Three properties, gated separately** — each can break without
breaking the others, and a run that is only two of the three is a
hybrid wearing a recipe's name:

1. *The forward is unchanged.* `tnn_detach` is forward-identity, so
   step 1 is **byte-identical to BP** (6.464970588684082 either way).
   That is what makes macro-vs-BP numbers comparable at all.
2. *The backward is different.* Steps 2+ diverge.
3. *The blocks really train from the injected error.* Changing
   `--dfa-b-seed` moves the curve — if the surrogate roots were not
   reaching the weights, the blocks would be frozen and the feedback
   seed could not matter. This is the assertion that would catch a
   silently-unwired macro build, which curve-watching would not.

**First numbers** (2-layer gate shape, 30 steps, seed 0): BP 3.514,
micro-DFA 3.541, **macro-DFA 5.076**, macro+RAdam 5.976. Note micro's
near-BP number is an artifact of scope — at the default `attn` scope
micro policies only attention qkv, so most of the net is still BP,
while macro policies the *entire stack*. **They are not comparable
arms**, which is exactly the confusion F15 exists to remove (and a
large part of why F1 "washed out"). Expect F15 to stay NEGATIVE at
scale: LightOn's own LM result is ppl 52 (DFA) vs 34 (BP).

**RAdam is an LR multiplier, and where that is not RAdam.** `r_t` is a
per-step scalar, so in the rectified regime "AdamW at lr·r_t" *is*
RAdam exactly — no engine work, no new kernel. In the un-rectified
early regime Liu et al. take a non-adaptive momentum step; we take
**no step** (`r_t = 0`). At β2=0.999 that is the first 4 steps, so a
5-step smoke under `radam` looks frozen — the runner therefore
**announces** `first_stepping_step=5` at startup rather than letting
someone debug a flat curve. `--optimizer radam` also carries
β2 = 0.999 as part of its identity (ρ∞ is a function of β2), which is
a real numerics change vs `adamw` and is recorded in run_start.

**CPU-only (tao#18), and the silent-ignore trap closed.** The CUDA
franken runner is *hand*-mirrored and does not implement macro, so
`--device cuda --dfa-granularity block` would have run micro DFA and
recorded it as macro. Rejected in the CLI *and* fails loud in the CUDA
runner itself, for callers that bypass the CLI.

**Drive-by fix.** The post-realize B upload in `llama_seq_engine.rb`
hardcoded `nb = d_head * vocab`, ignoring the per-weight `b_douts`
toy#151 introduced — masked only because the runners call
`franken_refresh_b!` before every step. toy#158's d_model-wide macro
taps would have widened the same trap, so it now reads `b_douts`.

## toy#154 (F18/T1) — SHIPPED, and it does NOT reproduce near-parity

`toy train ctr` (runner `libexec/toy-train-ctr`, engine
`lib/toy/llm/engine/ctr_engine.rb`, gate `prep/ctr_gate.rb`): per-field
embedding tables → concat → MLP tower → **scalar** sigmoid head →
logloss, per-tower-layer `--policy chain|dfa|frozen`, metric AUC
(exact Mann-Whitney, ties at 0.5). CPU-only.

**The construction had to be toy#158's, not toy#152's.** The ticket
asks for "DFA the tower, embeddings stay chain" — and toy#152's
direct-gradient rule propagates *nothing*, so the tables below the
tower would have sat at init. This lane therefore uses a surrogate
loss root per policied tower layer; the lowest one's gradient
continues into the embeddings, which is exactly what "the tables train
by backprop" means. The gate asserts it directly (the frozen-tower arm
still improves 0.529 → 0.620 AUC, which is only possible if the tables
are learning through the frozen tower).

**Result, 2000 steps, 40 teacher pairs, matched init + seed, 2048
held-out rows:**

| seed | all-BP | all-DFA | frozen | BP−DFA | DFA−frozen |
|---|---|---|---|---|---|
| 0 | 0.706 | 0.639 | 0.620 | .067 | .019 |
| 1 | 0.701 | 0.643 | 0.619 | .058 | .024 |
| 2 | 0.697 | 0.638 | 0.617 | .059 | .022 |

**The ticket's success target — DFA within ~0.01 AUC of BP — is NOT
met.** DFA clears the frozen control convincingly and reproducibly, but
lands 6 points behind BP, not 1. LR is not the explanation (swept
0.0003 / 0.001 / 0.003 / 0.01; 0.003 is DFA's *best*).

### Why, and why this does not contradict LightOn

At output dim 1 the DFA update is **provably rank-1**:

    delta_l = B_l · e  with e a [1, B] scalar error
            = b_l ⊗ e            (a FIXED vector times a per-sample scalar)
    grad W_l = b_l ⊗ ( sum_i e_i a_i )

so every policied layer can only ever move its output along **one
fixed direction** `b_l`; the rest of the layer stays at random init.
Note this is equally true at 2 classes (softmax CE makes the two error
rows negatives of each other), where toy#152 found DFA *matching* BP —
so rank-1 feedback is sufficient when the hidden layers only need a
simple transformation, and insufficient when they must synthesise
genuinely new features. **The output-dim lens is really two effects**:
fewer directions to align (helps) versus less information per step
(hurts), and this lane is where the second one bites.

**The reconciliation is measurable.** DeepFM — the architecture the
ticket cites — is an FM branch **plus** a DNN tower, summed. With the
FM branch enabled (`--fm-branch`, first-order + the exact second-order
pairwise term):

| arm | AUC |
|---|---|
| BP tower + FM | 0.806 |
| **frozen tower + FM** | **0.813** |
| DFA tower + FM | 0.651 |

**The frozen tower scores as well as the trained one.** On a DeepFM
shape the tower contributes ~nothing here, so a "DFA ≈ BP on DeepFM"
observation is *not* evidence that DFA trains towers well — the FM
branch is doing the work. That is the most likely reading of the
literature number, and it is worth knowing before F18 is designed
around it.

Third: in that regime DFA is **16 points worse than frozen** — the
toy#141 shape again (frozen experts beat both dfa and chain). A
DFA-trained module summed into a path a BP-trained component already
handles is net-harmful, and that has now happened on two independent
lanes.

### The task had to be built to discriminate

At the shipped defaults (12 pairs, `--lin-scale` 1.0) the additive part
of the teacher dominates, embeddings + head alone capture nearly all of
it, and the three arms land within **0.003 AUC with the frozen control
ahead** (.795 frozen / .793 dfa / .792 chain) — an unfalsifiable bar,
the toy#152-with-blobs failure exactly. `--lin-scale 0.25` with 40
pairs makes the crosses carry the signal. The gate asserts
BP − frozen > 0.05 *first*, so a future change that quietly makes the
task easy fails loudly instead of reporting a free pass.

## toy#153 (F17/T1) — SHIPPED, and it is the program's SECOND positive
## — the first where DFA BEATS BP

`toy train gnn` (runner `libexec/toy-train-gnn`, engine
`lib/toy/llm/engine/gnn_engine.rb`, task `lib/toy/io/toy_gnn_task.rb`,
gate `prep/gnn_gate.rb`): message passing over the symmetric-normalised
adjacency `S = D^-1/2 (A+I) D^-1/2`, per-layer `--policy
chain|dfa|frozen`, a MASKED cross-entropy over the labelled nodes only,
and DFA-GNN's structure-aware feedback as `--feedback-route structure`.
CPU-only. Runs on a seeded contextual-SBM graph by default and on **the
real Cora citation graph** via `GNN_GRAPH=data/gnn_cora`
(`ruby prep/fetch_cora.rb`, one 168 KB download).

**Result on Cora** (canonical 2-layer GCN: 1 hidden layer + head, width
64, lr 0.01, 100 steps, matched init + seed, val = all 2568 non-training
nodes):

| seed | all-BP | all-DFA | DFA(structure) | frozen |
|---|---|---|---|---|
| 0 | 0.697 | **0.762** | 0.764 | 0.618 |
| 1 | 0.661 | **0.759** | — | 0.590 |
| 2 | 0.643 | **0.760** | — | 0.524 |

**The ticket's success target — all-DFA matches BP node accuracy on
Cora — is MET and exceeded.** DFA is above BP on every seed and clears
the frozen control by .14–.24. This is the program's first lane where
DFA does not merely approach BP.

### Four things that keep it honest

1. **BP is not under-tuned, and it is not stopped at the wrong
   moment.** lr 0.01 is BP's OWN best on this graph (BP val .351 /
   .661 / .697 / .668 / .593 at lr .001 / .003 / .01 / .03 / .1).
   Weight decay does not rescue it (`GNN_WD` swept 0 → .05 moves BP
   .695 → .698). And the step budget is not the explanation either —
   **DFA is ahead at every budget**, and gets there far faster:

   | steps | all-BP | all-DFA | DFA(structure, 3 hops) | frozen |
   |---|---|---|---|---|
   | 20  | 0.405 | **0.768** | 0.778 | 0.582 |
   | 50  | 0.668 | **0.769** | 0.775 | 0.610 |
   | 100 | 0.697 | **0.762** | 0.773 | 0.618 |
   | 200 | 0.695 | **0.716** | 0.733 | 0.618 |

   BP's best (.697 @100) is 7 points under DFA's best (.769 @50), and
   at 20 full-graph steps DFA is already at its peak while BP is barely
   off the floor. Whatever DFA is doing on this architecture, it is
   also enormously more step-efficient with 140 labels.
2. **DFA is dramatically more seed-stable than BP** — .759–.762
   (spread .003) against BP's .643–.697 (spread .054). That is the
   signature of a regulariser, not of a better gradient.
3. **THE ALIGNMENT PHASE DOES NOT HAPPEN HERE.** toy#152's MLP anchor
   reproduced Refinetti's "align, then memorise" (cos(g_dfa, g_bp)
   climbing to .6 and staying). On this lane cos is transient at best —
   seeded graph seed 0 peaks at +.359 by step 61 then falls to −.124,
   seed 2 never leaves the noise — and on Cora it never exceeds +.096
   and **ends NEGATIVE (−.057)**. So on the architecture where DFA
   BEATS BP, the DFA update is mildly ANTI-correlated with the BP one.
   Whatever is working here, it is not DFA approximating backprop. That
   is the ticket's own skeptic hypothesis (an implicit
   regulariser/architecture effect) surviving its first real test, and
   `prep/gnn_gate.rb` asserts it so a future change that makes cos
   converge fails loudly instead of quietly rewriting the mechanism.
4. **Our BP is below a fully-regularised published GCN** (~.815 with
   dropout + weight decay + width 16 + early stopping). So the claim is
   "DFA beats BP AT MATCHED RECIPE", not "DFA beats a tuned GCN" —
   DFA's .762 is also below .815.

**Structure-aware feedback is real but small.** On Cora it adds
+.002 (1 hop) → +.007 (2) → +.011 (3) over vanilla DFA; on the seeded
graph at lr .003 it adds +.07. The big win is DFA itself, not the
graph-aware twist DFA-GNN is named for.

### The seeded graph cannot carry the bar, and that is measured

The mandatory success bar presupposes that **BP** beats the frozen
control — otherwise the cell says nothing about anybody
([[control-arm-must-be-able-to-lose]]). On the seeded graph BP − frozen
FLIPS SIGN across seeds (+.086 / +.033 / +.020 at 300 steps; +.034 /
−.012 / +.021 at 600; negative at features 128). In a GNN,
**neighbourhood aggregation is ARCHITECTURE, not learning** — a frozen
random hidden stack still smooths features over the graph — so on a
synthetic graph whose labels are largely recoverable from that
smoothing, training the hidden layers buys almost nothing.

Cora escapes this because its 1433-dim bag-of-words forces the hidden
layer to **compress**, and a random projection loses what a learned one
keeps. That is the single property that makes a frozen control able to
lose in this architecture, and it is why the seeded defaults are
features 64 / hidden 32 rather than the other way round. The gate
states the bar on Cora (a Makefile prerequisite, not a skippable leg)
and pins the seeded graph's own limit in a separate leg.

`--task community` (labels == community id) is this lane's `blobs`: a
FROZEN net already scores .922 against BP's .982, so no
credit-assignment rule can be distinguished near that ceiling. Selectable,
measured, and asserted — not assumed.

### Two construction notes

- **The first propagation is folded into the host preprocessing.** The
  engine is handed `S·X`, not `X`. `S` and `X` are both constant so
  `S(XW) == (SX)W` exactly, and it keeps an N²×feat_dim matmul out of
  every step — on Cora the difference between 10.5 GFLOP and 0. It
  changes no gradient: neither BP nor DFA differentiates a constant.
- **The propagation sits BEFORE the weight in each layer**, so no `S`
  appears in the DFA backward except the one `--feedback-route` puts
  there. That is what makes direct-vs-structure a clean contrast — and
  the gate pins it with an identity: at `--degree 0` the graph is
  edgeless, `S = I`, `S^k e == e`, so `structure` must be
  BYTE-IDENTICAL to `direct` there and differ once edges exist. One
  assertion proves both "the hops are applied" and "they go through the
  adjacency".

### For Tao (F17)

```
ruby prep/fetch_cora.rb        # once; 168 KB, cached in data/
toy train gnn --graph data/gnn_cora --layers 1 --hidden 64 \
  --lr 0.01 --steps 100 --seed 0 --policy chain
toy train gnn ... --policy dfa
toy train gnn ... --policy dfa --feedback-route structure --feedback-hops 3
toy train gnn ... --policy frozen
```

Arms differ ONLY in `--policy` / `--feedback-route`. `train:` and
`val:` lines ride stdout (the train/val GAP separates the arms as much
as val accuracy does); run_start carries a `dfa` object with
`feedback` + `feedback_hops`, and `cost.propagation_flops_per_step`
separately from the params-only count, because message passing is
parameter-free and a params-only cost reads this lane as far cheaper
than it is.

**CiteSeer/PubMed are NOT wired.** `prep/fetch_cora.rb` writes a
generic 5-file bundle and the loader is dataset-agnostic, so CiteSeer is
a fetcher away. PubMed (19717 nodes) is NOT reachable with a dense
[N,N] adjacency — 389 M floats and 25 GFLOP per propagation — and would
need a sparse matmul in the shim. Say so rather than quietly reporting
two of three.

## toy#155 (F19/T2) — SHIPPED: the ticket's structural argument, INVERTED

`toy train ssm` (runner `libexec/toy-train-ssm`, engine
`lib/toy/llm/engine/ssm_engine.rb`, task `lib/toy/io/toy_ssm_task.rb`,
gate `prep/ssm_gate.rb`): a stack of channel-wise **selective** linear
recurrences (input-dependent decay + causal depthwise conv + gating), a
small-output sequence-classification head reading the LAST timestep, and
per-layer `--policy chain|dfa|frozen` on a `--dfa-cut layer|step` axis.
CPU-only.

**The recurrence is unrolled from differentiable primitives, and that is
not a shortcut.** ggml ships fused SSM_SCAN/SSM_CONV and the shim
exposes them, but `ggml_compute_backward` covers 43 ops and neither is
among them — they are inference-only and abort under autodiff. It is
also the honest construction: the BPTT graph the DFA arm claims to avoid
has to exist before avoiding it means anything.

### The result: DFA matches BP, but ONLY at the layer cut

Delayed-cue task (one marked cue in the first quarter, noise everywhere
else, readout at step 64), 600 steps, matched init + seed:

| selection | seed | BP | **DFA (layer cut)** | DFA (step cut) | frozen |
|---|---|---|---|---|---|
| selective | 0 | 1.000 | **0.996** | 0.250 | 0.227 |
| selective | 1 | 0.992 | **0.988** | 0.250 | 0.227 |
| selective | 2 | 0.988 | **1.000** | 0.238 | 0.238 |
| lti | 0 | 0.750 | 0.645 | 0.645 | 0.305 |
| lti | 1 | 0.730 | 0.621 | 0.668 | 0.270 |
| lti | 2 | 0.738 | 0.656 | 0.617 | 0.266 |

**The success bar is MET at the layer cut** — DFA tracks BP to within
.012 and clears the frozen control by .76. That is the program's third
positive, on the architecture with no DFA precedent at all.

### And the ticket's structural argument comes out backwards

toy#155's thesis: *"the SSM core is a LINEAR RECURRENCE, and a
random-feedback error injected per step composes through the same linear
operator, so random-feedback credit assignment is mathematically natural
here."* Its caveat: *"in the purely-LINEAR case FA collapses to plain
gradient descent, so the interesting alignment MUST live in the
NONLINEAR parts."*

Read the DFA(step) column against that. **Cutting BPTT entirely is FREE
in the linear model** (step .645 vs layer .645 — equal to within noise)
**and CATASTROPHIC in the selective one** (.99 → chance). And it is not
an LR artifact: swept 3e-5 … 1e-2, the best per-step cell is .355
against a .227 frozen control and a 1.000 BP.

So the linearity that makes random feedback compose neatly is exactly
what makes the per-step cut cost nothing — and the selection that makes
the model *good* is exactly what makes it fail. **The kill-BPTT value
prop is available only in the regime where the model cannot do the
task.** That is a sharper statement than "DFA works here" and it is what
F19 should be designed around.

Leg 7 of the gate pins BOTH halves, because either alone is
misreadable: the lti equality alone looks like "the step cut works", the
selective collapse alone looks like a broken build. Together they are
the finding — and the lti equality doubles as the proof that the step
cut is correctly *wired*, since a dead cut could not learn at all.

### The activation-memory claim does NOT hold up in this harness

The ticket's success target is "matches BP at k-times-less activation
memory". Measured (realized graph nodes; every arm linear in T):

| T | BP | DFA (layer) | DFA (step) |
|---|---|---|---|
| 16 | 814 | 858 | 1077 |
| 32 | 1646 | 1722 | 2165 |
| 64 | 3310 | 3450 | 4341 |
| 128 | 6638 | 6906 | 8693 |

The step cut's graph is **31% BIGGER**, not smaller — the detach dups
and the per-step surrogate terms are net additions. In a graph-based
autodiff every forward tensor is materialised whatever the credit rule,
so the streaming memory win DFA promises is not expressible as a graph
rewrite; it needs a streaming implementation. `cost.graph_nodes` carries
that caveat in the bundle so nobody reads the number as bytes saved.

### Two construction findings worth carrying to F20/F21

1. **A per-step linear surrogate needs a tap that can change the
   prediction.** The first build tapped the layer output at every step;
   with a last-step readout, the FINAL layer's `o_t` for t < T-1 has no
   functional path to the prediction at all, so the global error is not
   a proxy target there — it is not a target at all, the surrogate has
   nothing to balance it, and the weights run away (measured: loss 1e24,
   then NaN). The state `h_t` is the only per-step quantity that always
   has such a path, because it is what propagates forward in time. This
   is a general rule for injecting DFA into any recurrent or
   late-readout architecture.
2. **The unreachable-input landmine fired again** (toy#154, franken-moe
   `t_hp` under `--optimizer sgd`): under the layer cut there are no
   state taps, so that tap family's feedback matrix is consumed by
   nothing, gets NO backend buffer, and the upload aborts inside
   `ggml_backend_tensor_set`. Allocate a B only if its tap family is
   non-empty.

`--task mean` is this lane's `blobs`: the class signal is spread over
every step, so neither memory nor selection is needed and a FROZEN
recurrence already integrates it. Selectable, measured, asserted.

### For Tao (F19)

```
toy train ssm --steps 600 --seed 0 --policy chain,chain
toy train ssm --steps 600 --seed 0 --policy dfa,dfa                    # layer cut
toy train ssm --steps 600 --seed 0 --policy dfa,dfa --dfa-cut step
toy train ssm --steps 600 --seed 0 --policy frozen,frozen
#  ... and the whole grid again under --selection lti
```

Arms differ ONLY in `--policy` / `--dfa-cut` / `--selection`. **No
`--align-events` on this lane**, and that is structural: the DFA update
arrives through autodiff from the surrogate roots, so it lands in the
SAME accumulator a BP run would use and there is no second tensor to
take a cosine against. The CLI rejects the flag rather than emitting
telemetry that silently means nothing. Gate on the B seed instead.

**The long-sequence CUDA twin is still not built** and, given the graph
measurement above, it would not show the memory win either — that needs
a streaming implementation, not a device port. Worth deciding before
anyone spends the twin.

## toy#156 (F20/T2) — SHIPPED: DFA BEATS BP at latent 4, and the lens bites earlier on a generative objective

`toy train diff` (runner `libexec/toy-train-diff`, engine
`lib/toy/llm/engine/diff_engine.rb`, task `lib/toy/io/toy_diff_task.rb`,
gate `prep/diff_gate.rb`): a time-conditioned eps-prediction MLP
denoiser over a LOW-DIM latent, per-layer `--policy chain|dfa|frozen`,
scored by a GENERATIVE metric. CPU-only. The DFA rule is toy#152's
canonical direct-gradient one (with the (*)f' factor), so a result here
is comparable to the MLP anchor straight across — only the output
dimension and the loss differ.

**The metric is generative, not denoising MSE**, because the ticket asks
for that and because the two genuinely disagree: a denoiser can have a
respectable MSE and generate nothing like the data. The headline is the
ENERGY DISTANCE between `eval_n` ancestrally-sampled points and `eval_n`
held-out reals — a proper metric, so **LOWER IS BETTER** and the success
bar inverts. A slightly negative value means "indistinguishable at this
sample size" (the estimator is unbiased), not a bug.

### The result: a POSITIVE, and the lens boundary

All three arms at ONE shared learning rate (3e-4), 10000 steps, so the
arms differ ONLY in `--policy`:

| latent | seed | all-BP | **all-DFA** | frozen |
|---|---|---|---|---|
| 4 | 0 | 0.102 | **0.010** | 0.590 |
| 4 | 1 | 0.184 | **0.135** | 1.084 |
| 4 | 2 | 0.131 | **0.072** | 1.005 |
| 16 | 0 | 0.130 | 0.979 | 9.020 |

**At latent 4 DFA BEATS BP on every seed** while both crush the frozen
control — the ticket's success target met and exceeded. At latent 16
DFA is 7.5x WORSE than BP (15-27x at each arm's own best LR across
seeds), while still beating frozen 9x.

The full lens at each arm's own best LR, 10000 steps, seed 0:

| latent | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| BP | -0.013 | 0.085 | 0.079 | 0.036 | 0.399 |
| DFA | 0.007 | 0.010 | 0.195 | 0.979 | 2.746 |
| frozen | 0.029 | 0.226 | 1.053 | 6.761 | 45.451 |

DFA's energy rises monotonically with the output dim and it beats the
frozen control at every one. **The boundary sits between 4 and 16 —
much lower than the classification lens, where toy#152 was still
comfortable at 10 classes.** The same lens applies to a generative
objective but bites earlier, which is worth carrying into F20's design.

At latent 2 the frozen control is nearly as good as BP (0.029 vs
-0.013), i.e. that cell is degenerate — the lens boundary and the
control-can-lose boundary are adjacent, which is exactly why the bar is
stated at latent 4.

### A schedule bug that was SILENT IN THE LOSS

Ho et al.'s betas (1e-4 ... 0.02) assume T = 1000. At this lane's
T = 100 they leave **abar_T = 0.60** — the forward process never reaches
noise, so the ancestral sampler, which starts from pure N(0, I), starts
OUT OF DISTRIBUTION. Measured at those defaults: BP had the **best**
denoising MSE (.684) and the **worst** energy distance (29.1, against
4.95 for an untrained net), with a perfectly healthy training curve
throughout. The defaults are now 1e-3 ... 0.15 (abar_T = 3.5e-4) and the
runner REFUSES to start if abar_T > 0.01, naming the fix. Gated.

`--task single` (one Gaussian instead of the mixture) is this lane's
`blobs`: chain -.0020 / dfa -.0011 / frozen -.0012, i.e. a FROZEN
denoiser reproduces it as well as a trained one. Measured, asserted.

### For Tao (F20)

```
toy train diff --steps 10000 --seed 0 --lr 0.0003 --latent 4 --policy chain,chain,chain
toy train diff ... --policy dfa,dfa,dfa
toy train diff ... --policy frozen,frozen,frozen
#  ... and again at --latent 8 / 16 / 32 for the lens
```

Arms differ ONLY in `--policy`. **Use lr 3e-4 for arm comparisons**:
BP's own best is 3e-3 (energy .036 at latent 16) but DFA diverges there
(4.6e6 at 1e-2), so a 3e-3 cell compares a converged BP against a broken
DFA. 3e-4 is the highest rate both arms tolerate — and DFA tolerating
less LR than BP is toy#152's finding restated. **10000 steps is also
load-bearing**: at 3000 steps BP has not converged at 3e-4 (latent 16:
BP .802 vs DFA .947, which reads as parity) and the lane's result would
invert.

This lane DOES carry `--align-events` (unlike ssm): it uses the direct
rule, so the DFA update is a separate tensor from the chain shadow and
cos(g_dfa, g_bp) is meaningful. It climbs, as in toy#152.

## toy#157 (F21/T3) — SHIPPED: the per-step cut HOLDS on a gated LSTM and COLLAPSES on the selective scan

`toy train lstm` (runner `libexec/toy-train-lstm`, engine
`lib/toy/llm/engine/lstm_engine.rb`, recipe
`lib/toy/llm/recipes/lstm_seq.rb`, gate `prep/lstm_gate.rb`): a stack of
textbook LSTM cells (i/f/o/g, 12 weights per layer) UNROLLED over T, a
small-output head reading the LAST timestep, per-layer
`--policy chain|dfa|frozen` on the same `--dfa-cut layer|step` axis
toy#155 introduced. CPU-only.

**It reuses toy#155's delayed-cue generator UNCHANGED**
(`lib/toy/io/toy_ssm_task.rb`). Holding the task fixed is what makes the
two lanes an architecture comparison instead of two anecdotes, and it is
why the contrast below can be read straight across.

### The headline: same cut, same task, opposite outcome

Each lane at its own fair cell, seed 0 (the LSTM row's other two seeds
are in the next table):

| architecture | BP | DFA (layer cut) | **DFA (step cut — no BPTT at all)** | frozen |
|---|---|---|---|---|
| selective scan (toy#155) | 1.000 | 0.996 | **0.250 — chance** | 0.227 |
| **gated LSTM (this lane)** | 1.000 | 0.242 | **1.000** | 0.266 |

On the SSM the per-step cut destroyed the task; on the LSTM it solves
it. The gates are the difference: an LSTM's forget gate can hold the
cue in `c_t` without any gradient crossing a timestep, so a per-step
random-feedback update has something to sharpen. The selective scan's
input-dependent decay — the part that makes it good — is exactly the
part that needs credit assigned ACROSS time. **"Can you delete BPTT?"
has no architecture-independent answer**, and that is what this lane was
built to establish. It agrees with Folchini et al. (ISC-HPC 2025).

### The bar: at BP's own best cell, on all three seeds

lr 0.02 + warmup 200, 4000 steps — the BP arm's best over a swept
(lr x warmup x seed) grid, so "matches BP" means matching a BP that
trained. Matched init, arms differ only in `--policy` / `--dfa-cut`:

| seed | BP | DFA (layer cut) | **DFA (step cut)** | frozen |
|---|---|---|---|---|
| 0 | 1.000 | 0.242 | **1.000** | 0.266 |
| 1 | 0.992 | 0.258 | **0.996** | 0.254 |
| 2 | 1.000 | 0.988 | **1.000** | 0.238 |

**The success bar is MET, and by the arm that keeps NO BPTT at all**:
the per-step cut tracks BP to within .004 on every seed and clears the
frozen control by .74. Note which arm fails: `--dfa-cut layer`, the one
that keeps BPTT *inside* the layer and injects the random feedback once,
is at chance on two of three seeds. It gets the worst of both — the
instability of 64-step BPTT plus a single-tap error signal — and that
ordering (step cut > BP > layer cut in robustness) is the opposite of
the intuition that more true gradient is safer.

### The second finding: BPTT is the FRAGILE arm here

Every arm on this task is bimodal — it either solves it (~1.000) or sits
at chance (.250 on 4 classes). Val accuracy at 4000 steps, no warmup:

| lr | BP (seeds 0/1/2) | DFA `--dfa-cut step` (seeds 0/1/2) |
|---|---|---|
| 0.005 | 0.504 / 1.000 / 0.996 | 1.000 / 1.000 / 1.000 |
| 0.010 | 0.250 / 1.000 / 0.996 | 1.000 / 1.000 / 1.000 |
| 0.020 | 0.250 / 0.996 / 1.000 | 1.000 / 0.996 / 1.000 |
| 0.030 | **1.000** / 0.250 / 0.238 | 1.000 / 1.000 / 0.996 |

**The per-step cut solves 12 of 12 cells. BPTT solves 7 of 12, and WHICH
learning rate works depends on the SEED** — seed 0 needs 0.03 and fails
below it; seeds 1 and 2 need 0.005–0.02 and fail at 0.03. Warmup (the
nearest thing this harness has to the gradient clipping that normally
stabilises long BPTT) moves BP but is itself cell-dependent — BP over
seeds 0/1/2 at 4000 steps:

| warmup | lr 0.005 | lr 0.01 | lr 0.02 |
|---|---|---|---|
| 0 | .504 / 1.000 / .996 | .250 / 1.000 / .996 | .250 / .996 / 1.000 |
| 200 | .730 / 1.000 / 1.000 | 1.000 / 1.000 / **.250** | **1.000 / .992 / 1.000** |
| 500 | **.258** / 1.000 / 1.000 | 1.000 / .992 / .996 | 1.000 / 1.000 / 1.000 |

Two cells make BP healthy at all three seeds (warmup 200 @ lr 0.02, and
warmup 500 @ lr 0.02); the bar is stated at the first. Steps alone also
work eventually — seed 0 at lr 0.005 with no warmup reaches .996 at 8000
steps and 1.000 at 16000, i.e. 4x the budget the per-step cut needs.

That asymmetry is not a subtlety of the bar, it IS the result. The arm
with no gradient crossing time never has a gradient to explode, so it is
indifferent to the rate; BPTT through 64 gated steps is threading a
needle. **The honest statement is therefore not "DFA matches BP" but
"DFA matches BP at BP's best cell, and reaches it from anywhere."**

**The caveat this section used to carry — "toy has no gradient clipping,
so some of BP's fragility may be ours" — is RESOLVED, and the claim
survived.** toy#162 built the clipping and it does not rescue BP: cells
solved out of 12 go 7 (unclipped) → 8 (clip 1.0) → 7 (clip 0.1), while
the per-step cut stays 12/12. See the toy#162 section.

### Why the bar is stated over THREE seeds, unlike every other lane

The half-built state of this lane reported the fair cell as lr 0.03 /
2000 steps, where BP reads 1.000 and both DFA arms read ~.99 — a clean
"DFA matches BP". At seeds 1 and 2 that same cell gives BP **.250 and
.270**, i.e. below its own frozen control. A one-seed bar there would
have shipped "DFA ties BP" on seed luck, and a one-seed bar anywhere in
the 0.005–0.02 band would have shipped "DFA BEATS BP" on the same luck
in the other direction. This is the THIRD time this lane nearly shipped
a wrong result from an undertrained BP arm (the first two: a bias-free
LSTM whose carry was dead at init, and the inherited 0.003 default).
Hence: the gate runs all three seeds, and the roadmap states the grid,
not a row.

### The memory claim: measured in the ticket's own units, and NOT met

toy#155 could only report graph NODE COUNT. This lane sums
`ggml_nbytes` over every node of the realized graph — the actual
materialised activation footprint the ticket states its target in:

| T | BP | DFA (layer) | DFA (step) |
|---|---|---|---|
| 16 | 3 312 132 | 3 387 912 | 3 879 552 |
| 32 | 6 638 084 | 6 763 016 | 7 779 072 |
| 64 | 13 289 988 | 13 513 224 | 15 578 112 |
| 128 | 26 593 796 | 27 013 640 | 31 176 192 |

The per-step cut costs **17% MORE bytes**, and the ratio does not
improve with L — the better instrument confirms toy#155's node count.
Same structural reason: in a graph autodiff every forward tensor is
materialised whatever the credit rule, so the streaming win is not
expressible as a graph rewrite. **Filed as tao#21** with both lanes'
tables and three options for what would actually measure it (cheapest:
an ANALYTIC streaming-bytes figure alongside the measured graph bytes).
tao#21 also recommends NOT spending tao#19's deferred F19 CUDA twin: a
device port changes throughput, not what the graph materialises.

### Construction notes

1. **The gate biases are load-bearing and were missing.** Without them
   f_t = sigmoid(0) = 0.5 at init, so the cell state halves every step
   and 0.5^64 is nothing — the carry is dead before training starts.
   Measured in that state: BPTT .199 (below chance) against per-step DFA
   .969, a spectacular-looking "DFA beats BPTT" that was really a
   crippled LSTM. `b_f` now initialises to 1.0 (Jozefowicz et al. 2015).
2. **One tap family is enough here**, unlike the SSM lane: `h_t` IS the
   layer output and it reaches the prediction from every step, so every
   per-step tap is well-posed. toy#155 needed a second family precisely
   because its per-step layer output was inert at the final layer.
3. **`--align-events` is rejected**, as on ssm: the DFA update arrives
   through autodiff from the surrogate roots, so it lands in the SAME
   accumulator a BP run uses and there is no second tensor to take a
   cosine against. The gate proves the feedback reaches the weights via
   the B SEED instead (toy#158's discipline).
4. The runner shipped with `run_start.name = "ssm"` — invisible in the
   curve, and it mislabels every bundle a consumer reads. Fixed, and the
   gate now asserts the lane's own name.

`--task mean` is this lane's `blobs`: the class signal is spread over
every step, so no carry across time is needed and a FROZEN recurrence
plus a trained head already integrates it. Measured, asserted.

### For Tao (F21)

```
toy train lstm --steps 4000 --lr 0.02 --warmup 200 --seed 0 --policy chain
toy train lstm --steps 4000 --lr 0.02 --warmup 200 --seed 0 --policy dfa                  # layer cut
toy train lstm --steps 4000 --lr 0.02 --warmup 200 --seed 0 --policy dfa --dfa-cut step
toy train lstm --steps 4000 --lr 0.02 --warmup 200 --seed 0 --policy frozen
#  ... and the whole grid at seeds 1 and 2, which is NOT optional here
```

**Write the cell out in full, every time.** It is not the default and
must not become one. toy#157 shipped the 200-step ramp as the runner's
default for a single commit; within hours a consumer's
`--lr 0.03 --steps 2000` inherited it and produced a 3-seed matrix under
a cell name that was not the cell it ran. The giveaway was that its
FROZEN row matched byte-for-byte while every trained arm moved — frozen
takes no optimizer step, so it is the one arm invariant to the LR
schedule. Defaults on this lane now change nothing.

Arms differ ONLY in `--policy` / `--dfa-cut`. **Run all three seeds** —
see above for what a one-seed reading of this lane produces. **No
`--align-events`** (rejected by the CLI). The memory half of the success
target is not met and is not measurable in this harness; take it up at
tao#21 rather than re-measuring here.

## toy#159 — the analytic streaming instrument: the memory claim, finally decidable

Both recurrent lanes reported the ticket's memory target as a negative,
in different units and for the same structural reason:

| lane | instrument | per-step cut vs BPTT |
|---|---|---|
| toy#155 | realized graph NODES | 31% MORE |
| toy#157 | realized graph BYTES | 17% MORE, no better with L |

No better measuring fixes that. **In a graph autodiff every forward
tensor is materialised whatever the credit rule**, so a harness that
BUILDS the whole unrolled graph cannot exhibit a streaming win — the
measurement was answering a different question from the one the ticket
asks. tao#21 listed the ways out; this is the cheapest, and the only one
needing no second execution engine.

`lib/toy/train/stream_bytes.rb` computes, from each cell's own shapes,
what a STREAMING implementation would have to hold. Both lanes now print
it beside the measured line:

```
graph:  nodes=2116 bytes=15578112                       # MEASURED — what toy builds
stream: bptt=3868160 sqrt_t=984576 cut=78336 cut_vs_bptt=49.37x cut_vs_sqrt_t=12.56x replay=2x_fwd
```

LSTM lane defaults (hidden 64, batch 32, 1 layer, 4 classes):

| T | BPTT | BPTT + sqrt-T checkpointing | **per-step cut (replay)** |
|---|---|---|---|
| 16 | 968 192 | 501 248 | **78 336** |
| 64 | 3 868 160 | 984 576 | **78 336** |
| 128 | 7 734 784 | 1 467 904 | **78 336** |
| 1024 | 61 867 520 | 3 884 544 | **78 336** |

**Constant in T, as the claim requires** — 49x under BPTT at T=64,
789x at T=1024, and still 12.6x / 49.6x against BPTT's own best
counter-move. That is the "k-times-less activation memory for sequence
length L" the ticket asks for, stated under a model rather than
measured on a harness that cannot exhibit it.

toy#155's lane gets the same instrument, so F19's table can finally be
restated in it (lane defaults: d_inner 48, d_model 24, batch 32,
2 layers, conv K=4):

| T | BPTT | BPTT + sqrt-T | **per-step cut** |
|---|---|---|---|
| 16 | 2 606 592 | 1 353 216 | **213 504** |
| 64 | 10 421 760 | 2 655 744 | **213 504** |
| 128 | 20 841 984 | 3 958 272 | **213 504** |

Two lane-specific details the gate pins. The causal conv's K-1 window
lands in the **carry**, not the per-step term — it is O(K) in the kernel
width, so it does not make the arm O(T). And `--selection lti` reports a
strictly smaller cell (127 488 vs 213 504) because its dt/C/gate
branches are weights rather than per-step activations; the gate asserts
that inequality, since equal figures would mean the instrument is not
reading the selection at all.

**The memory claim and the accuracy claim now point opposite ways on
this lane, and that is the honest F19 summary**: toy#155 measured the
per-step cut collapsing to chance under selection, so the arm that would
buy the 48x is the arm that cannot do the task there. On toy#157's
gated LSTM the same arm solves it — which is why the cross-architecture
contrast, not the memory number, is the transferable result.

### The sharp version, which is not what the tickets assumed

The win is **not a property of the credit rule alone.** The surrogate's
error `e = softmax(logits) − labels` is known only after the last-step
readout, and step t's update needs step t's own activations. So the cut
has exactly two ways to spend itself: keep every step's activations
until the error arrives (O(T) — what this harness does, hence the bigger
graph), or **replay** the forward once the error is known and update
each step as it passes (O(1), at ~2x forward compute).

Replay is legal precisely BECAUSE no gradient crosses a timestep: the
steps are independent and can be revisited in any order. Under BPTT it
is not available at all — the backward must traverse time in order.
**What cutting the time axis buys is the LEGALITY of the replay, not a
smaller graph.** Both numbers ship, with the compute price inline
(`replay=2x_fwd`), because either alone misleads: the measured one reads
as "DFA costs more memory", the analytic one as a free win.

The instrument is honest in the other direction too — it quotes
sqrt-T checkpointing (O(√T) at ~1.5x compute) so the cut is not
compared only against BP's worst case, and it excludes weights,
optimizer moments and gradient buffers, which are equal across arms and
independent of T.

### What it does NOT do

It does not make toy stream. It replaces "the target is unmeasurable
here" with "here is the target's value under a stated model, next to the
measured cost of not implementing it" — enough for F19/F21 to be decided
rather than deferred. It also confirms tao#21's recommendation against
spending the deferred F19 CUDA twin: a device port changes throughput,
not what the graph materialises, and the quantity that moves the claim
is not measured on a device at all.

## toy#160 (F22/T4) — SHIPPED: ATTENTION IS NOT DFA-HOSTILE; it was the output dim

`toy train gtx` (runner `libexec/toy-train-gtx`, engine
`lib/toy/llm/engine/gtx_engine.rb`, task `lib/toy/io/toy_gtx_task.rb`,
gate `prep/gtx_gate.rb`): masked self-attention whose mask **is an
adjacency**, per-block `--policy chain|dfa|frozen`, a small
relation-classification head, and the `--dfa-cut layer|step` axis
extended to the token-mixing. CPU-only.

**The question.** The program had one unresolved negative — the
transformer LM — and it CONFOUNDED two things: every LM run was DFA at
a ~50k-vocab output, while F13/F18 established that DFA degrades as the
OUTPUT DIMENSION grows. Attention, or the vocab? Nobody had separated
them. This lane keeps the attention and shrinks the head to 16 classes.

### The result: a POSITIVE, and the confound resolves in attention's favour

1500 steps, seeds 0/1/2, **each arm at ITS OWN best LR**:

| arm | lr | s0 | s1 | s2 | mean |
|---|---|---|---|---|---|
| chain (BP) | 0.003 | .983 | .990 | .981 | **.985** |
| **dfa (layer cut)** | 0.001 | .872 | .947 | .940 | **.920** |
| dfa (step cut) | 0.001 | .127 | .154 | .165 | .149 |
| frozen | 0.003 | .090 | .138 | .105 | .111 |

**Block-tap DFA reaches 93% of BP on a transformer, and beats the frozen
control by .81.** So attention is not the problem: the transformer-LM
negative was about the output dimension, and F22's route A is worth
pursuing with attention intact.

**The per-arm LR is not generosity, it is the whole result.** At BP's
own rate (0.01) the DFA arm reads **.064 — chance**. A single-LR matrix
would have published "attention is DFA-hostile", the exact opposite of
the truth, and it would have been quoted against the entire graph-LLM
route. DFA tolerating less LR than BP is toy#152's finding restated;
this is the lane where ignoring it would have cost the most.

### The mixing cut collapses — and that completes a four-lane pattern

`--dfa-cut step` (no gradient through the attention probabilities, with
Q/K getting their own random-feedback taps) scores .149 against the
layer cut's .920. Attention's PATTERN has to be learned through the true
gradient; random feedback at the block boundary is fine, random feedback
*into the mixing* is not.

Read across the lanes that share this axis:

| architecture | what the mixing does | per-step / mixing cut |
|---|---|---|
| gated LSTM (#157) | carries state through a forget gate | **HOLDS** (1.000) |
| selective scan (#155) | input-dependent selection | COLLAPSES (.250 = chance) |
| graph transformer (#160) | learned attention retrieval | COLLAPSES (.149) |

The pattern: where the architecture's own mechanism must be LEARNED
through the mixing — selection, attention — cutting the gradient there
kills it. Where the mixing is a gated carry that per-step random
feedback can still shape, it survives. "Can you delete the gradient
through the mixing?" has an architecture-dependent answer, and this is
now measured on three of them.

### The task took FOUR builds, and the frozen control caught every bad one

A structural attention mask aggregates each node's neighbourhood for
free, so a frozen random transformer is already a neighbourhood
averager — this lane's control is load-bearing in a way the others' are
not. Measured, in order:

1. **verbatim key match** → frozen **1.000**. A random projection
   approximately preserves inner products (Johnson-Lindenstrauss), so
   random Q/K already scored the match above the distractors: retrieval
   needed no learning at all.
2. **permuted key, pair-level split** → frozen **.957**. The head
   memorised each entity's fingerprint; because every entity appeared in
   training, held-out PAIRS leaked.
3. **permuted key, entity-level split** → BP itself stuck at **.246**.
   Memorising 36 training entities was cheaper than learning the rule,
   so nothing forced the rule.
4. **permuted key, content redrawn every step** → works. An entity index
   means nothing across steps, so the only learnable thing is the
   retrieval itself.

Every one of those was caught by asking "can the control lose?" rather
than by reading the code. `--task local` (type readable from the node's
own features) is the degenerate control and doubles as the DFA sanity
check: DFA solves it, which is what makes the step-cut collapse a
finding rather than a wiring bug.

### Construction note: the error lives on PAIRS, the taps on NODES

The CE error is [R, P] over pairs while every tap is [d_model, N] over
nodes. The error is routed onto nodes through a constant incidence
matrix (1 where a node is an endpoint of that pair), which is toy#153's
structure-aware route in its simplest form. Consequence, recorded
because it is honest rather than incidental: ATTRIBUTE nodes are
endpoints of no pair and receive zero direct error. Weights are shared
across nodes so they still learn from the entity columns; spreading the
node error along the adjacency is the obvious next axis if it matters.

### For Tao (F22)

```
toy train gtx --steps 1500 --seed 0 --lr 0.003 --policy chain,chain
toy train gtx --steps 1500 --seed 0 --lr 0.001 --policy dfa,dfa                  # layer cut
toy train gtx --steps 1500 --seed 0 --lr 0.001 --policy dfa,dfa --dfa-cut step
toy train gtx --steps 1500 --seed 0 --lr 0.003 --policy frozen,frozen
#  ... and all three seeds
```

**State the LR with the arm — the arms do not share one here**, and a
matrix run at BP's rate reports the opposite conclusion. `--lm-init` is
NOT built: with the content-resampled task, from-scratch BP reaches .985
and the frozen control sits at .111, so the acceptance bar is already
stateable without pretrained init. GLM's from-scratch-craters result is
about real ConceptNet with a T5-sized model; it does not bind here, and
the lane says so with a measurement rather than assuming either way.

## toy#162 — the FAIR BPTT control: BP's fragility is BPTT's, not ours

toy#157 measured BPTT as the fragile arm and Tao began recording that as
a *stabiliser* mechanism for DFA. The lane's own write-up flagged the
hole: **toy had no gradient clipping**, which is the standard fix for
exactly that failure mode (Pascanu et al. 2013), so part of the
fragility might have been the harness. This closes it.

`--clip-grad NORM` (env `LSTM_CLIP_GRAD`) implements GLOBAL-norm
clipping — one norm over all stepped parameters, applied in-graph
BEFORE the AdamW moments. **Off by default and byte-null when absent**,
so every cell the lane already published still means what it said.

### It does not rescue BP

Cells SOLVED (val ≥ .9) out of the 12 (lr × seed) at 4000 steps:

| arm | unclipped | clip 1.0 | clip 0.1 |
|---|---|---|---|
| chain (BP) | 7/12 | **8/12** | **7/12** |
| dfa (step cut) | 12/12 | **12/12** | — |

Clipping moves individual cells but not the fragility. What it actually
does is **re-roll which cells land in the good basin** — at clip 0.1
seed 0 solves at every rate while seed 2 mostly fails, the exact inverse
of the unclipped pattern:

| cell | unclipped | clip 1.0 | clip 0.1 |
|---|---|---|---|
| lr .01, seed 0 | .250 | .250 | **1.000** |
| lr .03, seed 1 | .250 | **.992** | **1.000** |
| lr .005, seed 2 | .996 | 1.000 | **.266** |

That non-monotonicity is itself the diagnosis. If this were plain
exploding gradients, a tighter clip would help monotonically; instead
the arm is BIMODAL — it either catches the cue or it does not — and
every hyperparameter, clipping included, just re-rolls the dice. **The
per-step DFA cut is the only arm invariant to all of them.**

So toy#157's stability finding stands with its fair control applied, and
F21 can record the stabiliser mechanism. The honest phrasing is not
"BPTT explodes and clipping is missing" but "on this task BPTT's success
is basin luck that no standard stabiliser removes, while cutting the
time axis removes the basin problem entirely".

### Scope

`lstm` only. The ssm lane's BP arm already reaches 1.000, so its
comparison was never at risk; the CLI rejects `--clip-grad` everywhere
else rather than offering a knob whose effect is unmeasured there.
Implemented from existing ops (`sum`/`sqrt`/`div`/`scale_bias`/`mul`
with a `[1,1]` broadcast), so there is no new shim function and no
CUDA/Metal mirror obligation.

**Why it could not be done on the learning rate.** Under Adam, clipping
and LR scaling are not the same operation: clipping changes what `m` and
`v` accumulate, an LR scale does not. A "clip" implemented on the hp
vector would have been a different experiment wearing this one's name.

## toy#161 (F12) — SHIPPED: DFA CAN retrofit a frozen BP-pretrained backbone

`toy train gtx --retrofit`. Phase 1 BP-pretrains the toy#160 backbone on
the 16-class relation task; phase 2 **freezes and detaches** it, adds a
pair-site adapter stack plus a fresh 4-class head, and trains only the
added capacity under `--adapter-policy chain|dfa|frozen`. Both phases
run in ONE process, so every arm's backbone is bit-identical by
construction rather than by a checkpoint round-trip that would have to
be trusted.

### The result: a POSITIVE, and the strongest one in the program

1500 + 1500 steps, seeds 0/1/2, each arm at its own LR:

| arm | lr | s0 | s1 | s2 | mean |
|---|---|---|---|---|---|
| bp-adapt (chain) | 0.003 | .990 | .987 | .979 | .985 |
| **dfa-adapt** | 0.001 | .989 | .991 | .980 | **.987** |
| frozen-adapt | 0.003 | .313 | .314 | .295 | .307 |

**DFA-adapt matches BP-adapt — a hair above it — and beats the frozen
control by .68.** This is the "DFA-LoRA" thesis: in the regime the
output-dim law says DFA works (F22/toy#160), DFA can cheaply adapt a
frozen pretrained backbone.

And unlike every other lane, **DFA here is insensitive to the learning
rate**: .985 / .987 / .985 across 0.0003 → 0.003, a 10x span. From
scratch it needed ~3x less than BP; adapting a frozen backbone through a
small adapter is an EASY regime for it.

### The task had to be many-to-one, and that is provable

The obvious retrofit shift — "new relation types" — does not work, and
the reason is worth stating because it will recur. The pretrain label
`ty_a*TY + ty_b` is a BIJECTION from pairs to classes, and so is any
relabeling of it. A retrained LINEAR head absorbs any bijection: set
`u_c[a] = 1{a = a*}` and `v_c[b] = 1{b = b*}` and the right class scores
2 while every other scores ≤ 1. So a frozen backbone plus a fresh head
solves it, the frozen control cannot lose, and the bar is vacuous.

The **modular sum** `(ty_a + ty_b) mod TY` is many-to-one and cannot be
absorbed: for TY=2 the four required inequalities sum to a contradiction,
and the argument runs for any TY. It needs an INTERACTION between the two
endpoint representations — which exists only where the pair is formed.
**That is why the adapters sit at the pair site and not inside the
backbone stack**, and it is a constraint on F12's design, not a
convenience: node-level adapters could not express the retrofit task at
all. Measured precondition: frozen-adapt .307 against chance .25.

### The cost claim: it is the FREEZE that buys it, not the credit rule

The ticket asks for the backbone-backward bytes avoided by dfa-adapt.
Measured honestly, that framing needs correcting:

- With `--freeze-backbone` the backbone output is **detached**, so
  neither arm backpropagates through it. `bb_grad_bytes_avoided` is
  therefore identical for chain and dfa (275 456 bytes at the lane
  defaults). DFA's own additional saving is the backward through the
  adapter stack only (32 768 bytes).
- The figure is **analytic and has to be**: the realized-graph counters
  exclude the backward extension, so a measured count cannot see an
  absent backward — freezing even reads *+1 node*, the detach itself.
  That is the mirror of tao#21: a measured graph cannot show a streaming
  win, and it cannot show a missing backward either.

So the cost win of a DFA retrofit at this site comes from freezing, and
DFA's structural advantage would only appear with capacity placed BELOW
the frozen stack's top. Worth knowing before F12 sells the cost half.

## toy#164 — the gtx checkpoint surface

`--ckpt-every K` writes the BACKBONE (the `@gx_backbone_count` span:
everything through the pretrain head) as a `toy-gtx/v1` GGUF into the
run dir; `--load-ckpt PATH` validates its shape metadata against the live
instrument, overwrites every backbone tensor by name, and **skips the
pretrain phase**. One pretrain, many retrofit arms — the workflow F12
asked for, and a large saving on sweeps that were re-deriving the same
backbone per arm.

Three deliberate constraints:

- **Only the backbone travels.** The retrofit capacity is never written.
  An adapter arriving with a "pretrained backbone" would start a retrofit
  somewhere other than the pretrained function, which is the one property
  the comparison rests on.
- **`--load-ckpt` requires `--retrofit`.** A loaded backbone on this lane
  exists to be retrofitted; a second meaning would make the flag
  ambiguous.
- **A shape mismatch is refused, naming both sides.** A backbone loaded
  under a different width or task shape produces confident garbage, and
  nothing downstream would look wrong.

The gate asserts the round trip is **bit-exact** — `bb_sig` compared as a
STRING across the writing and loading runs, not to a tolerance, because a
round trip that is merely close is one that lost bits.

**The one-process form remains how the BAR is measured.** Bit-identical
by construction beats a round trip that itself has to be gated; toy#164
removes the cost of re-pretraining for sweeps that do not need that
guarantee, and changes nothing about toy#161's result.

## Landmines that apply

- New runner = own compilation unit (landmine #16).
- Persistent inputs ALLOC before `finalize_weights` (toy#133) — a
  compute-context input allocated later reads zeros in SILENCE.
- Any `extend_backward_graph` wiring comes AFTER `tnn_build_backward`
  (toy#150) — earlier is silently discarded.
- New gate leg: capture `n0 = failures.length` at the leg start; never
  summarise with `failures.empty?` (d878143).
- Sweeps that shell out in zsh: never `env $var` (see
  [[zsh-env-var-wordsplit]]) — it silently drops every setting and both
  arms run the default.
