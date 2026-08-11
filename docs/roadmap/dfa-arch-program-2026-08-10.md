# DFA-arch program (toy#152–#158) — resolved decisions + build order

Tao filed six new architecture lanes plus F15. This records what is
SETTLED (tao#18, tao#19) so each lane can be built without re-litigating
it, and flags the one thing I think is still ambiguous.

Build order: **#152 → #158 → #154 → #153 / #155 / #156 / #157**.
Status: **#152, #158, #154 and #153 shipped** (see their sections below);
**#155 / #156 / #157 next**.

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
