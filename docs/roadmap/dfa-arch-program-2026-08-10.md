# DFA-arch program (toy#152–#158) — resolved decisions + build order

Tao filed six new architecture lanes plus F15. This records what is
SETTLED (tao#18, tao#19) so each lane can be built without re-litigating
it, and flags the one thing I think is still ambiguous.

Build order: **#152 → #158 → #154 → #153 / #155 / #156 / #157**.
Status: **#152 shipped** (see "toy#152 — SHIPPED" below); #158 next.

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

Two results, both first-of-their-kind for this program:

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
