# DFA-arch program (toy#152–#158) — resolved decisions + build order

Tao filed six new architecture lanes plus F15. This records what is
SETTLED (tao#18, tao#19) so each lane can be built without re-litigating
it, and flags the one thing I think is still ambiguous.

Build order: **#152 → #158 → #154 → #153 / #155 / #156 / #157**.

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
