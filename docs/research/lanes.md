# Research fixtures — the task lanes

toy ships twelve **fixture** recipes: synthetic task lanes that exist so toy can
test its credit-assignment capabilities *independently of any one research
question*. They are supported, gated, and reproducible — they are simply not the
product path, and a newcomer never needs them.

| | |
|---|---|
| **Capability** — credit rules (`--policy chain\|dfa\|frozen`), feedback matrices (`DfaB`, Kolen–Pollack, nDFA, LDFA), optimizers, LR schedules, checkpointing, instruments | framework features, documented in [`cli.md`](../cli.md) |
| **Fixture** — the lanes below | this document |

The lanes are `mlp ctr gnn ssm lstm gtx diff difflm ae franken franken-moe truck`. Each is
its own compiled binary (`libexec/toy-train-<lane>`), driven by the CLI as a pure
ENV-marshalling shim, and pinned by its own gate.

> **Planned:** these move to `toy research train <lane>`, with `toy train <lane>`
> retained as a deprecating alias. See tao#25 for the migration and the
> byte-reproducibility guarantee.

The reference below is written in the register of the research programme that
produced it (toy#152–#173) — measurements, arms, and the reasoning behind each
knob. That is deliberate: its audience is someone running these lanes.

---

#### The DFA lanes in one paragraph

`--policy chain,dfa,…` sets credit assignment **per layer**:
`chain` = backprop, `dfa` = fixed random feedback, and on `mlp`, `ctr`
and `gnn` also `frozen` (the control arm — the layer stays at init while
the head still trains). `--policy-scope attn|ffn|all` selects which tensors a `:dfa`
layer policies (dense lanes; default `attn`).
`--dfa-granularity matmul|block` selects **micro** (per-weight) vs
**macro** (one tap per block output, full BP inside the block) DFA.
`--align-events` emits the `align` telemetry described in
[events.md](../events.md#align-credit-assignment-telemetry--the-dfa-lanes).
`--optimizer adamw|muon|radam` (`radam` is `franken`-only; `sgd` is
`franken-moe`-only).

`gnn` adds `--feedback-route direct|structure` (+ `--feedback-hops N`):
`structure` spreads the error along the graph before the random
projection, so an UNLABELLED node — which has exactly zero direct error
in this semi-supervised setting — receives a pseudo-error from its
labelled neighbourhood. This is deliberately **not** spelled
`--dfa-feedback`: `franken-moe`'s flag of that name selects how the
feedback matrix B is *updated* (`fixed|kolen-pollack`), a different axis.
A different meaning gets a different name (the tao#18 `--policy-scope`
discipline).

`ssm` and `lstm` add `--dfa-cut layer|step`; `--selection selective|lti`
is `ssm`-only. `layer` cuts only the layer boundary and injects the
random feedback once, at the readout step, with BPTT intact inside the
layer; `step` additionally detaches the state at every timestep (on
`lstm`, both `h_{t-1}` and `c_{t-1}`), so no gradient crosses time at
all. It is **not** spelled `--dfa-granularity`: that one
picks matmul-vs-block in *depth*, this picks layer-vs-step in *time*.
`--align-events` is REJECTED on `ssm` and `lstm` — their DFA update
arrives through autodiff from the surrogate roots, so it lands in the
same accumulator a BP run uses and a cosine against it would mean
nothing.

The two recurrent lanes share `--seq`, `--classes`, `--task cue|mean`,
`--cue-span` and `--noise` because `lstm` reuses `ssm`'s delayed-cue
generator **unchanged** — that is what makes the pair an architecture
comparison rather than two anecdotes. They do *not* share
`--d-inner`/`--conv-k`/`--selection`/`--dt-init`, which name
selective-scan parts an LSTM has no counterpart for; those are rejected
on `lstm` by the same matrix.

`lstm` also carries `--clip-grad NORM` (toy#162): global-norm gradient
clipping, applied in-graph before the AdamW moments, **off by default
and byte-null when absent**. It exists as the fair control for the
lane's stability claim rather than as a tuning knob — measured, it does
*not* make the BP arm reliable (7/12 solved cells unclipped, 8/12 at
norm 1.0, 7/12 at norm 0.1), it re-rolls which cells work. It is
rejected on every other recipe.

`lstm`'s BP arm is **bimodal** on this task — it solves it (~1.000) or
sits at chance (.250), and which learning rate works depends on the
*seed*. Its fair cell is therefore
`--lr 0.02 --warmup 200 --steps 4000`, the BP arm's own best over a
swept (lr x warmup x seed) grid, and that cell is always written out in
full. It is deliberately **not** folded into the defaults: toy#157
shipped the 200-step ramp as a default for exactly one commit, and a
consumer's `--lr 0.03 --steps 2000` silently inherited it and relabelled
a 3-seed matrix. Defaults on this lane change nothing (`--warmup 0`);
the cell is spelled out instead.

The `mlp`, `ctr` and `gnn` lanes print extra stdout lines after the
curve — `val: acc=… loss=… n=…`, `val: auc=… logloss=… n=… pos=…`, and
for `gnn` a `train:` line as well — because held-out accuracy / AUC, not
loss, is the metric their success criteria are stated in. `gnn` reports
both sides of the split because its arms separate as much by their
train/val GAP as by val accuracy. `ssm` also prints
`graph: nodes=…`, the realized graph size — how a sweep over `--seq`
reads the arms' scaling. Read it as node count, never as bytes: in a
graph autodiff every forward tensor is materialised whatever the credit
rule. For what a *streaming* implementation would hold instead, read the
`stream:` line beneath it (toy#159, below).

`lstm` prints the same line with the missing half — `graph: nodes=…
bytes=…`, where `bytes` is `ggml_nbytes` summed over every node of the
realized graph, i.e. the materialised activation footprint its ticket's
memory target is stated in. The caveat above still travels with it and
is the finding: cutting BPTT does **not** shrink the graph. Measured at
T=64, the per-step cut costs 17% MORE bytes than BPTT (15 578 112 vs
13 289 988), and the gap grows with `--seq`.

#### The `stream:` line — ANALYTIC, not measured (toy#159)

Both recurrent lanes print a second memory line, and the distinction
between the two is the whole point:

```
graph:  nodes=2116 bytes=15578112                       # MEASURED — what toy builds
stream: bptt=3868160 sqrt_t=984576 cut=78336 cut_vs_bptt=49.37x cut_vs_sqrt_t=12.56x replay=2x_fwd
```

`stream:` is computed from the cell's shapes, **not** by walking a
graph: it is what a streaming implementation would have to hold, which
is the quantity the F19/F21 success target is actually about and the one
no graph measurement here can exhibit.

- `bptt` — every step's activations live for the ordered backward: O(T).
- `cut` — the per-step cut, replaying the forward once the error is
  known and updating each step as it passes: **O(1) in T**. Identical
  across `--seq 16/64/128`, and that invariance is gated.
- `sqrt_t` — BPTT's own best counter-move, sqrt-T checkpointing at
  ~1.5x forward compute, quoted so the cut is not compared only against
  BP's worst case.
- `replay=2x_fwd` — the price of `cut`, stated inline. It is not free.

The win is real but it is not a property of the credit rule alone: the
surrogate's error is known only after the last-step readout, so an O(1)
implementation must replay the forward. **What cutting the time axis
buys is the LEGALITY of that replay** — steps become independent, so
they can be revisited in any order — which is precisely what BPTT
cannot do, and precisely why the *measured* graph comes out bigger
rather than smaller. Weights, optimizer moments and gradient buffers are
excluded from both sides: they are equal across arms and independent of
T, so they would only dilute the ratio.

`gtx` puts **attention** on the same `--dfa-cut layer|step` axis: `layer`
taps each block's output with BP intact inside the block, `step`
additionally detaches the attention **probabilities** so no gradient
crosses the token-mixing (Q and K then get their own random-feedback
taps, or the arm would just be "attention frozen"). Its `--task
relational|local` mirrors the other lanes' degenerate control: under
`local` a node's type is readable from its own features, so no retrieval
is needed at all.

**The `gtx` arms do not share a learning rate, and that is a result
rather than an inconvenience.** BP's cell is `--lr 0.003`; the DFA arms
want ~3x less, and at BP's rate DFA reads chance. A single-LR matrix on
this lane would report "attention is DFA-hostile" — the opposite of what
it measures. Always state the LR with the arm.

#### `gtx --retrofit` — adapting a frozen pretrained backbone (toy#161)

One process, two phases: `--pretrain-steps` of BP on the 16-class
relation task, then the backbone is **frozen and detached** and only
added capacity trains on a *different* task — a 4-class **modular sum**
over the same graph — under `--adapter-policy chain|dfa|frozen`.

```
toy train gtx --retrofit --pretrain-steps 1500 --steps 1500   --adapter-policy dfa --pretrain-lr 0.003 --lr 0.001
```

Three things about it are load-bearing:

- **The retrofit label is many-to-one on purpose.** Any *bijective*
  relabeling — including "new relation types" — is absorbed by a
  retrained linear head, which would leave the frozen control unable to
  lose. The modular sum provably needs an **interaction** between the
  two endpoint representations, which is why the adapters sit at the
  **pair site** rather than inside the backbone.
- **Both phases run in one process**, so every arm's backbone is
  bit-identical by construction. The runner prints `backbone: sig_pre=…
  sig_post=…`; under a freeze the two are identical, which is a
  measurement rather than a promise that nothing was stepped.
**The two-command workflow (toy#164).** `--ckpt-every K` writes the
backbone as a `toy-gtx/v1` GGUF into the run dir, and `--load-ckpt PATH`
loads it and **skips the pretrain phase**, so one pretrain can serve many
retrofit arms:

```
toy train gtx --steps 1500 --ckpt-every 1500 --out $BB
toy train gtx --retrofit --load-ckpt $BB/step_1500.gguf --adapter-policy dfa --lr 0.001
toy train gtx --retrofit --load-ckpt $BB/step_1500.gguf --adapter-policy chain --lr 0.003
```

Only the BACKBONE is written — never the adapters, which must start at
identity. A shape mismatch is refused naming both sides, and
`--load-ckpt` requires `--retrofit` on this lane. The gate asserts the
round trip is **bit-exact** (`bb_sig` string-equal across write and
load), because a round trip that is merely close is one that lost bits
and every arm sweeping off that checkpoint would inherit them.

For the *bar*, prefer the one-process form (no `--load-ckpt`): it makes
every arm's backbone bit-identical **by construction** rather than by a
round trip that itself has to be trusted.

- **The cost line is analytic, and it credits the freeze, not DFA.**
  `retrofit: … bb_grad_bytes_avoided=…` is what *freezing* saves, and it
  is the same for both credit rules at this adapter site; DFA's own
  saving is the backward through the adapter stack only. The realized
  graph counters cannot show this (they exclude the backward
  extension — freezing even reads +1 node, the detach), which is the
  mirror of tao#21's caveat.

#### `gtx --dfa-feedback-precond ndfa` — the nDFA error-side preconditioner (toy#172/E1)

`--task bytelm` only, and only on a policy that actually has a `dfa`
block. It left-multiplies the broadcast error by
`lambda (C_E + lambda I)^-1` — the inverse local-error second moment,
ridge-regularised — which is **folded into the feedback matrix `B`**
host-side rather than added to the graph:

```
toy train gtx --task bytelm --text data/ae_shak_a2504 --vocab 4096 \
  --policy dfa,dfa,dfa,dfa --dfa-cut layer --lr 0.00003 --steps 4000 \
  --dfa-feedback-precond ndfa --ndfa-lambda 0.0001 --ndfa-every 500 --ndfa-samples 256
```

| flag | meaning |
| --- | --- |
| `--dfa-feedback-precond none\|ndfa` | `none` (default) is byte-identical to the runner before the flag existed |
| `--ndfa-lambda R` | the ridge. **Required** with `ndfa`; there is no default |
| `--ndfa-every K` | steps between refreshes (default 500) |
| `--ndfa-samples M` | error vectors per refresh (default 256) |
| `--ndfa-gain preserve\|raw` | `preserve` (default) renormalises `B'` back to `‖B‖_F` |

Five things about it are load-bearing:

- **`lambda` is the experiment, not a tuning detail**, which is why it
  has no default and is refused if omitted. `P` is the **identity** as
  `lambda` grows and a **projector** as it shrinks; pick it against the
  `lambda_max` the Phase 1.1 instrument reports (`GTX_INSTRUMENT=1`),
  which was measured falling from 0.121 at rank 65 to 0.0079 at rank
  2504.
- **The normalisation is `lambda (C_E + lambda I)^-1`, not
  `(C_E + lambda I)^-1`.** The first tends to `I`, so large `lambda` is
  a *byte identity* with the unpreconditioned arm (gated); the second
  tends to `I/lambda` and would silently rescale every DFA update — an
  LR change wearing a conditioning costume.
- **Nothing `[V x V]` is ever formed.** Woodbury turns the inverse into
  an `[m x m]` Cholesky on the Gram `G = EᵀE`, so `B' = B - (B E)(lambda
  m I + G)^-1 Eᵀ`. No new graph node, no new readout path.
- **`gain preserve` is the default because a global gain on `B` is
  indistinguishable from an LR change on this lane.** `P`'s eigenvalues
  are `lambda/(lambda + s)` in `(0, 1]`, so nDFA can only shrink `B`;
  `preserve` gives the norm back and leaves only the *direction*
  reweighting under test. The pre-renorm ratio is printed as `b_shrink`
  either way.
- **The samples come from the training steps themselves.** Running extra
  forward passes at the refresh point would consume the corpus RNG and
  step the Adam moments (lr=0 does **not** freeze `m` and `v`), so the
  large-`lambda` arm would diverge from the unpreconditioned one for
  reasons having nothing to do with `B`.

The run prints `ndfa: on=1 lambda=… every=… m=… gain=… refreshes=…
err_pre=… err_post=… b_shrink=…`. `err_pre`/`err_post` are `‖B E‖_F` and
`‖B' E‖_F`, the broadcast error as the taps actually receive it; the
runner **aborts** if `err_post` is not finite, because an inf/nan `B'`
would otherwise just stop the loss being a number partway through a run.
`refresh_ms` reads `unmeasured-no-run-dir` without `--out`: the clock
this lane has returns 0.0 when no event file is open, and printing that
would be a metric silently reading zero.

#### `gtx --dfa-feedback-rank` — LDFA, adaptive low-rank feedback (toy#172/E2)

`--task bytelm` only, and only on a policy that actually has a `dfa`
block. It factorises the feedback matrix as `B_eff = Q[dout x r] . P[r x
V]` and folds the product into the **same uploaded `B` tensor** — no new
graph node, no new readout path, the same route nDFA took:

```
toy train gtx --task bytelm --text data/ae_shak_a2504 --vocab 4096 \
  --policy dfa,dfa,dfa,dfa --dfa-cut layer --lr 0.00003 --steps 4000 \
  --dfa-feedback-rank 64 --dfa-feedback-adapt oja --ldfa-eta 0.05 \
  --ldfa-every 500 --ldfa-samples 128
```

| flag | meaning |
| --- | --- |
| `--dfa-feedback-rank full\|R` | `full` (default) is byte-identical to the runner before the flag existed |
| `--dfa-feedback-adapt none\|oja` | whether `P` tracks the error's top-`r` subspace by Oja's rule |
| `--ldfa-eta E` | Oja step size (default 0.05). `0` is legal and relabels itself `adapt=oja-frozen` |
| `--ldfa-every K` | steps between refreshes (default 500) |
| `--ldfa-samples M` | error vectors per refresh (default 128) |

Four things about it are load-bearing:

- **The scale is matched, and that is what makes the arms comparable.**
  A rank-`r` `Q.P` has a completely different Frobenius norm from the
  full-width `B` it replaces, so without a rescale "low rank hurts"
  would be indistinguishable from "the updates got smaller" — a global
  gain on `B` is an LR change wearing a rank costume. `B_eff` is
  rescaled to the **realised** `‖B‖_F` of the full-width draw from the
  same seed, and **both norms are printed** (`b_eff_fro`, `b_full_fro`,
  `scale_ratio`) so a reader can check it rather than assume it.
- **`rank_eff` is not `r`.** `rank(Q.P) <= min(dout, r)` and `dout` is
  `d_model`, so on the P6 fixture (`d_model` 128) `r = 256` buys the
  *same matrix rank as full width* and only confines the row space. Both
  numbers are on the provenance line for exactly that reason.
- **`P` is orthonormalised at init in BOTH modes** (`p_ortho=init`).
  Oja's rule requires orthonormal rows, so an un-orthonormalised fixed
  arm would differ from the adaptive one in two ways at once. The only
  difference between `none` and `oja` is whether the Oja update is
  applied — both collect the same samples on the same steps.
- **`p_energy` is the convergence check.** It is the fraction of the
  error's energy `P` captures, against `p_energy_rand = r/V` for a
  random orthonormal `P`. An adaptation that ran but learned nothing
  sits at the random baseline, which is otherwise indistinguishable from
  one that worked.

The run prints `ldfa: rank=… rank_eff=… dout=… v=… adapt=… eta=…
every=… m=… refreshes=… b_eff_fro=… b_full_fro=… scale_ratio=…
p_row_min=… p_row_max=… p_offdiag_max=… p_energy=… p_energy_rand=…`. The
`p_row_*`/`p_offdiag_max` numbers are measured **before**
re-orthonormalisation — after MGS they would be 1 and 0 by construction,
and a diverging `eta` would go unnoticed behind a tautology. The runner
**aborts** if the basis goes non-finite rather than renormalising it,
which would look like a healthy adaptation.

LDFA and nDFA are refused together: they fold into the same `B` and are
opposite interventions (nDFA *whitens* the dominant error directions;
LDFA *compresses to* them), so a composed cell is neither arm.

`diff` prints `gen: energy=…` — the ENERGY DISTANCE between generated
and held-out real samples. It is the lane's headline metric and **lower
is better**, the opposite direction from every accuracy-scored lane
here; a slightly negative value means the two sets are indistinguishable
at that sample size, not that something broke.


---

### `ae` — the per-token latent autoencoder (toy#165, capstone P1a)

The diffusion-text-LM capstone needs a per-token continuous latent of 4–8
dims (F20/toy#156's DFA-favourable window). Whether **text** survives such
a latent is unrun in the literature, and P1a is the cheapest decisive
test. It is **all BP** — no `--policy`, no DFA, no diffusion; those are
P1c and P1b.

```
ruby prep/fetch_text.rb --all          # once: three pinned byte corpora
toy train ae --text data/ae_names --latent 8 --context 256 --steps 4000 \
             --noise-eval 0,0.25,0.5,1,2 --seed 0
```

**Clean reconstruction is vacuous and is not the headline.** Packing a few
dozen codepoints into 4 continuous dims is ample analog capacity, so clean
accuracy sits at ~1.0 at *every* latent and cannot tell 4 from 32. The
read is the **noise margin**: reconstruction accuracy as the latent is
perturbed by Gaussian noise scaled to the latent's own per-dim std (so it
is comparable across `d`), summarised by the **half-accuracy SNR** — the
sigma at which accuracy crosses half of that cell's clean value. When the
curve never crosses inside the grid it is reported as `>=<max sigma>`,
never clamped.

**The alphabet is the second axis** (tao#22). The margin is packing-limited,
so a curve is unscoped without the number of symbols the head actually had
to separate: at N=27 the `d=4` problem is about as hard as N=256 at `d=8`,
and the alphabet alone can manufacture a `go`. Three pinned corpora ship:

| pack | source | bytes | distinct |
|---|---|---|---|
| `data/ae_names` | makemore names list | 228 K | 27 |
| `data/ae_shakespeare` | tinyshakespeare | 1.1 M | 65 |
| `data/ae_udhr` | UDHR, 388 languages, UTF-8 | 5.7 M | 201 |

Every run reports both the pack alphabet and the `val_alphabet` actually
observed in the scored windows. **Preliminary** surface (seed 0, 800 steps,
context 128, `d_model` 128 — Tao's cells are 4000 steps, so read these as
shape, not as the result):

| corpus (val alphabet) | d=4 | d=8 | d=32 |
|---|---|---|---|
| names (27) | 0.82 | 1.25 | ≥2.0 |
| shakespeare (53) | 0.66 | 0.98 | 1.93 |
| udhr (104) | 0.70 | 0.93 | 1.62 |

There is **no `--vocab`** on this lane. `--vocab` already means an integer
pack width on `franken`/`franken-moe`, and the `ae` head is byte-wide (256)
on every corpus by construction — sizing it to the observed alphabet would
confound the alphabet axis with head capacity. `--latent` is deliberately
the **same flag** the `diff` lane uses: it is the same quantity, and the
capstone compares those two numbers directly.

`--context` is also the **batch** — the encoder attends within a window, so
its T positions are the T reconstruction targets and there is no separate
`--batch`.

The controls: a **zeroed** latent leaves the head with only its bias, so it
lands at the unigram floor by construction — reported, but not gated, since
it is an identity. The **shuffled** latent (permuted across positions) is
the one with teeth and the one `prep/ae_gate.rb` gates: each position
decodes a real latent from the same distribution, just the wrong one, so it
*can* score above the floor if the head learned a prior.

#### `--target-ce` — comparing cells at matched FIDELITY, not matched steps

A noise margin is **not invariant to convergence**, and a sweep run at a
fixed step budget measures both. Measured on this lane: with clean accuracy
pinned at 1.000 throughout, names d=32 goes half-SNR **1.92 → 2.44** as CE
falls 1.9e-3 → 1.5e-9 (cross-entropy keeps rewarding wider logit separations
long after accuracy stops moving); and udhr d=4 goes **0.599 → 0.479** while
accuracy *rises* 0.951 → 0.993 (newly-learned rare bytes have to be packed
into the same `d` dims). Two opposite-signed biases, so the confound distorts
the shape of a surface, not just its scale.

The first 36-cell P1a surface was run at matched steps and its clean CE
spanned **seven orders of magnitude**. Re-run at matched CE it spans a factor
of 5.4, and the fitted law changes qualitatively.

```
toy train ae --text data/ae_udhr --latent 8 --context 256 \
             --steps 12000 --target-ce 0.05 --eval-every 10
```

Each cell trains until held-out CE crosses the target, then reports
`converged: target_ce=… achieved_ce=… steps=… matched=1`. A cell that never
reaches it reports `matched=0 … NOT REACHED` — an unmatched cell must not
join a matched surface silently.

**The probe must not perturb training, and that is not free.** ggml's AdamW
kernel is `m = m*b1 + g*(1-b1); v = v*b2 + g²*(1-b2); w = w*(1-lr*wd) - lr*mh/vh`,
so the `lr=0` eval hp every lane uses freezes the **weights** but still lets
the moments absorb the probe's gradients. Harmless at end-of-run; corrupting
for a periodic probe. The probe hp sets `b1 = b2 = 1.0`, making both moment
updates the identity. `prep/ae_gate.rb` asserts the consequence: a probed
run's training curve is byte-identical to an unprobed one.

`--probe-batches K` (default 4) runs the stopping check on a subset so the
interval can be fine. A coarse interval lets fast cells overshoot the target
by an order of magnitude — the exact mismatch the flag exists to remove. The
full held-out CE on the `val:` line stays the authoritative matched quantity;
both are printed.

Off by default and byte-null when off.

---

### `truck` — the truck backer-upper, the first CLOSED-LOOP lane (toy#189)

Every lane above is a **static dataset**. This one is not: the policy determines
the states it subsequently sees, which is the whole reason the
`dfa-for-dynamic-control` arc exists. Nothing measured on a static fixture says
whether DFA's noisier updates drift and compound in a loop or act as exploration
that makes a policy robust to its own mistakes.

The plant is Nguyen & Widrow's truck-and-trailer on Schoenauer & Ronald's arcsin
reformulation (`lib/toy/io/toy_truck_task.rb`, toy#188), and the target is to
**reproduce S&R's ICEC'94 result with a DFA-trained net in place of their GA**.

**Why it is an experiment and not a re-run.** The GA has *no error signal* — one
number per episode. So "DFA instead of GA" cannot mean BPTT-through-the-plant
with `W2ᵀ` swapped for a random matrix; that keeps the entire gradient path the
GA never had. It means an episodic **state error, randomly projected**, with no
plant Jacobian anywhere (`dfa_tb`). Textbook DFA still hands the readout its true
`dL/dy`; here `dL/du_t` does not exist without the plant.

| arm | hidden `W1` | readout `W2` | role |
|---|---|---|---|
| `ga` | evolution | evolution | the paper's pole, reproduced |
| `bptt` | exact plant Jacobian | exact | the ceiling |
| `frozen` | fixed at init | exact | **the fixture gate** |
| `dfa_tb` | `B1·e` | `B2·e` | the headline arm |
| `dfa_rx` | `B1·e` | exact | disambiguation |

`frozen` is not a formality: four inputs into nine random sigmoids is a good
random-feature basis. If `bptt` does not beat it with margin at matched seeds the
fixture cannot discriminate and every DFA row on it is void — read that row first.

| flag | |
|---|---|
| `--arm ga\|bptt\|frozen\|dfa_tb\|dfa_rx` | which credit-rule pair (default `bptt`) |
| `--start-scheme ensemble\|point\|yard\|lesson\|half_yard\|half_yard_neg` | `ensemble` = the paper's 15 fixed starts (default). The two half-yard schemes are C2b's sign test (toy#192) |
| `--obs 4\|3\|8` | the paper's three input sets; `8` is its rescaled-duplicate trick |
| `--hidden N` | hidden width (default 9 — with `--obs 4` that is the paper's 55 weights) |
| `--act sigmoid\|tanh` | the paper's logistic (default) or the frontend's tanh. Also selects the output→steering map: `2σ−1` vs identity |
| `--plant-r R` / `--step-cap N` | the paper's `r = 3` and 300-step episodes (defaults) |
| `--lr R` / `--clip-grad R` | **per arm.** See the note below |
| `--budget N` | the **matched-work** budget, in plant steps — the only comparable currency between a GA and a gradient arm |
| `--loss best\|terminal` | which step the arms **descend**, as distinct from the nearest approach they are **scored** at. `best` is the default and descends exactly what it scores |
| `--load PATH` / `--trace PATH` | roll a trained controller out headless to a `tbu-traces/1` bundle (toy#190) |
| `--trace-scheme` / `--trace-n` / `--trace-seed` / `--stride` | which starts to roll, how many, from which seed, and how much of each trace to keep |
| `--ga-pop/--ga-gens/--ga-agg` | the paper's pole. `--ga-agg mean\|min`, both of which it reports as acceptable |
| `--export PATH` | the controller in the frontend's format (`[layer][unit][w…, bias]`, bias **last**) plus a provenance sidecar |

Each `eval:` line also carries `mean_abs_signal`, `frac_sat` (|signal| > 0.99),
`frac_clamped_runs` and `median_path_len` (toy#192). They exist because
`mean_d2`/`dock5` cannot separate **two different routes into the same absorbing
state**: bang-bang steering with the wrong sign, and an under-actuated passive
jack-knife, both end a few metres from where they started with the clamp touched
in nearly every rollout. Measured on the far set, `dfa_tb` runs at |signal| ~1.0
with most steps saturated while `dfa_rx` sits at ~0.04 and never saturates — the
same `mean_d2`, opposite failures.

**The score is the NEAREST APPROACH over the episode**, `d² = x² + y² +
min(θs², (θs−2π)², (θs+2π)²)` — the GA scored the best point on the trajectory,
not where it stopped. `far` and `near` start sets are reported **separately on
every run**, never pooled: the paper's own split is that the gradient-free method
owned the far field and lost the close-up field, and which side a randomly
projected error lands on is a pre-registered question.

#### Three things measured here that will cost you time otherwise

- **`--lr` is per arm, over four orders of magnitude.** Measured at 5000 updates
  × 3 seeds: `bptt` peaks at `6.0` (interior — `12.0` collapses), `frozen` at
  `0.1`, `dfa_rx` at `0.2`. Reading `bptt` and `frozen` at one shared `lr=1.0`
  reports a 7% margin for a pair whose real separation is three orders of
  magnitude.
- **`--clip-grad` is effectively required** (default `1.0`). The objective is a
  *physical* `d²` of order 10⁴ chained through ~30 plant steps, so one unclipped
  update saturates the steering and every later episode ends at step 0 — a flat
  loss curve, not an error.
- **Neither loss recovers the paper's single-point start; `best` cannot even
  train on it.** From `(20, 10, −2)` backing up *increases* x, so an untrained
  policy never gets closer than its start. Under `best` the nearest approach IS
  the start, `d²` there is a **constant** no weight can move, the gradient is
  identically zero, and every arm sits on a plateau at exactly `504.0` (0/10
  seeds, all four arms). `terminal` removes the plateau — the loss moves — and
  the evaluated result is **still** exactly `504.0`, 0/10 seeds, all four arms,
  at every LR tried. Only `ga` reaches `d² < 5` there (5/5 seeds), which is the
  paper's own claim about why gradient methods are a poor fit on this plant.
- **`terminal` costs the ensemble and buys nothing measurable.** `bptt` under
  `best`: mean `d²` 2.5, docking 0.911. Under `terminal`: 2798.7 and 0.311,
  optimum interior at `lr 6.0`. It was briefly the default on toy#189 — on the
  reasoning that `best` cannot train the single point at all — and the
  measurement above withdrew that: it removes the plateau without recovering the
  single point. `best` is the default again, and it is also the one that keeps
  the descended and reported objectives identical.

The GA's fitness carries `fitness=reconstructed` in its provenance: the paper's
*amended* fitness is a bitmap figure the scan did not preserve, and the form used
(`1/(ε+d²) · 1/(1+γl)`) satisfies everything its text states but is not a
citation. `d²` itself is cited and unaffected.

Gates: `gate-truck-plant` (the plant's parity against a golden trajectory from
the frontend, and its analytic Jacobian against finite differences) and
`gate-truck-lane` (the lane's own BPTT gradient, self-checked the same way).

#### `--trace` — the `tbu-traces/1` bundle (toy#190)

`--load` a controller (the `--export` format plus its sidecar) and roll it out
headless over the single point, the 15-start ensemble, or `N` seeded yard draws;
`--trace` writes one bundle the frontend overlays. The frontend never runs a
forward pass, so there is no second controller implementation to keep in
parity — **and the rollout shares `tk_rollout` with the training loop for the
same reason**, rather than re-deriving a trajectory.

Per-run summaries are carried **and recomputable from the trace**, which
`gate-truck-lane` asserts by recomputing `best_d2`, `terminal_d2` and every `u`
from the rows. Two consequences worth knowing:

- `--stride k` keeps every k-th row **plus the last row and the best-approach
  row, always**. Those are the two rows every summary is computed from; a stride
  that dropped them would leave a bundle whose own numbers cannot be recovered
  from its own trace. At `k=10` a 300-step run keeps 32 rows, ~9x smaller.
- Yard starts are reproducible **from the seed alone** (the plant's own
  `sample_yard!`), so a bundle from this engine and one from the frontend's CLI
  overlay on identical states. `start.seed` and `start.idx` are recorded per run.

`--load` is checked against the net shape: a 4-9-1 controller loaded into a
3-input net is refused naming both counts, because a silently half-applied load
would leave the tail of the net at init and read as a slightly worse arm.

There is no gzip step in the runner — it has no zlib. Pipe it:
`gzip -9 bundle.json`.

#### `half_yard` — C2b's sign test (toy#192)

The candidate mechanism for `dfa_tb` failing while `dfa_rx` lands exactly on
`frozen`: a **fixed** `B` cannot carry a **state-dependent sign**. From `y > 0`
the correcting steering has one sign and from `y < 0` the other, the error's `dy`
component flips with it, so a fixed random map from error to weight direction is
right on one half of the yard and wrong on the other — and over a symmetric
start distribution the two halves cancel into a constant push.

`--start-scheme half_yard` draws the usual yard restricted to `y ≥ 0`;
`half_yard_neg` is its mirror. Same seeded LCG and the same x/angle bounds, so a
half-yard bundle reproduces from its seed exactly as the full yard does.

**Measured, and it is not the mechanism.** By the test's own criterion — dock
something on a half and nothing on the full yard, or it isn't the sign problem —
`dfa_tb` docks nothing on either half at any LR in {0.003 … 2.0}, 3 seeds, 4000
updates. What it does instead is collapse to a **constant, fully saturated,
single-sign policy**: steering `< 0` on 100% of 18,000 traced steps, whichever
half it trained on and whichever half the run starts in. That is stronger than
the cancellation story and not the same claim — cancellation would have been
cured by training on one half.

At `lr = 0.003` it does *not* saturate (`frac_sat` 0.000) and still docks
nothing, so saturation is a symptom rather than the cause.
