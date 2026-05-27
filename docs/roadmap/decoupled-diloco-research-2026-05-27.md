# Decoupled DiLoCo — research notes for the "laptop + remote GPU" use case

**Date:** 2026-05-27. **Status:** thinking-out-loud doc; not a commitment.
**Context:** companion to `backends-and-scale-2026-05-27.md`. That doc
covers single-host DP / multi-host DP at a high level; this one drills
into a specific algorithm family (DiLoCo, Decoupled DiLoCo) and the
specific dream of "start training on a MacBook, have a CUDA box join in
mid-run."

## TL;DR

- Paper is real: **"Decoupled DiLoCo for Resilient Distributed
  Pre-training"**, Douillard et al., DeepMind, arxiv 2604.21428
  (April 2026). Resilience-focused follow-up to vanilla DiLoCo (2023)
  and Streaming DiLoCo. Headline claim: "strictly zero global downtime"
  under massive simulated failures, with competitive quality.
- **Structurally** it fits the "join mid-training" idea: explicit
  learner-recovery, asynchronous attachment, K=1 quorum (one learner
  alone can drive progress). It **does not** validate the extreme
  asymmetry the user is imagining (laptop ↔ cloud-GPU, ~50× compute
  gap, Tailscale 100ms RTT, household-grade reliability). Closer-fit
  published variants for that story: **OpenDiLoCo** (Prime Intellect
  2024) and **Async-Local-SGD / Asynchronous DiLoCo** (Liu 2024,
  Pluralis 2025).
- For toy, smallest credible primitive: **synchronous vanilla DiLoCo
  over a 2-process socket transport**. Each replica trains its own
  full model on its own shard; every H steps they exchange pseudo-
  gradients; outer Nesterov-SGD step applied identically on both.
  Gated by the existing batching prereq, not by DiLoCo itself.
- Realistic effort (assuming batching is shipped): ~2 weeks for sync
  DiLoCo, ~4-6 weeks for an async / join-capable variant. Gating
  dependency is **not** the transport (sockets are plenty) and **not**
  the outer optimizer (~30 LOC). It's **dtype + numerical consistency**
  across heterogeneous hosts and the **membership protocol**.
- Honest failure modes: bf16-on-CPU vs f32-on-CUDA divergence in the
  outer step (we've been burned by exactly this — see
  `project_cpu_cuda_lora_train_divergence_2026_05_21`); straggler
  asymmetry making the laptop a near-no-op contributor; outer-step
  desync when the two sides disagree on the current step; Tailscale
  RTT amplifying any synchronous barrier.

## 1. What is Decoupled DiLoCo?

**Vanilla DiLoCo** (Douillard et al. 2023, arxiv 2311.08105) is a two-
level optimizer. Each of M replicas holds the **full model**, trains
independently on its own data shard for **H inner steps** with AdamW
(typically H=500), then all replicas synchronize by computing
**pseudo-gradients** (Δ = θ_replica − θ_global at the round start) and
applying an **outer step** of Nesterov-SGD on the averaged
pseudo-gradient. Communication is the all-reduce of one model copy
every H steps — orders of magnitude less than DDP.

**Decoupled DiLoCo** (arxiv 2604.21428) keeps that shape but breaks
the remaining synchronous barrier. Mechanisms reported in the paper:

- **Central synchronizer** (CPU-based) holds authoritative state;
  learners push parameter fragments asynchronously rather than
  peer-all-reducing.
- **Minimum quorum K** — synchronizer aggregates as soon as K of M
  learners reported. Paper uses **K=1** in most experiments: one
  learner can drive progress alone.
- **Adaptive grace window** — bounded extra wait within available
  bandwidth headroom to gather more contributors without stalling.
- **Token-weighted merging** — each learner reports `(steps, tokens)`;
  merge weight ≈ `tokens × tokens/steps`. Optional **Radial-Directional
  Averaging** separates outer-gradient norms and directions.
- **Fragmented sync** (P=24, τ=2 overlap) — Streaming-DiLoCo inheritance.
- **Learner recovery** (Appendix E.3) — a learner that drops off can
  rejoin by pulling current state; its synchronizer shard is held at
  weight 0 until return.

Communication-vs-compute: the press summary claims 0.84 Gbps across 8
datacenters vs ~200 Gbps for traditional DDP at comparable quality on
12B training, and >20× wall-clock speedup vs sync baselines in the
geo-distributed setting [deepmind.google/blog/decoupled-diloco/]. I
have not independently verified the 20× figure against the paper table.

## 2. Does it match the "join mid-training" use case?

**Structurally yes, empirically not really.** The mechanisms are the
right shape: K=1 quorum lets the laptop train solo until the CUDA box
appears; learner-recovery handles lid-closes; token-weighted merging
gracefully down-weights a slow contributor; async push fits Tailscale.

But the paper's regime is **homogeneous datacenter pods on 2-5 Gbps
links**, not consumer laptop + cloud GPU. Not validated:

- **Extreme compute asymmetry** (50×). Paper reports "slowest learners
  trailed fastest by 18%" on mixed TPUv5e/v5p — that's the published
  heterogeneity envelope. M-series ↔ GB10 is 50-100×.
- **Adversarial reachability.** 100ms RTT is fine (DiLoCo pushes once
  per H steps), but Tailscale connections silently die / re-establish.
  The paper's failure model is "TPU rebooted," not "home internet
  dropped for 4 hours."
- **Asymmetric ownership.** Paper assumes one synchronizer + N
  homogeneous learners. The user's mental model is "laptop owns, CUDA
  box opportunistically joins" — laptop is *both* synchronizer and a
  learner. Plausible but unbenchmarked.

The closer-fit published work for this scenario is **Async-Local-SGD**
(Liu 2024, arxiv 2401.09135) and **Asynchronous DiLoCo** (Pluralis
2025) — both scale per-worker inner-steps by speed (fast worker
H=500, slow worker H=50 in the same wall-clock) with momentum-
correction for stale gradients. **OpenDiLoCo** (Prime Intellect 2024,
arxiv 2407.07852) is the production-grade open-source implementation
that actually trained models across continents on mixed hardware,
using Hivemind's DHT for membership.

For toy's target scenario, OpenDiLoCo is the better reference than
the new Decoupled paper: Prime Intellect solved "decentralized
membership + heterogeneous hardware"; DeepMind solved "datacenter
resilience." Related but distinct shapes.

## 3. What would toy need to ship?

Tier-list, smallest primitive first:

**Tier 0 — prereqs already in the roadmap.** Batching (B>1) and
mixed precision are in `backends-and-scale-2026-05-27.md`. A DiLoCo
replica stuck at batch=1 isn't a serious training contributor.

**Tier 1 — synchronous vanilla DiLoCo, 2 processes, same host.** Two
`06_train_from_scratch.rb` processes on gx10, different RNG seed over
the same corpus. Every H=500 steps both write weights to a file, a
coordinator script averages them via Nesterov outer step, both reload.
DiLoCo *without DiLoCo's transport* — validates the algorithm shape
against toy's checkpoint plumbing. ~1 week including correctness
comparison vs single-process training.

**Tier 2 — synchronous DiLoCo over sockets, two hosts.** Replace
file-swap with a small TCP coordinator. Exchange parameter deltas as
flat f32 (or bf16) buffers. Rendezvous = config file. No late-join.
A 100M-param model is 400MB f32, exchanged once per ~500 steps ≈ 10
minutes — trivially fine over Tailscale. ~1 week on top of Tier 1.

**Tier 3 — async / late-join DiLoCo.** Now you need: a **membership
protocol** (cheapest: laptop as designated synchronizer, learners
reconnect; heaviest: Hivemind DHT, overkill for 2 nodes); **per-learner
inner-step scaling** (H proportional to recent step-rate from
`events.jsonl`); **staleness handling** in the outer step (Async-Local-
SGD's momentum correction); **reconnect/resume** (CUDA box continues
solo when laptop drops, laptop re-pulls on return). ~3-4 weeks on
top of Tier 2.

**What toy already has that helps:** `events.jsonl` is the right
substrate for "which step is each replica at." The GGUF checkpoint
writer is the right primitive for "send my weights to the peer."
Gradient sentinels notice divergence early. Observability is a real
head start.

## 4. Realistic effort

Assuming Tier 0 (batching + mixed precision) is done — call that 2-3
weeks of existing roadmap independent of DiLoCo.

- Tier 1 (single-host two-process sync DiLoCo, file-swap): **~1 week.**
  ~50 LOC coordinator + ~30 LOC outer-step + a correctness smoke test.
- Tier 2 (two-host sync DiLoCo over sockets): **+~1 week.** Socket
  plumbing + rendezvous config; algorithm unchanged.
- Tier 3 (async + late-join): **+3-4 weeks.** Membership, staleness,
  reconnect — the real work.

**Gating dependencies, ranked:** (1) **Batching** — replicas must
saturate hardware. (2) **Dtype discipline** — outer step must produce
near-identical results on both backends, see the prior CPU/CUDA LoRA
divergence note for evidence this isn't yet true; must root-cause
before any DP scheme is trustworthy. (3) **Transport** — sockets are
fine; NCCL/GLOO overkill at DiLoCo's once-per-H-steps cadence. (4)
**Outer optimizer + membership** — conceptually ~100 LOC. (5)
**Async / staleness** — where research-flavor work lives.

## 5. What could go wrong

- **Numerical drift between backends.** Already known: see the
  CPU/CUDA LoRA divergence note. Two replicas starting from identical
  weights but computing on CPU-bf16 vs CUDA-f32 will produce diverging
  pseudo-gradients; outer step compounds it. Mitigation: f32 master
  weights everywhere; bit-identical outer step (sorted reduce, Kahan).
  Single most likely failure mode.
- **Effective single-worker training.** If laptop is 50× slower and
  token-weighted merging down-weights it accordingly, laptop is ~2% of
  the signal. CUDA box is essentially training alone with mild noise.
  Not a bug — that's the math — but it calls the "heterogeneous
  training" framing into question. Honest version: "laptop participates
  symbolically; CUDA does the work."
- **Tailscale flakiness.** 100ms steady-state RTT is fine. Connections
  that die and silently re-establish over 30s are not, if the protocol
  assumes a stable socket. Need explicit per-message timeout +
  reconnect + replay. Standard distributed-systems hygiene, easy to
  underestimate.
- **Outer-step desync on dropout.** Without one source-of-truth replica,
  the two ends can end up at different parameter values both labelled
  "step N." Decoupled DiLoCo solves this with a central synchronizer;
  any toy implementation must commit to one source-of-truth replica.
- **Control-plane masquerading as numerics.** Distributed training
  failures usually present as "loss went up" but are actually "one
  replica fell behind, stale gradient overrode good signal." Toy's
  `events.jsonl` and sentinels help — but only if extended to track
  **inter-replica state**, which today they don't.
- **Quality regression at small scale.** DiLoCo's empirical sweet spot
  is ≥400M params on long runs. Toy's typical training is small models
  on short runs — exactly where inner-loop drift between sync rounds
  is largest relative to total signal. Algorithm may simply not pay
  off at toy's characteristic scale. Worth an ablation before plumbing.
- **The 20× speedup is geo-distributed-vs-naive, not vs good DDP.** The
  Decoupled DiLoCo headline is for WAN-separated datacenters vs
  baselines that ran poorly on that topology. Within one colo,
  well-tuned DDP still wins. For two-host toy this is mostly
  irrelevant — no DDP baseline to lose to — but worth being honest
  that the framing is topology-conditional.

## References

- Douillard et al., **"Decoupled DiLoCo for Resilient Distributed
  Pre-training"**, arxiv 2604.21428 (April 2026) —
  https://arxiv.org/abs/2604.21428
- Douillard et al., **"DiLoCo: Distributed Low-Communication Training
  of Language Models"**, arxiv 2311.08105 (2023, updated 2024) —
  https://arxiv.org/abs/2311.08105
- DeepMind blog, **"Decoupled DiLoCo: Resilient, Distributed AI
  Training at Scale"** — https://deepmind.google/blog/decoupled-diloco/
- AIForSWEs, **"Decoupled DiLoCo: How Google Is Enabling Multi-Region
  Distributed LLM Pretraining"** —
  https://www.aiforswes.com/p/decoupled-diloco
- ArXivIQ summary —
  https://arxiviq.substack.com/p/decoupled-diloco-for-resilient-distributed
- Jose et al. (Prime Intellect), **"OpenDiLoCo"**, arxiv 2407.07852 —
  https://arxiv.org/abs/2407.07852 ; blog at
  https://www.primeintellect.ai/blog/opendiloco
- Liu et al., **"Asynchronous Local-SGD Training for Language Modeling"**,
  arxiv 2401.09135 — https://arxiv.org/abs/2401.09135
- Pluralis, **"Asynchronous DiLoCo: Efficient Training on Heterogenous
  GPUs"** — https://blog.pluralis.ai/p/efficient-asynchronous-low-bandwidth
- HALoS (related, hierarchical async local SGD), arxiv 2506.04531 —
  https://arxiv.org/abs/2506.04531
- Smoothing DiLoCo with Primal Averaging, arxiv 2512.17131 —
  https://arxiv.org/abs/2512.17131

**Internal cross-refs:**
- `docs/roadmap/backends-and-scale-2026-05-27.md` — batching / DP
  prereqs and the strategic "does toy own DP at all?" question.
- Memory note `project_cpu_cuda_lora_train_divergence_2026_05_21.md`
  — concrete prior evidence that toy's CPU/CUDA numerical agreement
  is not yet good enough to take a DiLoCo-style outer step seriously.
