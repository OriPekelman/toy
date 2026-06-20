# Scaling story — one experiment, a spectrum of couplings

**Date:** 2026-06-18. **Status:** design skeleton. One decision locked
(own-loose / lean-tight, below); the rest is a proposed shape, gated.
**Context:** the API-first consolidation of the three 2026-05-27
future-directions docs —
[`backends-and-scale`](backends-and-scale-2026-05-27.md) (prerequisites,
multi-GPU modes, hybrid offload),
[`training-backends`](training-backends-2026-05-27.md) (the "add a backend
without rewriting the graph" question), and
[`decoupled-diloco-research`](decoupled-diloco-research-2026-05-27.md) (the
low-communication multi-machine path). Those captured the *mechanisms*;
this captures the *API and the order*, so that the same toy program runs
from a single laptop CPU to a multi-architecture WAN cluster.

## The ask

A scaling hierarchy, from a user sketch:

1. Single machine, single CPU
2. Single machine, multiple CPUs
3. Multiple machines, multiple CPUs, same arch, high-bandwidth LAN
   — 3.1 …on a WAN — 3.2 …multiple architectures (Arm/x86/RISC-V) on a WAN
4. Single machine, multiple CPUs + single GPU
   — 4.1 …multiple GPUs, same card — 4.2 …multiple GPUs, different cards
5. Multiple machines + multiple GPUs on a high-bandwidth LAN
   — 5.1 …multiple architectures on a WAN with mixed GPUs

With two guiding ideas: a **happy path** ("when everything is aligned, use
the *most effective* mechanism") that degrades to a **resilient path**
("when everything is misaligned, use the *most compatible/resilient*
mechanism"); and **dynamic scaling** ("if you can fail, you can probably
add nodes mid-run"). Not every rung must be built — but the *ground-up
shape* should make a future where "the same API runs small-local and
multi-GPU-cluster" believable.

## The core idea: it is one axis, not a dozen features

The hierarchy and the happy-path/resilient duality are **the same axis**.
The distributed-ML literature states it directly:

> Homogeneous + fast network → *tighter coupling buys speed.*
> Heterogeneous + slow/flaky network → *looser coupling buys survival.*

So we don't expose a dozen knobs. You write the experiment **once**; the
only thing that changes from laptop to cluster is **how tightly the
replicas of that experiment are coupled**. A second result makes the
"dynamic scaling" half free:

> Failure-recovery and elastic-resize are the **same three primitives** —
> re-form membership, re-shard, reload state.
> (torchft survived 1,015 injected failures *and* live-resized 30→25→30
> nodes with the same machinery; INTELLECT-1 trained 10B params across 3
> continents on this principle.)

**If a coupling can lose a node, it can gain one.** We build that once, at
the loose end, and both fall out of it.

### The coupling spectrum

Tightest → loosest, each a *real* mechanism that already exists or nearly
does in toy's substrate:

| Coupling | Mechanism | Sweet spot | Toy status |
|---|---|---|---|
| **Fused** | one ggml graph, one device | single CPU/GPU | shipped |
| **Threaded** | one graph, CPU threadpool | multi-CPU, 1 box | ~5 LOC (`set_n_threads` is never called today) |
| **Sharded** | one graph split across local backends (`ggml_backend_sched`) | multi-GPU, 1 box | C scaffold exists (`tnn_session_new_on`, `g_engine_cuda[8]`) |
| **Pooled** | remote devices joined into one graph (ggml **RPC backend**) | LAN memory aggregation | header vendored, not compiled |
| **Federated** | N independent replicas, sync pseudo-gradients every *H* steps (**DiLoCo**) | WAN / heterogeneous / elastic | fits as a Ruby outer loop |
| **Gossip** | async pairwise averaging, no barrier | maximal churn | future |

The architectural fact that makes the loose end cheap: **toy's inner
training step is one fused ggml graph (forward + CE + backward + AdamW)
driven by a thin Ruby loop, and DiLoCo's outer optimizer fits as pure Ruby
orchestration *around* that loop — without unfusing the fast, byte-exact
inner step.** See "Grounding" below.

## The API

One principle: **the experiment is invariant; the topology is the only
dial; placement and sync are policy, not code.**

```ruby
# ── the experiment: written ONCE, byte-identical at every scale ──────────
cfg    = Toy::SmolLM2Config.mha(627, 64, 4, 128, 2, 32, 1e4, 1e-5)
recipe = Toy::Device.from_scratch_recipe
recipe.realize!(cfg, Toy::RecipeOptions.new.tap { |o| o.t_seq = 32; o.seed = 0 })

# ── the run: a recipe + a corpus + a budget + a topology ────────────────
Toy::Run.train(recipe, corpus: "data/ts_seqs.bin", steps: 10_000,
               on: Toy::Topology.local)   # ← the ONLY line that changes
```

Scaling is swapping `on:`, and it maps directly onto the hierarchy:

```ruby
on: Toy::Topology.local                       # 1        single CPU
on: Toy::Topology.threads(:all)               # 2        multi-CPU (one graph, N threads)
on: Toy::Topology.devices(:all)               # 4/4.1/4.2 every local GPU
on: Toy::Topology.pool(%w[gx10:50052 mac])    # 3/5      LAN, RPC memory pool
on: Toy::Topology.federation(peers, sync_every: 500) # 3.1/3.2/5.1 DiLoCo: WAN, multi-arch, elastic
on: Toy::Topology.auto                         # negotiate the tightest coupling the substrate sustains
```

For the explicit, readable loop (toy's single-file-forward ethos), the
distribution stays visible but tiny:

```ruby
Toy::Run.train(recipe, on: topo) do |run|
  run.each_shard("data/ts_seqs.bin", steps: 10_000) do |batch, step|
    loss = run.step!(batch)        # inner step — fused, on-device, byte-exact
    run.sync!(step)                # NO-OP unless this is a federation round boundary
    puts "step #{step}: loss=#{loss}" if run.rank.zero?
  end
end
```

`run.sync!` is the whole trick — **one method, polymorphic on coupling**:

- **Fused / Threaded / Sharded / Pooled** → no-op (the coupling lives
  *inside* the ggml graph; the substrate already shares the work).
- **Federated** → every *H* steps: download the weight set (existing
  `download_row_major` + the GGUF checkpoint writer), all-reduce the
  pseudo-gradient (`Δ = θ_local − θ_round_start`) with the peers present,
  apply the Ruby outer Nesterov step, upload back. Reductions are
  sorted + Kahan-summed → deterministic.
- **Gossip** → exchange with one neighbor, no barrier.

`Toy::Topology.auto` *is* the happy-path/resilient duality: it inspects
the cohort and picks the tightest coupling that is safe — same-arch +
high-bandwidth LAN → Sharded/Pooled (effective); mixed-arch / WAN / flaky
→ Federated (resilient). And it **composes by network tier** (the standard
pattern: tensor/shard *inside* a node → data/federate *across* nodes), so
rung 5.1 is simply `federation(nodes)` where each node is itself
`devices(:all)`.

**Elastic = the loose end's membership, reused.** `federation` owns a peer
set + a quorum `K`; at a round, `sync!` merges whoever showed up
(token-weighted), and a missing peer is simply excluded from that round —
*identical* to one that never joined. A late or newly-added peer rejoins by
pulling current weights, held at merge-weight 0 until it has caught up
(the decoupled-DiLoCo recovery design toy already documented). One
protocol → fault-tolerance *and* dynamic add/remove.

## Grounding: toy's real seams today

From the architecture map (file:line in the seam survey; key points):

- **Single-process, single-session, single-graph.** One `toy train` =
  one OS process (CRuby CLI shells out to a Spinel runner via `Open3`) =
  one `LlamaSeqEngine` with one `@sess` and one combined `graph_b`.
- **The optimizer is fused into the graph**, not a separable Ruby step
  (`build_training_step` emits `opt_step_adamw` nodes into `graph_b`;
  `tnn_compute_backward` runs fwd+bwd+adam in one `graph_compute`). This is
  deliberate (`from_scratch.rb`: "there is no Ruby optimizer to wrap").
  **DiLoCo keeps the inner AdamW fused on-device (cheap, byte-exact) and
  adds the outer optimizer in Ruby** — the grad/weight handles
  (`download_row_major`, `ft_weights`/`ft_globals_weights`) and the GGUF
  fold already exist, so no unfusing is required.
- **Multi-GPU substrate is already in C, validated only at device 0:**
  `tnn_session_new_on(kind, device)`, per-device engine array
  `g_engine_cuda[8]`, `tnn_cuda_get_device_count`, device-aware init/
  teardown. Filed as GH#3 "mode-1 replicated inference." The Ruby
  factory (`Toy::Device.llama_engine`) just needs a `device` argument
  threaded through.
- **The RPC backend is the intended multi-machine-tight seam:**
  `vendor/ggml/include/ggml-rpc.h` is vendored but `librpc` is not built.
- **Multi-CPU is a 5-LOC gap:** ggml's CPU backend threads matmuls, but
  toy never calls `ggml_backend_cpu_set_n_threads` — there is no
  `n_threads` knob anywhere.
- **The data loop is a single sequential byte-offset corpus reader** with
  no shard/rank concept (`train.rb`) — the natural place to add
  `each_shard(rank, world)`.
- **Observability is a head start:** `events.jsonl` already records which
  step each run is at; gradient sentinels already catch divergence early —
  both are exactly what a federation round-coordinator needs.

## Decision locked: own the loose end, lean on ggml for the tight end

The strategic question the prior docs left open — *does toy own
distributed training, or delegate it?* — is **resolved**:

> **Toy owns the loose/federated (DiLoCo-family) end** — it is pure Ruby
> orchestration over toy's existing checkpoint + observability primitives,
> it fits the readable-single-file ethos, and it is small.
> **Toy leans entirely on ggml for the tight end** (RPC + `backend_sched`
> + `layer`/`tensor` split-modes). We do **not** build our own collective
> ops — "PyTorch + NCCL is two decades of engineering."

That single decision is what keeps an ambitious end-state conservative:
the only genuinely new distributed code we author lives in the Ruby
orchestration layer at the loose end; everything tight is the substrate's.

## Build order (each tier independently useful)

- **Tier 0 — single-machine maturity (no distribution; pure wins).**
  `n_threads` knob (→ Threaded, ~5 LOC), batched input `B>1`, gradient
  accumulation (~20 LOC), mixed precision. These speed up the laptop
  happy-path *and* are prerequisites for everything below.
- **Tier 1 — the seam.** `Toy::Topology` + `Toy::Run` with `:local` and
  `:threads` only. Zero distribution risk; establishes the
  invariant-experiment API. Highest leverage, lowest risk.
- **Tier 2 — tight, one box.** `devices(:all)`: replicated (mode-1, the C
  scaffold is built) then sched-sharded (mode-2).
- **Tier 3 — Pooled.** Compile ggml's RPC backend; `pool(...)` for LAN
  inference first (lowest risk, ggml-native), training later.
- **Tier 4 — Federated.** DiLoCo outer loop: single-host two-process →
  two-host TCP → async/late-join + membership. **Hard-gated** (below).
- **Tier 5 — `auto` negotiation + Gossip + full elastic membership.**

## The hard gates (real, from toy's own evidence)

1. **Numeric agreement across backends is not yet good enough for *any*
   cross-replica averaging.** The CPU/CUDA LoRA divergence incident plus
   the still-open ggml `backend_sched` buffer-aliasing bug (masked today by
   `tnn_pin_all_graph_b_nodes`) mean a DiLoCo outer step would compound
   drift. **Root-cause this before Tier 4** — it is the gating dependency,
   not the transport. Mitigation target: f32 master weights everywhere +
   bit-identical (sorted/Kahan) outer step.
2. **DiLoCo may not pay off at toy's characteristic scale** (small models,
   short runs; its sweet spot is ≥400M params, long runs). Run an
   *ablation* before plumbing Tier 4.
3. **Multi-arch bit-exactness is unachievable** (FMA contraction, reduction
   order differ per ISA) and that is *fine* — DiLoCo converts a
   bit-agreement requirement into a statistical-averaging one. But it means
   the tight couplings (Sharded/Pooled) must stay *within* a homogeneous
   tier; cross-arch is Federated-only.

## Open questions for later

- Hybrid CPU+GPU offload (ZeRO-Offload-style m/v on CPU backend) — the
  "missing strategic option" from `backends-and-scale`; only matters at
  ≥1B params toy doesn't yet target.
- Whether `pool` (RPC) training (not just inference) is worth it given
  RPC's cold-load-streams-the-whole-model cost.
- The asymmetric-ownership topology (a laptop that is *both* synchronizer
  and learner) — plausible but unbenchmarked.

## What this is NOT

- Not a commitment to build every rung — it is the *shape* that makes the
  rungs reachable with one API.
- Not toy-native collective ops / a NCCL competitor (see the locked
  decision).
- Not started: every tier sits behind Tier 0, and Tier 4 behind gate #1.
