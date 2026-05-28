# Toy as a framework: roadmap (DRAFT)

**Status:** roadmap / critical path. Companion to
[`toy-framework-design-2026-05-28.md`](toy-framework-design-2026-05-28.md)
— the design doc is "what we're building"; this is "how, in what
order, with what gates".
**Date:** 2026-05-28.

## Goal

Reach a state where:

- **Toy is a framework**, not a kit of examples. New users `toy new
  myproj` and have a working ML project in minutes — no Makefile
  archaeology, no per-backend setup decisions, no environment
  variables to chase.
- **Algo classes follow the contract.** Arches, Blocks, Primitives,
  Recipes — each is one Ruby file the user can read top to bottom.
  Each produces a Card via derivation from `realize`, not via
  hand-written `algorithm` methods.
- **The repo is clean.** The accidentals of "this was an
  exploratory monorepo" are gone. `lib/` holds the framework
  primitives + stdlib modules; `examples/` holds tiny scenario
  drivers (or retires entirely); `tep_demo/` is one
  `openai_api_llama` server; tests gate the contract.
- **Tao consumes toy** through the documented Card surface (events
  + Recipe introspection), not by reverse-engineering `lib/`.

This roadmap is the path to that state. Six phases; each has a
deliverable, an exit gate, and a cleanup arc.

## Critical path summary

```
P0 design lock  →  P1 derivation  →  P2 layering  →  P3 CLI MVP
                                                       │
                                                       ▼
                              P4 CLI complete  ←─  P5 generators
                                      │
                                      ▼
                              P6 Prism lowerer  (optional)
```

P0 → P1 → P2 are inside-toy-repo refactors that don't change user
behaviour. P3 → P4 are the user-facing surface. P5 → P6 are
optimisation / quality-of-life.

The critical-path bet: **prove the contract under real pressure
(P1 + P2) before exposing it to users (P3+).** If the contract
doesn't fit the existing code, we want to find out before users
write algos against it.

---

## P0 — Design lock + Tao coord

**Deliverable:** this doc + the design doc landed and reviewed; a
Tao-side issue filed announcing the `runs/<id>/` layout.

**Inputs:** `toy-framework-design-2026-05-28.md` v3 (committed); a
Tao GitHub issue opened and linked.

**Gate (exit criteria):**

- Design doc decisions table (§13) all marked locked.
- Tao issue filed; Tao acknowledges layout (no objections, or
  objections resolved here).
- This roadmap merged.

**Risks:** Tao has hidden assumptions that conflict with our
proposed layout. Mitigation: file the issue early, not at P0 exit.

**Cleanup arc:** none yet. Pure design.

**Estimated effort:** 1 day.

---

## P1 — Card derivation refactor

**Deliverable:** the runtime probe path lands in `lib/toy_card.rb`
+ `lib/toy_describe_flow.rb`. `Toy::SmolLM2` and the seq-mode
training graph produce Cards by derivation (probe → graph walk →
structural Card → renderer) instead of via hand-written
`algorithm` methods. New parity test: derived Card vs current
hand-written one, asserted equivalent.

**Gate:**

- `Toy::SmolLM2#card` (derived) renders the same pseudocode as
  today's hand-written `algorithm` for at least three reference
  configs (135M, 1.5B-shape, ViT-Tiny-shape).
- `LlamaSeqForwardFFICache`'s forward graph derives a valid
  Recipe Card (skeletal — Arch + the implicit Trainer for now).
- No behavioural change to existing examples / benches / tests.
- `make check-cards` still passes (drift detector should now be
  comparing derived-vs-derived; trivially identical).

**Risks:**

- Runtime probe may need very small dims to be fast — keep the
  Card emission decoupled from cfg size.
- Derived Cards may *lose* nuance present in hand-written ones
  (e.g., math notation). Accept: hand-written `card` override
  stays available for prose-quality renders; the framework
  doesn't *require* it.
- Some algos rely on conditional branches in `realize` that a
  runtime probe doesn't see. Document as a known gap; Prism
  lowerer (P6) closes it.

**Cleanup arc:**

- Hand-written `algorithm` methods on `Toy::SmolLM2`,
  `Toy::SmolLM2Block`, `Toy::GPT2`, etc. are **deprecated** (kept
  for the parity test; not deleted until P6).
- `lib/toy_card.rb`'s authoring API (`step_bind`, `step_update`,
  etc.) becomes "framework-internal" — users never call it.

**Estimated effort:** 5-7 days (the parity test surface area is
the long pole; the derivation itself is ~300 LOC).

---

## P2 — Five-layer refactor of stdlib

**Deliverable:** `lib/llama_seq_forward_ffi.rb` (and its mirrors)
factored into the L1/L2/L3 hierarchy:

- **L1:** named primitives in `lib/toy/llm/primitives/` — one file
  per primitive (RMSNorm, RoPE, GQA, MHA, SwiGLU, ...). Each
  registers itself.
- **L2:** blocks in `lib/toy/llm/blocks/` — TransformerBlock
  composes primitives. SSMBlock proves the contract holds for
  Mamba.
- **L3:** archs in `lib/toy/llm/archs/` — LlamaArch, GPT2Arch,
  ViTArch each one file. Each composes blocks.
- **L4:** recipes in `lib/toy/llm/recipes/` — FromScratchRecipe,
  LoRARecipe, WarmStartRecipe, plus a CurriculumRecipe template
  showing the multi-stage shape.

**Gate:**

- All existing examples (`examples/01_inference.rb` through
  `examples/09_warm_start_train.rb` + the smoke_*.rb battery)
  still build clean and produce bit-identical CE / loss / logits
  to before P2.
- Bench numbers (`make bench`) within ±2% of pre-P2 baseline
  (the refactor shouldn't introduce overhead; if it does, fix it
  before exiting the phase).
- One new SSMBlock (or stub) proves the Block contract
  generalises beyond transformer. Mamba-shaped smoke that builds
  + runs at toy dims is sufficient.
- Per-layer overrides demonstrably work: `LlamaArch` with
  `n_heads: [16, 16, 8, 4, 2, 1]` realizes and runs.

**Risks:**

- The CUDA/Metal mirror generator (`prep/gen_cuda_mirror.rb`)
  was sized for the monolithic `llama_seq_forward_ffi.rb`. May
  need extension for the layered structure. Worst case: keep the
  monolithic mirror generator and have it walk the layered
  source as input. Decide before refactoring.
- Spinel poly-dispatch landmines around inheritance + ivar typing
  (the same landmines that made us write the per-class mirrors)
  may bite when blocks compose primitives. Mitigation: hold a
  Spinel-friendliness review at the gate.

**Cleanup arc:**

- `lib/llama_seq_forward_ffi.rb` retires — its contents distributed
  across L1/L2/L3 files. CUDA/Metal mirrors regenerated against
  the new structure.
- `lib/transformer.rb`'s `Toy::*Block` classes either move into
  L2 or get absorbed (depending on whether they're worth keeping
  as pure-Ruby readable references).
- `lib/toy_smollm2.rb` slims down to "LlamaArch with SmolLM2
  default cfg".

**Estimated effort:** 10-15 days. This is the heaviest refactor.

---

## P3 — Core + CLI MVP

**Deliverable:** `bin/toy` as CRuby (not yet Spinel-compiled).
Five commands: `new`, `install`, `list`, `fetch`, `describe`.
Project layout (`toy new`) creates the conventional directory
tree. `toy --manifest` lands.

**Gate:**

- `toy new myproj && cd myproj && toy install` works end-to-end
  on a fresh checkout (macOS Apple Silicon + Linux x86_64 + Linux
  aarch64 verified).
- `toy fetch ...` drops `data/<basename>.gguf` symlinks reliably.
- `toy list` finds anything in HF / Ollama / LM Studio caches +
  project `data/`.
- `toy describe <model>` reads GGUF metadata and renders the
  arch's derived Card.
- `toy --manifest` emits a JSON manifest CC can consume.
- An empty `toy.yml` is honoured (defaults for run-id template +
  algo-discovery path; backend auto-detected at run time).

**Risks:**

- The "five commands" surface may already be too constrained;
  watch for missing primitives during dogfooding.
- Project-dir-vs-framework-dir behaviour edge cases (running
  outside a project should give a clear error, not a stack
  trace).

**Cleanup arc:**

- `examples/05_list_models.rb` retires (superseded by `toy list`).
- `prep/fetch_model.sh` either becomes `toy fetch`'s
  implementation or stays as the script that `toy fetch` shells
  out to. Decide based on which is simpler at the gate.
- `make hello` retires (replaced by `toy new myproj` flow).

**Estimated effort:** 5-7 days.

---

## P4 — CLI complete (train / serve / infer / eval)

**Deliverable:** `toy infer`, `toy serve`, `toy train`, `toy eval`.
Each composes the layered algo classes from P2 through a clean
CLI surface. Events emission becomes automatic (the framework
wraps the inner loop).

**Gate:**

- Every existing example shape is reproducible via the CLI:
  - `examples/01_inference.rb` → `toy infer <model> --prompt ...`
  - `examples/06_train_from_scratch.rb` → `toy train from-scratch
    --arch llama --data tinystories ...`
  - `examples/09_warm_start_train.rb` → `toy train warm-start
    --arch llama --donor <gguf> --pca-init ...`
  - `examples/08_lmc.rb` → `toy eval lmc --ckpt A --other B`
  - `tep_demo/openai_api_llama.rb` → `toy serve <model> --port N`
- A CurriculumRecipe (multi-stage) runs end-to-end via
  `toy train curriculum --stages ...`.
- Tao consumes a sample run via the Recipe Card + events stream;
  Tao confirms no reverse-engineering of `lib/` required.

**Risks:**

- The `toy train` knob surface is wide (LR, schedule, batch,
  grad accum, dtype, ...). Easy to over-design. Mitigation:
  let the Recipe's `.with_hyper / .per_layer / ...` chain be the
  knob surface; CLI flags are thin sugar.
- Tao integration may surface gaps in the Card spec. Tao file
  issues; we fix during P4.

**Cleanup arc:**

- `examples/01`, `06`, `08`, `09` retire to a
  `docs/scenarios/` directory as canonical references (or get
  rewritten as `Toy::Recipes::*` definitions that the CLI just
  invokes).
- `examples/03_finetune_lora.rb` retires (replaced by `toy
  train lora ...`).
- `examples/04_serve_http.rb` deletes (already known-broken;
  `toy serve` replaces it).
- `tep_demo/openai_api_llama.rb`'s endpoint code moves into
  `lib/toy/serve/openai/` as a Server class.
- `tep_demo/openai_api.rb` (the legacy GPT-2 server) either
  modernises into a registered Server or retires.

**Estimated effort:** 10-12 days.

---

## P5 — Generators

**Deliverable:** `toy g arch <name> --based-on <family>`,
`toy g recipe <name>`, `toy g primitive <name>`. Generators
scaffold thin files that require framework primitives (per §9 of
the design doc), not bulk copies.

**Gate:**

- `toy g arch my-llama --based-on llama` produces an
  `algos/my_llama.rb` that compiles + runs + composes correctly.
- Generators emit only the files the user needs to *vary*;
  shared bits stay required from the framework.
- Generated code passes `toy describe my-llama` cleanly (the
  derived Card looks right).

**Risks:**

- Generators are where opinions bake in. Be slow + conservative
  here. Resist scaffolding too much.
- Spinel-friendliness — generated code must compile under
  current Spinel without landmine surprises.

**Cleanup arc:**

- `prep/gen_cuda_mirror.rb` may retire if the multi-backend
  story consolidates (e.g., if `toy generate-mirror` subsumes
  it). Otherwise stays.
- Possibly retire `consuming-toy.md` (the vendoring story) in
  favour of "use `toy g` from inside a project" — depends on
  whether vendoring still has independent value at this point.

**Estimated effort:** 5-7 days.

---

## P6 — Prism lowerer integration (optional)

**Deliverable:** `toy build` runs the Prism lowerer over
`algos/*.rb`, producing Spinel-friendly specialised Ruby. Same
walker emits the structural Card statically. AST-derivation
becomes the canonical Card source; runtime probe becomes a
fallback / cross-check.

**Gate:**

- Lowerer's Card output matches runtime probe's Card for all
  stdlib archs (parity gate).
- At least one Spinel landmine retires because the lowerer
  emits Spinel-friendly code (the demonstrable concrete value).
- Build-time overhead acceptable (< 5s for a typical algo file).

**Risks:**

- Lowerer scope creep: the lowerer-design doc estimates ~500 LOC
  for the Prism walker. Hold the line.
- Some Spinel landmines won't be addressable by the lowerer;
  document the residue.

**Cleanup arc:**

- Hand-written `algorithm` methods (deprecated in P1) finally
  retire — only one source of truth (the realize body + AST
  walker) remains.
- `make check-cards` retires.
- The `def card` override path stays available but documented
  as exceptional.

**Estimated effort:** 10-15 days. Defer until P5 patterns settle.

---

## Phase dependencies + parallelism

| Phase | Depends on | Can start when |
| --- | --- | --- |
| P0 | — | now |
| P1 | P0 lock | P0 exit |
| P2 | P1 derivation working on `Toy::SmolLM2` | P1 exit |
| P3 | nothing (CLI is independent of layering!) | P0 exit — can run in parallel with P1+P2 |
| P4 | P2 layering + P3 CLI shell | both exits |
| P5 | P4 patterns observed | P4 exit + observation period |
| P6 | P5 generators stable | P5 exit |

**Parallelisation note:** P3 (CLI MVP — `new`, `install`, `list`,
`fetch`, `describe`) can technically start at P0 exit because it
doesn't depend on the algo layering refactor. The `describe`
command for *existing* archs uses today's hand-written Cards;
after P1 it transparently uses derived ones. Same for `list` and
`fetch` (which don't touch algos at all).

Whether to actually parallelise depends on team capacity. If
single-threaded, the order is P0 → P1 → P2 → P3 → P4 → P5 → P6.
If two tracks, P3 can run alongside P1+P2.

---

## Risks across the whole arc

**Risk 1: Spinel-friendliness regressions.** The layering refactor
(P2) and the generators (P5) both produce more Ruby that Spinel
has to compile. Mitigation: run `make bench` + the example smoke
battery at every phase gate. Any new landmines get filed
upstream; if they block, the affected layer stays monolithic
until Spinel fixes.

**Risk 2: Tao alignment drift.** We coordinate at P0 (filed
issue). If Tao's needs evolve mid-roadmap and contradict the
Recipe Card surface, we have to choose: extend the Card spec
(slow) or let Tao reverse-engineer (defeats the contract). The
right move is to extend the spec; build slack into P4 for this.

**Risk 3: `toy.yml` minimum creeps.** The decision (§13 of design
doc) is to keep `toy.yml` as empty as possible. At each phase we
verify no new "this should be in `toy.yml`" wishes have landed.
If genuinely needed: add, but cite the use case in the doc.

**Risk 4: Spinel version churn.** The decision is to *not* pin
Spinel — toy serves as a test case for Spinel breakage. This
means each phase may discover new Spinel landmines. Cleanup
arc per phase explicitly notes any new patterns to memorialise.

**Risk 5: User-visible surface paint.** Once `toy <cmd>`
commands ship to users (P3+), changing them is costly. Defer
the commit until P3 gate — earlier phases stay inside toy's
repo. (This is why P1 + P2 happen first.)

---

## Cleanup arc — the "on the other side" state

By end of P6 the repo should look like:

```
toy/
├── bin/toy                  # the CLI binstub (CRuby)
├── lib/toy/                 # framework Core + Stdlib
│   ├── core/                # CLI dispatch, Card IR, registry, …
│   ├── llm/                 # toy-llm: primitives, blocks, archs, recipes
│   ├── ggml/                # toy-ggml: the FFI bridge (lib/tinynn*.rb consolidates here)
│   ├── train/               # toy-train: the trainer base + standard trainers
│   ├── serve/               # toy-serve: openai_api server class + transport
│   └── bench/               # toy-bench: the regression-gate framework
├── prep/                    # data-side scripts (HF converters, pretokenizers)
├── docs/                    # current reference + roadmap + archive
└── (no examples/ — retired or moved to docs/scenarios/)
└── (no tep_demo/ — folded into lib/toy/serve)
```

What deletes between today and P6 end:

- `examples/*.rb` (retired or moved)
- `tep_demo/openai_api*.rb` (folded into lib/toy/serve)
- `tep_demo/_tep_lib/` (already retired)
- `prep/gen_cuda_mirror.rb` (potentially, if generators subsume)
- `Makefile` shrinks dramatically (the `toy` CLI replaces most targets)

What stays (the framework's home):

- `lib/toy/` as above
- `tinynn/` (the C shim — the FFI bridge to ggml is still there)
- `vendor/` (ggml + ggml-cuda + ggml-metal builds)
- `prep/` (data-side; not algo-side)
- `docs/`
- `bench/` (regression baselines + check.rb)

---

## What this roadmap is NOT

- **Not a commitment to do everything.** P6 in particular is
  optional. If P5 generators feel good and the runtime
  derivation path is "good enough", we may never build the
  Prism lowerer.
- **Not a feature list.** Specific algo additions (MoE, diffusion,
  new modern-LLM primitives) are orthogonal — they happen
  inside the layered structure once it's built. They don't
  block phases.
- **Not a Tao roadmap.** Tao's L5 work runs in parallel and is
  tracked separately. We coordinate at the Recipe Card surface.

---

_Draft v1 of the roadmap. Amend in place; mark phase completions
with `[DONE 2026-XX-XX]`, gate slippages with `[BLOCKED: ...]`,
risks materialised with `[REAL: ...]`._
