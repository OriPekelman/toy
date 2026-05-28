# Toy as a framework: design doc (DRAFT v3)

**Status:** design doc; v3 after second review round. All open
questions answered; this is the "what" — the roadmap doc
([`toy-framework-roadmap-2026-05-28.md`](toy-framework-roadmap-2026-05-28.md))
is the "how".
**Date:** 2026-05-28.
**Companion docs:**
[`toy-framework-roadmap-2026-05-28.md`](toy-framework-roadmap-2026-05-28.md),
[`lowerer-design.md`](lowerer-design.md),
[`spinelgems-tep-adoption-2026-05-27.md`](spinelgems-tep-adoption-2026-05-27.md),
[`backends-and-scale-2026-05-27.md`](backends-and-scale-2026-05-27.md).

## TL;DR

Promote toy from "library you `require_relative` from your own
Spinel-compiled experiment" to **"framework you create projects
inside of"** — `toy new myproj`, `toy train ...`, `toy serve ...`,
with a CLI that owns the per-project plumbing.

The lever: **`Toy::Card` as the IR contract**, organized in four
granularity layers (Primitive / Block / Arch / Recipe — L5 is Tao
territory) and filled by five kind slots (Arch / Trainer /
Decoder / Eval / Server, plus DataSpec). The Card describes the
work; the framework composes, introspects, and runs.

Key insight from review: **Cards are derived, not authored.** Users
write the realize implementation; the framework derives the Card
via graph-walk (today, leveraging `ToyDescribeFlow`) or AST walk
(future, via the Prism lowerer). No more hand-writing
`step_update("v^j", "x · W_V^j + b_V^j", ...)`.

**Why this matters.** ML practitioners currently deal with a lot
of Python ugliness and ceremony; for newcomers it makes
understanding the actual ML flow difficult. Toy comes from the
belief that ML is going to be a larger part of our lives — so
understanding the structure and algorithms is primordial. We want
to avoid gate-keeping and ceremony.

Explicit non-goal in v1: a generic "Spinel project generator base
library". We may *accidentally* land something extractable; we will
not design *for* it.

---

## 1. Why now — the lever

Two observations converging.

**(a) On-boarding friction is real.** A new Toy-based project
currently has installation woes (the `make` flow is addressing
those) and, more importantly, a question of *responsibilities*.
What does the framework own? What does the user own? What does a
consumer like Tao that wants to generate or operate on toy code
own? Without a contract surface, every consumer has to
reverse-engineer toy's internals.

**(b) `Toy::Card` is already an IR.** It exists at
`lib/toy_card.rb`. `SmolLM2` and `SmolLM2Block` build Cards from
`algorithm` methods. `lib/toy_describe_flow.rb` walks the runtime
compute graph and emits structured descriptions. The pieces of the
introspection substrate are in place; what's missing is the
framework layer that *consumes* them and surfaces them through a
CLI.

**Operating philosophy.** Conventions (Rails-style) remove
decisions the user shouldn't be making, but compose explicit
primitives, not magic. We are in **explicit territory** —
no metaprogramming "advanced magic". Code generation is cheap
these days; clarity and simplicity aren't. Every convention has
an escape hatch; every abstraction is one file you can read.

## 2. Two distinctions that matter

**Library vs framework:** today toy is consumable as a library
(gemspec shipped 2026-05-27, toy#19). The framework layer adds a
*third* place beyond toy's `lib/` and `examples/`: the **user's
project directory**. The CLI runs there, not inside toy's checkout.

**Card layers vs algo kinds — orthogonal axes.** Don't conflate.

| Axis | What it specifies |
| --- | --- |
| **Card LAYER** (§4) | Granularity of the IR — how big a unit is. Primitive / Block / Arch / Recipe. (L5 Experiment is Tao territory.) |
| **Kind SLOT** (§5) | Role of the unit in a workflow — what it does. Arch / Trainer / Decoder / Eval / Server / DataSpec. |

A *Recipe* (Layer 4) might be "FromScratch trainer + Llama arch +
TinyStories dataset" — composing one of each Kind. A Layer-1
Primitive (`RMSNorm`) is also one of a Kind (Arch-side primitive).
Layers describe *size*; Kinds describe *role*.

## 3. Three concentric circles

```
┌─────────────────────────────────────────────────────────┐
│  Add-ons (community gems)                               │
│  toy-diffusion, toy-vision-tower, toy-music-moe, …      │
├─────────────────────────────────────────────────────────┤
│  Stdlib (bundled, each its own gem-shaped module)       │
│  toy-ggml · toy-llm · toy-train · toy-serve · toy-bench │
├─────────────────────────────────────────────────────────┤
│  Core (always thin)                                     │
│  CLI · project layout · Toy::Card IR · event stream     │
│  composition operators · backend interface · manifest   │
└─────────────────────────────────────────────────────────┘
```

**Core invariants:** ~500-1000 LOC. No GGML dependency. No Gemfile
assumption. All commands return JSON when `--json` is passed.

**Stdlib boundary:** if you want LLMs, you pull `toy-llm`. If you
want training, `toy-train`. Weird agents wanting only event stream
+ CLI: just Core.

**Add-ons:** same interface as Stdlib modules. Stdlib is just
"what ships with toy by default" — no privileged status.

**Distribution.** Ship as a published gem with a `toy` binstub.
`gem install toy` works. Whatever Ruby manager / installer the
user prefers (rbenv, rv, asdf, system Ruby) is fine — the
framework doesn't care.

## 3.5 Filesystem layout (framework + project)

The layering is real, so it shows in the directory tree. Not
everything flat in `lib/`. Filesystem structure mirrors the
contract.

**Framework side** (the `toy` gem):

```
lib/toy/
├── core/                    # the always-thin framework
│   ├── card.rb              # Toy::Card IR
│   ├── card_renderer.rb     # → pseudocode / mermaid / json
│   ├── registry.rb          # arch_name / trainer_name registration
│   ├── cli/                 # one file per CLI command
│   │   ├── new.rb · install.rb · list.rb · fetch.rb · describe.rb
│   │   └── infer.rb · serve.rb · train.rb · eval.rb · manifest.rb
│   ├── backend.rb           # backend interface (not impls)
│   └── events.rb            # toy/v1 event stream emitter
│
├── ggml/                    # toy-ggml: backend impl on ggml
│   ├── tinynn.rb · tinynn_cuda.rb · tinynn_metal.rb  (consolidated)
│   └── …
│
├── llm/                     # toy-llm: LLM-specific stdlib
│   ├── primitives/          # L1 — one file per named op
│   │   ├── rms_norm.rb · rope.rb · softmax.rb
│   │   ├── mha.rb · gqa.rb · swiglu.rb
│   │   ├── token_embed.rb · patch_embed.rb · …
│   ├── blocks/              # L2
│   │   ├── transformer.rb
│   │   └── ssm.rb           # for Mamba; proves Block contract generalises
│   ├── archs/               # L3 — one file per arch
│   │   ├── llama.rb · gpt2.rb · vit.rb · mamba.rb
│   └── recipes/             # L4 — one file per training plan
│       ├── from_scratch.rb · lora.rb · warm_start.rb
│       └── curriculum.rb
│
├── train/                   # toy-train: training infrastructure
│   ├── trainer.rb · optimizers/ · schedule/
│
├── serve/                   # toy-serve: HTTP serving
│   ├── openai/              # server class + endpoint handlers
│   └── transport/
│
└── bench/                   # toy-bench: regression-gate framework
    └── check.rb · …
```

The five stdlib modules (`ggml`, `llm`, `train`, `serve`, `bench`)
each map to a published gem. `core/` ships in the main `toy`
gem. Add-on gems (`toy-diffusion`, `toy-music-moe`) follow the
same shape under their own `lib/<gem>/`.

**Project side** (what `toy new myproj` creates):

```
myproj/
├── toy.yml                  # minimal config (run-id template, algo path)
├── algos/                   # user code — same layering as framework's llm/
│   ├── primitives/          # custom primitives (rare)
│   ├── blocks/              # custom blocks (rare)
│   ├── archs/               # custom architectures (common)
│   │   └── my_llama.rb
│   └── recipes/             # training plans / curricula (common)
│       └── my_curriculum.rb
├── data/                    # GGUFs (HF-cache symlinks ok), corpora
├── runs/                    # event streams + checkpoints (Tao reads here)
│   └── <run_id>/
│       ├── events.jsonl
│       └── weights/{step_N.gguf, latest}
└── bin/toy                  # optional binstub
```

User project mirrors framework shape. A custom arch lives at the
same logical path inside the user's project (`algos/archs/`) as a
stdlib arch lives inside the framework (`lib/toy/llm/archs/`).
The framework's registry / discovery walks both transparently.

Subdirectories under `algos/` are optional — a project with just
a single custom arch can drop `algos/my_llama.rb` and the
framework infers the layer. The convention scales up to deeper
projects without imposing on small ones.

## 4. The four Card layers (granularity)

A Card exists at every level. Same IR type; layered composition.

| Layer | Unit | Examples (stdlib) | What practitioners vary |
| --- | --- | --- | --- |
| **L1 Primitive** | A single named op | `LayerNorm`, `RMSNorm`, `RoPE`, `Softmax`, `MHA`, `GQA`, `SwiGLU`, `PatchEmbed` | Implementation of one op |
| **L2 Block** | One state-threading unit | `TransformerBlock`, `SSMBlock` (Mamba) | Which primitives, with what cfg |
| **L3 Arch** | Stack of blocks + embedding + head | `LlamaArch`, `GPT2Arch`, `ViTArch`, `MambaArch` | Block count, per-layer overrides, embedding/head choice |
| **L4 Recipe** | A training plan — one or more stages, each composing Arch + Trainer + DataSpec (+ optional Decoder + Eval) | `FromScratchRecipe`, `LoRARecipe`, `WarmStartRecipe`, `CurriculumRecipe` | Stage sequence, optimizer choice, schedule, dataset progression |

**Recipes include curriculum.** A Recipe is "a training plan",
which may be one stage (single dataset, single optimizer, one
loop) or several (curriculum learning — progressively harder data,
phase transitions, schedule changes). Each stage composes Arch +
Trainer + DataSpec; state threads between stages; the Recipe owns
the sequence.

```ruby
recipe = Toy::Recipes::Curriculum
  .arch(:llama)
  .stage(:warmup,    trainer: :from_scratch, data: :tinystories_short, steps: 100)
  .stage(:expand,    trainer: :from_scratch, data: :tinystories_full,  steps: 500)
  .stage(:finetune,  trainer: :lora,         data: :domain_specific,   steps: 200)
```

One-stage Recipes are the default; multi-stage curricula are
a strict superset.

**L5 Experiment** (varying a Recipe along an axis — sweeps,
ablations, LMC pairs) is **Tao's territory**, not toy's. Tao reads
the Recipe Card to know what knobs exist, then drives the sweep.
Toy's top layer is L4.

Every layer is a file you can read and copy. Layer N composes
Layer N-1.

## 5. The five Kind slots (role)

Each Kind has its own contract — what it receives, owns, emits.

| Kind | Receives | Owns | Emits |
| --- | --- | --- | --- |
| **Arch** | cfg, sess, mode | forward graph, cache | (none) |
| **Trainer** | arch_cache, batch | opt step | `step`, `drift`, `grad`, `sample` events |
| **Decoder** | arch_cache, prompt | autoregressive loop, sampling | token IDs |
| **Eval** | arch_cache, corpus | metric computation | `eval` events |
| **Server** | state, transport | endpoint dispatch | server-side events with `phase: "serve"` |

**Sixth kind: DataSpec.** A unified "this is the data" object
(tokenizer + corpus loader + sequence shape + curriculum stage
marker). Today split between Ruby loaders and Python
pretokenizers. First-class to remove the "where does the data
come from" footnote from every other kind.

None of these depend on toy-ggml directly. They use whatever
session was built with — be it ggml, future pure-Ruby, or
something else. Backend is a separate axis (§8).

## 6. The Block contract

Generalized enough to fit transformers AND Mamba/SSM AND future:

```ruby
class Toy::Block
  # Explicit shape declarations — used for static compat checking
  # and for shape annotations in derived Cards.
  input  :x,         shape: "[T, D]"             # main input
  state  :s_in,      shape: "[L_max, 2, T, D_h]" # state from previous step
  output :h,         shape: "[T, D]"             # main output
  state_out :s_out,  shape: "[L_max, 2, T+1, D_h]"

  # Required: realize against a backend session.
  # mode ∈ {:infer, :train}.
  # Returns a cache exposing (h_tensor, s_out_tensor).
  def realize(cfg, sess, mode); ...; end
end
```

State is **the** abstraction that lets a Block be a transformer
block (state = KV cache), a Mamba block (state = SSM hidden),
diffusion (state = timestep), etc.

Explicit shapes are not optional. They drive:
- Compatibility checks before realize (does this Arch fit this
  Block?).
- Shape annotations in the derived Card without runtime probing.
- Error messages that say "expected `[T, D]`, got `[T, D_h]`".

## 7. Composition operators (the API surface)

Cards (at any layer) expose:

| Operator | Effect |
| --- | --- |
| `card.with_hyper(k, v)` | Override a scalar hyperparameter |
| `card.per_layer(field, list)` | Vary a field across the layer axis (the killer one — supports heads-per-layer `[16,16,8,4,2,1]`) |
| `card.replace_step(name, new_step)` | Swap a named step (e.g., replace attention with ALiBi-attn) |
| `card.tap(name) { |tensor| ... }` | Runtime hook on a named intermediate |
| `card.make_trainable(param)` | Convert a constant to a trainable param (e.g., learnable `rope_base`) |
| `card.replace_primitive(name, impl)` | Swap a registered primitive without subclassing |

These compose Cards declaratively. Researchers write:

```ruby
recipe = Toy::Recipes::FromScratch
  .arch(:llama)
  .with_hyper(:d_model, 512)
  .per_layer(:n_heads, [16, 16, 8, 4, 2, 1])     # pyramid-down
  .replace_primitive(:attention, :mha_with_alibi)
  .make_trainable(:rope_base)
```

`per_layer` is first-class. Without it, "vary anything across L"
forces fork-and-edit.

The operator set is machine-readable so Tao (L5) can introspect
what knobs a given Recipe Card exposes and systematically vary
them across sweep arms.

## 8. Cards are derived, not authored

The biggest concern from review: hand-writing
`step_update("v^j", "x · W_V^j + b_V^j", "v^j ∈ R^{T×D_h}", "V not rotated")`
is dangerous. Easy to get wrong; silently drifts from code over
time; high bar for users.

**Resolution:** users do not write Cards. Users write `realize`.
The framework derives the Card.

Two derivation paths, both producing the same Card type:

| Path | When | Mechanism |
| --- | --- | --- |
| **Runtime probe** (today) | Always available | Probe session with tiny dims → `realize` → walk the resulting compute graph (extend `lib/toy_describe_flow.rb`) → emit structural Card |
| **Static (Prism lowerer)** (future) | Build time | Walk AST of `realize` → emit Card with conditional/loop branches a runtime probe couldn't see → optionally also emit Spinel-friendly lowered Ruby ([`lowerer-design.md`](lowerer-design.md)) |

A `def card` method becomes **optional** — only used to override
the derived version with hand-tuned prose (Phuong-Hutter
pseudocode descriptions, math notation, etc.). The default is
"derived from realize, rendered as pseudocode by the framework".

This eliminates the drift footgun. Code IS the contract.
Introspection is automatic.

`make check-cards` (today's Ripper-based drift detector) retires
when the lowerer ships — because there's only one source of
truth.

## 9. DRY: require, don't copy

`toy g arch my-llama --based-on llama` does NOT copy stdlib's
primitives into the user's project. It scaffolds a thin
`algos/my_llama.rb` that *requires* the primitives from the
framework and composes them:

```ruby
# algos/archs/my_llama.rb — scaffolded by toy g arch my-llama --based-on llama
require "toy/llm/primitives"   # RMSNorm, RoPE, GQA, SwiGLU, ...
require "toy/llm/blocks"        # TransformerBlock

class MyLlama < Toy::Arch
  arch_name "my-llama"
  arch_family "llama"            # for cross-arch tooling; further-precision allowed
  config :d_model, :n_layers, :n_heads, :vocab

  def realize(cfg, sess, mode)
    tok = Toy::Primitives::TokenEmbed.new(cfg.vocab, cfg.d_model)
    blocks = cfg.n_layers.times.map do |li|
      Toy::Blocks::Transformer.new(cfg,
        attn: Toy::Primitives::GQA.new(cfg.n_heads, cfg.n_kv),
        norm: Toy::Primitives::RMSNorm.new(cfg.d_model, cfg.rms_eps),
        ffn:  Toy::Primitives::SwiGLU.new(cfg.d_model, cfg.d_ff))
    end
    # ... wire them up; framework derives the Card from this body
  end
end
```

To override a specific primitive, the user vendors just that
primitive into their project's `algos/` and registers it. The rest
keeps requiring from the framework. No bulk copy.

This balances "single file per arch" (the file is yours, readable
top to bottom) with DRY (shared math doesn't get duplicated).

## 10. Multi-backend extensibility

Today's per-class mirror generation (CPU + CUDA + Metal mirrors
generated by `prep/gen_cuda_mirror.rb`) is a Spinel-poly workaround,
not a fundamental constraint.

Three pragmatic paths for add-on algos:

- **(a) CPU-only first.** Algo ships only the CPU path. User runs
  `toy generate-mirror algos/my.rb` for CUDA/Metal when needed.
- **(b) Use only `toy-ggml` primitives.** Those already dispatch
  through `lib/tinynn{,_cuda,_metal}.rb`. Your algo is
  automatically multi-backend.
- **(c) Future:** if Spinel ships polymorphism, mirrors retire.
  The Card stays valid regardless.

(b) is the recommended path; (a) is the fallback for algos with
custom ops; (c) is upstream-dependent.

## 11. CC-tools-friendly defaults

- `toy <cmd> --json` everywhere — structured for CC, pretty for
  humans by default.
- `toy --manifest` — JSON manifest of commands + args. CC
  discovers the surface without parsing `--help`.
- Exit codes distinguish "your input was wrong" (2) from "I failed
  to do the thing" (1).
- Stable error JSON shape.

Slash commands / skills are wrappers around the CLI, not separate
surfaces.

## 12. The Tep-generator non-goal

Tep would also benefit from a generator. Resist extracting
"spinelgen" base library in v1. Web-shape and ML-shape projects
diverge sharply. Common base would be ~200 LOC of orthogonal
plumbing (argv parsing, dispatch, JSON output, manifest). Build
toy first; if after shipping the plumbing is genuinely identical
across Toy and Tep, extract then.

Don't design *for* extraction. Don't design *against* it either.

## 13. Decisions (all locked after review round 2)

| Decision | Resolution |
| --- | --- |
| **Card layers** | Four layers in toy (Primitive / Block / Arch / Recipe). L5 Experiment is Tao's. |
| **Kind slots** | Five plus DataSpec (the sixth). |
| **Block contract** | `(input, state) → (output, state)` with explicit shape declarations. |
| **Composition operators** | `with_hyper / per_layer / replace_step / tap / make_trainable / replace_primitive`. |
| **Cards derived, not authored** | `def card` is optional override only. |
| **DRY for primitives** | `toy g` scaffolds require framework primitives; user vendors specific ones to override. |
| **Recipes include curriculum** | Multi-stage by design; single-stage is the common case. |
| **Tao coordination** | We file a Tao-side issue noting the layout. We decide here; Tao follows. |
| **`toy.yml`** | Minimal — defaults for run-id template + algo-discovery path. Backend is runtime (achieves parity); not in config. |
| **Naming + discovery** | Minimal `arch_name "llama"`; allow further precision (`"llama2"`, `"qwen3.6"`, etc.). `arch_family` for cross-arch tooling. |
| **First deliverable** | Card-derivation refactor (proves §8 against `Toy::SmolLM2` + `LlamaArch` *inside toy's repo*) **before** any user-facing CLI surface ships. |
| **Framework gem packaging** | One `toy` gem for now. Split as needed later. |
| **Pin Spinel** | **NO.** Early days; toy is a test-case for Spinel. Breakage is good signal. Always use whatever Spinel the user has. |
| **L5 ownership** | Tao. Toy's top layer is L4 (Recipe). |
| **Backward compatibility** | **NONE required.** Zero external users; Tep + Tao are the only integrations and we own them. Breaking changes land freely; Tep + Tao re-adapt in the same arc that breaks them. Aggressive cleanup over soft deprecation. |

---

_Draft v3. Open follow-ups now live in the roadmap doc, not here.
This doc is the contract; the roadmap is the execution plan._
