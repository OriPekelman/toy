# Toy as a framework: design doc (DRAFT v2)

**Status:** design doc for review. No implementation yet.
**Date:** 2026-05-28 (rev 2 after first round of in-place review).
**Companion docs:**
[`lowerer-design.md`](lowerer-design.md),
[`spinelgems-tep-adoption-2026-05-27.md`](spinelgems-tep-adoption-2026-05-27.md),
[`backends-and-scale-2026-05-27.md`](backends-and-scale-2026-05-27.md).

## TL;DR

Promote toy from "library you `require_relative` from your own
Spinel-compiled experiment" to **"framework you create projects
inside of"** — `toy new myproj`, `toy train ...`, `toy serve ...`,
with a CLI that owns the per-project plumbing.

The lever: **`Toy::Card` as the IR contract**, organized in five
granularity layers (Primitive / Block / Arch / Recipe / Experiment)
and filled by five kind slots (Arch / Trainer / Decoder / Eval /
Server, + a likely DataSpec). The Card describes the work; the
framework composes, introspects, and runs.

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
| **Card LAYER** (§5) | Granularity of the IR — how big a unit is. Primitive / Block / Arch / Recipe / Experiment. |
| **Kind SLOT** (§6) | Role of the unit in a workflow — what it does. Arch / Trainer / Decoder / Eval / Server / DataSpec. |

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

## 4. Distribution via `rv`

Spinel-Coop's [`rv`](https://github.com/spinel-coop/rv) is the
`uv`/`pipx` equivalent for Ruby.

| Command | Use case |
| --- | --- |
| `rvx toy install` | Ephemeral one-shot |
| `rv tool install toy` | Persistent install (recommended) |
| `rv clean-install` (in project) | Project-pinned Ruby + gems |

`rv` adoption is documented as the recommended install path, not
required. `gem install toy` stays as a fallback. The CLI doesn't
shell out to `rv` from its own code.

## 5. The five Card layers (granularity)

A Card exists at every level. Same IR type; layered composition.

| Layer | Unit | Examples (stdlib) | What practitioners vary |
| --- | --- | --- | --- |
| **L1 Primitive** | A single named op | `LayerNorm`, `RMSNorm`, `RoPE`, `Softmax`, `MHA`, `GQA`, `SwiGLU`, `PatchEmbed` | Implementation of one op |
| **L2 Block** | One state-threading unit | `TransformerBlock`, `SSMBlock` (Mamba) | Which primitives, with what cfg |
| **L3 Arch** | Stack of blocks + embedding + head | `LlamaArch`, `GPT2Arch`, `ViTArch`, `MambaArch` | Block count, per-layer overrides, embedding/head choice |
| **L4 Recipe** | Arch + Trainer + DataSpec + Decoder + Eval | `FromScratchRecipe`, `LoRARecipe`, `WarmStartRecipe` | Optimizer choice, schedule, dataset |
| **L5 Experiment** | Vary a Recipe along an axis | Granite-style sweeps, ablations, LMC | The axis itself |

Every layer is a file you can read and copy. Layer N composes
Layer N-1.

## 6. The five Kind slots (role)

Each Kind has its own contract — what it receives, owns, emits.

| Kind | Receives | Owns | Emits |
| --- | --- | --- | --- |
| **Arch** | cfg, sess, mode | forward graph, cache | (none) |
| **Trainer** | arch_cache, batch | opt step | `step`, `drift`, `grad`, `sample` events |
| **Decoder** | arch_cache, prompt | autoregressive loop, sampling | token IDs |
| **Eval** | arch_cache, corpus | metric computation | `eval` events |
| **Server** | state, transport | endpoint dispatch | server-side events with `phase: "serve"` |

**Sixth kind (PROPOSED): DataSpec.** A unified "this is the
data" object (tokenizer + corpus loader + sequence shape). Today
split between Ruby loaders and Python pretokenizers. A first-class
Kind for it removes the "where does the data come from" footnote
from every other kind.

None of these depend on toy-ggml directly. They use whatever
session was built with — be it ggml, future pure-Ruby, or
something else. Backend is a separate axis (§9).

## 7. The Block contract

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

## 8. Composition operators (the API surface)

Cards (at any layer) expose:

| Operator | Effect |
| --- | --- |
| `card.with_hyper(k, v)` | Override a scalar hyperparameter |
| `card.per_layer(field, list)` | Vary a field across the layer axis (the killer one — supports heads-per-layer `[16,16,8,4,2,1]`) |
| `card.replace_step(name, new_step)` | Swap a named step (e.g., replace attention with ALiBi-attn) |
| `card.tap(name) { |tensor| ... }` | Runtime hook on a named intermediate |
| `card.make_trainable(param)` | Convert a constant to a trainable param (e.g., learnable `rope_base`) |
| `card.replace_primitive(name, impl)` | Swap a registered primitive without subclassing |

These compose Cards declaratively. Researchers write recipes like:

```ruby
recipe = Toy::Recipes::FromScratch
  .arch(:llama)
  .with_hyper(:d_model, 512)
  .per_layer(:n_heads, [16, 16, 8, 4, 2, 1])     # pyramid-down
  .replace_primitive(:attention, :mha_with_alibi)
  .make_trainable(:rope_base)
```

`per_layer` is first-class. Without it, "vary anything across L"
forces fork-and-edit. With it, heterogeneous archs are a one-liner.

## 9. Cards are derived, not authored

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

## 10. DRY: require, don't copy

`toy g arch my-llama --based-on llama` does NOT copy stdlib's
primitives into the user's project. It scaffolds a thin
`algos/my_llama.rb` that *requires* the primitives from the
framework and composes them:

```ruby
# algos/my_llama.rb — scaffolded by toy g arch my-llama --based-on llama
require "toy/llm/primitives"   # RMSNorm, RoPE, GQA, SwiGLU, ...
require "toy/llm/blocks"        # TransformerBlock

class MyLlama < Toy::Arch
  arch_name "my-llama"
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

## 11. Multi-backend extensibility

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

## 12. CC-tools-friendly defaults

- `toy <cmd> --json` everywhere — structured for CC, pretty for
  humans by default.
- `toy --manifest` — JSON manifest of commands + args. CC
  discovers the surface without parsing `--help`.
- Exit codes distinguish "your input was wrong" (2) from "I failed
  to do the thing" (1).
- Stable error JSON shape.

Slash commands / skills are wrappers around the CLI, not separate
surfaces.

## 13. The Tep-generator non-goal

Tep would also benefit from a generator. Resist extracting
"spinelgen" base library in v1. Web-shape and ML-shape projects
diverge sharply. Common base would be ~200 LOC of orthogonal
plumbing (argv parsing, dispatch, JSON output, manifest). Build
toy first; if after shipping the plumbing is genuinely identical
across Toy and Tep, extract then.

Don't design *for* extraction. Don't design *against* it either.

## 14. Decisions locked (after review round 1)

- ✅ **Card layers** — five layers (Primitive / Block / Arch /
  Recipe / Experiment).
- ✅ **Kind slots** — five plus DataSpec (proposed sixth).
- ✅ **Block contract** — `(input, state) → (output, state)` with
  explicit shape declarations.
- ✅ **Composition operators** — `with_hyper`, `per_layer`,
  `replace_step`, `tap`, `make_trainable`, `replace_primitive`.
- ✅ **Cards are derived, not authored** — `def card` is optional
  override only.
- ✅ **DRY for primitives** — `toy g` scaffolds require the
  framework's primitives; user vendors specific ones to override.
  No bulk copy.
- ✅ **rv as recommended install** — `gem install` as fallback;
  CLI doesn't shell out to `rv`.

## 15. Remaining open questions

- **Q1.** Tao coordination: `runs/<id>/` layout. Lock before
  `toy new` ships. Should we file a Tao-side issue?
- **Q2.** `toy.yml` minimum contents — backend choice, data dir,
  default run-id template, default algo-discovery path?
- **Q3.** Naming + discovery: is `arch_name "llama"` enough, or
  do algo classes need a richer manifest (version, deps, license)?
- **Q4.** First concrete deliverable: a `toy new` + MVP commands
  (install / list / fetch / describe) that proves the on-ramp, OR
  a Card-derivation refactor that proves §9 against `Toy::SmolLM2`
  + `LlamaArch` before any user-facing surface ships? Lean
  strongly toward the second — see §17.
- **Q5.** Framework gem packaging: one `toy` gem (CLI + Core +
  Stdlib facade) or split (`toy-framework` CLI/Core, `toy` the
  existing lib-vendoring gem, `toy-llm` etc. as separate gems)?
- **Q6.** Pin Spinel? Framework gem vendors a known-good revision,
  or always uses whatever the user has? Lean toward pin
  (reproducibility matters; landmines retire on a schedule).
- **Q7.** Should L5 (Experiment) be a toy concept or a Tao one?
  Tao already runs sweeps / comparisons. If L5 is Tao's, toy's
  top layer is L4 (Recipe). Probably yes — defer L5 to Tao.

## 16. Sequencing

**Phase 0** — design lock (this doc + Tao coord on Q1 + Q4
answered).

**Phase 1** — derive Cards from realize. Implement the runtime
probe path in Core; refactor `Toy::SmolLM2` and the seq-mode
training graph to produce Cards via derivation rather than the
hand-written `algorithm` methods. **Critically inside toy's repo**,
not yet in user-facing land. Proves §9 holds.

**Phase 2** — refactor stdlib archs into the five-layer hierarchy
(L1 primitives, L2 blocks, L3 archs). Llama-family first (most
exercise); ViT-Tiny second (proves generality). Recipes (L4)
follow naturally.

**Phase 3** — Core + CLI MVP. `bin/toy` as CRuby. Five commands:
`new`, `install`, `list`, `fetch`, `describe`. No `train` /
`serve` yet — focus on "can a new user get from clone to identified
model".

**Phase 4** — `toy infer`, `toy serve`, `toy train`, `toy eval`.
Each composes the algo classes from phase 2.

**Phase 5** — Generators (`toy g arch`, `toy g recipe`). After
patterns settle.

**Phase 6** — Prism lowerer (the static derivation path). When
the runtime path's limits start hurting.

Phases 1–4 don't paint into corners. Phase 5 is where opinions
start to bake; lock the contract first.

---

## Appendix — what the CLI surface *might* look like (sketch)

Not locked. Illustrative.

```sh
# Install toy (rv path recommended)
brew install rv && rv tool install toy
# or: gem install toy

# Bootstrap a project (outside any toy repo)
toy new myproj && cd myproj

# Install the right backend for this host
toy install                       # auto-detect; or --backend cpu|cuda|metal

# Discover + fetch models
toy list
toy fetch bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf

# Describe a model (Card + GGUF metadata)
toy describe SmolLM2-135M-Instruct-Q8_0

# Run inference (CLI resolves model name through `toy list`)
toy infer SmolLM2-135M-Instruct-Q8_0 --prompt "Once upon" --max-tokens 32

# Serve
toy serve SmolLM2-135M-Instruct-Q8_0 --port 4567

# Scaffold a new arch (DRY — requires framework primitives, doesn't copy)
toy g arch my-llama --based-on llama

# Train using a Recipe (composes Arch + Trainer + DataSpec)
toy train from-scratch --arch my-llama --data tinystories \
  --d-model 512 --heads-per-layer "[16,16,8,4,2,1]"   # per_layer in action

# Evaluate
toy eval lmc --ckpt runs/abc/weights/latest --other runs/def/weights/latest

# Inspect events
toy events runs/abc                  # tail
toy events runs/abc --kind sample    # filter

# Discover the surface (for CC etc.)
toy --manifest                       # JSON of commands + args
toy <any-cmd> --json                 # JSON output instead of pretty
```

---

_Draft v2. Amend in place; mark resolved questions with
`[ANSWERED]`, contested ones with `[DEBATE]`, add new ones
as they surface._
