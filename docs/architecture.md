# Architecture

Toy is a Spinel-compiled toy LLM framework: a CRuby CLI that owns the
per-project plumbing, shelling Spinel-compiled compute runners. This
doc is the design contract — the *what* and *why* of the layering. For
the command surface see [`cli.md`](cli.md); for writing your own algos
see [`authoring.md`](authoring.md); for the live execution plan and
deferred work see [`roadmap.md`](roadmap.md).

## Operating philosophy

Conventions (Rails-style) remove decisions the user shouldn't be making,
but they compose explicit primitives, not magic. We are in explicit
territory: no metaprogramming, no "advanced magic". Code generation is
cheap; clarity is not. Every convention has an escape hatch, and every
abstraction is one file you can read top to bottom.

ML practitioners deal with a lot of Python ceremony that makes the
actual flow hard to see. Toy exists to expose the structure and the
algorithms — to avoid gate-keeping and ceremony.

**Non-goals / hard rules.**

- **No Spinel pin.** Toy is a test-case for Spinel; breakage is good
  signal. Always run against whatever Spinel the user has.
- **Zero backward-compat.** The only integrations are Tep and Tao, and
  we own both. Breaking changes land freely; Tep and Tao re-adapt in
  the same arc that breaks them. Aggressive cleanup over soft
  deprecation.
- **CLI is pure MRI.** The CLI files under `lib/toy/core/` are plain
  CRuby and must *never* require the Spinel-compiled libs. They shell
  the runners instead.

## Two orthogonal axes

Two distinctions that should not be conflated:

| Axis | What it specifies |
| --- | --- |
| **Card layer** | Granularity of the IR — how big a unit is. Primitive / Block / Arch / Engine / Recipe. |
| **Kind slot** | Role of the unit in a workflow — what it does. Arch / Trainer / Decoder / Eval / Server / DataSpec. |

A Recipe (Layer 5) might be "from-scratch trainer + Llama arch +
TinyStories dataset" — composing one of each Kind. An `RMSNorm`
(Layer 1) is also of a Kind (an Arch-side primitive). Layers describe
*size*; Kinds describe *role*.

The unifying contract is `Toy::Card`, the IR. Crucially, **Cards are
derived, not authored**: users write `realize`, and the framework
derives the Card by walking the runtime compute graph
(`lib/toy/dev/toy_describe_flow.rb`, leveraged by `lib/toy/dev/toy_card.rb`). No
hand-written step descriptions to drift from the code.

## The six Card layers (granularity)

A Card exists at every level; same IR type, layered composition. Each
unit is one file under `lib/toy/llm/`. Layer N composes Layer N-1.

| Layer | Unit | Stdlib files | What practitioners vary |
| --- | --- | --- | --- |
| **L1 Primitive** | A single named op | `primitives/{rope,swiglu,rms_norm,gqa}.rb` | Implementation of one op |
| **L2 Block** | One state-threading unit | `blocks/transformer_block.rb` | Which primitives, with what cfg |
| **L3 Arch** | Stack of blocks + embedding + head | `archs/llama_arch.rb` | Block count, per-layer overrides, embedding/head choice |
| **L4 Engine** | A realized session: graph + weights + step/decode driver | `engine/{llama_seq_engine,llama_kv_engine,gpt2_seq_engine,vit_tiny_engine}.rb` | Execution mode (seq-train vs KV-decode), weight residency, backend session |
| **L5 Recipe** | A training plan (one or more stages) | `recipes/{from_scratch,lora,warm_start,vit_tiny}.rb` | Stage sequence, optimizer, schedule, dataset progression |

The Engine is the layer the L1–L3 units realize *into*: it owns the FFI
session, allocates the persistent tensors, builds the forward (and, for
training, backward + AdamW) graph by calling down into Arch/Block/
Primitive code, and exposes the flat step/decode surface that Recipes
(and the serve/infer runners) drive. Engines are deliberately plain
classes with no taxonomy of their own — one engine per execution shape:

- `llama_seq_engine.rb` — seq-mode forward + training + LoRA (the
  retired monolith).
- `llama_kv_engine.rb` — KV-cache decode (was `lib/toy_smollm2_ffi_kv.rb`).
- `gpt2_seq_engine.rb` — GPT-2 from-scratch training (self-contained).
- `vit_tiny_engine.rb` — ViT-tiny training.

The exact tree (verified):

```
lib/toy/llm/
├── primitives/   rope.rb  swiglu.rb  rms_norm.rb  gqa.rb   (+ _cuda / _metal mirrors)
├── blocks/       transformer_block.rb                      (+ _cuda / _metal mirrors)
├── archs/        llama_arch.rb                             (+ _cuda / _metal mirrors)
├── engine/       llama_seq_engine.rb  llama_kv_engine.rb
│                 gpt2_seq_engine.rb  vit_tiny_engine.rb    (+ _cuda / _metal mirrors;
│                                                            vit_tiny_engine is CPU-only)
└── recipes/      from_scratch.rb  lora.rb  warm_start.rb   (+ _cuda / _metal mirrors)
                  vit_tiny.rb                               (CPU-only)
```

**Recipes include curriculum.** A Recipe is a training plan: one stage
(single dataset, single optimizer, one loop) or several (curriculum
learning — progressively harder data, phase transitions, schedule
changes). Each stage composes Arch + Trainer + DataSpec; state threads
between stages; the Recipe owns the sequence. One-stage Recipes are the
common case; multi-stage curricula are a strict superset.

> **Design intent, not yet shipped.** `recipes/curriculum.rb` is
> **absent** — the multi-stage Recipe shape is deferred. The
> from_scratch recipe currently introduces no separate `Trainer` /
> `DataSpec` / `Eval` classes: the optimizer lives behind the FFI, so a
> Ruby `Toy::Trainers::AdamW` wrapper would be empty. The full
> Trainer/DataSpec/Eval taxonomy is deferred to the multi-stage
> recipes. See [`roadmap.md`](roadmap.md).

**L6 Experiment** (varying a Recipe along an axis — sweeps, ablations,
LMC pairs) is **Tao's territory**, not Toy's. Tao reads the Recipe Card
to learn what knobs exist, then drives the sweep. Toy's top layer is L5.

## The Kind slots (role)

Each Kind has its own contract — what it receives, owns, emits. None
depend on the ggml backend directly; they use whatever session was
built (backend is a separate axis, below).

| Kind | Receives | Owns | Emits |
| --- | --- | --- | --- |
| **Arch** | cfg, sess, mode | forward graph, cache | (none) |
| **Trainer** | arch_cache, batch | opt step | `step`, `drift`, `grad`, `sample` events |
| **Decoder** | arch_cache, prompt | autoregressive loop, sampling | token IDs |
| **Eval** | arch_cache, corpus | metric computation | `eval` events |
| **Server** | state, transport | endpoint dispatch | events with `phase: "serve"` |
| **DataSpec** | (source) | tokenizer + corpus loader + sequence shape + stage marker | batches |

DataSpec is the unifying "this is the data" object, first-class to
remove the "where does the data come from" footnote from every other
kind.

## The Block contract

A Block is generalized enough to fit transformers, Mamba/SSM, and
future kinds:

```ruby
class Toy::Block
  input     :x,     shape: "[T, D]"               # main input
  state     :s_in,  shape: "[L_max, 2, T, D_h]"   # state from previous step
  output    :h,     shape: "[T, D]"               # main output
  state_out :s_out, shape: "[L_max, 2, T+1, D_h]"

  # Realize against a backend session; mode ∈ {:infer, :train}.
  # Returns a cache exposing (h_tensor, s_out_tensor).
  def realize(cfg, sess, mode); ...; end
end
```

**State is the abstraction** that lets a Block be a transformer block
(state = KV cache), a Mamba block (state = SSM hidden), diffusion
(state = timestep), and so on: `(input, state) → (output, state)`.

Shapes are explicit, not optional. They drive: compatibility checks
before realize (does this Arch fit this Block?), shape annotations in
the derived Card without runtime probing, and error messages that name
the expected vs. actual shape.

## Composition operators (the API surface)

Cards at any layer expose a small, machine-readable operator set, so
Tao (L6) can introspect what knobs a Recipe exposes and vary them
systematically across sweep arms.

| Operator | Effect |
| --- | --- |
| `card.with_hyper(k, v)` | Override a scalar hyperparameter |
| `card.per_layer(field, list)` | Vary a field across the layer axis (load-bearing — e.g. heads-per-layer `[16, 16, 8, 4, 2, 1]`) |
| `card.replace_step(name, step)` | Swap a named step (e.g. attention → ALiBi-attn) |
| `card.tap(name) { |t| ... }` | Runtime hook on a named intermediate |
| `card.make_trainable(param)` | Convert a constant to a trainable param (e.g. learnable `rope_base`) |
| `card.replace_primitive(name, impl)` | Swap a registered primitive without subclassing |

`per_layer` is the one that earns its keep: without it, "vary anything
across L" forces fork-and-edit.

## The Arch data model

An Arch (L3) is **data the graph builder reads**, not a monolithic
model class. The per-model architecture struct lives at `lib/toy/models/arch.rb`:
plain `attr_reader` fields, no inheritance, no metaprogramming. All
fields are required (no nil-defaults) so a new arch declares every
choice explicitly — that surfaces silent assumptions.

| Group | Fields |
| --- | --- |
| Identity | `family`, `name` |
| Dimensions | `vocab_size`, `d_model`, `n_layers`, `n_heads_q`, `n_heads_kv`, `d_head`, `d_ff`, `max_position`, `untied_lm_head` |
| Attention | `attention_kind` (`:mha`/`:gqa`/`:mqa`, via `gqa?`), `qkv_bias`, `qk_norm`, `swa_window` (nil = none, via `swa?`) |
| RoPE | `rope_freq_base`, `rope_freq_scale`, `rope_partial_factor` (1.0 default; 0.5 for GLM/Phi) |
| Norm | `norm_kind` (`:rms`/`:layer`), `norm_eps` |
| FFN | `ffn_kind` (`:swiglu`/`:geglu`/`:gelu_mlp`), `ffn_bias` |
| MoE | `moe`, `n_experts`, `n_experts_used`, `n_shared_experts`, `expert_gating` (`:softmax`/`:sigmoid`) |
| Tokenizer | `tokenizer_kind` (`:gguf_embedded`/`:external`), `bos_id`, `eos_id`, `pad_id`, `unk_id`, `add_bos_by_default` |
| Embed | `embed_scale` (1.0 default) |

**Detection.** `Arch.from_gguf(path)` reads GGUF kv metadata and
detects the family by **presence of tensors/keys** rather than trusting
`general.architecture` (the converter's value is unreliable). For
example, `qkv_bias` is detected from the presence of `blk.0.attn_q.bias`,
and `untied_lm_head` from the presence of `output.weight`. The detected
family selects how the rest of the fields are read. Unsupported
architectures fail loud.

`llama_arch.rb` is the L3 graph builder that consumes this struct: one
linear `build_forward` reading the Arch fields and emitting ops
(QKV projection with/without bias, optional QK-norm, RoPE with the
configured base/factor, KV-cache slot write, GQA-aware attention, SWA
mask when present, residual + norm, SwiGLU FFN or MoE dispatch, final
norm, tied/untied lm_head). No subclassing, no strategy pattern — a
reader who knows what a transformer is can follow it.

### Per-model worked examples

| Field | Qwen2.5-1.5B | Llama-3.2-3B | Mistral-7B-v0.2 |
| --- | --- | --- | --- |
| family | qwen2 | llama | llama |
| d_model | 1536 | 3072 | 4096 |
| n_layers | 28 | 28 | 32 |
| n_heads_q | 12 | 24 | 32 |
| n_heads_kv | 2 | 8 | 8 |
| qkv_bias | true | false | false |
| qk_norm | false | false | false |
| rope_freq_base | 1e6 | 5e5 | 1e6 |
| rope_partial_factor | 1.0 | 1.0 | 1.0 |
| swa_window | nil | nil | nil (v0.2 dropped SWA) |
| ffn_kind | :swiglu | :swiglu | :swiglu |
| moe | false | false | false |

## CLI + runner design

The CLI is CRuby (under `lib/toy/core/cli/`, one file per command).
The `COMMANDS` registry in `lib/toy/core/cli.rb` is the **single source
of truth** for `--help`, `--manifest`, and dispatch — change the
surface there and all three follow.

The CLI shells four Spinel-compiled compute runners; it never requires
the compiled libs itself:

```
lib/toy/run/infer.rb  → libexec/toy-infer
lib/toy/run/train.rb  → libexec/toy-train
lib/toy/run/eval.rb   → libexec/toy-eval
lib/toy/run/serve.rb  → libexec/toy-serve
```

The runners are built by `toy install` / `make`. They are **CPU-only**;
GPU runners (a `--device` flag) are deferred (see
[`roadmap.md`](roadmap.md)).

**CC-tools-friendly defaults.**

- `--json` everywhere — structured for tools, pretty for humans by
  default.
- `--manifest` emits a JSON manifest (`format: "toy/manifest-v1"`) of
  commands + args, so tools discover the surface without parsing
  `--help`.
- Exit codes distinguish bad input (**2**) from execution failure
  (**1**); 0 is success.

The full command roster and flags are in [`cli.md`](cli.md). Serving is
OpenAI-compatible and IDs-in/IDs-out (`lib/toy/serve/openai/`); chat
templating is deferred.

## Backends axis

Backend is orthogonal to both the layer and the kind axes. Three
backends:

- **CPU** — default, the only backend the CLI runners build today.
- **CUDA** — GB10, sm_121.
- **Metal** — Mac.

L1–L5 units ship CPU `.rb` plus `_cuda` / `_metal` mirrors (exceptions:
`vit_tiny_engine.rb` and `recipes/vit_tiny.rb` are CPU-only today). The
mirrors are **generated**, not hand-maintained: `prep/gen_cuda_mirror.rb`
rewrites the CPU source (driven by `MIRRORABLE` markers in that script).
The gate `make verify-mirrors` runs the generator with `--verify` and
exits non-zero if any committed mirror has drifted. This per-class
mirroring is a Spinel-polymorphism workaround, not a fundamental
constraint — if Spinel ships polymorphism the mirrors retire and the
Card stays valid regardless.

For the vendored ggml patches and the CUDA BYO-pointer path, see
[`reference/backends.md`](reference/backends.md).

## Where it's going

L6 (experiments / sweeps) belongs to Tao. A `toy g` generator surface
and a Prism-based static Card lowerer are future/optional — there is
currently **no `toy g` and no `lib/toy/core/cli/curriculum.rb`**; those
remain design intent. The deferred list and live research notes live in
[`roadmap.md`](roadmap.md).
