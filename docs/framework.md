# Using toy as a framework

toy is a CLI you can play with — but it is, first, a **library of named,
readable ML algorithms** you build on. This page is the tour of that
surface: the layered stack, the one-require entry point, recipes, and
the two ways to start a project of your own.

## The stack, as your API

Everything model-shaped lives in four layers. Each layer only calls the
one below it, every file is plain Ruby, and every layer is gated
bit-identical against a reference — so when you swap a piece, the gates
tell you exactly what changed.

| Layer | What lives there | Example |
|---|---|---|
| **L1 primitives** | the math: matmul, softmax, layer_norm, RoPE | `Toy.softmax(scores)` |
| **L2 blocks** | named compositions: attention, FFN, the transformer block | `transformer_block(x, block)` |
| **L3 archs** | whole models with an `algorithm_card`: GPT-2, SmolLM2/Llama, ViT | `Toy::SmolLM2.forward(ids)` |
| **L4 recipes** | training plans: from-scratch, warm-start, LoRA, ViT | `Recipes::FromScratch` |

The promise across all four: **the code reads like the paper**. Shapes
are annotated, blocks are named after the math, and each L3 arch emits
Phuong–Hutter-style pseudocode (`toy describe <model>`) that round-trips
back to the constructing Ruby (`docs/authoring.md` § Cards).

## One require, the whole compute surface

```ruby
require "toy/compute"
```

That single require loads the full compute API: the TinyNN FFI bridge,
`Toy::Mat`, the L1–L3 stack, the engines
(`Toy::LLM::Engine::LlamaSeqEngine`, `Gpt2SeqEngine`, `VitTinyEngine`),
the recipes, the GGUF loader and tokenizer, AdamW and label helpers.
It is the entry point for **library consumers** — anything that isn't
the toy CLI itself.

## A training run in one screen

The recipe surface is deliberately flat: `realize!` builds the whole
forward + loss + backward + optimizer graph natively (AdamW is baked
into the ggml graph, not a Ruby loop), and `step!` drives one training
step. You own the config and the per-step data; the recipe owns the
graph.

```ruby
require "toy/compute"

recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, context_len, 1,    # your Toy::LLM config + shapes
                0, false, false,        # weight dtype, untied, qkv_bias
                0, 1.0)                 # seed, init scale

steps.times do |step|
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  puts "step #{step}: loss=#{loss}"
end
```

That is the *entire* contract — the same one toy's own gates train
through (`examples/smoke_compute_surface.rb` is the living version).
`WarmStart` resumes from a GGUF instead of random init; `LoRA` trains
adapters over frozen base weights; `VitTiny` is the same shape for
images. Writing your own recipe is ~100 lines of plain Ruby over an
engine — `docs/authoring.md` § L4 walks through it.

## Starting a project

Two scaffolds, depending on what you're building:

```sh
toy new mylab           # an experiment tree: data/, runs/, examples wiring
toy new mylib --lib     # a Ruby library that consumes toy as a gem
```

The `--lib` scaffold gives you a Gemfile (`gem "toy"`), the vendor
wiring for Spinel compilation, and a compilable hello-train. From
there the consumer story is ordinary Ruby: `bundle lock`, vendor, and
your code `require "toy/compute"`s its way to the whole surface.
The full consumer reference — including self-contained native
vendoring (ggml builds inside *your* tree, relocatable, no absolute
paths) and the CUDA/Metal opt-ins — is
[`consuming-toy.md`](consuming-toy.md).

## Where to go deeper

- [`authoring.md`](authoring.md) — add a primitive, block, arch, or
  recipe; the card round-trip; gate-before-merge.
- [`architecture.md`](architecture.md) — how the plain-Ruby CLI drives
  Spinel-compiled native runners; the five-layer map.
- [`consuming-toy.md`](consuming-toy.md) — toy as a dependency of your
  gem or app.
- [`gating.md`](gating.md) — what "gated bit-identical" means
  mechanically.
