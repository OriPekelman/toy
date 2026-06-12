# Using toy as a framework

toy is a CLI you can play with — but it is, first, a **library of named,
readable ML algorithms** you build on. This page is the tour of that
surface: the layered stack, the one-require entry point, recipes, and
the two ways to start a project of your own.

## The stack, as your API

Everything model-shaped lives in five layers. Each layer only calls the
ones below it, every file is plain Ruby, and every layer is gated
bit-identical against a reference — so when you swap a piece, the gates
tell you exactly what changed.

| Layer | What lives there | Example |
|---|---|---|
| **L1 primitives** | the math: matmul, softmax, layer_norm, RoPE | `Toy.softmax(scores)` |
| **L2 blocks** | named compositions: attention, FFN, the transformer block | `transformer_block(x, block)` |
| **L3 archs** | whole models with an `algorithm_card`: GPT-2, SmolLM2/Llama, ViT | `Toy::SmolLM2.forward(ids)` |
| **L4 engines** | realized graphs: forward + training + KV decode on a backend | `Engine::LlamaSeqEngine` |
| **L5 recipes** | training plans over an engine: from-scratch, warm-start, LoRA | `Recipes::FromScratch` |

(L6 — varying a recipe across sweeps and ablations — is experiment
territory: yours, or [Tao](https://github.com/OriPekelman/tao)'s.)

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

For **MRI dev-runs** (notebook cells, REPL pokes — no Spinel build),
`require "toy/mri"` loads the same surface under plain CRuby: pure-Ruby
paths (configs, Mat, `TransformerLM` + `Toy::Trainer`) work for real;
native calls raise a named `Toy::MRI::NativeCallError` instead of dying
in a `NoMethodError`. See `consuming-toy.md` § MRI dev-runs (toy#71).

## A training run in one screen

The recipe surface is deliberately flat: `realize!` builds the whole
forward + loss + backward + optimizer graph natively (AdamW is baked
into the ggml graph, not a Ruby loop), and `step!` drives one training
step. Knobs are named setters, not positional walls; per-step data
rides a validated `TrainingBatch`; you own the config and the data, the
recipe owns the graph.

```ruby
require "toy/compute"

cfg  = Toy::SmolLM2Config.tiny          # or .mid, or build your own shape
opts = Toy::LLM::RecipeOptions.new
opts.t_seq = 32                         # context window (required)
opts.seed  = 42                         # everything else has sane defaults

adamw = Toy::AdamW.for_from_scratch
batch = Toy::LLM::TrainingBatch.new(cfg.vocab, opts.t_seq, 1)

recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, opts)

steps.times do |step|
  batch.fill!(next_seq_ids)             # validates shapes, rebuilds labels
  batch.hp = adamw.hp(step)
  loss = recipe.step!(batch.seq_ids, batch.positions, batch.labels,
                      batch.hp, step == 0)
  puts "step #{step}: loss=#{loss}"
end
```

That is the *entire* contract — the same one toy's own gates train
through (`prep/smokes/smoke_compute_surface.rb` is the gate fixture;
`examples/01_train_tiny.rb` is the narrated, runnable version).
`WarmStart` resumes from a GGUF instead of random init; `LoRA` trains
adapters over frozen base weights (`recipe.realize!(gguf, cfg, rank,
opts)`); `VitTiny` is the same shape for images. Writing your own
recipe is ~100 lines of plain Ruby over an engine —
`docs/authoring.md` § L4 walks through it.

Afterwards, every run is queryable from plain Ruby:

```ruby
best = Toy::RunLog.scan("runs").first   # sorted by final loss
puts best.config, best.final_loss, best.loss_curve.first(5)
```

## Starting a project

Two scaffolds, depending on what you're building:

```sh
toy new mylab           # an experiment tree: data/, runs/, examples wiring
toy new mylib --lib     # a Ruby library that consumes toy as a gem
```

The experiment scaffold's hello recipe reads its hyperparameters from
ENV — one compile, many runs:

```sh
spinel algos/recipes/hello.rb -o hello
D_MODEL=128 N_LAYERS=4 STEPS=10 ./hello
D_MODEL=256 STEPS=10 ./hello            # no recompile
```

The `--lib` scaffold gives you a Gemfile (`gem "toy"`), the vendor
wiring for Spinel compilation, a device-agnostic `experiment.rb`, and a
multi-arch build script. **Devices are chosen at compile time**: your
source requires the device-neutral surface, the per-device entry shims
(`main_cpu.rb` / `main_cuda.rb` / `main_metal.rb` → `toy/compute`,
`toy/compute_cuda`, `toy/compute_metal`) pick the backend, and

```sh
./build.sh cpu cuda     # → ./experiment_cpu, ./experiment_cuda
```

builds one native binary per device from the same experiment source.
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
