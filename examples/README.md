# examples/ — the narrated teaching set

Eight single-file examples, curated (toy#60) onto the current API: the
one-require compute surface, the L5 recipes, and the named value
objects (`SmolLM2Config` / `RecipeOptions` / `TrainingBatch` / `AdamW`
— the tour is [`docs/framework.md`](../docs/framework.md)). Each file
opens with a narrated header: **what you'll see, how long it takes,
what to tweak**. Each compiles to one native binary via one `make`
target (06 is plain CRuby — no build; 08 drops to the FFI graph since the
Dragon/GDN block has no recipe yet).

Most day-to-day tasks are the `toy` CLI (`toy train|infer|eval|serve`,
see [`docs/cli.md`](../docs/cli.md)); these examples are the
**library-API reads** — what the CLI does under the hood, in one page
of Ruby each.

## The set

| # | File | What you'll see | Runtime | Needs |
|---|---|---|---|---|
| 01 | [`01_train_tiny.rb`](01_train_tiny.rb) | **Start here.** A tiny Llama trained from scratch on the bundled corpus through `Recipes::FromScratch` + the value-object quartet; writes a `runs/` bundle. | ~2 s | nothing (bundled corpus) |
| 02 | [`02_finetune_warm_start.rb`](02_finetune_warm_start.rb) | Warm-start: donor embeddings from a real GGUF uploaded into the training graph (`realize_scratch! → realize_warm! → build!`); `INIT=scratch` to diff what the donor buys. | ~3 s | any llama-family GGUF |
| 03 | [`03_lora.rb`](03_lora.rb) | LoRA/QLoRA: rank-8 adapters over a frozen mmap'd base; CE 9.2 → 3.6 in 20 steps while the base stays untouched. | ~10 s | a *native-layout* GGUF |
| 04 | [`04_generate.rb`](04_generate.rb) | Load a GGUF, KV-cache decode, print text (or raw ids for tokenizer-less models). | ~5 s | any llama-family GGUF |
| 05 | [`05_eval_logprobs.rb`](05_eval_logprobs.rb) | Per-token top-K log-probabilities at a decode position — the perplexity/calibration building block. | ~3 s | any llama-family GGUF |
| 06 | [`06_runlog_compare.rb`](06_runlog_compare.rb) | **CRuby, no build**: `Toy::RunLog.scan` over `runs/` → a comparison table, best final loss first. | instant | runs from 01 / `toy train` |
| 07 | [`07_vit_tiny.rb`](07_vit_tiny.rb) | The same recipe contract for IMAGES: ViT-Tiny memorizes the bundled smoke image (CE 2.30 → ~0.001). | ~20 s | nothing (committed corpus) |
| 08 | [`08_gdn_block.rb`](08_gdn_block.rb) | **v0.9.0 / Dragon.** A from-scratch model whose mixer is a trainable Gated-DeltaNet block (not attention); CE 2.99 → 1.94 in 14 steps. The one example that builds the train graph by hand (GDN has no L5 recipe yet). | <1 s | nothing (bundled) |

## Build + run

```sh
make setup           # one-time: ggml backend (auto-detects CUDA/Metal/CPU)
make example_01      # builds examples/example_01_train_tiny
./examples/example_01_train_tiny
LR=0.01 RUN_ID=lr-01 ./examples/example_01_train_tiny   # no recompile
ruby examples/06_runlog_compare.rb                      # compare the two
```

Every knob is ENV (compile once, run many); each file's header lists
its own. `make examples-curated` builds all the compiled ones.

## Models for 02–05

02–05 want a GGUF. `toy list` shows what's already cached (HF /
Ollama / LM Studio caches are scanned); otherwise:

```sh
toy fetch bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf
```

For 03 (LoRA) the base must be *native layout* (mmap'd in place):
convert with `./prep/convert_smollm2_to_gguf.py --ggml-native`.

## Where the rest went

- **Gates** (`smoke_*.rb`) are not examples — they live in
  [`prep/smokes/`](../prep/README.md) and are indexed by
  [`docs/gating.md`](../docs/gating.md).
- **The full interleaved attention+GDN hybrid** (v0.9.0 / Dragon) has no
  numbered example — example 08 trains the GDN block on its own; the full
  hybrid stack drives the engine directly in its own runner: `make
  gate-gdn-hybrid` (`libexec/toy-train-hybrid`). The design and the
  `LlamaArch#set_gdn_layer!` seam are written up in
  [`../docs/roadmap/dragon-gdn-arch-2026-06-20.md`](../docs/roadmap/dragon-gdn-arch-2026-06-20.md).
- **Superseded tutorials** live in [`legacy/`](legacy/README.md) — the
  pure-Ruby teaching GPT, the full-knobs instrumented trainer
  (events/checkpoints/CKA — still built by the mixed-precision gate),
  the GPT-2 engine demo, and the CLI-covered demos (lmc, metal
  inference, …). They still build.
- **HTTP serving** is `toy serve` (lib/toy/serve/) and the Tep demos in
  [`tep_demo/`](../tep_demo/README.md).
- **Exhaustive per-model drivers** are under `demos/`.

## After the examples

- [`../docs/framework.md`](../docs/framework.md) — toy as a library:
  the five layers, recipes, starting your own project (`toy new`).
- [`../docs/authoring.md`](../docs/authoring.md) — add your own
  primitive/block/arch/recipe, gate-before-merge.
- [`../docs/consuming-toy.md`](../docs/consuming-toy.md) — toy as a gem
  dependency with native vendoring.
