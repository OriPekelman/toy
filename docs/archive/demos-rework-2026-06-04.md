# Demos / examples rework — plan (2026-06-04)

Release-prep (NOT releasing yet). Goal: **fewer, current** demos that leverage
the post-refactor form (the `toy` CLI + `lib/toy/` engine + L1–L4), used as a
**design review** — if a port comes out clean, the refactor earned its keep; if
it's ugly, revisit the design.

## The load-bearing distinction (don't break this)

`examples/` holds TWO different things, both Makefile-built:

1. **Gates / wire-smokes (`smoke_*.rb`)** — the byte-identical TEST suite, wired
   into `docs/gating.md` + `make gate-*` / `make verify` / `verify-mirrors`.
   **These are NOT demos. They STAY.** Moving them breaks CI.
2. **Tutorials (numbered + `train_from_scratch.rb`) + `tep_demo/`** — the
   user-facing pedagogy. **These are the rework target.**

## The design-review finding

The `toy` CLI now covers the headline tasks the numbered tutorials predate:
`toy infer` / `toy train from-scratch|lora|warm-start [--device cuda]` /
`toy eval [lmc]` / `toy serve` / `toy list`. So **the CLI is the primary demo**;
example *scripts* are only worth keeping when they show something the CLI can't:
library-API usage, instrumentation, or an arch with no CLI surface.

## Inventory + disposition (tutorials only; gates untouched)

| Demo | Now covered by | Disposition |
|---|---|---|
| `train_from_scratch.rb` (BLESSED) | — (library-API tutorial) | **KEEP** — the canonical "use toy as a library" read (L4 `FromScratch` + `SmolLM2Config.mha` + `Toy::Labels` + `Toy::AdamW`). The exemplar of the current form. |
| `06_train_from_scratch.rb` (+cuda/metal) | — (instrumentation ref) | **KEEP** — the events/checkpoints/drift/CKA/Tao-harness full-knobs reference; distinct purpose from the CLI. |
| `07_train_vit_tiny.rb` | — (no ViT CLI) | **KEEP** — only path to ViT until a CLI surface exists. |
| `02_train_custom_gpt.rb` | — (pure-Ruby teaching GPT) | **PORT → GPT-2** — refresh as the showcase for the new GPT-2 training arch (the `gpt2-train` work); replace the pure-Ruby `lib/transformer.rb` path with the engine GPT-2 once `toy train --arch gpt2` lands. |
| `01_inference_metal.rb` | `toy infer --device metal` | **RETIRE** — CLI covers it. |
| `03_finetune_lora.rb` (+cuda) | `toy train lora` | **RETIRE or fold** into one library-API example (LoRA is `recipe = Lora.new; realize!/step!`). |
| `08_lmc.rb` | `toy eval lmc` | **RETIRE** — CLI covers it. |
| `09_warm_start_train.rb` | `toy train warm-start` | **RETIRE or fold** into the library-API example. |
| `tep_demo/openai_api_llama` | `toy serve` | already moved (README says so) — **drop the leftover**. |
| `tep_demo/{hello,inference,openai}_api.rb` | — (Tep+Spinel demos) | **KEEP a single** minimal Tep serving demo (`hello_api`); the GPT-2 `openai_api` rides on the #30 serve convergence (currently blocked). |

**Target: ~4 example scripts** — `train_from_scratch` (library API),
`06_train_from_scratch` (instrumentation), `07_train_vit_tiny` (ViT), and a new
`gpt2_train` (the new arch) — plus the CLI as the primary surface, plus the
`smoke_*` gate suite (unchanged). Down from ~10 tutorials.

## Safe execution sequence (each step keeps `make` + gates green)

1. **Stage, don't delete.** `git mv` retirees to `examples/legacy/` (keeps history,
   stays out of the curated set). Update the README "What's here" table to the
   curated set; move the rest under a "Legacy (superseded by the CLI)" note.
2. **Drop their Makefile targets** AND any `.PHONY`/umbrella aggregations that
   reference them; re-run `make` + `make verify` to confirm nothing else depended
   on them. (`verify-mirrors` only touches the `_cuda`/`_metal` twins — check the
   retired ones aren't in the mirror set.)
3. **Re-home GPT-2.** Promote `prep/gpt2_train_min.rb` into a real
   `examples/gpt2_train.rb` once `toy train --arch gpt2` (engine integration) lands,
   so the demo uses the public surface, not the inline prep proof.
4. **Reframe the README** around the CLI as the primary path; the curated scripts
   are "library API / things the CLI doesn't cover yet."
5. **Note design friction.** Anything that ports ugly → a roadmap item to revisit
   the API, not a workaround in the demo.

## Sequencing

After the GPT-2 arch tail (so the GPT-2 demo uses `toy train --arch gpt2`, not the
inline prep proof). Until then this plan is the spec; the destructive moves (step
1–2) are a deliberate, gates-green pass — not to be rushed at the tail of a long
session.
