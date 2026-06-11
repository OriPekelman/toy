# examples/legacy/

Demos superseded by the `toy` CLI or by the curated set (toy#60; see
[`../README.md`](../README.md)). Kept for reference and history — they
still build (their `make` targets point here), but they're no longer
part of the teaching set. Prefer the CLI or the numbered examples one
level up.

| Legacy file | Use instead |
|---|---|
| `01_inference_metal.rb` | `toy infer <model.gguf> --device metal` |
| `02_train_custom_gpt.rb` | the pure-Ruby teaching GPT (no FFI engine); still unique, but pre-recipe — read `../01_train_tiny.rb` first (`make example_train` still builds this) |
| `03_finetune_lora.rb` / `_cuda.rb` | `toy train lora`, or `../03_lora.rb` for the library read (`make example_finetune` still works) |
| `06_train_from_scratch.rb` | the full-knobs instrumented trainer (events, checkpoints, drift sentinels, CKA taps, WEIGHT_DTYPE). Still the instrumentation reference and still a GATE SUBJECT — `prep/mixed_precision_gate.rb` builds and drives it (`make example_train_from_scratch`); its `_cuda`/`_metal` mirrors are still generated (it stays in `MIRRORABLE`). |
| `07_train_vit_tiny.rb` | `../07_vit_tiny.rb` (recipe-based; this one drives the engine directly with the timm AugReg warm-start knobs) |
| `08_lmc.rb` | `toy eval lmc --ckpt A --other B` |
| `09_warm_start_train.rb` | `toy train warm-start`, or `../02_finetune_warm_start.rb` for the library read |
| `gpt2_train.rb` | `toy train from-scratch --arch gpt2` (`make gpt2_train` still builds this) |
| `train_from_scratch.rb` | `../01_train_tiny.rb` (same recipe, on the Phase-A value objects; this was the pre-#60 blessed tutorial — `make example_train_from_scratch_blessed` still works) |

These were the right shape before the CLI / the recipe value objects
existed. The byte-exact behaviour is preserved by the gates
(`prep/*_gate.rb`, `prep/smokes/smoke_*`), which are the test suite and
live under `prep/`, not here.
