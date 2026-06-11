# examples/legacy/

Demos superseded by the `toy` CLI (or by a newer example). Kept for reference
and history — they still build (their `make` targets point here), but they're
no longer part of the curated example set. Prefer the CLI.

| Legacy file | Use instead |
|---|---|
| `01_inference_metal.rb` | `toy infer <model.gguf> --device metal` |
| `03_finetune_lora.rb` / `_cuda.rb` | `toy train lora` (or `examples/03_finetune_lora` → `make example_finetune` still works) |
| `08_lmc.rb` | `toy eval lmc --ckpt A --other B` |
| `09_warm_start_train.rb` | `toy train warm-start` |

These were the right shape before the CLI existed; the CLI now drives the same
compute runners with a controlled environment. The byte-exact behaviour is
preserved by the gates (`prep/*_gate.rb`, `prep/smokes/smoke_*`), which are the
test suite and live at `examples/`, not here.
