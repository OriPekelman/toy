# toy

<p align="center">
  <img src="toy_logo.png" alt="toy" width="240" />
</p>

**v0.7.0-pre-alpha** · not API-stable
&nbsp;·&nbsp; [CHANGELOG](CHANGELOG.md)
&nbsp;·&nbsp; [docs](docs/architecture.md)
&nbsp;·&nbsp; [framework guide](docs/framework.md)

**Readable machine learning.** toy is a transformer LM framework in
Ruby, [Spinel](https://github.com/matz/spinel)-compiled to native
binaries. The whole forward pass fits on one screen, every shape is
annotated, and the building blocks are named after the math — and it
still runs real HuggingFace models (SmolLM2, Llama 3, Qwen 2.5/3,
Mistral, Gemma 2, OLMoE) with output matched against PyTorch, on CPU,
CUDA, and Metal.

```ruby
# One transformer block (GPT-2 family — Llama swaps LN→RMSNorm and
# adds RoPE inside self_attention; same one-screen shape).
def transformer_block(x, block)
  h  = layer_norm(x, block.ln1_gamma, block.ln1_beta)
  x.add!(self_attention(h, block))
  h2 = layer_norm(x, block.ln2_gamma, block.ln2_beta)
  x.add!(feed_forward(h2, block))
  x
end
```

This is not pseudocode next to the real implementation — it **is** the
implementation. Every model also carries an `algorithm_card` emitting
Phuong–Hutter-style pseudocode (arXiv:2207.09238) with shape
annotations, and the round-trip closes: `toy describe <model>` renders
the card from a GGUF's metadata, and the card parses back into the Ruby
that constructs the model.

Inference (KV-cache decode, F32/Q8, zero-copy mmap), training
(from-scratch, warm-start, LoRA), eval (per-token logprobs), and an
OpenAI-compatible server — each an end-to-end single native binary.

## Five minutes of play

Requires Ruby, [Spinel](https://github.com/matz/spinel), and a C
compiler. The CLI is plain MRI Ruby; the native compute runners are
built on demand.

```sh
toy install                                  # build/verify the CPU backend
toy fetch ggml-org/models \
    tinyllamas/stories15M-q4_0.gguf          # grab a tiny model into ./data
toy infer data/stories15M-q4_0.gguf \
    --prompt "Once upon a time"              # greedy decode
```

Then train a small transformer from scratch, inspect it, serve it:

```sh
toy train from-scratch --steps 20 --seed 0   # runs/<id>/: weights, events, loss curve
toy eval data/SmolLM2-135M-Instruct-Q8_0.gguf --top-k 5
toy serve data/SmolLM2-135M-Instruct-Q8_0.gguf --port 4567 --name smol
toy list                                     # finds GGUFs in HF / Ollama / LM Studio caches
toy describe data/SmolLM2-135M-Instruct-Q8_0.gguf   # the algorithm card, from metadata
```

`toy --help` shows all 9 commands (`new`, `install`, `list`,
`describe`, `fetch`, `infer`, `train`, `eval`, `serve`);
[`docs/cli.md`](docs/cli.md) has flags, exit codes, and the
machine-readable `--manifest` contract. If `toy list` shows nothing,
any of `toy fetch`, `huggingface-cli download`, `ollama pull`, or LM
Studio will populate a cache it sees.

## Using toy as a framework

The CLI is the front door; the framework is the house. Everything is a
layered stack — **primitives → blocks → archs → recipes** — each layer
plain Ruby, each gated bit-identical against a reference, all of it
loaded with one require:

```ruby
require "toy/compute"

recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, context_len, 1, 0, false, false, 0, 1.0)
steps.times do |step|
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
end
```

`realize!` builds the entire forward + loss + backward + AdamW graph
natively; `step!` drives one training step. That's the whole training
contract — the same one toy's own gates use. Start your own project
with `toy new mylab` (an experiment tree) or `toy new mylib --lib` (a
library consuming toy as a gem, native vendoring included).

**[The framework guide](docs/framework.md)** is the tour;
[`docs/authoring.md`](docs/authoring.md) shows how to add your own
primitive, block, arch, or recipe; [`docs/consuming-toy.md`](docs/consuming-toy.md)
is the full dependency story.

## Models and backends

Seventeen checkpoints run today — across GPT-2, SmolLM2, TinyLlama,
Llama 3.2, Mistral, Qwen 2.5/3, Gemma 2, and OLMoE (MoE) — in F32 and
Q8_0, with three tokenizer flavors and RoPE scaling auto-detected from
the GGUF. CPU is the gated reference
backend; CUDA and Metal mirror it, held bit-identical by
`make verify-mirrors`.

The honest per-model/per-backend matrix (including the footnotes —
what's validated vs. expected-to-work vs. not wired) lives in
[`docs/models.md`](docs/models.md); per-op coverage vs PyTorch in
[`docs/coverage.md`](docs/coverage.md).

## Documentation

- [`docs/framework.md`](docs/framework.md) — **start here to build with
  toy**: the stack, recipes, `toy/compute`, project scaffolds.
- [`docs/architecture.md`](docs/architecture.md) — the five-layer
  algorithm stack and how the CLI shells to compute runners.
- [`docs/cli.md`](docs/cli.md) — the 9 commands, flags, exit codes,
  and the manifest contract.
- [`docs/authoring.md`](docs/authoring.md) — adding a primitive, block,
  arch, or recipe; the algorithm-card round-trip.
- [`docs/models.md`](docs/models.md) — the supported-models matrix,
  tokenizers, RoPE, opt-in performance knobs.
- [`docs/consuming-toy.md`](docs/consuming-toy.md) — toy as a gem
  dependency: vendoring, native builds, CUDA/Metal opt-ins.
- [`docs/gating.md`](docs/gating.md) — the bit-identical
  reproducibility gates that hold all of the above together.
- [`docs/events.md`](docs/events.md) — the `toy/v1` event schema.
- [`docs/roadmap.md`](docs/roadmap.md) — deferred work and live
  research directions.
- [`examples/`](examples/README.md) — focused, single-file entry
  points compiled to native binaries.

## Acknowledgments

A heartfelt thank-you to **Ninoslav Milenović** for graciously handing over
the [`toy`](https://rubygems.org/gems/toy) gem name on RubyGems. A good name
is a gift, and giving one up for someone else's project is the kind of quiet
generosity the Ruby community runs on. We don't take it lightly — thank you,
Ninoslav. 🙏
