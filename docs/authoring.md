# Authoring algorithms

How to add or modify an algorithm in toy's standard library. The
algorithms live under `lib/toy/llm/` in five layers; each unit is one
file, and a layer only ever calls down to the layer beneath it.

```
lib/toy/llm/
  primitives/   L1  one named op            rope, swiglu, rms_norm, gqa
  blocks/       L2  (x, ctx) -> x'          transformer_block
  archs/        L3  full forward graph      llama_arch
  recipes/      L4  training plan           from_scratch, lora, warm_start
```

Two cross-cutting rules govern every layer and are non-negotiable
because they are gated:

1. **Mirror discipline** — you write the CPU file; the `_cuda` / `_metal`
   mirrors are generated, and `make verify-mirrors` fails the build if a
   committed mirror drifts from its CPU source.
2. **Cards are derived, not authored** — you write the graph-building
   code; the framework derives the algorithm description by walking the
   built graph.

Both are explained below.

---

## L1 — a primitive

A primitive is a single named op. It is a **pure module**: `self.`
methods only, no module ivars, no state. Inputs (the session handle, the
input tensors, a small config value object) come in as arguments; the
output tensor handle comes back. Weight allocation and any per-layer
state belong to the L2 block that calls the primitive, never here.

The four shipped primitives are `rms_norm`, `rope`, `swiglu`, `gqa`
(`lib/toy/llm/primitives/`). Their entry points:

```ruby
Toy::LLM::Primitives::RMSNorm.build(sess, x, gamma, eps)          # -> tensor
Toy::LLM::Primitives::RoPE.apply_2d(sess, t_pre, positions,
                                    freq_factors, cfg, t_seq, t_batch)
Toy::LLM::Primitives::SwiGLU.gate(sess, t_gate, t_up)             # -> tensor
Toy::LLM::Primitives::GQA.attention(sess, t_k, t_q, t_vt,
                                    attn_mask, scale, batch)       # -> tensor
```

`rope.rb` is the canonical shape to copy. It declares `NAME = :rope`, a
small `Cfg` value object carrying plain scalars, and one `self.apply_2d`
that does integer/shape math plus FFI passthrough. It does **not**
`require_relative "tinynn"` — the loading module
(`lib/toy/llm/engine/llama_seq_engine.rb`) has already loaded the correct backend's
`TinyNN` before this file is required, and leaving the require out is
what lets the mirror generator pick the backend (see Mirrors, below).

Why pure modules: a module with only `self.` methods has no ivar layout,
so it sidesteps Spinel's whole-program ivar/accessor type-unification (a
recurring source of mis-compiles — see
[reference/memory-design.md](reference/memory-design.md) and
[dependencies.md](dependencies.md)). Composition is then mechanical: a
block just calls the primitives in order.

Spinel hygiene that applies at L1: config value objects take **all args
positionally with no defaults** (default-arg poisoning); no `STDERR`, no
`Optional` returns, no `Array<Array<mixed>>` destructuring inside the op.

## L2 — a block

A block composes L1 primitives into one repeatable unit and **owns its
weight handles**. The shipped block is `transformer_block.rb`
(`Toy::LLM::Blocks::TransformerBlock`): RMSNorm + GQA-with-RoPE attention
+ SwiGLU FFN, in sequence-mode forward.

The forward signature is:

```ruby
block.build_forward(sess, t_x, ctx) -> t_resid
```

`ctx` is a `TransformerBlockCtx` — a plain positional class carrying the
per-forward read-only context (scale, eps, dims, positions, RoPE cfg,
mask, dtype, LoRA flag). The block reads its weights off `self.t_seq_*`
and returns a single residual-stream handle. Note this is **seq-mode**:
no per-block KV `state` is threaded in or out (incremental KV-cache
decode is the separate `lib/toy_smollm2_ffi_kv.rb` path, out of scope for
this layer).

The block also owns the **allocate / load** of its weights for each
realize path: `alloc_trainable_f32_weights!` (random-init training),
`load_from_gguf_mmap!` (mmap'd GGUF base), and the Q8-stays-Q8 pair
`alloc_q8_typed_from_gguf!` / `copy_q8_bytes_from_gguf!`. Each takes every
value it needs as an argument — no ivar reads off the cache.

Spinel hygiene that applies at L2: `TransformerBlockCtx` is a plain class
with an explicit positional `initialize` (no kwargs, no defaults) —
**never `Struct.new`**, because a Struct's synthesized accessors unify
their return type across every class that exposes the same accessor name,
mis-compiling unrelated callers. The member names keep the verbose
`@seq_*` / `@t_seq_*` prefixes precisely to keep each accessor's inferred
type local. No `Card` / `step_bind` / FFI `:str` argument at class-load
time.

## L3 — an arch

An arch is the full forward orchestration: token-embed `get_rows` ->
(optional projection-lens matmul) -> stacked L2 blocks -> final RMSNorm
-> tied/untied logits matmul. The shipped arch is `llama_arch.rb`
(`Toy::LLM::Archs::LlamaArch`), covering the llama/qwen family.

```ruby
arch.build_forward(sess, t_token_ids, t_positions, t_rope_freq_factors,
                   t_attn_mask, seq_eps, seq_d_head, seq_n_kv, seq_n_heads,
                   seq_group_size, seq_has_qkv_bias, seq_weight_dtype,
                   seq_lora_q_enabled, seq_t, seq_b, seq_n_layers,
                   seq_has_untied_output)
  -> LlamaArchForwardOut(t_seq_x_embed, t_seq_x_final, t_seq_logits)
```

The arch owns the arch-level persistent handles (`t_seq_token_embed`,
`t_seq_final_norm_gamma`, `t_seq_output`, `t_seq_w_proj`, and the
`seq_blocks_ffi` array). It builds the shared `TransformerBlockCtx`
**once**, then loops `block.build_forward` over the blocks. `seed_blocks!`
fills the block array; `load_globals_from_gguf_mmap!` loads the
arch-level globals for the mmap path. The graph **input** handles
(`token_ids`, `positions`, mask, freq-factors) are allocated by the cache
and passed in.

`LlamaArchForwardOut` is again a hand-written positional class — never a
Struct — for the same type-isolation reason as the block.

## L4 — a recipe

A recipe is a training plan that wraps the existing FFI training loop.
The shipped recipes are `from_scratch.rb`, `lora.rb`, `warm_start.rb`
(`Toy::LLM::Recipes::*`). `from_scratch.rb` is the template:

```ruby
r = Toy::LLM::Recipes::FromScratch.new
r.realize!(cfg, t_seq, t_batch, weight_dtype, untied, qkv_bias, seed, init_scale)
loss = r.step!(seq_ids, positions, m_labels, m_hp, is_first)  # one step
```

`realize!` delegates to the cache (`realize_for_random_init` then
`build_training_step`); `step!` runs one step (reset, four ordered
uploads, `compute_backward`, download loss). **AdamW is baked into the
ggml backward graph** by `build_training_step`, so there is no Ruby
optimizer to wrap — the recipes deliberately introduce **no
`Trainer`/`Stage` abstraction**. The three recipes differ only in
`realize!`:

- **FromScratch** — `realize_for_random_init` + `build_training_step`,
  fused (nothing to upload between).
- **LoRA** — `enable_lora_q!` + `enable_lora_q_adamw!` +
  `realize_for_mmap` (frozen mmap'd GGUF base) + `upload_lora_q_init!` +
  `build_training_step`. Same `step!` as FromScratch.
- **WarmStart** — splits realize/build into `realize_scratch!` /
  `realize_warm!` / `build!` so the fixture can upload the donor
  embedding (and/or PCA lens) **between** realize and build.
  `realize_warm!` is optional (scratch init skips it).

Experiment-specific config — the GGUF path, ranks, tokens, labels,
hyperparameters, LR schedule, corpus streaming — stays in the **fixture
or example**, never in the recipe (library-vs-example scope). The
duplicated ~8-line `step!` across sibling recipes is intentional; a
shared module would be over-abstraction.

`curriculum.rb` is **deferred** — the file does not exist. The
multi-stage `each_stage` / `Stage` / `DataSpec` / `Eval` shape sketched
in `lib/toy/llm/recipes/README.md` is a target, not current code.

The L4 recipes ship CPU-only; their CUDA mirrors are deferred alongside
the GPU-runner deferral (see [roadmap.md](roadmap.md)).

---

## Mirror discipline

Backends are CPU (default), CUDA (GB10), and Metal. L1/L2/L3 each carry a
generated `_cuda` and `_metal` sibling. **You only write the CPU file.**

The generator is `prep/gen_cuda_mirror.rb`. The set of files it mirrors
is the `MIRRORABLE` list at the top of that script; the L1-L3 algorithm
files are all in it. To add a new mirrored algorithm file you add it to
`MIRRORABLE` and give it a per-file substitution entry. The mechanical
rewrite renames `TinyNN` -> `TinyNNCuda`/`TinyNNMetal`, switches the
session backend id, and rewrites any `require_relative "tinynn"`.

Workflow:

```sh
ruby prep/gen_cuda_mirror.rb          # regenerate every mirror
make verify-mirrors                   # CI gate: fails on drift
```

The generated files carry an `AUTO-GENERATED by prep/gen_cuda_mirror.rb.
Do not edit by hand` banner. Never hand-edit a `_cuda` / `_metal` file —
edit the CPU source and regenerate. `make verify-mirrors` exits non-zero
if any committed mirror no longer matches what the generator would
produce; it runs in the build, so a stale mirror blocks merge. See
[gating.md](gating.md) for where this sits in the gate suite.

The L1 files deliberately omitting `require_relative "tinynn"` (above) is
what makes this clean: the backend is selected by the monolith's require
rewrite, so the primitive body itself is backend-agnostic and the
substitution table stays tiny.

## Cards are derived, not authored

A Card is toy's structural description of an algorithm. **Users do not
hand-write Cards** — hand-writing pseudocode silently drifts from the
code. Instead you write the graph-building `realize`/`build_forward` code,
and the framework derives the Card.

Today's mechanism is a **runtime graph-walk**: realize the graph (with
tiny dims), then walk the built ggml graph and classify every tensor.
This lives in `lib/toy/dev/toy_describe_flow.rb` — `ToyDescribeFlow.card(sess)`
emits a structural Card (param leaves, input leaves, compute-node count,
the output node), and `.text` / `.json` / `.mermaid` emit the same DAG in
other forms. Because the Card comes from the realized graph, it cannot
drift from the code that built it.

> Note: the `toy describe <model.gguf>` CLI command is a **different**
> path — it builds a Card from GGUF *metadata* plus the known
> transformer shape (`lib/toy/core/cli/describe.rb`, MRI-clean, no FFI),
> not from a realized graph. The graph-walk derivation above is the
> author-facing path for the algorithms you write.

A `def card` method is therefore **optional** — it exists only to
override the derived Card with hand-tuned prose (math notation,
textbook-style pseudocode). The default is "derived from the graph,
rendered by the framework."

A future **static (Prism) lowerer** would walk the `realize` AST at build
time and emit a Card that captures conditional/loop branches a runtime
probe cannot see. That is optional future work — see
[roadmap.md](roadmap.md).

## DRY — require, don't copy

A new arch should `require` the framework primitives and compose them,
not bulk-copy them into the project. To override one primitive, vendor
just that primitive into the project's `algos/` and register it; the rest
keeps requiring from the framework. This keeps "one readable file per
arch" without duplicating shared math.

The `toy g` scaffolding surface that would generate these thin files
(`toy g arch ... --based-on llama`) is **deferred — no generators surface
exists yet** (P5). Until it ships, compose by writing the file directly,
following the layer shapes above.

## Gate-before-merge

Every new layer ships with a smoke fixture and an entry in the gate
harness. The current fixtures live in `examples/`:

| Layer | Fixture |
| --- | --- |
| L1-L3 realize (random-init forward) | `examples/smoke_projection_lens.rb` (+ `_cuda`, `_metal`) |
| L4 FromScratch | `examples/smoke_recipe_from_scratch.rb` |
| L4 LoRA | `examples/smoke_recipe_lora.rb` |
| L4 WarmStart | `examples/smoke_recipe_warm_start.rb` |

Each fixture is built and run as a gate (the `examples/smoke_recipe_*`
and `examples/smoke_projection_lens*` Make targets). The shape contract
is **bit-identical output**: an extracted layer must reproduce the prior
op order exactly, and a generated mirror must match its CPU source.
Compute-runner behavior (infer/train/eval/serve) is gated separately via
`prep/{infer,train,eval,serve}_gate.rb`.

For the full gate suite, fixtures, and the verification gates that block
merge, see [gating.md](gating.md). For the layered overview and the
runtime/compute boundary, see [architecture.md](architecture.md).
