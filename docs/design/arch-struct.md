# Generic `TransformerLM` + `Arch` struct — design

**Status:** draft, pre-Phase-0
**Authors:** Ori Pekelman + Claude
**Date:** 2026-05-19
**Predecessor:** `lib/toy_smollm2_*.rb` (Qwen2.5-shaped, currently the only path)

## Goal

Replace the Qwen2.5-shaped `SmolLM2KV` / `SmolLM2KVFFICache(Cuda)`
implementation with a generic `TransformerLM` driven by an `Arch`
struct read from each GGUF's metadata. After Phase 0:

- One graph builder. Per-architecture differences live in data
  (the `Arch` instance), not in branched code paths.
- New models are added by writing a one-page `Arch` constructor —
  no graph code is touched.
- Refactor is the prerequisite for Llama-3.2, Qwen3 dense, Qwen3
  MoE, GLM-4.7, and GPT-OSS support. Done badly, MoE forces a
  second refactor pass; done well, MoE is a `:moe` branch in two
  places.

This document fixes the abstraction *before* any code moves.

## Why a refactor is the right call here

The current code is honest about its scope ("smollm2" in the
filename) — it's not pretending to be general. But:

- Five new architectures are on deck. Cloning `SmolLM2KVFFICache`
  five times will produce five subtly-diverging copies that all
  need the same bug fix (e.g. the Phase 3 V-matmul flip — there
  are currently *two* copies of it because the CUDA variant
  diverged).
- The CPU and CUDA classes already drift (per memory: the CUDA
  variant still has the pre-Phase-3 V-matmul order). One graph
  builder eliminates the class.
- MoE introduces a routing layer + expert dispatch *inside* the
  FFN block. Adding MoE as a branch of a Qwen-shaped decoder is
  manageable; adding it as a branch of six near-clones is not.

## Architecture detection

Read `general.architecture` from GGUF (already exposed via
`tnn_gguf_kv_get_*`). The string maps to an `Arch` factory:

| GGUF `general.architecture` | `Arch` factory      | Models                           |
| --------------------------- | ------------------- | -------------------------------- |
| `qwen2`                     | `Arch.qwen2(gguf)`  | Qwen2.5-0.5B/1.5B/3B/7B (current) |
| `qwen3`                     | `Arch.qwen3(gguf)`  | Qwen3.6-27B dense, WebWorld-8B    |
| `qwen3moe`                  | `Arch.qwen3moe(gguf)` | Qwen3.6-35B-A3B                  |
| `llama`                     | `Arch.llama(gguf)`  | Llama-3.2 1B/3B, Mistral-7B-v0.2 (Llama-style after SWA was dropped) |
| `gpt_oss` / TBD             | `Arch.gpt_oss(gguf)` | GPT-OSS-20b (MoE + MXFP4 — deferred) |
| `chatglm` / `glm4`          | `Arch.glm4(gguf)`   | GLM-4.7-Flash                     |
| `smollm` (alias)            | `Arch.qwen2(gguf)`  | SmolLM2 (it's Qwen2 in disguise)  |

The factory reads GGUF kv metadata and returns an `Arch` instance.
Unknown architectures raise `Arch::UnsupportedError`.

## The `Arch` struct

Plain Ruby struct, no inheritance. Booleans for capabilities,
symbols for enum-like choices, integers/floats for dimensions and
hyperparameters. All fields are required (no nil-defaults) — a
new arch has to declare every choice explicitly. That's a feature:
it surfaces silent assumptions.

```ruby
class Arch
  # Identity
  attr_reader :architecture        # Symbol: :qwen2, :qwen3, :qwen3moe, :llama, :glm4, :gpt_oss
  attr_reader :name                # String: "Qwen2.5-1.5B", "Llama-3.2-3B-Instruct", etc.

  # Dimensions
  attr_reader :vocab_size
  attr_reader :d_model
  attr_reader :n_layers
  attr_reader :n_heads_q           # Query heads
  attr_reader :n_heads_kv          # KV heads (== n_heads_q for MHA, < for GQA, 1 for MQA)
  attr_reader :d_head              # Usually d_model / n_heads_q; can differ (Llama-3.2)
  attr_reader :d_ff                # Intermediate / per-expert in MoE
  attr_reader :max_position        # Context length
  attr_reader :untied_lm_head      # Bool: false = lm_head shares weights with embed

  # Attention
  attr_reader :attention_kind      # :mha | :gqa | :mqa
  attr_reader :qkv_bias            # Bool — Qwen2/3 yes, Llama/Mistral no
  attr_reader :qk_norm             # Bool — Qwen3 yes (RMS-norms Q and K before RoPE)
  attr_reader :swa_window          # Integer or nil — window size for sliding-window attention

  # RoPE
  attr_reader :rope_freq_base      # 10000 (GPT-2), 500000 (Llama-3.2), 1e6 (Qwen2/Mistral)
  attr_reader :rope_freq_scale     # 1.0 default; YaRN / linear scaling stored here
  attr_reader :rope_partial_factor # 1.0 default; 0.5 for GLM / Phi (rotate only half of d_head)

  # Norm
  attr_reader :norm_kind           # :rms | :layer
  attr_reader :norm_eps            # 1e-5 / 1e-6 — read from GGUF
  attr_reader :final_norm          # Bool — usually true; placement is always pre-lm_head

  # FFN
  attr_reader :ffn_kind            # :swiglu | :geglu | :gelu_mlp | :relu_mlp
  attr_reader :ffn_bias            # Bool — Llama/Qwen no, GPT-2 yes

  # MoE (only meaningful when moe? is true)
  attr_reader :moe                 # Bool
  attr_reader :n_experts           # Integer (0 when not moe)
  attr_reader :n_experts_used      # top-k (0 when not moe)
  attr_reader :n_shared_experts    # always-on experts (DeepSeek-style; 0 if absent)
  attr_reader :expert_gating       # :softmax | :sigmoid (DeepSeek)

  # Tokenizer (Phase 0 = GGUF-embedded only; later phases extend this)
  attr_reader :tokenizer_kind      # :gguf_embedded for now
  attr_reader :bos_id              # Integer or nil
  attr_reader :eos_id              # Integer or nil
  attr_reader :pad_id              # Integer or nil
  attr_reader :unk_id              # Integer or nil
  attr_reader :add_bos_by_default  # Bool

  # Embed scaling (some models multiply token_embd by sqrt(d_model))
  attr_reader :embed_scale         # Float — 1.0 default

  def moe?; @moe; end
  def gqa?; @n_heads_kv < @n_heads_q; end
  def swa?; !@swa_window.nil?; end
end
```

### Per-model values (worked examples)

| Field             | Qwen2.5-1.5B | Llama-3.2-3B | Mistral-7B-v0.2 | Qwen3.6-35B-A3B          | GLM-4.7-Flash       |
| ----------------- | ------------ | ------------ | --------------- | ------------------------ | ------------------- |
| architecture      | :qwen2       | :llama       | :llama          | :qwen3moe                | :glm4               |
| d_model           | 1536         | 3072         | 4096            | TBD                      | TBD                 |
| n_layers          | 28           | 28           | 32              | TBD                      | TBD                 |
| n_heads_q         | 12           | 24           | 32              | TBD                      | TBD                 |
| n_heads_kv        | 2            | 8            | 8               | TBD                      | TBD                 |
| qkv_bias          | true         | false        | false           | false (Qwen3 dropped it) | false               |
| qk_norm           | false        | false        | false           | true                     | false               |
| rope_freq_base    | 1e6          | 5e5          | 1e6             | 1e6                      | model-dep           |
| rope_partial      | 1.0          | 1.0          | 1.0             | 1.0                      | 0.5                 |
| swa_window        | nil          | nil          | nil (v0.2)      | nil                      | nil                 |
| ffn_kind          | :swiglu      | :swiglu      | :swiglu         | :swiglu                  | :swiglu             |
| moe               | false        | false        | false           | true (128 / top-8)       | true                |
| tokenizer_kind    | :gguf_embedded | :gguf_embedded | :gguf_embedded | :gguf_embedded         | :gguf_embedded      |

(TBD rows are filled at Phase B/C kickoff after reading the actual GGUFs.)

## Generic `TransformerLM`

One class. Single decode-graph builder. Replaces
`SmolLM2KVFFICache` and `SmolLM2KVFFICacheCuda`.

```ruby
class TransformerLM
  def initialize(arch, backend) # backend: :cpu or :cuda
    @arch    = arch
    @backend = backend
    @sess    = (backend == :cuda) ? TinyNNCuda.tnn_session_new(1)
                                  : TinyNN.tnn_session_new(0)
    @weights = nil  # populated by load_gguf
    @kv      = nil  # KV cache, allocated at realize time
  end

  def load_gguf(path);        end # mmap + attach via BYO-pointer
  def realize(max_seq);       end # build decode graph for max_seq context
  def prefill(input_ids);     end # bulk encode prompt → KV cache
  def decode_step(token, pos); end # one-token-at-a-time decode (autoregressive)
  def sample(logits, opts);   end # delegates to Sampler

  private

  def build_decode_graph
    # Reads @arch fields and emits ops.
    # - attention block: Q/K/V projection (with/without bias depending on arch.qkv_bias)
    # - QK norm (if arch.qk_norm)
    # - RoPE with arch.rope_freq_base / arch.rope_partial_factor
    # - KV-cache slot write (the cpy-into-strided pattern, now fixed by ggml patch)
    # - GQA-aware attention (broadcast KV across query head groups)
    # - SWA mask if arch.swa?
    # - residual + norm (placement depends on arch.norm_kind)
    # - FFN: routed to expert dispatch if arch.moe?, else single SwiGLU/GeGLU
    # - final norm + lm_head (tied or untied per arch.untied_lm_head)
  end
end
```

The graph builder is large but linear: one method per block, with
two-three conditionals per block on `@arch` fields. No subclassing,
no strategy pattern, no metaprogramming. A reader who knows what a
transformer is can follow it.

## Tokenizer (Phase 0 scope)

Two parts:

1. **At GGUF load**, extract these from kv metadata:
   - `tokenizer.ggml.tokens` (vocab table, IDs → strings)
   - `tokenizer.ggml.merges` (BPE merge pairs, ordered)
   - `tokenizer.ggml.bos_token_id`, `eos_token_id`, etc.
   - `tokenizer.ggml.add_bos_token`
   - Store on the `Arch` (the tokenizer fields), and keep the full
     vocab/merges in a `Tokenizer` object owned by `TransformerLM`.

2. **Decode side (IDs → text)** — Ruby `Tokenizer#decode(ids)`.
   Sufficient for all phases: server still receives token IDs and
   emits tokens. Output side resolves IDs back to text using the
   embedded vocab.

3. **Encode side (text → IDs)** — *deferred to Phase D*. Clients
   tokenize externally for now (HF tokenizer, llama.cpp tokenize,
   etc.). When we ship Stage 2 tokenizers, the same `Tokenizer`
   object grows an `encode(text)` method.

This matches the current `tep_demo/openai_api_*` design (per
memory: server speaks token IDs). The refactor doesn't change
the API surface — just adds the decode side cleanly.

## Sampler

New module `lib/sampler.rb`. Composable transforms applied to
the logits vector before argmax:

```ruby
module Sampler
  # All transforms take (logits, ctx) and return new logits.
  # ctx is { generated_ids:, prompt_ids:, arch: }.

  def self.temperature(logits, t);             end # logits / t
  def self.repetition_penalty(logits, ctx, p); end # penalize tokens in ctx[:generated_ids]
  def self.top_k(logits, k);                   end # keep top-k, -inf the rest
  def self.top_p(logits, p);                   end # nucleus
  def self.argmax_or_multinomial(logits, opts); end # final pick

  class Config
    attr_accessor :temperature, :top_k, :top_p, :rep_penalty, :stop_tokens, :max_tokens
  end
end
```

Default config per arch lives on the `Arch` instance (sensible
defaults — temperature 0.7, top_p 0.9, etc. — but callers can
override).

Stop tokens are checked after each decode step against
`arch.eos_id` plus any per-call additions.

## Migration plan (how to refactor without breaking Qwen2.5)

The current path that *must* keep working through the refactor:

- `demos/qwen25_direct_native_mmap.rb` (CPU mmap inference)
- `demos/qwen25_direct_native_mmap_cuda.rb` (CUDA mmap inference)
- `demos/smollm2_kv_cuda.rb` (legacy path)
- `tep_demo/openai_api_*` binaries

Sequence:

1. **Add `lib/arch.rb` and `lib/arch/qwen2.rb`** with no callers.
   Verify `Arch.qwen2(gguf_path)` reads correctly for Qwen2.5-1.5B
   and matches the hardcoded values in `SmolLM2KVFFICache`.
2. **Add `lib/tokenizer.rb`** with GGUF-embedded decode. Test
   round-trip on a few prompts: tokens → decode → original-ish text.
3. **Add `lib/sampler.rb`** with the four transforms + Config.
   Unit-test each transform.
4. **Add `lib/transformer_lm.rb`** initially as a wrapper that
   delegates to `SmolLM2KVFFICache` based on `arch.architecture`.
   Switch one demo to it; verify bit-identical output.
5. **Inline the decode graph** into `TransformerLM`. Delete the
   delegation. At this point Qwen2.5 runs through the new code path.
6. **Verify all Qwen2.5 demos** produce identical output to the
   pre-refactor baseline. CPU + CUDA, F32 + Q8.
7. Only after step 6 is green: write `Arch.llama` and the Phase A
   tests.

The old `lib/toy_smollm2_*.rb` files stay in place during steps
1-5 (delegated to) and get deleted at step 6.

## Test plan

**Per-phase acceptance gate.** No phase is "done" until:

1. **Layer-by-layer parity** vs reference (HF transformers run on
   the same prompt): max-abs-diff < 1e-3 on the final logits at N=1
   layer and at full depth.
2. **Bit-identical CPU vs CUDA** on all 28+ layers (the per-layer
   harness from [[project_phase2_cuda_byo_2026_05_18]]).
3. **End-to-end sample**: 50-token generation with the sampler at
   `temperature=0`, deterministic, matches HF reference greedy
   output bit-by-bit.
4. **Sampler smoke**: same prompt at `temperature=0.7, top_p=0.9,
   rep_penalty=1.1`, seed fixed, output deterministic across runs.

Phase 0 acceptance: all four gates above pass for Qwen2.5-1.5B
*through the new code path*. If any gate fails, the refactor is
not done.

## Open questions / future work

- **YaRN / linear RoPE scaling.** Not in the Phase-0 `Arch`. Add
  when the first model that needs it shows up (probably Qwen3 32k+).
- **Encoder–decoder.** Out of scope. `Arch` is decoder-only.
- **Quantization-aware dispatch.** Q4/Q5 GGUFs work today; we add
  a `weight_quant_kind` field on `Arch` only when Phase C3 (GPT-OSS
  MXFP4) lands.
- **Attention sinks / register tokens.** Out of scope for now; some
  recent models (HunYuan) use these.
- **Phi-style parallel attention+FFN block.** Out of scope (Phi-2
  was dropped from targets), but worth noting as an `Arch.block_kind`
  knob if it ever comes back.

## Acceptance for this design doc itself

When this doc compiles in someone's head — i.e. they can predict
what `Arch.qwen3moe(gguf)` returns and how `TransformerLM` consumes
it for a Phase C model — we're ready to start Phase 0.
