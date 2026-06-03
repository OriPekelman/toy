# Roadmap

The one consolidated forward-looking document for Toy. Phase status,
the live deferred list, the modern-LLM-primitives priorities, and the
known issues we're carrying. This is "what's done, what's next, what
we've decided not to chase."

For the backends/scale future-directions trio (batched/grad-accum
training, multi-GPU, DiLoCo, non-ggml backend options) see
[`roadmap/backends-and-scale-2026-05-27.md`](roadmap/backends-and-scale-2026-05-27.md)
and [`roadmap/decoupled-diloco-research-2026-05-27.md`](roadmap/decoupled-diloco-research-2026-05-27.md).
For the op/backend coverage matrix see
[`reference/coverage.md`](coverage.md).

---

## Phase status

The framework was built in six phases. P0–P4 are done; P5–P6 are
future and optional.

| Phase | Scope | Status |
| --- | --- | --- |
| **P0** | Design lock — five-layer algo contract, `runs/<id>/` layout, `toy.yml` minimalism | **DONE** |
| **P1** | Card derivation — Cards come from a runtime graph-walk (`lib/toy/dev/toy_describe_flow.rb`), not hand-written `algorithm` methods | **DONE** |
| **P2** | Five-layer refactor of the LLM stdlib into L1–L4 | **DONE** |
| **P3** | Core + CLI MVP — `new`/`install`/`list`/`describe`/`fetch` + `--manifest` | **DONE** (2/3 platforms gated) |
| **P4** | `infer`/`train`/`eval`/`serve` CLI commands | **DONE + gated on `main`** |
| **P5** | Generators (`toy g arch|recipe|primitive`) | **FUTURE / optional** |
| **P6** | Prism lowerer (`toy build`) | **FUTURE / optional** |

### P2 — complete

The five layers landed and are gated bit-identical:

- **L1** primitives: `rope`, `swiglu`, `rms_norm`, `gqa`
  (`lib/toy/llm/primitives/`, `_cuda` + `_metal` mirrors).
- **L2** block: `transformer_block`.
- **L3** arch: `llama_arch`.
- **engine**: `llama_seq_engine`, `vit_tiny_engine` (`lib/toy/llm/engine/`).
- **L4** recipes: `from_scratch`, `lora`, `warm_start`, `vit_tiny`.

The former top-level monolith `lib/llama_seq_forward_ffi.rb` is **retired**
— relocated to `lib/toy/llm/engine/llama_seq_engine.rb` as
`Toy::LLM::Engine::LlamaSeqEngine`. All four `realize` paths
(`random_init`, `mmap`, `q8_copy`, `full_finetune`) are decomposed onto
`TransformerBlock` / `LlamaArch`; the `full_finetune` lift is gated
byte-exact by `prep/full_finetune_gate.rb` (the 6th realize gate). The
`_cuda` / `_metal` mirrors are generated at build time (not committed).

Two realize residuals remain out-of-scope:

- **llama3 `rope_freq_factors` realize-wiring** is genuinely
  un-gateable here: the toy forward is rope-angle-insensitive at the
  logit level, so the tensor gate only covers the standalone RoPE
  primitive, never the wired-up arch path. (The angle-insensitivity is
  itself a parked finding worth a dedicated look — primitive *is*
  sensitive in isolation; the full forward is not.)
- **GQA-divergent on mmap/q8** has no divergent-head GGUF to gate
  against; the round-trip pins non-divergence and `w_o` stays
  hard-square (never unified with `random_init`'s divergent shape).

A **CurriculumRecipe** is DEFERRED — the L4 file is absent.

### P3 — platform gate at 2/3

All 9 commands plus `--manifest`/`--help`/`--version` and the `toy.yml`
loader are verified on **gx10 (aarch64)** and **Mac (M2 / Metal,
install + inference end-to-end)**. The only unverified leg is
**Linux x86_64**. Accept 2/3, or verify on an x86_64 box if the full
3-platform gate matters.

Packaging (issue #28) resolved to **option (c)**: the gem ships the
backend build inputs (Makefile, `vendor-patches/`, prep filters, tinynn
CPU/Metal/CUDA shim sources), and `toy install` clones+patches+builds
ggml at install time. The fat-gem option (prebuilt per-platform `.a`)
is deferred.

### P4 — complete

`infer`, `train`, `eval`, `serve` all ship and are gated byte-for-byte
on `main`. Each follows the same shape: a CRuby CLI command
(`lib/toy/core/cli/<cmd>.rb`) shells a Spinel-compiled runner
(`lib/toy/run/<cmd>.rb` → `libexec/toy-<cmd>`) with a controlled env.
All four runners are **CPU-only**. The serve endpoint logic lives in
`lib/toy/serve/openai/{server,handlers,api_json,embeddings_handler}.rb`
(folded out of the retired `tep_demo` server), gated by
`prep/serve_gate.rb`.

**`serve` is the only Tep-coupled command** (Tep is a build-dep only —
`infer` / `train` / `eval` are tep-free). It carried an inbound-TLS
link blocker: the vendored `vendor/spinel/tep/lib/tep/net.rb` was
missing the `ffi_lib "ssl"` / `ffi_lib "crypto"` markers upstream tep
added, so the TLS object linked without `-lssl -lcrypto`
(`undefined reference to TLS_server_method@OPENSSL_3.0.0`). The fix was
to **re-vendor tep**; the current `net.rb` carries the ssl/crypto
markers and `libexec/toy-serve` links against `libssl`/`libcrypto`.
Keep re-vendoring in mind whenever upstream tep's net layer moves.

---

## Live deferred list

Features that are scoped and understood but not built. These don't
block phases — they land inside the layered structure when picked up.

- **GPU runners (`--device`).** The four `libexec/toy-*` runners are
  CPU-only. The CUDA/Metal inference paths use hand-written
  `ToyLMCuda` / `ToyLMMetal` classes with a different constructor arity
  than the seq-forward runner, so they can't be mechanically mirrored
  yet. A `--device cuda|metal` flag is the future surface. Until then
  the `*_metal` / `*_cuda` examples are kept as the GPU path.
- **`train` recipe variants.** Only `from-scratch` is exposed on the
  CLI. `lora`, `warm-start`, and `curriculum` recipes exist (or are
  deferred, for curriculum) but aren't wired to `toy train` yet.
- **`eval lmc`** — the two-checkpoint linear-mode-connectivity eval
  (slice 2). Slice 1 (CE / per-token logprobs) shipped.
- **train→infer checkpoint round-trip.** `toy train` writes
  `runs/<id>/weights/step_N.gguf`, but the checkpoint uses
  training-graph tensor naming and is **not loadable by `toy infer`**
  yet (the same per-head-vs-fused naming theme as the GGUF round-trip
  gate). The fix is a fuse-on-save / loadable-name pass; until then the
  train gate asserts checkpoint *existence* only. This matters for
  "train then run your model" and is a prerequisite for a clean,
  stable Toy.
- **ViT / vision in the CLI** — the ViT examples and image-loader
  smokes have no CLI command yet.
- **GPT-2 arch in `toy train`** — only the Llama arch is exposed.
- **Realize residuals** — llama3 `rope_freq_factors` wiring (un-gateable;
  the toy forward is rope-angle-insensitive at the logit level), and
  GQA-divergent on mmap/q8 (no divergent-head GGUF to gate against). The
  `full_finetune` 6th gate is built (`prep/full_finetune_gate.rb`).
- **MoE.** `ggml_mul_mat_id` produces garbage on K-quant expert weights
  (Q4_K / Q5_K / Q6_K) — see [`reference/coverage.md`](coverage.md)
  and ggml-org/ggml#1506. Workaround: **Q8_0 experts** (non-expert
  tensors can stay K-quant). `realize_for_mmap` warns loud when it sees
  K-quant experts.
- **Tep, then Tao, re-adaptation** — deferred until Toy is fully
  stabilized. Tep is a build-dep only today; the "Tep consumes toy's
  serve surface" and "Tao consumes the events stream" arcs come after.

---

## Modern-LLM primitives — priority three

The shortest path that unlocks the most current open-weight families.
The key insight: **most "missing" primitives aren't missing in ggml —
they're missing in our FFI surface.** Widening bindings is cheaper than
touching kernels.

1. **Bind `MUL_MAT_ID` + `ADD_ID` + `TOP_K` + `ARGSORT`** (sparse MoE
   expert dispatch + router). All exist in ggml; we have zero bindings.
   Unblocks **five families**: Mixtral, Qwen3-MoE, DeepSeek-V3 FFN,
   Llama-4-MoE, GLM-MoE. (Carries the K-quant-expert caveat above —
   ship with Q8_0 experts until ggml#1506 lands.)

2. **Apply the widened RoPE.** The C and Ruby `tnn_rope_ext` binding
   *already* exposes `freq_scale`, `ext_factor`, `attn_factor`,
   `beta_fast`, `beta_slow`, and a `freq_factors` tensor
   (`tinynn/tinynn_ggml.h`, `lib/tinynn.rb`, consumed by
   `lib/toy/llm/primitives/rope.rb`). The remaining work is **wiring
   the scaling args through per-arch realize paths** — YaRN
   (Qwen3 long-context), Llama-3 freq-scaling, Phi-3 LongRoPE
   `freq_factors`, GLM partial RoPE — which is exactly the
   `rope_freq_factors` realize-wiring residual noted under the P2
   ceiling.

3. **SWA mask + per-layer KV sizing.** No new op needed — sliding-window
   attention is a different mask tensor fed to `soft_max_ext`, plus a
   per-layer KV-cache size policy. Pure Ruby + graph-builder. Gates
   Gemma 2/3 (interleaved local/global) and Phi-3-mini-4k; also feeds
   Llama-4 chunked attention.

After the three: **DeepSeek MLA (multi-head latent attention) is the
biggest single-family delta remaining** and the natural next milestone.
Gemma adds a logit soft-cap (bind `GGML_UNARY_OP_TANH`); Qwen3 dense is
small additional work (QK-norm + bias-off).

**Next concrete targets:** Mixtral · Mamba-Jamba · Gemma3 ·
DeepSeek-V3-MLA · Llama4-MoE.

Out of scope on this list: Mamba/SSM and RWKV (ggml has the ops, but the
architectures deserve their own scoping); diffusion; encoder-decoder.

---

## Known issues and takeaways

- **Heavy LoRA-train perf is exhausted at the app layer.** The ~2×
  gap to PyTorch on `make bench-heavy` (2.01× at LoRA-1.5B) is **upstream
  ggml-cuda single-stream serialization**, not Toy's to close. Five
  candidate optimizations (head-fusion, pinned upload, OUT_PROD vs
  MUL_MAT, macro-op fusion, one-hot upload) were all refuted or
  attributed upstream with measured evidence. ggml-cuda already
  auto-fuses several relevant patterns; Toy gets those for free. The
  real fix needs multi-stream dispatch or `cudaGraph` capture upstream.
- **CPU/CUDA LoRA-train numeric divergence.** Root cause is a
  still-OPEN ggml-cpu `ggml_backend_sched` buffer-slot aliasing bug on
  long backward chains. It is **masked** by a
  `tnn_pin_all_graph_b_nodes` call in `lib/toy/llm/engine/llama_seq_engine.rb`
  (and the CUDA/Metal mirrors) between build-backward and
  realize-backward, which prevents slot reuse. With the pin in place,
  CPU and CUDA gradients agree to float32 tolerance. The upstream fix
  is still a TODO.
- **`FLASH_ATTN_BACK` aborts upstream** (`GGML_ABORT`) — flash
  attention stays unbound until that's resolved; the current attention
  path is correct, this would be a perf upgrade.
- **Chat templating is deferred.** `toy serve` is **IDs-in / IDs-out**
  (`POST /v1/completions` with token-ID prompts; `/v1/chat/completions`
  returns 501). The agreed backend shape is a two-method split —
  `generate_from_messages` + `generate_from_tokens` — where Toy
  implements the token form today and the message form lands once Toy
  ships per-arch chat templates.

---

## What this roadmap is NOT

- **Not a commitment to do everything.** P5 and especially P6 are
  optional. If the runtime Card-derivation path stays "good enough", the
  Prism lowerer may never be built.
- **Not a feature list.** Specific algo additions (MoE families,
  new primitives) happen inside the layered structure once it exists;
  they don't gate phases.
- **Not a Tao roadmap.** Tao's work runs in parallel; coordination
  happens at the Recipe Card + events surface.
