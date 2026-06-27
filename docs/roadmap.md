# Roadmap

The one consolidated forward-looking document for Toy. Phase status,
the live deferred list, the modern-LLM-primitives priorities, and the
known issues we're carrying. This is "what's done, what's next, what
we've decided not to chase."

For the **scaling story** — the API-first design where one toy program
runs from a single laptop CPU to a multi-arch WAN cluster via a single
"coupling" dial (`Toy::Topology`), consolidating the trio below — see
[`roadmap/scaling-story-2026-06-18.md`](roadmap/scaling-story-2026-06-18.md).
For the backends/scale future-directions trio it builds on (batched/grad-accum
training, multi-GPU, DiLoCo, non-ggml backend options) see
[`roadmap/backends-and-scale-2026-05-27.md`](roadmap/backends-and-scale-2026-05-27.md),
[`roadmap/training-backends-2026-05-27.md`](roadmap/training-backends-2026-05-27.md),
and [`roadmap/decoupled-diloco-research-2026-05-27.md`](roadmap/decoupled-diloco-research-2026-05-27.md).
For the **Dragon / Gated-DeltaNet trainable hybrid arch** see
[`roadmap/dragon-gdn-arch-2026-06-20.md`](roadmap/dragon-gdn-arch-2026-06-20.md).
Phases 1–5 are **done**: the GDN forward + 4 L1 primitives, the trainable
recurrence (Path B — an unrolled autograd composition, no hand-written kernel
backward), the per-layer `LayerSpec` int-kind dispatch seam, a trainable
`GDNBlock`, and a self-contained from-scratch **attention+GDN hybrid runner**
(`libexec/toy-train-hybrid`, `gate-gdn-hybrid`). Folding the hybrid into the
shared llama engine for plain `toy train` is deferred behind a union-pin Spinel
codegen block — the mechanical re-apply is in
[`roadmap/gdn-hybrid-engine-reintegration.md`](roadmap/gdn-hybrid-engine-reintegration.md).
P6 (a Dragon→toy converter for real weights) has no known path yet.
For the op/backend coverage matrix see
[`coverage.md`](coverage.md).

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
The CPU runners are the gated reference; `infer`/`eval`/`train` also
have `--device cuda` twins (Metal source-wired, Mac-gating). The serve
endpoint logic lives in
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

## Shipped post-v0.7 (was deferred)

These landed on `main` after the phase table above was written (the
post-v0.7 deferred-queue session, `main@507e160` and follow-ups):

- **GPU runners (`--device`).** `--device cuda` is wired for `infer` +
  `eval` (`acf0d0b`) and `train from-scratch` (`88b8ded`), plus
  `lora` / `warm-start` CUDA twins (`429c38d`) — all gated CUDA-vs-CPU.
  Metal twins are source-wired on the `metal-source-wiring` branch
  (runtime-gating on the Mac; not yet merged).
- **`train` recipe variants.** `lora` and `warm-start` dispatch by name
  on `toy train` (`33a5efb`). Only `curriculum` stays deferred (no L4 file).
- **`eval lmc`** — two-checkpoint linear-mode-connectivity shipped
  (`dfaeb64`, `libexec/toy-eval-lmc`, `make gate-lmc`).
- **train→infer round-trip.** From-scratch checkpoints are now loadable
  by `toy infer` (fuse-on-save), gated by `prep/ckpt_roundtrip_gate.rb`
  (`fbd0f35`).

## Live deferred list

Features that are scoped and understood but not built. These don't
block phases — they land inside the layered structure when picked up.

- **ViT / vision in the CLI** — the ViT examples and image-loader
  smokes have no CLI command yet.
- **GPT-2 arch in `toy train`** — only the Llama arch is exposed.
  *In progress on the `gpt2-train` branch:* the two ggml backward kernels
  (`ggml_gelu_back`, `ggml_norm_back` — vendor-patches/0007) are landed and
  finite-diff validated (2/3 of our ggml#1514); the training arch + gate
  (`prep/gpt2_train_gate.rb`) is the next step.
- **Realize residuals** — llama3 `rope_freq_factors` wiring (un-gateable;
  the toy forward is rope-angle-insensitive at the logit level), and
  GQA-divergent on mmap/q8 (no divergent-head GGUF to gate against). The
  `full_finetune` 6th gate is built (`prep/full_finetune_gate.rb`).
- **MoE — RESOLVED (2026-06-05).** OLMoE Q4_K_M decode was incoherent and was
  long misfiled as a ggml `mul_mat_id` K-quant kernel bug (ggml#1506). The op is
  correct for K-quants (op-level + real-bytes reproducers in `tinynn/ggml1506_*`
  all clean); the real cause was `head_nbytes` returning 0 for K-quant ATTENTION
  weights, collapsing every attention head onto head 0. Fixed. K-quant MoE
  experts (incl. OLMoE's mixed q4_K+q6_K `down_exps`) now decode coherently; the
  misleading runtime warning was removed and `make gate-moe-kquant` guards it.
  ggml#1506 is closeable upstream as a non-bug. See `docs/notes/mul_mat_id_quants.md`.
- **Tep, then Tao, re-adaptation** — deferred until Toy is fully
  stabilized. The serve convergence onto tep's `Backend` (#30) is gated on a
  **Spinel pin-bump**: `libexec/toy-serve` won't compile at toy's current
  Spinel (`Tep::Scheduler.spawn_fiber` → `FiberSlot.new` incompatible-pointer +
  a JSON-key monomorphization), but **both are clean at tep's Spinel pin
  `f6d5eef`** — it's a Spinel regression toy hit by running newer Spinel than
  tep, NOT a tep bug and NOT needing a tep release. Aligning toy's Spinel build
  to `f6d5eef` unblocks it (tracked as OriPekelman/tep#198, the pin-bump
  tracker). `main` keeps its hand-rolled handlers until the bump lands. tep is
  consumed as the released RubyGems gem (#31 done).

---

## Modern-LLM primitives — priority three

The shortest path that unlocks the most current open-weight families.
The key insight: **most "missing" primitives aren't missing in ggml —
they're missing in our FFI surface.** Widening bindings is cheaper than
touching kernels.

1. **MoE expert dispatch — binding DONE; the work now is routing variants.**
   The four ops (`MUL_MAT_ID` / `ADD_ID` / `TOP_K` / `ARGSORT`) are bound
   across CPU + CUDA + Metal (`lib/toy/ffi/tinynn{,_cuda,_metal}.rb`,
   shims `tinynn/tinynn_ggml.c`, landed `15cdeb3`), and a full **softmax-gated
   top-k routed FFN** runs in the decode path
   (`SmolLM2KVFFICache#build_moe_ffn`, `llama_kv_engine.rb:1333`: router →
   softmax → top_k → 3× `mul_mat_id` → silu·up → weighted sum). OLMoE-Q4_K_M
   decodes coherently (`make gate-moe-kquant`); **Mixtral / Qwen3-MoE-shape
   already work** (softmax, top-k, no shared expert, no bias).

   What's left is **completing the routing variants** for the newer families
   (the metadata is parsed in `arch.rb` but the forward hardcodes softmax /
   0 shared / no bias):
   - **P1 — gating variants + renorm + loader keys** (highest leverage):
     branch `build_moe_ffn` on `expert_gating` (softmax vs **sigmoid**), add
     optional top-k weight renormalization, and actually *read*
     `*.expert_gating_func` / `*.expert_weights_norm` / `*.expert_shared_count`
     (stop hardcoding `:softmax`/`0` at `arch.rb:249`) + add `qwen3moe.*` /
     `deepseek2.*` arch prefixes. → **Qwen3-MoE** (`norm_topk_prob`) + the
     sigmoid-gated FFN of **DeepSeek-V2 / GLM-MoE / Llama-4**.
   - **P2 — shared experts**: load `*_shexp` weights when
     `n_shared_experts > 0`, build a plain SwiGLU on them, add to the routed
     output. → **DeepSeek-V2 / Qwen3-Next / GLM-MoE** FFN.
   - **P3 — DeepSeek-V3 routing** (hardest): aux-loss-free bias on router
     logits before top-k + node-limited **grouped top-k**. → **DeepSeek-V3**
     FFN (its MLA attention is a separate milestone below).

   All decode-path / coherence-gated (not byte-exact), so union-pin-safe and
   clear of the `:int_array` master bug. Each new family gets its own decode
   gate modelled on `gate-moe-kquant`. Training MoE (the seq engine has no
   MoE block) is a separate, later effort.

2. **Apply the widened RoPE.** The C and Ruby `tnn_rope_ext` binding
   *already* exposes `freq_scale`, `ext_factor`, `attn_factor`,
   `beta_fast`, `beta_slow`, and a `freq_factors` tensor
   (`tinynn/tinynn_ggml.h`, `lib/toy/ffi/tinynn.rb`, consumed by
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
biggest single-family delta remaining** and the natural next milestone —
designed in [`deepseek-mla-arch.md`](roadmap/deepseek-mla-arch.md) (a second
attention engine: latent KV cache + per-head up-projection + decoupled YaRN
RoPE; DeepSeek-V2-Lite is the test model, also needing per-layer dense/MoE
dispatch and the shared experts shipped in MoE-P2).
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
