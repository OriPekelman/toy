# DeepSeek MLA (Multi-head Latent Attention) — design

Status: **MLA-A + MLA-B SHIPPED** (2026-06-28; CPU). Grounds the implementation
of `deepseek2` inference (DeepSeek-V2-Lite first) in the real GGUF structure. MLA
is the roadmap's "biggest single-family delta" and a *second attention engine* —
fundamentally different from toy's per-head MHA.

`gate-deepseek-mla` greedy-decodes "The capital of France is" → "Paris." (matches
llama.cpp's reference continuation), and asserts **MLA-A == MLA-B** (identical
greedy decode).
- **MLA-A** (`SmolLM2KVFFICache#build_mla_block_step`, default): correctness-
  first, caches the expanded per-head K(192)/V(128).
- **MLA-B** (`#build_mla_b_block_step`, opt-in `KV_MLA_LATENT=1`): caches the
  compressed latent c_kv[512] + shared k_rope[64] and up-projects the history
  through `attn_kv_b` at attention time — **8.89× smaller KV cache**
  (`prep/deepseek_mla_bench.rb`: 5120 → 576 floats/token/layer; at 32K context
  18.1 GB → 2.0 GB), at ~29% more decode time in this naive (non-absorbed) form.
  The **absorbed** form (fold `attn_kv_b` into `attn_q`/`attn_output`, attend in
  latent space) would recover the speed — the natural next optimization.

CUDA: **DeepSeek MLA-A + MLA-B decode coherently on GPU** ("Paris.", "cold."),
via the mechanically-mirrored engine (`gen_cuda_mirror.rb`; `--verify` clean) and
the CUDA driver wired for MoE + MLA-B. Getting there required fixing a CUDA
buffer bug (NOT MLA-specific — it also blocked Qwen3-MoE/Qwen2-MoE on CUDA):
`ggml_backend_cuda_buffer_from_ptr`'s init_tensor zeroed quantized row-padding
with a `cudaMemset` into the **read-only mmap**, an illegal write for any
quantized tensor with `ne0 % 512 != 0` (DeepSeek/Qwen3 `down_exps`,
ne0=expert_ff=1408/768; OLMoE's 1024 was exempt — which is why only some MoE
models crashed). Fixed by `vendor-patches/0010-*` (override init_tensor=NULL on
the BYO buffer; the zeroing is redundant since matmul zero-pads the input). Metal
is unverifiable off-Darwin but mirrored identically.

Test model (downloaded, gitignored): `data/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf`
(`mradermacher/DeepSeek-V2-Lite-Chat-GGUF`, ~9.7 GB). 27 layers, d_model 2048.

## What MLA is

Standard MHA caches per-head K and V (`n_heads · d_head` each). MLA instead
projects the hidden state to a small **latent** `c_kv` (`kv_lora_rank`) plus a
**decoupled RoPE key** `k_rope` (`qk_rope_head_dim`, shared across heads), and
caches *those*. At attention time the latent is up-projected to per-head
`k_nope` and `v`. The cache shrinks from `n_heads·(192+128)=5120`/token to
`512+64=576`/token — the headline DeepSeek win.

## DeepSeek-V2-Lite structure (from the GGUF)

Config (`deepseek2.*`): `head_count=16`, `key_length=192`, `value_length=128`,
`kv_lora_rank=512`, **`q_lora_rank=None`** (Lite has *no* Q compression — the
simplest MLA), `rope.dimension_count=64` (= qk_rope_head_dim), so
`qk_nope_head_dim = 192-64 = 128`. `rope.scaling.type="yarn"`.
MoE: `expert_count=64`, `expert_used_count=6`, `expert_shared_count=2`,
**`leading_dense_block_count=1`** (layer 0 is dense, layers 1–26 are MoE).

Per-block attention tensors (gguf ne):
```
attn_q        [2048, 3072]   d_model → n_heads·qk_head_dim   (16·192=3072)
attn_kv_a_mqa [2048,  576]   d_model → kv_lora_rank+qk_rope  (512+64=576)
attn_kv_a_norm[ 512]         RMSNorm over the latent
attn_kv_b     [ 512, 4096]   kv_lora_rank → n_heads·(qk_nope+v_head) (16·256)
attn_output   [2048, 2048]   n_heads·v_head_dim → d_model    (16·128=2048)
```

## Forward (decode, single token, no q_lora)

Given hidden `x` [2048] at position `pos`:
1. `q = attn_q·x` → [3072] → per head [16, 192]; split `q_nope`[128], `q_rope`[64].
2. `kv_a = attn_kv_a_mqa·x` → [576]; split `c_kv`[512], `k_rope`[64] (shared).
3. `c_kv ← RMSNorm(c_kv, attn_kv_a_norm)`.
4. RoPE (YaRN) applied to `q_rope` (per head) and `k_rope` (shared) at `pos`.
5. **Cache** `c_kv`[512] and `k_rope`[64] for `pos`.
6. Attention over cached `j ≤ pos`:
   - `kvb = attn_kv_b·c_kv[j]` → [4096] → [16, 256]; split `k_nope[j]`[16,128], `v[j]`[16,128].
   - `k[j][h] = concat(k_nope[j][h], k_rope[j])` → [16, 192].
   - `q[h] = concat(q_nope[h], q_rope[h])` → [192].
   - `score[h,j] = (q[h]·k[j][h]) · softmax_scale` (YaRN mscale-adjusted).
   - `attn[h] = Σ_j softmax_j(score[h,:]) · v[j][h]` → [128].
7. `out = attn_output · concat_h(attn[h])` → [2048].

## Integration into toy's KV engine

Dispatch on `is_mla` (detect by `deepseek2` arch / presence of
`blk.0.attn_kv_a_mqa.weight`). Two phases:

- **MLA-A — correctness first (reuse the cache).** New projection path
  (`attn_q`, `attn_kv_a_mqa`→`kv_a_norm`→`attn_kv_b`, decoupled RoPE) that emits
  per-head K[192]/V[128], then reuse toy's existing per-head attention +
  KV cache. Requires the cache to support **asymmetric K/V head dims**
  (`t_K` uses `qk_head_dim=192`, `t_V` uses `v_head_dim=128`) — today `t_K`/`t_V`
  share `@d_head`, so add `@d_head_k`/`@d_head_v`. No memory win yet; correct
  output. This is the milestone to land + gate first.
- **MLA-B — latent cache (the memory win).** Cache `c_kv`[512]+`k_rope`[64];
  up-project per cached position at attention time (or the llama.cpp "absorbed"
  form folding `attn_kv_b` into `attn_q`/`attn_output`). Optimization, separate.

## Prerequisites (land before/with MLA)

1. **Per-layer dense/MoE dispatch.** `@is_moe` is engine-wide; DeepSeek has a
   dense layer 0 then MoE. Make FFN dispatch per-block (a `blk.is_moe` flag set
   by `ffn_gate_inp` presence at load), honoring `leading_dense_block_count`.
2. **Decoupled / partial RoPE.** RoPE applies only to the `qk_rope_head_dim=64`
   slice, not the full head. toy's `tnn_rope_ext` already takes a rope dim; the
   work is slicing `q_rope`/`k_rope` and applying RoPE to just those.
3. **YaRN scaling.** `rope.scaling.type="yarn"`; the `tnn_rope_ext` binding
   already exposes `ext_factor`/`attn_factor`/`beta_fast`/`beta_slow` — wire the
   `deepseek2.rope.scaling.*` keys through (overlaps roadmap priority #2). The
   YaRN `mscale` also adjusts `softmax_scale`.
4. **Routing.** 64 experts top-6 + 2 shared (shared experts already shipped,
   P2). DeepSeek-V2-Lite is softmax-scored, `n_group=1` (no grouping); the
   grouped/bias routing is V3 (a later P3).

## Validation

- New `deepseek2` arch prefix in `detect_arch_prefix` (gives dims/experts).
- Decode-coherence gate `gate-deepseek-mla` (model-gated, ~10 GB), greedy,
  factual-completion assertion like `gate-qwen3moe`.
- Regression: `gate-cpu` byte-exact, `gate-moe-kquant`/`gate-qwen3moe`/
  `gate-qwen2moe-shexp` unchanged (the MLA path is null-gated for non-MLA).
- Mirror to CUDA/Metal via `gen_cuda_mirror.rb`; bind any new FFI op in both.

## Order of work

1. ✅ `detect_arch_prefix` + dims for `deepseek2`; per-layer dense/MoE dispatch
   (`cfg.leading_dense`; FFN dispatches dense for `li < leading_dense`).
2. ✅ MLA-A projection path + asymmetric K(192)/V(128) cache + decoupled RoPE
   (CPU), `gate-deepseek-mla`.
3. ✅ YaRN wiring — `tnn_rope_ext_yarn` (new C shim: the old `tnn_rope_ext`
   hardcodes `n_ctx_orig=0`, which degenerates the YaRN ramp to plain linear
   scaling, AND hardcodes NEOX mode). The shim takes an explicit `mode` +
   `n_ctx_orig`. `RopeScaling.deepseek_yarn` folds the mscale into the softmax
   `kq_scale = mscale²/√d_head_k`, with the rope `attn_factor` back-compensated
   so the rotation magnitude stays 1.0.
4. ✅ MLA-B latent cache (the memory win) — `#build_mla_b_block_step`, opt-in
   `KV_MLA_LATENT=1`; caches c_kv[512]+k_rope[64], up-projects the history at
   read time. 8.89× smaller cache, numerically == MLA-A. Naive (non-absorbed);
   the absorbed form is the next optimization to recover the ~29% speed cost.
5. ✅ CUDA/Metal engine mirror — `gen_cuda_mirror.rb` regenerates both with the
   MLA methods (`--verify` clean); CUDA driver wired (MoE + MLA-B). DeepSeek
   MLA-A + MLA-B decode coherently on CUDA after fixing the BYO-buffer
   init_tensor padding-memset crash (`vendor-patches/0010-*`; also unblocked
   Qwen3-MoE/Qwen2-MoE on CUDA). Metal unverifiable off-Darwin.

## Bring-up notes (gotchas hit during MLA-A)

- **add_bos_token.** DeepSeek-V2 sets `tokenizer.ggml.add_bos_token=1` and the
  attention sink lives on BOS (100000). Without it the decode degenerates to
  "is is is". The byte-level BPE encode path now honors `@add_bos` (gated, so
  every toy-converted fixture — which omits the key — stays bit-identical).
- **RoPE mode = NORM, not NEOX.** The decoupled-RoPE pe slices in these GGUFs
  (llama.cpp `convert_hf_to_gguf.py` output) are laid out for `GGML_ROPE_TYPE_NORM`
  (0; GPT-J interleaved pairs), not the NEOX (2) the rest of toy uses. With NEOX,
  fact recall garbles ("The opposite of hot is is") and the full-logit Pearson r
  vs llama.cpp is only 0.94; with NORM it is 0.999 and the argmax matches. This
  was THE bring-up bug — found by sweeping `mode` against the llama.cpp reference.
- **Massive activation.** Layer-3's routed-expert `down` output spikes the
  residual to ±1130 (DeepSeek's known early-layer "super weight"); it is stable
  and not a bug.
- **Splits.** Per head, attn_q = [q_nope(128), q_rope(64)] and attn_kv_b =
  [k_nope(128), v(128)], both head-major; kv_a = [c_kv(512), k_rope(64)] with
  k_rope shared across heads and RoPE'd once.

DeepSeek-V3 adds grouped top-k + the aux-loss-free expert bias (MoE P3) on top
of this MLA; out of scope here.
