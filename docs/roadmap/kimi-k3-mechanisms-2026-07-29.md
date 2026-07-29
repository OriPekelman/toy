# Kimi K3 mechanism coverage: the K-series roadmap (design assessment, 2026-07-29)

**Verdict up front: GO, phased, with one honest descope candidate (native
vision).** K3 (moonshotai/Kimi-K3, 2.8T/104B MoE, tech report in-repo at the
link below) is the current open-weights SOTA, and its mechanism list decomposes
almost entirely onto seams toy already owns: GDN (Gated DeltaNet) is KDA's
direct ancestor and lives IN the engine; MLA is shipped; the MoE instrument
has runtime E/dm/dff/vocab/T and a routing-policy axis; the hybrid-layer seam
(GDN_LAYERS / per-layer kinds) is exactly where a 3:1 KDA:MLA pattern lands.
We do NOT run K3 itself on this box — every phase below is a toy-scale,
byte-gated instrument of a K3 mechanism, spec-callable for Tao, composing with
the credit-assignment program (the house thesis: the K3-shaped instrument ×
DFA/BP axes is the natural F-series successor).

Sources: `k3_tech_report.pdf` (MoonshotAI/Kimi-K3), HF card, and two
inference references useful as EXACT-PATH ORACLES when we need semantics
pinned: gavamedia/deltafin (pure-PyTorch KDA port from flash-linear-attention;
bit-exact MXFP4 dequant tables) and JustVugg/colibri#676 (CPU engine with a
numpy reference validated to ≤2.2e-6 against real K3 weights, MXFP4 kernels).
Neither is a training implementation; both pin forward semantics.

## 1. The K3 mechanism inventory (report §2–§4, Table 1)

| # | Mechanism | One-line description |
|---|-----------|----------------------|
| M1 | **KDA** (Kimi Delta Attention) | delta-rule recurrence with CHANNEL-wise decay: S_t = (I − β_t k_t k_tᵀ)Diag(α_t)S_{t−1} + β_t k_t v_tᵀ; q/k = L2Norm(Swish(ShortConv(Wx))), v = Swish(ShortConv(Wx)); decay logits via low-rank W↑W↓x + per-head bias; **lower-bounded decay** g = g_min·σ(e^{A_h}z), g_min=−5 (bounded reciprocal ⇒ dense tensor-core tiles in the chunkwise/UT form); **full-rank output gate** y = W_o[σ(W_g x) ⊙ RMSNorm(õ)] |
| M2 | **Gated MLA** | DeepSeek MLA (latent c_t = W_c x, up-projected K/V) + **NoPE** on all MLA layers + the same channel-wise full-rank output gate; FP32 attention output during training |
| M3 | **Hybrid 3:1** | per block: 3 KDA layers then 1 Gated MLA; final layer always global (K3: 69 KDA + 24 MLA + 1 dense over 93 layers) |
| M4 | **AttnRes** (Attention Residuals) | per-layer learnable pseudo-query attends over ALL preceding layer outputs (kernel exp(qᵀRMSNorm(k)), softmax over depth); Block form partitions L into N=8 blocks (O(Nd) memory) |
| M5 | **Stable LatentMoE** | routed experts operate in a LATENT space ℓ = d/2 (W↓ → experts → RMSNorm → W↑), 2 full-width shared experts; K3: 896 routed / 16 active / sparsity 56 |
| M6 | **SiTU-GLU** | GLU with softcap on both factors: [β₁tanh(W_g x/β₁) ⊙ σ(W_g x)] ⊙ [β₂tanh(W_u x/β₂)], β₁=4, β₂=25 — bounded where SwiGLU is not |
| M7 | **Quantile Balancing** (QB) | aux-loss-FREE routing: per-expert bias b_j set from the (1−k/n)-quantile of router-score margins via Top-(k+1) cutoffs, one forward pass; bias frozen at inference (histogram estimator only needed at scale) |
| M8 | **NoPE** | no positional encoding anywhere; position lives in KDA's recurrence ⇒ 1M-token extrapolation with no RoPE retuning; progressive context curriculum (8K→64K→256K→1M) |
| M9 | **Per-Head Muon** | Muon (momentum + Newton–Schulz orthogonalization) with per-HEAD partitioning of attention projections; + K2's weight-clipping; cosine decay beat WSD under per-schedule tuning |
| M10 | **MTP layer** | one multi-token-prediction layer mirroring a backbone block (also the EAGLE-3 draft seed via LK loss — inference-side) |
| M11 | **MXFP4/MXFP8 QAT** | expert weights MXFP4, activations MXFP8, quantization-aware from SFT onward; non-expert components higher precision |
| M12 | **MoonViT-V2** | 401M from-scratch vision tower (next-token, NOT contrastive init), pixel-shuffle 2×2, native interleaved multimodal |

Out of scope by construction (not model mechanisms): the RL/MOPD post-training
pipeline, million-token infra (KDA context parallelism, MoonEP, sandboxes),
serving. The scaling-law/data-recipe content informs Tao designs, not toy
surfaces.

## 2. Asset map (what toy already owns per mechanism)

- **M1**: `gdn_block.rb` — Gated DeltaNet per-head recurrence with q/k norm,
  decay g, and a β write-strength ALREADY (`recur_unrolled(qn,kn,v,g,beta,
  state0)`); KDA = channel-wise Diag(α) decay + ShortConv/Swish/L2Norm
  parameterization + lower-bounded scaled-sigmoid decay + full-rank output
  gate ON TOP of this. ggml has conv_1d; tanh/sigmoid/silu/rms_norm/scale
  compose the rest.
- **M2**: MLA-A/B shipped CPU (r=0.999 vs full attention, latent cache
  8.89×; docs/roadmap/deepseek-mla-arch.md). Gate + NoPE are ops we have.
  CUDA MLA was blocked by the old CUDA MoE-FFN crash — re-probe, it predates
  the 2026-07 engine wave.
- **M3**: the GDN_LAYERS seam (per-layer kind table, KIND_ATTENTION guard
  rails in the franken policy) IS the hybrid mechanism; add kinds + a preset.
- **M4**: new, but all ops exist and toy depth is small (L≤6) — the FULL form
  is affordable; Block form is a flag for provenance parity.
- **M5–M7**: the franken-moe instrument (runtime dm/dff/vocab/E/T, dense+top1
  routing, Switch aux, batch, checkpoints, held-out eval) is the chassis;
  LatentMoE = a W↓/RMSNorm/W↑ sandwich; shared experts = an always-on dense
  branch; SiTU-GLU = tanh softcaps; QB = a Ruby-side bias-update rule (exact
  quantile at toy scale — no histogram needed).
- **M8**: NoPE = skip rope (flag); composes with --context/--corpus; the
  toy-scale analog of progressive extension is a Tao curriculum, not new code.
- **M9**: NEW optimizer — but Newton–Schulz is ~5 matmuls per update matrix
  per step, buildable IN-GRAPH with existing ops (the AdamW opt-step seam
  generalizes); per-head partitioning matches the per-head QKV layout the
  engine already carries. m/v buffers exist (m = momentum; v unused by Muon).
- **M10**: an extra block + shifted-label loss head; the MTP loss is a second
  CE root (the two-loss-roots pattern is proven — toy#121 aux).
- **M11**: gate on vendored-ggml MXFP4 support (upstream added it for
  gpt-oss); toy-scale QAT = fake-quant (quantize→dequantize) on expert
  weights per step. Check the vendor tree before scheduling.
- **M12**: vit-tiny exists (classification only). Native-multimodal joint
  next-token is a LARGE lift with the least transfer to the credit-assignment
  program. Descope candidate — decide after K4.

## 3. Phases (each = spec-callable surface + byte gates + provenance, the F-series delivery style)

**K0 — spec + axes (small).** This doc; `--act swiglu|situ-glu`,
`--moe-balance aux|qb|none`, `--optimizer adamw|muon`, `--rope rope|nope`,
layer-kind presets into RecipeOptions/CLI matrix (each axis lands with its
phase; K0 just fixes names so Tao specs don't churn).

**K1 — cheap high-value drops (each byte-null, independently landable).**
SiTU-GLU on llama FFN + MoE experts; NoPE flag; Gated-MLA output gate;
**QB on franken-moe** (direct F8-line science: QB vs the α-swept Switch aux
at E=8/16 — K3's answer to exactly the starvation question F8b/F8c mapped);
cosine-decay schedule flag. CUDA twins where the runner has one.

**K2 — KDA (the big one).** Upgrade the GDN block to the KDA
parameterization: channel-wise decay, ShortConv+Swish+L2Norm q/k/v,
low-rank+bias decay logits, lower-bounded scaled-sigmoid decay (gate the
bound: every α ≥ e^{g_min}), full-rank output gate. RECURRENT form first
(CPU + CUDA twins, byte-gated vs recorded fixtures; deltafin's fla port is
the semantics oracle). The chunkwise/UT tensor-core form is a SEPARATE perf
leg (same split as MLA-A/B) — correctness never waits on it.

**K3 — hybrid + AttnRes.** Layer-kind preset `(kda,kda,kda,mla)×n + mla`
(the 3:1 contract + final-global); AttnRes full form with the Block form
as a flag; ablation axes (--attnres on|off, ratio sweep) for Tao.

**K4 — Stable LatentMoE.** Latent expert sandwich + N_s shared experts +
SiTU-GLU experts + QB composed at E≥8 on the MoE instrument. Exit criterion:
**toy-k3** — a ~10–50M-param instrument with M1–M8 composed, training on the
fineweb pack at --batch on CUDA, eval-ce'd, cost-accounted. This is the
F-series successor instrument: K3-shaped × credit-assignment axes (which
parts of a K3-shaped model tolerate DFA is a paper-shaped question).

**K5 — Per-Head Muon.** In-graph Newton–Schulz opt step; per-head for QKV,
whole-matrix elsewhere; K2 weight clipping. Science hook: Muon × DFA
(orthogonalized DFA updates are unexplored territory).

**K6 — MTP.** One MTP layer mirroring a block + second CE root; provenance +
eval story (t+2 accuracy in eval-ce events).

**K7 — MXFP4/8 QAT (gated on vendor support).** Probe vendored ggml for
MXFP4; if present: fake-quant expert weights per step (toy-scale, no fused
kernels); if absent: vendor-patch cost assessment FIRST, may defer. colibri's
kernels + deltafin's tables are the bit-exactness oracles.

**K8 — native vision (descope candidate).** vit-tiny as a MoonViT-mini
feeding interleaved image tokens into the LM stream. Decide after K4;
default position: descope unless a Tao ask materializes — least transfer to
the program, largest data-pipeline cost.

## 4. Landmines & constraints carried in

- One sched-alloc on graph_b (F1.1); one-graph twins; persistent inputs
  ALLOC before finalize_weights (the toy#133 mask lesson); FFI mirrors via
  gen_cuda_mirror only; monomorphic drivers; landmine #16 unit splits (KDA
  and Muon likely each want their own compiled unit probes early).
- Every phase lands with byte-null defaults + the order-swap/round-trip
  null styles where applicable; the CLI flag matrix gets one row per axis.
- cost accounting must learn linear-attention arithmetic in K2 (KDA is
  O(T·d²)-class, not O(T²·d)) — quality-per-FLOP comparisons between KDA
  and MLA layers are exactly the demonstrator's currency.
- Muon changes the hp contract (no bias-correction slots) — slot-5/6
  dual-meaning discipline (hp memory) must not be silently extended a third
  way; a separate hp layout for muon, fail-loud guarded.

## 5. Descope triggers

- K2 chunkwise form: drop if the recurrent form's CUDA twin already saturates
  toy shapes (measure first; the tensor-core claim matters at K3 scale, not
  necessarily at ours).
- K7: defer if vendored ggml lacks MXFP4 and the vendor-patch is >0011-class
  effort.
- K8: default descope; revisit only on explicit Tao ask.
