# OLMoE Q4_K_M corruption — RESOLVED (was per-head attention stride, NOT mul_mat_id)

**Status (2026-06-05): FIXED.** Root cause was `head_nbytes()` in
`lib/toy_smollm2_ffi_kv.rb` returning **0** for K-quant attention weights,
collapsing every attention head onto head 0's weight slice. It had **nothing to
do with `mul_mat_id` or the experts** — the op was correct for K-quants all
along. ggml#1506 is a non-bug on the ggml side (closeable). Fix: generalize
`head_nbytes` via `tnn_row_size` (correct for F32, Q8_0, and all K-quants).

> **THE ACTUAL BUG.** MoE GGUFs are force-routed through `realize_for_mmap`
> (`load_cpu`, the `if flags.is_moe → is_native = true` branch). That path slices
> the fused `attn_q/k/v.weight` into per-head tensors at byte offset
> `off_base + hq * head_nbytes(type, d_head, d_model)`. `head_nbytes` only had
> arms for F32 (type 0) and Q8_0 (type 8); **every K-quant fell to the `else → 0`**.
> Stride 0 ⇒ all heads read head 0's slice ⇒ multi-head attention collapses ⇒
> degenerate repeating output, compounding across 16 layers. It surfaced only on
> K-quant **MoE** models because (a) non-MoE legacy K-quant GGUFs take the *copy*
> loader (`realize_for` + `load_kv_cache_auto`, which never calls `head_nbytes`),
> and (b) Q8_0 MoE GGUFs hit the working type-8 arm. The smoking gun in the trace:
> `L0.concat` min/max == `L0.head0` (all heads identical) on Q4_K_M, but wider than
> `head0` on Q8_0. After the fix, `concat` matches the Q8_0 reference.
>
> **How it was localized** (all reproducers kept in `tinynn/`):
> - `ggml1506_mul_mat_id_kquant_repro.c` — synthetic op: K-quant `mul_mat_id` clean.
> - `ggml1506_mmap_byo_repro.c` — REAL q6_K/q4_K `down_exps` bytes via the mmap
>   BYO-pointer attach vs copied vs F32: bit-identical, experts definitively fine.
> - `ggml1506_broadcast_repro.c` — the broadcast (`b->ne1==1`) gate/up call shape:
>   also clean for K-quants.
> - `gguf_all_types.c` — diff of Q4_K_M vs q8_0 tensor dtypes showed **all**
>   weights differ (attn + embed + output), not just experts → premise was wrong.
> - `gguf_requant_q4k.c` — built a non-MoE Q4_K tinyllama; it ran coherently
>   (copy path), proving the defect was MoE-path-specific, not general K-quant.
> - `trace_olmoe.rb` / `lib/toy/run/infer_trace.rb` (→ `libexec/toy-infer-trace`)
>   — per-tap min/max/|mean|/nan dump; revealed the `concat == head0` collapse.

**Affected**: MoE inference via `tnn_mul_mat_id` (used in OLMoE,
Mixtral, Qwen-MoE, Granite-MoE, and any other arch with routed
experts) when the expert weight tensors are stored as K-quants
(Q4_K, Q5_K, Q6_K — the "K-family" quantization formats).

## What goes wrong

M2.3 (commit 42b8609) integrated MoE inference end-to-end. Two
GGUFs of OLMoE-1B-7B-Instruct were tested:

|              | per-expert dtype | output                                           |
|--------------|------------------|--------------------------------------------------|
| Q4_K_M GGUF  | Q4_K (some Q6_K) | "The capital of France is **Dub Dub Dub Dub…**"  |
| Q8_0 GGUF    | Q8_0             | "The capital of France is **called Paris.**"    |

Same model, same inference code, same router weights (F16 in both).
Only the expert tensor dtype changes. Q4_K produces degenerate
repeating output; Q8_0 produces coherent factual text.

The model graph itself (router → softmax → top_k → 3× mul_mat_id →
silu·up → weighted sum) is identical across both runs. The only
difference is which kernel `ggml_mul_mat_id` dispatches to.

## Why we suspect a kernel gap

`vendor/ggml/tests/test-backend-ops.cpp::test_mul_mat_id` only
exercises `ggml_mul_mat_id` with F32, F16, and Q8_0 source weights
(per the test registrations around lines 8264–8439 of that file).
K-quants are not in the test matrix. Our M2.3 finding suggests
they don't actually work for mul_mat_id even though the kernel
dispatch presumably accepts them — output is garbage rather than
a hard error.

This is consistent with:
- ggml's mul_mat (non-id) DOES support K-quant sources on the
  same code paths we hit (regular per-head matmul, K-cache reads).
  OLMoE's attention weights are also Q4_K and they work fine.
- mul_mat_id's expert-indexing logic appears to mishandle the
  K-quant block layout. The legacy quants (Q4_0 / Q5_0 / Q8_0)
  have a simpler per-block scale; K-quants have a hierarchical
  scale + per-sub-block offset that requires different per-block
  unpacking.

## Workaround (HISTORICAL — no longer needed)

Before the `head_nbytes` fix the workaround was "use Q8_0 for MoE expert
weights." That is obsolete: K-quant MoE (incl. OLMoE Q4_K_M with its mixed
q4_K+q6_K `down_exps`) now loads and runs coherently. The sections below are
kept for the historical record of the (wrong) kernel-gap hypothesis.

## Reproducing

Pure-Ruby repro is impractical because we'd need to generate Q4_K
data offline (the block layout has a hierarchical scale + per-sub-
block offset structure that's not trivial to synthesize). The
real-model repro:

Prefer the **token-ID-level** repro: it is decode-independent (the text path
has a separate space-dropping bug, toy#34, that mangles the rendered string).
Same prompt ("The capital of France is" → ids `510 38479 1171 33639 261`),
same graph, greedy decode; **only the expert tensor dtype differs**:

```bash
# Q4_K_M experts — degenerate (token 20065 = "Dub", repeated)
toy infer data/OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf \
  --prompt-ids '510 38479 1171 33639 261' --n 10
# ids: … 261 | 20065 20065 20065 20065 20065 42859 42859 42859 42859 42859

# Q8_0 experts — coherent / varied
toy infer data/OLMoE-1b-7b-0924-Instruct-q8_0.gguf \
  --prompt-ids '510 38479 1171 33639 261' --n 10
# ids: … 261 | 1373 1171 33639 15 11368 3 2303 23604 22473 1138
```

(The text form `--prompt '…'` shows "Dub Dub…" only when the tokenizer decode
is intact; on current `main` toy#34 strips spaces, so compare IDs.)

## Upstream

Reported to ggml-org/ggml:
<https://github.com/ggml-org/ggml/issues/1506> (filed 2026-05-24).

2026-06-03: a maintainer couldn't reproduce on M1 Metal + current master. We
re-confirmed it at the ID level on **aarch64 CPU @ ggml `41e7949`** and noted
the backend/arch difference.

2026-06-04: the maintainer (@devYRPauli) followed up with thorough op-level
isolation — `test_mul_mat_id` at OLMoE's REAL topology (`n_mats=64, n_used=8`,
Q2_K–Q6_K + IQ) on **CPU *and* Metal**, all pass on master; opened #1518 adding
the OLMoE-sized shapes. This supersedes our "no CPU fix landed" note (their CPU
test passes). Likely either (a) a fix landed between `41e7949` and master, or
(b) the corruption is in our FFI/graph layer, not the op. **NEXT (fresh session):
rebuild against current ggml master on aarch64 CPU + re-run the end-to-end OLMoE
Q4_K_M decode** — if coherent → fixed upstream, re-vendor + close; if it survives
only at the old pin → close (already fixed); if it survives on master → file a
minimal C reproducer (`ggml_mul_mat_id` direct, no FFI). Still OPEN.

2026-06-04 (op-level re-test — **bug is almost certainly OURS, not ggml's**):
There are **zero CPU-backend `mul_mat_id` changes** between our pin `41e7949`
and master `1e33fed` (85 commits; the `mul_mat_id` commits are all other
backends). So the CPU op is identical at both pins. A standalone op-level
reproducer (`/tmp/ggml1506_repro.c`, no FFI/no model: synthesize → quantize →
`ggml_mul_mat_id` on the CPU sched path at `n_mats=64,n_used=8,k=2048`, scattered
per-token ids, vs an F32 reference) shows **all quant types within normal quant
noise**, monotonic in bit-width: Q8_0 3e-5, Q6_K 2e-4, Q4_K 3e-3. No anomaly, no
gross corruption. **Yet the end-to-end OLMoE Q4_K_M decode STILL corrupts** at the
same pin (`… 261 | 20065×5 42859×5`). Clean op + corrupt end-to-end ⇒ the defect is
in **our loader / graph wiring** (realize/mmap of K-quant expert stacks, or a
Q4_K+Q6_K per-tensor mix our loader mishandles), NOT `ggml_mul_mat_id`. Posted to
#1506 (likely closeable on the ggml side). **NEXT: root-cause on the toy side** —
inspect how `realize_for_mmap` lays out the MoE expert tensors vs what the op
expects (block alignment / per-expert stride / mixed-type stack), and build a
toy-side op-test that feeds our mmap'd tensors directly.

**The runtime warning was MISATTRIBUTED and is now REMOVED.** It blamed ggml's
mul_mat_id kernel; the real cause was `head_nbytes` (per-head ATTENTION stride),
not the experts. Removed in the same change as the fix.

DONE (2026-06-05):
- `head_nbytes` generalized via `tnn_row_size` (handles K-quants); fails loud
  on a 0/invalid stride instead of silently collapsing heads.
- Misleading mul_mat_id-K-quant warning removed from `realize_for_mmap`.
- Verified: OLMoE Q4_K_M decode now coherent and tracks the q8_0 reference
  (`510 38479 1171 33639 261 …` → varied IDs, no `20065×N` degeneration).
- Regression-checked: Q8_0 mmap path, legacy-copy K-quant path (tinyllama Q4_K),
  and F32 path all still coherent.

Remaining nice-to-have: a committed smoke that loads OLMoE Q4_K_M and asserts
the first generated id ≠ the degenerate token.

## Coverage-doc cross-reference

`docs/coverage.md` shows `GGML_OP_MUL_MAT_ID` as `bound: yes` on all
three backends (CPU/CUDA/Metal). That's accurate at the *binding*
level. The kernel-quality story is the (op, src_type) axis we
explicitly noted in the strategic reframe — covered by THIS doc.

## 2026-06-05 — maintainer dump + our verification (NOT the per-role stride)

@devYRPauli dumped the OLMoE-1B-7B Q4_K_M expert metadata and pointed at the
mixed-quant `ffn_down_exps` as the likely role-stride trap. We confirmed the
mixed quant with our own metadata dumper (`tinynn/gguf_expert_dump.c`, ggml's
`gguf_init_from_file(no_alloc=true)`):

- `down_exps`: **8x q4_K (nb2=1,179,648) + 8x q6_K (nb2=1,720,320)** — MIXED across
  the 16 layers. `gate_exps`/`up_exps`: uniform q4_K. router (`ffn_gate_inp`): f32.
  Every tensor passes `nb[1]==ggml_row_size(type,ne0)` and `nb[2]==nb[1]*ne[1]`.

**But our loader does NOT make the per-role mistake.** `toy_smollm2_ffi_kv.rb`'s
MoE path reads `type`+`offset` PER TENSOR (`tnn_gguf_tensor_type`/`_file_offset`
= direct `gguf_get_tensor_type`/`_offset` on `blk.<li>.ffn_down_exps.weight`) and
hands `(ne0,ne1,ne2,type,offset)` to `ggml_new_tensor_3d` (ggml derives `nb[]`).
`@d_ff=1024` is OLMoE's expert FFN length (`olmoe.feed_forward_length`), so the
shape is right too. So type+offset+shape+nb are all correct per tensor — it's
not the role-stride bug for us, and our op reproducer + the maintainer's
`test_mul_mat_id` both run the op clean.

**Remaining suspects (narrowed):** (a) the mmap BYO-pointer path (real data via
`ggml_backend_tensor_alloc(buf, tensor, external_ptr)` — untested by the op
reproducers, though for a CPU backend `tensor->data=ptr` shouldn't change the op);
(b) the MoE GRAPH wiring (router->softmax->top_k->3x mul_mat_id->weighted-sum) or
an OLMoE-topology interaction. NEXT decisive test: run a `mul_mat_id` over the REAL
q6_K `down_exps` bytes vs an F32 dequant reference — if clean, the experts are
definitively fine and the bug is graph-side. ggml side OK to close.
