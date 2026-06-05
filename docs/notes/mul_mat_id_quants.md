# `mul_mat_id` × K-quantized source weights — OLMoE Q4_K_M corruption

**Status (2026-06-05): NOT a ggml bug — it's ours, and narrowed but not yet
found.** Workaround stands: use Q8_0 for MoE expert weights. ggml#1506 is
closeable on the ggml side (maintainer agrees).

> **FRESH-SESSION RESUME — start here.** What's verified: the op is clean
> (`tinynn/ggml1506_mul_mat_id_kquant_repro.c`); our loader reads expert
> `type`+`offset`+`shape` PER TENSOR (`@d_ff=1024` is OLMoE's expert FFN length),
> so the per-role-stride bug @devYRPauli flagged does NOT apply to us; the MoE
> graph is identical between Q4_K_M (corrupt) and Q8_0 (coherent), so it's not a
> pure graph bug. See the **2026-06-05 section at the bottom** for the full
> verification + the `tinynn/gguf_expert_dump.c` metadata dump. **THE DECISIVE
> NEXT TEST:** a ~120-line C reproducer that runs `mul_mat_id` over the REAL q6_K
> `down_exps` bytes (read from the GGUF at `data_offset+tensor_offset`, attached
> via `ggml_backend_tensor_alloc` like our loader's mmap path) vs an F32 dequant
> reference (`ggml_get_type_traits(type).to_float`). Corrupt → real op bug w/
> specific data → minimal C repro for ggml. Clean → experts fine, hunt the
> weighted-sum / contiguity / mmap-attach path. Models: `data/OLMoE-1b-7b-0924-
> Instruct-{Q4_K_M,q8_0}.gguf`. Repro shape: n_mats=64, n_used=8, k=1024 (down).

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

## Workaround (current)

1. **Convert MoE models at Q8_0 or higher**. Avoid Q4_K_M, Q5_K_M,
   Q6_K for MoE expert weights specifically. Non-expert weights
   (embeddings, attention, norms) can stay at K-quants — only the
   MoE expert stacks need Q8_0.

2. **Runtime warning**: `realize_for_mmap` emits a warning when
   it detects MoE expert tensors with type ∈ {Q4_K, Q5_K, Q6_K,
   ...K_M / K_S / K_XS variants}. The model will still load and
   run, but output will be wrong. This makes the failure mode
   loud rather than silent.

3. **Choose your GGUF**: For OLMoE-1B-7B, `Meshwa/OLMoE-1b-7b-0924-
   Instruct-gguf` ships both Q4_K_M and Q8_0 variants. Use the
   Q8_0 (~7 GB, ~13 tok/s on gx10 CPU).

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

**The runtime warning is now MISATTRIBUTED.** `realize_for_mmap` says "ggml's
mul_mat_id kernel produces wrong output for K-quants" — the op-level reproducer
shows that's false. Reword it to "toy's K-quant MoE expert path produces wrong
output (cause under investigation; use Q8_0)" until the toy-side root cause lands;
do NOT remove it (the symptom is still live).

When the toy-side root cause is fixed:
- Update/remove the runtime warning in realize_for_mmap.
- Add a smoke that loads OLMoE Q4_K_M and confirms coherent output.

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
