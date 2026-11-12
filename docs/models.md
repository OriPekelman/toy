# Supported models

The canonical "what's actually wired" reference is
[`docs/coverage.md`](coverage.md) (per-op coverage vs PyTorch). This page
is the model-level view: which checkpoints run, in which precision, on
which backend — and the footnotes that keep the table honest.

> **Verified at:** toy `d605aea` / spinel `a699cf9` / ggml `41e7949`
> on 2026-06-11 (toy#61 re-verification pass; serve leg at spinel
> `f6d5eef`, see spinel-dev#14). Every ✓ below was re-run on that
> date on the gx10 (GB10, aarch64) except where a footnote says
> otherwise. Greedy `toy infer` decode, CUDA asserted token-identical
> to CPU; GPT-2 family additionally parity-checked against recorded
> HuggingFace/PyTorch logits.

| Model              | Family               | Params      | F32 | Q8_0 | CPU | CUDA F32 | CUDA Q8 | Metal F32 |
| ------------------ | -------------------- | ----------- | --- | ---- | --- | -------- | ------- | --------- |
| DistilGPT-2        | GPT-2                | 82M         | ✓   |      | ✓   | ✓        |         | ‡         |
| GPT-2 small        | GPT-2                | 124M        | ✓   |      | ✓   | ✓        |         | ‡         |
| SmolLM2-135M       | Llama family         | 135M        | ✓   | ✓    | ✓   | ✓        |         | ✓◇        |
| SmolLM2-360M       | Llama family         | 360M        | ✓   |      | ✓   | ✓        |         | ‡         |
| TinyLlama-1.1B     | Llama + SPM-BPE tok  | 1.1B        | ✓   | ✓    | ✓   | ✓        |         | ‡         |
| Llama-3.2-1B       | Llama family         | 1B          | ✓   |      | ✓   | ✓        |         | ‡         |
| Llama-3.2-3B       | Llama family         | 3B          | ✓   |      | ✓   | ✓        |         | ‡         |
| Mistral-7B-v0.2    | Llama + SPM-BPE tok  | 7B          | ✓   | ✓    | ✓   |          | ✓       |           |
| Qwen2.5-0.5B       | Llama + QKV bias     | 0.5B        | ✓   | ✓    | ✓   | ✓        |         | ‡         |
| Qwen2.5-1.5B       | Llama + QKV bias     | 1.5B        | ✓   | ✓    | ✓   | ✓        | ✓†      | ‡         |
| Qwen2.5-3B         | Llama + QKV bias     | 3B          | ✓   | ✓    | ✓   | ✓        | ✓†      | ‡         |
| Qwen2.5-7B         | Llama + QKV bias     | 7B          | ✓   | ✓    | ✓   |          | ✓       |           |
| Qwen3-0.6B         | Qwen3 + QK-norm      | 0.6B        | ✓   |      | ✓   | ✓※       |         | ‡         |
| Qwen3-1.7B         | Qwen3 + QK-norm      | 1.7B        | ✓   | ✓°   | ✓   | ✓※       |         | ‡         |
| Qwen3-4B           | Qwen3 + QK-norm      | 4B          | ✓   |      | ✓   |          |         | ‡         |
| OLMoE-1B-7B-Instr  | Llama + MoE + QK-norm⁂  | 7B (1B act) |     | ✓    | ✓   |          |         | ‡         |
| Gemma 2-2b-it      | Llama + Gemma extras⁂⁂  | 2B          |     | ✓    | ✓   |          |         | ‡         |

⁂ OLMoE uses OLMoE-style QK-norm (gamma at `[d_model]`, per-head
sliced) and 64 experts × top-8 routing. MoE expert weights **must be
Q8_0** — Q4_K / Q5_K / Q6_K trigger a known ggml `mul_mat_id` bug
([ggml#1506](https://github.com/ggml-org/ggml/issues/1506)). The loader
emits a runtime warning if expert weights are K-quantized.

⁂⁂ Gemma 2 needs four extras: embedding scaling by sqrt(d_model),
logit soft-cap (tanh) on attention + final logits, pre+post-norms on
each sublayer, and per-layer alternating SWA/full attention. All
auto-detected from the GGUF (`gemma2.*` metadata). Its tokenizer is
SPM-Unigram (no merges), enabled automatically.

† These cells used to read "aborts on CUDA at weight-load time
(ggml-cuda quantized matmul required `d_ff` aligned to 512)". At the
current ggml pin (`41e7949`) that restriction no longer reproduces:
Qwen2.5-1.5B/3B Q8 run on CUDA and decode token-identical to the CPU
Q8 and F32 paths (re-verified 2026-06-11).

One case of it did remain until toy#205: a file whose **last** tensor
is quantized with a row width that is not a multiple of 512 aborted at
load on CUDA (`ggml-backend.cpp: GGML_ASSERT(… get_alloc_size …)`).
ggml-cuda may read up to one padding row segment past such a tensor;
mid-file that is the next tensor, at the end of the file it was past
the mapping. Qwen2.5-0.5B Q8 (`d_ff` 4864) and a pruned model with
`d_ff` 9472 hit it. Every GGUF is now mapped with a
zero tail, and these files load on CUDA and decode the CPU's ids
(`make gate-q8-tail-cuda`).

**Open (found 2026-11-12, not fixed):** SmolLM2-135M converted with
`--ggml-native --quantize q8_0` loads on CUDA but decodes different ids
from the CPU, which agrees with F32. The legacy-layout Q8 of the same
model is correct on CUDA, and so is the native file with any single
tensor quantized, so the Q8 ✓ in the SmolLM2 row holds for the legacy
file only. The runner from before toy#205 behaves the same. Cause not
yet known.

※ These cells used to read ✗: Qwen3 CUDA decode was degenerate while
CPU was coherent (caught by the 2026-06-11 re-verification, toy#76).
Root cause was toy-side, not a ggml kernel: the hand-written CUDA
loader (`lib/toy/models/transformer_lm_cuda.rb`) never wired the
detected `qk_norm` flags through to `realize_for_mmap` — it passed 5
of the 6 args and Spinel zero-fills missing call args without a
diagnostic — so the per-head Q/K RMS-norms were never built on the
CUDA graph. Fixed in the #76 PR; re-verified 2026-06-11 (spinel
`a699cf9` / ggml `41e7949`, gx10): 0.6B and 1.7B greedy decode
token-identical to CPU across prompts, `toy eval` top-5 ids identical,
and the KV_Q8 / FLASH_ATTN opt-ins parity-hold on CUDA too.

° Qwen3-1.7B Q8 was not re-verified on 2026-06-11: no Q8 checkpoint of
it on the bench box (re-quantizing needs an HF download we don't keep).
The ✓ is carried over from the original v0.5 validation.

‡ Metal: validated end-to-end only on SmolLM2-135M F32 (bit-identical
to CPU). Other Llama-family models should work — same FFI surface,
same graph builder, same ggml-metal kernel coverage — but haven't been
smoked. GPT-2-family isn't wired on Metal. Quantized weights on Metal
are untested. The Metal path uses copy-load rather than mmap, so
multi-GB models pay the copy cost — and the copy-load path has no
QK-norm support, so QK-norm models (Qwen3, OLMoE) abort loudly on
Metal rather than decode degenerate (#76).

◇ The SmolLM2-135M Metal ✓ is Mac-gated (#27). Re-confirmed on
2026-06-12 on Apple M2 (toy `d5ffbc5` / spinel `a699cf9`, pinned per
the toy#27 brief; vendored ggml `57ea0bc`): `make
example_inference_metal` builds + runs end-to-end (Metal kernels JIT,
fusion/concurrency on), F32, deterministic ids — identical to the
prior 2026-05-31 Mac validation (`9b5131a`).

## Which GGUF files load

Three kinds of file are accepted by the llama-family runners (`toy infer`,
`toy eval`, `toy serve`), and the loader tells them apart itself:

| Written by | Metadata keys | Weight layout | How it is recognised |
| --- | --- | --- | --- |
| `prep/convert_smollm2_to_gguf.py --ggml-native` (and toy's own checkpoint writer) | `llama.*` | ggml-native | `toy.ggml_native = true` |
| the same converter without `--ggml-native` (legacy) | `llama.*` | transposed | no flag, and `blk.0.ffn_gate.weight` stored `[d_ff, d_model]` |
| **llama.cpp `convert_hf_to_gguf.py`** / `llama-quantize` | `qwen2.*`, `qwen3.*`, `llama.*`, … | ggml-native | no flag, and `blk.0.ffn_gate.weight` stored `[d_model, d_ff]` |

llama.cpp-converted files are a supported input as of toy#203. Before
it they failed twice over: the `qwen2.*` keys were not probed, so every
dimension read back `-1`; and layout was decided by the flag alone, so
such a file took the legacy transposing path and decoded wrong tokens
with exit 0. Layout is now decided by the stored shape of
`blk.0.ffn_gate.weight` (the flag still wins when present), in one
place — `tnn_gguf_is_native_layout` in the shim.

Verified on the gx10, CPU, greedy decode, against the toy-converted Q8
of the same weights (a Qwen2.5-7B pruned to 22 layers / d_ff 9472,
3.98 B): the llama.cpp **Q8_0** (4.23 GB, embeddings and head quantised
too) and **Q4_K_M** (2.48 GB) both produce identical ids. `qwen3.*`
resolves but has not been run from a llama.cpp file. `make
gate-llamacpp-gguf` keeps this honest on a manufactured fixture, CPU;
the same gate passes against `toy-infer-cuda` on the small F32 fixture.
Quantised weights on CUDA are a separate question (toy#205).

Which architectures resolve is an explicit list in the shim
(`tnn_gguf_arch_prefix`: llama, olmoe, gemma2, qwen3moe, qwen2moe,
deepseek2, qwen2, qwen3), not "whatever `general.architecture` says" —
a GPT-2 file has perfectly valid `gpt2.*` dimensions and no llama-family
graph. Anything else is refused before a graph is built, with the
architecture named:

```
Arch.from_gguf: model.gguf: unsupported architecture — general.architecture=gpt2, ...
```

and a file whose prefix resolves but whose `embedding_length`,
`block_count`, `attention.head_count` or `feed_forward_length` is
missing is refused the same way, listing the four values.

## Converting a checkpoint

`prep/convert_smollm2_to_gguf.py` converts a llama-family checkpoint
(Llama, SmolLM2, Qwen2.x, Qwen3) from a HuggingFace repo or from a local
directory:

```
uv run --script prep/convert_smollm2_to_gguf.py \
    --model-dir /path/to/checkpoint --ggml-native --with-tokenizer \
    --quantize q8_0 --quantize-embd --out model-q8.gguf
```

- **`--model-dir`** reads `config.json`, `model.safetensors` (or
  `model.safetensors.index.json` and its shards) and, with
  `--with-tokenizer`, `tokenizer.json`, `tokenizer_config.json` and
  `chat_template.jinja`. Nothing is downloaded, so a pruned or
  fine-tuned model converts without being published. Shapes come from
  the config. `--repo-id` is the HuggingFace path, as before.
- **RoPE base.** Read from top-level `rope_theta`, or from
  `rope_parameters.rope_theta`, which is where transformers 5 writes it.
  A config with neither is refused; `--rope-theta <base>` states it.
  Before toy#204 the second form was not read and the file got
  `rope.freq_base = 10000`, which decodes plausible wrong text.
- **`--quantize q8_0|q4_0|q5_0`** quantizes the 2-D weights, including
  an untied `output.weight`. `token_embd.weight` stays F32 unless
  **`--quantize-embd`** is also given.

On the 3.98 B student above, `--quantize q8_0` gives 5.83 GB and adding
`--quantize-embd` gives 4.23 GB, the size of the llama.cpp Q8_0. 266 of
its 267 tensors are byte-identical to the llama.cpp file; `token_embd`
differs only in blocks whose scale is zero (never-trained rows), which
dequantize to zero in both. Greedy ids are the same from both files, on
CPU and CUDA. The conversion holds the whole checkpoint in memory: about
20 GB resident for that 8 GB BF16 checkpoint.

`make gate-converter` covers the above on a tiny random checkpoint built
at gate time.

## Tokenizer / RoPE coverage

Three tokenizer flavors are auto-detected from the GGUF:

- **Byte-level BPE** (GPT-2 byte-map): GPT-2 family, SmolLM2,
  Llama-3.2, Qwen2.5, Qwen3.
- **SPM-BPE** (SentencePiece with merges): TinyLlama-1.1B, Mistral.
- **SPM-Unigram** (SentencePiece with scores, no merges): Gemma 2.
  Greedy longest-match; auto-prepends BOS when
  `tokenizer.ggml.add_bos_token` is set.

RoPE scaling (YaRN / llama3-style / linear) propagates through the
converter and loader, so Llama-3.x and Qwen3 long-context
configurations work without extra wiring.

## Opt-in performance features

Environment variables the runners read:

- **`KV_Q8=1`** — K and V caches stored as Q8_0 (3.75× smaller),
  auto-enables flash attention. Validated on Qwen3-1.7B: 13% faster
  wallclock at N_NEW=32, bit-identical token output.
- **`FLASH_ATTN=1`** — fused `ggml_flash_attn_ext` per Q-head. 12%
  faster on Qwen3-1.7B at N_NEW=32 vs the scale→softmax→matmul triplet.
- **`TOY_EVENTS=path.jsonl`** — emit the structured `toy/v1` event
  stream for live TUIs and post-run analysis (see
  [`events.md`](events.md)).
