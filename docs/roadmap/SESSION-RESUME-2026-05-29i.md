# Session resume — 2026-05-29 (i): P2 COMPLETE (to the gated ceiling); P3 (Core + CLI MVP) next

> Supersedes (h). The P2 five-layer refactor is DONE: L1–L4 landed, and
> the realize bulk is decomposed to the approved 5-gate ceiling (3/4
> paths). Moving to P3 — the user-facing CLI. P3 is greenfield feature
> work with a FUNCTIONAL gate (commands work), not the bit-identical
> gating of P0–P2.

## P2 — COMPLETE (gated)

| Layer | Unit(s) | Gate |
| --- | --- | --- |
| L1 | RoPE, SwiGLU, RMSNorm, GQA | bit-identical CPU |
| L2 | TransformerBlock | bit-identical CPU |
| L3 | LlamaArch (`build_forward` + globals alloc) | CPU+CUDA |
| L4 | FromScratch, LoRA, WarmStart recipes (Curriculum deferred) | bit-identical loss curve |
| realize | 5 shared + per-block loaders for random_init / mmap / q8 | 7-way gated |

**Monolith `lib/llama_seq_forward_ffi.rb`: 1918 → 1333 lines.** Per-block
weight loading for 3 of 4 realize paths now lives on `TransformerBlock` /
`LlamaArch`. Everything verified bit-identical (independent A/B each
step); all on `main`, pushed to both remotes.

## Gates built this effort (all deterministic, committed)

`examples/`: smoke_projection_lens[_cuda] (random_init CPU/GPU),
smoke_gguf_roundtrip (mmap-F32, via `lib/toy_gguf_fuse.rb`),
smoke_gate_{gqa_divergent, b_gt_1, qkv_bias, q8_preserve} (config
variants), smoke_gate_llama3_tensor (RoPE primitive),
smoke_recipe_{from_scratch, lora, warm_start}.

## Realize residual — DOCUMENTED CEILING (user accepted, not pursuing)

3/4 realize paths decomposed. NOT decomposed:
- **full_finetune** (path 4) — left INLINE (`lib/llama_seq_forward_ffi.rb`
  ~462-648 + ft_load_from_gguf ~1008-1072). Works; just not lifted onto
  the block. A 6th gate is CONSTRUCTIBLE (demos/smollm2_seq_full_finetune)
  if ever wanted — user chose to accept the ceiling instead.
- **llama3 rope_freq_factors realize-wiring** — genuinely UN-GATEABLE: the
  toy forward is rope-ANGLE-insensitive at the logit level (finding —
  worth a separate look someday), and the tensor gate only covers the
  standalone RoPE primitive.
- **GQA-divergent on mmap/q8** — no divergent-head GGUF; the round-trip
  pins non-divergence. Kept w_o hard-square verbatim (never unified with
  random_init's divergent shape).

## Findings parked for later

- **Rope-angle insensitivity** of the full toy forward at logit level
  (llama3-vs-none, base 500k-vs-10k, positions i-vs-2i all = 0 logit
  diff; primitive IS sensitive in isolation). Either a tiny-config/
  random-weight artifact or a latent forward bug. Worth a dedicated
  investigation; out of scope here.
- **Spinel matz/spinel#1043** (Struct.new accessor type-merge) filed;
  **ggml#1506** (mul_mat_id K-quant) noted for MoE. See memory.

## P3 — Core + CLI MVP (IN PROGRESS)

**Slice 1 DONE** (6ca853f + fix bfa040b): `bin/toy` (CRuby, 11 files, zero
Spinel) — `new <path> [--force] [--json]`, `list [--json]`, `describe
<model.gguf> [--json]`, global `--manifest`/`--help`/`--version`. Exit
codes 0/2/1. `toy.yml` loader (run_id_template + algos_path; empty=defaults,
unknown keys warn). JSON tags `toy/{manifest,list,describe,new}-v1`.
gemspec → `bin/["toy"]`. Pure-CRuby GGUF metadata reader
(`lib/toy/core/gguf_meta.rb`) + scan (`model_scan.rb`) + Card via
`toy_card.rb`. All commands verified running on gx10; clean errors
outside a project. **Correctness fix:** describe now reads
`general.architecture` and DECLINES non-llama arches (gpt2/gemma2/olmoe)
instead of rendering a wrong llama Card; list shows true families.

**Slice 2 DONE** (98e50c8): `toy fetch <repo> [<file>] [--json]` (HF
download via hf/huggingface-cli/curl → cache → relative `data/` symlink;
verified with a real 19MB download, `toy list` sees it) + `toy install
[--json]` (locates toy root via TOY_HOME / dev-checkout walk-up / gem
dir, verifies-or-builds the CPU backend, fail-loud if none; verified on
aarch64) + cleanup arc (deleted 05_list_models, 04_serve_http,
fetch_model.sh; removed `make hello` + FIRST-TIME banner; retargeted docs
to `toy` CLI). `make verify-mirrors` still passes; surviving targets
resolve. `--manifest` lists all 5 commands.

**P3 = COMPLETE on gx10 (aarch64) + Mac (M2/Metal).** All 5 commands +
--manifest/--help/--version + toy.yml work on both. Mac validation pass
(2026-05-30): all 5 commands pass on M2, Metal install+inference verified
end-to-end, 2 bugs fixed (0949885 install Metal+source-sentinel, 2153294
binstub fail-loud), issue #29 closed.

**Packaging (#28) RESOLVED → option (c)** (dcb052e): gemspec ships the
backend build inputs (Makefile, vendor-patches, prep filters, tinynn
CPU/Metal/CUDA shim sources). Verified aarch64: gem-install → `toy
install` from /tmp locates the gem dir, clones+patches+builds ggml. #28
closed; fat-gem (a) deferred. **Dedup polish** (f302fe1): `toy list`
dedups by realpath so a fetched model shows once.

**Remaining for the full 3-platform P3 gate:** only **Linux x86_64** is
unverified (gx10=aarch64, Mac=arm64-darwin both pass). Verify on an x86_64
Linux box if the full gate matters, else accept 2/3. Then → P4.

### (earlier) Remaining for the P3 GATE:

1. **OPEN DESIGN QUESTION — `toy install` gem-packaging** (user decision):
   `gem install toy` yields NO backend today. install currently works via
   TOY_HOME / dev-checkout / gem-dir locate-and-build. The packaging story
   is unresolved: (a) fat gem ships prebuilt per-platform `.a`; (b) gem
   builds backend on install (native-ext/post-install); (c) gem ships
   sources + `toy install` builds (closest to current). This MUST be
   settled for the Mac/cross-platform gate to pass.
2. **Cross-platform fresh-checkout verification** (macOS-AS + Linux-x86_64)
   — the USER's separate task (only aarch64 verified here).
3. **Minor polish:** `toy list` shows a fetched model TWICE (data/ symlink
   + HF cache original); dedup should canonicalize symlinks (File.realpath).

After P3 gate → **P4 (train / serve / infer / eval CLI commands).**

### Original P3 spec (reference)


**Deliverable (roadmap §P3):** `bin/toy` as CRuby (NOT Spinel-compiled).
Five commands: `new`, `install`, `list`, `fetch`, `describe`. `toy new`
creates the conventional dir tree. `toy --manifest` (JSON for CC). Empty
`toy.yml` honoured (run-id template + algo-discovery path; backend
auto-detected).

**Gate (functional, not bit-identical):**
- `toy new myproj && cd myproj && toy install` end-to-end on fresh
  checkout (macOS AS + Linux x86_64 + Linux aarch64).
- `toy fetch …` drops `data/<basename>.gguf` symlinks.
- `toy list` finds HF / Ollama / LM Studio caches + project `data/`.
- `toy describe <model>` reads GGUF metadata + renders the arch's
  DERIVED Card (reuses P1 Card machinery — `lib/toy_card.rb`,
  `lib/toy_describe_flow.rb`, `lib/model_index.rb`).
- `toy --manifest` emits JSON CC can consume.

**Cleanup arc (in-phase, aggressive):** delete examples/05_list_models.rb
(→ `toy list`), examples/04_serve_http.rb (known-broken), fold
prep/fetch_model.sh into `toy fetch`, delete `make hello` + the "FIRST
TIME" banner.

**Layout (roadmap §):** `bin/toy` binstub; `lib/toy/core/` (CLI dispatch,
Card IR, registry, events). NOTE: P3 is CRuby + functional-gated — no
Spinel mirrors, no bit-identical gate. Reuse existing `lib/model_index.rb`
(`toy list`/`fetch` likely already have machinery there) + the P1 Card
derivation (`toy describe`).

**Risks (roadmap):** 5-command surface may be too constrained; project-
dir-vs-framework-dir edge cases must give clear errors, not stack traces.

**Suggested first move:** the CLI shell + `toy new` + `toy list`/
`describe` (which reuse existing model_index + Card machinery) before
`install`/`fetch` (more environment-coupled). Recon what
`lib/model_index.rb` + `prep/fetch_model.sh` + `examples/05_list_models.rb`
already provide so the CLI wraps rather than reinvents.

## Reference paths

- Layers: `lib/toy/llm/{primitives,blocks,archs,recipes}/`
- Monolith: `lib/llama_seq_forward_ffi.rb` (realize bulk + delegators)
- Card machinery (for `toy describe`): `lib/toy_card.rb`,
  `lib/toy_describe_flow.rb`, `lib/model_index.rb`
- Roadmap P3-P6: `docs/roadmap/toy-framework-roadmap-2026-05-28.md:245+`
- Design CLI/layout: `docs/roadmap/toy-framework-design-2026-05-28.md`
- Gate fixtures: `examples/smoke_*` + `lib/toy_gguf_fuse.rb`
- Landmines: `memory/feedback_spinel_type_inference_landmines.md` (#16)
  — NOTE: P3 CLI is CRuby, so Spinel landmines DON'T apply to bin/toy.
