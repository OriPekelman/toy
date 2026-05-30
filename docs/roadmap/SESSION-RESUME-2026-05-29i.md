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

## P4 — CLI complete (IN PROGRESS)

**Slice 1 DONE — `toy infer` + the CRuby→runner COMPUTE BRIDGE** (4692c17):
- THE BRIDGE (reused by train/serve/eval): CRuby CLI can't compute, so
  `toy infer` ensures the Spinel runner is built (`make
  examples/example_inference` via the shared `lib/toy/core/toy_root.rb`
  root-locator — extracted + DRY'd with install), then `Open3`-shells it
  with a CONTROLLED env hash (GGUF/PROMPT/N_NEW — no stale-env leak).
  Matches the install/fetch shell-out precedent; "Core-no-ggml" intact.
- `toy infer <model.gguf> --prompt X [--n N] [--json]`, greedy/argmax
  (deterministic). Registered in COMMANDS (manifest/help auto-include).
  Clean errors (no file → 1, no args → 2, no backend → fail-loud hint).
- RUNNER is TRANSITIONAL (B): shells the existing example_inference;
  01_inference.rb KEPT (it IS the runner) — deletion deferred to the
  P4 cleanup arc. The long-term shape is a lib-side Spinel entrypoint.
- GATE: `prep/infer_gate.rb` — `toy infer` reproduces example_inference
  BYTE-FOR-BYTE (verified: ids 6403 1980 253 … on smollm2-135m, greedy).

**Slice 1 NOTE:** the build agent committed cleanly then hit the benign
harness "no StructuredOutput" error (4th time); commit + gate verified
by hand.

**Slice 1b DONE — CLEAN lib-side runner** (af3c86f, user asked for clean
now not transitional): the RUNNER PATTERN all P4 commands follow —
- SOURCE: `lib/toy/run/<cmd>.rb` (Spinel entrypoint; reads config from a
  controlled ENV the CLI sets; lib-vs-example scope, no baked config).
- BINARY: `libexec/toy-<cmd>` (gitignored build artifact).
- BUILD: Makefile rule `libexec/toy-<cmd>: lib/toy/run/<cmd>.rb <deps>
  tinynn/libtinynn_ggml.a | libexec` via `$(SPINEL)`.
- CLI: `lib/toy/core/cli/<cmd>.rb` builds it (toy_root + make) then
  Open3-shells it with the controlled env. Greedy/deterministic.
- GATE: recorded-baseline (`prep/fixtures/*` + `prep/<cmd>_gate.rb`),
  byte-for-byte.
infer: `lib/toy/run/infer.rb` → `libexec/toy-infer`; `01_inference.rb`
RETIRED; gate passes byte-for-byte (2 fixtures); verify-mirrors green.
NOTE: runner is CPU-ONLY — GPU deferred (CUDA/Metal inference use
hand-written ToyLMCuda/ToyLMMetal with different ctor arity → can't
mechanically mirror; no `--device` flag yet). `01_inference_metal.rb`
KEPT as the GPU path until a `--device` slice. Bonus: the new runner
handles the tok/text fixture the old example SEGFAULTED on (Spinel
tokenizer-GC regression).

**Slice 2 DONE — `toy train from-scratch`** (a5d3ec4): clean runner
`lib/toy/run/train.rb` → `libexec/toy-train` composing the FromScratch
recipe; `toy train from-scratch [--steps N] [--seed S] [--out D] [--json]`
resolves run-id from toy.yml → `runs/<id>/` (mkdir), runner emits valid
`events.jsonl` (run_start/step×N/run_end) + `weights/step_N.gguf` + `latest`
symlink. GATE byte-for-byte vs smoke_recipe_from_scratch loss curve
(prep/train_gate.rb). CPU-only (out of MIRRORABLE like infer);
verify-mirrors green; manifest lists 7 cmds.

**KNOWN GAP (tracked follow-up): train→infer round-trip does NOT close.**
The checkpoint GGUF uses training-graph tensor naming → NOT loadable by
`toy infer` yet (same per-head-vs-fused-name theme as the GGUF round-trip
gate). The train gate asserts checkpoint EXISTENCE only, no reload. A
"train→infer round-trip" gate (write loadable names / a fuse-on-save) is
the fix — important for "train then run your model" + a clean stable Toy.
Minor doc nit: docs/events-schema.md uses ".runs/" vs the "runs/" used
everywhere else — align in a doc pass.

**Slice 3 DONE — `toy eval`** (eaa75de): clean runner `lib/toy/run/eval.rb`
→ `libexec/toy-eval` (CE/logprobs via toy_logprobs), `toy eval <model.gguf>
[--top-k N] [--json]`, gate byte-for-byte vs smoke_decode_logprobs
(prep/eval_gate.rb). CPU-only; verify-mirrors green; manifest = 8 cmds.
ROADMAP NOTE: slice 1 is CE/logprobs; the roadmap's named eval gate is
`toy eval lmc` (two-ckpt) → eval slice 2.

**P4 COMMANDS COMPLETE: infer ✓ train ✓ eval ✓ serve ✓ (all gated, on main).**

serve landed (ba5bc54, FF'd to main): endpoint logic MOVED from
tep_demo/openai_api_llama.rb into `lib/toy/serve/openai/{server,handlers,
api_json,embeddings_handler}.rb` + runner `lib/toy/run/serve.rb` →
`libexec/toy-serve`; `toy serve <model> [--port N] [--name NAME]`
foreground-exec (persistent). HTTP gate (prep/serve_gate.rb): POST
/v1/completions fixed IDs → byte-identical to baseline recorded from the
OLD openai_api_llama; guaranteed teardown (ensure+at_exit+traps, kills
pgroup), verified no leak. tep build-dep ONLY (Tep untouched). CLI = 9
commands. All 4 gates (infer/train/eval/serve) green at HEAD.

**DEFERRED (post-P4): cleanup arc (NEXT), then:** eval lmc (slice 2,
two-ckpt), train warm-start/lora/curriculum variants, train→infer
checkpoint round-trip (naming), GPU runners (--device), Tep/Tao
re-adaptation (until Toy stable).

**NEXT — the CLEANUP ARC (user mandate):**
- Retire superseded `examples/` (01-09 etc. now reproduced by CLI cmds)
  + `tep_demo/openai_api_llama.rb` (folded into lib/toy/serve/openai/).
  **PRESERVE the gate/mirror fixtures:** `smoke_projection_lens` (in
  MIRRORABLE), and the smoke_recipe_*/smoke_gate_*/smoke_gguf_roundtrip
  fixtures + `prep/fixtures/*` (gate re-recording). Deleting wholesale
  breaks verify-mirrors + the gates.
- Shrink the Makefile (drop what the CLI replaced).
- **DOCS: FRESH START** — new clean-slate docs tree, PORT worthwhile
  content (don't refactor the sprawl).
- **AUDIT all markdowns/design-docs** — keep only still-relevant,
  INCLUDING live future-directions; prune the rest.
- Goal: clean-state repo.

**TEP CONSUMPTION CLEANED UP (2dea006):** toy now consumes tep as a gem
from `main` via `git:` (Gemfile) through the spinelgems convention — NO
@TEP_*@ trick. `prep/post_vendor_tep.rb` DELETED (it consumed the dead
ffi_manifest #97; tep main ships spinel-ext.json #98, and spinel-compat
vendor wires C-exts natively). Makefile `vendor-tep` = `bundle lock` →
`spinel-compat vendor`. Verified: `make vendor-tep` → openai_api_llama
builds+links clean. Notes: `bundle` needs a user Ruby env (rbenv/rv;
doc-only); tep's optional pg C-ext opted out (SPINEL_EXT_DISABLE=pg) —
filed **spinelgems#8** (libpq cflags not wired to split @TEP_PG_*@
entries). So `serve` is now buildable — re-running its workflow.

**serve BLOCKER (confirmed 2026-05-30):** `make tep_demo/openai_api_llama`
fails to link — `vendor/spinel/tep/lib/tep/net.rb` is STALE: it lacks the
`ffi_lib "ssl"` + `ffi_lib "crypto"` markers that upstream `~/sites/tep/
lib/tep/net.rb:15-16` added for inbound TLS. Upstream's sphttp.o was
rebuilt with `TLS_server_method`/`SSL_accept` (OpenSSL 3.0), but the
vendored net.rb wasn't re-synced → Spinel links the TLS object without
`-lssl -lcrypto` → `undefined reference to TLS_server_method@@OPENSSL_3.0.0`.
Fix = RE-VENDOR tep (re-run the vendoring so net.rb regains ssl/crypto +
re-points sphttp.o). That's a Tep-sync chore + upstream tep is mid-flux
(TLS just landed). infer/train/eval are tep-free and unaffected.
DECISION (pending user): serve is the Tep-coupled command → defer it (and
the tep_demo→lib/toy/serve fold) to the Tep phase, and proceed now with
the tep-free STABILIZATION work (train→infer round-trip + cleanup +
fresh docs + markdown audit). OR re-vendor tep now to finish serve.

`serve` recon: tep_demo/openai_api_llama.rb is a Spinel HTTP server using
`Tep::Handler` (requires vendor/spinel/tep + ../vendor/spinel/deps),
token-IDs-in/IDs-out, backed by Toy::SmolLM2KVFFICache. Building toy serve
USES tep as a build-dep (NOT the deferred "Tep re-adaptation" = Tep
consuming toy). serve is a PERSISTENT process → different gate shape
(start server → POST /v1/completions fixed IDs → deterministic IDs back →
stop), not the one-shot run-twice-diff.

**CLEANUP-ARC TENSION (must preserve):** the gate harnesses now use
RECORDED prep/fixtures/*.txt, but `make verify-mirrors` still MIRRORS
`examples/smoke_projection_lens` (it's in MIRRORABLE), and re-recording any
gate baseline needs the smoke_recipe_*/smoke_gate_*/smoke_gguf_roundtrip
fixtures. So the aggressive "delete all examples/" cleanup MUST preserve
(or migrate to test/) the gate-relevant fixtures + their MIRRORABLE entries
— deleting wholesale breaks verify-mirrors + gate re-recording.

Deferred: `toy eval lmc` (slice 2), train warm-start/lora/curriculum
variants, train→infer checkpoint round-trip, GPU runners (--device).
**Tep then Tao re-adaptation DEFERRED until Toy fully stabilized.**

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
