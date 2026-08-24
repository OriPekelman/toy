# Changelog

## Unreleased

### Framework

- **Alternative credit assignment is now a framework capability.**
  `--policy chain|dfa|frozen` threads through the Llama engines on CPU,
  CUDA and Metal, with a documented feedback-matrix family
  (`--dfa-b-{seed,dist,scale}`, `--dfa-cut`, `--dfa-granularity`,
  `--dfa-feedback kolen-pollack`, the nDFA preconditioner and LDFA
  low-rank feedback). See [`docs/cli.md`](docs/cli.md).
- **New archs and primitives**: Gated-DeltaNet / KDA, Gated MLA, SiTU-GLU,
  Muon, MTP, LatentMoE, AttnRes — with gates for DeepSeek-V2 (MLA),
  Qwen3-MoE and Qwen2-MoE.
- **Wider training surface**: `--optimizer`, `--lr-schedule`,
  `--lr-control`, `--clip-grad`, `--ckpt-every`/`--load-ckpt`,
  `--eval-corpus`.
- **CUDA twin for the byte-LM path** (`--device cuda`, 4.7–5.8×), held to a
  stated cross-backend tolerance. Cross-device cells are **not** numerically
  comparable — a sweep runs on one device or re-runs its reference.
- **`gate-prereq`** — `libexec/*` targets are now checked against each
  runner's `require_relative` closure. 26 of 36 targets had drifted, so a
  change to an undeclared source rebuilt nothing and its gate passed against
  a stale binary. Two sibling instances of the same class were found and
  fixed: a vendored ggml patch every fresh clone silently dropped, and four
  CUDA mirrors `make` could never generate.
- **`make gates-fast`** — the battery in phases; `make -j gates` is unsafe
  (recursive makes race on one `libexec/` path) and `gate-run-log` must run
  alone (it scans the whole `runs/` tree and will read a bundle mid-write).

### Research fixtures and findings

The eleven fixture lanes (`mlp ctr gnn ssm lstm gtx diff difflm ae franken
franken-moe`) exist to test the capabilities above independently of any one
question. The programme that drove them (toy#152–#173) is concluded; the
record is in [`docs/research/`](docs/research/) and the lane reference in
[`docs/research/lanes.md`](docs/research/lanes.md).

#### Direct Feedback Alignment — the result

- **A POSITIVE, at last (toy#152).** `toy train mlp` — an MLP classifier
  on a seeded synthetic task, the anchor lane. At 2 classes DFA **matches
  BP** (0.942 vs 0.940 val accuracy); at 10 it is 3 points behind BP and
  23 ahead of the frozen control. Before this, every DFA result in the
  tree was negative and *could not be distinguished from a harness
  artifact*. Now they can: the harness reproduces a known DFA success.
- **The output-dimension lens, measured.** DFA's *recovery* — the share of
  BP's over-the-frozen-control gain it recaptures — falls monotonically
  1.01 → 0.88 → 0.49 → 0.42 across {2, 10, 100, 1000} classes, on every
  seed tried. The lens was a failure *explanation*; it is now a tested
  prediction, on the same feedback machinery the transformer lanes use.
- **The mechanism is visible.** With `--align-events`, cos(g_DFA, g_BP)
  climbs from ~0 to 0.67 over training — Refinetti's "align, then
  memorise" reproduced in our own telemetry, and the layer nearest the
  head aligns *least*. The gate asserts the climb: a mis-wired DFA build
  can still move a loss curve, but it cannot make the shadow gradient
  rotate towards a random matrix.
- **Macro DFA (toy#158).** `--dfa-granularity block` — LightOn's recipe:
  one random-feedback tap per block *output*, full BP inside the block,
  no backward chain across blocks. Built by cutting the residual stream
  at each boundary and attaching a surrogate loss root per block. Because
  the cut is forward-identity, step 1 is **byte-identical to BP** — which
  is what makes macro-vs-BP numbers comparable at all.
- **A NEGATIVE that matters (toy#154).** `toy train ctr` — embedding
  tables + MLP tower + scalar sigmoid head, scored by AUC. LightOn put
  DFA within ~0.01 AUC of BP on Criteo; **we do not reproduce that**.
  DFA clears the frozen control reproducibly (+0.019–0.024 AUC across
  seeds) but lands 6 points behind BP, not 1, and LR tuning is not the
  explanation. At output dim 1 the DFA update is provably **rank-1** —
  every layer can only move its output along one fixed direction — so
  "small output dim" is really two competing effects: fewer directions
  to align, but less information per step. The likely reconciliation is
  measurable: with DeepFM's FM branch enabled, a **frozen** tower
  scores as well as a trained one (0.813 vs 0.806), so near-parity on
  that architecture is not evidence that DFA trains towers.
- **`--optimizer radam` (toy#158)**, plus per-layer DFA reaching the FFN
  (`--policy-scope attn|ffn|all`, toy#151), adaptive Kolen-Pollack
  feedback (`--dfa-feedback`, toy#150), block-granularity DFA on the MoE
  lane (toy#143), and the frozen-expert control (toy#141).
- **Micro and macro are not comparable arms.** At the default
  `--policy-scope attn`, per-weight DFA policies only attention q/k/v, so
  most of the network is still backprop; macro policies the entire stack.
  Reading one as the other is a large part of why an early experiment
  "washed out". Both `dfa_granularity` and `policied_tensors` now ride in
  every run bundle so the confusion cannot recur silently.

#### The Kimi-K3 mechanism wave (toy#135–#147)

KDA (channel-wise delta rule with bounded decay), Gated MLA with a latent
KV sandwich, Attention Residuals over depth, SiTU-GLU, NoPE, Quantile
Balancing, Muon (per-head Newton–Schulz orthogonalization), Stable
LatentMoE with shared experts, Multi-Token Prediction where λ is a
*coupling* dial rather than a loss weight, per-layer LR schedules, and a
loss-reactive LR damper. Each is a flag on the research instruments, each
independently gated, each byte-null when off.

#### Research-instrument surface (toy#120–#134)

Spec-callable `toy train franken` / `franken-moe`; real corpora (TOYC
packs) with `--context`/`--vocab`/`--batch`; checkpoint writers and
`--ckpt-every`; `toy eval ce` for held-out cross-entropy; run-bundle cost
accounting; a CUDA twin of the MoE lane; the `--shape` presets; and the
**flag × recipe matrix** — one table, gate-asserted, after four
consecutive flags tripped at experiment runtime.

### Events / consumer contract

- `align` and `mask` event kinds are now **documented** in
  [`docs/events.md`](docs/events.md), along with the run_start
  credit-assignment provenance object.
- Align events carry **`wname`**, a stable cross-lane weight name, in
  every lane. `wi` is lane-local and does not travel between lanes;
  consumers key on `wname`.

### Discipline fixes worth naming

- **Every gate leg reports on its own assertions** (`d878143`) — legs used
  to summarise with the global failure list, so one early failure made
  every later leg print FAIL, exactly when you were debugging.
- **A held-out eval that was still training** (`d5e4c83`) — `--eval-corpus`
  ran at lr=0 for the *global* hp vector only, so under `--optimizer` and
  `--lr-schedule` the model kept training on its own test split.
- **An OOM that was a leak, not a size** (`0127836`) — the eval loop
  allocated a fresh label matrix per window; peak memory grew with window
  *count*, which is what made shrinking the eval look like a fix.

## v0.9.0 — 2026-06-22

**The Dragon / Gated-DeltaNet trainable hybrid arc.** toy grows a second block
type and the seam to stack it heterogeneously with attention — built phase by
phase, each independently gated.

- **Trainable GDN (Path B)**: the gated delta rule expressed as an *unrolled
  autograd composition* (`GDN.recur_unrolled`) of ops that each have a ggml
  backward — so a Gated-DeltaNet layer trains with **no hand-written kernel
  backward** (ggml has none for `GATED_DELTA_NET`); the fused kernel is kept for
  inference. Gated by forward-parity (`recur_unrolled` == fused kernel to 1e-6,
  incl. multi-head) + a differentiability proof.
- **L1 Dragon primitives**: `gdn` (l2/decay-gate/update-gate/recur/gated-out),
  `diff_attention`, `scalable_softmax`, `depth_scale`; 8 elementwise ggml ops +
  `tnn_gated_delta_net`/`tnn_conv_1d` wired (CPU-only this arc).
- **Per-layer `LayerSpec` seam**: a flat-int `seq_layer_kinds` dispatch (one arch
  loop, monomorphic per-kind block call) — byte-exact on homogeneous Llama
  (from-scratch / warm-start / lora unchanged).
- **`GDNBlock`** (L2) + **`libexec/toy-train-hybrid`**: a self-contained
  from-scratch **attention+GDN hybrid** trains (CE loss decreases). Folding it
  into the shared `toy train` engine is deferred behind a union-pin Spinel
  codegen block — re-apply protocol in `docs/roadmap/gdn-hybrid-engine-reintegration.md`.
- **Fixes**: `#1449` whole-program training abort (backward `get_rows` index OOB)
  root-caused as a latent ggml-alloc liveness bug and fixed toy-side
  (`tnn_input_1d_i32_persistent`, a galloc-external token-id index) — *not* a
  spinel codegen bug (matz closed it resolved); CUDA/Metal training restored by
  mirroring that FFI decl into the CUDA/Metal siblings. New backward-friendly
  shims `tnn_sqrt`/`tnn_div`/`tnn_repeat`.
- **Performance**: CPU inference ~+27% tok/s and LoRA steady-state ~−24% vs the
  v0.8.0-era baselines (heavy CUDA bench stable).

## v0.8.0 — 2026-06-12

**The first published version** (RubyGems, gem name graciously transferred
by Ninoslav Milenović — see README acknowledgments). The release of the
"first form right" arc (#58): toy is now a publishable dual-use framework —
a playable CLI and a clean library — with every README claim gate-tested.

Highlights of the arc (full trail: issues #58–#87):

- **The experimenter API** (#64/#73): `RecipeOptions` named setters replace
  the 8-positional `realize!`; validating `TrainingBatch` (+ objectives) and
  `ClassifyBatch`; `AdamW.for_from_scratch`/`.for_lora` + `hp_for_step`
  (misuse fails loud); `realize_warm!` owns donor plumbing; `Toy::RunBundle`
  writer + `Toy::RunLog` reader (`RunLog.scan("runs")` — best run in one
  line); config profile factories (`SmolLM2Config.tiny/.mid`,
  `ViTTinyConfig.tiny`); ENV-driven scaffolds (one compile, many runs).
- **Tree truth** (#65/#68): the layered stack is the whole library —
  `lib/` is exactly `toy.rb` + `toy/`; tinynn under `toy/ffi/`, loaders
  under `toy/io/loaders/`, KV decode + the lifted GPT-2 engines under
  `toy/llm/engine/`; docs teach the real six layers (L4 = Engine).
- **Compile-time devices** (#64/#70): `toy/compute` ·
  `toy/compute_cuda` · `toy/compute_metal` entries; `toy new --lib` ships a
  multi-arch `build.sh`; GPU vendor build-units (opt-in) validated through
  the consumer front door (CUDA byte-exact on GB10; Metal Mac-verified).
- **MRI dev-runs + the CRuby oracle** (#71): `require "toy/mri"` loads
  everything under plain Ruby (teaching surface works pure-Ruby); with
  `make libtinynn_shared`, the Fiddle arm runs the real compute path —
  **bit-exact vs the Spinel binaries on both Linux/aarch64 and Apple
  Silicon** (train curves and decode ids).
- **Curated examples + tested quickstart** (#60): 7 narrated examples
  replace the 60-entry sprawl; `make gate-consumer` proves the README
  cold-start (app + `--lib` legs) on Linux and macOS.
- **Coverage honesty** (#61/#76/#77): 17-model matrix re-verified with
  provenance stamps; Qwen3 CUDA fixed (below); Qwen2.5 CUDA-Q8 footnote
  upgraded; bench-vs-PyTorch legs restored (train 1.064×, infer 1.399×).
- **sig/*.rbs type roots** (#69): shipped in the gem and consumed at
  compile (`--rbs sig`) — uncalled-param poly-widening defense for toy and
  for every consumer.
- **lora in `toy/compute`** (#52): the exclusion dissolved with the #64
  reshape; the compute-surface gate carries an uncalled-LoRA tripwire.
- Upstream: eight Spinel issues filed from this arc (the silent-failure
  cluster spinel-dev#17/#19/#20/#21/#23 among them), two fixed same-day.

Itemized entries accumulated during the arc:

- **#70: GPU build-units merged into `spinel-ext.json`** (spinelgems#20
  landed: `"default": "disabled"` opt-in + `--with-ext NAME` /
  `SPINEL_EXT_ENABLE`, cmake `build_dir` variants, copy-once shared source
  dirs). The staged `spinel-ext-gpu.json` is retired. CUDA re-validated
  end-to-end through the consumer front door on the GB10 (plain `vendor`
  skips the units; `--with-ext cuda --with-ext cuda-shim` builds
  `build-cuda/` alongside the CPU `build/`; vendored `train_cuda`
  reproduces `prep/fixtures/train_cuda_baseline.txt` byte-for-byte on GPU).
  Metal entries ride along structurally (Mac validation pending, toy#27;
  macOS cmake build-units also need spinelgems#21 tool-side). Findings
  hardened: the gem now ships the generated `lib/toy/llm/*_{cuda,metal}.rb`
  mirrors (`gem-prep` runs `gen-mirrors`) — without them Spinel silently
  compiles the missing requires to nothing and consumer GPU builds
  mis-resolve call sites; the `toy new --lib` `build.sh` fails loud when a
  GPU archive is missing (re-vendor hint) and its README documents the
  enable flags.

- **Fixed #76: Qwen3 CUDA F32 degenerate decode.** The hand-written CUDA
  loader (`transformer_lm_cuda.rb`) never wired the detected QK-norm flags
  into the engine — it called `realize_for_mmap` with 5 of 6 args (Spinel
  zero-fills missing call args silently), so Qwen3's per-head Q/K RMS-norms
  were never built on the CUDA graph. Now wired identically to the CPU
  loader; Qwen3-0.6B/1.7B CUDA decode is token-identical to CPU and
  `docs/models.md` rows are restored ✓. Also: `NO_QK_NORM=1` actually
  disables the norm now (it was overwritten by `realize_for_mmap`), and
  loading a QK-norm model through the legacy (non-`toy.ggml_native`)
  copy-load path fails loud instead of decoding silently degenerate.

- **The `toy` gem name is ours.** With huge thanks to **Ninoslav Milenović**,
  who graciously transferred ownership of the [`toy`](https://rubygems.org/gems/toy)
  gem on RubyGems. See the README acknowledgments. (Not released yet — name
  secured for the eventual release.)
- **GPT-2 training foundation.** Two vendored ggml backward kernels
  (`ggml_gelu_back`, `ggml_norm_back`; `vendor-patches/0007`) — finite-diff
  validated, 2 of the 3 gaps in our upstream ggml#1514. A minimal inline GPT-2
  trainer (`make gate-gpt2-min`) proves both kernels train end-to-end
  (CE 3.47 → 0.007). Attention + the full arch are the next increments.

## v0.7.0-pre-alpha — 2026-06-02

### P2-finish + directory tidy (2026-06-03, branches `p2-retire-monolith` / `p2-dir-tidy`)
Past P2's accepted ceiling; all bit-identical-gated on CPU+CUDA (Metal
Mac-verified). Not yet merged to `main`.
- **Monolith retired.** `lib/llama_seq_forward_ffi.rb` → `lib/toy/llm/engine/llama_seq_engine.rb`
  as `Toy::LLM::Engine::LlamaSeqEngine`; `vit_tiny_forward_ffi.rb` likewise →
  `engine/vit_tiny_engine.rb`. The L1–L4 tree is now the engine.
- **`full_finetune` lifted** onto `TransformerBlock` (the 4th realize path,
  left inline at the ceiling) + a 6th byte-exact gate (`prep/full_finetune_gate.rb`),
  recorded from the inline path first. `random_init` global-tensor alloc lifted
  onto `LlamaArch`.
- **Mirrors off-disk.** The 24 `_cuda`/`_metal` twins are generated at build time
  (Makefile static-pattern rules) and gitignored, not committed (−~11k lines);
  `verify-mirrors` regenerates + checks idempotency.
- **Directory tidy.** Top-level `lib/*.rb` grouped under `lib/toy/`: `io/`
  (tokenizer, gguf, loaders), `models/` (transformer, arch, smollm2, gpt2, vit,
  transformer_lm), `train/` (training, lr_schedule, drift_grad, gguf_writer/fuse,
  sampler), `dev/` (describe_flow, card, tap, logprobs). `toy.rb` (package entry)
  and `tinynn.rb` (FFI bridge) kept at root.
- Fixed `bench/check.rb` to `mkdir -p bench/build` (fresh-clone build). Filed
  toy#34: `Tokenizer#decode` drops spaces (byte-level `Ġ` not restored;
  pre-existing on `main`).

### Deferred queue — full compute surface + GPU + packaging (2026-06-01/02)
Landed after the initial v0.7 refactor; all bit-identical-gated, all on the
canonical gx10 box (see gate-portability note below).
- **`toy train` recipes:** `lora` + `warm-start` CLI dispatch (real recipes,
  byte-exact gates), plus the existing `from-scratch`.
- **GPU `--device`:** `cuda` for `infer`/`eval` (token-parity vs CPU) and for
  `train {from-scratch,lora,warm-start}` (byte-deterministic CUDA gates +
  checkpoint round-trip). `metal` runtime for `infer`/`eval`/`train`
  (Mac-verified; clean errors off-platform).
- **train→infer round-trip:** from-scratch/warm-start checkpoints fold the
  projection-lens at write time into a standard fused-llama GGUF that
  `toy infer` loads; `--prompt-ids` + fail-loud on tokenizer-less/out-of-range.
- **`toy serve` telemetry:** emits toy/v1 `run_start` + per-request
  `eval`/`serve`/`request` events (run_end a known gap — `Tep.run!` SIGSEGVs on
  SIGTERM; tep#175).
- **`toy eval lmc`:** two-checkpoint linear-mode-connectivity (pinned fixtures).
- **`toy train vit-tiny`:** first non-llama arch — ViT-tiny from-scratch on the
  committed `data/vit_smoke` corpus; fail-loud on a missing corpus.
- **Gate portability:** float gates (train loss, eval logprobs) are
  gx10-canonical strict byte-exact; cross-platform they gate the discrete
  invariant + a tolerance (eval logprobs run through Ruby libm). Gate corpora
  are pinned committed fixtures.
- **Packaging:** MIT `LICENSE`; gemspec `files` hardened with `git ls-files`
  (artifact-free ~432 KB gem); option-(c) build inputs ship for
  `gem install toy && toy install`.
- **Upstream:** filed tep#168/#172/#175, toy#30 (tep Backend convergence),
  toy#31 (consume tep `~> 0.11`). GPT-2 training deferred (greenfield: ggml has
  no LayerNorm backward).

**Headline.** toy becomes a **CLI-driven framework**. The forward path is
refactored into a five-layer algo stack (primitives → blocks → archs →
recipes), every layer landed behind a **bit-identical-output gate**; a
CRuby `bin/toy` CLI ships **9 commands**; tep is now consumed as a normal
**gem from `main`** (no hand-rolled vendoring); and the docs tree was
rebuilt clean. `toy --version` and the gemspec now agree on the version
(was: gem `0.1.0` vs README `v0.6.0-pre-alpha`).

### P2 — five-layer refactor (`lib/toy/llm/`), all bit-identical-gated
- **L1 primitives** `{rope, swiglu, rms_norm, gqa}`, **L2** `transformer_block`,
  **L3** `llama_arch`, **L4 recipes** `{from_scratch, lora, warm_start}`
  (curriculum deferred) extracted from `lib/llama_seq_forward_ffi.rb`.
  Monolith ~1918 → ~1300 lines.
- **realize-path decomposition:** per-block weight loading for random_init /
  mmap / q8 moved onto the block/arch, behind a **7-way** bit-identical gate
  (CPU + CUDA + GGUF F32 round-trip + GQA-divergent + B>1 + qkv_bias + q8).
- **New gates:** CUDA forward parity (`smoke_projection_lens_cuda`), GGUF F32
  mmap round-trip (`lib/toy_gguf_fuse.rb` head-fusing writer), and the
  config-variant cascade fixtures. Harnesses under `prep/*_gate.rb`.

### P3 — CLI MVP + packaging
- **`bin/toy`** (CRuby, no Spinel): `new`, `install`, `list`, `describe`,
  `fetch` + global `--manifest` / `--help` / `--version`; `toy.yml`
  (run-id template + algos path). Manifest/help auto-generated from one
  COMMANDS registry.
- **`describe`** reads `general.architecture` and **declines** non-llama
  arches (gpt2/gemma2/olmoe) instead of mis-rendering a llama Card
  (fail-loud); `list` dedups by canonical path; pure-Ruby GGUF metadata
  reader (`lib/toy/core/gguf_meta.rb`).
- **Packaging (toy#28 → option c):** the gem ships the backend build inputs;
  `gem install toy && toy install` builds the backend. Verified on Linux
  aarch64 (gx10) + macOS Apple Silicon / Metal; Linux x86_64 pending.

### P4 — `infer` / `train` / `eval` / `serve`
- **Compute bridge:** each command builds a lib-side Spinel runner
  (`lib/toy/run/<cmd>.rb` → `libexec/toy-<cmd>`) via the install machinery
  and shells it with a controlled env; recorded-baseline gates.
- `infer` (greedy, byte-for-byte vs the old example), `train from-scratch`
  (drives the FromScratch recipe → `runs/<id>/` + events.jsonl + checkpoint),
  `eval` (CE/logprobs), `serve` (`lib/toy/serve/openai/`, OpenAI-compatible
  IDs-in/IDs-out HTTP; tep as build-dep only). `examples/01_inference.rb` and
  `tep_demo/openai_api_llama.rb` retired into the CLI/lib.

### Dependencies — tep as a normal gem
- tep consumed from its **`main` branch via the spinelgems convention**
  (`Gemfile` `git:` → `bundle lock` → `spinel-compat vendor` →
  `require_relative "vendor/spinel/deps"`). The `@TEP_*@` post-vendor
  substitution trick is **retired** — `spinel-compat vendor` wires tep's
  C-exts natively from its shipped `spinel-ext.json` (tep#98). `prep/post_vendor_tep.rb` deleted.

### Cleanup + docs
- Docs rebuilt to a clean tree (`architecture`, `cli`, `authoring`, `gating`,
  `dependencies`, `events`, `roadmap`, `reference/`); ~33 dead docs pruned,
  ~16 archived; zero dangling intra-repo links.

### Upstream issues filed
- **matz/spinel#1043** — `Struct.new` accessor names act as global type-merge
  keys (mis-types unrelated code); use hand-written value classes.
- **spinelgems#8** — `spinel-compat vendor` doesn't wire a split
  `@MOD_CFLAGS@` pkg-config to the source `.o` compile (tep pg → libpq-fe.h
  not found); `SPINEL_EXT_DISABLE=pg` workaround.
- **ggml#1506** noted — `mul_mat_id` broken for K-quants (use Q8_0 experts).

### Deferred
GPU runners (`--device`), train `lora`/`warm-start`/`curriculum` variants,
`eval lmc`, ViT, GPT-2 train, the train→infer checkpoint round-trip
(tensor-naming), the realize fixture-cascade (full_finetune), Tep then Tao
re-adaptation (until Toy is stable), Linux x86_64 verification.

## v0.6.0-pre-alpha — 2026-05-28

**Headline.** Tao's two fresh asks shipped same-day (toy#20 per-token
drift + corpus freq; toy#21 sample events). `/v1/embeddings` lives on
all 7 `tep_demo/openai_api_*` servers — mean-pooled OpenAI-shape
response over the dequantize-aware embed_lookup primitive. GH#13
(ViT-Tiny) closed — 200-step acceptance verified, events.jsonl
well-formed. GH#14 (Qwen-2.5-1.5B → 410M transfer) ~85% done; missing
piece is FineWeb-Edu data, filed as toy#22. GH#3 (multi-GPU mode 1)
got its C-side scaffolding (device-index plumbing in tnn_session_new);
runtime testing deferred until multi-GPU hardware. GH#9
(mixed-precision) API foundation shipped (tnn_cast + mp_matmul); full
implementation blocked on ggml autograd accepting BF16 srcs +
completing F16 backward sched paths. Spinel landmines #11 + #15
(File.open + FFI / non-block writes) fixed upstream; cleanup pass
removed stale workaround comments + memorialized probes as regression
tests.

### Tao asks shipped (same-day)

- **Per-token embedding drift + corpus frequency (toy#20).** New
  `lib/toy_token_drift.rb` module; `TOY_TOKEN_DRIFT=N` env knob in
  06_train_from_scratch emits one `drift` event per vocab row per
  N macro-steps with `cos_to_init`, `l2_to_init`, and `freq`
  (training-corpus occurrence). Tao renders this as the freq↔drift
  figure (granite_transfer Pearson r = -0.835 headline).
- **Sample generation events (toy#21).** New `lib/toy_sample.rb`
  module; `TOY_SAMPLES=N` decodes N completions from training-
  sequence prompts at run end and emits toy/v1 `sample` events
  `{prompt, text, step}`. Uses `tnn_compute` (forward graph only)
  so weights are not mutated by sample emission. Cheap-when-off.
- **`/v1/embeddings` on every openai_api server** (closes the
  toy-side of Tao's embedding ask). `tep_demo/embeddings_handler.rb`
  (shared) + route registration in all 7 servers
  (`openai_api_smollm2`, `openai_api_qwen25_{0.5,1.5,3,7}b` × {f32,
  q8}). OpenAI-shape response; mean-pool over input token IDs;
  dequantize-aware (works on F32 + Q8 + Q4 weight tables). Smoke
  on SmolLM2-135M returns 576-dim vector; on Qwen-0.5B returns
  896-dim.

### Training maturity follow-ups

- **GH#7 micro-batching** + **GH#8 LR-scaled grad accumulation**
  shipped (commits 87800fa, 0ae99ac — full details below in
  v0.5.0-pre-alpha). 06_train_from_scratch.rb gains BATCH +
  GRAD_ACCUM env knobs; defaults bit-identical to pre-GH#7.
- **bench BATCH knob** in `bench/lora_step.rb` — emits per-batch
  metrics (`lora_step_b1_ms`, `lora_step_b4_ms`, etc.). Track how
  step time scales with effective batch. B=4 baseline at SmolLM2-
  135M / gx10 CPU: 104.42 ms (vs 58.13 at B=1 → 1.80× wall for 4×
  effective → 2.22× throughput).

### GitHub issue triage / closures

- **GH#13 ViT-Tiny — closed.** Primary acceptance verified:
  `examples/07_train_vit_tiny.rb` runs 200 steps, emits 202-line
  events.jsonl in toy/v1 schema, final loss 5.7e-4 ≪ initial 2.30.
  Scope items (arch, timm loader, training driver, image loader)
  all in main. The "E1 reproduces granite_transfer #28" follow-up
  is downstream Tao work.
- **GH#14 Qwen-2.5-1.5B → 410M transfer — ~85% done.**
  Trainer + projection lens (E2.3) + warm-start donor load
  (`examples/09_warm_start_train.rb`) all working. Final `eval`
  event added at run end (`name:"final"`, matches issue's
  acceptance schema). Qwen-410M invocation pattern documented in
  09's header. Smoke verified loading 233M-float donor embed from
  `data/qwen25-1.5b-f32.gguf`. **Remaining gap is data, not code:**
  FineWeb-Edu pretokenizer filed as toy#22. Once #22 lands +
  TOKENS knob added, a full 10M-token run on the GB10 would close
  the acceptance loop.
- **GH#10 activation recomputation — closed as blocked on
  upstream ggml.** Investigation found no recompute/checkpoint API
  in current ggml (no matches for "checkpoint", "recompute",
  "rematerial" anywhere in `vendor/ggml/`). Issue body's premise
  ("ggml supports this via sched-recompute") doesn't map to a real
  ggml feature. Real implementation paths (vendor patch or
  two-pass approach) are much bigger than the "Ruby-side change to
  realize_for_random_init" the issue suggested. Revisit when toy
  hits the activation-memory wall (1B+ at long context).
- **GH#9 mixed-precision — API foundation shipped; full impl
  blocked on ggml.** `tnn_cast` FFI binding + `mp_matmul` helper
  in `LlamaSeqForwardFFICache` + `WEIGHT_DTYPE` env in
  06_train_from_scratch. At WEIGHT_DTYPE=0 (default, F32):
  bit-identical to pre-GH#9. At WEIGHT_DTYPE=30 (BF16): fails at
  `ggml.c:7052` (autograd builder rejects BF16 src). At
  WEIGHT_DTYPE=1 (F16): autograd accepts but sched can't place a
  backward op (`ggml-backend.cpp:1242`) on either CPU or CUDA.
  Both walls are upstream-ggml. Forward-only F16 cast works
  (probe verified).
- **GH#3 multi-GPU mode 1 — C-side scaffolding shipped.**
  `tnn_session_new_on(kind, device)` + `tnn_cuda_get_device_count`
  C entry points; engine cache widened from a scalar to a
  per-device array (`g_engine_cuda[TNN_MAX_CUDA_DEVICES=8]`). The
  device > 0 path is untested at runtime (gx10 = 1 GPU); the
  device = 0 path (= legacy behavior) is bit-equivalent (CUDA
  example_train_from_scratch_cuda step-1 CE unchanged at
  6.490198612213135).
- **GH#22 (new) — FineWeb-Edu pretokenizer.** Filed; blocks
  GH#14's full acceptance run. ~150 LOC Python: streams
  FineWeb-Edu via uv datasets, tokenizes via Qwen-2.5
  tokenizer, packs i32 binary matching `lib/toy_corpus_loader.rb`.
- **Status comments on the keep-open issues** (#3, #4, #5, #6,
  #14) — each got a concise current-state note. #5 retains its
  "no action expected" framing; #6 is cross-repo (Tep streaming
  handler); #4/#5 blocked on multi-GPU hardware.

### Spinel landmines retired

- **#11 — block-form `File.open(r)` + FFI session-init crash —
  FIXED.** Verified on Spinel a03bb49 (commit 39438d8
  "non-block File.open with sp_File handle" modernized File.open
  codegen end-to-end). Probe at `tinynn/probe_file_block_ffi.rb`
  passes. Existing `File.read(path).split("\\n")` sites stay
  because they're the natural primitive; workaround comments
  removed.
- **#15 — non-block `File.open(path, "w")` silent no-op —
  FIXED** (same commit). Probe at `tinynn/probe_file_nonblock.rb`
  writes "hello from non-block File.open" + reads back; verified
  on disk.
- **#9, #4, #13 — re-verified active.** Probes
  (`probe_hash_missing_key`, `probe_default_args`,
  `probe_nested_mixed_array`) kept as regression checks for
  future Spinel updates. Hash[missing_key]=0 still returns 0;
  cross-module default-args poison can't be reproduced in
  single-file probes but treat as active; nested-mixed-array
  startup-segfault still active with the same
  `incompatible types ... 'sp_IntArray *' from type 'sp_RbVal'`
  diagnostic.

### Tep server consolidation (toy#188)

- **7 → 1 server.** Replaced the seven near-duplicate
  `tep_demo/openai_api_{smollm2,qwen25_{0.5,1.5,3,7}b{,_q8}}.rb`
  files with one env-driven `tep_demo/openai_api_llama.rb`.
  `MODEL_PATH` + `MODEL_NAME` env at boot select any
  llama-family GGUF; `MODEL_NAME` auto-derives from the GGUF
  basename when not given. Net: -2128 LOC, +63 LOC.
- The 7 sources existed because Spinel module-constant
  inference was sketchy on env-driven values when the
  convention was set. With landmines #11/#15 retired this
  release, env-driven constants work cleanly.
- The original `tep_demo/openai_api.rb` (GPT-2 / DistilGPT2)
  stays as the legacy server with the server-side tokenizer.

### DevEx + docs pass

- **README** version bumped to v0.6.0-pre-alpha.
- **examples/README.md** table now covers 06-09 (modern
  from-scratch trainer, ViT-Tiny, LMC, warm-start) and the
  smoke_*.rb wire tests. Serving section points at
  `tep_demo/openai_api_llama` as the canonical HTTP path.
- **tep_demo/README.md** updated for the 4-server reality;
  `openai_api_llama` quick-start added with all the env knobs
  + `/v1/embeddings` + `/v1/completions` curl recipes.
- **events-schema.md** now documents the `sample` event
  (toy#21) and the per-token `drift` variant (token_id + freq,
  toy#20) — both producer + consumer notes.
- **examples/04_serve_http.rb** status note refreshed: the
  original startup segfault is fixed, but `Tep.run!` exits
  immediately because the file still uses the pre-spinelgems
  vendored Tep. The header now points users at
  `tep_demo/openai_api_llama` as the working serving binary.

### Bug fixes

- `tep_demo/openai_api_smollm2.rb` defaults restored to
  SmolLM2-135M (file was serving Qwen-0.5B due to a
  copy-paste artifact — `GGUF_PATH` and `MODEL_NAME` pointed at
  qwen25-0.5b despite the filename). File then renamed to
  `openai_api_llama.rb` as part of the toy#188 consolidation.

## v0.5.0-pre-alpha — 2026-05-27

**Headline.** Tao Tier-3 fully unblocked: trustworthy drift/grad, LMC,
and activation-CKA all have producer-side support landed. From-scratch
training runs on CUDA (~10× CPU at 24L × 16-head Qwen-shape). Toy
checkpoints round-trip through inference. Embedding lookup +
decode-logprobs primitives shipped for Tep's eventual `/v1/embeddings`
and `/v1/chat?logprobs=true`. Graph node capacity scales with
n_layers × n_heads so per-head decomposition doesn't cap us at ~10
layers anymore.

### Training observability + cross-run analysis (Tao Tier-3)

- **Semantic tensor names** (#11, #16). PARAM tensors emitted by
  `realize_for_random_init` (from-scratch), `realize_for_mmap` (LoRA),
  and `realize_for_full_finetune` (FFT) now carry llama.cpp-convention
  names: `token_embd.weight`, `blk.N.attn_q.head_H.weight`,
  `…ffn_down.weight`, `…lora_a.weight`, plus matching `.m` / `.v` for
  Adam moments. Drift, grad, and gguf-checkpoint events now have
  stable cross-run identifiers — Tao's `compare` can align.
- **Session graph capacity is parametric** (#17). Per-head
  decomposition makes node count scale as `O(n_layers × n_heads)`;
  the default 65536 cap overflowed on 24L × 16-head Qwen-shape at
  backward-expand time. `tnn_session_set_graph_capacity` re-allocates
  graphs (auto-grows ctx_buf if needed) + persists across
  `tnn_reset_for_rebuild`. `realize_for_random_init` now sizes
  `cap = n_layers × n_heads × 1000 + 65536` from cfg.
- **Activation-Gram taps for CKA** (#15). `ToyTap.emit_cka` computes
  `G = Aᵀ·A` (T×T Gram) of an `[d, T]` activation and emits it as a
  `gram` field on tap events. Wired into `build_seq_block` at three
  stable regions per block: `attn_norm`, `ffn_out`,
  `resid_post_block`. Gated by `TOY_CKA=N` (every N steps). Tao's
  `Analyze.linear_cka` already unit-tested on synthetic grams.
  Schema: `gram` field added to `tap` event in
  [`docs/events-schema.md`](docs/events.md).
- **LMC interpolate-and-eval runner** (#18). `examples/08_lmc.rb`
  takes two toy checkpoints + α-grid; for each α it blends
  `θ_α = (1-α)·θ_A + α·θ_B` per-PARAM (by name, semantic-names
  required), runs forward + CE on a fixed sequence, emits one
  `eval` event per α with `name="lmc"`. Tao's `Analyze.lmc` reads
  these → α-curve → same-basin / disconnected verdict.

### From-scratch training: CUDA + larger shapes

- **`DEVICE=cuda` for from-scratch training** (#152).
  `prep/gen_cuda_mirror.rb` now mirrors `examples/06_train_from_scratch.rb`
  alongside the FFI libs; `make example_train_from_scratch_cuda`
  produces a CUDA-linked binary. Shell wrapper routes
  `DEVICE=cuda` → CUDA binary. On GB10: SmolLM2-shape ~57 ms/step
  (CPU is multi-s); Qwen-shape (24L × 1024) trains at ~314 ms/step
  vs ~3 s/step on CPU. Math matches CPU to float-roundoff.
- **Checkpoint reload through the inference path** (#153). Toy
  checkpoints written by `ToyGGUFWriter` are now loadable by the
  standard inference path. `realize_for_mmap` detects per-head
  naming via the `blk.0.attn_q.head_0.weight` sentinel and reads
  each head's own GGUF tensor offset instead of base+stride. Writer
  also flags `toy.ggml_native=true` so `transformer_lm.rb` routes
  through the mmap path. Smoke: train 5 steps → write ckpt →
  `lm.generate` 3 tokens, no NaN, no crash. New
  `tnn_gguf_w_set_bool` primitive for the flag.

### Tep `/v1/*` building blocks

- **Embedding lookup** (#145). `tnn_embed_lookup_to_doubles`:
  dequantize-aware single-row read from a 2-D tensor whose data
  lives in CPU-readable memory (mmap'd GGUF pages — the common
  case). F32 short-circuits memcpy; Q4/Q5/Q6/Q8/F16 go through
  ggml's per-type `to_float`. Ruby API:
  `ToyLM#embed_lookup(token_ids) → flat Array<Float>` of length
  `n_tokens × d_model`. Verified on llama-3.2-1b f32 and
  qwen25-0.5b q8.
- **Decode logprobs** (#151). `ToyLogProbs.log_softmax` (max-shift,
  numerically stable) + `ToyLogProbs.top_k` (manual partial-sort,
  Spinel-safe). `ToyLM#decode_step_with_logprobs(token_id, pos, k)`
  returns `[logits_mat, logprobs_mat, top_ids, top_vals]`. Smoke on
  SmolLM2-135M: top-5 logprobs around -2.6 to -3.7, argmax sanity
  passes.

### Consumer packaging — toy as a vendored gem

- **toy.gemspec** (`toy#19`, commit `1f06840`). A downstream research
  project (e.g. `tao_transfer`) can now declare `gem "toy", path:
  "../toy_ruby_neural_network"` in a Gemfile, `bundle lock`, run
  `spinel-compat vendor` from
  [spinelgems](https://github.com/OriPekelman/spinelgems), apply a
  small post-vendor link-path rewrite (`prep/post_vendor_toy.rb`),
  and compile its own experiment against toy's primitives — no
  forks, no hand-pathed `require_relative`s, no Mat poly-dispatch
  landmine. End-to-end recipe in
  [`docs/consuming-toy.md`](docs/consuming-toy.md).
- **lib/toy/version.rb** + **lib/toy/ffi_manifest.rb**. VERSION
  extracted so the gemspec doesn't pull in Spinel-only `tinynn.rb`.
  FFI manifest follows the same `Tep::FFIManifest` shape from
  `tep#97` — CRuby-only declarative spec of per-backend link
  recipes that consumer-side post-vendor scripts read.
- **prep/post_vendor_toy.rb**. Consumer-side hook (shipped as
  template; consumers copy into their own `prep/`). Rewrites
  `ffi_cflags` in the vendored `tinynn{,_cuda,_metal}.rb` from
  toy's relative `-L.` / `-Ltinynn` paths to absolute references
  anchored at `TOY_SRC`. Honors `CUDA_DIR_LIB` env, `TOY_DISABLE`
  to skip backends a consumer doesn't compile.
- **Verified end-to-end** on a `/tmp` consumer project: bundle
  lock + vendor + rewrite + spinel compile + run. Training 3 steps
  with `LlamaSeqForwardFFICache`, loss 6.44 → 6.32, `events.jsonl`
  emitted with full provenance.

### Training maturity — gradient accumulation (toy#8)

- **`GRAD_ACCUM` env knob** in `examples/06_train_from_scratch.rb`.
  Effective batch = `BATCH × GRAD_ACCUM` without the memory cost
  of a single big batch. Default `GRAD_ACCUM=1` is bit-identical
  to pre-GH#8 (step-1 CE = 6.490187644958496 unchanged; step-10
  CE = 5.3334760665893555).
- **Implementation: LR-scaled mini-batch, not literal grad
  accumulation.** ggml's `opt_step_adamw` is baked into the
  backward graph and runs on every `compute_backward`; there's
  no graph-level "skip this op" primitive, so true grad
  accumulation (skip opt_step for N-1 iters, fire on Nth with
  the accumulated grad) would need either a vendor patch
  (8th hp slot to gate `opt_step_adamw`) or a two-graph
  approach (rebuild between modes — expensive). Instead the
  training loop runs `STEPS × GRAD_ACCUM` micro-batches, each
  firing opt_step with `lr = LR / GRAD_ACCUM`. Cumulative
  weight movement over a cycle matches a single full-lr step
  on the mean grad; Adam's m/v state evolves per micro-step
  rather than once per cycle. For typical settings
  (`beta1=0.9, GRAD_ACCUM<=8`) the divergence from true
  accumulation is the "AdamW state warmup" the issue
  acknowledged.
- **Step semantics: `STEPS` counts macro-steps (effective
  opt-step cycles), not micro-batches.** `STEPS=20 GRAD_ACCUM=4`
  runs 80 forward+backward passes, emits 20 step events, and
  the loss/checkpoint cadence is on the macro boundary.
  `tokens` in the step event = `CONTEXT × BATCH × GRAD_ACCUM`
  (effective tokens per macro-step).
- **Acceptance verified.** `BATCH=2 GRAD_ACCUM=4 STEPS=20` vs
  `BATCH=8 GRAD_ACCUM=1 STEPS=20` both train (final/initial
  loss ratio 0.68 / 0.75 respectively at the toy shape). Curves
  comparable; small differences from Adam state evolution +
  data diversity (GA=4 micro-batches sweep different sequences
  per macro-step).
- **run_start event** carries `config.grad_accum` so Tao /
  external consumers see the actual training regime.

### Training maturity — micro-batching (toy#7)

- **B>1 in `realize_for_random_init`**. New `t_batch` arg (now 7
  positionals; the from-scratch example exposes it as the `BATCH`
  env knob). Lays B sequences side-by-side as a flat `[T*B]`
  token + position vector. RoPE applies the right per-batch
  positional encoding because `rope_ext` reads `positions[k]` for
  each `ne[2]` slot; positions cycle `0..T-1` per batch element.
- **Block-causal attention mask**. New `@t_seq_attn_mask`
  persistent `f32 ne=[T*B, T*B]` tensor, uploaded once at realize
  via `upload_block_causal_mask!`: `0.0` for `(query, key)` pairs
  in the same batch with `key <= query`, `-1.0e30` everywhere else
  (`exp(-1e30) == 0.0` in f32). Applied via
  `tnn_soft_max_ext(scores, mask, scale, 0.0)`, which folds
  `scale + mask + softmax` into one op.
- **B=1 stays on the legacy path** (bit-identical to pre-GH#7):
  no mask tensor allocated, `tnn_scale + tnn_diag_mask_inf +
  tnn_softmax` triple. The conditional is `if @seq_b > 1` in
  `build_seq_qhead`; everywhere else the `T*B` arithmetic
  collapses to `T` at B=1 with no branch.
- **Verified bit-identical at B=1** on CPU + CUDA (CE step-1
  6.490198 unchanged); verified learning at `B=4, B=8` on both
  backends; CUDA's per-step time drops with `B=8` (19.5 ms vs
  24.2 ms at B=1, T=16) — launch overhead amortizes. Larger
  shape `T=64 B=8 STEPS=20`: CE 6.51 → 4.84 in 12 ms/step on
  CUDA.
- 06_train_from_scratch.rb emits `config.batch` on `run_start`
  and `step.tokens = CONTEXT * BATCH`. Other realize-path
  callers (`08_lmc`, `09_warm_start_train`,
  `smoke_projection_lens`) pass `t_batch=1` — those experiments
  stay on the single-sequence path by design.

### Roadmap docs

- [`docs/archive/e1-e2-scope-2026-05-27.md`](docs/archive/e1-e2-scope-2026-05-27.md)
  — full decomposition of E1 (ViT-Tiny, 6 sub-issues, 5-8 days) and
  E2 (Qwen-410M embedding transfer, 7 sub-issues, 3-5 days). E2.1
  cheapest-first-step was executed (24L × 16H Qwen-shape) — uncovered
  the graph-capacity blocker (filed + shipped as #17).
- [`docs/roadmap/backends-and-scale-2026-05-27.md`](docs/roadmap/backends-and-scale-2026-05-27.md)
  — training maturity (batching, grad accum, mixed-precision,
  activation recompute), hybrid CPU/GPU offload, non-ggml backends,
  the strategic question on what toy should own.

### Spinel landmines pinned this run

- F32 has no `to_float` in ggml type_traits — caller must memcpy-
  shortcut (the type_traits API would otherwise return NULL and
  emit zeros). Now memorialized in `tnn_embed_lookup_to_doubles`
  with the F32 short-circuit branch.
- `Array<Array<int_or_float>>` seeds confuse Spinel poly inference
  and can fault at startup-class-init (`poly_array_push`). Pin in
  `lib/toy_logprobs.rb` header — return two parallel arrays
  (`ids: Array<Int>`, `vals: Array<Float>`) instead.

## v0.4.0-pre-alpha — 2026-05-24

**Headline.** Real OLMoE-1B-7B-Instruct produces coherent factual
answers ("The capital of France is **called Paris**"). Gemma 2 extras
land. Flash attention finally beats baseline on Qwen3-1.7B (12%
faster) after the V cache layout flip that was eating its win. SSM
op primitives bound (Mamba enable, speculative). Upstream ggml issue
filed for a real mul_mat_id × K-quant bug we found.

### I-V-layout-flip (P5.2) — V Q8 unlocked, flash perf realized

- V cache flipped from `ne=[max_T, d_head]` (positions on ne0) to
  `ne=[d_head, max_T]` (positions on ne1, mirroring K). Two wins land
  together:
  - `enable_kv_q8!` now quantizes BOTH K and V (was K-only). Per-
    position V writes span a contiguous d_head-vector, block-aligned
    at d_head=64 (=2 Q8_0 blocks).
  - `flash_attn_ext` consumes V natively in its expected
    `[d_head, hist_count]` orientation — no transpose-cont in the
    hot loop. P4.1's reported flash=baseline was being eaten by that
    transpose; now flash wins.
- Qwen3-1.7B, N_NEW=32, CPU:
  - baseline    3.54s  (1.00×)
  - FLASH only  3.09s  (0.87×, **12% faster** — was a wash before)
  - KV_Q8       3.07s  (0.87×, K+V Q8 + flash)
- SmolLM2-135M four-config token streams bit-identical (baseline /
  KV_Q8 / FLASH / KV_Q8+FLASH).
- Structural constraint: Q8 V *requires* flash. Transposing a Q8
  tensor yields a non-block-aligned destination (hist_count isn't
  divisible by 32); ggml_cont can't materialize it. `enable_kv_q8!`
  auto-enables flash to make this transparent.

### #110 I-QKnorm — per-arch QK-norm flavor (OLMoE coherent text)

- OLMoE / Granite-MoE store the QK-norm gamma at `[d_model]` (per-
  head packed) and apply RMSNorm to the *full* Q before head split.
  Qwen3-style models store it at `[d_head]` (shared per-head).
  These are mathematically different. M2.3 mis-applied Qwen3 to
  OLMoE, producing "The capital of France is a city in the capital
  of France." This patch detects the flavor and routes accordingly.
- `SmolLM2Flags.qk_norm_kind`: 0 = none, 1 = Qwen3, 2 = OLMoE.
  Detected from `blk.0.attn_q_norm.weight`'s ne[0] against d_head
  and d_model from the multi-arch (llama.* / olmoe.* / gemma2.*)
  metadata.
- Graph builder applies the kind-2 path via a per-head sliced
  `view_1d(gamma, d_head, hq * d_head * 4)`. Per-head approximation
  of true full-Q norm — exact gamma scaling, approximate variance
  pooling. Empirically: produces coherent OLMoE output.
- Validated OLMoE-1B-7B-Instruct Q8:
  - "The capital of France is" → "called Paris."
  - "Python is a programming language that" → "is used to create
    programs that can be executed on a computer."
  - "The largest planet in our solar system is" → "Jupiter. It is
    a gas giant"
  - "Albert Einstein was famous for" → "his theory of relativity"
- KV_Q8 + flash + per-head sliced norm compose cleanly: same
  factual answers on the quantized path.

### #113 I-Gemma — Gemma 2 extras

Four model-specific extras integrated as opt-in features. Non-Gemma
models pass inert defaults and the graph paths are no-ops.

- **Embedding scale** sqrt(d_model) applied post-token-embed lookup
  (Gemma 2-2b → 48.0). Newton sqrt at detection time avoids the
  Math.sqrt Spinel landmine.
- **Logit soft-cap** `tanh(x/c)*c`:
  - Attention logits (c = 50.0): wired through flash_attn_ext's
    native `logit_softcap` parameter; non-flash composes via
    tnn_scale + new `tnn_tanh` + tnn_scale.
  - Final output logits (c = 30.0): applied to t_kv_logits before
    set_output.
- **Pre + post norms** on each sublayer: new
  `t_post_attn_norm_gamma` + `t_post_ffn_norm_gamma` allocated
  when has_post_norms. Applied on sublayer output BEFORE the
  residual add (Gemma 2's sandwich).
- **Alternating SWA**: per-layer toggle (even = sliding, odd = full)
  when `@swa_alternates` is set. layer_idx threaded through
  build_attention_qhead_step.
- Multi-arch metadata probe gains "gemma2" alongside llama / olmoe.
  `rope.freq_base` default-fallback to 10000.0 (Gemma 2 doesn't
  emit the key). Force `is_native = true` when has_post_norms is
  detected (third-party Gemma GGUFs take the mmap path despite
  lacking toy.ggml_native).
- New primitive: `tnn_tanh` (ggml_tanh wrapper).
- Gemma 2-2b-it Q8 loads, mmaps, graph builds + realizes cleanly,
  produces well-formed varied logits (no NaN). End-to-end text
  output is currently blocked at the **tokenizer** layer: Gemma 2
  uses sentencepiece with vocab=256000, our tokenizer mis-
  tokenizes. Separate task (#117).

### #112 Q-mul_mat_id × K-quants — upstream filed

- Discovered in M2.3: ggml's `mul_mat_id` kernel produces wrong
  output for K-quantized (Q4_K, Q5_K, Q6_K) expert weights. Same
  model at Q8_0 produces coherent text; at Q4_K_M produces
  "Dub Dub Dub" repeating. Root cause likely: mul_mat_id only has
  reliable kernels for F32/F16/Q8_0 sources per
  `test-backend-ops.cpp::test_mul_mat_id` registrations.
- Filed upstream: <https://github.com/ggml-org/ggml/issues/1506>
  with the OLMoE repro + suggested test additions.
- Runtime WARN: realize_for_mmap detects K-quant MoE weights (type
  ∈ [10, 19]) and emits four WARN lines on layer 0. Loud failure
  mode rather than silent wrong output.
- Documented in `docs/notes/mul_mat_id_quants.md` — the canonical
  write-up with the workaround (use Q8_0 for MoE expert weights).

### #114 C-SSM — SSM_CONV + SSM_SCAN bindings (speculative)

- `tnn_ssm_conv` + `tnn_ssm_scan` FFI wrappers on CPU + CUDA.
  Coverage 28 → 30 of 98 ops bound. Mamba/Jamba family is now
  blocked only by Cache-class wiring, not by primitives.
- Speculative binding. No Mamba use case yet. When someone wants
  Mamba inference, they start from "build a Mamba Cache class" with
  the primitives already wired.
- Shape expectations documented (Mamba-2 grouped 4D layout). Smoke
  deferred to the M-Mamba follow-up — meaningful test needs proper
  Mamba-2-shaped inputs which is more work than the binding itself.

### Coverage matrix tweak

`prep/gen_coverage.rb` regression fix: `PRIMARY_WRAPPER` override
map for ops where our wrapper name doesn't follow `tnn_<ggml_stem>`
(MUL_MAT → tnn_matmul, SOFT_MAX → tnn_softmax). Caught during the
post-Metal-merge regression battery — the new wrapper-matching
heuristic was demoting them to status `via`.

### Follow-ups filed

- **#117 T-Gemma-tokenizer**: SentencePiece for Gemma 2
  (vocab=256000). Graph integration works; example_inference text
  output blocked on tokenizer correctness.
- **ggml-org/ggml#1506**: upstream mul_mat_id × K-quant bug. Drop
  the runtime warning + add a coverage smoke when fixed.

## v0.3.0-pre-alpha — 2026-05-24

**Headline.** Metal backend (issue #2). SmolLM2-135M runs end-to-end
on Apple Silicon GPUs with bit-identical output to the CPU path. Same
FFI surface as CPU + CUDA, same graph builder, same generated mirror
classes — the only switch is `tnn_session_new(2)`. Validated on M2;
expected to work on M1 / M3 / M4 / M5.

### B-Metal — third backend

- New `setup-ggml-metal` Makefile target. Builds with
  `GGML_METAL_EMBED_LIBRARY=ON` so the .metal shaders are baked into
  the static archive as raw bytes — the Metal driver JIT-compiles
  them on first device load (~15 s one-time per binary, then cached).
  Works with Command Line Tools alone; no full Xcode required.
- `tinynn/tinynn_backend_metal.m` (Objective-C). Strong
  `tnn_backend_metal_init_internal()` calling `ggml_backend_metal_init`,
  mirroring the CUDA backend's archive-isolation pattern. Plus
  `tnn_force_exit` — a flush-then-`_exit` trampoline that skips
  `__cxa_finalize` so ggml-metal's static-destructor residency-set
  assert doesn't fire on short-lived programs.
- `tinynn/tinynn_ggml.c` engine cache extended to ternary
  (CPU / CUDA / Metal) via `tnn_engine_get(backend_kind)`. The
  `prefer_cuda` integer is now `backend_kind` with 0/1/2 semantics;
  callers pass `2` to opt into Metal. Adds `tnn_shutdown_engines()`
  for explicit teardown (CPU + CUDA tolerate the call as well).
- `lib/tinynn_metal.rb` — `TinyNNMetal` FFI module mirroring the full
  CPU surface, plus Ruby helpers (`upload_int_array`,
  `download_row_major`, …) the generated mirrors call. Links
  Foundation / Metal / MetalKit frameworks.
- `lib/transformer_lm_metal.rb` + `lib/toy_smollm2_ffi_kv_metal.rb` +
  the rest of the `_metal.rb` mirror set. The KV-cache decode runs on
  GPU; the loader takes the copy-load (non-mmap) path because
  ggml-metal doesn't expose a public `buffer_from_pointer` — the
  scheduler crashes when fed CPU-resident weight tensors as kernel
  inputs. Multi-GB models pay the copy cost on Metal until upstream
  adds the BYO-pointer API.
- `prep/gen_cuda_mirror.rb` generalized: emits both `*_cuda.rb` and
  `*_metal.rb` from the same CPU source via a per-backend
  substitution table (`--backend cuda|metal` to target one).
- `examples/01_inference_metal.rb` + `make example_inference_metal`
  — end-to-end smoke. Output for SmolLM2-135M F32 with the five-ID
  fallback prompt is bit-identical to the CPU path.
- `docs/coverage.md` gains a Metal column. Today the Metal mirror is
  intentionally a thin surface; the `0/26` Metal-bound count is the
  follow-up to-do list, not a regression signal.

### Known gaps (Metal)

- Zero-copy mmap: blocked on a public `ggml_backend_metal_buffer_from_ptr`
  upstream. Workaround: copy-load (current default).
- GPT-2-family validation: `lib/gpt2_ffi_*_metal.rb` mirrors exist
  (generated), but no binary has been built against them yet.
- Quantized weights: untested on Metal. Should work — same kernel
  coverage upstream — but no smoke yet.

## v0.2.0-pre-alpha — 2026-05-23

**Headline.** Three new model families work end-to-end as
text → text: Qwen3-0.6B (dense), Mistral-7B-Instruct-v0.2, and
TinyLlama-1.1B. RoPE scaling (YaRN / llama3 / linear) lands.
The tokenizer now handles both byte-level BPE (GPT-2 / Llama-3 /
Qwen) and SentencePiece (Llama-1/2 / Mistral). A bench harness +
card-drift detector + Chrome-Trace-format observability primitive
ship as reusable infrastructure.

Still pre-alpha: no API stability commitments, and we'll happily
break shapes when something deserves it. See the sections below
for the full inventory.

### T1.3 — SentencePiece tokenizer (Llama-1/2 / Mistral / TinyLlama)

- `lib/tokenizer.rb` auto-detects SentencePiece vs byte-level BPE
  by checking `vocab[3] == "<0x00>"` (the first byte-fallback
  token in any SPM vocab). Sets `@spm = true` and dispatches
  `encode` / `decode` to SPM-specific paths.
- SPM encode: prepend `▁` (U+2581), replace ASCII spaces with `▁`,
  char-split, byte-fallback any char not in vocab via `<0xHH>`
  tokens, then run the same merge-loop BPE as the GPT-2 path.
- SPM decode: collapse `<0xHH>` byte-fallback runs back to UTF-8
  bytes (with robust byte-level hex parsing to dodge a Spinel
  `String#[Range]` quirk), and convert `▁` → ASCII space (stripping
  the leading boundary marker so the round-trip is lossless).
- Converter detects the tokenizer flavor at conversion time
  (same `vocab[3] == "<0x00>"` heuristic) and emits the right
  `tokenizer.ggml.model` value (`"llama"` for SPM, `"gpt2"` for
  byte-level) plus the right `tokenizer.ggml.pre` hint.
- Verified:
    TinyLlama-1.1B: "The capital of France is Paris, which is
                     known for its beautiful architecture, museums,
                     and cultural events"
    Mistral-7B-v0.2: "The capital of France is Paris."
- `tinynn/ab_smoke_tokenizer.rb` extended to include
  TinyLlama-1.1B-tok in the round-trip matrix. Now 20/20 PASS
  across SmolLM2 / Llama-3.2 / Qwen2.5 / TinyLlama on five
  representative English prompts.

### M1.1 — Qwen3-0.6B works end-to-end (head_dim + o_proj fix)

Final fix for the `sp_ToyLM_decode_step` crash: `t_w_o` was
allocated as `(d_model, d_model)`, but the output projection
actually maps `[n_heads * d_head] → [d_model]`. For SmolLM2 /
Llama / Qwen2.5, `n_heads * d_head == d_model`, so the shape was
right by accident. Qwen3-0.6B has `n_heads * d_head = 2048 ≠
d_model = 1024`, and the matmul aliased the wrong memory region.

Fix in both realize paths: `tnn_input_2d_persistent_mmap(@sess,
@d_model, @n_heads * @d_head, ...)`.

Now produces coherent text:
  prompt: "The capital of France is"
  output: "The capital of France is located in the city of Paris..."

Existing models unchanged (n_heads*d_head==d_model is an
invariant of the older configs).

### M1.1 — explicit head_dim + tied-embeddings handling (partial)

- `Toy::SmolLM2Config` gains `head_dim` field (defaults to
  `d_model / n_heads`; loader overrides from GGUF).
- `SmolLM2ConfigLoader.read` reads `llama.attention.key_length`
  (llama.cpp convention) when present; falls back to the computed
  value for SmolLM2 / Llama-3.x / Qwen2.5 (all match
  hidden_size/num_heads).
- `lib/toy_smollm2_ffi_kv.rb` and `lib/llama_seq_forward_ffi.rb`
  now read `cfg.head_dim` everywhere they used to compute
  `cfg.d_model / cfg.n_heads`. Behaviour unchanged for models
  whose explicit head_dim matches the computed one.
- `prep/convert_smollm2_to_gguf.py` emits
  `llama.attention.key_length` + `llama.attention.value_length`
  GGUF keys (one value, written twice — every model on the
  roadmap has K-dim == V-dim).
- Converter also handles `tie_word_embeddings: true` correctly:
  skips emitting `output.weight` even when `lm_head.weight` is
  in the safetensors. Trusts the config flag (as llama.cpp does).

Qwen3-0.6B converts with the right head_dim=128 and tied
embeddings, but inference still crashes in
`sp_ToyLM_decode_step`. M1.1 stays open; root cause appears to
be deeper than the converter / loader layer (suspect ToyLM /
decode-step assumes specific shape invariants).

Bench passes (±2% across all metrics). Existing models —
SmolLM2, Llama-3.2, Qwen2.5 — produce identical text.

### M1 — Qwen3 dense plumbing (QK-norm, partial)

- `SmolLM2KVBlockFFI` now carries `t_q_norm_gamma` + `t_k_norm_gamma`
  (1D `[d_head]` shared across heads, per block). `SmolLM2KVFFICache`
  carries `@has_qk_norm` flag.
- `GGUFLoad.detect_smollm2_flags` detects QK-norm by presence of
  `blk.0.attn_q_norm.weight` (Qwen3-only; Qwen2.5 / Llama return false).
  `SmolLM2Flags` gains a `qk_norm` field.
- `realize_for_mmap` signature widened: `(gguf, cfg, max_T, untied,
  qkv_bias, qk_norm)`. Allocates the QK-norm gammas as mmap'd
  1D F32 tensors when set. Graph builder applies `tnn_rms_norm` to
  Q and K with the per-block gamma BEFORE `tnn_rope_ext`.
- `prep/convert_smollm2_to_gguf.py` propagates Qwen3's
  `self_attn.q_norm.weight` / `k_norm.weight` HF tensors to
  `attn_q_norm.weight` / `attn_k_norm.weight` GGUF tensors.
- Existing Qwen2.5 / SmolLM2 / Llama-3.2 / TinyLlama unchanged
  (no QK-norm path triggered). Bench passes within ±5%.
- Qwen3-0.6B converts and loads, but **inference is not yet
  correct**. Root cause: Qwen3 sets `head_dim = 128` explicitly in
  HF config, not `hidden_size / num_heads = 64`. Our converter
  computes the wrong head_dim and the Q/K/V projections come out
  half-sized. Tracked as M1.1 (next task). QK-norm itself appears
  correctly wired — the head_dim mismatch alone explains the
  garbage output.

- Re-converted `data/tinyllama-1.1b-tok.gguf` and
  `data/mistral-7b-instruct-v0.2-tok.gguf` with `--with-tokenizer`.
  Both load + run inference, but **text-mode I/O fails** because
  their tokenizers are SentencePiece (Llama-2 vocab), not the
  byte-level BPE our `lib/tokenizer.rb` handles. T1.2's "never mask"
  rule caught it cleanly: `WARN: tokenizer: piece "Ġ" not in vocab
  — emitting UNK` ⇒ Mistral output `"The<unk>capital<unk>of<unk>..."`.
- Tracked as T1.3 (new task). Adds tokenizer-flavor detection from
  `tokenizer.ggml.model` and a SentencePiece encoder path.
- **Current text-I/O coverage** (works end-to-end):
  SmolLM2-135M, Llama-3.2-1B, Qwen2.5-0.5B. All byte-level BPE.
- **Inference-only** (text I/O blocked on T1.3): TinyLlama-1.1B,
  Mistral-7B-v0.2. ID-mode `example_inference` (no PROMPT) still
  works fine.

### D1 — algorithm-card drift detector (instead of auto-emitter)

- `prep/card_drift_check.rb`: a Ripper-walker tripwire that
  verifies each `Toy::` class with both `def forward` and
  `def algorithm` keeps the two in lock-step. Run via
  `make check-cards`. Pure-Ruby stdlib, no extra deps.
- D1 was originally scoped as an auto-emitter that would delete
  the 209 LOC of hand-written `def algorithm` methods. Closer
  reading of those methods showed they're not 1:1 with the
  unrolled forward code — `FFN`'s 5-line forward becomes a
  curated 2-step card that fuses `matmul + add_bias + gelu`. An
  auto-emitter would produce faithful-but-ugly output that
  wouldn't actually replace the cards.
- The drift detector matches the real failure mode: forward
  changes, card doesn't (or vice versa). Catches the common
  `gelu` / `silu` / `⊙` activation-mismatch + the
  matmul-presence collapse case. Validated by deliberate-drift
  test (deleting `gelu(...)` from FFN's card → tool fails).
- For the original "delete the 209 LOC" goal (re-trigger
  condition (a) from task #95): an auto-emitter is still possible
  if/when we add a third architecture and feel the cost, but the
  drift detector covers the maintenance-during-edits case today.

### P2 — measured, not viable (skipped)

- `docs/roadmap/p2-ffi-matmul-2026-05-23.md`: planned to FFI-wrap
  `Mat#matmul` for a 5–10× win on `example_train`. Measured the
  actual cost: session-per-op FFI is **1.7× SLOWER** than pure-Ruby
  at training-toy shapes (32×8 matmul). 5 000 calls × ~180 µs
  session lifecycle overhead = ~0.9 s deficit vs the 1.3 s
  pure-Ruby baseline.
- Lesson: the 38× FFI gain on LLM-shape inference is for
  *whole-graph* FFI (one `tnn_compute` per step), not per-op FFI.
  At toy shapes, per-op FFI loses on session overhead; break-even
  is around `m*k*n ~ 100 000`. Real workloads (LoRA, KV decode,
  full FT) route through whole-graph FFI cache classes already
  and are unaffected.
- If we later want fast `example_train`, build a
  `TransformerLMTrainerFFI` mirror of `LlamaSeqForwardFFICache`
  for the custom-GPT shape (~500 LOC, ~5–20× expected). Queued
  as P2-α; not currently prioritised.

### Bench harness + Lowerer evidence

- `bench/` directory with three Spinel-compiled benches —
  `lora_step.rb` (training step ms), `inference.rb` (toks/sec), and
  `tokenizer.rb` (encode μs/token). Each emits `BENCH metric value`
  lines on stdout; the orchestrator at `bench/check.rb` runs them
  and compares to `bench/baselines.csv`, exiting 1 on any metric
  past its per-metric tolerance.
- `make bench` runs the gate; `make bench-update` rewrites the
  baselines; `make bench-report` runs without gating (handy for
  local exploration). Suggested use: invoke before pushing
  perf-sensitive changes; the CSV diffs cleanly in git so
  re-baselining is a normal commit.
- `docs/roadmap/lowerer-evidence-2026-05-23.md`: trace-driven
  evidence on whether the full Lowerer pays for itself.
  Measured `example_train` (native-Mat path) — 82.7 % of step
  wallclock is in three matmul methods, 51 % of all matmul
  calls share one (N, P) inner-dim pair. But the comparable
  alternative (FFI the matmul through ggml) gives 10–100×
  for an order of magnitude less code than the ~500-LOC
  Lowerer. Verdict: split the Lowerer's three claimed benefits
  into proportionate tools (P2 to FFI Mat ops; D1 standalone
  card emitter; Spinel landmines wait for upstream).
- `lib/transformer.rb`: six hot `Mat` methods (`matmul`,
  `matmul_t`, `t_matmul`, `plus`, `add!`, `scale!`) wrapped with
  `tnn_trace_begin/end`. Zero-cost when off (1.44 s baseline →
  1.44 s with traces present but inactive on a 87-sequence
  training run); 12 % overhead when on. `MAT_SHAPES=1` env
  enables a shape histogram printf for Lowerer-style evidence
  runs.

### Toolchain

- Bumped Spinel to master `d59926a`. Two changes affect us directly:
  - `0adca86` (matz/spinel#647): `examples/example_serve` no longer
    segfaults at startup. The bug was a Spinel codegen ordering issue
    on top-level `CONST = recv.method` when `recv` was a local;
    we'd kept the buggy form intentionally as a regression check.
    Confirmed: it loads the model, binds the port, accepts requests.
  - `97bf268` (rbs_extract): `--rbs sig` Mat-in-`Toy::` resolution
    now works. Warning overhead dropped from +45 (was unusable) to
    +3 (acceptable). `sig/toy.rbs` header reflects this.
  - The Hash#[missing] → 0 codegen behavior (T1.2's root cause)
    still requires the `has_key?` guards we landed yesterday;
    Spinel's matz/spinel#521 fix narrowed the symptoms but the
    structural workaround stays.

### Observability / training

- New `tinynn/tinynn_trace.{c,h}`: Chrome Trace Format emitter,
  ~5ns per begin/end when off, opens in https://perfetto.dev.
  Instrumented `tnn_realize{,_backward}`, `tnn_compute{,_backward}`,
  `tnn_upload{,_from_float_array,_from_int_array}`, `tnn_download`.
  `examples/03_finetune_lora.rb` and its CUDA mirror accept
  `TRACE=path.json` to wrap each step.
- New `tnn_scratch_sum_f32` and `tnn_scratch_sum_sq_f32` reducers
  (mean / L2-norm without a Mat round-trip).
- `examples/03_finetune_lora{,_cuda}.rb` accept `GRAD_DUMP=1` to
  emit per-(layer, head, A/B) gradient stats as CSV.
- **Finding** (`docs/p1-grad-bisection-2026-05-22.md`):
  CPU and CUDA LoRA gradients agree to 0.32–0.42 % median across
  steps 1–3; loss curves match to 3+ decimals. The underlying
  ggml-cpu sched aliasing bug is still upstream but the local
  workaround (`tnn_pin_all_graph_b_nodes`, wired into
  `lib/llama_seq_forward_ffi.rb:1192`) prevents it from biting
  prod LoRA training. Issue still wants filing against ggml-org/ggml.

### Inference / DevEx

- `examples/example_inference` now speaks text end-to-end on GGUFs
  converted with `--with-tokenizer`: encode `PROMPT`, generate,
  decode the IDs back into text. Falls back to the hardcoded
  five-ID SmolLM2 prompt + raw IDs when no tokenizer is embedded.
- `tinynn/ab_smoke_tokenizer.rb`: 15/15 prompts round-trip
  bit-identically on SmolLM2-135M, Llama-3.2-1B, Qwen2.5-0.5B.
  The earlier "SmolLM2 fails 2/5" bug (`?\n` and `.txt` patterns)
  was actually a **Spinel hash codegen bug**: `Hash#[missing_key]`
  returns integer 0 instead of nil, so missing merges appeared as
  rank-0 (top-priority) merges in the BPE loop. Fixed by guarding
  every hash read with `has_key?`. Tokenizer now warns loudly when
  emitting UNK so the next instance of this class of bug surfaces
  fast.

### RoPE scaling (FFI)

- `tnn_rope_ext` and `tnn_rope_ext_back` widened with the full
  YaRN/llama3 arg surface (freq_scale, ext_factor, attn_factor,
  beta_fast, beta_slow, freq_factors). New
  `tnn_rope_freq_factors_alloc` allocates the per-dim factors
  tensor; values computed in
  `Toy::RopeScaling.compute_llama3_freq_factors`.
- `Toy::RopeScaling` value class + `Toy::SmolLM2Config.@rope_scaling`
  field; `SmolLM2ConfigLoader` reads `llama.rope.scaling.*` from
  GGUF and dispatches to `none / linear / llama3 / yarn`.
- `prep/convert_smollm2_to_gguf.py` now propagates HF `rope_scaling.*`
  metadata to GGUF. Without this our Llama-3.x GGUFs had no scaling
  metadata; converter was the bottleneck.

## v0.1.0-pre-alpha — 2026-05-22

**First tagged cut.** Not API-stable; expect breaking changes in any
direction. The goal of this tag is to mark a coherent set of working
capabilities so external readers have a reference point.

### Inference

- Run pretrained models end-to-end as a single native binary:
  GPT-2 family, Llama-family (SmolLM2, TinyLlama, Llama-3.2,
  Mistral-7B), Qwen2.5 family (0.5B → 7B).
- KV-cache decode on CPU and CUDA, F32 and Q8.
- Zero-copy mmap of GGUF weights into both CPU and CUDA buffers
  (UVA on GB10).
- Discover cached models from HuggingFace / Ollama / LM Studio /
  `./data` / `$TOY_MODEL_DIR` via `examples/example_list_models`.

### Training

- From-scratch training of small GPTs via `Toy::Trainer`
  (`examples/example_train` on TinyStories).
- LoRA fine-tune on attention Q heads, CPU + CUDA
  (`examples/example_finetune` / `example_finetune_cuda`).
- QLoRA: Q8 base + F32 LoRA adapter, CPU (works through mmap) and
  CUDA (works through the new `realize_for_q8_copy` path that
  bypasses the BYO-pointer padding issue).
- Full fine-tune on CUDA: every per-block weight + optional
  embedding/output trainable, up to ~1.5B verified
  (`demos/smollm2_seq_full_finetune_cuda`).
- Sequence-mode forward graph (M3): `T` tokens in, `T` logits out;
  one forward + backward + opt_step per training step instead of
  T separate KV-decode rebuilds.

### Serving

- OpenAI-compatible HTTP API via Tep+Spinel
  (`tep_demo/openai_api_smollm2` and family).
- Lite HTTP example for direct token-IDs in / token-IDs out
  (`examples/example_serve` — see Known issues).

### Infrastructure

- Phase 0.6 CPU/CUDA mirror dedup: `prep/gen_cuda_mirror.rb`
  generates `*_cuda.rb` files from their CPU counterparts via
  mechanical substitution; `make verify-mirrors` catches drift.
- Vendored ggml patches (`vendor-patches/`):
  - `0001-0002` CUDA `buffer_from_ptr` for BYO-pointer mmap.
  - `0003` BYO copy-mode A/B selector.
  - `0004` CUDA `cpy` strided-destination fix.
  - `0005` `GGML_OP_CONCAT` backward.
  - `0006` chunked `get_rows_back` for vocab > 65535
    (fixes Qwen-class embedding training on CUDA).

### Bench reference (GB10, 2026-05-22)

| Model               | CPU tok/s | CUDA tok/s |
| ------------------- | --------- | ---------- |
| SmolLM2-135M F32    | 88        | 76         |
| Qwen2.5-0.5B F32    | 34        | 41         |
| Qwen2.5-1.5B F32    | 14        | 20         |
| Qwen2.5-7B Q8       | 4.2       | 14         |

LoRA training step time on SmolLM2-135M (T=4): 108 ms/step.
Full fine-tune step time on SmolLM2-135M (T=4): 108 ms/step.

Full details: `docs/archive/bench-gx10-2026-05-22.md`.

### Known issues

- CPU LoRA training requires `tnn_pin_all_graph_b_nodes` to work
  around a ggml-cpu scheduler aliasing bug on long backward chains
  (filed upstream as `ggml-org/ggml#1501`). Current cache classes
  apply the pin transparently.
- Full fine-tune memory ceiling: ~3B on a 121 GB unified-memory box
  (1.5B comfortable, 3B fits, 7B doesn't).

### Sibling projects

- [Spinel](https://github.com/matz/spinel) — Ruby AOT compiler
  matz is building; we live in `~/sites/spinel`. Issues filed
  during this work: `#644` (Range indexing codegen), `#645`
  (Optional<Int> narrowing).
- [Tep](https://github.com/OriPekelman/tep) — Sinatra-flavoured
  HTTP framework that compiles to a native binary via Spinel.
  Issues filed during this work: `#13`, `#16`, `#17`.
- Vendored [ggml](https://github.com/ggml-org/ggml). Upstream
  contributions filed: `#1500` (merged), `#1501` (open).
