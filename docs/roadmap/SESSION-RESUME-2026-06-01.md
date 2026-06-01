# Session resume — 2026-06-01 (gx10 wind-down)

This session drove the post-v0.7 deferred queue on gx10. Winding down here;
the **metal followup continues on the Mac**, then we return to gx10.

## main is at `507e160` (clean, pushed origin+github, 0/0, no binary bloat)
Commit chain this session:
- `33a5efb` train variants (lora + warm-start) — bit-identical gates
- `acf0d0b` `--device cuda` for infer + eval — CUDA-vs-CPU token parity
- `fbd0f35` train→infer checkpoint round-trip (projection-lens fold at write time)
- `bc721b4` wire toy/v1 serving events + serve gate; **glob .gitignore fix**
- `88b8ded` `toy train from-scratch --device cuda` — strong byte gate
- `507e160` gates: platform-aware float comparison (gx10-canonical strict / portable elsewhere)

## IN FLIGHT — blocks the next gx10 work
**`metal-source-wiring` branch** (`8d52fbe`, pushed github+origin, NOT merged).
Metal runtime twins for infer/eval/train are source-wired + gx10-verified
(codegen-no-link, structural parity vs `_cuda`, clean-error-on-Linux, no
regression). **Runtime-unverified — being gated on the Mac.** Failure to fix:
`make gate-metal` → `metal-infer` produces nil at runtime. Full handoff in
**`docs/roadmap/METAL-FOLLOWUP-2026-06-01.md`** (debug cmd, highest-risk seam =
CPU-TinyNN-reads-Metal-buffer, done-criteria). Mac must **rebase on `507e160`**
(for the portable gates) before its final no-regression run, then merge.

## NEXT gx10 work (RESUME HERE once metal is merged)
HELD until metal merges (all touch `cli/{train,eval}.rb` + `Makefile`):
1. **`lora` / `warm-start --device cuda`** — finish CUDA training coverage
   (mirror `from_scratch_cuda.rb`/`train_cuda.rb`; lora needs its own binary
   like `train_lora`). Main target = Linux+CUDA, so this is the priority.
2. **#5 `eval lmc`** — two-checkpoint linear-mode-connectivity.
3. **#6 ViT + GPT-2 train coverage** — largest scope, lowest priority.
Deferred (toy's call, once stable): **#8 toy#30** — collapse `lib/toy/serve/openai/`
onto tep's `Tep::Llm::OpenAI::Backend` (tep side ready incl. embeddings tep#176);
de-risk with a cross-repo Spinel `use(Backend)` spike first. May also resolve the
serve-shutdown SIGSEGV (tep#175). serve `--device cuda` likely folds into this.

## Filed hand-offs (read-only; owned by their repos)
- tep#168 (OpenAI battery convergence — Part A safe-now done by tep), tep#172
  (SQLite int64 truncation), tep#175 (Tep.run! SIGSEGV on SIGTERM).
- toy#30 (reciprocal: adopt tep Backend) = our #8.

## Hard-won lessons this session (also in gx10 memory)
- **NEVER `git add -A` in a Workflow execute step** — it swept ~680 MB CUDA
  build binaries into a commit (github's 100 MB limit caught it). Stage explicit
  paths + a `>5 MB` pre-commit guard. `.gitignore` is now glob-based.
- **NEVER run two non-isolated Workflows in one working tree** — they collide on
  `git checkout -b`. Sequential, or `isolation:'worktree'`.
- **Float gates are gx10-canonical**: byte-exact loss/logprob baselines
  reproduce on aarch64-linux only; cross-platform gate the discrete invariant +
  tolerance (done in `507e160`). Discrete gates (token IDs) are portable.
