# Toy as a framework: design doc (DRAFT)

**Status:** design doc for review. No implementation yet.
**Date:** 2026-05-28.
**Audience:** ourselves, Tao maintainers (layout coordination).
**Companion docs:**
[`lowerer-design.md`](lowerer-design.md),
[`spinelgems-tep-adoption-2026-05-27.md`](spinelgems-tep-adoption-2026-05-27.md),
[`backends-and-scale-2026-05-27.md`](backends-and-scale-2026-05-27.md).

## TL;DR

Promote toy from "library you `require_relative` from your own
Spinel-compiled experiment" to **"framework you create projects
inside of"** — `toy new myproj`, `toy train ...`, `toy serve ...`,
with a CLI that absorbs the per-project plumbing the lived Mac
fresh-clone walkthrough showed is currently friction.

The lever: **`Toy::Card` as the IR contract** between user code and
framework tooling. Cards already exist; today they're hand-written
on Arch classes. Promoting them to the contract surface that
Trainers, Decoders, Evals, and Servers also produce/consume lets
the framework compose pieces without baking ML-specific opinions
into Core.

Explicit non-goal in v1: a generic "Spinel project generator base
library". We may *accidentally* land something extractable; we will
not design *for* it.

---

## 1. Why now — the moment + the lever

Two observations are converging.

**(a) The lived Mac fresh-clone walkthrough** (2026-05-28) showed
that today's surface is too low-level. The user's path was: clone →
`make` (fails on vendor-tep) → `make help` → `make setup-ggml-metal`
(builds only metal libs) → `make example_inference` (fails: linker
can't find libggml from build-metal/) → `make example_list_models`
(builds; runs; "No GGUF models found" despite a successful
`prep/fetch_model.sh` of the right repo). Each of those is a
decision the user shouldn't have been asked to make.

**(b) `Toy::Card` is already an IR.** It exists at
`lib/toy_card.rb`. SmolLM2 + SmolLM2Block already build Cards from
their `algorithm` methods. The lowerer roadmap
([`lowerer-design.md`](lowerer-design.md)) is explicitly about
generating those Cards from Prism. The IR is real; what's missing
is the framework layer that consumes it.

Together: there's a CLI shape that eliminates the Mac pains *by
construction*, and a contract substrate that lets us add commands
without inheriting opinions.

The Rails-moment lever (DHH's original): convention removes
decisions the user shouldn't be making, but does so by composing
explicit primitives, not by hiding magic. We aim for the same — if
the convention doesn't fit, escape hatches everywhere.

## 2. Library vs framework (where toy is, where we want)

| | Library | Framework |
| --- | --- | --- |
| **You require it from** | your code | your project |
| **It owns the entry point** | no | yes (the CLI) |
| **It dictates layout** | no | yes (project dir conventions) |
| **It composes pieces for you** | no | yes (Arch × Trainer × Decoder × …) |
| **Toy today** | ✅ (gemspec shipped 2026-05-27, toy#19) | partial — `examples/*` play this role manually |

The lib-vs-example architectural rule from
[`feedback_lib_vs_example_scope.md`](../../.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_lib_vs_example_scope.md)
still holds: experiment-specific configs in `examples/` or `prep/`,
generic primitives in `lib/`. The framework layer doesn't change
that — it adds a *third* place: the **project directory**, which is
neither examples/ (toy's repo) nor lib/. It's the user's working
directory.

## 3. The Rails-moment ambition + the risks

**Ambition:** Toy could do for Spinel what Rails did for Ruby —
provide a credible, batteries-included on-ramp for an ML project,
with conventions that get out of the way when you outgrow them.

**The risks the user already named:**

1. **Over-opinionated framework paints into a corner.** ML hasn't
   converged on shapes the way web did on MVC/REST. A weird
   agent / MoE-music user shouldn't have to fork around our
   opinions to use Spinel-flavoured numerics.
2. **YAGNI on the meta-framework.** "A generic Spinel scaffold
   library" is tempting (Tep would also benefit from one). Resist:
   ML and web converge differently. Build a great toy first; if
   patterns crystallize that obviously generalize, extract later.
3. **Hand-rolled CLI vs Thor/dry-cli vs Spinel-compiled binary.**
   First cut should be CRuby (fast iteration). Spinel-compiling
   the CLI itself comes later if ever.

**The mitigations:**

- Core is **thin**. Stdlib is **modular** (drop any module). Algos
  are **gem-shaped**, not built-in. (See §6.)
- The contract is the **Card IR**, not behavioural interfaces
  (no "implement these 7 methods"). If your algo produces a Card,
  the framework can introspect / compose / serve it.
- Every convention has an escape hatch. `toy infer` resolves names
  via `toy list`, but `GGUF=...` still works. `toy serve` accepts
  raw paths.

## 4. Mac onboarding as design input

The walkthrough revealed not bugs (those got fixed in
a98b136 / d34e00d) but **friction points** — each is a CLI
invariant.

| Mac pain | Today's friction | Framework invariant |
| --- | --- | --- |
| `make` no-args | Fails on vendor-tep | `toy` (no args) → shows commands; never fails |
| Backend choice | User picks setup-ggml-{,cuda,metal} | `toy install` decides; user doesn't pick |
| Metal libggml needs Metal framework | CPU example linker fails | Framework links each example with the right framework set |
| HF cache symlinks invisible | `tnn_list_walk` used lstat | (fixed in d34e00d — invariant: `toy list` always sees what `toy fetch` put down) |
| `example_inference` default GGUF path missing after `fetch_model.sh` | Hardcoded `data/smollm2-135m-f32.gguf` | `toy fetch` symlinks into project's `data/<basename>.gguf` immediately; `toy infer <model>` resolves names |
| 4× `fetch_model.sh` invocations because user couldn't see model | Quiet success, no canonical lookup | `toy fetch` reports the resolved local path on stdout + `toy list` shows it next call |
| Repeated decisions per session | Per-target env vars (GGUF=, DEVICE=) | `toy.yml` project config + per-command `--flag` overrides |

These are the *acceptance criteria* for the CLI MVP. If any of them
still requires the user to make a decision they shouldn't, we
haven't shipped the framework yet.

## 5. Three-layer model — what's Core

```
┌─────────────────────────────────────────────────────────┐
│  Add-ons (community gems)                               │
│  toy-diffusion, toy-vision-tower, toy-music-moe, ...   │
├─────────────────────────────────────────────────────────┤
│  Stdlib (bundled but each its own module)               │
│  toy-ggml · toy-llm · toy-train · toy-serve · toy-bench │
├─────────────────────────────────────────────────────────┤
│  Core (the always-thin framework)                       │
│  CLI · project layout · Toy::Card IR · event stream     │
│  backend interface (NOT impls) · skill manifest         │
└─────────────────────────────────────────────────────────┘
```

**Core invariants:**

- ~500-1000 LOC.
- No GGML dependency at this layer.
- No Gemfile assumption.
- No Spinel assumption at the API level (Core could in theory power
  a pure-CRuby project — though the algos in stdlib will all need
  Spinel for performance).
- All commands return JSON when `--json` is passed; pretty otherwise.

**Stdlib boundary:** if you want to write an LLM, you pull
`toy-llm`. If you want training infrastructure, you pull
`toy-train`. If you're doing weird agents and want only the event
stream + project layout + CLI dispatch, you pull none of stdlib —
just Core.

**Add-ons:** same interface as stdlib modules. There is no privileged
status; stdlib is just "the modules that ship with toy by default".

## 5.5 Distribution + install path — `rv` / `rvx`

Spinel-Coop's [`rv`](https://github.com/spinel-coop/rv) is the
`uv`/`pipx` equivalent for Ruby — fast Ruby version + gem manager
from the same team that's investing in Spinel. Three install shapes
fall out naturally if toy ships as a gem with a `toy` binstub:

| Command | Shape | Use case |
| --- | --- | --- |
| `rvx toy install` | Ephemeral run (pull + execute) | One-shot: a new user wants to try toy without commitment |
| `rv tool install toy` | Persistent isolated install (`toy` on PATH) | "I'll be using this regularly" — same UX as a `brew install` |
| `rv clean-install` (in a project) | Install project's pinned Ruby + gems | Inside a `toy new`-created project, after `Gemfile.lock` lands |

**Why this matters for the framework story:**

- Removes the "which Ruby" decision from the on-ramp. `rv` installs
  the right Ruby version in ~2s if missing.
- Cross-platform consistency (macOS, Linux, Windows). The Mac
  fresh-clone walkthrough wouldn't have needed any platform-specific
  troubleshooting.
- Coordinates with Spinel-Coop's tooling direction — same team
  ships Spinel and `rv`, so adopting it isn't a bet on a stranger.
- CC-friendly: `rv tool install toy && toy --manifest` is two
  commands to get from "nothing" to "agent-discoverable surface".

**The distribution model becomes:**

- Toy is a published gem (`toy-framework` or just `toy` — naming
  TBD; current `toy.gemspec` already exists for the lib-vendoring
  story, may need a sibling for the CLI binstub or a single
  gemspec with both).
- The gem includes the `toy` binstub.
- Stdlib modules (`toy-llm`, `toy-train`, etc.) are separate gems
  the framework gem depends on by default.
- Add-on gems install via standard `gem install` or
  `rv tool install`; the framework discovers them via standard
  gem path walks (or an explicit `toy plugin add ...`).

**Risks to flag:**

- `rv` is new (active development, pre-1.0 territory). Adopting
  it as the *recommended* install path is a small bet. The fallback
  (`gem install toy`) always works for users without `rv`.
- Toy's gem must keep Ruby-version compat broad (3.2+) since `rv`
  manages whichever version the user lands on.

**Decision (PROPOSED):** ship toy as a published gem with a `toy`
binstub. Document `rv tool install toy` as the recommended
install path. Keep `gem install toy` working as a no-friction
fallback. Don't make `rv` a hard requirement.

## 6. Project layout (project dir, NOT framework dir)

`toy new myproj` creates:

```
myproj/
├── toy.yml              # project config (backend choice, data dir, ...)
├── algos/               # user algorithm classes — one per file
│   └── my_arch.rb       #   each registers itself on require
├── data/                # GGUFs (symlinks to HF cache or local), corpora
├── runs/                # event streams + checkpoints (Tao reads here)
│   └── <run_id>/
│       ├── events.jsonl
│       └── weights/{step_N.gguf, latest}
├── bin/toy              # thin binstub → resolves the framework
└── (no Gemfile required, no Rakefile, no Makefile)
```

**Coordinate with Tao** on the `runs/` layout. Tao already has
opinions about per-run subdirs. Lock this before `toy new` ships.

**The big design point:** the `toy` CLI runs in *this* directory, not
inside the framework's checkout. The framework lives in a gem
(via spinelgems vendor or system gem) and provides the binstub.

## 7. Algo class taxonomy

**Five kinds.** Each is a category of Card it produces, plus the
framework concept it slots into.

| Kind | Produces | Today's example | Framework slot |
| --- | --- | --- | --- |
| **Arch** | Forward-pass graph | `Toy::SmolLM2`, `ViTTinyConfig` | `toy infer`, `toy describe` |
| **Trainer** | Loop that updates an Arch's params | `examples/06_train_from_scratch.rb` | `toy train` |
| **Decoder** | Generation strategy on a realized Arch | KV-cache decode | (used by `toy infer` + `toy serve`) |
| **Eval** | Reads a checkpoint, emits eval events | `examples/08_lmc.rb` | `toy eval` |
| **Server** | Wires Arch + Decoder behind a transport | `tep_demo/openai_api_llama.rb` | `toy serve` |

**Status (OPEN):** is this taxonomy right? Two questions:

- Should **Trainer** and **Decoder** be one kind (both are "loops over
  an Arch")? Probably not — Trainer mutates weights, Decoder reads
  them. The mutation distinction is structural enough to keep them
  separate.
- Is there a sixth kind for **DatasetLoader**? Currently
  `lib/toy_corpus_loader.rb` and `prep/pretokenize_fineweb_edu.py`
  are split between Ruby and Python. A unified "data spec" might be
  its own kind. Lean: yes, sixth kind = **DataSpec**.

## 8. Card-as-contract spec (PROPOSED)

The minimum-viable algo class — sketched against today's Cards:

```ruby
class MyArch < Toy::Arch
  arch_name "my-arch"
  config :d_model, :n_layers, :n_heads, :vocab

  # Required: produce a Card for static introspection.
  def card
    Toy::Card.new(arch_name, kind: :arch)
      .add_input("x", "{1..V}^T", "token IDs")
      .add_hyper("D", cfg.d_model.to_s)
      .step_bind("e", "embed(x)", "[T, D]")
      # ... etc.
  end

  # Required: realize the forward graph against a backend session.
  # The framework allocates session + ctx_w (for the requested mode).
  # Returns a cache object exposing the documented public surface:
  #   sess, t_logits, t_token_ids, t_positions (Arch contract).
  def realize(cfg, sess, mode)
    # mode ∈ {:infer, :train}
  end

  # Optional: declare backend support. Default: framework infers
  # from operations used.
  backends [:cpu, :cuda, :metal]
end
```

Trainer:

```ruby
class FromScratch < Toy::Trainer
  trainer_name "from-scratch"

  def card
    Toy::Card.new(trainer_name, kind: :trainer)
      .add_param("optimizer", "AdamW")
      .add_hyper("lr", cfg.lr.to_s)
      # ... step pseudocode, dataset spec, schedule
  end

  # Framework calls this. Event-stream emission (step / drift /
  # sample) is automatic — wrap of `step` records the event.
  def step(arch_cache, batch)
    # one forward + backward + opt_step against arch_cache
  end
end
```

**Card-shape question (OPEN):**

- **Path A:** One `Toy::Card` type with optional fields per kind
  (current trajectory).
- **Path B:** `Toy::ArchCard`, `Toy::TrainerCard`, etc. as
  subclasses — separate field sets, separate renderers.

Recommend **path B** for Spinel-friendliness — one Card type with
optional fields invites poly-dispatch landmines (see
[`feedback_spinel_type_inference_landmines.md`](../../.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md)).

## 9. Composition / interop

Three composition axes, each orthogonal:

1. **Vertical**: Trainer × Arch × Decoder × Eval. Any trainer pairs
   with any arch matching its dataset spec (vocab, sequence shape).
   Card declares the shape; framework checks compat before running.
2. **Horizontal**: An Algo gem ships a coherent set (`toy-diffusion`
   provides an Arch + a Decoder + an Eval that reference each
   other). Or à-la-carte. Discovery via `arch_name` /
   `trainer_name` registration on require.
3. **Backend**: Set at session-creation time. Arch authors writing
   in `toy-ggml` primitives get multi-backend automatically. CUDA
   mirror generation for Spinel-poly workarounds is a build-time
   step the framework owns (`toy generate-mirror algos/my.rb`).

## 10. Assumptions each kind makes (the seams)

| Kind | Receives | Owns | Emits |
| --- | --- | --- | --- |
| Arch | cfg, sess, mode | forward graph, cache object | (none) |
| Trainer | arch_cache, batch | opt step | `step`, `drift`, `grad`, `sample` events |
| Decoder | arch_cache, prompt | autoregressive loop, sampling | token IDs |
| Eval | arch_cache, corpus | metric computation | `eval` events |
| Server | State, transport | endpoint dispatch | server-side `step`/`eval` events |

Critically: **none of these depend on toy-ggml directly**. They
use whatever session was built with — be it a future pure-Ruby
backend, ggml, or something else. The Card is backend-agnostic
authored-intent; `.describe` (graph walk) is what actually ran.

## 11. Multi-backend extensibility

Today's per-class mirror generation (CPU + CUDA + Metal mirrors
generated by `prep/gen_cuda_mirror.rb`) is a Spinel-poly workaround,
not a fundamental constraint.

Three pragmatic paths for add-on algos:

- **(a) CPU-only first.** Algo ships only the CPU path. User runs
  `toy generate-mirror algos/my.rb` to get CUDA/Metal mirrors when
  acceleration matters.
- **(b) Use only `toy-ggml` primitives.** Those already dispatch
  through `lib/tinynn{,_cuda,_metal}.rb`. Your algo is automatically
  multi-backend.
- **(c) Future:** if Spinel ships polymorphism, mirrors retire
  entirely. The Card stays valid regardless.

(b) is the recommended path; (a) is the fallback for algos that need
custom ops; (c) is dependence-on-upstream.

## 12. Prism lowerer integration (when built)

The lowerer ([`lowerer-design.md`](lowerer-design.md)) integrates as
a build step over `algos/*.rb`:

- `toy build` runs the lowerer → emits Spinel-friendly specialized
  Ruby → Spinel compiles → native binary.
- Same walker produces the algo's `Toy::Card` → registration is
  automatic; no hand-written `card` method needed.
- Algos opting out (custom Card) write it themselves (current
  pattern).

**Relationship to `.describe`:** Card = authored intent (static,
Prism-derived or hand-written). `.describe` = runtime graph walk
(actual execution shape). Two views of the same thing. `toy
describe my_algo` could use either path; framework checks they
agree.

Lowerer is *not* a prerequisite for the framework. Framework first,
lowerer when justified.

## 13. CC-tools-friendly defaults (minimal)

For Claude Code adoption, the framework needs:

- `toy <cmd> --json` everywhere — structured output for CC, pretty
  for humans.
- `toy --manifest` — JSON manifest of commands + args +
  descriptions. Enough for CC to discover the surface without
  parsing `--help`.
- Exit codes that distinguish "your input was wrong" (2) from "I
  failed to do the thing" (1).
- Stable error JSON shape (no flapping between runs).

Slash commands / skills are *wrappers around the CLI*, not separate
surfaces. Build the CLI; the wrappers follow.

## 14. The Tep-generator YAGNI question

The user raised: Tep would also benefit from a generator. Should we
extract a "base library for Spinel generators" that toy and tep
both consume?

**Argument for now:** doing it twice and extracting is the honest
path. Doing it generic from day 1 means designing without knowing
which abstractions hold.

**Argument against now:** even if Tep needs a generator, web-shape
projects and ML-shape projects diverge sharply. Tep's `tep new` would
scaffold routes/handlers/views (Sinatra-style). Toy's `toy new`
scaffolds algos/runs/data. The CLI commands barely overlap. A common
base would mostly be: argv parsing, command dispatch, JSON output,
skill manifest. That's ~200 LOC of orthogonal-to-the-domain
plumbing.

**Recommendation:** build `toy` first, in CRuby, in toy's own gem.
If after shipping we notice the dispatch/manifest/output plumbing
is genuinely the same across Toy and Tep, extract `spinelgen` (or
similar) and have both consume it. Not before.

Explicit non-goal in v1: don't design `toy` such that it
*precludes* a future extraction. Don't design *for* one either.

## 15. Decisions to lock first

In dependency order:

1. **Algo categories** — exactly the five (or six with DataSpec)
   from §7, or a different cut? Locking this defines the contract
   surface every algo authors against.

2. **Card-shape: path A (one type) vs path B (subclasses)?**
   Recommend **B**. Lock before refactoring Cards.

3. **Project layout coordination with Tao** — `runs/<id>/` shape,
   events.jsonl location, checkpoint subdir. Lock before `toy new`
   ships.

4. **What goes in Core, exactly?** Recommend the list in §5.

5. **`toy install` behaviour** — does it just build ggml? Does it
   detect+install Spinel if missing? Lock before the first
   onboarding pass against the new CLI.

## 16. Open questions for review

These are *not yet decided*; flagged for amend during review:

- **Q1.** Is the five-kind taxonomy right, or do we need
  DataSpec / others?
- **Q2.** Path A vs B for Cards.
- **Q3.** What `toy.yml` actually contains. Backend choice, data
  dir, default run-id template, default model, default
  algo-discovery path? Minimum viable subset?
- **Q4.** Coordination with Tao: does Tao read `runs/<id>/` from
  the toy project, or does Tao have its own dir? Locking this
  affects the project layout. Should we open a Tao-side issue
  asking?
- **Q5.** Is `toy build` (Spinel compile of algos/) the right
  command, or should compile be implicit on first `toy infer`
  / `toy train`?
- **Q6.** Naming + discovery: is `arch_name "llama"` enough? Or do
  algo classes need a richer manifest (version, deps, license)?
- **Q7.** First concrete deliverable: a `toy new` + MVP commands
  (install / list / fetch / infer / describe) that proves the
  on-ramp, OR a Card-refactor pass that turns `examples/06`,
  `examples/09`, etc. into proper `Toy::Trainer` subclasses so we
  *see the abstraction working under real pressure* before users
  meet it?
- **Q8.** Do we want the framework gem to vendor a known-good
  Spinel revision (pin) or always use whatever the user has? Lean
  toward pin (Spinel landmines are real; reproducibility matters).
- **Q9.** `rv` adoption depth: document as the *recommended*
  install path (proposed §5.5), or go further and *require* it for
  some commands (e.g. `toy install` shells out to
  `rv` for Ruby management)? Recommend the lighter touch —
  document `rv tool install toy` as the canonical install, keep
  `gem install` working, don't shell out to `rv` from the CLI's
  own code.
- **Q10.** Gem packaging: one `toy` gem with the CLI binstub +
  lib, or split `toy-framework` (CLI + Core) from `toy` (lib for
  vendoring, the current gemspec)? Affects how `toy.gemspec` and
  `consuming-toy.md` evolve.

## 17. Recommended sequencing

Assuming the above is broadly OK after review:

**Phase 0 — design lock** (this doc + Tao coord + Q1/Q2 answered).

**Phase 1 — refactor existing examples to the contract.** Take
`examples/06_train_from_scratch.rb`, `examples/09_warm_start_train.rb`,
`examples/03_finetune_lora.rb`, `tep_demo/openai_api_llama.rb` and
re-express them as `Toy::Trainer` / `Toy::Server` subclasses against
the proposed Arch contract. **Critically: this happens in toy's
repo, not yet in user-facing land.** It proves the abstraction
holds without committing the user-facing CLI to a shape that won't
survive pressure.

**Phase 2 — Core + CLI MVP.** `bin/toy` as CRuby. Five commands:
`new`, `install`, `list`, `fetch`, `describe`. No `train` / `serve`
yet — focus on "can a new user get from clone to identified model".

**Phase 3 — `toy infer`, `toy serve`, `toy train`, `toy eval`.** Each
composes the algo classes refactored in phase 1.

**Phase 4 — Generators (`toy g algo`, `toy g trainer`).** Only after
the patterns settle and we know what the boilerplate-to-replace is.

**Phase 5 — Prism lowerer integration** (if + when patterns from
phase 4 justify it).

Phases 1–3 don't paint into corners. Phase 4 is where opinions
start to bake; lock the contract first.

---

## Appendix A — known gotchas to encode in the framework

These are the lived realities the CLI should make invisible:

- The 7-file consolidation (toy#188) shows Spinel
  module-constant inference can be sketchy. The CLI should never
  *exec* a Spinel-compiled binary with env vars the binary's
  module constants depend on — instead, write a config file and
  pass its path.
- HF cache symlinks (fixed in d34e00d) — `toy fetch` should always
  also drop a `data/<basename>.gguf` symlink so the user sees the
  model land.
- `make example_inference` failing because libggml in build-metal/
  needs Metal framework — `toy install` should always build both
  CPU + accelerator libs on platforms that support them.
- The default GGUF path mismatch (`smollm2-135m-f32.gguf` doesn't
  exist after fetching Q8) — `toy infer <name>` always goes through
  `toy list` resolution; never a hardcoded path.

## Appendix B — what the CLI surface *might* look like (sketch)

Not locked. Illustrative.

```sh
# Install toy itself (one-time, via rv tool install — recommended)
brew install rv                   # or curl install
rv tool install toy               # `toy` now on PATH in an isolated env
# Alternatively:
#   gem install toy               # works without rv
#   rvx toy <subcommand>          # ephemeral; no install at all

# Bootstrap a new project (outside any toy repo)
toy new myproj && cd myproj

# Install the right backend for this host
toy install                       # auto-detect; or --backend cpu|cuda|metal

# Discover + fetch models
toy list                          # walks project data/, HF, Ollama, LM Studio caches
toy fetch bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf

# Describe a model (introspection via Card + GGUF metadata)
toy describe SmolLM2-135M-Instruct-Q8_0

# Run inference (CLI resolves model name through `toy list`)
toy infer SmolLM2-135M-Instruct-Q8_0 --prompt "Once upon" --max-tokens 32

# Serve over HTTP
toy serve SmolLM2-135M-Instruct-Q8_0 --port 4567

# Train from scratch
toy train --arch llama --trainer from-scratch \
          --data data/tinystories.bin --steps 200 \
          --d-model 64 --n-layers 2 --n-heads 4

# Evaluate
toy eval --ckpt runs/abc123/weights/latest --metric lmc \
         --other runs/def456/weights/latest

# Inspect events
toy events runs/abc123                  # tails the events.jsonl
toy events runs/abc123 --kind sample    # filter

# Discover the CLI surface (for CC etc.)
toy --manifest                          # JSON of commands + args
toy <any-cmd> --json                    # JSON output instead of pretty
```

All of these compose existing primitives. None require new ML
research. The framework is the on-ramp; the algos do the work.

---

_End of draft. Amend in place; mark resolved questions with
`[ANSWERED]`, contested ones with `[DEBATE]`, and add new ones as
they surface._
