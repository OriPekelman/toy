# Spinel-landmine tech-debt audit (toy)

**Date:** 2026-05-25. **Purpose:** catalog every place in toy where we
work around a Spinel codegen/inference issue, so when Spinel ships a
fix we know exactly what tech debt we can pay back. The list also
serves as a "regression budget" — Spinel changes that touch any of
these patterns deserve a heads-up.

Spinel revision when audited: `master = d500374` (~2 h ago).
Toy revision: `dd4b131` at audit; `5ffea87` after comment refinement.

**Project-family Spinel pin: `ef3f1f3`** ("fix(codegen): explicit 0
return in compile_yield_inline_capture", ~5 h before audit). Tep
pinned here to avoid the `0ec6b1d (*StrHash _get returns NULL,
Phase 2)` runtime-behaviour change. Toy verified clean against this
pin: `make bench-heavy` reports all 8 metrics ok (-1.4 % on train
mean, within tolerance), all 5ffea87 builds reproduce identical
behaviour, no new warnings. Toy does not depend on the StrHash NULL
change either way (we use `Hash<String, Int>` everywhere, not
`<String, String>`), so the pin is cost-free for us.

## Inventory of workaround patterns

### 1. Type-pinning empty arrays (seed-and-pop)

```ruby
# Toy currently writes:
@ft_weights = [TinyNNCuda.tnn_null_ptr]; @ft_weights.pop
out         = [0.0];                     out.pop
paths       = [""];                      paths.pop

# Plain Ruby would write:
@ft_weights = []
out         = []
paths       = []
```

**What it fixes:** without the seed, Spinel infers `[]` as
`sp_IntArray *`. Pushing tensor pointers (8 bytes) or floats
(8 bytes) into an IntArray (4-byte slots) causes silent corruption
at runtime — this is exactly the bug fixed today
(`dd4b131` for `t_k_per_kv` / `t_vt_per_kv`).

**Site count:** ~94 across the codebase (63 in `lib/*.rb`, 31 in
`tinynn/*.rb`). High-traffic files:
- `lib/llama_seq_forward_ffi_cuda.rb` (12+)
- `lib/llama_seq_forward_ffi.rb` (12+)
- `lib/llama_seq_forward_ffi_metal.rb` (12+)
- `lib/toy_smollm2_ffi_kv_{cuda,metal,}.rb` (multiple)
- `lib/model_index.rb`, `lib/toy_smollm2.rb`
- `tinynn/ab_smoke_moe_ffn.rb`, `tinynn/ab_smoke_flash_attn.rb`

**Unlock condition:** Spinel infers `[]` as a polymorphic array OR
adopts first-push type pinning OR exposes a typed-empty syntax
(`Array<X>.new` etc.). Until then every new empty-array literal in a
Spinel-compiled file is a footgun.

### 2. `Hash#[missing_key]` returns 0/empty-string, not nil

```ruby
# Toy currently writes (lib/tokenizer.rb, ~10 sites):
if @merge_rank.has_key?(key)
  r = @merge_rank[key]
  if r < best_rank
    ...
  end
end

# Plain Ruby (and idiomatic Spinel-target) would write:
r = @merge_rank[key]
if r != nil && r < best_rank
  ...
end
```

**What it fixes:** Spinel's `*IntHash#[]` returns `0` on miss,
`*StrHash#[]` returns `""` — both are truthy by Spinel's coercion
rules and trip the missing-key check.

**Site count:** ~10 in `lib/tokenizer.rb`; a handful in other loader
code. (`lib/toy_smollm2_loader.rb`, `lib/toy_smollm2_ffi_kv.rb`).

**Unlock condition:** Spinel's `docs/HASH-NULLABLE.md` design doc
proposes a 5-phase fix landing `nil`-semantics for `*StrHash` (Phase
2-3) and `*IntHash` via `SP_INT_NIL` sentinel (Phase 4). Doc is in
Phase 1 (survey) as of audit. **The risk of Phase 4 cascading
through optcarrot is real — Phases 2-3 might land first.** When
Phase 2-3 ships we can drop the `has_key?` guards on the StrStr
side; Phase 4 unlocks the StrInt / SymInt side.

### 3. No `STDERR`

```ruby
# Toy currently writes:
puts "WARN: MoE expert weights are K-quantized…"

# Plain Ruby would write:
STDERR.puts "WARN: MoE expert weights are K-quantized…"
```

**Site count:** 7+ across `lib/tokenizer.rb`,
`lib/toy_smollm2_loader.rb`, `lib/toy_smollm2_ffi_kv.rb`.

**Unlock condition:** Spinel adds a `STDERR` constant bound to
file descriptor 2. Cosmetic — these are diagnostic prints that
should not be mixed into stdout when toy is run inside a pipeline.

### 4. No `File.basename` / `File.expand_path` / `Dir.entries`

```ruby
# Toy currently writes (via C shim in tinynn/tinynn_gguf.c):
TinyNN.tnn_list_dir_basenames(path)   # returns String[]
```

**Site count:** the shim is one file (~50 LOC C) but a real reason
for it. `lib/model_index.rb` uses it.

**Unlock condition:** Spinel adds these stdlib functions. Then we
drop the shim and replace with idiomatic Ruby.

### 5. No string interpolation in compiled code

```ruby
# Toy currently writes (104+ sites in lib/):
puts "step " + step.to_s + ": loss=" + loss.to_s

# Plain Ruby would write:
puts "step #{step}: loss=#{loss}"
```

**Site count:** ~104 `.to_s + …` manual concatenations across `lib/`.
Only 1 `#{...}` interpolation site in `lib/*.rb` — confirming it's
not in the Spinel-compiled subset (or developers avoid it).

**Unlock condition:** Spinel supports `#{expr}` interpolation
properly. Cosmetic but big readability win — manual concat is one of
the most-cited Spinel uglinesses.

### 6. No default arguments (`def f(x = nil)`)

Memory note (landmine #4): default args widen the parameter to
`RbVal` across the whole program, cascading into every caller.

**Site count:** Hard to count without a poly-inference report. The
absence is itself the signal — toy splits methods rather than using
defaults.

**Unlock condition:** Spinel handles default args without poly-
contagion. Tier-2 priority; the split-methods style isn't actively
painful, just verbose.

### 7. Local-variable name collision across files (landmine #12)

```ruby
# Toy's transformer.rb has a method:
def rms_norm_backward(value, rms)
  value / rms                                  # mrb_float divide
end

# Any OTHER file in the same compilation defining a local var
# named `rms` (e.g. a smoke iterating Newton's method on rms_err)
# widens transformer.rb's `rms` param to RbVal at C codegen.
```

**Workaround:** the smoke local was renamed to `rms_err` (committed
2026-05-23 per the memory). Not undone.

**Unlock condition:** Spinel uses scope-isolated names for cross-
module inference. Until then, smoke-author hygiene rule: use
project-unique local variable names.

### 8. `File.open ... do |f| ... end` + FFI session-init = runtime crash

Memory note (landmine #11): block-form File.open in a program that
also does `tnn_session_new + tnn_finalize_weights` segfaults at
runtime. Use `File.read(path).split("\n")` or `File.foreach` instead.

**Site count:** `lib/training.rb`, `lib/bpe.rb`, several
`tinynn/gpt2_*.rb` files still use the block form — those are
non-FFI contexts so they work. **Hazard:** any new Spinel-compiled
program mixing both is a footgun.

**Unlock condition:** Spinel fixes the FFI / block-form interaction.

## Other Spinel features we depend on

These aren't workarounds — they're "if Spinel's behaviour here
regresses, toy breaks". Listing them as awareness:

- `ggml`-backed FFI: `:int_array`, `:float_array`, `:ptr`, `:size_t`,
  `:str` parameter passing. Heavily relied on across tinynn FFI.
- `Mat` row-major flat storage via `mat.flat[i]` indexing. Toy's
  uploads pass `mat.flat` directly to C.
- Block-less `File.read(path).split("\n")` — used by `lib/training.rb`,
  `lib/bpe.rb`. Memory note 11 makes this the recommended form.
- `**` exponentiation on Floats (`(1.0 - (0.9 ** step.to_f))` in the
  bench). Verified working.
- The `1.0e-8`-style float literal. Verified working.

## Tech-debt reduction worth doing

Ranked by effort × value:

1. **Wait for `HASH-NULLABLE.md` Phase 2-3** (StrStrHash → NULL). Then
   drop the `has_key?` guards in lib/tokenizer.rb that target string-
   valued hashes. ~5 sites. Low risk, ~50 LOC delete.

2. **If Spinel adds STDERR** — swap the `puts "WARN:..."` lines to
   `STDERR.puts`. ~7 sites. Cosmetic but right.

3. **If Spinel adds File.basename / Dir.entries** — drop the tinynn
   C shim (`tnn_list_dir_basenames` etc.). ~50 LOC C delete + Ruby
   simplification.

4. **If Spinel infers `[]` as poly** — collapse all ~94 seed-and-pop
   sites back to `[]`. Big footprint, all-or-nothing change.

5. **If Spinel adds string interpolation** — the 104 manual concat
   sites would simplify but the gain is purely readability. Lowest
   priority; do as opportunistic cleanup, not a sweep.

## What we should NOT do until Spinel stabilises

- Don't rip out the seed-and-pop idiom on the BET that Spinel will
  fix `[]` inference. The fix path could go several ways and each
  imposes different requirements. The current workaround is ugly but
  reliably correct.
- Don't trust empty-array literals in new code. Every `[]` in a
  Spinel-compiled file is a potential landmine. The `t_k_per_kv` /
  `t_vt_per_kv` bug fixed today went undetected for an unknown number
  of days because the type mismatch was non-crashing at first.

## What landed in Spinel master in the last 22 hours (verified)

40 commits — extremely active. Highlights relevant to toy:

| Landmine | Spinel commit | Status |
| --- | --- | --- |
| #1 (empty `[]` → IntArray default) | `f292232` (issue #688) | **Partial.** Local var gets promoted on first `:ptr` push, but the function-parameter type is locked in BEFORE the promotion is observed. **Verified empirically**: reverting toy's `t_k_per_kv = [TinyNNCuda.tnn_null_ptr]; .pop` to bare `[]` re-fires the `sp_IntArray * vs sp_PtrArray *` warning. Seed-and-pop still required for cross-function passes. |
| #9 (Hash `[missing]` returns 0/"" not nil) | `0ec6b1d` (Phase 2) | **Partial.** `*StrHash _get` returns NULL now. `*IntHash` (which is what `lib/tokenizer.rb` uses) is **deferred** — see `ccbe436` "defer Phase 4, document cascade findings". Re-plan is the poly-wrapper route (boxes every lookup); cost unclear. has_key? guards stay. |
| (new) Empty `{}` literal | `19b81dc` | Empty `{}` coerces to expected hash variant. Cosmetic; we don't have many empty-hash literal sites. |
| #10 (Array<String> poisons poly dispatch) | `ac7720e` | Possibly fixed: "same-named attr_accessor on unrelated classes no longer widens reader to poly". Different surface but same root mechanism. Not yet re-tested against toy's lib/model_index.rb path. |
| (new useful) `fetch(k, nil)` on int-leaf hash | `7d3262e` | Returns `int?` with SP_INT_NIL. Could let tokenizer write `r = @merge_rank.fetch(k, nil); if r != nil` instead of has_key? + indexed read. Saves one branch per merge step; same semantics. |

### What this means for toy

- **Seed-and-pop stays.** The 94 sites in toy remain necessary. The
  fix is incomplete for our usage pattern (cross-function ptr array
  passes), and we shouldn't bet on a complete fix soon.

- **`has_key?` guards on `lib/tokenizer.rb` stay.** Phase 4 was tried
  today on Spinel and reverted after three cascade failures (self-host
  bootstrap, `&&=`/`||=` codegen using raw C truthy on SP_INT_NIL,
  existing tests that expected `0` output). The poly-wrapper re-plan
  has perf implications and isn't on a near-term track.

- **Possible incremental win**: rewrite the tokenizer's `has_key? ;
  then []` pairs to `fetch(k, nil)` + nil-check. Same semantics under
  Spinel `7d3262e`. Idiomatic Ruby and slightly more concise.
  ~10 sites in `lib/tokenizer.rb`. **Low risk, modest cleanup.**

- **The Tep regression suspect window**: with Spinel on a 40-commit/day
  cadence and several deep codegen/analyze changes (hash-variant
  conversions, exception lowering, kwargs binding), there are many
  candidates. Recent particularly suspicious for cross-project
  regressions:
  - `5399659 fix(codegen): wider-hash-variant return coerces to declared narrower variant` — return-coerce changes can silently change call shapes
  - `3d92a27 fix(codegen,analyze): cross-key-shape hash merge promotes to str_poly_hash`
  - `0ec6b1d fix(runtime): *StrHash _get returns NULL` — runtime behaviour change that could break Tep code that read string-hash misses as `""`
  - `63494c0 fix(codegen): sym_X_hash -> sym_poly_hash conversion at call boundary`

Spinel HEAD on this gx10 box is detached at `0ec6b1d` (the StrHash
NULL change), master is 3 commits ahead. If Tep is on `0ec6b1d`
and seeing the regression, the suspect is `0ec6b1d` itself or
something close to it. To bisect Tep's regression: compare against
the last known-good Spinel revision Tep was using.

## Concrete cleanup PRs to consider

Cheapest-first:

1. **`has_key?` → `fetch(k, nil)` in tokenizer** — leverages Spinel
   `7d3262e`. Same semantics, slightly more idiomatic. ~10 sites in
   `lib/tokenizer.rb`. Verify on next compile that the C output
   matches the has_key? path. No risk.

2. **Re-test landmine #10 (Array<String> poly poisoning)** after
   `ac7720e`. If the lib/model_index.rb workaround can be relaxed,
   simplify the file. May open back up an "Array<String> in main
   scope is OK" pattern.

3. Keep watching `f292232` for a follow-up that propagates the
   ptr-array promotion to function parameter types. That's the
   commit that, when it lands, lets us nuke the 94 seed-and-pop
   sites.

## What we should NOT do until Spinel stabilises

- Same as before. The seed-and-pop sites stay until #688 follow-up.
- Don't update Spinel HEAD without testing toy's bench-heavy gate
  first — the 40-commit-day surface has too many ways to break.
  Tep already flagged one regression; treat the gate as the proof.
