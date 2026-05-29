# Session resume — 2026-05-29

> **Read this first** at the start of the next session. It pins
> exactly where we are in the toy-framework refactor and what the
> next concrete move is. ≤ 5 minutes to load full context.

## Where we are

**Phase status:**

| Phase | Status | Commit |
| --- | --- | --- |
| P0 — Design lock + Tao coord | done (design + roadmap merged) | bc1b40e |
| P1 — Card derivation refactor | **done** | 07aa547 |
| **P2 — Five-layer refactor of stdlib** | **0.0–0.2 done; 0.3 next** | ceb4aef |
| P3 — Core + CLI MVP | pending | — |
| P4 — CLI complete | pending | — |
| P5 — Generators | pending | — |
| P6 — Prism lowerer | pending (optional) | — |

**What's actually on disk:**

- `docs/roadmap/toy-framework-design-2026-05-28.md` — design v3 (locked).
- `docs/roadmap/toy-framework-roadmap-2026-05-28.md` — phase plan with
  P2.0 survey findings + mirror decision recorded inline.
- `lib/toy/llm/{primitives,blocks,archs,recipes}/README.md` — empty
  directories with contract sketches (P2.2).
- `lib/toy_describe_flow.rb#card(sess)` — runtime Card derivation
  (counts only, per-step bind deferred; see Spinel landmine note).
- `examples/smoke_card_derive.rb` — verification gate, passing.

## The next concrete move: P2.3 — extract RoPE as the pilot L1

**Why RoPE first** (re-confirmed from P2.0 survey):

- 8-arg parameter signature → encapsulation actually pays.
- Pure function, no weights → no ivar typing landmines.
- Used in two places (K path + Q path) → DRY win is visible.
- RMSNorm is too thin (one tnn call = rename theater); GQA is too
  thick (the head loop is half the block). RoPE = Goldilocks.

**Exact call sites in the monolith** (`lib/llama_seq_forward_ffi.rb`):

| Where | Line | What |
| --- | --- | --- |
| K path inside `build_seq_block` | 1765–1772 | `tnn_rope_ext` on K |
| Q path inside `build_seq_qhead` | 1845–1852 | `tnn_rope_ext` on Q |

Both follow the same shape-lift idiom:

```ruby
tb = @seq_t * @seq_b
t_pre3 = TinyNN.tnn_reshape_3d(@sess, t_pre, @seq_d_head, 1, tb)
t3     = TinyNN.tnn_rope_ext(@sess, t_pre3, @t_seq_positions,
                              @seq_d_head, @seq_rope_base,
                              @seq_rope_scaling.freq_scale,
                              @seq_rope_scaling.ext_factor,
                              @seq_rope_scaling.attn_factor,
                              @seq_rope_scaling.beta_fast,
                              @seq_rope_scaling.beta_slow,
                              @t_seq_rope_freq_factors)
t      = TinyNN.tnn_reshape_2d(@sess, t3, @seq_d_head, tb)
```

**Target after extraction:**

```ruby
# new in lib/toy/llm/primitives/rope.rb
module Toy::LLM::Primitives::RoPE
  NAME = :rope

  # Cfg value object — concrete fields, Spinel-safe.
  class Cfg
    attr_accessor :d_head, :base,
                  :freq_scale, :ext_factor, :attn_factor,
                  :beta_fast, :beta_slow

    def initialize(d_head, base, freq_scale, ext_factor,
                   attn_factor, beta_fast, beta_slow)
      # ... assign all 7
    end
  end

  # Applies the shape-lift + rope + un-lift in one call.
  # Returns the rotated 2D tensor.
  def self.apply_2d(sess, t_pre, positions, freq_factors, cfg, t_seq, t_batch)
    tb = t_seq * t_batch
    t_pre3 = TinyNN.tnn_reshape_3d(sess, t_pre, cfg.d_head, 1, tb)
    t3     = TinyNN.tnn_rope_ext(sess, t_pre3, positions,
                                 cfg.d_head, cfg.base,
                                 cfg.freq_scale, cfg.ext_factor,
                                 cfg.attn_factor, cfg.beta_fast,
                                 cfg.beta_slow, freq_factors)
    TinyNN.tnn_reshape_2d(sess, t3, cfg.d_head, tb)
  end
end
```

The Q and K call sites then collapse to:

```ruby
t_k = Toy::LLM::Primitives::RoPE.apply_2d(
        @sess, t_k_pre, @t_seq_positions, @t_seq_rope_freq_factors,
        @seq_rope_cfg, @seq_t, @seq_b)
```

The cache class needs an `@seq_rope_cfg` ivar built once at realize
time (currently the params are spread across `@seq_rope_base` +
`@seq_rope_scaling.*`). That's a 5-line constructor refactor in
`realize_for_mmap` + the 3 other realize paths.

## How to execute P2.3 (step-by-step)

1. **Write `lib/toy/llm/primitives/rope.rb`** with the module + Cfg
   above. Don't `require` it from anywhere yet.
2. **Add the require + the `@seq_rope_cfg` ivar build** at the END
   of each of the 4 realize paths in
   `lib/llama_seq_forward_ffi.rb` (lines 222 / 512 / 810 / 1046).
3. **Replace the two `tnn_rope_ext` call sites** (1765–1772 and
   1845–1852) with `RoPE.apply_2d(...)`. Remove the surrounding
   `reshape_3d` / `reshape_2d` — they moved into the primitive.
4. **Add `lib/toy/llm/primitives/rope.rb` to `MIRRORABLE`** in
   `prep/gen_cuda_mirror.rb` with a `subs_for` case (just the
   `common_module_tail` should be enough — no class names to
   rename, the module is `Toy::LLM::Primitives::RoPE` which doesn't
   collide with `TinyNN`). Add the require-relative rewrite for the
   `llama_seq_forward_ffi*.rb` mirror.
5. **Regenerate mirrors:** `ruby prep/gen_cuda_mirror.rb`.
6. **Smoke battery:**
   - `make examples/smoke_card_derive && ./examples/smoke_card_derive`
   - Whatever example(s) exercise the seq-mode forward — pick the
     fastest one that hits the rope path, run it before and after,
     compare logits byte-by-byte. If you can't find a deterministic
     fixture in 10 min, build one as part of P2.3.
7. **CUDA smoke too** — the mirror path is the riskier surface.
8. **Commit** with title `P2.3: extract RoPE as pilot L1 primitive`.

## Spinel landmines to step around

These are the active ones — all four bit me or other recent passes
in the last 2 weeks. Re-check `memory/feedback_spinel_type_inference_landmines.md`
before writing code.

| Landmine | Detection | Workaround |
| --- | --- | --- |
| **Default args poison** (#4) | Spinel widens the param type permanently. | Explicit overloads or always-supplied args. |
| **`Hash[missing_key]=0`** (#9) | Silent failure. | Pre-init Hash entries before increment. |
| **Array<Array<mixed>> destructure** | `a, b = pair` widens fields. | `pair[0]`, `pair[1]` direct index. |
| **`step_bind` with FFI :str args** (NEW 2026-05-28) | Static-init SIGSEGV. | Don't pass FFI `:str` returns to seeded-typed builders. For RoPE this isn't an issue (no Card calls inside the primitive itself). |

For the Cfg class, the `initialize` will have 7 args — that's a lot,
but Spinel handles fixed-arity ctors fine. **Do NOT** add optional
defaults to the Cfg ctor (default-arg poisoning).

## Gate (what "P2.3 done" means)

- `lib/toy/llm/primitives/rope.rb` exists, ~40 LOC.
- Both rope call sites in the monolith are gone (grep for
  `tnn_rope_ext` should only match `vendor/` and the primitive).
- Mirrors regenerated; the two `_cuda.rb` / `_metal.rb` files build
  clean.
- Forward parity: a chosen deterministic example produces
  byte-identical logits before vs after. Bit-identical, not "close".
- `make verify-mirrors` (`prep/gen_cuda_mirror.rb --verify`) passes.

If any of those fail, the right move is `git reset` and re-plan —
not patching around the failure.

## After P2.3

The order: SwiGLU → RMSNorm → GQA → L2 TransformerBlock → L3
LlamaArch (P2.4 / P2.5). At each step the same gate applies:
bit-identical logits + clean mirror regen + no Spinel warnings.

The realize paths (the 1750-line bulk) become tractable once the
block is the unit of allocation. That's a P2.6/P2.7 concern, NOT a
P2.3 concern.

## What NOT to do in the next session

- Don't try to extract multiple primitives in one commit.
- Don't touch the realize paths until L1 + L2 are clean.
- Don't add optional-arg defaults to Cfg ctors (Spinel landmine #4).
- Don't pin Spinel (durable: this project is a Spinel test-case).
- Don't open a PR yet — keep landing on main, push to origin when
  ready.

## Open questions for the user (optional; only if relevant comes up)

- Tao-side issue for `runs/<id>/` layout coordination is still
  pending (per "no intermediary tickets" we'll batch all issues at
  P2 close).
- The deferred per-step Card detail (the step_bind landmine
  workaround) — restore via C-side `tnn_card_push_step` or via
  Card variant? Defer to P5/P6 unless it blocks something.

## Reference paths

- Monolith: `lib/llama_seq_forward_ffi.rb`
- Mirror generator: `prep/gen_cuda_mirror.rb`
- Card IR: `lib/toy_card.rb`
- Card derivation: `lib/toy_describe_flow.rb#card`
- Design: `docs/roadmap/toy-framework-design-2026-05-28.md`
- Roadmap: `docs/roadmap/toy-framework-roadmap-2026-05-28.md`
- Landmines: `~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/feedback_spinel_type_inference_landmines.md`
- step_bind landmine: `~/.claude/projects/-home-oripekelman-sites-toy-ruby-neural-network/memory/project_step_bind_landmine_2026_05_28.md`
