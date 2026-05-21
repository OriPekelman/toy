# Workaround audit — 2026-05-21 (post Spinel `f5bc710` bump)

Triggered by the Spinel update that landed five #626 sub-issue 2
commits (RBS arbitration) plus `04f8929 fix(codegen): FFI numeric arg
unboxes a poly value before the cast`. The latter is directly relevant
to our #626 #1 reproducer.

## Spinel state delta since last audit (2026-05-13)

| Closed | Title | Affected us? |
|---|---|---|
| #627 | rescue ParentClass / is_a?(ParentClass) class hierarchy | not used |
| #628 | yield typed Hash loses type at block-local | not used |
| #629 | Two same-shape sp_String ivars widen to sp_RbVal | maybe (lib/tokenizer.rb has many string ivars; never hit it) |
| #630 | Destructuring (parallel assignment) of `[UserClass, UserClass]` return value segfaults | **yes** — we have `Mat#plus` rename in transformer.rb worked around it (used to be `Mat#add` returning multiple). Re-test below. |
| #631 | is_a?(Hash)-narrowed value doesn't unbox at typed-Hash param call site | not used |
| #634 | Nullable param `= nil` widens to non-nullable | possibly affects lib/transformer.rb's `def Mat.from_array(arr, nrows=nil, ncols=nil)` family |
| #636 | Tep::Server::Scheduled segfaults under concurrent HTTP/1.1 keep-alive | **yes** — affects our tep_demo/openai_api_smollm2 serving path. Re-test below. |

| Open | Title | Affected us? |
|---|---|---|
| #626 | Three blockers… (`:str` fixed, sub-issue 1 partial via `04f8929`, sub-issue 2 phased) | **yes — sub-issue 1 partially fixed** |
| #638-639 | RBS over-strictness regressions from sub-issue 2 work | not used (no RBS files in tree) |

## #626 #1 status after `04f8929` + `f5bc710`

Tested project repro `tinynn/_spinel_626_issue1_repro.rb` against the
new compiler:

- BEFORE: error in FFI cast — `tnn_input_2d_f32(... (int)(lv_a->iv_nrows) ...)` failed
  "aggregate value used where an integer was expected" because Mat#nrows
  was widened to `sp_RbVal` but the cast assumed `mrb_int`.
- AFTER (`f5bc710`): that specific FFI cast site now compiles cleanly
  (the new commit unboxes the poly value before the cast).
- Remaining: `Mat#initialize`'s `@flat = Array.new(nrows * ncols, 0.0)`
  hits a SECOND poly-widening: `sp_poly_mul(lv_nrows, lv_ncols)` returns
  `sp_RbVal`, then tries to initialize `mrb_int _n` from it — same
  type-mismatch family, different code path (non-FFI internal use of
  the widened ivars).

So `04f8929` is a partial fix — the FFI surface is no longer the
trigger, but every NON-FFI use of `nrows*ncols`, `nrows+ncols`, etc.
inside Mat methods that the compiler hot-paths into `mrb_int` still
fails when Tokenizer is co-loaded.

Phase 0.6 re-trigger condition #3 stays open. Will follow up on
#626 with this narrower repro.

## Workarounds that are simplifiable now

None this session. The two main candidates were:

1. **Drop `lib/gguf_kv.rb` decouple, merge tokenizer back into the
   tinynn compilation unit** — blocked by the `sp_poly_mul` widening
   above. Tokenizer-only binaries can stay loose-coupled until #626 #1
   fully fixes.
2. **Revert `Mat#plus` to `Mat#add`** — was a workaround for #630.
   With #630 closed we can attempt the rename, BUT the original
   conflict was with Spinel's arg-type-narrowing on the name `add`
   used by multiple receivers (per the comment at
   `lib/transformer.rb:154-156`), not specifically #630. Test
   deferred — low-priority cosmetic.

## Workarounds that are still needed

| Workaround | Where | Spinel bug it dodges |
|---|---|---|
| `lib/gguf_kv.rb` decouple from tinynn      | `lib/tokenizer.rb:21`                                    | #626 sub-issue 1 (Mat#nrows widening) |
| Mat ivar names + literal-seed pattern      | `lib/transformer.rb:252,488,492,740` etc                 | iterative type inference name-collision |
| `Mat#plus` rename (no longer `Mat#add`)    | `lib/transformer.rb:154`                                 | arg-type-narrowing on shared method name |
| `Result` wrappers instead of tuple returns | `lib/transformer.rb:432`                                 | tuple-return type loss |
| Class-name mirror discipline (`*Cuda`)     | `lib/toy_smollm2_ffi_kv_cuda.rb:...`                     | class-collapse across compilation units |
| Lazy byte-table build in Tokenizer         | `lib/tokenizer.rb` `build_byte_tables`                   | Spinel edge case with large ivars+vocab |
| Per-binary GGUF path for OpenAI demos      | `tep_demo/openai_api_qwen25_*` (Makefile note)           | env-var constant mistyping |
| `require_relative "toy_smollm2_loader"` order constraint | `lib/toy_smollm2_ffi_kv.rb` header               | GC-mark crash in decode_step at certain require-order |

All of these are documented inline at their use sites with the
filed-upstream issue number where applicable.

## Regression check at f5bc710

After bumping Spinel and rebuilding:

- `make qwen25_acceptance` → 4/4 PASS  (Phase 0.7 gates green)
- `make smollm2_lora_forward` → PASS  (LoRA-B=0 bit-identical to baseline)
- `tinynn/ab_smoke_train_micro{,2,3,4,5}` → all PASS  (F1.1 micros)
- `demos/smollm2_lora_train_ce` → FAIL  (gate FAIL is correct — task #70 CPU sched bug)
- `demos/smollm2_lora_train_ce_cuda` → PASS  (CUDA training works)
- `demos/smollm2_lora_train_ce_pinned` → PASS  (task #70 diagnostic)

No regressions from the Spinel bump.

## Conclusion

Spinel `f5bc710` is a strict improvement over `6513d2d` for our use
cases. The big-ticket item is the partial #626 #1 fix (FFI cast path)
which doesn't yet unblock Phase 0.6, but narrows the remaining surface
to non-FFI internal uses of widened ivars. We carry no new
workarounds; one workaround (Mat#plus revert) is now technically
attemptable but not session-priority.

Next: file the ggml-cpu sched-aliasing issue upstream (task #71).
Continue F1.2 / F2 work on CUDA-side training where the chain works
correctly.
