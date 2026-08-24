# toy#150 — adaptive (Kolen–Pollack) DFA feedback: design + resume point

Status: **SHIPPED** (6e5616b + 29e4639, toy#150). This document is the
design record; the implementation landed as `--dfa-feedback kolen-pollack`
and is gated. The status line below used to read "NOT implemented" and was
never updated when the work shipped — corrected 2026-08-24.

Everything below was derived against the real code, not sketched.

The issue delegates the modelling choice ("the precise rule is your
call — pick what's cleanest and note it"). This document IS that note.

## The rule

For each feedback matrix `B_l`, the quantity it is standing in for is
the **effective output path** `P_l`: the map from the activation
`B_l` feeds back into, forward to the logits. Fixed-DFA freezes `B_l`
at a random draw and hopes the forward rotates toward it. Kolen–Pollack
instead drives `B_l` toward `P_l`.

The gradient of the loss w.r.t. that path matrix is exactly

```
∇_P L  =  e · a_outᵀ
```

where `e` is the output error (`[vocab, T]`, already in the graph as
`e_b`) and `a_out` is the activation at `B_l`'s **output width**. So
the update is

```
B_l  ←  B_l  −  η · (e · a_outᵀ)  −  η·λ · B_l
```

which is Kolen–Pollack with weight decay: the feedback receives the
same descent the forward path receives, and the decay is what makes the
two converge rather than merely co-move (Akrout et al. 2019).

**This is why it is cheap:** that is precisely `ggml`'s SGD step,
`opt_step_sgd(B, g, hp)` with `hp = [η, λ]` — no new kernel, no new
optimizer arm. `B` stops being an inert persistent input and becomes a
tensor with a step node, nothing more.

### Why not "B tracks the immediate forward weight"

The issue offers that alternative. It does not typecheck here: the DFA
is DIRECT (output error → layer), so `B_l` is `[vocab, d_out]` while
the immediate forward weight is `[d_in, d_out]`. There is no shape in
which one mirrors the other. The path formulation is the only one whose
dimensions close, and it is also the one the DKP literature uses for
direct feedback.

## Shapes, verified against the code

`tnn_matmul(A, B)` = `ggml_mul_mat` → contracts `ne0`, result
`ne = [A.ne1, B.ne1]`.

| tensor | allocation | `ne` | `d_out` | `a_out` (width `d_out`) |
|---|---|---|---|---|
| `b_ups[bi]` | `(sess, dfv, vocabv)` | `[vocab, dfv]` | `dfv` | `tap_as[bi]` ✅ exists |
| `b_glus[bi]` | `(sess, dfv, vocabv)` | `[vocab, dfv]` | `dfv` | `tap_as[bi]` ✅ exists |
| `b_downs[bi]` | `(sess, ew_b, vocabv)` | `[vocab, ew_b]` | `ew_b` | `o_i` ❌ **NOT TAPPED** |
| `b_rs[l]` | `(sess, nev, vocabv)` | `[vocab, nev]` | `nev` | `t_gates` ✅ exists |

The update node, mirroring `dfa_grad`'s existing transpose idiom:

```ruby
e_t   = tnn_cont_2d(sess, tnn_transpose(sess, e_b),   tv, vocabv)  # [T, vocab]
a_t   = tnn_cont_2d(sess, tnn_transpose(sess, a_out), tv, d_out)   # [T, d_out]
g_b   = tnn_matmul(sess, e_t, a_t)                                 # [vocab, d_out] == B
tnn_opt_step_sgd(sess, b, g_b, t_hp_fb)                            # hp = [eta, lambda]
```

`[vocab, d_out]` matches `B` exactly. Verified by hand against
`franken_moe_parts.rb:920` (`dfa_grad`) which builds the same shape one
transpose differently.

## The ONE missing piece

`o_i` — the expert down-projection output, `franken_moe_parts.rb:730`
(`o_i = tnn_matmul(sess, tw.pp[down_idx(l, ei)], a_i)`) — is not
tapped. Add `tap_os` beside `tap_as` (same layer-major `l * E + i`
indexing, same accessor/init/push pattern) in BOTH
`franken_moe_parts.rb` and its HAND-MIRRORED twin
`franken_moe_parts_cuda.rb`.

## Telemetry — and what already answers the question

The issue asks for `cos(B, forward)` to separate "B moved" from "B
moved toward the forward weights". Note first that **the existing
`align` event already separates them**: `cos(g_dfa, g_bp)` is high
exactly when `B ≈ P`, because `g_dfa = a_inᵀ(B e)ᵀ` and `g_bp` is the
true gradient. Its rising IS the result the ticket is after.

Add on top of it:

- `dfa_b_sig` — sum of squares over every `B`, in `run_start` and
  `run_end`. Proves `B` MOVES (the `experts_sig` / K4b discipline: a
  coupling that silently never fires looks identical on a loss curve).
  Under `fixed` the two MUST be bit-identical.
- `b_cos_head` — a DIRECT `cos(B, P)`, available in one place exactly:
  the **last** layer's `b_downs`, whose path to the logits is just the
  tied head `pp[0]`. Everything earlier has intervening blocks in its
  path, so the metric is scoped to that one matrix and must be labelled
  as such rather than presented as a global alignment.
  Index mapping (ggml `ne0` fastest): `B.ne = [vocab, dm]`,
  `pp[0].ne = [dm, vocab]`, so `B.flat[d*vocab + v]` pairs with
  `pp0.flat[v*dm + d]`. Only valid when `ew_b == dm` (no `--moe-latent`).

## Implementation order

1. `tap_os` in `franken_moe_parts.rb` + the CUDA twin (hand-mirrored).
2. Env + guards in `train_franken_moe_cli.rb` + CUDA twin:
   `FRANKEN_DFA_FEEDBACK` (`fixed` default | `kolen-pollack`),
   `FRANKEN_DFA_FEEDBACK_DECAY` (λ ≥ 0), `FRANKEN_DFA_FEEDBACK_LR`
   (η, default: tie to `--lr`). `kolen-pollack` requires a DFA policy
   (`dfa-experts`) — reject on chain/bp-spine, fail loud.
3. A PERSISTENT `t_hp_fb = tnn_input_1d_f32_persistent(sess, 2)`
   allocated BEFORE `finalize_weights` — the toy#133 landmine: a
   compute-context input allocated later reads ZEROS in silence, which
   here would mean η=0, i.e. a KP arm that quietly never adapts.
4. The update nodes, wired only under `kolen-pollack`.
5. `--dfa-feedback*` on the CLI + flag×recipe matrix row.
6. Provenance: `dfa_feedback` + decay + lr + the modelling note
   (`"path"`, so a bundle records WHICH rule ran).
7. Gate leg: `fixed` byte-null vs absent (structural); `dfa_b_sig`
   bit-identical under `fixed` and provably MOVED under
   `kolen-pollack`; `b_cos_head` RISES over a short run; determinism;
   composes with `--dfa-granularity block` / `--shape deep` /
   `--routing dense`; guards. **No claim about final loss** — that is
   F13's science, not the gate's.

## Landmines that apply here

- `t_hp_fb` persistent-before-finalize (item 3 above) — same class as
  toy#146's per-layer vectors.
- Both `franken_moe_parts*.rb` and both `train_franken_moe_cli*.rb` are
  HAND-MIRRORED committed twins. (Unlike `llama_arch_cuda.rb`, which is
  GENERATED — do not confuse them.)
- New gate leg → capture `n0 = failures.length` at the leg start
  (d878143); never summarise with `failures.empty?`.
