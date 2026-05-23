# Lowerer evidence (2026-05-23)

Measured the Lowerer's three claimed benefits before committing to
the ~500-LOC build phase. Verdict: don't build it as spec'd — there's
a cheaper alternative for the perf claim (FFI the hot Mat ops) and
the other two benefits don't independently justify the architectural
tax.

## Methodology

- Instrumented six pure-Ruby `Mat` methods (`matmul`, `matmul_t`,
  `t_matmul`, `plus`, `add!`, `scale!`) in `lib/transformer.rb` with
  `tnn_trace_begin/end` from the Day-1 trace primitive. Zero-cost
  when off (1.44 s baseline → 1.44 s with traces present but
  inactive). 12 % overhead when on, which is acceptable for
  diagnostic runs.
- Added a `MAT_SHAPES=1` flag that prints a shape triple per matmul
  call (`MAT_SHAPE matmul M N P`) so we can build a shape histogram
  by `sort | uniq -c`.
- Ran `examples/example_train` (1 epoch over 87 TinyStories
  sequences, GPT-2-style native-Ruby path; D_MODEL=32, D_FF=64,
  N_HEADS=4, N_LAYERS=2, CONTEXT=64) under both traces.

## (1) Perf benefit — real but cheaper to capture another way

Top of step wallclock:

| op            | count   | total (ms) | mean (μs) | % of step |
|---------------|--------:|-----------:|----------:|----------:|
| `Mat.matmul`   |   4 785 |    368.26  |    76.96  |   27.64 % |
| `Mat.matmul_t` |   4 089 |    369.21  |    90.29  |   27.71 % |
| `Mat.t_matmul` |   3 393 |    364.45  |   107.41  |   27.36 % |
| `Mat.add!`     |   2 175 |     17.55  |     8.07  |    1.32 % |
| `Mat.plus`     |     696 |      7.31  |    10.50  |    0.55 % |
| `Mat.scale!`   |   1 064 |      2.40  |     2.26  |    0.18 % |

**82.7 % of step time is in the three matmul flavours.** Add/plus/scale
are 2.05 %. Everything else (loss, optimizer step, softmax, embedding
lookups, LayerNorm, GeLU) is the remaining ~15 %.

Shape concentration: 15 228 matmul calls across 512 unique shape
triples, but the inner-loop dimension (`N` × `P`, the parts that
matter for compile-time specialization) clusters tightly:

| op        | top (N, P) pair  | calls | % of op |
|-----------|-----------------:|------:|--------:|
| matmul    | (32, 8)          | 3 608 | 50.7 %  |
| matmul_t  | (8, 32)          | 2 152 | 46.0 %  |
| t_matmul  | (32, 8)          | 2 128 | 62.0 %  |

The `M` dim (output row count) is just sequence length T, which
varies 2–64 across sequences. So a Lowerer specialising on `(N, P)`
and leaving `M` dynamic would still see >50 % hit rate on every
matmul.

**But:** the comparable alternative is to FFI the matmul through
`tinynn` → `ggml_mul_mat`, which is hand-tuned C with vectorisation.
Other ggml-bound paths in this repo see 38× speedups vs the same
pure-Ruby code (`project_m1_full_forward_shipped_2026_05_14`). A
Lowerer's shape-specialisation buys you 2–4×; FFI buys you 10–100×
for an order of magnitude less code.

## (2) Spinel landmine relief — measurable but modest

Counts in `lib/`:

- **46** `.pop` sites (mostly the seed-then-pop pattern, e.g.
  `arr = [""]; arr.pop` to type-pin `Array[String]`).
- **2** `= "" + var` type-pin sites (in `lib/model_index.rb` only).
- **9** documented Spinel codegen landmines
  (`feedback_spinel_type_inference_landmines.md`).

A Lowerer with RBS-style typed arrays would eliminate the seed-pop
sites and probably most of the type-pin sites. That's a real win
for code legibility — the pattern looks weird, surprises new
contributors, and is documented in the memory file. But 46 sites in
a working codebase isn't *expensive*; it's a recognisable papercut,
not a structural problem. With Spinel master `d59926a` improving
RBS arbitration (`97bf268`), some of these papercuts will heal
without our doing anything.

## (3) Algorithm-card emission — real, mostly orthogonal

Counts:

- **~20** `algorithm` / `algorithm_card` / `algorithm_card_full`
  methods across `lib/toy.rb`, `lib/toy_gpt2.rb`, `lib/toy_smollm2.rb`.
- **209** LOC total inside those methods.

These methods describe the model's forward pass in
Phuong–Hutter card form (pseudocode + shapes). They're
hand-maintained today. A Lowerer that traces Mat ops at build time
could emit cards automatically, deleting all 209 LOC.

But: this benefit is mostly orthogonal to the perf/landmine claims.
A standalone "card emitter" tool that doesn't touch the runtime
would capture (3) with maybe 100 LOC of Prism-walker code and zero
architectural-tax to the rest of the codebase.

## Decision

Skip the full Lowerer. Instead, pursue the three benefits separately
with proportionate tools:

1. ~~**Perf** → FFI-wrap `Mat#matmul`, `Mat#matmul_t`, `Mat#t_matmul`
   (and the cache classes). Estimated 5-10× on `example_train`
   wallclock, ~100 LOC, no architectural change. Tracked as
   **P2** (new task).~~ **MEASURED 2026-05-23: estimate was wrong**
   — session-per-op FFI is 1.7× *slower* at training-toy shapes
   (32×8) because session lifecycle (~180 µs) dwarfs the matmul
   (~77 µs). See `p2-ffi-matmul-2026-05-23.md`. The 38× number
   cited above is for whole-graph FFI (the cache pattern), not
   per-op. Real workloads already use cache pattern via
   `lib/llama_seq_forward_ffi.rb` etc.; nothing to do on those.
2. **Spinel landmine relief** → wait for upstream Spinel RBS
   maturation; chase items only when they re-bite. The pattern is
   already documented in the memory file.
3. **Algorithm-card emission** → if/when we want to delete the 209
   LOC, write a standalone card-emitter (Prism walker over
   `def forward(...)` bodies). No runtime impact. Tracked as
   **D1** (design task, not blocking).

The Lowerer-as-spec'd would do all three at once for ~500 LOC plus
ongoing build-phase complexity. The above splits it into three
right-sized pieces; nothing gets built that isn't paying its way.

Re-trigger conditions for revisiting the full Lowerer:

- Pure-Ruby Mat workloads become a primary path (currently every
  serious workload routes through FFI).
- Spinel landmines start eating significant dev time.
- Card maintenance falls behind the model implementation.
