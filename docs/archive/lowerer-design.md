# Lowerer design — Roundhouse-style preprocessing for toy

This doc records the design we'd implement *if* we decided to lower
Toy's source before Spinel sees it. We haven't built it. It exists
because there's a concrete proposal on the table
([tep#6 comment by Sam Ruby][tep-6]) and because the structured
`Toy::Card` IR — built first for the `algorithm` / round-trip story —
is the natural output target for the lowerer when we do build it.

The two ideas are independent in implementation but converge on the
same IR.

## Background — Sam Ruby's pitch

Roundhouse is Sam Ruby's [Rails refactoring proposal][roundhouse]: turn
the polymorphic, string-keyed `params[:title]` bag into typed accessors
on per-controller classes (`ArticleParams.title`). The compilers (mruby,
Spinel) were already capable of optimizing typed code — what was
missing was input that wasn't deliberately polymorphic.

For toy, [Sam's adaptation][tep-6] is: keep `Mat` in the *authored*
source but lower it to per-shape specialized classes
(`TensorF32_2D_768x768`, `TensorF32_2D_768x3072`, …) at build time. The
"axis" is small — a real model uses ~20–30 distinct tensor shapes total
— so the lowerer's combinatorics are bounded.

Pragmatic shape: toy becomes a preprocessor + smaller runtime library,
not a runtime library alone. Build phase adds `make lower` between the
Spinel translate and `cc`.

## What this buys

A few wins:

1. **Spinel sees fully-typed input.** Every Mat operation specialises
   to known nrows/ncols/dtype. The `l_*` / `g_*` field-name prefixes,
   the literal-seed-then-push pattern, the careful avoidance of
   reassigned ivars — all of those exist because we're asking Spinel
   to recover shape information that the lowerer would just *make
   explicit*. Most of them retire.

2. **Per-shape matmul loops.** `tensor_768_3072.matmul(tensor_3072_768)`
   generates a specialized C function instead of dispatching through
   `Mat#matmul(Mat)`. The C compiler then has constant trip counts and
   can unroll / SIMD-ize. Expected: 2–4× over today's generic Mat
   inner loops at LLM shapes. (Caveat: we already FFI most hot matmuls
   to ggml — so this win lands in the *backward* and *training* paths,
   plus the inference-host scaffolding, not the matmul kernel itself.)

3. **Same producer for the algorithm card.** Walking the source to
   recover shapes also recovers the *structure*: assignments, loops,
   the data-flow graph. That structure is what `Toy::Card` already
   represents. So the lowerer's Prism walker can also emit Cards —
   replacing the hand-authored `algorithm` methods on each class.

## Cost

The reasons we haven't built it:

- ~500 lines of Prism-based walker for the lowerer itself, plus a
  symbol table, plus per-op specializers. Three weeks of work, not
  three days.
- A new build phase to maintain. `make lower` is one more thing that
  can go wrong; failures need to be diagnosable.
- The current code is reasonably clean post-Spinel #537/#538. We're
  not blocked on type-inference pain *right now*. The right time to
  pay this cost is when (a) a new shape makes the seed-call pattern
  intolerable, or (b) a perf measurement says the un-specialised
  matmul loops are the bottleneck.

## Sketch — what the build phase looks like

```text
lib/toy_smollm2.rb          (authored: uses Mat freely)
        │
        ▼   make lower
build/toy_smollm2_lowered.rb (generated: TensorF32_2D_576x576 etc)
        │
        ▼   make build/...
build/spinel-out/...        (Spinel translate output)
        │
        ▼   cc
demos/smollm2_kv             (binary)
```

Authored source stays pleasant Ruby. Lowered source is generated
boilerplate (huge but read-only). Both are checked into the repo so
debugging works in either layer.

## Walker shape

```ruby
# prep/lower.rb — sketch only
require "prism"

result = Prism.parse_file("lib/toy_smollm2.rb")
walker = Toy::ShapeWalker.new(symbol_table)
result.value.accept(walker)
walker.emit_lowered("build/toy_smollm2_lowered.rb")
```

The walker recognises:

- `Mat.new(R, C)` → infer concrete shape from arg expressions
  (use the symbol table to resolve `cfg.d_model`, `n_heads * d_head`,
  etc — these are statically computable for a fixed config).
- `x.matmul(y)`   → emit a call site to the specialized matmul for
  shapes `(R_x, C_x) · (R_y, C_y)`.
- `Mat#flat[i * d + j]` → preserve as-is (the lowerer doesn't change
  internal kernels, just call sites).

It does NOT recognize arbitrary Ruby. The lowerer is a
pattern-recognizer over Toy's API, not a general transpiler. Cases it
doesn't recognise → fall back to today's polymorphic Mat call,
unspecialised.

## How `Toy::Card` plugs in

The same walker that produces the lowered code can — with a different
visitor — produce a `Toy::Card` for each `def forward` method:

```ruby
class CardWalker < Prism::Visitor
  def initialize(card); @card = card; end

  def visit_local_variable_write_node(node)
    rhs = render_expression(node.value)
    shape = infer_shape(node.value)
    if @card.steps.any? { |s| s.var == node.name.to_s }
      @card.step_update(node.name.to_s, rhs, shape, "")
    else
      @card.step_bind(node.name.to_s, rhs, shape)
    end
    super
  end

  def visit_while_node(node) ; ... end
  def visit_call_node(node)  ; ... end
end
```

The Card is the *common interface*. Today each model writes its
`algorithm` method by hand using the Card builder API. With the
lowerer, that method becomes:

```ruby
def algorithm
  Toy::SourceWalker.card_for("Toy::SmolLM2", :forward)
end
```

— and the source itself is the algorithm. Round-trip parsing then has
a real target (the lowerer emits a Card; the parser produces a Card;
tree equality is the test). That's stronger than today's
parse-the-rendered-string-with-regex contract.

## Status

Not built. The `Toy::Card` IR is in `lib/toy_card.rb`; the model
classes use its builder API in their `algorithm` methods. If/when we
build the lowerer, it lands in `prep/lower.rb` and emits to the same
Card type.

[tep-6]: https://github.com/OriPekelman/tep/issues/6#issuecomment-4466841401
[roundhouse]: https://intertwingly.net/blog/2026/05/09/The-Compilers-Were-Ready.html
