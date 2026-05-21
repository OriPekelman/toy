# Spinel issue: three blockers found while building a Ruby tokenizer

**Target repo:** `matz/spinel`
**Type:** Bug report (three findings, escalating severity)
**Reproducer commit:** [oripekelman/toy_ruby_neural_network@HEAD](https://github.com/OriPekelman/toy)
**Spinel HEAD tested:** `1e2394d` (also `7beeb54`, same behaviour)

## Context

Building a standalone Ruby BPE tokenizer (`lib/tokenizer.rb`) for
the project's LLM inference path. Three independent issues block
the full implementation. Reporting together because they're all
discovered by the same scaffold and may share a root cause.

## Issue 1: cross-class type inference widens Mat#nrows to sp_RbVal

### Symptom

Compiling a file that requires both `lib/transformer.rb` (which
defines `Mat`) and `lib/tokenizer.rb` (which has its own attr_readers)
produces generated C that fails to compile:

```
out.c:XXXX:5: error: aggregate value used where an integer was expected
  lv_ta = tnn_input_2d_f32(((void *)lv_sess), ((int)(lv_a->iv_nrows)), ...);
                                                ^
spinel: C compilation failed
```

### Root cause

`Mat`'s ivars are declared as `mrb_int iv_nrows; mrb_int iv_ncols`
when only `lib/transformer.rb` is loaded. But adding
`require_relative "../lib/tokenizer"` causes the analyzer to widen
them to `sp_RbVal iv_nrows; sp_RbVal iv_ncols`. The `(int)(...)`
cast then tries to cast a `sp_RbVal` struct to int, which is illegal.

### Reproducer (minimal)

```ruby
# tinynn/_repro_min.rb
require_relative "../lib/transformer"
require_relative "../lib/tokenizer"
t = Tokenizer.from_gguf("data/smollm2-135m-tok.gguf")
puts t.vocab_size
```

Compile:
```
spinel tinynn/_repro_min.rb -o /tmp/_repro
# fails with "aggregate value used where an integer was expected"
```

`lib/tokenizer.rb` shape:
```ruby
class Tokenizer
  attr_reader :vocab_size, :bos_id, :eos_id, :pad_id, :unk_id, :present
  def initialize(vocab, bos_id, eos_id, pad_id, unk_id)
    @vocab = vocab; @vocab_size = vocab.length
    @bos_id = bos_id; ...
  end
  def self.from_gguf(path); ...; Tokenizer.new(...); end
end
```

### Workaround applied

Decoupled `lib/tokenizer.rb` to use a new `lib/gguf_kv.rb` module
(metadata-only FFI binds, no Mat dependency). This sidesteps the
issue because the demo binary doesn't require both files together
in the same compilation unit.

### Suspected root cause (guess)

`Tokenizer`'s `attr_reader`s on integer ivars (`vocab_size`, `bos_id`,
etc.) get cross-referenced with `Mat`'s `attr_accessor`s on integer
ivars (`nrows`, `ncols`), and a poly-dispatch widens the union to
`sp_RbVal`. The `--rbs sig/` flag with explicit `Integer` types on
both classes does NOT prevent this.

## Issue 2: --rbs flag is advisory and doesn't pin types

Tested by adding a deliberately-wrong RBS:
```ruby
class Mat
  attr_accessor nrows: String   # wrong on purpose
end
```

Spinel still compiled (silently). The RBS file has no effect on
analyzer output — the error from Issue 1 persists with or without
`--rbs sig/`.

This may be intentional ("advisory" per `spinel --help`) but it
means the workaround for Issue 1 cannot be done declaratively. If
RBS were strictly enforced, the user could pin Mat's ivars as
Integer and avoid the polymorphic widening.

### Suggested fix

Either:
- Enforce RBS as ground truth when present (recommended), or
- Document that RBS is hint-only + provide an alternate way to
  pin ivar types (e.g. typed-attribute pragma).

## Issue 3: :str FFI return aliases C-owned memory; large arrays segv

### Symptom

Reading 49 152 strings from a GGUF via a `:str`-returning FFI and
pushing them into a Ruby array, then constructing an object that
holds the array, segfaults inside the constructor.

### Reproducer

```ruby
# Loop reads 49 152 strings from a GGUF kv array.
n = GgufKV.tnn_gguf_arr_n(handle, "tokenizer.ggml.tokens")  # 49152
vocab = [""]; vocab.pop
i = 0
while i < n
  s = GgufKV.tnn_gguf_arr_str(handle, "tokenizer.ggml.tokens", i)
  vocab.push(s)   # or "" + s, or rebuild-via-chars — all segv
  i = i + 1
end
GgufKV.tnn_gguf_free(handle)   # invalidates ggml-owned pointers
Tokenizer.new(vocab, ...)       # → SIGSEGV inside SP_GC_ROOT(self)
```

The FFI binding:
```ruby
ffi_func :tnn_gguf_arr_str, [:ptr, :str, :int], :str
```

C side:
```c
const char *tnn_gguf_arr_str(void *handle, const char *key, int i) {
    return gguf_get_arr_str(ctx, key_id, i);  // ggml-owned const char *
}
```

### Tests

- ≤ 10 strings: works ✓
- All 49 152 strings, dummy via `vocab.push("tok_" + i.to_s)`: works ✓
- All 49 152 strings via FFI `:str`, then keeping `handle` alive: segv ✗
- All 49 152 strings via FFI `:str`, char-by-char rebuild: segv ✗
- All 49 152 strings via FFI `:str`, `"" + s` concat: segv ✗

The crash is in the GC scan over `@vocab` (which holds the array)
during `Tokenizer.new`'s `SP_POOL_NEW` call (which triggers GC).

### Suspected root cause

`:str` return values are wrapped as Ruby strings that share storage
with the C-side `const char *`. Ruby-side copy operations
(`s.dup`, `"" + s`, char-by-char) under Spinel do not produce a
true Ruby-owned copy — they appear to retain the alias to the
original C pointer.

If that's the case, freeing the GGUF handle invalidates every
"copy" in the array. But the crash happens even with the handle
still alive (just at large array size), suggesting the GC mark may
also be walking the underlying C bytes and tripping on something
else (alignment? pointer tagging?).

### Suggested fix

Either:
- Make `:str` FFI return values always copy into Ruby-owned storage
  (slower but correct), or
- Document the alias semantics + provide a `String.copy_from_cstr(p)`
  primitive that callers can use explicitly.

## Impact

These three issues together prevent shipping a full Ruby BPE
tokenizer in this project. The workarounds:

- Issue 1: decoupled FFI module (lib/gguf_kv.rb) — works.
- Issue 2: no workaround; just live with the looser typing.
- Issue 3: cap vocab at ~256 entries (demo-only); no production use
  until upstream fix.

Filing because each issue is independently reproducible and the
project would benefit from any one of them being fixed.
