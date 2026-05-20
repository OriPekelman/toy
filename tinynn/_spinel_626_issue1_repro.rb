# Spinel #626 sub-issue 1 — Mat#nrows widens to sp_RbVal when Tokenizer
# is co-loaded. Reproducer for matz/spinel#626 comment requesting a
# self-contained 2-file repro.
#
# Tested Spinel HEAD: 6513d2d (2026-05-20). Bug still reproduces.
#
# Run from the project root:
#   spinel tinynn/_spinel_626_issue1_repro.rb -o /tmp/_x
#
# Expected (current behaviour, repro target):
#   error: aggregate value used where an integer was expected
#   lv_tb_t = tnn_input_2d_f32(..., ((int)(lv_b->iv_ncols)), ((int)(lv_b->iv_nrows)));
#
# Expected (after fix): clean compile, no errors.
#
# Note (2026-05-20): building a from-scratch two-file minimal (Mat +
# Tokenizer in /tmp with synthetic FFI bindings) did NOT trigger the
# widening — the trigger requires the project's specific cross-class
# inference graph (lib/transformer.rb's full Mat surface + lib/tokenizer.rb's
# attr_readers + transitively required gguf_kv.rb / tinynn.rb FFI modules).
# Pull the project at HEAD and run from project root.

require_relative "../lib/transformer"
require_relative "../lib/tokenizer"

m = Mat.new(2, 3)
t = Tokenizer.from_gguf("data/smollm2-135m-tok.gguf")
puts m.nrows.to_s
puts t.vocab_size.to_s
