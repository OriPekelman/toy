# lib/toy/mri.rb — the MRI (CRuby) dev-run entrypoint. toy#71 Stage A+B.
#
#   ruby -Ilib -e 'require "toy/mri"'    # the whole compute surface loads
#
# WHY THIS EXISTS. `ffi_lib` / `ffi_func` / `ffi_cflags` are Spinel
# COMPILE-TIME intrinsics with no CRuby counterpart, so under plain MRI
# every `require` that transits lib/toy/ffi/tinynn.rb (i.e. the whole
# compute surface) dies with NoMethodError — even though the pure-Ruby
# teaching path (Mat, configs, RecipeOptions, RunLog, cards) never
# executes an FFI branch. This file gives the three intrinsics an MRI
# story, in TWO ARMS:
#
# STUB ARM (Stage A): DECLARATION RECORDERS whose generated methods raise
# a named, actionable Toy::MRI::NativeCallError. The pure-Ruby surface
# works end to end under MRI (livebook/tao dev-run cells, REPL pokes);
# anything that actually needs ggml fails loud at the exact call site.
#
# NATIVE ARM (Stage B — the CRuby oracle): when the shared library
# tinynn/libtinynn_ggml_shared.so exists (`make libtinynn_shared`),
# `ffi_func` binds the declaration against it via Fiddle instead of
# raising — the SAME toy program then runs real ggml compute under plain
# MRI. The 186 unique declarations ARE the spec; no second surface.
# Differential property (spinel-dev#6 phase 1): one program, two
# runtimes (MRI+Fiddle vs the Spinel-compiled binary) over the identical
# C library — divergence isolates a Spinel codegen bug, not a numerics
# question. Gated by prep/mri_gate.rb's native leg (train losses vs the
# recorded Spinel gate curve, KV-decode ids vs the infer baseline).
# Opt out with TOY_MRI_NATIVE=0 (forces the stub arm).
#
# NEVER COMPILED BY SPINEL. This file must not appear in any compiled
# require chain (lib/toy/compute*.rb, libexec runners, prep/smokes).
# The declarations themselves live in lib/toy/mri/shim.rb, which the
# files that declare native functions also load under CRuby (behind a
# RUBY_ENGINE guard Spinel resolves at compile time — toy#175), so
# `require "toy"` and `require "toy/compute"` work under CRuby too; this
# file remains the documented entry. No sig/ lockstep: Spinel never sees
# either file.

require_relative "mri/shim"

# With the DSL live, the full compute surface loads under MRI.
require_relative "compute"
