# lib/toy/mri.rb — the MRI (CRuby) dev-run entrypoint. toy#71 Stage A.
#
#   ruby -Ilib -e 'require "toy/mri"'    # the whole compute surface loads
#
# WHY THIS EXISTS. `ffi_lib` / `ffi_func` / `ffi_cflags` are Spinel
# COMPILE-TIME intrinsics with no CRuby counterpart, so under plain MRI
# every `require` that transits lib/toy/ffi/tinynn.rb (i.e. the whole
# compute surface) dies with NoMethodError — even though the pure-Ruby
# teaching path (Mat, configs, RecipeOptions, RunLog, cards) never
# executes an FFI branch. This file gives the three intrinsics an MRI
# story: DECLARATION RECORDERS whose generated methods raise a named,
# actionable Toy::MRI::NativeCallError. The pure-Ruby surface then works
# end to end under MRI (livebook/tao dev-run cells, quick REPL pokes);
# anything that actually needs ggml fails loud at the exact call site.
#
# NEVER COMPILED BY SPINEL. This file must not appear in any compiled
# require chain (lib/toy/compute*.rb, libexec runners, prep/smokes).
# It is the CRuby side of the seam, like lib/toy/core/ — a SEPARATE
# entry file, NOT a conditional require (Spinel silently compiles
# conditional require_relative to 0 — spinel-dev#20, probed a699cf9).
# Consequently there is NO sig/ lockstep for this file: Spinel never
# sees it, so no .rbs mirror exists or is needed.
#
# STAGE B (toy#71, follow-on): the registry this file keeps
# (Toy::MRI.declarations / .libs / .cflags) is consumed by the Fiddle
# arm — `ffi_func` declarations ARE the spec the Fiddle backend binds
# against a shared libtinynn_ggml.so. Keep the registry shape stable.

module Toy
  # MRI-side runtime support for the Spinel FFI DSL.
  module MRI
    # Raised when MRI code calls a declared-but-unbound native function.
    # Named (not NoMethodError) so callers/tests can rescue it precisely.
    class NativeCallError < StandardError; end

    # Raised when the DSL shim would clobber an existing definition
    # (double-load, a foreign stub like tao's, or — impossibly — a
    # Spinel runtime). Fail loud, never silently redefine.
    class ShimCollisionError < StandardError; end

    @declarations = {}   # Module => [[name(Symbol), argtypes(Array), rettype(Symbol)], ...]
    @libs         = {}   # Module => [libname(String), ...]
    @cflags       = {}   # Module => [flags(String), ...]

    class << self
      # Public registry — Stage B's Fiddle arm consumes these.
      attr_reader :declarations, :libs, :cflags

      def record_lib(mod, name)
        (@libs[mod] ||= []) << name
        nil
      end

      def record_cflags(mod, flags)
        (@cflags[mod] ||= []) << flags
        nil
      end

      def record_func(mod, name, argtypes, rettype)
        (@declarations[mod] ||= []) << [name.to_sym, argtypes, rettype]
        nil
      end
    end

    # The deliberate NON-RAISING declarations — every other declared
    # function raises NativeCallError. Two families, both with an exact,
    # documented pure-value semantic (no computation is being masked):
    #
    # 1. `tnn_null_ptr` — returns a typed NULL `void *`; exists only as
    #    a Spinel type-inference workaround (`:ptr` ivars seed with it
    #    instead of `nil`, else post-85a4670 inference boxes them as
    #    sp_RbVal — see lib/toy/ffi/tinynn.rb FFNFFICache#initialize).
    #    Under MRI the honest equivalent of a typed NULL handle is
    #    `nil`. Without this, `TransformerLM.new` — the PURE-RUBY
    #    teaching model, the whole point of Stage A — raises at
    #    construction despite never executing an FFI branch.
    #
    # 2. The trace OFF-state (tinynn/tinynn_trace.c): until
    #    `tnn_trace_open` succeeds, natively begin→0, end/mark→nothing,
    #    active→0, op_capture_active→0. Under MRI tracing can never
    #    open (`tnn_trace_open` RAISES — a state change we cannot
    #    honor, so it fails loud, as does tnn_trace_set_op_capture),
    #    so the off-state values ARE the native semantics, forever.
    #    Without this, every Mat op raises — Mat#matmul et al. are
    #    unconditionally instrumented with tnn_trace_begin/end.
    OFF_STATE_NATIVES = {
      tnn_null_ptr:                nil,
      tnn_trace_begin:             0,
      tnn_trace_end:               nil,
      tnn_trace_mark:              nil,
      tnn_trace_active:            0,
      tnn_trace_op_capture_active: 0,
    }.freeze
  end
end

# ── The DSL shim ──────────────────────────────────────────────────────
# Guard FIRST: if any of the three is already a method on Module, some
# other definition got there before us (loaded twice? tao's interim
# stub? the ffi gem in the same process?). Silently clobbering would
# mask which semantics are live — fail loud instead.
%i[ffi_lib ffi_func ffi_cflags].each do |dsl|
  if Module.method_defined?(dsl) || Module.private_method_defined?(dsl)
    raise Toy::MRI::ShimCollisionError,
          "Module##{dsl} is already defined — refusing to clobber. " \
          "toy/mri must be the FIRST and ONLY provider of the FFI DSL " \
          "in this process (did it get required twice, or after a " \
          "foreign stub / the ffi gem?) — toy#71"
  end
end

class Module
  private

  # Records the library name; linking is a Spinel-build concern.
  def ffi_lib(name)
    Toy::MRI.record_lib(self, name)
  end

  # Records the flags; compiling is a Spinel-build concern.
  def ffi_cflags(flags)
    Toy::MRI.record_cflags(self, flags)
  end

  # Records the declaration and defines a module_function `name` that
  # raises Toy::MRI::NativeCallError. The declaring module's call shape
  # (TinyNN.tnn_session_new(0)) thus resolves under MRI — to a loud,
  # named failure at the exact native boundary.
  def ffi_func(name, argtypes = [], rettype = :void)
    Toy::MRI.record_func(self, name, argtypes, rettype)
    mod  = self
    sym  = name.to_sym
    impl =
      if Toy::MRI::OFF_STATE_NATIVES.key?(sym)
        value = Toy::MRI::OFF_STATE_NATIVES[sym]
        ->(*_args) { value }
      else
        lambda do |*_args|
          raise Toy::MRI::NativeCallError,
                "native call `#{sym}` (declared in #{mod}) requires the " \
                "Spinel-compiled binary or the fiddle backend — toy#71"
        end
      end
    define_method(sym, &impl)
    if instance_of?(Module)
      module_function(sym)
    else
      # Defensive: every current declaration site is a module, but if a
      # class ever declares one, give it the same singleton call shape.
      singleton_class.define_method(sym, &impl)
    end
    sym
  end
end

# With the DSL live, the full compute surface loads under MRI.
require_relative "compute"
