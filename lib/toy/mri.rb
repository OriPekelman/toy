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
# It is the CRuby side of the seam, like lib/toy/core/ — a SEPARATE
# entry file, NOT a conditional require (Spinel silently compiles
# conditional require_relative to 0 — spinel-dev#20, probed a699cf9).
# Consequently there is NO sig/ lockstep for this file: Spinel never
# sees it, so no .rbs mirror exists or is needed.

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

    # The Stage B shared library (CPU ggml + tinynn, self-contained —
    # `make libtinynn_shared`). Repo-relative: this file lives at
    # <root>/lib/toy/mri.rb, the artifact at <root>/tinynn/.
    NATIVE_LIB = File.expand_path("../../tinynn/libtinynn_ggml_shared.so", __dir__)

    @declarations  = {}   # Module => [[name(Symbol), argtypes(Array), rettype(Symbol)], ...]
    @libs          = {}   # Module => [libname(String), ...]
    @cflags        = {}   # Module => [flags(String), ...]
    @native_handle = nil  # Fiddle::Handle when the native arm is live

    class << self
      # Public registry — the Fiddle arm binds against exactly these.
      attr_reader :declarations, :libs, :cflags, :native_handle

      # True when the Stage B native arm is live (shared lib dlopened).
      def native?
        !@native_handle.nil?
      end

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

      # ── Stage B: the Fiddle binder ──────────────────────────────────
      # Called once at load. Honors TOY_MRI_NATIVE=0 (explicit stub-arm
      # opt-out, silent); otherwise dlopens the shared lib when present,
      # or hints (one stderr line) at the make target when absent.
      def try_open_native!
        return if ENV["TOY_MRI_NATIVE"] == "0"
        if File.file?(NATIVE_LIB)
          require "fiddle"
          @native_handle = Fiddle.dlopen(NATIVE_LIB)
        else
          warn "toy/mri: stub arm (native calls raise) — `make libtinynn_shared` " \
               "builds tinynn/libtinynn_ggml_shared.so for real MRI compute (toy#71)"
        end
      end

      # Spinel's FFI lowering, pinned by probing generated C (a699cf9):
      #   :ptr → void*            (Ruby side: integer-valued handle)
      #   :int → int              :long → long          :size_t → size_t
      #   :double → double        :str → const char*    :void → void
      #   :int_array   → const int64_t* (Spinel Array<Integer> storage;
      #                  == native long on LP64 → pack("l!*"))
      #   :float_array → const double*  (Array<Float> storage — NOT
      #                  float; the C side narrows where needed → pack("d*"))
      # The tinynn headers declare exactly these widths per family
      # (verified: int64_t*/long* for the 2 int_array sigs, double* for
      # the 6 float_array sigs).
      def fiddle_type(t)
        case t
        when :ptr, :str, :int_array, :float_array then Fiddle::TYPE_VOIDP
        when :int    then Fiddle::TYPE_INT
        when :long   then Fiddle::TYPE_LONG
        when :size_t then Fiddle::TYPE_SIZE_T
        when :double then Fiddle::TYPE_DOUBLE
        when :void   then Fiddle::TYPE_VOID
        else
          raise NativeCallError,
                "unknown FFI type #{t.inspect} in a declaration — extend " \
                "Toy::MRI.fiddle_type in lockstep with Spinel's DSL (toy#71)"
        end
      end

      INT_ARRAY_FMT   = "l!*"  # native long == int64_t on LP64
      FLOAT_ARRAY_FMT = "d*"   # C double

      # Build the native (Fiddle-backed) implementation for one
      # declaration. Spinel's array specs are ZERO-COPY (the C side reads
      # AND writes the Ruby array's storage); Fiddle can't share storage
      # with an MRI Array, so we mirror the semantics: pack → call →
      # unpack → Array#replace IN PLACE on the caller's array. That keeps
      # output-array conventions (pre-sized Array filled by C, e.g.
      # tnn_read_i32_file, tnn_download_to_f64_array) working unchanged.
      def native_impl(mod, name, argtypes, rettype)
        sym =
          begin
            @native_handle[name.to_s]
          rescue Fiddle::DLError
            nil
          end
        if sym.nil?
          # Loud, named, at the CALL site (a require-time raise would
          # take down the whole surface for one stale artifact).
          return lambda do |*_args|
            raise NativeCallError,
                  "native call `#{name}` (declared in #{mod}) is missing from " \
                  "#{File.basename(NATIVE_LIB)} — stale artifact? rebuild with " \
                  "`make libtinynn_shared` (toy#71)"
          end
        end
        fn = Fiddle::Function.new(sym,
                                  argtypes.map { |t| fiddle_type(t) },
                                  fiddle_type(rettype))
        lambda do |*args|
          if args.length != argtypes.length
            raise ArgumentError,
                  "#{mod}.#{name}: #{args.length} args for " \
                  "#{argtypes.length}-arg native declaration"
          end
          arrays = nil
          cargs = Array.new(args.length)
          i = 0
          while i < args.length
            a = args[i]
            cargs[i] =
              case argtypes[i]
              when :ptr then a.nil? ? 0 : a
              when :int, :long, :size_t then Integer(a)
              when :double then Float(a)
              when :str then a # String buffer → const char*
              when :int_array
                buf = a.pack(INT_ARRAY_FMT)
                (arrays ||= []) << [a, buf, INT_ARRAY_FMT]
                buf
              when :float_array
                buf = a.pack(FLOAT_ARRAY_FMT)
                (arrays ||= []) << [a, buf, FLOAT_ARRAY_FMT]
                buf
              end
            i += 1
          end
          r = fn.call(*cargs)
          arrays&.each { |orig, buf, fmt| orig.replace(buf.unpack(fmt)) }
          case rettype
          when :ptr  then r.to_i
          when :str  then r.null? ? nil : r.to_s
          when :void then nil
          else r
          end
        end
      end
    end

    # The deliberate NON-RAISING declarations of the STUB ARM ONLY —
    # every other declared function raises NativeCallError there. In the
    # NATIVE arm this table is BYPASSED ENTIRELY: tnn_null_ptr and the
    # trace family bind through Fiddle like everything else (real calls,
    # real off-state values from tinynn_trace.c, real NULL → 0 from
    # tnn_null_ptr). Two families, both with an exact, documented
    # pure-value semantic (no computation is being masked):
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

  # Records the declaration and defines a module_function `name`:
  # NATIVE arm — a Fiddle::Function binding against the shared lib
  # (real ggml compute under MRI); STUB arm — a loud, named
  # Toy::MRI::NativeCallError at the exact native boundary. Either way
  # the declaring module's call shape (TinyNN.tnn_session_new(0))
  # resolves under MRI.
  def ffi_func(name, argtypes = [], rettype = :void)
    Toy::MRI.record_func(self, name, argtypes, rettype)
    mod  = self
    sym  = name.to_sym
    impl =
      if Toy::MRI.native?
        Toy::MRI.native_impl(mod, sym, argtypes, rettype)
      elsif Toy::MRI::OFF_STATE_NATIVES.key?(sym)
        value = Toy::MRI::OFF_STATE_NATIVES[sym]
        ->(*_args) { value }
      else
        lambda do |*_args|
          raise Toy::MRI::NativeCallError,
                "native call `#{sym}` (declared in #{mod}) requires the " \
                "Spinel-compiled binary or the fiddle backend " \
                "(`make libtinynn_shared`) — toy#71"
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

# Arm selection happens ONCE, before any declaration is processed, so
# every ffi_func in the chain below binds consistently.
Toy::MRI.try_open_native!

# With the DSL live, the full compute surface loads under MRI.
require_relative "compute"
