# Toy FFI manifest — declares the build-time link recipe for each
# tinynn backend. Read by downstream consumers' post-vendor readers
# (the toy-side counterpart of tep#97's ffi_manifest).
#
# BUILD-TIME, CRuby-ONLY. lib/toy.rb's require chain pulls in
# tinynn.rb which depends on Spinel directives — that's why this
# file isn't required from anywhere in the runtime path. Consumers
# `load` it from their post-vendor script.
#
# Why this exists: tinynn.rb's `ffi_cflags "-L. -Ltinynn -L…"` uses
# paths relative to toy's repo root. For toy's own builds (run from
# the toy root), that's correct. For a downstream consumer that
# vendored toy via `spinel-compat vendor` and is compiling from a
# different working dir, the `-L.` etc. resolve to the wrong place
# and the link fails. The consumer's post-vendor reader walks the
# vendored tinynn{,_cuda,_metal}.rb and rewrites each `-L<rel>` to
# `-L<TOY_SRC>/<rel>`. Toy's own files are never touched.
#
# Recipe is *backend* × *relative library path list*. The post-vendor
# reader joins them into the absolute -L args and substitutes into
# the vendored Ruby. Plus a fixed pass-through tail (`-Wno-int-conversion`,
# `-framework Metal`, etc.) per backend.

module Toy
  module FFIManifest
    # Per-backend link recipe. Order matters (matches what
    # tinynn.rb / _cuda / _metal ship today).
    BACKENDS = {
      cpu: {
        ffi_file: "tinynn.rb",
        # Relative paths from toy's repo root that get absolutized:
        ld_paths: [
          ".",                                   # toy root (libtinynn_ggml.a is in tinynn/ — kept for legacy compat)
          "tinynn",                              # libtinynn_ggml.a
          "vendor/ggml/build/src",               # libggml.a, libggml-base.a
          "vendor/ggml/build/src/ggml-cpu",      # libggml-cpu.a
        ],
        # Trailing flags that go after the -L paths.
        tail: "-Wno-int-conversion",
      },
      cuda: {
        ffi_file: "tinynn_cuda.rb",
        ld_paths: [
          ".",
          "tinynn",
          "vendor/ggml/build-cuda/src",
          "vendor/ggml/build-cuda/src/ggml-cpu",
          "vendor/ggml/build-cuda/src/ggml-cuda",
        ],
        # CUDA libdir is system-absolute, NOT relative to TOY_SRC.
        # Default is the common /usr/local/cuda location; consumers can
        # override via CUDA_DIR.
        tail: "-Wno-int-conversion",
        extra_ld_paths_abs: ["/usr/local/cuda/lib64"],
        extra_ld_paths_env: "CUDA_DIR_LIB",
      },
      metal: {
        ffi_file: "tinynn_metal.rb",
        ld_paths: [
          ".",
          "tinynn",
          "vendor/ggml/build-metal/src",
          "vendor/ggml/build-metal/src/ggml-cpu",
          "vendor/ggml/build-metal/src/ggml-metal",
        ],
        tail: "-Wno-int-conversion -framework Foundation -framework Metal -framework MetalKit",
      },
    }.freeze

    # The string literally in tinynn.rb today. Post-vendor readers
    # gsub this exact line; if toy's source changes the literal, bump
    # this constant in lockstep (committed as part of the toy PR
    # that touches tinynn.rb).
    CURRENT_FFI_CFLAGS = {
      cpu:   '-L. -Ltinynn -Lvendor/ggml/build/src -Lvendor/ggml/build/src/ggml-cpu -Wno-int-conversion',
      cuda:  '-L. -Ltinynn -Lvendor/ggml/build-cuda/src -Lvendor/ggml/build-cuda/src/ggml-cpu -Lvendor/ggml/build-cuda/src/ggml-cuda -L/usr/local/cuda/lib64 -Wno-int-conversion',
      metal: '-L. -Ltinynn -Lvendor/ggml/build-metal/src -Lvendor/ggml/build-metal/src/ggml-cpu -Lvendor/ggml/build-metal/src/ggml-metal -Wno-int-conversion -framework Foundation -framework Metal -framework MetalKit',
    }.freeze

    # Build the absolute ffi_cflags string for `backend_key` against a
    # caller-supplied TOY_SRC directory. Honors any per-backend env
    # overrides (e.g. CUDA_DIR_LIB).
    def self.absolute_cflags(backend_key, toy_src)
      cfg = BACKENDS.fetch(backend_key)
      parts = cfg[:ld_paths].map { |rel| "-L#{File.join(toy_src, rel)}" }
      if cfg[:extra_ld_paths_abs]
        abs_extra = cfg[:extra_ld_paths_env] ? ENV[cfg[:extra_ld_paths_env]] : nil
        parts.concat(abs_extra ? [abs_extra].map { |p| "-L#{p}" } : cfg[:extra_ld_paths_abs].map { |p| "-L#{p}" })
      end
      parts << cfg[:tail]
      parts.join(" ")
    end
  end
end
