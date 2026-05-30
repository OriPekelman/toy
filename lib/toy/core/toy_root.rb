# lib/toy/core/toy_root.rb — locate the toy install root + drive `make`.
#
# CRuby ONLY (the CLI shell). Plain MRI Ruby — NO Spinel, NO ffi_lib. This
# is the reusable spine of the CRuby→runner compute bridge: `toy infer`
# (and, in later slices, `toy train` / `toy serve` / `toy eval`) cannot
# compute in-process — every lib that bears an `ffi_lib` directive crashes
# under MRI (see cli.rb:3-8). So they must shell out to a Spinel-compiled
# binary, and that binary must first be built. This module is the shared
# locate-root + ensure-built half of that pattern.
#
# Extracted verbatim from install.rb's private locate_root /
# toy_source_tree? so there is ONE resolver, no drift. install.rb now
# delegates here.
#
# Root-location priority (same as install.rb's documented order):
#   1. ENV["TOY_HOME"]                 — explicit override
#   2. dev checkout discovered by walking up from __dir__ (max 20 levels)
#   3. Gem.loaded_specs["toy"].gem_dir — gem-installed root (if present)
#
# Source-tree sentinel: Makefile + tinynn/tinynn_ggml.c. The C shim source
# is the definitive marker — it ships in the source tree and is never a
# build artifact (unlike vendor/ggml/, which only exists after `make
# setup`).

require "open3"

module Toy
  module Core
    module ToyRoot
      module_function

      # Priority: TOY_HOME → dev-checkout walk-up → gem dir. All three are
      # validated with the same #source_tree? sentinel so they stay in sync
      # — a stale TOY_HOME or empty gem dir fails the same way as a non-toy
      # walk-up hit instead of silently returning garbage. Returns the
      # absolute root path, or nil if none resolves.
      def locate_root
        env = ENV["TOY_HOME"]
        if env && !env.empty?
          expanded = File.expand_path(env)
          return expanded if source_tree?(expanded)
        end

        dir = __dir__
        20.times do
          return dir if source_tree?(dir)
          parent = File.dirname(dir)
          break if parent == dir
          dir = parent
        end

        if defined?(Gem) && Gem.loaded_specs["toy"]
          gd = Gem.loaded_specs["toy"].gem_dir
          return gd if gd && source_tree?(gd)
        end
        nil
      end

      # Sentinel test: does `dir` look like a toy source tree we can run
      # `make` in? Requires the Makefile plus a source-only marker
      # (tinynn_ggml.c) that is always shipped and is never a build
      # artifact. vendor/ggml/ is NOT a valid sentinel — setup CREATES it,
      # so a bone-fresh clone fails it.
      def source_tree?(dir)
        File.file?(File.join(dir, "Makefile")) &&
          File.file?(File.join(dir, "tinynn", "tinynn_ggml.c"))
      end

      # Build `target` (a Makefile target, e.g. "examples/example_inference")
      # in `root` via `make`. Idempotent — make no-ops when the target is
      # up-to-date, so callers may invoke this unconditionally rather than
      # reimplement mtime staleness. The first build on a fresh tree can
      # take minutes (the target deps pull the full lib inference stack), so
      # echo a notice unless `quiet`.
      #
      # Build progress (the notice + make's own output) goes to $stderr so
      # that callers which parse the runner's $stdout (e.g. `toy infer`) get
      # a clean compute stream uncontaminated by build chatter. Returns
      # [true, nil] on success, or [false, message] with a clean one-line
      # failure (last 20 lines of make output) — never raises, so callers
      # can fail loud with their own command prefix.
      def ensure_built(root, target, quiet: false)
        $stderr.puts "building #{target} in #{root} (this may take a few minutes)..." unless quiet
        out, status = Open3.capture2e("make", "-C", root, target)
        if status.success?
          $stderr.print out unless quiet
          [true, nil]
        else
          tail = out.lines.last(20).join
          [false, "`make #{target}` failed in #{root}:\n#{tail}"]
        end
      end
    end
  end
end
