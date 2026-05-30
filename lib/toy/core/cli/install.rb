# lib/toy/core/cli/install.rb — `toy install`.
#
# Build (or verify) the backend a project links against: ggml's static
# archive(s) and toy's FFI shim(s). On Linux/CPU that is just
# vendor/ggml/build/src/libggml.a + tinynn/libtinynn_ggml.a. On Darwin
# Metal is added: vendor/ggml/build-metal/src/libggml.a +
# tinynn/libtinynn_ggml_metal.a (so example_inference_metal links).
# CRuby ONLY — NO Spinel; this shells out to `make` for the actual build.
#
# DESIGN NOTE (open question — see the P3 report): the docs name this
# command but specify no mechanism, and a fresh `toy new` project has no
# Makefile / vendor sources of its own. So install LOCATES toy's install
# root and verifies-or-builds there, in priority order:
#
#   1. ENV["TOY_HOME"]                 — explicit override (new concept,
#                                        mirrors TOY_MODEL_DIR precedent)
#   2. dev checkout discovered by walking up from __dir__ until a dir
#      that looks like the toy source tree is found
#   3. Gem.loaded_specs["toy"].gem_dir — gem-installed root (if present)
#
# Source-tree sentinel: Makefile + tinynn/tinynn_ggml.c. The C shim
# source is the definitive marker — it ships in the source tree and is
# never absent from a buildable checkout. (The old sentinel was
# vendor/ggml/, but that directory only exists AFTER `make setup` has
# run, so install failed on a bone-fresh clone.)
#
# Then:
#   - all expected .a present + readable → VERIFY (ar t sanity) → ready
#   - missing .a but root has a buildable source tree → `make setup`
#     (platform auto-detect) + tinynn shim(s) then re-verify
#   - neither prebuilt .a nor buildable toolchain → FAIL LOUD with a clean
#     message naming what's missing (never a stacktrace; never-mask rule)

require "json"
require "open3"
require "rbconfig"
require_relative "exit_codes"

module Toy
  module Core
    module CLI
      class Install
        FORMAT = "toy/install-v1"

        # CPU artifacts — built on every platform.
        GGML_A   = "vendor/ggml/build/src/libggml.a"
        TINYNN_A = "tinynn/libtinynn_ggml.a"

        # Metal artifacts — built on Darwin only. The Metal libggml.a is
        # a side-effect of `make setup` on Darwin; the Metal tinynn shim
        # is a separate target that install must request explicitly,
        # else example_inference_metal can't link on a fresh Mac build.
        GGML_METAL_A   = "vendor/ggml/build-metal/src/libggml.a"
        TINYNN_METAL_A = "tinynn/libtinynn_ggml_metal.a"

        DARWIN = !!(RbConfig::CONFIG["host_os"] =~ /darwin/i)

        def initialize(argv)
          @argv = argv
          @json = false
        end

        def run
          return EXIT_BAD_INPUT unless parse_args

          root = locate_root
          unless root
            return fail_out(
              "could not locate toy's install root. Set TOY_HOME to a toy " \
              "checkout (one with a Makefile + tinynn/tinynn_ggml.c), or run " \
              "install from inside the toy source tree. A gem-installed toy " \
              "does not yet ship a buildable backend (see packaging note)."
            )
          end

          artifacts = expected_artifacts(root)
          missing = artifacts.reject { |a| usable?(a[:path]) }

          if missing.empty?
            return report_ready(root, artifacts, built: false)
          end

          # Not prebuilt. Can we build here?
          unless buildable?(root)
            names = missing.map { |a| a[:relpath] }.join(", ")
            return fail_out(
              "no usable backend at #{root}: missing #{names}, and no " \
              "Makefile + tinynn/tinynn_ggml.c to build from. This is the " \
              "gem-install gap — point TOY_HOME at a dev checkout."
            )
          end

          build_result = build(root)
          return build_result if build_result.is_a?(Integer)

          still_missing = artifacts.reject { |a| usable?(a[:path]) }
          if still_missing.empty?
            report_ready(root, artifacts, built: true)
          else
            names = still_missing.map { |a| a[:relpath] }.join(", ")
            fail_out("build completed but backend artifacts are still missing at #{root}: #{names}")
          end
        end

        private

        # The set of static archives this install must produce/verify on
        # this platform. Tagged by name so report_ready can speak in terms
        # of (ggml-cpu, tinynn-cpu, ggml-metal, tinynn-metal) instead of
        # one undifferentiated "cpu" backend.
        def expected_artifacts(root)
          list = [
            { name: "ggml-cpu",   relpath: GGML_A,   path: File.join(root, GGML_A) },
            { name: "tinynn-cpu", relpath: TINYNN_A, path: File.join(root, TINYNN_A) }
          ]
          if DARWIN
            list << { name: "ggml-metal",   relpath: GGML_METAL_A,   path: File.join(root, GGML_METAL_A) }
            list << { name: "tinynn-metal", relpath: TINYNN_METAL_A, path: File.join(root, TINYNN_METAL_A) }
          end
          list
        end

        def parse_args
          @argv.each do |tok|
            case tok
            when "--json" then @json = true
            else
              $stderr.puts "toy install: unknown argument #{tok.inspect}"
              return false
            end
          end
          true
        end

        # Priority: TOY_HOME → dev-checkout walk-up → gem dir. All three
        # paths are validated with the same #toy_source_tree? sentinel
        # (Makefile + tinynn/tinynn_ggml.c) so they stay in sync — a stale
        # TOY_HOME or an empty gem dir fails the same way as a non-toy
        # walk-up hit instead of silently returning garbage.
        def locate_root
          env = ENV["TOY_HOME"]
          if env && !env.empty?
            expanded = File.expand_path(env)
            return expanded if toy_source_tree?(expanded)
          end

          dir = __dir__
          20.times do
            return dir if toy_source_tree?(dir)
            parent = File.dirname(dir)
            break if parent == dir
            dir = parent
          end

          if defined?(Gem) && Gem.loaded_specs["toy"]
            gd = Gem.loaded_specs["toy"].gem_dir
            return gd if gd && toy_source_tree?(gd)
          end
          nil
        end

        # Sentinel test: does `dir` look like a toy source tree from which
        # we can run `make setup`? Requires the Makefile plus a source-only
        # marker (tinynn_ggml.c) that is always shipped in the source tree
        # and is never a build artifact. vendor/ggml/ is NOT a valid
        # sentinel — it's CREATED by setup, so a bone-fresh clone fails it.
        def toy_source_tree?(dir)
          File.file?(File.join(dir, "Makefile")) &&
            File.file?(File.join(dir, "tinynn", "tinynn_ggml.c"))
        end

        # True if `root` has the source files needed to drive a build. Same
        # predicate as toy_source_tree?; kept named for the call-site that
        # decides between "build or fail" after a missing-artifacts check.
        def buildable?(root)
          toy_source_tree?(root)
        end

        # A static archive is usable if it exists, is readable, and `ar t`
        # lists at least one member (cheap sanity, not a link test).
        def usable?(path)
          return false unless File.file?(path) && File.readable?(path)
          out, status = Open3.capture2e("ar", "t", path)
          status.success? && !out.strip.empty?
        rescue StandardError
          false
        end

        # Shell out to `make setup` in the located root. setup auto-detects
        # the platform (Darwin→ggml-cpu + ggml-metal; nvcc→ggml-cpu + ggml-
        # cuda; else CPU) and is idempotent via sentinels. Then build the
        # tinynn shim(s) explicitly (setup only builds ggml). On Darwin the
        # Metal tinynn shim is a separate target — without this line a
        # fresh Mac install leaves example_inference_metal unlinkable.
        # Streams nothing in --json mode; otherwise echoes.
        def build(root)
          puts "building backend in #{root} (this may take a few minutes)..." unless @json
          targets = ["setup", TINYNN_A]
          targets << TINYNN_METAL_A if DARWIN
          targets.each do |target|
            out, status = Open3.capture2e("make", "-C", root, target)
            unless status.success?
              tail = out.lines.last(20).join
              return fail_out("`make #{target}` failed in #{root}:\n#{tail}")
            end
            print out unless @json
          end
          true
        end

        # Backend label for the output: lists what GPU paths are linkable.
        # "cpu" on Linux/CPU, "cpu+metal" on Darwin, "cpu+cuda" if a CUDA
        # build is found (parity hook for the day install learns to drive
        # setup-ggml-cuda too).
        def backend_label(artifacts)
          have_metal = artifacts.any? { |a| a[:name] == "ggml-metal" }
          have_metal ? "cpu+metal" : "cpu"
        end

        def report_ready(root, artifacts, built:)
          if @json
            puts JSON.pretty_generate(
              "format" => FORMAT,
              "root" => root,
              "backend" => backend_label(artifacts),
              "built" => built,
              "artifacts" => artifacts.each_with_object({}) { |a, h| h[a[:name]] = a[:path] },
              "status" => "ready"
            )
          else
            verb = built ? "built" : "verified"
            puts "backend #{verb} at #{root} (#{backend_label(artifacts)})"
            artifacts.each do |a|
              puts format("  %-13s %s (%d bytes)", "#{a[:name]}:", a[:path], File.size(a[:path]))
            end
            ready_msg = artifacts.any? { |a| a[:name] == "ggml-metal" } ?
                          "Ready to link CPU and Metal examples." :
                          "Ready to link CPU examples."
            puts ready_msg
          end
          EXIT_OK
        end

        def fail_out(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy install: #{msg}"
          end
          EXIT_FAILURE
        end
      end
    end
  end
end
