# lib/toy/core/cli/install.rb — `toy install`.
#
# Build (or verify) the CPU backend a project links against: ggml's
# static archive (vendor/ggml/build/src/libggml.a) and toy's FFI shim
# (tinynn/libtinynn_ggml.a). CRuby ONLY — NO Spinel; this shells out to
# `make` for the actual build.
#
# DESIGN NOTE (open question — see the P3 report): the docs name this
# command but specify no mechanism, and a fresh `toy new` project has no
# Makefile / vendor sources of its own. So install LOCATES toy's install
# root and verifies-or-builds there, in priority order:
#
#   1. ENV["TOY_HOME"]                 — explicit override (new concept,
#                                        mirrors TOY_MODEL_DIR precedent)
#   2. dev checkout discovered by walking up from __dir__ until a dir
#      with BOTH Makefile and vendor/ggml is found
#   3. Gem.loaded_specs["toy"].gem_dir — gem-installed root (if present)
#
# Then:
#   - both .a present + readable → VERIFY (ar t sanity) → ready, EXIT_OK
#   - .a absent but root has Makefile + vendor sources → `make setup`
#     (platform auto-detect) then re-verify
#   - neither prebuilt .a nor buildable toolchain → FAIL LOUD with a clean
#     message naming what's missing (never a stacktrace; never-mask rule)

require "json"
require "open3"
require_relative "exit_codes"

module Toy
  module Core
    module CLI
      class Install
        FORMAT = "toy/install-v1"

        GGML_A   = "vendor/ggml/build/src/libggml.a"
        TINYNN_A = "tinynn/libtinynn_ggml.a"

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
              "checkout (one with a Makefile + vendor/ggml), or run install " \
              "from inside the toy source tree. A gem-installed toy does not " \
              "yet ship a buildable backend (see packaging note)."
            )
          end

          ggml   = File.join(root, GGML_A)
          tinynn = File.join(root, TINYNN_A)

          if usable?(ggml) && usable?(tinynn)
            return report_ready(root, ggml, tinynn, built: false)
          end

          # Not prebuilt. Can we build here?
          makefile = File.join(root, "Makefile")
          vendor   = File.join(root, "vendor", "ggml")
          unless File.file?(makefile) && File.directory?(vendor)
            return fail_out(
              "no usable backend at #{root}: missing #{GGML_A} and " \
              "#{TINYNN_A}, and no Makefile + vendor/ggml to build them. " \
              "This is the gem-install gap — point TOY_HOME at a dev checkout."
            )
          end

          build_result = build(root)
          return build_result if build_result.is_a?(Integer)

          if usable?(ggml) && usable?(tinynn)
            report_ready(root, ggml, tinynn, built: true)
          else
            fail_out("build completed but backend artifacts are still missing at #{root}")
          end
        end

        private

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

        # Priority: TOY_HOME → dev-checkout walk-up → gem dir.
        def locate_root
          env = ENV["TOY_HOME"]
          return File.expand_path(env) if env && !env.empty? && File.directory?(env)

          dir = __dir__
          20.times do
            if File.file?(File.join(dir, "Makefile")) &&
               File.directory?(File.join(dir, "vendor", "ggml"))
              return dir
            end
            parent = File.dirname(dir)
            break if parent == dir
            dir = parent
          end

          if defined?(Gem) && Gem.loaded_specs["toy"]
            gd = Gem.loaded_specs["toy"].gem_dir
            return gd if gd && File.directory?(gd)
          end
          nil
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
        # the platform (Darwin→metal, nvcc→cuda, else CPU) and is idempotent
        # via sentinels. Then build the tinynn shim explicitly (setup only
        # builds ggml). Streams nothing in --json mode; otherwise echoes.
        def build(root)
          puts "building backend in #{root} (this may take a few minutes)..." unless @json
          %w[setup tinynn/libtinynn_ggml.a].each do |target|
            out, status = Open3.capture2e("make", "-C", root, target)
            unless status.success?
              tail = out.lines.last(20).join
              return fail_out("`make #{target}` failed in #{root}:\n#{tail}")
            end
            print out unless @json
          end
          true
        end

        def report_ready(root, ggml, tinynn, built:)
          if @json
            puts JSON.pretty_generate(
              "format" => FORMAT,
              "root" => root,
              "backend" => "cpu",
              "built" => built,
              "artifacts" => {
                "ggml" => ggml,
                "tinynn" => tinynn
              },
              "status" => "ready"
            )
          else
            verb = built ? "built" : "verified"
            puts "backend #{verb} at #{root}"
            puts "  ggml:   #{ggml} (#{File.size(ggml)} bytes)"
            puts "  tinynn: #{tinynn} (#{File.size(tinynn)} bytes)"
            puts "Ready to link CPU examples."
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
