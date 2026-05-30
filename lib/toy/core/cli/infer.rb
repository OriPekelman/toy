# lib/toy/core/cli/infer.rb — `toy infer <model.gguf> --prompt "X" [--n N]`.
#
# Generate text from a GGUF model with greedy (argmax) decode. CRuby ONLY
# (the CLI shell) — NO Spinel. This is the first user of the CRuby→runner
# COMPUTE BRIDGE: the CLI cannot compute (every ffi_lib-bearing lib crashes
# under MRI; see cli.rb:3-8), so it locates the toy root, ensures a
# Spinel-compiled runner is built (`make examples/example_inference`), then
# shells out to it via Open3 — exactly how `toy install` / `toy fetch`
# already shell external tools. train/serve/eval will copy this shape.
#
# RUNNER (slice 1, transitional): the existing examples/example_inference
# Spinel binary, whose compute source is examples/01_inference.rb. It reads
# config from ENV only (GGUF/PROMPT/N_NEW). We pass those as a controlled
# env hash (first positional to Open3) so the caller's stale GGUF/PROMPT/
# N_NEW can NEVER leak into the child. The runner prints a multi-line
# preamble (Arch.summary + "prompt: …"); we parse only its `text:` line
# (embedded tokenizer) or `ids:` line (none), and fail loud if neither is
# present — no positional assumptions about stdout.
#
# DETERMINISM: 01_inference calls generate(..) with no sampler_config →
# Sampler.argmax (first-max-wins, no rand/temperature). Greedy, no seed.
#
# DELIBERATELY NOT inherited from 01_inference.rb: its ModelIndex
# auto-select fallback (a silent fallback the never-mask rule forbids).
# `toy infer` requires an explicit .gguf path and errors on a bad one.

require "json"
require "open3"
require_relative "exit_codes"
require_relative "../toy_root"

module Toy
  module Core
    module CLI
      class Infer
        FORMAT = "toy/infer-v1"

        DEFAULT_PROMPT = "Once upon a time" # parity with 01_inference.rb:24
        DEFAULT_N      = 16                  # parity with N_NEW default :25
        RUNNER_TARGET  = "examples/example_inference"

        def initialize(argv)
          @argv = argv
          @json = false
          @model = nil
          @prompt = DEFAULT_PROMPT
          @n = DEFAULT_N
        end

        def run
          parsed = parse_args
          return parsed unless parsed == true

          path = File.expand_path(@model)
          unless File.file?(path)
            return bad_input("no such file: #{path}")
          end

          root = ToyRoot.locate_root
          unless root
            return fail_out(
              "could not locate toy's install root. Set TOY_HOME to a toy " \
              "checkout (one with a Makefile + tinynn/tinynn_ggml.c), or run " \
              "from inside the toy source tree. Then `toy install` to build " \
              "the backend."
            )
          end

          ok, err = ToyRoot.ensure_built(root, RUNNER_TARGET, quiet: @json)
          return fail_out(err) unless ok

          runner = File.join(root, RUNNER_TARGET)
          unless File.file?(runner) && File.executable?(runner)
            return fail_out(
              "runner missing after build: #{runner}. Run `toy install` to " \
              "build the backend, then retry."
            )
          end

          out, status = Open3.capture2e(
            { "GGUF" => path, "PROMPT" => @prompt, "N_NEW" => @n.to_s },
            runner
          )
          unless status.success?
            tail = out.lines.last(20).join
            return fail_out("runner exited #{status.exitstatus}:\n#{tail}")
          end

          text = scan_line(out, "text: ")
          if text
            return emit(path, text: text)
          end
          ids = scan_line(out, "ids:")
          if ids
            return emit(path, ids: ids.strip)
          end

          tail = out.lines.last(20).join
          fail_out("runner produced no `text:`/`ids:` line; output was:\n#{tail}")
        end

        private

        # Parse argv. Returns true, or an Integer exit code (message already
        # emitted). model is a REQUIRED positional .gguf path; --prompt /
        # --n / --json are flags. No auto-resolve.
        def parse_args
          rest = []
          i = 0
          while i < @argv.length
            tok = @argv[i]
            case tok
            when "--json"
              @json = true
            when "--prompt"
              i += 1
              val = @argv[i]
              return bad_arg("--prompt requires a value") if val.nil?
              @prompt = val
            when "--n"
              i += 1
              val = @argv[i]
              return bad_arg("--n requires a value") if val.nil?
              unless val =~ /\A\d+\z/ && val.to_i > 0
                return bad_arg("--n must be a positive integer, got #{val.inspect}")
              end
              @n = val.to_i
            when /\A--prompt=(.*)\z/m
              @prompt = $1
            when /\A--n=(.*)\z/
              val = $1
              unless val =~ /\A\d+\z/ && val.to_i > 0
                return bad_arg("--n must be a positive integer, got #{val.inspect}")
              end
              @n = val.to_i
            when /\A-/
              return bad_arg("unknown flag #{tok.inspect}")
            else
              rest << tok
            end
            i += 1
          end

          if rest.empty?
            return bad_arg("missing required argument <model.gguf>")
          end
          if rest.length > 1
            return bad_arg("unexpected extra arguments: #{rest[1..].join(' ')}")
          end
          @model = rest.first
          true
        end

        # Pull the first line beginning with `prefix` and return the text
        # AFTER the prefix (stripped of trailing newline). nil if absent.
        def scan_line(output, prefix)
          line = output.lines.find { |l| l.start_with?(prefix) }
          return nil unless line
          line[prefix.length..].chomp
        end

        def emit(path, text: nil, ids: nil)
          if @json
            payload = { "format" => FORMAT, "model" => path,
                        "prompt" => @prompt, "n" => @n }
            if text
              payload["text"] = text
            else
              payload["ids"] = ids
              payload["note"] = "no embedded tokenizer; emitting raw IDs"
            end
            puts JSON.pretty_generate(payload)
          elsif text
            puts text
          else
            $stderr.puts "toy infer: no embedded tokenizer; emitting raw IDs"
            puts "ids: #{ids}"
          end
          EXIT_OK
        end

        # Bad CLI input (missing/extra args, unknown flag, bad --n) → exit 2.
        def bad_arg(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy infer: #{msg}"
          end
          EXIT_BAD_INPUT
        end

        # A model path that exists-or-not check that is "bad input" (a
        # missing file the user named) → exit 2, mirroring describe.rb's
        # treatment of a named-but-missing model.
        def bad_input(msg)
          bad_arg(msg)
        end

        # Execution failure (no root, build failed, runner crashed, no
        # parseable line) → exit 1. Clean one-liner, never a stacktrace.
        def fail_out(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy infer: #{msg}"
          end
          EXIT_FAILURE
        end
      end
    end
  end
end
