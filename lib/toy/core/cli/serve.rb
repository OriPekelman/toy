# lib/toy/core/cli/serve.rb — `toy serve <model.gguf> [--port N] [--name NAME]`.
#
# Serve a GGUF model over an OpenAI-compatible HTTP API (CPU). CRuby ONLY
# (the CLI shell) — NO Spinel. Follows the infer/train/eval run-flow for
# validation + ensure_built, then DIVERGES at the launch step.
#
# THE LOAD-BEARING DIVERGENCE: serve is a PERSISTENT process. infer.rb
# uses Open3.capture2e (run-once, capture stdout). That would block the
# CLI forever here — the server never returns. So `toy serve` instead
# Kernel.exec's the runner: the CRuby process is REPLACED by libexec/
# toy-serve, the server owns the terminal, and SIGINT/SIGTERM (Ctrl-C)
# go straight to Tep's signal handlers. cli.rb's `rescue Interrupt`
# never fires (intended) — control never returns past exec.
#
# Because exec replaces the process, ALL validation MUST run BEFORE the
# exec (after it, there is no "after"). The controlled env hash is exec's
# first positional arg, giving the same no-leak guarantee Open3 had:
#   {"MODEL_PATH"=>path, "MODEL_NAME"=>name, "PORT"=>port.to_s}
# The runner (lib/toy/run/serve.rb) reads ONLY those env vars.
#
# Exit-code contract (peer parity with infer/eval):
#   2  bad input  — missing/extra args, unknown flag, bad --port,
#                   named-but-missing model
#   1  failure    — no toy root, build failed, runner missing. (A
#                   valid-but-corrupt GGUF or an in-use port surfaces as
#                   the RUNNER's own nonzero exit AFTER exec, which IS
#                   `toy serve`'s exit code since the process was replaced.)
#
# NO --json: serve is persistent and returns no result envelope.

require_relative "exit_codes"
require_relative "../toy_root"

module Toy
  module Core
    module CLI
      class Serve
        DEFAULT_PORT = 4567
        # NOTE: target name MUST equal the output path — ToyRoot.ensure_built
        # runs `make <RUNNER_TARGET>` and File.join(root, RUNNER_TARGET) is
        # the binary. `make libexec/toy-serve` outputs libexec/toy-serve.
        RUNNER_TARGET = "libexec/toy-serve"

        def initialize(argv)
          @argv = argv
          @model = nil
          @port = DEFAULT_PORT
          @name = nil
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

          ok, err = ToyRoot.ensure_built(root, RUNNER_TARGET)
          return fail_out(err) unless ok

          runner = File.join(root, RUNNER_TARGET)
          unless File.file?(runner) && File.executable?(runner)
            return fail_out(
              "runner missing after build: #{runner}. Run `toy install` to " \
              "build the backend, then retry."
            )
          end

          # Default the model name from the GGUF basename (the runner does
          # the same derivation; we echo a friendly name pre-exec).
          name = @name || File.basename(path).sub(/\.gguf\z/, "")

          # The toy-branded line BEFORE exec; after exec the child streams
          # its own "[openai_api_llama] ready; serving" + Tep's listening
          # line. Goes to stderr so it never pollutes any piped capture.
          $stderr.puts "toy serve: starting #{name} on http://127.0.0.1:#{@port} (Ctrl-C to stop)"

          # DIVERGENCE: replace this process with the persistent runner.
          # The controlled env hash is exec's first positional arg — the
          # caller's stale MODEL_PATH/MODEL_NAME/PORT can never leak in.
          # Control never returns past this line.
          Kernel.exec(
            { "MODEL_PATH" => path, "MODEL_NAME" => name, "PORT" => @port.to_s },
            runner
          )
        end

        private

        # Parse argv. Returns true, or an Integer exit code (message already
        # emitted). model is a REQUIRED positional .gguf path; --port / --name
        # are flags.
        def parse_args
          rest = []
          i = 0
          while i < @argv.length
            tok = @argv[i]
            case tok
            when "--port"
              i += 1
              val = @argv[i]
              return bad_arg("--port requires a value") if val.nil?
              p = parse_port(val)
              return p unless p == true
            when /\A--port=(.*)\z/
              p = parse_port($1)
              return p unless p == true
            when "--name"
              i += 1
              val = @argv[i]
              return bad_arg("--name requires a value") if val.nil?
              @name = val
            when /\A--name=(.*)\z/m
              @name = $1
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

        # Validate a --port value: numeric, 1..65535. Returns true on
        # success (sets @port), or an Integer exit code (bad_arg → 2).
        def parse_port(val)
          unless val =~ /\A\d+\z/
            return bad_arg("--port must be a positive integer, got #{val.inspect}")
          end
          n = val.to_i
          unless n > 0 && n <= 65535
            return bad_arg("--port must be in 1..65535, got #{n}")
          end
          @port = n
          true
        end

        # Bad CLI input (missing/extra args, unknown flag, bad --port,
        # named-but-missing model) → exit 2. Clean one-liner.
        def bad_arg(msg)
          $stderr.puts "toy serve: #{msg}"
          EXIT_BAD_INPUT
        end

        # A model path the user named that doesn't exist → bad input (2),
        # mirroring infer.rb/eval.rb peer convention.
        def bad_input(msg)
          bad_arg(msg)
        end

        # Execution failure (no root, build failed, runner missing) → exit 1.
        def fail_out(msg)
          $stderr.puts "toy serve: #{msg}"
          EXIT_FAILURE
        end
      end
    end
  end
end
