# lib/toy/core/cli/eval.rb — `toy eval <model.gguf> [--top-k N] [--json]`.
#
# Score a GGUF model: report the top-K per-token logprobs at a frozen,
# deterministic eval point. CRuby ONLY (the CLI shell) — NO Spinel. Uses the
# same CRuby→runner COMPUTE BRIDGE as `toy infer`/`toy train`: the CLI cannot
# compute (every ffi_lib-bearing lib crashes under MRI; see cli.rb:3-8), so it
# locates the toy root, ensures a Spinel-compiled runner is built
# (`make libexec/toy-eval`), then shells out to it via Open3.
#
# RUNNER: the lib-side compute runner libexec/toy-eval, whose Spinel source is
# lib/toy/run/eval.rb. It reads config from ENV only (GGUF/TOP_K). We pass
# those as a controlled env hash (first positional to Open3) so the caller's
# stale GGUF/TOP_K can NEVER leak into the child. The runner prints one
# "logprob: <id> <logprob>" line per rank; we parse those and fail loud if
# none is present — no positional assumptions about stdout.
#
# SLICE 1: single-model per-token logprobs only. CE/perplexity over a --data
# corpus and the two-checkpoint LMC eval (`toy eval lmc --ckpt A --other B`)
# are LATER slices; --data is intentionally NOT a flag here yet.
#
# DETERMINISM: the runner does a pure CPU f32 forward + log_softmax + manual
# top-K — no sampler, no rand, no seed. Byte-for-byte reproducible; this is
# what prep/eval_gate.rb gates against a recorded baseline.

require "json"
require "open3"
require_relative "exit_codes"
require_relative "../toy_root"

module Toy
  module Core
    module CLI
      class Eval
        FORMAT = "toy/eval-v1"

        DEFAULT_TOP_K = 5 # parity w/ lib/toy/run/eval.rb TOP_K default
        # NOTE: target name MUST equal the output path — ToyRoot.ensure_built
        # runs `make <RUNNER_TARGET>` and File.join(root, RUNNER_TARGET) is
        # the binary. `make libexec/toy-eval` outputs libexec/toy-eval.
        RUNNER_TARGET = "libexec/toy-eval"

        def initialize(argv)
          @argv = argv
          @json = false
          @model = nil
          @top_k = DEFAULT_TOP_K
          @device = "cpu"
        end

        def run
          parsed = parse_args
          return parsed unless parsed == true

          # metal is accepted by the parser but only buildable in a macOS
          # build — gate it HERE, before any build/Open3, so the gx10 build
          # never tries to compile the (Apple-only) metal runner.
          if @device == "metal" && RUBY_PLATFORM !~ /darwin/
            return fail_out("metal is only available in a macOS build")
          end

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

          # Per-device binary: the CPU target stays byte-unchanged (its link
          # line never sees the CUDA archive); cuda builds the sibling runner.
          target = case @device
                   when "cuda"  then "libexec/toy-eval-cuda"
                   when "metal" then "libexec/toy-eval-metal"
                   else RUNNER_TARGET
                   end

          ok, err = ToyRoot.ensure_built(root, target, quiet: @json)
          return fail_out(err) unless ok

          runner = File.join(root, target)
          unless File.file?(runner) && File.executable?(runner)
            return fail_out(
              "runner missing after build: #{runner}. Run `toy install` to " \
              "build the backend, then retry."
            )
          end

          env = { "GGUF" => path, "TOP_K" => @top_k.to_s, "DEVICE" => @device }
          # Metal: disable ggml's residency-set optimization so the runner exits
          # cleanly. See lib/toy/core/cli/infer.rb for the full rationale — the
          # ggml-metal static-destructor teardown asserts the residency set is
          # empty and aborts at exit; disabling it keeps compute byte-identical.
          env["GGML_METAL_NO_RESIDENCY"] = "1" if @device == "metal"
          out, status = Open3.capture2e(env, runner)
          unless status.success?
            tail = out.lines.last(20).join
            return fail_out("runner exited #{status.exitstatus}:\n#{tail}")
          end

          lines = out.lines.map(&:chomp).select { |l| l.start_with?("logprob:") }
          if lines.empty?
            tail = out.lines.last(20).join
            return fail_out("runner produced no `logprob:` line; output was:\n#{tail}")
          end

          emit(path, lines)
        end

        private

        # Parse argv. Returns true, or an Integer exit code (message already
        # emitted). model is a REQUIRED positional .gguf path; --top-k /
        # --json are flags.
        def parse_args
          rest = []
          i = 0
          while i < @argv.length
            tok = @argv[i]
            case tok
            when "--json"
              @json = true
            when "--top-k"
              i += 1
              val = @argv[i]
              return bad_arg("--top-k requires a value") if val.nil?
              unless val =~ /\A\d+\z/ && val.to_i > 0
                return bad_arg("--top-k must be a positive integer, got #{val.inspect}")
              end
              @top_k = val.to_i
            when /\A--top-k=(.*)\z/
              val = $1
              unless val =~ /\A\d+\z/ && val.to_i > 0
                return bad_arg("--top-k must be a positive integer, got #{val.inspect}")
              end
              @top_k = val.to_i
            when "--device"
              i += 1
              val = @argv[i]
              return bad_arg("--device requires a value") if val.nil?
              unless %w[cpu cuda metal].include?(val)
                return bad_arg("--device must be one of cpu, cuda, metal, got #{val.inspect}")
              end
              @device = val
            when /\A--device=(.*)\z/
              val = $1
              unless %w[cpu cuda metal].include?(val)
                return bad_arg("--device must be one of cpu, cuda, metal, got #{val.inspect}")
              end
              @device = val
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

        # Parse a runner "logprob: <id> <logprob>" line into [id, logprob].
        def parse_logprob(line)
          body = line[("logprob:".length)..].strip
          id_s, val_s = body.split(" ", 2)
          [id_s.to_i, val_s]
        end

        def emit(path, lines)
          if @json
            pairs = lines.map do |l|
              id, val = parse_logprob(l)
              { "id" => id, "logprob" => val.to_f }
            end
            payload = { "format" => FORMAT, "model" => path,
                        "top_k" => @top_k, "logprobs" => pairs }
            puts JSON.pretty_generate(payload)
          else
            lines.each { |l| puts l }
          end
          EXIT_OK
        end

        # Bad CLI input (missing/extra args, unknown flag, bad --top-k) → 2.
        def bad_arg(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy eval: #{msg}"
          end
          EXIT_BAD_INPUT
        end

        # A named-but-missing model file is "bad input" → exit 2.
        def bad_input(msg)
          bad_arg(msg)
        end

        # Execution failure (no root, build failed, runner crashed, no
        # parseable line) → exit 1. Clean one-liner, never a stacktrace.
        def fail_out(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy eval: #{msg}"
          end
          EXIT_FAILURE
        end
      end
    end
  end
end
