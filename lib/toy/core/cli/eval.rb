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
require "fileutils"
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

        # LMC subcommand (`toy eval lmc --ckpt A --other B [--alphas ...]`).
        # CPU-only this slice; a cuda LMC twin is a later slice.
        LMC_RUNNER_TARGET = "libexec/toy-eval-lmc"
        DEFAULT_ALPHAS    = "0,0.25,0.5,0.75,1.0"

        # CE subcommand (`toy eval ce --ckpt <gguf> --corpus <pack> ...`) —
        # toy#130: mean next-token CE over disjoint held-out windows.
        # CPU-only like lmc.
        CE_RUNNER_TARGET = "libexec/toy-eval-ce"

        def initialize(argv)
          @argv = argv
          @json = false
          @model = nil
          @top_k = DEFAULT_TOP_K
          @device = "cpu"
          @lmc_a = nil
          @lmc_b = nil
          @alphas = DEFAULT_ALPHAS
        end

        def run
          # LMC subcommand dispatch — `toy eval lmc ...`. The single-model path
          # below is BYTE-UNCHANGED for the non-lmc case.
          return run_lmc(@argv.drop(1)) if @argv.first == "lmc"
          return run_ce(@argv.drop(1)) if @argv.first == "ce"

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

        # `toy eval lmc --ckpt A --other B [--alphas a,b,c] [--json]` — Linear
        # Mode Connectivity. Build+invoke libexec/toy-eval-lmc with a CONTROLLED
        # ENV {LMC_A,LMC_B,LMC_ALPHAS}; print the byte-deterministic α→loss
        # curve (one "lmc: <alpha>=<loss>" line per α). CPU-only this slice (no
        # --device). Returns true→never (always an Integer exit code).
        def run_lmc(argv)
          # --- Parse flags ---
          i = 0
          while i < argv.length
            tok = argv[i]
            case tok
            when "--json"
              @json = true
            when "--ckpt"
              i += 1
              val = argv[i]
              return bad_arg_lmc("--ckpt requires a value") if val.nil?
              @lmc_a = val
            when /\A--ckpt=(.*)\z/m
              @lmc_a = $1
            when "--other"
              i += 1
              val = argv[i]
              return bad_arg_lmc("--other requires a value") if val.nil?
              @lmc_b = val
            when /\A--other=(.*)\z/m
              @lmc_b = $1
            when "--alphas"
              i += 1
              val = argv[i]
              return bad_arg_lmc("--alphas requires a value") if val.nil?
              return bad_arg_lmc("--alphas must be comma-separated numbers, got #{val.inspect}") unless valid_alphas?(val)
              @alphas = val
            when /\A--alphas=(.*)\z/
              val = $1
              return bad_arg_lmc("--alphas must be comma-separated numbers, got #{val.inspect}") unless valid_alphas?(val)
              @alphas = val
            when "--device", /\A--device(=.*)?\z/
              return bad_arg_lmc("--device is not supported for lmc (cpu-only in this slice)")
            when /\A-/
              return bad_arg_lmc("unknown flag #{tok.inspect}")
            else
              return bad_arg_lmc("unexpected argument #{tok.inspect} (lmc takes no positionals)")
            end
            i += 1
          end

          # --- Required-flag + file validation ---
          return bad_arg_lmc("--ckpt is required") if @lmc_a.nil?
          return bad_arg_lmc("--other is required") if @lmc_b.nil?

          path_a = File.expand_path(@lmc_a)
          path_b = File.expand_path(@lmc_b)
          return bad_arg_lmc("no such file: #{path_a}") unless File.file?(path_a)
          return bad_arg_lmc("no such file: #{path_b}") unless File.file?(path_b)

          # --- Build + locate the runner ---
          root = ToyRoot.locate_root
          unless root
            return fail_out_lmc(
              "could not locate toy's install root. Set TOY_HOME to a toy " \
              "checkout (one with a Makefile + tinynn/tinynn_ggml.c), or run " \
              "from inside the toy source tree. Then `toy install` to build " \
              "the backend."
            )
          end

          target = LMC_RUNNER_TARGET
          ok, err = ToyRoot.ensure_built(root, target, quiet: @json)
          return fail_out_lmc(err) unless ok

          runner = File.join(root, target)
          unless File.file?(runner) && File.executable?(runner)
            return fail_out_lmc(
              "runner missing after build: #{runner}. Run `toy install` to " \
              "build the backend, then retry."
            )
          end

          # --- CONTROLLED ENV (first positional to Open3 — no stale leak). ---
          env = { "LMC_A" => path_a, "LMC_B" => path_b, "LMC_ALPHAS" => @alphas }
          out, status = Open3.capture2e(env, runner)
          unless status.success?
            tail = out.lines.last(20).join
            return fail_out_lmc("runner exited #{status.exitstatus}:\n#{tail}")
          end

          lines = out.lines.map(&:chomp).select { |l| l.start_with?("lmc:") }
          if lines.empty?
            tail = out.lines.last(20).join
            return fail_out_lmc("runner produced no `lmc:` line; output was:\n#{tail}")
          end

          emit_lmc(path_a, path_b, lines)
        end

        # Every comma-field of `s` must parse as a Float.
        def valid_alphas?(s)
          fields = s.split(",")
          return false if fields.empty?
          fields.all? do |f|
            begin
              Float(f.strip)
              true
            rescue ArgumentError, TypeError
              false
            end
          end
        end

        # Emit the α→loss curve. non-json: the gated lines verbatim. json: a
        # parsed curve payload.
        def emit_lmc(path_a, path_b, lines)
          if @json
            curve = lines.map do |l|
              body = l[("lmc:".length)..].strip
              a_s, v_s = body.split("=", 2)
              { "alpha" => a_s.to_f, "loss" => v_s.to_f }
            end
            payload = { "format" => FORMAT, "mode" => "lmc",
                        "ckpt" => path_a, "other" => path_b, "curve" => curve }
            puts JSON.pretty_generate(payload)
          else
            lines.each { |l| puts l }
          end
          EXIT_OK
        end

        # toy#130 — `toy eval ce`: checkpoint + pack → mean held-out CE.
        # Mirrors run_lmc's shape (parse → validate → build → controlled
        # env → parse the gated line).
        def run_ce(argv)
          ce_ckpt = nil
          ce_pack = nil
          ce_context = 32
          ce_tokens = 8192
          ce_offset = 0
          ce_out = nil
          i = 0
          while i < argv.length
            tok = argv[i]
            case tok
            when "--json"
              @json = true
            when "--ckpt"
              i += 1
              val = argv[i]
              return bad_arg_ce("--ckpt requires a value") if val.nil?
              ce_ckpt = val
            when /\A--ckpt=(.*)\z/m
              ce_ckpt = $1
            when "--corpus"
              i += 1
              val = argv[i]
              return bad_arg_ce("--corpus requires a value") if val.nil?
              ce_pack = val
            when /\A--corpus=(.*)\z/m
              ce_pack = $1
            when "--context"
              i += 1
              val = argv[i]
              return bad_arg_ce("--context requires a value") if val.nil?
              return bad_arg_ce("--context must be an integer >= 2, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 2
              ce_context = val.to_i
            when /\A--context=(.*)\z/
              val = $1
              return bad_arg_ce("--context must be an integer >= 2, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 2
              ce_context = val.to_i
            when "--eval-tokens"
              i += 1
              val = argv[i]
              return bad_arg_ce("--eval-tokens requires a value") if val.nil?
              return bad_arg_ce("--eval-tokens must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              ce_tokens = val.to_i
            when /\A--eval-tokens=(.*)\z/
              val = $1
              return bad_arg_ce("--eval-tokens must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              ce_tokens = val.to_i
            when "--offset"
              i += 1
              val = argv[i]
              return bad_arg_ce("--offset requires a value") if val.nil?
              return bad_arg_ce("--offset must be a non-negative integer (tokens), got #{val.inspect}") unless val =~ /\A\d+\z/
              ce_offset = val.to_i
            when /\A--offset=(.*)\z/
              val = $1
              return bad_arg_ce("--offset must be a non-negative integer (tokens), got #{val.inspect}") unless val =~ /\A\d+\z/
              ce_offset = val.to_i
            when "--out"
              i += 1
              val = argv[i]
              return bad_arg_ce("--out requires a value") if val.nil?
              ce_out = val
            when /\A--out=(.*)\z/m
              ce_out = $1
            when "--device", /\A--device(=.*)?\z/
              return bad_arg_ce("--device is not supported for ce (cpu-only in this slice)")
            when /\A-/
              return bad_arg_ce("unknown flag #{tok.inspect}")
            else
              return bad_arg_ce("unexpected argument #{tok.inspect} (ce takes no positionals)")
            end
            i += 1
          end

          return bad_arg_ce("--ckpt is required (a fused-llama checkpoint GGUF)") if ce_ckpt.nil?
          return bad_arg_ce("--corpus is required (a packed-i32/TOYC pack)") if ce_pack.nil?
          ckpt_path = File.expand_path(ce_ckpt)
          pack_path = File.expand_path(ce_pack)
          return bad_arg_ce("no such file: #{ckpt_path}") unless File.file?(ckpt_path)
          return bad_arg_ce("no such file: #{pack_path}") unless File.file?(pack_path)

          root = ToyRoot.locate_root
          unless root
            return fail_out_ce(
              "could not locate toy's install root. Set TOY_HOME to a toy " \
              "checkout (one with a Makefile + tinynn/tinynn_ggml.c), or run " \
              "from inside the toy source tree. Then `toy install` to build " \
              "the backend."
            )
          end
          ok, err = ToyRoot.ensure_built(root, CE_RUNNER_TARGET, quiet: @json)
          return fail_out_ce(err) unless ok
          runner = File.join(root, CE_RUNNER_TARGET)
          unless File.file?(runner) && File.executable?(runner)
            return fail_out_ce("runner missing after build: #{runner}. Run `toy install`, then retry.")
          end

          env = { "GGUF" => ckpt_path, "PACK" => pack_path,
                  "CONTEXT" => ce_context.to_s, "EVAL_TOKENS" => ce_tokens.to_s,
                  "EVAL_OFFSET" => ce_offset.to_s }
          if ce_out
            FileUtils.mkdir_p(ce_out)
            env = env.merge("TAO_RUN_DIR" => File.expand_path(ce_out))
          end
          out, status = Open3.capture2e(env, runner)
          unless status.success?
            tail = out.lines.last(20).join
            return fail_out_ce("runner exited #{status.exitstatus}:\n#{tail}")
          end
          lines = out.lines.map(&:chomp).select { |l| l.start_with?("eval_ce:") }
          if lines.empty?
            tail = out.lines.last(20).join
            return fail_out_ce("runner produced no `eval_ce:` line; output was:\n#{tail}")
          end
          if @json
            body = lines.first[("eval_ce:".length)..].strip
            fields = {}
            body.split(" ").each do |kv|
              k, v = kv.split("=", 2)
              fields[k] = v
            end
            payload = { "format" => FORMAT, "mode" => "ce", "ckpt" => ckpt_path,
                        "corpus" => pack_path, "windows" => fields["windows"].to_i,
                        "tokens" => fields["tokens"].to_i, "ce" => fields["ce"].to_f }
            puts JSON.pretty_generate(payload)
          else
            lines.each { |l| puts l }
          end
          EXIT_OK
        end

        def bad_arg_ce(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy eval ce: #{msg}"
          end
          EXIT_BAD_INPUT
        end

        def fail_out_ce(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy eval ce: #{msg}"
          end
          EXIT_FAILURE
        end

        # Bad CLI input for the lmc subcommand → exit 2, "toy eval lmc:" prefix.
        def bad_arg_lmc(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy eval lmc: #{msg}"
          end
          EXIT_BAD_INPUT
        end

        # Execution failure for the lmc subcommand → exit 1.
        def fail_out_lmc(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy eval lmc: #{msg}"
          end
          EXIT_FAILURE
        end

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
