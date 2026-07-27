# lib/toy/core/cli/train.rb — `toy train from-scratch [--steps N] [--seed S]
# [--out <dir>] [--json]`.
#
# Train a tiny Llama-shape model from scratch and record a run bundle under
# runs/<id>/. CRuby ONLY (the CLI shell) — NO Spinel. Second user of the
# CRuby→runner COMPUTE BRIDGE that `toy infer` established: the CLI cannot
# compute (every ffi_lib-bearing lib crashes under MRI; see cli.rb:3-8), so it
# locates the toy install root, ensures the Spinel-compiled runner is built
# (`make libexec/toy-train`), resolves a run id, creates runs/<id>/, then
# shells out via Open3 with a CONTROLLED ENV.
#
# RUNNER: lib/toy/run/train.rb → libexec/toy-train. It reads config from ENV
# only (STEPS/SEED/TAO_RUN_DIR/TOY_RUN_ID). We pass those as a controlled env
# hash (first positional to Open3) so a stale caller TAO_RUN_DIR/STEPS can
# NEVER leak into the child. The runner prints byte-deterministic
# "step N: loss=" lines to stdout (the gated loss curve); events.jsonl + the
# checkpoint go to runs/<id>/ (file-side, never stdout).
#
# THIN FLAGS (roadmap risk): only --steps/--seed/--out/--json. The recipe's
# with_hyper/realize! chain is the real knob surface, deferred. Only the
# "from-scratch" recipe in this slice — warm-start / lora / curriculum are
# LATER slices and are rejected with a clean bad-input error.
#
# TWO ROOTS (do not conflate): ToyRoot.locate_root = the TOY install tree
# (for `make`); Dir.pwd = the PROJECT (where toy.yml + runs/ live). infer
# needs only the former; train needs both.

require "json"
require "open3"
require "fileutils"
require_relative "exit_codes"
require_relative "../toy_root"
require_relative "../config"

module Toy
  module Core
    module CLI
      class Train
        FORMAT = "toy/train-v1"

        # NOTE: target name MUST equal the output path — ToyRoot.ensure_built
        # runs `make <RUNNER_TARGET>` and File.join(root, RUNNER_TARGET) is the
        # binary. `make libexec/toy-train` outputs libexec/toy-train.
        #
        # from-scratch + warm-start share libexec/toy-train (both random-init).
        # lora dispatches to a SEPARATE binary, libexec/toy-train-lora: its
        # realize_for_mmap path cannot share a Spinel compilation unit with the
        # random-init path without a cfg type-merge miscompile (landmine #16;
        # see lib/toy/run/train_lora.rb header). Same byte-gated stdout
        # contract — only the binary differs.
        RUNNER_TARGET      = "libexec/toy-train"
        LORA_RUNNER_TARGET = "libexec/toy-train-lora"
        # CUDA from-scratch runner — a SEPARATE per-device binary (single-type
        # binary, landmine #16). Selected only for --device cuda + from-scratch.
        CUDA_RUNNER_TARGET = "libexec/toy-train-cuda"
        # CUDA lora runner — a SEPARATE single-type binary (landmine #16): its
        # realize_for_mmap cfg path is monomorphic and cannot share a Spinel
        # compilation unit with the random-init path. Selected only for
        # --device cuda + lora.
        LORA_CUDA_RUNNER_TARGET = "libexec/toy-train-lora-cuda"
        # Metal from-scratch runner — a SEPARATE per-device binary (single-type
        # binary, landmine #16). Selected only for --device metal + from-scratch,
        # and only on macOS (the build target is macOS-guarded).
        METAL_RUNNER_TARGET = "libexec/toy-train-metal"
        # ViT-Tiny from-scratch CPU runner — a SEPARATE binary (landmine #16):
        # ViTTinyConfig must NOT share a Spinel compilation unit with
        # SmolLM2Config. CPU-only this slice. Binary path EQUALS the make
        # target so ToyRoot.ensure_built builds + locates it.
        VIT_RUNNER_TARGET = "libexec/toy-train-vit"
        # GPT-2 from-scratch CPU runner — a SEPARATE binary (landmine #16): the
        # GPT2SeqEngine realize path can't share a Spinel unit with the llama one.
        # Selected by `--arch gpt2` (from-scratch, CPU only this slice). Backward
        # of its LayerNorm + GELU rides the vendored kernels (vendor-patches/0007).
        GPT2_RUNNER_TARGET = "libexec/toy-train-gpt2"
        # Franken credit-assignment runner (toy#109/#112) — its own binary
        # (landmine #16): FrankenFromScratch alongside FromScratch stays out
        # of the byte-gated toy-train unit.
        FRANKEN_RUNNER_TARGET = "libexec/toy-train-franken-llama"
        FRANKEN_CUDA_RUNNER_TARGET = "libexec/toy-train-franken-llama-cuda"
        FRANKEN_MOE_RUNNER_TARGET = "libexec/toy-train-franken-moe-cli"
        # GPT-2 GPU twins (--arch gpt2 --device cuda|metal). SEPARATE single-type
        # binaries (landmine #16); link the generated CUDA/Metal engine mirrors.
        # The GELU/LayerNorm backward ops fall back to the CPU backend on GPU.
        GPT2_CUDA_RUNNER_TARGET  = "libexec/toy-train-gpt2-cuda"
        GPT2_METAL_RUNNER_TARGET = "libexec/toy-train-gpt2-metal"

        DEFAULT_STEPS = 5  # the gate config (smoke_recipe_from_scratch)
        DEFAULT_SEED  = 0

        # The from-scratch arch family — substituted for {arch} in the
        # run_id_template. The runner hardcodes a llama-shape model.
        ARCH = "llama"

        def initialize(argv)
          @argv  = argv
          @json  = false
          @recipe = nil
          @steps = DEFAULT_STEPS
          @seed  = DEFAULT_SEED
          @out   = nil
          @model = nil   # lora GGUF path
          @rank  = nil   # lora rank (Integer)
          @corpus = nil  # warm-start corpus path
          @init  = nil   # warm-start init mode
          @device = "cpu"  # cpu | cuda | metal (from-scratch only for non-cpu)
          @policy = nil    # franken: per-layer credit tokens
          @dfa_b_seed  = nil  # franken B axes
          @dfa_b_dist  = nil
          @dfa_b_scale = nil
          @align_events = false
          @routing    = nil   # franken-moe: dense | top1
          @moe_policy = nil   # franken-moe: chain | dfa-experts
          @moe_aux    = nil   # franken-moe: top1 aux-loss alpha
          @arch   = ARCH   # llama | gpt2 (gpt2 = from-scratch CPU only this slice)
        end

        def run
          parsed = parse_args
          return parsed unless parsed == true

          # The TOY INSTALL root (for `make`) — may differ from Dir.pwd.
          # metal is accepted by the parser but only buildable in a macOS
          # build — gate it HERE, before any build/Open3 (mirrors infer.rb).
          if @device == "metal" && RUBY_PLATFORM !~ /darwin/
            return fail_out("metal is only available in a macOS build")
          end

          # --arch gpt2 is from-scratch only (CPU/CUDA/Metal). Metal is gated to
          # macOS by the @device check above; CUDA/Metal back the GELU/LayerNorm
          # backward on the CPU fallback backend (no GPU kernel yet).
          if @arch == "gpt2" && @recipe != "from-scratch"
            return fail_out("--arch gpt2 supports only the `from-scratch` recipe")
          end

          # Existence checks on user-suppliable paths BEFORE any build/shell
          # (spinel-dev#17 class: the runner side also guards, but the CLI
          # names the file and the fix first — and exits 2 like infer's /
          # describe's named-but-missing model). Paths are cwd-relative to
          # the PROJECT (the runner runs in Dir.pwd).
          if @recipe == "lora"
            lora_gguf = @model || "data/smollm2-135m-native.gguf"
            unless File.file?(lora_gguf)
              return bad_arg("no such file: #{lora_gguf} (lora needs a " \
                             "native-layout base GGUF; see `toy list`, or convert one " \
                             "with prep/convert_smollm2_to_gguf.py --ggml-native)")
            end
          elsif @recipe == "warm-start"
            ws_corpus = @corpus || "data/ts_seqs.bin"
            unless File.file?(ws_corpus)
              return bad_arg("no such file: #{ws_corpus} (warm-start streams " \
                             "packed-i32 tokens; `toy new` seeds data/ts_seqs.bin, or pass --corpus)")
            end
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

          # Per-device-AND-recipe binary (single-type, landmine #16).
          #   lora + cuda      -> toy-train-lora-cuda (monomorphic mmap cfg)
          #   lora + cpu       -> toy-train-lora      (unchanged)
          #   from-scratch /   -> toy-train-cuda (the warm-start branch lives
          #     warm-start +cuda                  in train_cuda.rb source)
          #   metal (fs only)  -> toy-train-metal
          #   cpu fs/warm-start-> toy-train
          target = if @recipe == "franken-moe"
                     FRANKEN_MOE_RUNNER_TARGET
                   elsif @recipe == "franken"
                     @device == "cuda" ? FRANKEN_CUDA_RUNNER_TARGET : FRANKEN_RUNNER_TARGET
                   elsif @recipe == "lora"
                     @device == "cuda" ? LORA_CUDA_RUNNER_TARGET : LORA_RUNNER_TARGET
                   elsif @recipe == "vit-tiny"
                     VIT_RUNNER_TARGET
                   elsif @arch == "gpt2"
                     case @device
                     when "cuda"  then GPT2_CUDA_RUNNER_TARGET
                     when "metal" then GPT2_METAL_RUNNER_TARGET
                     else              GPT2_RUNNER_TARGET
                     end
                   elsif @device == "cuda"
                     CUDA_RUNNER_TARGET
                   elsif @device == "metal"
                     METAL_RUNNER_TARGET
                   else
                     RUNNER_TARGET
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

          # Resolve a run id + create runs/<id>/ in the PROJECT cwd (the
          # train-specific net-new step, CRuby-side, BEFORE Open3 — the runner
          # assumes TAO_RUN_DIR pre-exists; tnn_events_open does no parent
          # mkdir).
          project  = Dir.pwd
          cfg      = Toy::Core::Config.load(project)
          # toy#115 — a caller-supplied TOY_RUN_ID wins over the internal
          # llama-<date>-<seq> counter (the tao#flow contract: the events
          # run_id and the Tao registry entry must name the run identically;
          # the controlled-env rebuild was silently dropping it).
          run_id   = ENV["TOY_RUN_ID"].to_s.length > 0 ? ENV["TOY_RUN_ID"] :
                       resolve_run_id(cfg.run_id_template, project,
                                      @arch == "gpt2" ? "gpt2" : arch_for(@recipe))
          run_dir  = @out ? File.expand_path(@out) : File.join(project, "runs", run_id)
          begin
            FileUtils.mkdir_p(run_dir)
          rescue SystemCallError => e
            return fail_out("could not create run dir #{run_dir}: #{e.message}")
          end

          # CONTROLLED ENV (first positional) so a stale caller env can't
          # leak. Built per-recipe (parallel to the runner's landmine-#16
          # branch discipline): each recipe's exact keys reproduce its gate.
          base = { "TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => run_id,
                   "RECIPE" => @recipe }
          if @recipe == "lora"
            # NO SEED key: lora seed=42 is hardcoded in the runner branch.
            env = base.merge("STEPS" => @steps.to_s,
                             "GGUF"  => (@model || "data/smollm2-135m-native.gguf"),
                             "RANK"  => (@rank || 8).to_s)
          elsif @recipe == "warm-start"
            env = base.merge("STEPS"  => @steps.to_s,
                             "SEED"   => @seed.to_s,
                             "CORPUS" => (@corpus || "data/ts_seqs.bin"),
                             "INIT"   => (@init || "scratch"))
          elsif @recipe == "vit-tiny"
            # CPU-only ViT from-scratch on the COMMITTED data/vit_smoke corpus.
            # Runner hard-codes 224/16/196/10 + AdamW hp; only STEPS/SEED vary.
            # data/vit_smoke is committed → no --corpus needed. vit IS seeded.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "IMG_DIR" => "data/vit_smoke")
          elsif @recipe == "franken-moe"
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "FRANKEN_MOE_ROUTING" => (@routing || "dense"),
                             "FRANKEN_MOE"         => (@moe_policy || "chain"),
                             "FRANKEN_MOE_AUX"     => (@moe_aux || "0"),
                             "FRANKEN_B_SEED"      => (@dfa_b_seed || 1234).to_s,
                             "FRANKEN_B_DIST"      => (@dfa_b_dist || ""),
                             "FRANKEN_B_SCALE"     => (@dfa_b_scale || ""),
                             "FRANKEN_ALIGN"       => (@align_events ? "1" : ""))
          elsif @recipe == "franken"
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "FRANKEN_POLICY"  => (@policy || ""),
                             "FRANKEN_B_SEED"  => (@dfa_b_seed || 0).to_s,
                             "FRANKEN_B_DIST"  => (@dfa_b_dist || ""),
                             "FRANKEN_B_SCALE" => (@dfa_b_scale || ""),
                             "FRANKEN_ALIGN"   => (@align_events ? "1" : ""))
          else
            # from-scratch — byte-identical to today plus the harmless RECIPE key.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s)
          end
          # Metal: disable ggml's residency-set optimization so the runner exits
          # cleanly. See lib/toy/core/cli/infer.rb for the full rationale — the
          # ggml-metal static-destructor teardown asserts the residency set is
          # empty and aborts at exit; disabling it keeps compute byte-identical.
          env["GGML_METAL_NO_RESIDENCY"] = "1" if @device == "metal"
          out, status = Open3.capture2e(env, runner)
          unless status.success?
            tail = out.lines.last(20).join
            # exitstatus is nil for a signal death (e.g. SEGV) — say so
            # instead of the formerly-masked "runner exited :".
            how = status.exitstatus ? status.exitstatus.to_s
                                    : "from signal #{status.termsig} (#{Signal.signame(status.termsig) rescue '?'})"
            tail = "(no output)" if tail.strip.empty?
            return fail_out("runner exited #{how}:\n#{tail}")
          end

          losses = out.lines.select { |l| l.start_with?("step ") }.map(&:chomp)
          emit(run_id, run_dir, losses)
        end

        private

        # Parse argv. Returns true, or an Integer exit code (message already
        # emitted). FIRST positional = recipe name (required; only
        # "from-scratch"). --steps/--seed/--out/--json are flags.
        def parse_args
          rest = []
          i = 0
          while i < @argv.length
            tok = @argv[i]
            case tok
            when "--json"
              @json = true
            when "--steps"
              i += 1
              val = @argv[i]
              return bad_arg("--steps requires a value") if val.nil?
              return bad_arg("--steps must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @steps = val.to_i
            when "--seed"
              i += 1
              val = @argv[i]
              return bad_arg("--seed requires a value") if val.nil?
              return bad_arg("--seed must be a non-negative integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @seed = val.to_i
            when "--out"
              i += 1
              val = @argv[i]
              return bad_arg("--out requires a value") if val.nil?
              @out = val
            when "--arch"
              i += 1
              val = @argv[i]
              return bad_arg("--arch requires a value") if val.nil?
              return bad_arg("--arch must be llama or gpt2, got #{val.inspect}") unless %w[llama gpt2].include?(val)
              @arch = val
            when /\A--arch=(.*)\z/
              val = $1
              return bad_arg("--arch must be llama or gpt2, got #{val.inspect}") unless %w[llama gpt2].include?(val)
              @arch = val
            when /\A--steps=(.*)\z/
              val = $1
              return bad_arg("--steps must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @steps = val.to_i
            when /\A--seed=(.*)\z/
              val = $1
              return bad_arg("--seed must be a non-negative integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @seed = val.to_i
            when /\A--out=(.*)\z/m
              @out = $1
            when "--policy"
              i += 1
              val = @argv[i]
              return bad_arg("--policy requires a value (e.g. chain,dfa)") if val.nil?
              @policy = val
            when /\A--policy=(.*)\z/
              @policy = $1
            when "--dfa-b-seed"
              i += 1
              val = @argv[i]
              return bad_arg("--dfa-b-seed must be an integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @dfa_b_seed = val.to_i
            when /\A--dfa-b-seed=(.*)\z/
              val = $1
              return bad_arg("--dfa-b-seed must be an integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @dfa_b_seed = val.to_i
            when "--dfa-b-dist"
              i += 1
              val = @argv[i]
              return bad_arg("--dfa-b-dist must be gaussian|uniform|rademacher") unless %w[gaussian uniform rademacher].include?(val)
              @dfa_b_dist = val
            when /\A--dfa-b-dist=(.*)\z/
              val = $1
              return bad_arg("--dfa-b-dist must be gaussian|uniform|rademacher") unless %w[gaussian uniform rademacher].include?(val)
              @dfa_b_dist = val
            when "--dfa-b-scale"
              i += 1
              val = @argv[i]
              return bad_arg("--dfa-b-scale requires a value") if val.nil?
              @dfa_b_scale = val
            when /\A--dfa-b-scale=(.*)\z/
              @dfa_b_scale = $1
            when "--align-events"
              @align_events = true
            when "--routing"
              i += 1
              val = @argv[i]
              return bad_arg("--routing must be dense|top1") unless %w[dense top1].include?(val)
              @routing = val
            when /\A--routing=(.*)\z/
              val = $1
              return bad_arg("--routing must be dense|top1") unless %w[dense top1].include?(val)
              @routing = val
            when "--moe-policy"
              i += 1
              val = @argv[i]
              return bad_arg("--moe-policy must be chain|dfa-experts") unless %w[chain dfa-experts].include?(val)
              @moe_policy = val
            when /\A--moe-policy=(.*)\z/
              val = $1
              return bad_arg("--moe-policy must be chain|dfa-experts") unless %w[chain dfa-experts].include?(val)
              @moe_policy = val
            when "--moe-aux"
              i += 1
              val = @argv[i]
              return bad_arg("--moe-aux must be a non-negative float, got #{val.inspect}") unless val =~ /\A\d+(\.\d+)?\z/
              @moe_aux = val
            when /\A--moe-aux=(.*)\z/
              val = $1
              return bad_arg("--moe-aux must be a non-negative float, got #{val.inspect}") unless val =~ /\A\d+(\.\d+)?\z/
              @moe_aux = val
            when "--model"
              i += 1
              val = @argv[i]
              return bad_arg("--model requires a value") if val.nil?
              @model = val
            when "--rank"
              i += 1
              val = @argv[i]
              return bad_arg("--rank requires a value") if val.nil?
              return bad_arg("--rank must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @rank = val.to_i
            when "--corpus"
              i += 1
              val = @argv[i]
              return bad_arg("--corpus requires a value") if val.nil?
              @corpus = val
            when "--init"
              i += 1
              val = @argv[i]
              return bad_arg("--init requires a value") if val.nil?
              @init = val
            when /\A--model=(.*)\z/m
              @model = $1
            when /\A--rank=(.*)\z/
              val = $1
              return bad_arg("--rank must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @rank = val.to_i
            when /\A--corpus=(.*)\z/m
              @corpus = $1
            when /\A--init=(.*)\z/m
              @init = $1
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
            return bad_arg("missing required argument <recipe> (only 'from-scratch' is supported in this slice)")
          end
          if rest.length > 1
            return bad_arg("unexpected extra arguments: #{rest[1..].join(' ')}")
          end
          @recipe = rest.first
          unless %w[from-scratch lora warm-start vit-tiny franken franken-moe].include?(@recipe)
            return bad_arg("unknown recipe #{@recipe.inspect}; supported: 'from-scratch', 'lora', 'warm-start', 'vit-tiny', 'franken', 'franken-moe'")
          end
          if @recipe != "lora" && (!@model.nil? || !@rank.nil?)
            return bad_arg("--model/--rank are only valid with recipe 'lora'")
          end
          if @recipe != "warm-start" && (!@corpus.nil? || !@init.nil?)
            return bad_arg("--corpus/--init are only valid with recipe 'warm-start'")
          end
          if !%w[franken franken-moe].include?(@recipe) && (@dfa_b_seed || @dfa_b_dist || @dfa_b_scale || @align_events)
            return bad_arg("--dfa-b-*/--align-events are only valid with recipe 'franken' or 'franken-moe'")
          end
          if @recipe != "franken" && @policy
            return bad_arg("--policy is only valid with recipe 'franken'")
          end
          if @recipe != "franken-moe" && (@routing || @moe_policy || @moe_aux)
            return bad_arg("--routing/--moe-policy/--moe-aux are only valid with recipe 'franken-moe'")
          end
          if @recipe == "franken-moe" && @moe_aux && @routing != "top1"
            return bad_arg("--moe-aux requires --routing top1 (the aux-loss rides the hard router)")
          end
          if @recipe == "franken-moe" && @moe_policy == "dfa-experts" && @routing == "top1"
            return bad_arg("--moe-policy is dense-only: top1 IS the fully-DFA lane (drop --moe-policy)")
          end
          if !@init.nil? && @init != "scratch"
            return bad_arg("--init #{@init.inspect} unsupported; only 'scratch' has a gate curve in this slice")
          end
          # cuda is valid for ALL three recipes (from-scratch + warm-start ->
          # toy-train-cuda; lora -> toy-train-lora-cuda). metal ships
          # from-scratch only (the macOS metal runner is from-scratch).
          if @device == "metal" && @recipe != "from-scratch"
            return bad_arg("--device metal is only supported with recipe 'from-scratch'")
          end
          if @recipe == "franken" && @device == "metal"
            return bad_arg("franken has no metal runner yet (CUDA + CPU only)")
          end
          # vit-tiny is CPU-only in this slice: reject cuda AND metal as clean
          # bad-input (the --device allow-list already caught unknown devices).
          if @recipe == "franken-moe" && @device != "cpu"
            return bad_arg("--device #{@device.inspect} is not supported for recipe 'franken-moe' (cpu-only instrument)")
          end
          if @recipe == "vit-tiny" && @device != "cpu"
            return bad_arg("--device #{@device.inspect} is not supported for recipe 'vit-tiny' (cpu-only in this slice)")
          end
          true
        end

        # Expand a run_id_template's brace tokens. Supported:
        #   {arch} → "llama"   {date} → YYYYMMDD   {time} → HHMMSS
        #   {seq}  → 3-digit zero-padded daily counter (max existing run sharing
        #            today's {date} substring, +1)
        # NET-NEW resolver — config.rb only STORES the template.
        # Arch label for the {arch} run-id token. The lora base is
        # smollm2-shape; from-scratch + warm-start are both llama-shape.
        # NOTE: this only affects the run_id string, NOT the byte-gated
        # stdout loss curve.
        def arch_for(recipe)
          return "smollm2" if recipe == "lora"
          return "vit"     if recipe == "vit-tiny"
          return "moe"     if recipe == "franken-moe"
          "llama"
        end

        def resolve_run_id(template, project, arch)
          now  = Time.now
          date = now.strftime("%Y%m%d")
          time = now.strftime("%H%M%S")
          seq  = next_seq(project, date)
          template
            .gsub("{arch}", arch)
            .gsub("{date}", date)
            .gsub("{time}", time)
            .gsub("{seq}", format("%03d", seq))
        end

        # Daily counter: scan existing runs/* dir names containing today's
        # date substring; return max+1 (starting at 1). Robust to whatever the
        # template put around {seq} — we just look for the date.
        def next_seq(project, date)
          runs_dir = File.join(project, "runs")
          return 1 unless File.directory?(runs_dir)
          max = 0
          Dir.children(runs_dir).each do |name|
            next unless name.include?(date)
            # Trailing run of digits = the seq for this naming.
            m = name[/(\d+)\z/, 1]
            n = m ? m.to_i : 0
            max = n if n > max
          end
          max + 1
        end

        def emit(run_id, run_dir, losses)
          final = losses.empty? ? nil : losses.last
          final_loss = final && final =~ /loss=(.+)\z/ ? $1 : nil
          if @json
            puts JSON.pretty_generate(
              "format"     => FORMAT,
              "run_id"     => run_id,
              "run_dir"    => run_dir,
              "steps"      => @steps,
              "seed"       => @seed,
              "final_loss" => final_loss
            )
          else
            # Echo the full byte-deterministic loss curve through to our own
            # stdout (the runner's per-step lines), then the run pointer. The
            # curve is what prep/train_gate.rb asserts byte-for-byte.
            losses.each { |l| puts l }
            puts "run #{run_id} → #{run_dir}"
          end
          EXIT_OK
        end

        # Bad CLI input (unknown recipe/flag, bad --steps/--seed) → exit 2.
        def bad_arg(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy train: #{msg}"
          end
          EXIT_BAD_INPUT
        end

        # Execution failure (no root, build failed, runner crashed) → exit 1.
        # Clean one-liner, never a stacktrace.
        def fail_out(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy train: #{msg}"
          end
          EXIT_FAILURE
        end
      end
    end
  end
end
