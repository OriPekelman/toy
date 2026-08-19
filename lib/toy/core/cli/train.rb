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
        # toy#152 (DFA-arch T0) — the MLP-classifier anchor. Own binary
        # (landmine #16) and CPU-only by decision (tao#18: no CUDA twins
        # for T0–T3).
        MLP_RUNNER_TARGET = "libexec/toy-train-mlp"
        # toy#154 (DFA-arch T1) — the CTR tower. Own binary, CPU-only.
        CTR_RUNNER_TARGET = "libexec/toy-train-ctr"
        # toy#153 (DFA-arch T1) — the GNN node-classification lane. Own
        # binary, CPU-only.
        GNN_RUNNER_TARGET = "libexec/toy-train-gnn"
        # toy#155 (DFA-arch T2) — the selective-scan / Mamba-lite lane.
        # Own binary, CPU-only this slice (tao#19 defers its CUDA twin to
        # the long-sequence memory measurement).
        SSM_RUNNER_TARGET = "libexec/toy-train-ssm"
        # toy#156 (DFA-arch T2) — the latent diffusion denoiser. Own
        # binary, CPU-only.
        DIFF_RUNNER_TARGET = "libexec/toy-train-diff"
        # toy#157 (DFA-arch T3) — the LSTM lane, the SSM rehearsal. Own
        # binary, CPU-only (tao#18/#19: no CUDA twin for T0–T3).
        LSTM_RUNNER_TARGET = "libexec/toy-train-lstm"
        # toy#160 (DFA-arch T4) — the graph transformer, which separates
        # ATTENTION from OUTPUT DIM on the program's one open negative.
        # Own binary. CPU serves all three tasks; `--task bytelm` ALSO has
        # a CUDA twin (tao#24) because toy#170/P3 grew that task by orders
        # of magnitude (vocab 4096, ctx 128, ~3.2 TFLOP/cell) while the
        # relational task it was scoped against is still 16 classes.
        GTX_RUNNER_TARGET = "libexec/toy-train-gtx"
        # tao#24 — the byte-LM CUDA twin. SEPARATE single-type binary
        # (landmine #16) built from the GENERATED runner mirror, which
        # refuses every task but bytelm. There is no metal twin.
        GTX_CUDA_RUNNER_TARGET = "libexec/toy-train-gtx-cuda"
        # toy#165 (capstone P1a) — the per-token latent autoencoder. All
        # BP: no DFA and no diffusion on this lane. Own binary, CPU-only.
        AE_RUNNER_TARGET = "libexec/toy-train-ae"
        # toy#166 (capstone P1b) — the latent-diffusion byte-LM. Three
        # models per run, built SEQUENTIALLY (two live sessions corrupt
        # each other). Own binary, CPU-only, all BP.
        DIFFLM_RUNNER_TARGET = "libexec/toy-train-difflm"
        FRANKEN_MOE_RUNNER_TARGET = "libexec/toy-train-franken-moe-cli"
        FRANKEN_MOE_CUDA_RUNNER_TARGET = "libexec/toy-train-franken-moe-cli-cuda"
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
          @policy_scope = nil # franken: attn | ffn | all — which tensors a :dfa layer policies (toy#151)
          @dfa_b_seed  = nil  # franken B axes
          @dfa_b_dist  = nil
          @dfa_b_scale = nil
          @align_events = false
          @align_every  = nil   # franken: thin align/mask emissions (toy#122)
          @lr           = nil   # franken: hp-slot-0 override (toy#126)
          @warmup       = nil   # franken: linear lr ramp over N steps (toy#126)
          @no_shadow    = false # franken/franken-moe: skip dfa shadow (toy#129)
          @context      = nil   # franken/franken-moe: context override (toy#129)
          @ckpt_every   = nil   # franken: mid-run checkpoint cadence (toy#129)
          @eval_corpus  = nil   # franken-moe: end-of-run held-out eval (toy#130)
          @eval_tokens  = nil
          @eval_offset  = nil
          @vocab        = nil   # franken/franken-moe: headerless-pack vocab (toy#129)
          @batch        = nil   # franken/franken-moe: windows per step (toy#133)
          @load_ckpt    = nil   # franken-moe: eval-only checkpoint reload (toy#131)
          @act          = nil   # franken: swiglu | situ-glu (toy#136/K1)
          @rope         = nil   # franken: rope | nope (toy#136/K1)
          @schedule     = nil   # franken/franken-moe: const | cosine (toy#136/K1)
          @moe_balance  = nil   # franken-moe: aux | qb | none (toy#136/K1)
          @attn_gate    = false # franken-moe: K3 attention output gate (toy#136/K1.1)
          @kda_layers   = nil   # franken: KDA layer indices (toy#137/K2b)
          @mla_layers   = nil   # franken: Gated-MLA layer indices (K-series M2)
          @mla_rank     = nil   # franken: KV latent rank r (0/nil = derived)
          @kda_conv_off = false # franken: disable KDA ShortConv (toy#137/K2c)
          @layer_pattern = nil  # franken: hybrid layer preset (toy#138/K3a)
          @attnres      = false # franken: attention residuals (toy#138/K3b)
          @mtp          = false # franken: Multi-Token Prediction (K-series M10)
          @mtp_lambda   = nil   # franken: MTP backbone-COUPLING weight, not a loss weight
          @optimizer    = nil   # franken-moe: adamw | muon | sgd (toy#139)
          @donor        = nil   # franken-moe: donor GGUF for embed transfer (toy#140)
          @donor_mode   = nil
          @freeze_embed = false
          @freeze_experts = false  # franken-moe: the R2 inert-experts control (toy#141)
          @moe_latent   = false # franken-moe: latent expert sandwich (toy#142/K4)
          @moe_shared   = nil   # franken-moe: N shared full-width experts (toy#142/K4)
          @dfa_granularity = nil # franken-moe: matmul | block (toy#143)
          @dfa_feedback = nil    # franken-moe: fixed | kolen-pollack (toy#150)
          @dfa_feedback_decay = nil
          @dfa_feedback_lr = nil
          @expert_act   = nil   # franken-moe: gelu | situ-glu (K4b/M6)
          @lr_schedule  = nil   # franken-moe: uniform | ramp-up | ramp-down (toy#146)
          @lr_lo        = nil   # franken-moe: ramp endpoint, layer 0 side
          @lr_hi        = nil   # franken-moe: ramp endpoint, last layer side
          @lr_control   = nil   # franken-moe: none | reactive (toy#148)
          @lr_control_window   = nil # EMA window for the smoothed loss
          @lr_control_patience = nil # non-improving steps before a cut
          @lr_control_factor   = nil # cut multiplier, 0 < F < 1
          @lr_control_recover  = nil # per-step restore toward 1.0, R >= 1
          @lr_control_floor    = nil # min ctrl, 0 < X <= 1
          @shape        = nil   # franken/franken-moe: preset (toy#124)
          @routing    = nil   # franken-moe: dense | top1
          # toy#152 (mlp): the T0 anchor's own shape/task knobs.
          @classes     = nil  # output dim — the axis under test
          @hidden      = nil  # hidden width
          @features    = nil  # input dim
          @mlp_layers  = nil  # hidden layer count
          @task        = nil  # teacher | blobs
          @teacher_dim = nil
          @task_seed   = nil
          @val_batches = nil
          # toy#154 (ctr): the tower's own shape/task knobs.
          @fields      = nil
          @cardinality = nil
          @numeric     = nil
          @emb         = nil
          @pairs       = nil
          @base_rate   = nil
          @lin_scale   = nil
          @fm_branch   = false
          # toy#153 (gnn): the graph lane's own shape/task/feedback knobs.
          @graph          = nil  # bundle prefix (prep/fetch_cora.rb)
          @nodes          = nil
          @degree         = nil
          @homophily      = nil
          @feat_signal    = nil
          @train_per_class = nil
          @feedback_route = nil  # direct | structure
          @feedback_hops  = nil
          @weight_decay   = nil
          # toy#155 (ssm): the sequence lane's own shape/task knobs.
          @seq            = nil
          @d_model        = nil
          # toy#160 (gtx): the graph transformer's own shape knobs.
          # --heads/--d-ff/--types are net-new; --nodes, --features,
          # --pairs, --task and --dfa-cut are shared with the lanes that
          # already own those names.
          @heads          = nil
          @d_ff           = nil
          @types          = nil
          @entities       = nil
          # toy#172 (E1 Phase 1.2): the nDFA error-side preconditioner on
          # the gtx byte-LM lane. All nil = OFF, and off is byte-null
          # against the runner as it stood before the flag existed.
          @dfa_feedback_precond = nil
          @ndfa_lambda    = nil
          @ndfa_every     = nil
          @ndfa_samples   = nil
          @ndfa_gain      = nil
          # toy#165 (ae): the latent-autoencoder lane (capstone P1a).
          #
          # --text is NOT --corpus and there is NO --vocab here. --corpus
          # names a TOYC pack on the LM lanes; --text names a byte pack
          # from prep/fetch_text.rb, which is a different thing and so
          # gets a different name (the tao#18 --policy-scope discipline).
          # --vocab is worse: it already means an INTEGER >= 2 on
          # franken/franken-moe, so honouring the spec's `--vocab byte`
          # would make one flag mean an integer on one recipe and a
          # string on another — the hp-slot-5/6 dual-meaning trap. This
          # lane is byte-level and its head is 256-wide on every corpus,
          # so a --vocab knob would have exactly one legal value anyway.
          # --latent is REUSED from the diff lane rather than a net-new
          # --latent-dim (which is what the P1a spec writes). It is the
          # SAME quantity — F20/toy#156's "latent 4" and P1a's bottleneck
          # 4 are the number the capstone compares directly — and two
          # names for one concept is how that comparison gets
          # mis-transcribed.
          @text           = nil
          @noise_eval     = nil
          @noise_seed     = nil
          @val_frac_pct   = nil
          # toy#165 follow-up: MATCHED-CE stopping. The published P1a
          # surface compared cells at matched STEPS and their clean CE
          # spanned seven orders of magnitude; a noise MARGIN is not
          # invariant to that, so the surface was partly a convergence
          # surface. --target-ce stops each cell at equal reconstruction
          # fidelity instead, which is also the question a decoder is
          # actually asked. OFF by default and byte-null when off.
          # toy#166 (difflm): the P1b arms and their knobs.
          @arm            = nil
          @ae_steps       = nil
          @tsteps         = nil
          @gen_bytes      = nil
          @judge_steps    = nil
          # toy#168 (P1b3): the stage-2 objective axis. Weights are applied
          # as a t-SAMPLING distribution, not a loss multiplier — this lane
          # draws one t per step and Adam is scale-invariant, so a
          # multiplier would be inert (toy#152's B-scale landmine).
          @loss_weight    = nil
          @minsnr_gamma   = nil
          @target_ce      = nil
          @eval_every     = nil
          @probe_batches  = nil
          # toy#162 (lstm): global-norm gradient clipping. nil = OFF, and
          # off is byte-null — every cell toy#157 published was measured
          # without it.
          @clip_grad      = nil
          # toy#161 (gtx retrofit): freeze a BP-pretrained backbone and
          # train ADDED capacity by chain|dfa|frozen.
          @retrofit        = false
          @pretrain_steps  = nil
          @pretrain_lr     = nil
          @adapter_policy  = nil
          @adapter_layers  = nil
          @adapter_rank    = nil
          @no_freeze_backbone = false
          @d_inner        = nil
          @conv_k         = nil
          @selection      = nil  # selective | lti
          @dfa_cut        = nil  # layer | step
          @cue_span       = nil
          @noise          = nil
          @dt_init        = nil
          # toy#156 (diff): the denoiser's own shape/task/schedule knobs.
          @latent         = nil
          @time_feat      = nil
          @modes          = nil
          @spread         = nil
          @mode_scale     = nil
          @diff_steps     = nil
          @beta_lo        = nil
          @beta_hi        = nil
          @eval_n         = nil
          # toy#157 (lstm) adds NO ivars of its own: it is deliberately
          # the same shape/task/cut axis as the ssm lane (--seq, --classes,
          # --task cue|mean, --cue-span, --noise, --dfa-cut layer|step) so
          # the two lanes read as one architecture comparison. What it does
          # NOT reuse is --selection/--d-inner/--conv-k/--dt-init, which
          # name selective-scan parts an LSTM does not have.
          @moe_policy = nil   # franken-moe: chain | dfa-experts
          @moe_aux    = nil   # franken-moe: top1 aux-loss alpha
          @experts    = nil   # franken-moe: expert count E>=2 (toy#128)
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
          elsif %w[franken franken-moe].include?(@recipe) && @corpus
            unless File.file?(@corpus)
              return bad_arg("no such file: #{@corpus} (#{@recipe} --corpus streams " \
                             "packed-i32 tokens; `toy new` seeds data/ts_seqs.bin)")
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
          target = if @recipe == "difflm"
                     DIFFLM_RUNNER_TARGET
                   elsif @recipe == "difflm"
            # Lane-local DL_* namespace. --latent defaults to 8: P1a's
            # PINNED operating point (matched-CE surface, tao cf308d3),
            # not a guess.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "DL_ARM"         => (@arm || "diff-selfcond"),
                             "DL_TEXT"        => (@text || ""),
                             "DL_LATENT"      => (@latent || 8).to_s,
                             "DL_CONTEXT"     => (@context || 256).to_s,
                             "DL_AE_STEPS"    => (@ae_steps || 2000).to_s,
                             "DL_D_MODEL"     => (@d_model || 128).to_s,
                             "DL_BLOCKS"      => (@mlp_layers || 2).to_s,
                             "DL_HEADS"       => (@heads || 4).to_s,
                             "DL_D_FF"        => (@d_ff || 256).to_s,
                             "DL_TSTEPS"      => (@tsteps || 100).to_s,
                             "DL_GEN_BYTES"   => (@gen_bytes || 16384).to_s,
                             "DL_JUDGE_STEPS" => (@judge_steps || 3000).to_s,
                             "DL_LR"          => (@lr || ""),
                             "DL_WARMUP"      => (@warmup || 0).to_s,
                             "DL_TASK_SEED"   => (@task_seed || 7).to_s,
                             "DL_NOISE_SEED"  => (@noise_seed || 4242).to_s,
                             "DL_LOSS_WEIGHT" => (@loss_weight || ""),
                             "DL_MINSNR_GAMMA" => (@minsnr_gamma || "5.0"))
          elsif @recipe == "ae"
                     AE_RUNNER_TARGET
                   elsif @recipe == "gtx"
                     # tao#24: cuda reaches here ONLY under --task bytelm
                     # (validate! refuses every other combination), so the
                     # twin binary is never selected for the relational lane.
                     @device == "cuda" ? GTX_CUDA_RUNNER_TARGET : GTX_RUNNER_TARGET
                   elsif @recipe == "lstm"
                     LSTM_RUNNER_TARGET
                   elsif @recipe == "diff"
                     DIFF_RUNNER_TARGET
                   elsif @recipe == "ssm"
                     SSM_RUNNER_TARGET
                   elsif @recipe == "gnn"
                     GNN_RUNNER_TARGET
                   elsif @recipe == "ctr"
                     CTR_RUNNER_TARGET
                   elsif @recipe == "mlp"
                     MLP_RUNNER_TARGET
                   elsif @recipe == "franken-moe"
                     @device == "cuda" ? FRANKEN_MOE_CUDA_RUNNER_TARGET : FRANKEN_MOE_RUNNER_TARGET
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
          elsif @recipe == "mlp"
            # toy#152: a LANE-LOCAL env namespace (MLP_*), not the
            # FRANKEN_* one. The two lanes share Toy::Train::DfaB, not a
            # flag vocabulary — `frozen` is an mlp policy token with no
            # franken counterpart, and franken's `mix:`/`mask*:` tokens
            # have no meaning here. One namespace per lane keeps a
            # copy-pasted command from silently half-applying.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "MLP_POLICY"       => (@policy || ""),
                             "MLP_LAYERS"       => (@mlp_layers || 3).to_s,
                             "MLP_HIDDEN"       => (@hidden || 64).to_s,
                             "MLP_FEATURES"     => (@features || 32).to_s,
                             "MLP_CLASSES"      => (@classes || 10).to_s,
                             "MLP_TASK"         => (@task || ""),
                             "MLP_TEACHER_DIM"  => (@teacher_dim || 32).to_s,
                             "MLP_TASK_SEED"    => (@task_seed || 7).to_s,
                             "MLP_BATCH"        => (@batch || 64).to_s,
                             "MLP_VAL_BATCHES"  => (@val_batches || 8).to_s,
                             "MLP_LR"           => (@lr || ""),
                             "MLP_WARMUP"       => (@warmup || 0).to_s,
                             "MLP_B_SEED"       => (@dfa_b_seed || 1234).to_s,
                             "MLP_B_DIST"       => (@dfa_b_dist || ""),
                             "MLP_B_SCALE"      => (@dfa_b_scale || ""),
                             "MLP_ALIGN"        => (@align_events ? "1" : ""),
                             "MLP_ALIGN_EVERY"  => (@align_every || 1).to_s)
          elsif @recipe == "diff"
            # Lane-local DIFF_* namespace, same discipline as the others.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "DIFF_POLICY"     => (@policy || ""),
                             "DIFF_LAYERS"     => (@mlp_layers || 3).to_s,
                             "DIFF_HIDDEN"     => (@hidden || 128).to_s,
                             "DIFF_LATENT"     => (@latent || 16).to_s,
                             "DIFF_TIME_FEAT"  => (@time_feat || 8).to_s,
                             "DIFF_TASK"       => (@task || ""),
                             "DIFF_MODES"      => (@modes || 8).to_s,
                             "DIFF_SPREAD"     => (@spread ? @spread.to_s : ""),
                             "DIFF_SCALE"      => (@mode_scale ? @mode_scale.to_s : ""),
                             "DIFF_DIFF_STEPS" => (@diff_steps || 100).to_s,
                             "DIFF_BETA_LO"    => (@beta_lo ? @beta_lo.to_s : ""),
                             "DIFF_BETA_HI"    => (@beta_hi ? @beta_hi.to_s : ""),
                             "DIFF_TASK_SEED"  => (@task_seed || 7).to_s,
                             "DIFF_BATCH"      => (@batch || 128).to_s,
                             "DIFF_EVAL_N"     => (@eval_n || 512).to_s,
                             "DIFF_LR"         => (@lr || ""),
                             "DIFF_WARMUP"     => (@warmup || 0).to_s,
                             "DIFF_B_SEED"     => (@dfa_b_seed || 1234).to_s,
                             "DIFF_B_DIST"     => (@dfa_b_dist || ""),
                             "DIFF_B_SCALE"    => (@dfa_b_scale || ""),
                             "DIFF_ALIGN"      => (@align_events ? "1" : ""),
                             "DIFF_ALIGN_EVERY" => (@align_every || 1).to_s)
          elsif @recipe == "difflm"
            # Lane-local DL_* namespace. --latent defaults to 8: P1a's
            # PINNED operating point (matched-CE surface, tao cf308d3),
            # not a guess.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "DL_ARM"         => (@arm || "diff-selfcond"),
                             "DL_TEXT"        => (@text || ""),
                             "DL_LATENT"      => (@latent || 8).to_s,
                             "DL_CONTEXT"     => (@context || 256).to_s,
                             "DL_AE_STEPS"    => (@ae_steps || 2000).to_s,
                             "DL_D_MODEL"     => (@d_model || 128).to_s,
                             "DL_BLOCKS"      => (@mlp_layers || 2).to_s,
                             "DL_HEADS"       => (@heads || 4).to_s,
                             "DL_D_FF"        => (@d_ff || 256).to_s,
                             "DL_TSTEPS"      => (@tsteps || 100).to_s,
                             "DL_GEN_BYTES"   => (@gen_bytes || 16384).to_s,
                             "DL_JUDGE_STEPS" => (@judge_steps || 3000).to_s,
                             "DL_LR"          => (@lr || ""),
                             "DL_WARMUP"      => (@warmup || 0).to_s,
                             "DL_TASK_SEED"   => (@task_seed || 7).to_s,
                             "DL_NOISE_SEED"  => (@noise_seed || 4242).to_s,
                             "DL_LOSS_WEIGHT" => (@loss_weight || ""),
                             "DL_MINSNR_GAMMA" => (@minsnr_gamma || "5.0"))
          elsif @recipe == "ae"
            # Lane-local AE_* namespace, same discipline as the others.
            # There is no synthetic default corpus: --text is REQUIRED and
            # the runner refuses without it, because a low-entropy
            # synthetic byte stream would inflate the noise margin at
            # exactly the small latent where the verdict is decided
            # (tao#22). `--context` is also the batch on this lane — the
            # encoder attends within a window, so T positions ARE the T
            # reconstruction targets and there is no separate --batch.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "AE_TEXT"         => (@text || ""),
                             "AE_LATENT"       => (@latent || 8).to_s,
                             "AE_CONTEXT"      => (@context || 256).to_s,
                             "AE_BLOCKS"       => (@mlp_layers || 2).to_s,
                             "AE_D_MODEL"      => (@d_model || 128).to_s,
                             "AE_HEADS"        => (@heads || 4).to_s,
                             "AE_D_FF"         => (@d_ff || 256).to_s,
                             "AE_NOISE_EVAL"   => (@noise_eval || ""),
                             "AE_NOISE_SEED"   => (@noise_seed || 4242).to_s,
                             "AE_VAL_BATCHES"  => (@val_batches || 16).to_s,
                             "AE_VAL_FRAC_PCT" => (@val_frac_pct || 10).to_s,
                             "AE_TASK_SEED"    => (@task_seed || 7).to_s,
                             "AE_TARGET_CE"    => (@target_ce || ""),
                             "AE_EVAL_EVERY"   => (@eval_every || 0).to_s,
                             "AE_PROBE_BATCHES" => (@probe_batches || 4).to_s,
                             "AE_LR"           => (@lr || ""),
                             "AE_WARMUP"       => (@warmup || 0).to_s)
          elsif @recipe == "gtx"
            # Lane-local GTX_* namespace, same discipline as the others.
            # NOTE the LR default (0.003) is BP's, and the DFA arms want
            # ~3x less — measured, and stated in the roadmap rather than
            # folded into a default, for the reason toy#157 learned the
            # hard way: a default that silently redefines a cell
            # relabels other people's experiments.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "GTX_POLICY"      => (@policy || ""),
                             "GTX_DFA_CUT"     => (@dfa_cut || ""),
                             "GTX_BLOCKS"      => (@mlp_layers || 2).to_s,
                             "GTX_D_MODEL"     => (@d_model || 64).to_s,
                             "GTX_HEADS"       => (@heads || 4).to_s,
                             "GTX_D_FF"        => (@d_ff || 128).to_s,
                             "GTX_ENTITIES"    => (@entities || 48).to_s,
                             "GTX_TYPES"       => (@types || 4).to_s,
                             "GTX_FEATURES"    => (@features || 16).to_s,
                             "GTX_NOISE"       => (@noise ? @noise.to_s : ""),
                             "GTX_TASK"        => (@task || ""),
                             "GTX_TEXT"        => (@text || ""),
                             # EXPLICIT default, never "": train_gtx reads
                             # (ENV["GTX_VOCAB"] || "256").to_i, and "" is
                             # TRUTHY in Ruby, so an empty string would
                             # silently become vocab 0 rather than 256.
                             "GTX_VOCAB"       => (@vocab || 256).to_s,
                             "GTX_PAIRS"       => (@batch || 128).to_s,
                             "GTX_VAL_BATCHES" => (@val_batches || 8).to_s,
                             "GTX_TASK_SEED"   => (@task_seed || 7).to_s,
                             "GTX_LR"          => (@lr || ""),
                             "GTX_WARMUP"      => (@warmup || 0).to_s,
                             "GTX_B_SEED"      => (@dfa_b_seed || 1234).to_s,
                             "GTX_B_DIST"      => (@dfa_b_dist || ""),
                             "GTX_B_SCALE"     => (@dfa_b_scale || ""),
                             "GTX_RETROFIT"       => (@retrofit ? "1" : ""),
                             "GTX_PRETRAIN_STEPS" => (@pretrain_steps || 1500).to_s,
                             "GTX_PRETRAIN_LR"    => (@pretrain_lr || ""),
                             "GTX_ADAPTER_POLICY" => (@adapter_policy || ""),
                             "GTX_ADAPTER_LAYERS" => (@adapter_layers || 2).to_s,
                             "GTX_ADAPTER_RANK"   => (@adapter_rank || 16).to_s,
                             "GTX_FREEZE_BACKBONE" => (@no_freeze_backbone ? "0" : "1"),
                             "GTX_CKPT_EVERY"     => (@ckpt_every || 0).to_s,
                             # toy#172/E1 Phase 1.2 — nDFA. EXPLICIT
                             # defaults, never "": the runner reads these
                             # with `(ENV[..] || "N").to_i` and an empty
                             # string is TRUTHY in Ruby, so "" would come
                             # back 0 rather than the default (the same
                             # trap GTX_VOCAB documents two blocks up).
                             # The lambda is the exception and is passed
                             # through verbatim — it has NO default and
                             # the runner refuses without it.
                             "GTX_NDFA"        => (@dfa_feedback_precond == "ndfa" ? "1" : "0"),
                             "GTX_NDFA_LAMBDA" => (@ndfa_lambda || ""),
                             "GTX_NDFA_EVERY"  => (@ndfa_every || 500).to_s,
                             "GTX_NDFA_M"      => (@ndfa_samples || 256).to_s,
                             "GTX_NDFA_GAIN"   => (@ndfa_gain || ""),
                             "GTX_LOAD_CKPT"      => (@load_ckpt || ""))
          elsif @recipe == "lstm"
            # Lane-local LSTM_* namespace, same discipline as the others.
            # Defaults CHANGE NOTHING here, deliberately. This lane's
            # fair cell is `--lr 0.02 --warmup 200 --steps 4000` and it
            # is written out wherever the lane's numbers are stated, NOT
            # folded into a default: toy#157 shipped the 200-step ramp as
            # a default for one commit and Tao's `--lr 0.03 --steps 2000`
            # silently inherited it, relabelling a 3-seed matrix. Flag
            # strings are experiment identity in this repo.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "LSTM_POLICY"      => (@policy || ""),
                             "LSTM_DFA_CUT"     => (@dfa_cut || ""),
                             "LSTM_LAYERS"      => (@mlp_layers || 1).to_s,
                             "LSTM_HIDDEN"      => (@hidden || 64).to_s,
                             "LSTM_FEATURES"    => (@features || 24).to_s,
                             "LSTM_SEQ"         => (@seq || 64).to_s,
                             "LSTM_CLASSES"     => (@classes || 4).to_s,
                             "LSTM_TASK"        => (@task || ""),
                             "LSTM_CUE_SPAN"    => (@cue_span ? @cue_span.to_s : ""),
                             "LSTM_NOISE"       => (@noise ? @noise.to_s : ""),
                             "LSTM_TASK_SEED"   => (@task_seed || 7).to_s,
                             "LSTM_BATCH"       => (@batch || 32).to_s,
                             "LSTM_VAL_BATCHES" => (@val_batches || 8).to_s,
                             "LSTM_LR"          => (@lr || ""),
                             "LSTM_WARMUP"      => (@warmup || 0).to_s,
                             "LSTM_B_SEED"      => (@dfa_b_seed || 1234).to_s,
                             "LSTM_B_DIST"      => (@dfa_b_dist || ""),
                             "LSTM_B_SCALE"     => (@dfa_b_scale || ""),
                             "LSTM_CLIP_GRAD"   => (@clip_grad ? @clip_grad.to_s : ""))
          elsif @recipe == "ssm"
            # Lane-local SSM_* namespace, same discipline as the others.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "SSM_POLICY"      => (@policy || ""),
                             "SSM_SELECTION"   => (@selection || ""),
                             "SSM_DFA_CUT"     => (@dfa_cut || ""),
                             "SSM_LAYERS"      => (@mlp_layers || 2).to_s,
                             "SSM_D_MODEL"     => (@d_model || 24).to_s,
                             "SSM_D_INNER"     => (@d_inner || 48).to_s,
                             "SSM_SEQ"         => (@seq || 64).to_s,
                             "SSM_CONV_K"      => (@conv_k || 4).to_s,
                             "SSM_CLASSES"     => (@classes || 4).to_s,
                             "SSM_TASK"        => (@task || ""),
                             "SSM_TEXT"        => (@text || ""),
                             "SSM_CUE_SPAN"    => (@cue_span ? @cue_span.to_s : ""),
                             "SSM_NOISE"       => (@noise ? @noise.to_s : ""),
                             "SSM_TASK_SEED"   => (@task_seed || 7).to_s,
                             "SSM_BATCH"       => (@batch || 32).to_s,
                             "SSM_VAL_BATCHES" => (@val_batches || 8).to_s,
                             "SSM_DT_INIT"     => (@dt_init ? @dt_init.to_s : ""),
                             "SSM_LR"          => (@lr || ""),
                             "SSM_WARMUP"      => (@warmup || 0).to_s,
                             "SSM_B_SEED"      => (@dfa_b_seed || 1234).to_s,
                             "SSM_B_DIST"      => (@dfa_b_dist || ""),
                             "SSM_B_SCALE"     => (@dfa_b_scale || ""))
          elsif @recipe == "gnn"
            # Lane-local GNN_* namespace, same discipline as MLP_*/CTR_*.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "GNN_POLICY"          => (@policy || ""),
                             "GNN_LAYERS"          => (@mlp_layers || 1).to_s,
                             "GNN_HIDDEN"          => (@hidden || 32).to_s,
                             "GNN_GRAPH"           => (@graph || ""),
                             "GNN_NODES"           => (@nodes ? @nodes.to_s : ""),
                             "GNN_FEATURES"        => (@features ? @features.to_s : ""),
                             "GNN_CLASSES"         => (@classes ? @classes.to_s : ""),
                             "GNN_DEGREE"          => (@degree ? @degree.to_s : ""),
                             "GNN_HOMOPHILY"       => (@homophily ? @homophily.to_s : ""),
                             "GNN_FEAT_SIGNAL"     => (@feat_signal ? @feat_signal.to_s : ""),
                             "GNN_TASK"            => (@task || ""),
                             "GNN_TEACHER_DIM"     => (@teacher_dim ? @teacher_dim.to_s : ""),
                             "GNN_TASK_SEED"       => (@task_seed || 7).to_s,
                             "GNN_TRAIN_PER_CLASS" => (@train_per_class ? @train_per_class.to_s : ""),
                             "GNN_FEEDBACK_ROUTE"  => (@feedback_route || ""),
                             "GNN_FEEDBACK_HOPS"   => (@feedback_hops ? @feedback_hops.to_s : ""),
                             "GNN_LR"              => (@lr || ""),
                             "GNN_WD"              => (@weight_decay ? @weight_decay.to_s : ""),
                             "GNN_WARMUP"          => (@warmup || 0).to_s,
                             "GNN_B_SEED"          => (@dfa_b_seed || 1234).to_s,
                             "GNN_B_DIST"          => (@dfa_b_dist || ""),
                             "GNN_B_SCALE"         => (@dfa_b_scale || ""),
                             "GNN_ALIGN"           => (@align_events ? "1" : ""),
                             "GNN_ALIGN_EVERY"     => (@align_every || 1).to_s)
          elsif @recipe == "ctr"
            # Lane-local CTR_* namespace, same discipline as MLP_*.
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "CTR_POLICY"      => (@policy || ""),
                             "CTR_FIELDS"      => (@fields || 8).to_s,
                             "CTR_CARD"        => (@cardinality || 64).to_s,
                             "CTR_NUMERIC"     => (@numeric || 4).to_s,
                             "CTR_EMB"         => (@emb || 8).to_s,
                             "CTR_HIDDEN"      => (@hidden || 64).to_s,
                             "CTR_LAYERS"      => (@mlp_layers || 3).to_s,
                             "CTR_PAIRS"       => (@pairs || 12).to_s,
                             "CTR_BASE_RATE"   => (@base_rate ? @base_rate.to_s : ""),
                             "CTR_LIN_SCALE"   => (@lin_scale ? @lin_scale.to_s : ""),
                             "CTR_WIDE"        => (@fm_branch ? "1" : ""),
                             "CTR_TASK_SEED"   => (@task_seed || 7).to_s,
                             "CTR_BATCH"       => (@batch || 128).to_s,
                             "CTR_VAL_BATCHES" => (@val_batches || 16).to_s,
                             "CTR_LR"          => (@lr || ""),
                             "CTR_WARMUP"      => (@warmup || 0).to_s,
                             "CTR_B_SEED"      => (@dfa_b_seed || 1234).to_s,
                             "CTR_B_DIST"      => (@dfa_b_dist || ""),
                             "CTR_B_SCALE"     => (@dfa_b_scale || ""))
          elsif @recipe == "franken-moe"
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "FRANKEN_MOE_ROUTING" => (@routing || "dense"),
                             "FRANKEN_MOE"         => (@moe_policy || "chain"),
                             "FRANKEN_MOE_AUX"     => (@moe_aux || "0"),
                             "FRANKEN_MOE_EXPERTS" => (@experts || 2).to_s,
                             "FRANKEN_NO_SHADOW"   => (@no_shadow ? "1" : ""),
                             "FRANKEN_CONTEXT"     => (@context || 0).to_s,
                             "FRANKEN_VOCAB"       => (@vocab || 0).to_s,
                             "FRANKEN_BATCH"       => (@batch || 0).to_s,
                             "FRANKEN_EVAL_CORPUS" => (@eval_corpus || ""),
                             "FRANKEN_EVAL_TOKENS" => (@eval_tokens || 0).to_s,
                             "FRANKEN_EVAL_OFFSET" => (@eval_offset || 0).to_s,
                             "FRANKEN_LR"          => (@lr || ""),
                             "FRANKEN_WARMUP"      => (@warmup || 0).to_s,
                             "FRANKEN_SCHEDULE"    => (@schedule || ""),
                             "FRANKEN_MOE_BALANCE" => (@moe_balance || ""),
                             "FRANKEN_OPTIMIZER"   => (@optimizer || ""),
                             "FRANKEN_DONOR"       => (@donor || ""),
                             "FRANKEN_DONOR_MODE"  => (@donor_mode || ""),
                             "FRANKEN_FREEZE_EMBED" => (@freeze_embed ? "1" : ""),
                             "FRANKEN_FREEZE_EXPERTS" => (@freeze_experts ? "1" : ""),
                             "FRANKEN_MOE_LATENT"  => (@moe_latent ? "1" : ""),
                             "FRANKEN_DFA_GRANULARITY" => (@dfa_granularity || ""),
                             "FRANKEN_DFA_FEEDBACK" => (@dfa_feedback || ""),
                             "FRANKEN_DFA_FEEDBACK_DECAY" => (@dfa_feedback_decay ? @dfa_feedback_decay.to_s : ""),
                             "FRANKEN_DFA_FEEDBACK_LR" => (@dfa_feedback_lr ? @dfa_feedback_lr.to_s : ""),
                             "FRANKEN_EXPERT_ACT" => (@expert_act || ""),
                             "FRANKEN_LR_SCHEDULE" => (@lr_schedule || ""),
                             "FRANKEN_LR_LO" => (@lr_lo ? @lr_lo.to_s : ""),
                             "FRANKEN_LR_HI" => (@lr_hi ? @lr_hi.to_s : ""),
                             "FRANKEN_LR_CONTROL" => (@lr_control || ""),
                             "FRANKEN_LR_CONTROL_WINDOW"   => (@lr_control_window   ? @lr_control_window.to_s   : ""),
                             "FRANKEN_LR_CONTROL_PATIENCE" => (@lr_control_patience ? @lr_control_patience.to_s : ""),
                             "FRANKEN_LR_CONTROL_FACTOR"   => (@lr_control_factor   ? @lr_control_factor.to_s   : ""),
                             "FRANKEN_LR_CONTROL_RECOVER"  => (@lr_control_recover  ? @lr_control_recover.to_s  : ""),
                             "FRANKEN_LR_CONTROL_FLOOR"    => (@lr_control_floor    ? @lr_control_floor.to_s    : ""),
                             "FRANKEN_MOE_SHARED"  => (@moe_shared || 0).to_s,
                             "FRANKEN_ATTN_GATE"   => (@attn_gate ? "1" : ""),
                             "FRANKEN_CKPT_EVERY"  => (@ckpt_every || 0).to_s,
                             "FRANKEN_MOE_LOAD"    => (@load_ckpt || ""),
                             "FRANKEN_SHAPE"       => (@shape || "base"),
                             "FRANKEN_B_SEED"      => (@dfa_b_seed || 1234).to_s,
                             "FRANKEN_B_DIST"      => (@dfa_b_dist || ""),
                             "FRANKEN_B_SCALE"     => (@dfa_b_scale || ""),
                             "FRANKEN_ALIGN"       => (@align_events ? "1" : ""),
                             "FRANKEN_ALIGN_EVERY" => (@align_every || 1).to_s,
                             "CORPUS"              => (@corpus || ""))
          elsif @recipe == "franken"
            env = base.merge("STEPS" => @steps.to_s, "SEED" => @seed.to_s,
                             "FRANKEN_POLICY"  => (@policy || ""),
                             "FRANKEN_POLICY_SCOPE" => (@policy_scope || ""),
                             "FRANKEN_B_SEED"  => (@dfa_b_seed || 0).to_s,
                             "FRANKEN_B_DIST"  => (@dfa_b_dist || ""),
                             "FRANKEN_B_SCALE" => (@dfa_b_scale || ""),
                             "FRANKEN_ALIGN"   => (@align_events ? "1" : ""),
                             "FRANKEN_ALIGN_EVERY" => (@align_every || 1).to_s,
                             "FRANKEN_SHAPE"   => (@shape || "base"),
                             "CORPUS"          => (@corpus || ""),
                             "FRANKEN_OPTIMIZER" => (@optimizer || ""),
                             "FRANKEN_DFA_GRANULARITY" => (@dfa_granularity || ""),
                             "FRANKEN_LR"      => (@lr || ""),
                             "FRANKEN_WARMUP"  => (@warmup || 0).to_s,
                             "FRANKEN_NO_SHADOW" => (@no_shadow ? "1" : ""),
                             "FRANKEN_CONTEXT" => (@context || 0).to_s,
                             "FRANKEN_VOCAB"   => (@vocab || 0).to_s,
                             "FRANKEN_BATCH"   => (@batch || 0).to_s,
                             "FRANKEN_ACT"     => (@act || ""),
                             "KDA_LAYERS"      => (@kda_layers || ""),
                             "MLA_LAYERS"      => (@mla_layers || ""),
                             "FRANKEN_MLA_RANK" => (@mla_rank ? @mla_rank.to_s : ""),
                             "KDA_CONV"        => (@kda_conv_off ? "0" : ""),
                             "FRANKEN_LAYER_PATTERN" => (@layer_pattern || ""),
                             "ATTNRES"         => (@attnres ? "1" : ""),
                             "FRANKEN_MTP"     => (@mtp ? "1" : ""),
                             "FRANKEN_MTP_LAMBDA" => (@mtp_lambda ? @mtp_lambda.to_s : ""),
                             "FRANKEN_NOPE"    => (@rope == "nope" ? "1" : ""),
                             "FRANKEN_SCHEDULE" => (@schedule || ""),
                             "FRANKEN_CKPT_EVERY" => (@ckpt_every || 0).to_s)
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

          # toy#130: the end-of-run eval_ce summary rides stdout alongside
          # the byte-gated step lines; toy#152 adds the mlp lane's "val:"
          # held-out line (accuracy is the metric its success bar is
          # stated in, so it must reach the caller, not just events).
          # toy#155 adds "train:" (gnn) and "graph:" (ssm): the graph-size
          # line is how a sweep over --seq reads the arms' scaling
          # without opening a bundle.
          # toy#159 adds "stream:" — the ANALYTIC activation-memory line
          # the recurrent lanes print beside the measured "graph:" one.
          # It has to reach the caller for the same reason "graph:" does:
          # a sweep over --seq reads the arms' scaling from stdout, and
          # this is the line the memory claim is actually stated in.
          # toy#165 adds the ae lane's four: "corpus:" (which alphabet was
          # actually separated — the margin curve is unscoped without it),
          # "noise:" (the curve itself, which IS that lane's result),
          # "half_snr:" (its one scalar per cell) and "control:" (the
          # can-it-lose precondition). A sweep over --latent-dim reads all
          # four off stdout, so filtering them here would leave the lane's
          # actual finding reachable only by opening a bundle.
          # toy#169/#170 add the byte-LM pair, and they are the reason the
          # CLI could not carry P2-P6 at all: "bytelm:" IS the lane's
          # result (held-out bpb — every P3/P4/P5/P6 number is quoted from
          # it), and "gtx:" is the wiring line the sweeps grep for
          # `vocab=` and `b_dim=` to prove a cell is the arm and the head
          # width it claims. Filtered, a CLI run trains correctly and
          # reports neither its result nor its provenance — which is
          # exactly how every cell ended up driven through the env
          # harness instead.
          losses = out.lines.select { |l| l.start_with?("step ") || l.start_with?("eval_ce:") || l.start_with?("val:") || l.start_with?("train:") || l.start_with?("graph:") || l.start_with?("stream:") || l.start_with?("gen:") || l.start_with?("corpus:") || l.start_with?("noise:") || l.start_with?("half_snr:") || l.start_with?("control:") || l.start_with?("latent_std:") || l.start_with?("converged:") || l.start_with?("stage1:") || l.start_with?("arm:") || l.start_with?("resid:") || l.start_with?("judge:") || l.start_with?("objective:") || l.start_with?("epsmse:") || l.start_with?("bytelm:") || l.start_with?("ndfa:") || l.start_with?("bcond") || l.start_with?("gtx: ") }.map(&:chomp)
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
              # toy#131: 0 is legal for the eval-only reload path
              # (--load-ckpt); the runner rejects 0 everywhere else.
              return bad_arg("--steps must be a non-negative integer, got #{val.inspect}") unless val =~ /\A\d+\z/
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
              # toy#131: 0 is legal for the eval-only reload path
              # (--load-ckpt); the runner rejects 0 everywhere else.
              return bad_arg("--steps must be a non-negative integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @steps = val.to_i
            when /\A--seed=(.*)\z/
              val = $1
              return bad_arg("--seed must be a non-negative integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @seed = val.to_i
            when /\A--out=(.*)\z/m
              @out = $1
            # ---- toy#152 (mlp): the T0 anchor's shape/task knobs ----
            # --classes is the AXIS UNDER TEST (the output dim our lens
            # says DFA degrades with), so it gets a real name rather
            # than riding --vocab: nothing here is a vocabulary.
            when "--classes", "--hidden", "--features", "--layers",
                 "--teacher-dim", "--val-batches"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              min = key == "--classes" ? 2 : 1
              return bad_arg("#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "--classes"     then @classes     = val.to_i
              when "--hidden"      then @hidden      = val.to_i
              when "--features"    then @features    = val.to_i
              when "--layers"      then @mlp_layers  = val.to_i
              when "--teacher-dim" then @teacher_dim = val.to_i
              else                      @val_batches = val.to_i
              end
            when /\A--(classes|hidden|features|layers|teacher-dim|val-batches)=(.*)\z/
              key = $1
              val = $2
              min = key == "classes" ? 2 : 1
              return bad_arg("--#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "classes"     then @classes     = val.to_i
              when "hidden"      then @hidden      = val.to_i
              when "features"    then @features    = val.to_i
              when "layers"      then @mlp_layers  = val.to_i
              when "teacher-dim" then @teacher_dim = val.to_i
              else                    @val_batches = val.to_i
              end
            when "--fields", "--cardinality", "--numeric", "--emb", "--pairs"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              min = key == "--cardinality" ? 2 : 0
              return bad_arg("#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "--fields"      then @fields      = val.to_i
              when "--cardinality" then @cardinality = val.to_i
              when "--numeric"     then @numeric     = val.to_i
              when "--emb"         then @emb         = val.to_i
              else                      @pairs       = val.to_i
              end
            when /\A--(fields|cardinality|numeric|emb|pairs)=(.*)\z/
              key = $1
              val = $2
              min = key == "cardinality" ? 2 : 0
              return bad_arg("--#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "fields"      then @fields      = val.to_i
              when "cardinality" then @cardinality = val.to_i
              when "numeric"     then @numeric     = val.to_i
              when "emb"         then @emb         = val.to_i
              else                    @pairs       = val.to_i
              end
            when "--base-rate", "--lin-scale"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be a float in (0, 1], got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0 && val.to_f <= 1.0
              key == "--base-rate" ? @base_rate = val.to_f : @lin_scale = val.to_f
            when /\A--(base-rate|lin-scale)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--#{key} must be a float in (0, 1], got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0 && val.to_f <= 1.0
              key == "base-rate" ? @base_rate = val.to_f : @lin_scale = val.to_f
            when "--fm-branch"
              @fm_branch = true
            # ---- toy#153 (gnn): the graph lane's own knobs ----
            when "--graph"
              i += 1
              val = @argv[i]
              return bad_arg("--graph requires a value") if val.nil?
              @graph = val
            when /\A--graph=(.*)\z/m
              @graph = $1
            when "--nodes", "--degree", "--train-per-class", "--feedback-hops"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              # --degree 0 is legal and meaningful: an EDGELESS graph
              # makes the propagation the identity, so the lane
              # degenerates to a plain MLP — the control that says
              # whether the graph is load-bearing at all.
              min = key == "--degree" ? 0 : 1
              return bad_arg("#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              return bad_arg("--feedback-hops must be in 1..8, got #{val.inspect}") if key == "--feedback-hops" && val.to_i > 8
              case key
              when "--nodes"           then @nodes           = val.to_i
              when "--degree"          then @degree          = val.to_i
              when "--train-per-class" then @train_per_class = val.to_i
              else                          @feedback_hops   = val.to_i
              end
            when /\A--(nodes|degree|train-per-class|feedback-hops)=(.*)\z/
              key = $1
              val = $2
              min = key == "degree" ? 0 : 1
              return bad_arg("--#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              return bad_arg("--feedback-hops must be in 1..8, got #{val.inspect}") if key == "feedback-hops" && val.to_i > 8
              case key
              when "nodes"           then @nodes           = val.to_i
              when "degree"          then @degree          = val.to_i
              when "train-per-class" then @train_per_class = val.to_i
              else                        @feedback_hops   = val.to_i
              end
            when "--homophily", "--feat-signal", "--weight-decay"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be a non-negative float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/
              return bad_arg("--homophily must be in [0, 1], got #{val.inspect}") if key == "--homophily" && val.to_f > 1.0
              case key
              when "--homophily"    then @homophily    = val.to_f
              when "--feat-signal"  then @feat_signal  = val.to_f
              else                       @weight_decay = val.to_f
              end
            when /\A--(homophily|feat-signal|weight-decay)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--#{key} must be a non-negative float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/
              return bad_arg("--homophily must be in [0, 1], got #{val.inspect}") if key == "homophily" && val.to_f > 1.0
              case key
              when "homophily"    then @homophily    = val.to_f
              when "feat-signal"  then @feat_signal  = val.to_f
              else                     @weight_decay = val.to_f
              end
            # ---- toy#156 (diff): the denoiser's own knobs ----
            when "--latent", "--time-feat", "--modes", "--diff-steps", "--eval-n"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              min = key == "--diff-steps" ? 2 : 1
              return bad_arg("#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "--latent"      then @latent     = val.to_i
              when "--time-feat"   then @time_feat  = val.to_i
              when "--modes"       then @modes      = val.to_i
              when "--diff-steps"  then @diff_steps = val.to_i
              else                      @eval_n     = val.to_i
              end
            when /\A--(latent|time-feat|modes|diff-steps|eval-n)=(.*)\z/
              key = $1
              val = $2
              min = key == "diff-steps" ? 2 : 1
              return bad_arg("--#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "latent"      then @latent     = val.to_i
              when "time-feat"   then @time_feat  = val.to_i
              when "modes"       then @modes      = val.to_i
              when "diff-steps"  then @diff_steps = val.to_i
              else                    @eval_n     = val.to_i
              end
            when "--spread", "--mode-scale", "--beta-lo", "--beta-hi"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0
              case key
              when "--spread"     then @spread     = val.to_f
              when "--mode-scale" then @mode_scale = val.to_f
              when "--beta-lo"    then @beta_lo    = val.to_f
              else                     @beta_hi    = val.to_f
              end
            when /\A--(spread|mode-scale|beta-lo|beta-hi)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--#{key} must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0
              case key
              when "spread"     then @spread     = val.to_f
              when "mode-scale" then @mode_scale = val.to_f
              when "beta-lo"    then @beta_lo    = val.to_f
              else                   @beta_hi    = val.to_f
              end
            # ---- toy#160 (gtx): the graph transformer's own knobs ----
            # --entities is NOT --nodes and the pairs-per-step count is
            # NOT --pairs: `--nodes` is the gnn lane's graph size and
            # `--pairs` is the ctr lane's feature crosses. A different
            # meaning gets a different name (the tao#18 --policy-scope
            # discipline); pairs-per-step is just --batch, which already
            # means "samples per step" on every lane that has one.
            when "--heads", "--d-ff", "--types", "--entities"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              min = key == "--types" ? 2 : 1
              return bad_arg("#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "--heads"    then @heads    = val.to_i
              when "--d-ff"     then @d_ff     = val.to_i
              when "--types"    then @types    = val.to_i
              else                   @entities = val.to_i
              end
            when /\A--(heads|d-ff|types|entities)=(.*)\z/
              key = $1
              val = $2
              min = key == "types" ? 2 : 1
              return bad_arg("--#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "heads"    then @heads    = val.to_i
              when "d-ff"     then @d_ff     = val.to_i
              when "types"    then @types    = val.to_i
              else                 @entities = val.to_i
              end
            # ---- toy#165 (ae): the latent-autoencoder lane's own knobs ----
            when "--text"
              i += 1
              val = @argv[i]
              return bad_arg("--text requires a value") if val.nil?
              @text = val
            when /\A--text=(.*)\z/
              @text = $1
            when "--arm"
              i += 1
              val = @argv[i]
              return bad_arg("--arm requires a value") if val.nil?
              unless %w[ar-baseline diff-selfcond diff-plain prior-floor].include?(val)
                return bad_arg("--arm #{val.inspect} unsupported (ar-baseline|diff-selfcond|diff-plain|prior-floor)")
              end
              @arm = val
            when /\A--arm=(.*)\z/
              val = $1
              unless %w[ar-baseline diff-selfcond diff-plain prior-floor].include?(val)
                return bad_arg("--arm #{val.inspect} unsupported (ar-baseline|diff-selfcond|diff-plain|prior-floor)")
              end
              @arm = val
            when "--loss-weight"
              i += 1
              val = @argv[i]
              return bad_arg("--loss-weight requires a value") if val.nil?
              unless %w[eps-uniform min-snr-gamma v-param nonuniform-t].include?(val)
                return bad_arg("--loss-weight #{val.inspect} unsupported (eps-uniform|min-snr-gamma|v-param|nonuniform-t)")
              end
              @loss_weight = val
            when /\A--loss-weight=(.*)\z/
              val = $1
              unless %w[eps-uniform min-snr-gamma v-param nonuniform-t].include?(val)
                return bad_arg("--loss-weight #{val.inspect} unsupported (eps-uniform|min-snr-gamma|v-param|nonuniform-t)")
              end
              @loss_weight = val
            when "--minsnr-gamma"
              i += 1
              val = @argv[i]
              return bad_arg("--minsnr-gamma requires a value") if val.nil?
              return bad_arg("--minsnr-gamma must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0
              @minsnr_gamma = val
            when /\A--minsnr-gamma=(.*)\z/
              val = $1
              return bad_arg("--minsnr-gamma must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0
              @minsnr_gamma = val
            when "--ae-steps", "--tsteps", "--gen-bytes", "--judge-steps"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              case key
              when "--ae-steps"    then @ae_steps    = val.to_i
              when "--tsteps"      then @tsteps      = val.to_i
              when "--gen-bytes"   then @gen_bytes   = val.to_i
              else                      @judge_steps = val.to_i
              end
            when /\A--(ae-steps|tsteps|gen-bytes|judge-steps)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--#{key} must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              case key
              when "ae-steps"    then @ae_steps    = val.to_i
              when "tsteps"      then @tsteps      = val.to_i
              when "gen-bytes"   then @gen_bytes   = val.to_i
              else                    @judge_steps = val.to_i
              end
            when "--target-ce"
              i += 1
              val = @argv[i]
              return bad_arg("--target-ce requires a value") if val.nil?
              return bad_arg("--target-ce must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0.0
              @target_ce = val
            when /\A--target-ce=(.*)\z/
              val = $1
              return bad_arg("--target-ce must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0.0
              @target_ce = val
            when "--eval-every", "--probe-batches"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              if key == "--eval-every" then @eval_every = val.to_i else @probe_batches = val.to_i end
            when /\A--(eval-every|probe-batches)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--#{key} must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              if key == "eval-every" then @eval_every = val.to_i else @probe_batches = val.to_i end
            when "--noise-seed", "--val-frac-pct"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              case key
              when "--noise-seed" then @noise_seed   = val.to_i
              else                     @val_frac_pct = val.to_i
              end
            when /\A--(noise-seed|val-frac-pct)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--#{key} must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              case key
              when "noise-seed" then @noise_seed   = val.to_i
              else                   @val_frac_pct = val.to_i
              end
            when "--noise-eval"
              i += 1
              val = @argv[i]
              return bad_arg("--noise-eval requires a value") if val.nil?
              return bad_arg("--noise-eval must be a comma list of non-negative floats, got #{val.inspect}") unless val =~ /\A\d*\.?\d+(,\d*\.?\d+)*\z/
              @noise_eval = val
            when /\A--noise-eval=(.*)\z/
              val = $1
              return bad_arg("--noise-eval must be a comma list of non-negative floats, got #{val.inspect}") unless val =~ /\A\d*\.?\d+(,\d*\.?\d+)*\z/
              @noise_eval = val
            # ---- toy#155 (ssm): the sequence lane's own knobs ----
            when "--seq", "--d-model", "--d-inner", "--conv-k", "--cue-span"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              min = key == "--d-model" ? 2 : 1
              return bad_arg("#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "--seq"      then @seq      = val.to_i
              when "--d-model"  then @d_model  = val.to_i
              when "--d-inner"  then @d_inner  = val.to_i
              when "--conv-k"   then @conv_k   = val.to_i
              else                   @cue_span = val.to_i
              end
            when /\A--(seq|d-model|d-inner|conv-k|cue-span)=(.*)\z/
              key = $1
              val = $2
              min = key == "d-model" ? 2 : 1
              return bad_arg("--#{key} must be an integer >= #{min}, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= min
              case key
              when "seq"      then @seq      = val.to_i
              when "d-model"  then @d_model  = val.to_i
              when "d-inner"  then @d_inner  = val.to_i
              when "conv-k"   then @conv_k   = val.to_i
              else                 @cue_span = val.to_i
              end
            # toy#162: a POSITIVE float. There is no "--clip-grad 0" —
            # omitting the flag is how you turn it off, so a 0 would be a
            # setting that silently means "unset".
            when "--retrofit"
              @retrofit = true
            when "--no-freeze-backbone"
              @no_freeze_backbone = true
            when "--adapter-policy"
              i += 1
              val = @argv[i]
              return bad_arg("--adapter-policy requires a value") if val.nil?
              return bad_arg("--adapter-policy must be chain, dfa or frozen, got #{val.inspect}") unless %w[chain dfa frozen].include?(val)
              @adapter_policy = val
            when /\A--adapter-policy=(.*)\z/
              val = $1
              return bad_arg("--adapter-policy must be chain, dfa or frozen, got #{val.inspect}") unless %w[chain dfa frozen].include?(val)
              @adapter_policy = val
            when "--pretrain-steps", "--adapter-layers", "--adapter-rank"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be an integer >= 1, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 1
              case key
              when "--pretrain-steps" then @pretrain_steps = val.to_i
              when "--adapter-layers" then @adapter_layers = val.to_i
              else                         @adapter_rank   = val.to_i
              end
            when /\A--(pretrain-steps|adapter-layers|adapter-rank)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--#{key} must be an integer >= 1, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 1
              case key
              when "pretrain-steps" then @pretrain_steps = val.to_i
              when "adapter-layers" then @adapter_layers = val.to_i
              else                       @adapter_rank   = val.to_i
              end
            when "--pretrain-lr"
              i += 1
              val = @argv[i]
              return bad_arg("--pretrain-lr requires a value") if val.nil?
              return bad_arg("--pretrain-lr must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0.0
              @pretrain_lr = val.to_f
            when /\A--pretrain-lr=(.*)\z/
              val = $1
              return bad_arg("--pretrain-lr must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0.0
              @pretrain_lr = val.to_f
            when "--clip-grad"
              i += 1
              val = @argv[i]
              return bad_arg("--clip-grad requires a value") if val.nil?
              return bad_arg("--clip-grad must be a positive float (omit it to disable clipping), got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0.0
              @clip_grad = val.to_f
            when /\A--clip-grad=(.*)\z/
              val = $1
              return bad_arg("--clip-grad must be a positive float (omit it to disable clipping), got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0.0
              @clip_grad = val.to_f
            when "--noise"
              i += 1
              val = @argv[i]
              return bad_arg("--noise requires a value") if val.nil?
              return bad_arg("--noise must be a non-negative float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/
              @noise = val.to_f
            when /\A--noise=(.*)\z/
              val = $1
              return bad_arg("--noise must be a non-negative float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/
              @noise = val.to_f
            # --dt-init is NEGATIVE by default (-5.0): it is a pre-softplus
            # logit, and a value near 0 gives a one-step half-life that no
            # arm can learn a delayed-cue task from. So this one flag needs
            # a SIGNED parser — the other float flags deliberately do not.
            when "--dt-init"
              i += 1
              val = @argv[i]
              return bad_arg("--dt-init requires a value") if val.nil?
              return bad_arg("--dt-init must be a float, got #{val.inspect}") unless val =~ /\A-?\d*\.?\d+\z/
              @dt_init = val.to_f
            when /\A--dt-init=(.*)\z/
              val = $1
              return bad_arg("--dt-init must be a float, got #{val.inspect}") unless val =~ /\A-?\d*\.?\d+\z/
              @dt_init = val.to_f
            when "--selection"
              i += 1
              val = @argv[i]
              return bad_arg("--selection requires a value") if val.nil?
              return bad_arg("--selection must be selective or lti, got #{val.inspect}") unless %w[selective lti].include?(val)
              @selection = val
            when /\A--selection=(.*)\z/
              val = $1
              return bad_arg("--selection must be selective or lti, got #{val.inspect}") unless %w[selective lti].include?(val)
              @selection = val
            when "--dfa-cut"
              i += 1
              val = @argv[i]
              return bad_arg("--dfa-cut requires a value") if val.nil?
              return bad_arg("--dfa-cut must be layer or step, got #{val.inspect}") unless %w[layer step].include?(val)
              @dfa_cut = val
            when /\A--dfa-cut=(.*)\z/
              val = $1
              return bad_arg("--dfa-cut must be layer or step, got #{val.inspect}") unless %w[layer step].include?(val)
              @dfa_cut = val
            when "--feedback-route"
              i += 1
              val = @argv[i]
              return bad_arg("--feedback-route requires a value") if val.nil?
              return bad_arg("--feedback-route must be direct or structure, got #{val.inspect}") unless %w[direct structure].include?(val)
              @feedback_route = val
            when /\A--feedback-route=(.*)\z/
              val = $1
              return bad_arg("--feedback-route must be direct or structure, got #{val.inspect}") unless %w[direct structure].include?(val)
              @feedback_route = val
            when "--task"
              i += 1
              val = @argv[i]
              return bad_arg("--task requires a value") if val.nil?
              return bad_arg("--task must be teacher, blobs, community, cue, mean, relational, local, bytelm, mixture or single, got #{val.inspect}") unless %w[teacher blobs community cue mean relational local bytelm mixture single].include?(val)
              @task = val
            when /\A--task=(.*)\z/
              val = $1
              return bad_arg("--task must be teacher, blobs, community, cue, mean, relational, local, bytelm, mixture or single, got #{val.inspect}") unless %w[teacher blobs community cue mean relational local bytelm mixture single].include?(val)
              @task = val
            when "--task-seed"
              i += 1
              val = @argv[i]
              return bad_arg("--task-seed requires a value") if val.nil?
              return bad_arg("--task-seed must be a non-negative integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @task_seed = val.to_i
            when /\A--task-seed=(.*)\z/
              val = $1
              return bad_arg("--task-seed must be a non-negative integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @task_seed = val.to_i
            when "--policy-scope"
              i += 1
              val = @argv[i]
              return bad_arg("--policy-scope requires a value") if val.nil?
              return bad_arg("--policy-scope must be attn, ffn or all, got #{val.inspect}") unless %w[attn ffn all].include?(val)
              @policy_scope = val
            when /\A--policy-scope=(.*)\z/
              val = $1
              return bad_arg("--policy-scope must be attn, ffn or all, got #{val.inspect}") unless %w[attn ffn all].include?(val)
              @policy_scope = val
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
            # ---- toy#172 / E1 Phase 1.2 (gtx --task bytelm): nDFA ----
            # The preconditioner is named on the FEEDBACK, not on the
            # optimizer, because that is what it changes: it folds
            # lambda(C_E + lambda I)^-1 into B and touches nothing else.
            # `none` is the default and is byte-identical to the runner
            # before the flag existed.
            when "--dfa-feedback-precond"
              i += 1
              val = @argv[i]
              return bad_arg("--dfa-feedback-precond requires a value") if val.nil?
              return bad_arg("--dfa-feedback-precond must be none or ndfa, got #{val.inspect}") unless %w[none ndfa].include?(val)
              @dfa_feedback_precond = val
            when /\A--dfa-feedback-precond=(.*)\z/
              val = $1
              return bad_arg("--dfa-feedback-precond must be none or ndfa, got #{val.inspect}") unless %w[none ndfa].include?(val)
              @dfa_feedback_precond = val
            # The ridge is kept as a STRING all the way to the env. It is
            # experiment identity on this lane, and `1e30`.to_f.to_s is
            # "1.0e+30" — a different cell label for the same cell, which
            # is exactly how a sweep's files stop matching its greps.
            when "--ndfa-lambda"
              i += 1
              val = @argv[i]
              return bad_arg("--ndfa-lambda requires a value") if val.nil?
              return bad_arg("--ndfa-lambda must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0.0
              @ndfa_lambda = val
            when /\A--ndfa-lambda=(.*)\z/
              val = $1
              return bad_arg("--ndfa-lambda must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0.0
              @ndfa_lambda = val
            when "--ndfa-every", "--ndfa-samples"
              key = tok
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be an integer >= 1, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 1
              if key == "--ndfa-every"
                @ndfa_every = val.to_i
              else
                @ndfa_samples = val.to_i
              end
            when /\A--ndfa-(every|samples)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--ndfa-#{key} must be an integer >= 1, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 1
              if key == "every"
                @ndfa_every = val.to_i
              else
                @ndfa_samples = val.to_i
              end
            when "--ndfa-gain"
              i += 1
              val = @argv[i]
              return bad_arg("--ndfa-gain requires a value") if val.nil?
              return bad_arg("--ndfa-gain must be preserve or raw, got #{val.inspect}") unless %w[preserve raw].include?(val)
              @ndfa_gain = val
            when /\A--ndfa-gain=(.*)\z/
              val = $1
              return bad_arg("--ndfa-gain must be preserve or raw, got #{val.inspect}") unless %w[preserve raw].include?(val)
              @ndfa_gain = val
            when "--align-events"
              @align_events = true
            when "--no-shadow"
              @no_shadow = true
            when "--attn-gate"
              @attn_gate = true
            when "--no-kda-conv"
              @kda_conv_off = true
            when "--attnres"
              @attnres = true
            when "--mtp"
              @mtp = true
            # K-series M10: lambda is a COUPLING dial, not a loss weight.
            # 0 = the second root exists and its weights train, but it
            # does not perturb the backbone at all (byte-identical to
            # --mtp off); 1 = full coupling. Values > 1 would amplify
            # MTP's pull on the backbone past the main loss's own, which
            # is not what the dial means — rejected rather than quietly
            # allowed.
            when "--mtp-lambda"
              i += 1
              val = @argv[i]
              return bad_arg("--mtp-lambda requires a value") if val.nil?
              return bad_arg("--mtp-lambda must be a float in [0, 1], got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f >= 0.0 && val.to_f <= 1.0
              @mtp_lambda = val.to_f
            when /\A--mtp-lambda=(.*)\z/
              val = $1
              return bad_arg("--mtp-lambda must be a float in [0, 1], got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f >= 0.0 && val.to_f <= 1.0
              @mtp_lambda = val.to_f
            when "--donor"
              i += 1
              val = @argv[i]
              return bad_arg("--donor requires a value") if val.nil?
              @donor = val
            when /\A--donor=(.*)\z/m
              @donor = $1
            when "--donor-mode"
              i += 1
              val = @argv[i]
              return bad_arg("--donor-mode requires a value") if val.nil?
              return bad_arg("--donor-mode must be tied or untied, got #{val.inspect}") unless %w[tied untied].include?(val)
              @donor_mode = val
            when /\A--donor-mode=(.*)\z/
              val = $1
              return bad_arg("--donor-mode must be tied or untied, got #{val.inspect}") unless %w[tied untied].include?(val)
              @donor_mode = val
            when "--freeze-embed"
              @freeze_embed = true
            when "--freeze-experts"
              @freeze_experts = true
            when "--moe-latent"
              @moe_latent = true
            when "--lr-schedule"
              i += 1
              val = @argv[i]
              return bad_arg("--lr-schedule requires a value") if val.nil?
              return bad_arg("--lr-schedule must be uniform, ramp-up or ramp-down, got #{val.inspect}") unless %w[uniform ramp-up ramp-down].include?(val)
              @lr_schedule = val
            when /\A--lr-schedule=(.*)\z/
              val = $1
              return bad_arg("--lr-schedule must be uniform, ramp-up or ramp-down, got #{val.inspect}") unless %w[uniform ramp-up ramp-down].include?(val)
              @lr_schedule = val
            when "--lr-lo", "--lr-hi"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0
              key == "--lr-lo" ? @lr_lo = val.to_f : @lr_hi = val.to_f
            when /\A--lr-(lo|hi)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--lr-#{key} must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0
              key == "lo" ? @lr_lo = val.to_f : @lr_hi = val.to_f
# toy#150: adaptive feedback. lambda/eta are meaningless
# without the coupling, so they are rejected without it.
when "--dfa-feedback"
  i += 1
  val = @argv[i]
  return bad_arg("--dfa-feedback requires a value") if val.nil?
  return bad_arg("--dfa-feedback must be fixed or kolen-pollack, got #{val.inspect}") unless %w[fixed kolen-pollack].include?(val)
  @dfa_feedback = val
when /\A--dfa-feedback=(.*)\z/
  val = $1
  return bad_arg("--dfa-feedback must be fixed or kolen-pollack, got #{val.inspect}") unless %w[fixed kolen-pollack].include?(val)
  @dfa_feedback = val
when "--dfa-feedback-decay"
  i += 1
  val = @argv[i]
  return bad_arg("--dfa-feedback-decay requires a value") if val.nil?
  return bad_arg("--dfa-feedback-decay must be a float >= 0, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][-+]?\d+)?\z/ && val.to_f >= 0.0
  @dfa_feedback_decay = val.to_f
when /\A--dfa-feedback-decay=(.*)\z/
  val = $1
  return bad_arg("--dfa-feedback-decay must be a float >= 0, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][-+]?\d+)?\z/ && val.to_f >= 0.0
  @dfa_feedback_decay = val.to_f
when "--dfa-feedback-lr"
  i += 1
  val = @argv[i]
  return bad_arg("--dfa-feedback-lr requires a value") if val.nil?
  return bad_arg("--dfa-feedback-lr must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][-+]?\d+)?\z/ && val.to_f > 0.0
  @dfa_feedback_lr = val.to_f
when /\A--dfa-feedback-lr=(.*)\z/
  val = $1
  return bad_arg("--dfa-feedback-lr must be a positive float, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][-+]?\d+)?\z/ && val.to_f > 0.0
  @dfa_feedback_lr = val.to_f
            when "--lr-control"
              i += 1
              val = @argv[i]
              return bad_arg("--lr-control requires a value") if val.nil?
              return bad_arg("--lr-control must be none or reactive, got #{val.inspect}") unless %w[none reactive].include?(val)
              @lr_control = val
            when /\A--lr-control=(.*)\z/
              val = $1
              return bad_arg("--lr-control must be none or reactive, got #{val.inspect}") unless %w[none reactive].include?(val)
              @lr_control = val
            # toy#148 controller constants. Ranges are checked HERE as
            # well as in the runner: the CLI is the surface most people
            # touch, and a factor of 2.0 ("double the LR") silently doing
            # the opposite of a damper is exactly the class of quiet
            # wrongness the gates exist to prevent.
            when "--lr-control-window", "--lr-control-patience"
              key = @argv[i]
              i += 1
              val = @argv[i]
              return bad_arg("#{key} requires a value") if val.nil?
              return bad_arg("#{key} must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 1
              key == "--lr-control-window" ? @lr_control_window = val.to_i : @lr_control_patience = val.to_i
            when /\A--lr-control-(window|patience)=(.*)\z/
              key = $1
              val = $2
              return bad_arg("--lr-control-#{key} must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 1
              key == "window" ? @lr_control_window = val.to_i : @lr_control_patience = val.to_i
            when "--lr-control-factor"
              i += 1
              val = @argv[i]
              return bad_arg("--lr-control-factor requires a value") if val.nil?
              return bad_arg("--lr-control-factor must be a float in (0, 1) — it is a CUT, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0 && val.to_f < 1.0
              @lr_control_factor = val.to_f
            when /\A--lr-control-factor=(.*)\z/
              val = $1
              return bad_arg("--lr-control-factor must be a float in (0, 1) — it is a CUT, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0 && val.to_f < 1.0
              @lr_control_factor = val.to_f
            when "--lr-control-recover"
              i += 1
              val = @argv[i]
              return bad_arg("--lr-control-recover requires a value") if val.nil?
              return bad_arg("--lr-control-recover must be a float >= 1.0 — it restores toward 1.0, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f >= 1.0
              @lr_control_recover = val.to_f
            when /\A--lr-control-recover=(.*)\z/
              val = $1
              return bad_arg("--lr-control-recover must be a float >= 1.0 — it restores toward 1.0, got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f >= 1.0
              @lr_control_recover = val.to_f
            when "--lr-control-floor"
              i += 1
              val = @argv[i]
              return bad_arg("--lr-control-floor requires a value") if val.nil?
              return bad_arg("--lr-control-floor must be a float in (0, 1], got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0 && val.to_f <= 1.0
              @lr_control_floor = val.to_f
            when /\A--lr-control-floor=(.*)\z/
              val = $1
              return bad_arg("--lr-control-floor must be a float in (0, 1], got #{val.inspect}") unless val =~ /\A\d*\.?\d+\z/ && val.to_f > 0.0 && val.to_f <= 1.0
              @lr_control_floor = val.to_f
            when "--expert-act"
              i += 1
              val = @argv[i]
              return bad_arg("--expert-act requires a value") if val.nil?
              return bad_arg("--expert-act must be gelu or situ-glu, got #{val.inspect}") unless %w[gelu situ-glu].include?(val)
              @expert_act = val
            when /\A--expert-act=(.*)\z/
              val = $1
              return bad_arg("--expert-act must be gelu or situ-glu, got #{val.inspect}") unless %w[gelu situ-glu].include?(val)
              @expert_act = val
            when "--dfa-granularity"
              i += 1
              val = @argv[i]
              return bad_arg("--dfa-granularity requires a value") if val.nil?
              return bad_arg("--dfa-granularity must be matmul or block, got #{val.inspect}") unless %w[matmul block].include?(val)
              @dfa_granularity = val
            when /\A--dfa-granularity=(.*)\z/
              val = $1
              return bad_arg("--dfa-granularity must be matmul or block, got #{val.inspect}") unless %w[matmul block].include?(val)
              @dfa_granularity = val
            when "--moe-shared"
              i += 1
              val = @argv[i]
              return bad_arg("--moe-shared requires a value") if val.nil?
              return bad_arg("--moe-shared must be a non-negative integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @moe_shared = val.to_i
            when /\A--moe-shared=(.*)\z/
              val = $1
              return bad_arg("--moe-shared must be a non-negative integer, got #{val.inspect}") unless val =~ /\A\d+\z/
              @moe_shared = val.to_i
            when "--optimizer"
              i += 1
              val = @argv[i]
              return bad_arg("--optimizer requires a value") if val.nil?
              return bad_arg("--optimizer must be adamw, muon, radam, or sgd, got #{val.inspect}") unless %w[adamw muon radam sgd].include?(val)
              @optimizer = val
            when /\A--optimizer=(.*)\z/
              val = $1
              return bad_arg("--optimizer must be adamw, muon, radam, or sgd, got #{val.inspect}") unless %w[adamw muon radam sgd].include?(val)
              @optimizer = val
            when "--layer-pattern"
              i += 1
              val = @argv[i]
              return bad_arg("--layer-pattern requires a value") if val.nil?
              return bad_arg("--layer-pattern must be hybrid or k3, got #{val.inspect}") unless %w[hybrid k3].include?(val)
              @layer_pattern = val
            when /\A--layer-pattern=(.*)\z/
              val = $1
              return bad_arg("--layer-pattern must be hybrid or k3, got #{val.inspect}") unless %w[hybrid k3].include?(val)
              @layer_pattern = val
            when "--kda-layers"
              i += 1
              val = @argv[i]
              return bad_arg("--kda-layers requires a value") if val.nil?
              return bad_arg("--kda-layers must be comma-separated 0-based integers, got #{val.inspect}") unless val =~ /\A\d+(,\d+)*\z/
              @kda_layers = val
            when /\A--kda-layers=(.*)\z/
              val = $1
              return bad_arg("--kda-layers must be comma-separated 0-based integers, got #{val.inspect}") unless val =~ /\A\d+(,\d+)*\z/
              @kda_layers = val
            when "--mla-layers"
              i += 1
              val = @argv[i]
              return bad_arg("--mla-layers requires a value") if val.nil?
              return bad_arg("--mla-layers must be comma-separated 0-based integers, got #{val.inspect}") unless val =~ /\A\d+(,\d+)*\z/
              @mla_layers = val
            when /\A--mla-layers=(.*)\z/
              val = $1
              return bad_arg("--mla-layers must be comma-separated 0-based integers, got #{val.inspect}") unless val =~ /\A\d+(,\d+)*\z/
              @mla_layers = val
            when "--mla-rank"
              i += 1
              val = @argv[i]
              return bad_arg("--mla-rank requires a value") if val.nil?
              return bad_arg("--mla-rank must be a positive integer, got #{val.inspect}") unless val =~ /\A[1-9]\d*\z/
              @mla_rank = val.to_i
            when /\A--mla-rank=(.*)\z/
              val = $1
              return bad_arg("--mla-rank must be a positive integer, got #{val.inspect}") unless val =~ /\A[1-9]\d*\z/
              @mla_rank = val.to_i
            when "--act"
              i += 1
              val = @argv[i]
              return bad_arg("--act requires a value") if val.nil?
              return bad_arg("--act must be swiglu or situ-glu, got #{val.inspect}") unless %w[swiglu situ-glu].include?(val)
              @act = val
            when /\A--act=(.*)\z/
              val = $1
              return bad_arg("--act must be swiglu or situ-glu, got #{val.inspect}") unless %w[swiglu situ-glu].include?(val)
              @act = val
            when "--rope"
              i += 1
              val = @argv[i]
              return bad_arg("--rope requires a value") if val.nil?
              return bad_arg("--rope must be rope or nope, got #{val.inspect}") unless %w[rope nope].include?(val)
              @rope = val
            when /\A--rope=(.*)\z/
              val = $1
              return bad_arg("--rope must be rope or nope, got #{val.inspect}") unless %w[rope nope].include?(val)
              @rope = val
            when "--schedule"
              i += 1
              val = @argv[i]
              return bad_arg("--schedule requires a value") if val.nil?
              return bad_arg("--schedule must be const or cosine, got #{val.inspect}") unless %w[const cosine].include?(val)
              @schedule = val
            when /\A--schedule=(.*)\z/
              val = $1
              return bad_arg("--schedule must be const or cosine, got #{val.inspect}") unless %w[const cosine].include?(val)
              @schedule = val
            when "--moe-balance"
              i += 1
              val = @argv[i]
              return bad_arg("--moe-balance requires a value") if val.nil?
              return bad_arg("--moe-balance must be aux, qb, or none, got #{val.inspect}") unless %w[aux qb none].include?(val)
              @moe_balance = val
            when /\A--moe-balance=(.*)\z/
              val = $1
              return bad_arg("--moe-balance must be aux, qb, or none, got #{val.inspect}") unless %w[aux qb none].include?(val)
              @moe_balance = val
            when "--ckpt-every"
              i += 1
              val = @argv[i]
              return bad_arg("--ckpt-every requires a value") if val.nil?
              return bad_arg("--ckpt-every must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @ckpt_every = val.to_i
            when "--load-ckpt"
              i += 1
              val = @argv[i]
              return bad_arg("--load-ckpt requires a value") if val.nil?
              @load_ckpt = val
            when /\A--load-ckpt=(.*)\z/m
              @load_ckpt = $1
            when "--eval-corpus"
              i += 1
              val = @argv[i]
              return bad_arg("--eval-corpus requires a value") if val.nil?
              @eval_corpus = val
            when /\A--eval-corpus=(.*)\z/m
              @eval_corpus = $1
            when "--eval-tokens"
              i += 1
              val = @argv[i]
              return bad_arg("--eval-tokens requires a value") if val.nil?
              return bad_arg("--eval-tokens must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @eval_tokens = val.to_i
            when /\A--eval-tokens=(.*)\z/
              val = $1
              return bad_arg("--eval-tokens must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @eval_tokens = val.to_i
            when "--eval-offset"
              i += 1
              val = @argv[i]
              return bad_arg("--eval-offset requires a value") if val.nil?
              return bad_arg("--eval-offset must be a non-negative integer (tokens), got #{val.inspect}") unless val =~ /\A\d+\z/
              @eval_offset = val.to_i
            when /\A--eval-offset=(.*)\z/
              val = $1
              return bad_arg("--eval-offset must be a non-negative integer (tokens), got #{val.inspect}") unless val =~ /\A\d+\z/
              @eval_offset = val.to_i
            when /\A--ckpt-every=(.*)\z/
              val = $1
              return bad_arg("--ckpt-every must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @ckpt_every = val.to_i
            when "--shape"
              i += 1
              val = @argv[i]
              return bad_arg("--shape must be base|wide|deep") unless %w[base wide deep].include?(val)
              @shape = val
            when /\A--shape=(.*)\z/
              val = $1
              return bad_arg("--shape must be base|wide|deep") unless %w[base wide deep].include?(val)
              @shape = val
            when "--align-every"
              i += 1
              val = @argv[i]
              return bad_arg("--align-every must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @align_every = val.to_i
            when /\A--align-every=(.*)\z/
              val = $1
              return bad_arg("--align-every must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @align_every = val.to_i
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
              return bad_arg("--moe-policy must be chain|dfa-experts|bp-router-dfa-experts|bp-spine") unless %w[chain dfa-experts bp-router-dfa-experts bp-spine].include?(val)
              @moe_policy = val
            when /\A--moe-policy=(.*)\z/
              val = $1
              return bad_arg("--moe-policy must be chain|dfa-experts|bp-router-dfa-experts|bp-spine") unless %w[chain dfa-experts bp-router-dfa-experts bp-spine].include?(val)
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
            when "--lr"
              i += 1
              val = @argv[i]
              return bad_arg("--lr requires a value") if val.nil?
              return bad_arg("--lr must be a positive number, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0
              @lr = val
            when /\A--lr=(.*)\z/
              val = $1
              return bad_arg("--lr must be a positive number, got #{val.inspect}") unless val =~ /\A\d*\.?\d+([eE][+-]?\d+)?\z/ && val.to_f > 0
              @lr = val
            when "--warmup"
              i += 1
              val = @argv[i]
              return bad_arg("--warmup requires a value") if val.nil?
              return bad_arg("--warmup must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @warmup = val.to_i
            when /\A--warmup=(.*)\z/
              val = $1
              return bad_arg("--warmup must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @warmup = val.to_i
            when "--experts"
              i += 1
              val = @argv[i]
              return bad_arg("--experts requires a value") if val.nil?
              return bad_arg("--experts must be an integer >= 2, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 2
              @experts = val.to_i
            when "--context"
              i += 1
              val = @argv[i]
              return bad_arg("--context requires a value") if val.nil?
              return bad_arg("--context must be an integer >= 2, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 2
              @context = val.to_i
            when "--batch"
              i += 1
              val = @argv[i]
              return bad_arg("--batch requires a value") if val.nil?
              return bad_arg("--batch must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @batch = val.to_i
            when /\A--batch=(.*)\z/
              val = $1
              return bad_arg("--batch must be a positive integer, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i > 0
              @batch = val.to_i
            when /\A--context=(.*)\z/
              val = $1
              return bad_arg("--context must be an integer >= 2, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 2
              @context = val.to_i
            when "--vocab"
              i += 1
              val = @argv[i]
              return bad_arg("--vocab requires a value") if val.nil?
              return bad_arg("--vocab must be an integer >= 2, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 2
              @vocab = val.to_i
            when /\A--vocab=(.*)\z/
              val = $1
              return bad_arg("--vocab must be an integer >= 2, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 2
              @vocab = val.to_i
            when /\A--experts=(.*)\z/
              val = $1
              return bad_arg("--experts must be an integer >= 2, got #{val.inspect}") unless val =~ /\A\d+\z/ && val.to_i >= 2
              @experts = val.to_i
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
          unless %w[from-scratch lora warm-start vit-tiny franken franken-moe mlp ctr gnn ssm lstm gtx diff ae difflm].include?(@recipe)
            return bad_arg("unknown recipe #{@recipe.inspect}; supported: 'from-scratch', 'lora', 'warm-start', 'vit-tiny', 'franken', 'franken-moe', 'mlp', 'ctr', 'gnn', 'ssm', 'lstm', 'gtx', 'diff', 'ae', 'difflm'")
          end
          # ---- toy#132: the flag x recipe MATRIX ----
          # Four llama-first flags in a row tripped franken-moe at Tao
          # runtime (--corpus toy#125, --align-every toy#127, --ckpt-every
          # toy#131, --lr/--warmup toy#132) because "only valid with"
          # checks were scattered one-per-flag. One table now: recipe
          # parity is a CELL FLIP here (plus the runner-side env wiring),
          # and prep/train_cli_matrix_gate.rb asserts every row rejects
          # under a wrong recipe. CONDITIONAL rules (flag-requires-flag,
          # top1-only, value checks) stay bespoke below — this table
          # covers recipe membership ONLY.
          flag_matrix = [
            ["--model/--rank",  %w[lora],                          (!@model.nil? || !@rank.nil?), ""],
            ["--corpus",        %w[warm-start franken franken-moe], !@corpus.nil?, ""],
            ["--init",          %w[warm-start],                     !@init.nil?, ""],
            ["--fields/--cardinality/--numeric/--emb/--pairs", %w[ctr], (!@fields.nil? || !@cardinality.nil? || !@numeric.nil? || !@emb.nil? || !@pairs.nil?), " (toy#154)"],
            ["--base-rate/--lin-scale/--fm-branch", %w[ctr], (!@base_rate.nil? || !@lin_scale.nil? || @fm_branch), " (toy#154)"],
            ["--dfa-b-*",       %w[franken franken-moe mlp ctr gnn ssm lstm gtx diff], (!@dfa_b_seed.nil? || !@dfa_b_dist.nil? || !@dfa_b_scale.nil?), ""],
            # toy#172/E1 Phase 1.2. gtx ONLY, and inside gtx `--task
            # bytelm` only (checked below): nDFA inverts the ERROR
            # COVARIANCE, and the question is about a WIDE error vector.
            ["--dfa-feedback-precond/--ndfa-*", %w[gtx],
             (!@dfa_feedback_precond.nil? || !@ndfa_lambda.nil? || !@ndfa_every.nil? ||
              !@ndfa_samples.nil? || !@ndfa_gain.nil?), " (toy#172/E1: the nDFA error-side preconditioner folds lambda(C_E+lambda I)^-1 into the gtx byte-LM lane's feedback matrix B)"],
            # --align-events deliberately EXCLUDES ssm AND lstm: on both,
            # the DFA update arrives through autodiff from the surrogate
            # roots, so it lands in the same accumulator a BP run would
            # use and there is no second tensor to take a cosine against.
            # Accepting the flag would emit telemetry that silently means
            # nothing. Those two lanes gate on the B SEED moving the dfa
            # curve instead (toy#158's discipline).
            ["--align-events",  %w[franken franken-moe mlp ctr gnn diff], @align_events, ""],
            ["--policy",        %w[franken mlp ctr gnn ssm lstm gtx diff],   !@policy.nil?, ""],
            # toy#153 (gnn). --feedback-route is DELIBERATELY not
            # --dfa-feedback: franken-moe's --dfa-feedback selects how B
            # is UPDATED (fixed|kolen-pollack), this selects how the
            # error is ROUTED (direct|structure). Different axis, so a
            # different name — the tao#18 --policy-scope discipline.
            ["--graph",         %w[gnn],                            !@graph.nil?, " (toy#153)"],
            ["--nodes/--degree/--homophily/--feat-signal", %w[gnn],
             (!@nodes.nil? || !@degree.nil? || !@homophily.nil? || !@feat_signal.nil?), " (toy#153)"],
            ["--train-per-class", %w[gnn],                          !@train_per_class.nil?, " (toy#153)"],
            ["--feedback-route/--feedback-hops", %w[gnn],           (!@feedback_route.nil? || !@feedback_hops.nil?), " (toy#153)"],
            ["--weight-decay",  %w[gnn],                            !@weight_decay.nil?, " (toy#153)"],
            # toy#155 (ssm). --dfa-cut is NOT --dfa-granularity:
            # franken's granularity picks matmul-vs-block in DEPTH, this
            # picks layer-vs-step in TIME. Different axis, different name.
            # toy#157 (lstm) shares the SEQUENCE knobs with ssm — same
            # task generator, same cut axis, which is what makes the two
            # lanes an architecture comparison. It does NOT share the
            # selective-scan-shaped ones (--d-inner/--conv-k/--selection/
            # --dt-init name parts an LSTM has no counterpart for), so
            # those rows stay ssm-only rather than widening.
            ["--seq",           %w[ssm lstm],                       !@seq.nil?, " (toy#155/#157)"],
            ["--d-model",       %w[ssm gtx ae difflm],                     !@d_model.nil?, " (toy#155/#160/#165)"],
            ["--d-inner/--conv-k", %w[ssm],
             (!@d_inner.nil? || !@conv_k.nil?), " (toy#155)"],
            ["--selection",     %w[ssm],                            !@selection.nil?, " (toy#155)"],
            # toy#160 puts ATTENTION on the same cut axis: `layer` taps
            # the block output with BP intact inside it, `step` cuts the
            # gradient through the attention pattern itself.
            ["--dfa-cut",       %w[ssm lstm gtx],                   !@dfa_cut.nil?, " (toy#155/#157/#160)"],
            ["--heads/--d-ff", %w[gtx ae difflm],
             (!@heads.nil? || !@d_ff.nil?), " (toy#160/#165)"],
            ["--types/--entities", %w[gtx],
             (!@types.nil? || !@entities.nil?), " (toy#160)"],
            ["--arm/--ae-steps/--tsteps/--gen-bytes/--judge-steps/--loss-weight/--minsnr-gamma", %w[difflm],
             (!@arm.nil? || !@ae_steps.nil? || !@tsteps.nil? || !@gen_bytes.nil? || !@judge_steps.nil? ||
              !@loss_weight.nil? || !@minsnr_gamma.nil?),
             " (toy#166)"],
            # toy#169/#170 SPLIT this row. It used to bundle seven flags
            # under one lane list, so widening it for `--text` silently
            # legalised --noise-eval/--probe-batches/--target-ce and the
            # rest on gtx/ssm, which ignore them — caught by
            # gate-train-cli-matrix. `--text` is the only one the byte-LM
            # lanes read, so it gets its own row and the ae/difflm probe
            # knobs keep theirs.
            ["--text", %w[ae difflm gtx ssm], !@text.nil?,
             " (toy#165 names a byte pack from prep/fetch_text.rb, NOT a TOYC corpus; toy#169/#170 add the byte-LM lanes, where it needs --task bytelm)"],
            ["--noise-eval/--noise-seed/--val-frac-pct/--target-ce/--eval-every/--probe-batches", %w[ae difflm],
             (!@noise_eval.nil? || !@noise_seed.nil? || !@val_frac_pct.nil? ||
              !@target_ce.nil? || !@eval_every.nil? || !@probe_batches.nil?),
             " (toy#165; the ae/difflm probe knobs — the byte-LM lanes have no noise probe)"],
            ["--cue-span",      %w[ssm lstm],                       !@cue_span.nil?, " (toy#155/#157)"],
            # toy#162: the FAIR BPTT control lives on the lane whose
            # stability claim needs it. Widening it is a follow-on, not
            # a default.
            ["--retrofit/--adapter-*/--pretrain-*/--no-freeze-backbone", %w[gtx],
             (@retrofit || !@adapter_policy.nil? || !@adapter_layers.nil? ||
              !@adapter_rank.nil? || !@pretrain_steps.nil? || !@pretrain_lr.nil? ||
              @no_freeze_backbone), " (toy#161)"],
            ["--clip-grad",     %w[lstm],                           !@clip_grad.nil?, " (toy#162)"],
            ["--noise",          %w[ssm lstm gtx],                   !@noise.nil?, " (toy#155/#157/#160)"],
            ["--dt-init",       %w[ssm],                            !@dt_init.nil?, " (toy#155)"],
            # toy#156 (diff). --latent is the OUTPUT DIM under test and
            # gets a real name rather than riding --classes: this lane
            # regresses epsilon, it does not classify.
            ["--latent",        %w[diff ae difflm],                        !@latent.nil?, " (toy#156/#165 — the SAME quantity on both lanes)"],
            ["--time-feat",     %w[diff],                           !@time_feat.nil?, " (toy#156)"],
            ["--modes/--spread/--mode-scale", %w[diff],
             (!@modes.nil? || !@spread.nil? || !@mode_scale.nil?), " (toy#156)"],
            ["--diff-steps/--beta-lo/--beta-hi", %w[diff],
             (!@diff_steps.nil? || !@beta_lo.nil? || !@beta_hi.nil?), " (toy#156)"],
            ["--eval-n",        %w[diff],                          !@eval_n.nil?, " (toy#156)"],
            # tao#18 item 1: --policy-scope is DELIBERATELY not accepted
            # on mlp. The attn|ffn|all meaning stays stable across
            # lanes; an MLP has no attention to scope, and a
            # head-vs-hidden split would get a DIFFERENT name
            # (--policy-tensors), never an overload of this one.
            ["--policy-scope",  %w[franken],                        !@policy_scope.nil?, " (toy#151; NOT accepted on 'mlp' — tao#18)"],
            ["--align-every",   %w[franken franken-moe mlp gnn diff], !@align_every.nil?, ""],
            ["--lr/--warmup",   %w[franken franken-moe mlp ctr gnn ssm lstm gtx diff ae difflm], (!@lr.nil? || !@warmup.nil?), " (toy#126/#132)"],
            ["--classes",       %w[mlp gnn ssm lstm],               !@classes.nil?, " (toy#152/#153/#155/#157)"],
            ["--features",      %w[mlp gnn lstm gtx],               !@features.nil?, " (toy#152/#153/#157/#160)"],
            ["--hidden",        %w[mlp ctr gnn lstm diff],          !@hidden.nil?, " (toy#152/#154/#153/#157/#156)"],
            ["--layers",        %w[mlp ctr gnn ssm lstm gtx diff ae difflm], !@mlp_layers.nil?, " (toy#152/#154/#153/#155/#157/#160/#156/#165)"],
            ["--task",          %w[mlp gnn ssm lstm gtx diff],      !@task.nil?, " (toy#152/#153/#155/#157/#160/#156)"],
            ["--teacher-dim",   %w[mlp gnn],                        !@teacher_dim.nil?, " (toy#152/#153)"],
            ["--task-seed",     %w[mlp ctr gnn ssm lstm gtx diff ae difflm], !@task_seed.nil?, " (toy#152/#154/#153/#155/#157/#160/#156/#165)"],
            # --val-batches stays mlp/ctr: the gnn lane is TRANSDUCTIVE,
            # its held-out set is a node mask over the one graph, so
            # "how many val batches" has nothing to size.
            ["--val-batches",   %w[mlp ctr ssm lstm gtx ae],        !@val_batches.nil?, " (toy#152/#154/#155/#157/#160/#165)"],
            ["--no-shadow",     %w[franken franken-moe],            @no_shadow, " (toy#129)"],
            # toy#165 SPLIT this row. `--context` now also means the ae lane's
            # window (which is also its batch); `--vocab` deliberately did NOT
            # come with it — it means an INTEGER pack width on the franken
            # lanes, and letting `ae` take a string there would make one flag
            # mean two different types depending on the recipe.
            ["--context",       %w[franken franken-moe ae difflm], !@context.nil?, " (toy#129/#165/#166)"],
            ["--vocab",         %w[franken franken-moe gtx],       !@vocab.nil?, " (toy#129 names a headerless pack vocab; toy#170/P5 on gtx it is the NOMINAL HEAD WIDTH under --task bytelm. On the ae lane the head is byte-wide by construction, so there is no --vocab)"],
            ["--batch",         %w[franken franken-moe mlp ctr ssm lstm gtx diff], !@batch.nil?, " (toy#133; on gtx it is the labelled PAIRS per step)"],
            ["--act",           %w[franken],                        !@act.nil?, " (toy#136/K1; MoE experts get their GLU in K4)"],
            ["--rope",          %w[franken],                        !@rope.nil?, " (toy#136/K1)"],
            ["--schedule",      %w[franken franken-moe],            !@schedule.nil?, " (toy#136/K1)"],
            ["--moe-balance",   %w[franken-moe],                    !@moe_balance.nil?, " (toy#136/K1)"],
            ["--attn-gate",     %w[franken-moe],                    @attn_gate, " (toy#136/K1.1; the llama lane's gate arrives with KDA in K2)"],
            ["--kda-layers",    %w[franken],                        !@kda_layers.nil?, " (toy#137/K2b)"],
            ["--mla-layers/--mla-rank", %w[franken],                (!@mla_layers.nil? || !@mla_rank.nil?), " (K-series M2)"],
            ["--no-kda-conv",   %w[franken],                        @kda_conv_off, " (toy#137/K2c)"],
            ["--layer-pattern", %w[franken],                        !@layer_pattern.nil?, " (toy#138/K3a)"],
            ["--attnres",      %w[franken],                        @attnres, " (toy#138/K3b)"],
            ["--mtp/--mtp-lambda", %w[franken],                    (@mtp || !@mtp_lambda.nil?), " (K-series M10)"],
            ["--optimizer",    %w[franken franken-moe],            !@optimizer.nil?, " (toy#139/K5)"],
            ["--donor/--donor-mode/--freeze-embed", %w[franken-moe], (!@donor.nil? || !@donor_mode.nil? || @freeze_embed), " (toy#140)"],
            ["--freeze-experts", %w[franken-moe],                    @freeze_experts, " (toy#141)"],
            ["--moe-latent/--moe-shared", %w[franken-moe],           (@moe_latent || !@moe_shared.nil?), " (toy#142/K4)"],
            ["--dfa-granularity", %w[franken franken-moe],           !@dfa_granularity.nil?, " (toy#143 moe / toy#158 dense)"],
            ["--dfa-feedback/--dfa-feedback-*", %w[franken-moe],     (!@dfa_feedback.nil? || !@dfa_feedback_decay.nil? || !@dfa_feedback_lr.nil?), " (toy#150)"],
            ["--expert-act", %w[franken-moe],                        !@expert_act.nil?, " (K4b/M6)"],
            ["--lr-schedule/--lr-lo/--lr-hi", %w[franken-moe],       (!@lr_schedule.nil? || !@lr_lo.nil? || !@lr_hi.nil?), " (toy#146)"],
            ["--lr-control/--lr-control-*", %w[franken-moe],         (!@lr_control.nil? || !@lr_control_window.nil? || !@lr_control_patience.nil? || !@lr_control_factor.nil? || !@lr_control_recover.nil? || !@lr_control_floor.nil?), " (toy#148)"],
            ["--ckpt-every",    %w[franken franken-moe gtx],        !@ckpt_every.nil?, " (toy#129/#131/#164)"],
            ["--load-ckpt",     %w[franken-moe gtx],                !@load_ckpt.nil?, " (toy#131 franken-moe: eval-only, pass --steps 0 + --eval-corpus; toy#164 gtx: needs --retrofit)"],
            ["--eval-corpus/--eval-tokens/--eval-offset", %w[franken-moe], (!@eval_corpus.nil? || !@eval_tokens.nil? || !@eval_offset.nil?), " (toy#130; the llama lane evals checkpoints offline via `toy eval ce`)"],
            ["--shape",         %w[franken franken-moe],            !@shape.nil?, ""],
            ["--routing/--moe-policy/--moe-aux", %w[franken-moe],   (!@routing.nil? || !@moe_policy.nil? || !@moe_aux.nil?), ""],
            ["--experts",       %w[franken-moe],                    !@experts.nil?, " (toy#128)"],
          ]
          flag_matrix.each do |label, recipes, used, note|
            next unless used && !recipes.include?(@recipe)
            names = recipes.map { |r| "'#{r}'" }.join(" or ")
            verb = label.include?("/") ? "are" : "is"
            return bad_arg("#{label} #{verb} only valid with recipe #{names}#{note}")
          end
          if (@eval_tokens || @eval_offset) && @eval_corpus.nil?
            return bad_arg("--eval-tokens/--eval-offset need --eval-corpus")
          end
          if (!@dfa_feedback_decay.nil? || !@dfa_feedback_lr.nil?) && @dfa_feedback != "kolen-pollack"
            return bad_arg("--dfa-feedback-decay/--dfa-feedback-lr need --dfa-feedback kolen-pollack")
          end
          if !@mtp_lambda.nil? && !@mtp
            return bad_arg("--mtp-lambda needs --mtp (there is no coupling to weight without the second root)")
          end
          if @eval_corpus && !File.file?(@eval_corpus)
            return bad_arg("no such file: #{@eval_corpus} (--eval-corpus streams packed-i32 tokens)")
          end
          # toy#129's rule, and it is about the FRANKEN lanes: there
          # `--vocab` names a headerless corpus pack's vocab, so it is
          # meaningless without `--corpus`. On gtx (toy#170/P5) the same
          # flag means the NOMINAL HEAD WIDTH and the data comes from
          # `--text`, so requiring `--corpus` there would refuse the one
          # configuration the flag exists for.
          if @vocab && @corpus.nil? && !%w[gtx].include?(@recipe)
            return bad_arg("--vocab needs --corpus (it names a headerless pack's vocab; TOYC packs declare their own)")
          end
          if @recipe == "franken-moe" && @context && @corpus.nil?
            return bad_arg("--context needs --corpus for franken-moe (the fixed-seq feed is the byte-gated 4-token contract)")
          end
          if @steps == 0 && @load_ckpt.nil?
            return bad_arg("--steps 0 is only meaningful with --load-ckpt (the eval-only reload, toy#131)")
          end
          if @load_ckpt && !File.file?(@load_ckpt)
            fmt = @recipe == "gtx" ? "toy-gtx/v1 backbone" : "toy-moe/v1"
            return bad_arg("no such file: #{@load_ckpt} (--load-ckpt takes a #{fmt} checkpoint from --ckpt-every)")
          end
          # toy#158: radam lives on the dense franken lane only — the moe
          # runner's optimizer allow-list does not know it, and a flag
          # the runner ignores is worse than one it rejects.
          if @recipe != "franken" && @optimizer == "radam"
            return bad_arg("--optimizer radam is only valid with recipe 'franken' (toy#158/F15: the RAdam rectification is wired in the dense lane's LR path)")
          end
          # tao#18: F15's small case is CPU-only. The CUDA twin is a
          # HAND-mirrored runner that does not read the granularity env
          # at all, so without this a `--device cuda --dfa-granularity
          # block` run would silently execute MICRO DFA and be recorded
          # as macro — the worst kind of wrong result. (The CUDA runner
          # also fails loud on its own, for callers that bypass the CLI.)
          if @recipe == "franken" && @dfa_granularity == "block" && @device != "cpu"
            return bad_arg("--dfa-granularity block is CPU-only on the franken lane (tao#18: F15's small case gets no CUDA twin)")
          end
          if @recipe == "franken" && @optimizer == "sgd"
            return bad_arg("--optimizer sgd is franken-moe-only (the llama lane's shared recipe uploads the AdamW hp unconditionally; muon keeps norms on adamw so it is fine there)")
          end
          if @donor && !File.file?(@donor)
            return bad_arg("no such file: #{@donor} (--donor takes a SAME-VOCAB GGUF; data/distilgpt2-f32.gguf is the GPT-2-vocab donor)")
          end
          if @donor_mode && @donor.nil?
            return bad_arg("--donor-mode needs --donor")
          end
          if @layer_pattern && @kda_layers
            return bad_arg("--layer-pattern and --kda-layers both set — the pattern computes the KDA layer list (toy#138)")
          end
          if @layer_pattern && @mla_layers
            return bad_arg("--layer-pattern and --mla-layers both set — the pattern computes the MLA layer list (K-series M2)")
          end
          if @mla_rank && !@mla_layers && @layer_pattern != "k3"
            return bad_arg("--mla-rank needs MLA layers — pass --mla-layers or --layer-pattern k3")
          end
          # The corpus rule is a TRANSFORMER-lane rule: there, B>1 means
          # B corpus windows per step. The mlp lane's batch is samples
          # from a seeded generator — there is no corpus to need.
          if @batch && @batch > 1 && @corpus.nil? && !%w[mlp ctr ssm lstm gtx diff].include?(@recipe)
            return bad_arg("--batch > 1 needs --corpus (the fixed-seq feed is the byte-gated single-window contract)")
          end
          if @no_shadow && @align_events
            return bad_arg("--no-shadow + --align-events: align telemetry compares DFA grads against the chain shadow acc, which a no-shadow build does not create — drop one")
          end
          if @recipe == "franken-moe" && @moe_aux && @routing != "top1"
            return bad_arg("--moe-aux requires --routing top1 (the aux-loss rides the hard router)")
          end
          if @recipe == "franken-moe" && @routing == "top1" && %w[chain dfa-experts].include?(@moe_policy)
            return bad_arg("--moe-policy #{@moe_policy} is dense-only; top1 takes no policy (fully-DFA lane) or --moe-policy bp-router-dfa-experts (toy#121)")
          end
          if @recipe == "franken-moe" && %w[bp-router-dfa-experts bp-spine].include?(@moe_policy) && @routing != "top1"
            return bad_arg("--moe-policy #{@moe_policy} requires --routing top1")
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
          if @recipe == "franken-moe" && @device == "metal"
            return bad_arg("franken-moe has no metal runner yet (CUDA + CPU only, toy#134)")
          end
          # tao#18 item 2: T0–T3 are CPU-only by decision, not by
          # accident — the anchors are small by construction, and a
          # hand-mirrored twin is exactly what drifted in toy#150/#151.
          if @recipe == "ctr" && @device != "cpu"
            return bad_arg("--device #{@device.inspect} is not supported for recipe 'ctr' (CPU-only by decision — tao#18)")
          end
          if @recipe == "mlp" && @device != "cpu"
            return bad_arg("--device #{@device.inspect} is not supported for recipe 'mlp' (CPU-only by decision — tao#18: the T0 anchor is small by construction and gets no CUDA twin)")
          end
          # --task's TOKEN SET is lane-local even though the flag name is
          # shared: `blobs` is the mlp anchor's degenerate control and
          # `community` is the gnn lane's. Accepting the wrong one would
          # silently run the default task under an experiment label.
          if (@task == "mixture" || @task == "single") && @recipe != "diff"
            return bad_arg("--task #{@task} is only valid with recipe 'diff' (toy#156)")
          end
          # cue|mean is the SHARED sequence-task vocabulary of the two
          # recurrent lanes — toy#157 reuses toy#155's generator verbatim,
          # so the token set has to be shared too or the same experiment
          # would need two names.
          if (@task == "cue" || @task == "mean") && !%w[ssm lstm].include?(@recipe)
            return bad_arg("--task #{@task} is only valid with recipe 'ssm' or 'lstm' (toy#155/#157)")
          end
          if (@task == "relational" || @task == "local") && @recipe != "gtx"
            return bad_arg("--task #{@task} is only valid with recipe 'gtx' (toy#160)")
          end
          # toy#169/#170 — the AR byte-LM task is SHARED by the two
          # generative-body lanes, for the same reason cue|mean is shared
          # by the recurrent pair: gtx and ssm run the same experiment on
          # different architectures, and one experiment gets one name.
          #
          # Every P2-P6 cell was driven through the GTX_*/SSM_* env
          # harness because this flag did not exist, which is what blocks
          # `tao conclude` from recording those runs against the registry.
          if @task == "bytelm" && !%w[gtx ssm].include?(@recipe)
            return bad_arg("--task bytelm is only valid with recipe 'gtx' or 'ssm' (toy#169/#170)")
          end
          # ...and the converse, which gate-ae caught: on gtx/ssm the
          # runner reads GTX_TEXT/SSM_TEXT ONLY under bytelm, so `--text`
          # on a relational or cue/mean run is accepted and then ignored.
          # A flag that silently does nothing is the failure this lane's
          # CLI checks exist to prevent, so refuse it rather than widen
          # the lane matrix and call it done.
          if !@text.nil? && %w[gtx ssm].include?(@recipe) && @task != "bytelm"
            return bad_arg("--text on recipe #{@recipe.inspect} is only valid with --task bytelm (toy#169/#170); the #{@recipe} lane's other tasks generate their own data and would ignore it")
          end
          # Same refusal as the ae/difflm lanes and for the same reason:
          # there is no synthetic fallback corpus, so a bytelm run without
          # --text would train on nothing and report a confident number.
          # Checked here as well as in the runner, and the pack is checked
          # to exist so a typo does not surface as "could not read".
          if @task == "bytelm"
            if @text.nil?
              return bad_arg("--task bytelm requires --text <pack-prefix> (prep/fetch_text.rb, or prep/remap_alphabet.rb for the dense packs)")
            end
            missing = %w[.meta.i32 .tok.i32].reject { |sfx| File.file?(@text + sfx) }
            unless missing.empty?
              return bad_arg("--text #{@text.inspect} is not a byte pack: missing #{missing.join(', ')}")
            end
          end
          # `--vocab` is SHARED with franken/franken-moe, where it names a
          # headerless pack's vocab (toy#129). On gtx it is the toy#170/P5
          # nominal head width, and it only means anything under bytelm —
          # for the relational task the head is TY*TY by construction.
          if !@vocab.nil? && @recipe == "gtx" && @task != "bytelm"
            return bad_arg("--vocab on recipe 'gtx' is only valid with --task bytelm (toy#170/P5: it sets the head, the embedding AND the DFA feedback matrix B together; the relational task's head is TY*TY)")
          end
          # ---- toy#172 / E1 Phase 1.2: the nDFA conditionals ----
          #
          # Three rules, and each of them exists because the alternative
          # is a flag that runs, reports itself on, and changes nothing:
          #   (a) bytelm only — the relational head is 16 classes and
          #       there is no wide error vector to whiten;
          #   (b) the ridge is REQUIRED — lambda is the experiment, not a
          #       tuning detail, and the identity/projector limits sit at
          #       its two ends;
          #   (c) the sub-knobs are meaningless with the preconditioner
          #       off, so passing them alone is refused rather than
          #       ignored.
          # The runner repeats (a) and (b) — a CLI check protects `toy
          # train`, and the sweeps call libexec/ directly.
          ndfa_sub = !@ndfa_lambda.nil? || !@ndfa_every.nil? ||
                     !@ndfa_samples.nil? || !@ndfa_gain.nil?
          if @recipe == "gtx" && (!@dfa_feedback_precond.nil? || ndfa_sub) && @task != "bytelm"
            return bad_arg("--dfa-feedback-precond/--ndfa-* on recipe 'gtx' require --task bytelm (toy#172/E1: nDFA inverts the ERROR COVARIANCE C_E and the question is about a WIDE error vector — the relational task's head is TY*TY = 16 classes, where the sample count passes the width on the first step and there is nothing to whiten)")
          end
          if @dfa_feedback_precond == "ndfa" && @ndfa_lambda.nil?
            return bad_arg("--dfa-feedback-precond ndfa requires --ndfa-lambda <ridge> — there is NO defensible default. P = lambda(C_E + lambda I)^-1 is the IDENTITY as lambda grows and a PROJECTOR as it shrinks, so the ridge IS the experiment. Read a scale off the Phase 1.1 instrument's lambda_max (GTX_INSTRUMENT=1)")
          end
          if ndfa_sub && @dfa_feedback_precond != "ndfa"
            return bad_arg("--ndfa-* is meaningless without --dfa-feedback-precond ndfa — the knobs would be parsed, wired and then never read, which is an inert flag that still labels the cell")
          end
          if @task == "community" && @recipe != "gnn"
            return bad_arg("--task community is only valid with recipe 'gnn' (toy#153); 'mlp' takes teacher|blobs")
          end
          if @task == "blobs" && @recipe != "mlp"
            return bad_arg("--task blobs is only valid with recipe 'mlp' (toy#152); 'gnn' takes teacher|community, 'ssm'/'lstm' take cue|mean")
          end
          if @recipe == "diff" && @device != "cpu"
            return bad_arg("--device #{@device.inspect} is not supported for recipe 'diff' (CPU-only by decision — tao#18)")
          end
          if @recipe == "ssm" && @device != "cpu"
            return bad_arg("--device #{@device.inspect} is not supported for recipe 'ssm' (CPU-only this slice — tao#19 defers the F19 CUDA twin to the long-sequence memory measurement)")
          end
          # tao#21: a device port changes THROUGHPUT, not what the graph
          # materialises — and this lane's success target is stated in
          # materialised bytes. So the CUDA twin would not move the
          # measurement it exists for, and is not spent.
          if @recipe == "lstm" && @device != "cpu"
            return bad_arg("--device #{@device.inspect} is not supported for recipe 'lstm' (CPU-only by decision — tao#18/#21: a device twin changes throughput, not the materialised activation bytes this lane measures)")
          end
          # toy#165 (ae): there is NO synthetic fallback corpus, on
          # purpose. A low-entropy synthetic byte stream would inflate the
          # noise margin at exactly the small latent where the verdict is
          # decided (tao#22), so a run without --text would produce a
          # confident and wrong `go`. Refuse it here as well as in the
          # runner, and check the pack exists rather than letting a typo
          # surface as "could not read".
          if @recipe == "difflm"
            if @text.nil?
              return bad_arg("recipe 'difflm' requires --text <pack-prefix> (prep/fetch_text.rb)")
            end
            missing = %w[.meta.i32 .tok.i32].reject { |sfx| File.file?(@text + sfx) }
            unless missing.empty?
              return bad_arg("--text #{@text.inspect} is not a byte pack: missing #{missing.join(', ')}")
            end
            if @device != "cpu"
              return bad_arg("--device #{@device.inspect} is not supported for recipe 'difflm' (CPU-only by decision — tao#18)")
            end
            if @policy
              return bad_arg("--policy is not valid on recipe 'difflm' — P1b is ALL BP by design; DFA arrives in P1c and attaches to the DENOISER (output dim = the latent), never to the 256-way decode head")
            end
          end
          if @recipe == "ae"
            if @text.nil?
              return bad_arg("recipe 'ae' requires --text <pack-prefix> — this lane measures a REAL text corpus and has no synthetic fallback (tao#22). Run: ruby prep/fetch_text.rb --all, then --text data/ae_names (or ae_shakespeare / ae_udhr)")
            end
            missing = %w[.meta.i32 .tok.i32].reject { |sfx| File.file?(@text + sfx) }
            unless missing.empty?
              return bad_arg("--text #{@text.inspect} is not a byte pack: missing #{missing.join(', ')}. Packs are built by prep/fetch_text.rb, not by --corpus (a TOYC pack is a different format)")
            end
            if @device != "cpu"
              return bad_arg("--device #{@device.inspect} is not supported for recipe 'ae' (CPU-only by decision — tao#18: the cross-architecture lanes are small by construction and get no CUDA twin)")
            end
            if @target_ce && @eval_every.nil?
              return bad_arg("--target-ce needs --eval-every N — a stopping criterion that is never checked would silently run to --steps and report an unmatched cell as matched")
            end
            if @eval_every && @target_ce.nil?
              return bad_arg("--eval-every without --target-ce costs a held-out pass every N steps and changes nothing")
            end
            if @latent && @d_model && @latent >= @d_model
              return bad_arg("--latent #{@latent} must be < --d-model #{@d_model} — a bottleneck at least as wide as the residual stream is not a bottleneck")
            end
          end
          if @recipe == "gtx" && @load_ckpt && !@retrofit
            return bad_arg("--load-ckpt needs --retrofit on recipe 'gtx' (toy#164: a loaded backbone on this lane exists to be retrofitted; a second meaning for it would make the flag ambiguous)")
          end
          if (@adapter_policy || @no_freeze_backbone) && !@retrofit
            return bad_arg("--adapter-policy/--no-freeze-backbone need --retrofit (there are no adapters and no frozen backbone outside a retrofit, so the flag would silently do nothing)")
          end
          # tao#24 — gtx is the ONE cross-architecture lane with a device
          # choice, and it is scoped to a single TASK rather than to the
          # recipe. The original CPU-only call (recorded in code as
          # "tao#18", a stale pre-GitHub number; the decision is tao#24)
          # was right for the relational task: d_model 64, 16 relation
          # classes, 1500 steps, where a GPU buys nothing. `--task bytelm`
          # (toy#170/P3) is a different workload by orders of magnitude —
          # vocab up to 4096, ctx 128, 4000 steps, ~3.2 TFLOP/cell measured
          # at ~32 GFLOP/s with the accelerator idle. The scope lapsed; the
          # decision was not wrong, and it lapsed for exactly one task.
          #
          # metal is refused outright: no metal gtx binary is built, and
          # accepting the flag to fail later at the build is worse than
          # saying so here.
          if @recipe == "gtx" && @device == "metal"
            return bad_arg("--device metal is not supported for recipe 'gtx' (tao#24 built a CUDA twin for --task bytelm only; there is no metal gtx binary)")
          end
          if @recipe == "gtx" && @device == "cuda" && @task != "bytelm"
            return bad_arg("--device cuda on recipe 'gtx' requires --task bytelm (tao#24: the CUDA twin was built for THAT task — vocab up to 4096, ctx 128, ~3.2 TFLOP/cell — and the relational/local tasks stay CPU-only because a 16-class head at d_model 64 gains nothing from a GPU). Drop --device, or add --task bytelm --text <pack>.")
          end
          # NOTE for anyone reading a number off this lane: CUDA cells are
          # NOT numerically comparable to the CPU P3-P6 cells. A sweep runs
          # entirely on one device, or it re-runs its reference there.
          if @recipe == "gnn" && @device != "cpu"
            return bad_arg("--device #{@device.inspect} is not supported for recipe 'gnn' (CPU-only by decision — tao#18)")
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
          return "mlp"     if recipe == "mlp"
          return "ctr"     if recipe == "ctr"
          return "gnn"     if recipe == "gnn"
          return "ssm"     if recipe == "ssm"
          return "lstm"    if recipe == "lstm"
          return "gtx"     if recipe == "gtx"
          return "ae"      if recipe == "ae"
          return "difflm"  if recipe == "difflm"
          return "diff"    if recipe == "diff"
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
