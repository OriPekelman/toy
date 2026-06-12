# lib/toy/io/run_bundle.rb — Toy::RunBundle: the runs/<id>/ event-bundle
# WRITER for compiled consumers (toy#73 item 1).
#
# Before this class, a compiled consumer that wanted the toy/v1 bundle
# hand-rolled JSON lines against the raw tnn_events_* FFI (example 01
# burned ~15 lines on it), and the append-only sink meant a re-run with
# the same run id DOUBLED the bundle. RunBundle owns the whole story:
#
#   bundle = Toy::RunBundle.new("runs", RUN_ID)   # mkdir + TRUNCATE-open
#   bundle.run_start!("llama", cfg.vocab, cfg.d_model, cfg.n_layers,
#                     cfg.ctx, STEPS, LR, SEED)
#   bundle.step!(step + 1, loss)                  # one per step
#   bundle.run_end!(STEPS, final_loss)            # emits run_end + closes
#
# The stream is the toy/v1 schema (docs/events.md): run_start carries
# kind/schema/t/started_at/run_id/phase/host{}/backend{}/[git{}/]
# model{}/config{}, step carries kind/phase/t/step/loss, run_end
# carries kind/t/ended_at/reason/final_step/final_loss/exit_code — the
# same shape the train runners emit (lib/toy/run/train.rb), minus the
# per-step runner extras (lr/tokens/wall_us — v1-optional keys; the
# schema is open, consumers ignore absences). Toy::RunLog reads the
# result back (round-tripped in prep/run_log_gate.rb).
#
# PROVENANCE (toy#73 A.3 seed b):
#   backend{kind} — AUTOMATIC. The device is compile-time known (you
#     picked it by requiring compute.rb / compute_cuda.rb /
#     compute_metal.rb), so run_start! stamps Toy::Device.name itself.
#     This is why the compute entries require this file AFTER defining
#     Toy::Device — Spinel needs the constant defined before a file
#     that references it compiles.
#   git{sha,branch} — OPT-IN via git!(sha, branch) before run_start!.
#     It CANNOT be automatic: the reader lives in SpinelKit::Git
#     (vendor/spinel/spinel_kit/, toy#44), and a require_relative from
#     here cannot name it in both layouts this file ships in — in the
#     toy repo it sits at ../../../vendor/spinel/spinel_kit/... (and
#     only exists after `make vendor-tep`, which gates on sibling
#     checkouts — a fresh clone must still compile example 01), while
#     vendored into a consumer (`toy new --lib`) this file lives at
#     vendor/spinel/toy/lib/toy/io/ and the gem sits at
#     ../../../../spinel_kit/... — a DIFFERENT depth, and Spinel
#     compiles a conditional require_relative to 0 (silently — see the
#     compute.rb header). So callers that DO load SpinelKit::Git (the
#     runners' pattern, by path) inject the two strings:
#       gp = SpinelKit::Git.read
#       bundle.git!(gp.sha, gp.branch)
#     Un-injected bundles simply omit git{} (v1-optional key).
#
# FRESH-BUNDLE SEMANTICS: the ctor opens events.jsonl via
# tnn_events_open_trunc (toy#73's new FFI hook) — a re-run with the
# same run id REPLACES the bundle instead of doubling it. The runners'
# append-only tnn_events_open stays untouched (the CLI creates a fresh
# run dir per run, so append is correct there).
#
# CHECKPOINT-DIR CONVENTION: weights_dir creates + returns
# runs/<id>/weights — the documented checkpoint home (docs/events.md
# "Run directory layout") — so consumers put GGUF snapshots where
# `toy infer` and the bundle tooling expect them.
#
# JSON: self-contained minimal escaper (same escape table as
# SpinelKit::Json::Builder) — NO vendor/spinel/spinel_kit require, so a
# fresh clone compiles the examples without `make vendor-tep`. Values
# here are run ids / arch names + numbers; control bytes beyond
# \" \\ \n \r \t pass through (don't put binary in a run id).
#
# BACKEND COUPLING: none in practice — the tnn_events_* / mkdir C
# symbols live in libtinynn_ggml.a, which EVERY backend links, and the
# CPU TinyNN module is defined in every compiled binary (transformer.rb
# requires ffi/tinynn). So this file is SHARED across the CPU / CUDA /
# Metal compute entries (like ToyCorpusLoader), NOT mirrored.
#
# Spinel hygiene: PLAIN class (no Struct.new — landmine #16), explicit
# ctor with NO default args (landmine #4), while-loops, NO #{}
# interpolation, uniquely rb_-prefixed ivars, FFI :str calls only
# inside method bodies (step_bind :str landmine 2026-05-28). Warns
# loud + disables on open failure (never silently swallows events).

module Toy
  class RunBundle
    attr_accessor :rb_root, :rb_run_id, :rb_dir, :rb_active,
                  :rb_git_sha, :rb_git_branch

    # Create runs_root/ and runs_root/run_id/, then TRUNCATE-open the
    # events.jsonl sink. On open failure (rc != 0) warns loud with the
    # rc + path and continues with events disabled (active == false) —
    # compute must not die because a bundle dir is unwritable.
    def initialize(runs_root, run_id)
      @rb_root       = runs_root
      @rb_run_id     = run_id
      @rb_dir        = runs_root + "/" + run_id
      @rb_active     = false
      @rb_git_sha    = ""   # empty = git{} omitted from run_start
      @rb_git_branch = ""
      TinyNN.tnn_filesystem_mkdir(runs_root)
      TinyNN.tnn_filesystem_mkdir(@rb_dir)
      rc = TinyNN.tnn_events_open_trunc(@rb_dir + "/events.jsonl")
      if rc == 0
        @rb_active = true
      else
        puts "warn: Toy::RunBundle could not open " + @rb_dir +
             "/events.jsonl (rc=" + rc.to_s + ") — run bundle disabled"
      end
    end

    def active
      @rb_active
    end

    def events_path
      @rb_dir + "/events.jsonl"
    end

    # The checkpoint-dir convention: runs/<id>/weights/, created on
    # first ask. Pass the returned path to ToyGGUFWriter.write_step (or
    # any GGUF writer) so the bundle layout matches docs/events.md.
    def weights_dir
      d = @rb_dir + "/weights"
      TinyNN.tnn_filesystem_mkdir(d)
      d
    end

    # OPT-IN git provenance (see the header — it cannot be automatic).
    # Call BEFORE run_start! with the two strings from SpinelKit::Git
    # (or any sha/branch pair); run_start! then includes
    # git{sha,branch} after backend{}. Returns nil.
    def git!(sha, branch)
      @rb_git_sha    = sha
      @rb_git_branch = branch
      nil
    end

    # The shared run_start prefix: kind/schema/t/started_at/run_id/
    # phase:"train" + host{} + backend{kind: Toy::Device.name} +
    # git{} when injected via git! — the same key order as the train
    # runners' Toy::Events.add_provenance. Callers append model{} +
    # config{} and emit.
    def run_start_prefix
      s = "{\"kind\":\"run_start\",\"schema\":\"toy/v1\"" +
          ",\"t\":" + TinyNN.tnn_events_now_seconds.to_s +
          ",\"started_at\":\"" + TinyNN.tnn_events_iso8601_now + "\"" +
          ",\"run_id\":\"" + RunBundle.json_escape(@rb_run_id) + "\"" +
          ",\"phase\":\"train\"" +
          ",\"host\":{\"name\":\"" +
            RunBundle.json_escape(TinyNN.tnn_provenance_host_name) +
          "\",\"os\":\"" + TinyNN.tnn_provenance_host_os +
          "\",\"arch\":\"" + TinyNN.tnn_provenance_host_arch + "\"}" +
          ",\"backend\":{\"kind\":\"" + Toy::Device.name + "\"}"
      if @rb_git_sha.length > 0
        s = s + ",\"git\":{\"sha\":\"" + RunBundle.json_escape(@rb_git_sha) +
            "\",\"branch\":\"" + RunBundle.json_escape(@rb_git_branch) + "\"}"
      end
      s
    end

    # Emit the toy/v1 run_start (exactly one per bundle, first):
    # schema + t + started_at + run_id + phase:"train" + host{} +
    # backend{} + [git{} if injected] + model{arch,vocab,d_model,
    # n_layers} + config{context,steps,lr,seed}.
    # arch is e.g. "llama" / "gpt2" / "vit". Returns nil.
    def run_start!(arch, vocab, d_model, n_layers, context, steps, lr, seed)
      if @rb_active
        TinyNN.tnn_events_emit(run_start_prefix +
          ",\"model\":{\"arch\":\"" + RunBundle.json_escape(arch) + "\"" +
          ",\"vocab\":" + vocab.to_s +
          ",\"d_model\":" + d_model.to_s +
          ",\"n_layers\":" + n_layers.to_s + "}" +
          ",\"config\":{\"context\":" + context.to_s +
          ",\"steps\":" + steps.to_s +
          ",\"lr\":" + lr.to_s +
          ",\"seed\":" + seed.to_s + "}}")
      end
      nil
    end

    # Emit one toy/v1 step event. `step` is the 1-indexed step number
    # (matches the runners' "step N: loss=" stdout line). Returns nil.
    def step!(step, loss)
      if @rb_active
        TinyNN.tnn_events_emit("{\"kind\":\"step\",\"phase\":\"train\"" +
          ",\"t\":" + TinyNN.tnn_events_now_seconds.to_s +
          ",\"step\":" + step.to_s +
          ",\"loss\":" + loss.to_s + "}")
      end
      nil
    end

    # Emit the toy/v1 run_end (exactly one, last) and CLOSE the sink.
    # reason:"completed" / exit_code:0 — RunBundle is for runs that ran
    # to completion; a crashed run just leaves the bundle without a
    # run_end (the documented torn-bundle case consumers handle).
    # Returns nil.
    def run_end!(final_step, final_loss)
      if @rb_active
        TinyNN.tnn_events_emit("{\"kind\":\"run_end\"" +
          ",\"t\":" + TinyNN.tnn_events_now_seconds.to_s +
          ",\"ended_at\":\"" + TinyNN.tnn_events_iso8601_now + "\"" +
          ",\"reason\":\"completed\"" +
          ",\"final_step\":" + final_step.to_s +
          ",\"final_loss\":" + final_loss.to_s +
          ",\"exit_code\":0}")
        TinyNN.tnn_events_close
        @rb_active = false
      end
      nil
    end

    # Minimal JSON string-body escaper (same table as
    # SpinelKit::Json::Builder.escape, minus the \u00XX arm — run ids
    # and arch names are ASCII-clean by construction).
    def self.json_escape(s)
      out = ""
      i = 0
      n = s.length
      while i < n
        c = s[i]
        if c == "\""
          out = out + "\\\""
        elsif c == "\\"
          out = out + "\\\\"
        elsif c == "\n"
          out = out + "\\n"
        elsif c == "\r"
          out = out + "\\r"
        elsif c == "\t"
          out = out + "\\t"
        else
          out = out + c
        end
        i = i + 1
      end
      out
    end
  end
end
