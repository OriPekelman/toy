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
# kind/schema/t/started_at/run_id/phase/host{}/model{}/config{}, step
# carries kind/phase/t/step/loss, run_end carries kind/t/ended_at/
# reason/final_step/final_loss/exit_code — the same shape the train
# runners emit (lib/toy/run/train.rb), minus the runner-only extras
# (backend{}, git{}, lr/tokens/wall_us — all v1-optional keys; the
# schema is open, consumers ignore absences). Toy::RunLog reads the
# result back (round-tripped in prep/run_log_gate.rb).
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
    attr_accessor :rb_root, :rb_run_id, :rb_dir, :rb_active

    # Create runs_root/ and runs_root/run_id/, then TRUNCATE-open the
    # events.jsonl sink. On open failure (rc != 0) warns loud with the
    # rc + path and continues with events disabled (active == false) —
    # compute must not die because a bundle dir is unwritable.
    def initialize(runs_root, run_id)
      @rb_root   = runs_root
      @rb_run_id = run_id
      @rb_dir    = runs_root + "/" + run_id
      @rb_active = false
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

    # Emit the toy/v1 run_start (exactly one per bundle, first):
    # schema + t + started_at + run_id + phase:"train" + host{} +
    # model{arch,vocab,d_model,n_layers} + config{context,steps,lr,seed}.
    # arch is e.g. "llama" / "gpt2" / "vit". Returns nil.
    def run_start!(arch, vocab, d_model, n_layers, context, steps, lr, seed)
      if @rb_active
        TinyNN.tnn_events_emit("{\"kind\":\"run_start\",\"schema\":\"toy/v1\"" +
          ",\"t\":" + TinyNN.tnn_events_now_seconds.to_s +
          ",\"started_at\":\"" + TinyNN.tnn_events_iso8601_now + "\"" +
          ",\"run_id\":\"" + RunBundle.json_escape(@rb_run_id) + "\"" +
          ",\"phase\":\"train\"" +
          ",\"host\":{\"name\":\"" +
            RunBundle.json_escape(TinyNN.tnn_provenance_host_name) +
          "\",\"os\":\"" + TinyNN.tnn_provenance_host_os +
          "\",\"arch\":\"" + TinyNN.tnn_provenance_host_arch + "\"}" +
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
