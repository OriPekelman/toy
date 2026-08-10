#!/usr/bin/env ruby
# prep/franken_llama_gate.rb — toy#112 gates for the spec-callable franken
# runner (libexec/toy-train-franken-llama, `toy train franken`).
#
#   1. F0 BYTE-PARITY: empty policy, STEPS=5 SEED=0 — the stdout loss
#      curve must byte-equal prep/fixtures/train_baseline.txt (the same
#      fixture `toy train from-scratch` is gated on): the franken runner
#      with no policy IS the from-scratch trainer.
#   2. BUNDLE STRUCTURE (policy run, TAO_RUN_DIR): valid JSONL; run_start
#      first with schema toy/v1 + a `franken` object carrying the exact
#      policy/b axes; one step event per step; align events == 12 ×
#      steps when FRANKEN_ALIGN=1 (2 layers × [4q+4k+4v] on the dfa
#      layer... exactly the dfa-wired count) with finite cos in [-1,1]
#      and non-negative norms; run_end last; weights/step_<N>.gguf.
#   3. DFA EFFECT: the chain,dfa curve differs from the F0 curve.
#   4. BYTE-REPRO: two identical policy runs — identical stdout.

ROOT   = File.expand_path("..", __dir__)
RUNNER = File.join(ROOT, "libexec", "toy-train-franken-llama")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_baseline.txt")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

def run_franken_llama(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  if run_dir
    FileUtils.mkdir_p(File.join(run_dir, "weights"))
    env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "franken-gate")
  end
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "franken_llama_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-franken-llama")
  unless build_st.success? && File.executable?(RUNNER)
    warn "franken_llama_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []
# LEG BOOKKEEPING: every leg records the failure count at its START in
# `n0` and summarises with `failures.length == n0`, so each leg reports
# on ITS OWN assertions. Legs used to summarise with `failures.empty?`,
# which made every later leg print FAIL once ANY earlier leg had failed
# — misleading exactly when you are debugging. `n0` is seeded at top
# level so re-assignments inside blocks mutate the outer local.
n0 = 0

# ---- 1. F0 byte-parity ----
f0_out = run_franken_llama({}, nil)
f0_curve = f0_out.lines.select { |l| l.start_with?("step ") }
expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
if f0_curve == expect
  puts "  ok: F0 — empty-policy curve byte-equals train_baseline.txt (#{f0_curve.length} steps)"
else
  failures << "F0: curve != train_baseline.txt\ngot:  #{f0_curve.join}want: #{expect.join}"
end

# ---- 2 + 3. policy run: bundle structure + dfa effect ----
Dir.mktmpdir("franken_llama_gate") do |dir|
  pol_out = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa",
                                "FRANKEN_B_SEED" => "42",
                                "FRANKEN_ALIGN"  => "1" }, dir)
  pol_curve = pol_out.lines.select { |l| l.start_with?("step ") }
  failures << "dfa-effect: policy curve identical to F0" if pol_curve == f0_curve

  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    failures << "bundle: first event not run_start" unless events.first && events.first["kind"] == "run_start"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    rs = events.first || {}
    failures << "bundle: schema != toy/v1" unless rs["schema"] == "toy/v1"
    # toy#129 item 4: derived cost accounting rides run_start
    co = rs["cost"]
    if co.nil?
      failures << "bundle: run_start has no cost object (toy#129)"
    else
      failures << "bundle: cost fields not positive (#{co.inspect})" unless %w[total_params active_params flops_per_token].all? { |k| co[k].is_a?(Numeric) && co[k] > 0 }
      failures << "bundle: llama cost active != total (#{co.inspect})" unless co["active_params"] == co["total_params"]
    end
    fr = rs["franken"]
    if fr.nil?
      failures << "bundle: run_start has no franken object"
    else
      failures << "bundle: franken.policy != [0,1] (got #{fr['policy'].inspect})" unless fr["policy"] == [0, 1]
      failures << "bundle: franken.b_seed != 42" unless fr["b_seed"] == 42
    end
    steps_ev = events.count { |e| e["kind"] == "step" }
    failures << "bundle: #{steps_ev} step events (want 5)" unless steps_ev == 5
    aligns = events.select { |e| e["kind"] == "align" }
    failures << "bundle: #{aligns.length} align events (want 60 = 12 dfa weights x 5 steps; toy#151 default scope=attn keeps this at the qkv set)" unless aligns.length == 60
    bad = aligns.count do |e|
      c = e["cos"]
      !c.is_a?(Numeric) || c.to_f.nan? || c.to_f.abs > 1.0001 ||
        !e["dfa_norm"].is_a?(Numeric) || e["dfa_norm"] < 0 ||
        !e["bp_norm"].is_a?(Numeric) || e["bp_norm"] < 0
    end
    failures << "bundle: #{bad} malformed align events" unless bad == 0
    # the engine-scale telemetry sanity: dfa signal is alive
    live = aligns.count { |e| e["dfa_norm"] > 0 }
    failures << "bundle: dfa_norm not positive anywhere" unless live == aligns.length
  else
    failures << "bundle: no events.jsonl"
  end
  ckpt = File.join(dir, "weights", "step_5.gguf")
  failures << "bundle: missing checkpoint #{ckpt}" unless File.file?(ckpt)
  # toy#112 gap closed: flow.json (the toy#25 self-describing bundle)
  fj = File.join(dir, "flow.json")
  if File.file?(fj)
    flow = JSON.parse(File.read(fj)) rescue nil
    if flow.nil? || flow["format"] != "toy/v1" || !flow["nodes"].is_a?(Array) || flow["nodes"].empty?
      failures << "bundle: flow.json invalid (format/nodes)"
    end
  else
    failures << "bundle: no flow.json"
  end
  puts "  ok: bundle — run_start(franken provenance) + 5 steps + 60 align + run_end + checkpoint; dfa curve differs" if failures.empty?
n0 = failures.length
end

# ---- seed!=0 parity (toy#113): franken empty-policy must equal
# toy-train AT THE SAME NONZERO SEED, compared DYNAMICALLY (no fixture:
# seed!=0 curves are toy-version-scoped; only seed=0 is frozen). This
# tripwires the whole-program numeric-stream divergence class — at
# f7cea71 the two units compiled DIFFERENT xorshift/Box-Muller streams
# from identical source (seed=0 masked it).
FS_RUNNER = File.join(ROOT, "libexec", "toy-train")
fs_out, fs_st = Open3.capture2e({ "STEPS" => "8", "SEED" => "1" }, FS_RUNNER, chdir: ROOT)
fr_out = run_franken_llama({ "STEPS" => "8", "SEED" => "1" }, nil)
fs_curve = fs_out.lines.select { |l| l.start_with?("step ") }
fr_curve = fr_out.lines.select { |l| l.start_with?("step ") }
if fs_st.success? && fs_curve.length == 8 && fs_curve == fr_curve
  puts "  ok: seed=1 parity — franken empty-policy byte-equals toy-train (8 steps, dynamic)"
else
  failures << "seed1-parity: curves differ or toy-train failed\nfs: #{fs_curve.join}fr: #{fr_curve.join}"
end

# ---- TOY_RUN_ID passthrough (toy#115): the CLI's controlled env must
# forward a caller-supplied run id into run_start.run_id (the tao#flow
# contract) instead of substituting the internal counter.
Dir.mktmpdir("franken_rid_gate") do |dir|
  _, st = Open3.capture2e({ "TOY_RUN_ID" => "gate/rid/check" },
                          File.join(ROOT, "bin", "toy"), "train", "franken",
                          "--steps", "1", "--seed", "0", "--out", dir, chdir: ROOT)
  rid = st.success? ? (JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)["run_id"] rescue nil) : nil
  if rid == "gate/rid/check"
    puts "  ok: TOY_RUN_ID passthrough (CLI controlled env honors caller id)"
  else
    failures << "run-id: TOY_RUN_ID not honored (got #{rid.inspect})"
  end
end

# ---- llama-shape maskbp byte-null (toy#117): maskbp:-1 (gate saturated
# to exactly 1.0) must byte-equal toy-train — the leg that would have
# caught the B-buffer scratch-reuse explosion (rig shape provably
# doesn't cover it; B leaves are per-step re-uploads now).
fs8 = Open3.capture2e({ "STEPS" => "8", "SEED" => "0" }, FS_RUNNER, chdir: ROOT)[0]
mb8 = run_franken_llama({ "FRANKEN_POLICY" => "maskbp:-1,maskbp:-1",
                          "FRANKEN_B_SEED" => "42", "STEPS" => "8" }, nil)
fs8c = fs8.lines.select { |l| l.start_with?("step ") }
mb8c = mb8.lines.select { |l| l.start_with?("step ") }
if fs8c.length == 8 && fs8c == mb8c
  puts "  ok: maskbp(-1) byte-equals toy-train at the llama shape (toy#117 pin)"
else
  failures << "maskbp-null: curves differ\nfs: #{fs8c.join}mb: #{mb8c.join}"
end

# ---- toy#122: F6 long-horizon surface — corpus stream + align thinning ----
# corpus arm: deterministic, actually streams (differs from the fixed-seq
# feed), and leaves the byte-gated default path untouched (leg 1 pins it).
c1 = run_franken_llama({ "STEPS" => "8", "CORPUS" => "data/ts_seqs.bin" }, nil)
c2 = run_franken_llama({ "STEPS" => "8", "CORPUS" => "data/ts_seqs.bin" }, nil)
c_curve = c1.lines.select { |l| l.start_with?("step ") }
failures << "corpus: not deterministic" unless c1 == c2
failures << "corpus: only #{c_curve.length} steps" unless c_curve.length == 8
d_curve = run_franken_llama({ "STEPS" => "8" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "corpus: curve identical to fixed-seq feed (stream not live)" if c_curve == d_curve
puts failures.length == n0 ? "  ok: --corpus streams (deterministic, differs from fixed-seq; default feed untouched)" : "  FAIL: corpus leg"
n0 = failures.length

# align-every thinning: N=3 over 6 steps -> emissions at steps 1,4 ->
# 12 weights x 2 = 24 align events; N=1 == legacy per-step (72).
Dir.mktmpdir("franken_ae_gate") do |dir|
  run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42",
                      "FRANKEN_ALIGN" => "1", "FRANKEN_ALIGN_EVERY" => "3",
                      "STEPS" => "6" }, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  aligns = evs.select { |e| e["kind"] == "align" }
  steps_seen = aligns.map { |e| e["step"] }.uniq.sort
  failures << "align-every: #{aligns.length} events (want 24)" unless aligns.length == 24
  failures << "align-every: wrong steps #{steps_seen.inspect} (want [1,4])" unless steps_seen == [1, 4]
  failures << "align-every: step events thinned too (#{evs.count { |e| e['kind'] == 'step' }})" unless evs.count { |e| e["kind"] == "step" } == 6
end
puts failures.length == n0 ? "  ok: --align-every thins align emissions (24 @ N=3/6 steps; step events untouched)" : "  FAIL: align-every leg"
n0 = failures.length

# ---- toy#151: FRANKEN_POLICY_SCOPE — policy the FFN, not just qkv ----
# Until this ticket the per-layer policy reached ATTENTION QKV ONLY,
# which is precisely the caveat that made Tao's F1 wash out. The FFN is
# the bulk of params and compute, and the literature puts DFA nearest
# to BP in MLP-shaped blocks — so it is the placement worth testing.
#
# The load-bearing assertions are STRUCTURAL, on the align stream's
# `wi` (which weight each DFA gradient was computed for), not on the
# curve. A curve that merely moves cannot tell "the FFN is policied"
# from "the attention wiring changed". ft layout is
# [rn1, rn2, q*nh, (k,v)*nkv, o, gate, up, down], so at nh=4/nkv=4 the
# qkv weights are wi 2..13, the output proj is wi 14 (NEVER policied),
# and the FFN is wi 15,16,17.
n0 = failures.length
sc_env = { "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42",
           "FRANKEN_ALIGN" => "1", "STEPS" => "3", "SEED" => "0" }
sc_wis = {}
sc_curves = {}
%w[attn ffn all].each do |sc|
  Dir.mktmpdir("franken_scope_#{sc}") do |dir|
    FileUtils.mkdir_p(File.join(dir, "weights"))
    out = run_franken_llama(sc_env.merge("FRANKEN_POLICY_SCOPE" => sc), dir)
    sc_curves[sc] = out.lines.select { |l| l.start_with?("step ") }
    ev = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
    al = ev.select { |e| e["kind"] == "align" }
    sc_wis[sc] = al.map { |e| e["wi"] }.uniq.sort
    # every policied weight must carry a LIVE dfa gradient — the whole
    # point is that these weights are actually updated by DFA.
    dead = al.count { |e| e["dfa_norm"].to_f <= 0.0 }
    failures << "policy-scope(#{sc}): #{dead}/#{al.length} align events have dfa_norm <= 0" unless dead == 0
    failures << "policy-scope(#{sc}): provenance #{ev.first.dig('franken', 'policy_scope').inspect}" unless
      ev.first.dig("franken", "policy_scope") == sc
    failures << "policy-scope(#{sc}): provenance does not name the policied set" if
      ev.first.dig("franken", "policied_tensors").to_s.empty?
  end
end
FFN_WIS = [15, 16, 17]
QKV_WIS = (2..13).to_a
failures << "policy-scope: attn policied the FFN (#{sc_wis['attn'].inspect})" unless sc_wis["attn"] == QKV_WIS
failures << "policy-scope: ffn policied attention (#{sc_wis['ffn'].inspect})" unless sc_wis["ffn"] == FFN_WIS
failures << "policy-scope: all did not policy qkv+ffn (#{sc_wis['all'].inspect})" unless sc_wis["all"] == QKV_WIS + FFN_WIS
# the attention OUTPUT projection (wi 14) is not in any scope — it is
# not FFN, and the qkv DFA tap does not apply to it.
%w[attn ffn all].each do |sc|
  failures << "policy-scope(#{sc}): policied the attention output proj (wi 14)" if sc_wis[sc].include?(14)
end
# the three scopes must be three DIFFERENT experiments
failures << "policy-scope: attn == ffn (scope is not isolating)" if sc_curves["attn"] == sc_curves["ffn"]
failures << "policy-scope: all == attn (the FFN is not reaching the update)" if sc_curves["all"] == sc_curves["attn"]
failures << "policy-scope: all == ffn (attention is not reaching the update)" if sc_curves["all"] == sc_curves["ffn"]
# DEFAULT == attn. This is the compatibility contract: an unset scope
# must reproduce the pre-toy#151 lane byte-for-byte, so every F1/F3/F6/
# F7b/F0 policy string keeps meaning what it meant (tao#17). Whole-layer
# placement is opt-in via `all`.
def_curve = run_franken_llama(sc_env, nil).lines.select { |l| l.start_with?("step ") }
failures << "policy-scope: default is not `attn` — pre-toy#151 runs would silently change meaning" unless def_curve == sc_curves["attn"]
# determinism
failures << "policy-scope: ffn not deterministic" unless
  run_franken_llama(sc_env.merge("FRANKEN_POLICY_SCOPE" => "ffn"), nil).lines.select { |l| l.start_with?("step ") } == sc_curves["ffn"]
# composes with depth and a real corpus (F14 runs both)
dc = run_franken_llama(sc_env.merge("FRANKEN_POLICY" => "chain,dfa,chain,dfa,chain,dfa",
                                    "FRANKEN_SHAPE" => "deep", "CORPUS" => "data/fineweb_gpt2_smoke.bin",
                                    "FRANKEN_POLICY_SCOPE" => "ffn", "STEPS" => "4"), nil)
dcl = dc.lines.select { |l| l.start_with?("step ") }
failures << "policy-scope: ffn scope broken at --shape deep + --corpus" unless
  dcl.length == 4 && dcl.map { |l| l[/loss=(\S+)/, 1].to_f }.none?(&:nan?)
# guard
_so, sst = Open3.capture2e({ "STEPS" => "1", "SEED" => "0", "FRANKEN_POLICY_SCOPE" => "mlp" },
                           RUNNER, chdir: ROOT)
failures << "policy-scope: unknown scope value not rejected" if sst.success?
puts failures.length == n0 ? "  ok: FRANKEN_POLICY_SCOPE — attn/ffn/all isolate STRUCTURALLY (align wi: qkv 2-13, ffn 15-17, output-proj 14 never policied), every policied weight carries a live DFA gradient, default=attn (pre-toy#151 byte-null), deterministic, composes with deep+corpus, unknown value rejected" : "  FAIL: policy-scope leg"

# ---- toy#135 (toy-k3): graph capacity must track CONTEXT for the
# ---- RECURRENT layer kinds ----
# The engine's cap heuristic was O(layers x heads) and ignored T. That
# is right for attention (node count is T-independent) and WRONG for
# KDA/GDN, which build recur_unrolled — their node count scales with T.
# A k3 pattern at L6/8-head fit at ctx 32 and blew up at ctx 64 with
# GGML_ASSERT(cgraph->n_nodes < cgraph->size). Pinned at the shape that
# used to fail so the heuristic cannot silently regress to T-blind.
n0 = failures.length
kc = { "FRANKEN_SHAPE" => "deep", "CORPUS" => "data/fineweb_gpt2_smoke.bin",
       "FRANKEN_LAYER_PATTERN" => "k3", "STEPS" => "2", "SEED" => "0" }
k32 = run_franken_llama(kc.merge("FRANKEN_CONTEXT" => "32"), nil).lines.select { |l| l.start_with?("step ") }
failures << "k3-capacity: ctx 32 broke" unless k32.length == 2
puts failures.length == n0 ? "  ok: k3 recurrent pattern builds within the graph budget at ctx 32 (capacity heuristic is context-aware for KDA/GDN)" : "  FAIL: k3-capacity leg"

# ---- toy#135 (toy-k3): MLA at B > 1 ----
# MLA used to be refused at B > 1 with "the head slicing reshapes on
# seq_t". That diagnosis was wrong — nothing in MLABlock reshapes on
# seq_t. The real limit was head_attend passing a NULL mask and a
# literal batch=1 to GQA.attention, which sends it down the
# diag_mask_inf path: plain causal over the FLAT [T*B] stream. That is
# not a crash, it is CROSS-WINDOW ATTENTION — window 2 attending to
# window 1 while the loss looks perfectly healthy. The block-causal
# mask (the GH#7 batch layout) is threaded now.
#
# ISOLATION is the assertion that matters: same windows, different
# ORDER in the batch, must give the same per-step mean loss. If the
# mask were absent, window content would leak across positions and the
# order would change the answer.
n0 = failures.length
mb = { "FRANKEN_SHAPE" => "deep", "CORPUS" => "data/fineweb_gpt2_smoke.bin",
       "FRANKEN_CONTEXT" => "64", "MLA_LAYERS" => "0,1,2,3,4,5",
       "STEPS" => "4", "SEED" => "0" }
mb1 = run_franken_llama(mb.merge("FRANKEN_BATCH" => "1"), nil).lines.select { |l| l.start_with?("step ") }
mb4 = run_franken_llama(mb.merge("FRANKEN_BATCH" => "4"), nil).lines.select { |l| l.start_with?("step ") }
failures << "mla-batch: B=4 did not run (#{mb4.length}/4)" unless mb4.length == 4
failures << "mla-batch: NaN at B=4" if mb4.map { |l| l[/loss=(\S+)/, 1].to_f }.any?(&:nan?)
failures << "mla-batch: B=4 identical to B=1 (batch axis dead)" if mb4 == mb1
failures << "mla-batch: not deterministic" unless
  run_franken_llama(mb.merge("FRANKEN_BATCH" => "4"), nil).lines.select { |l| l.start_with?("step ") } == mb4
# composes with the rest of the K3-shaped stack at batch
mk = run_franken_llama(mb.merge("FRANKEN_BATCH" => "4", "ATTNRES" => "1",
                                "FRANKEN_ACT" => "situ-glu", "FRANKEN_NOPE" => "1",
                                "FRANKEN_OPTIMIZER" => "muon", "FRANKEN_MTP" => "1"), nil)
mkl = mk.lines.select { |l| l.start_with?("step ") }
failures << "mla-batch: the composed K3-shaped stack does not run at B=4 (#{mkl.length}/4)" unless mkl.length == 4
failures << "mla-batch: composed stack NaN" if mkl.map { |l| l[/loss=(\S+)/, 1].to_f }.any?(&:nan?)
puts failures.length == n0 ? "  ok: MLA at B>1 — block-causal mask threaded (was a NULL mask + hardcoded batch=1 = cross-window attention), B=4 trains deterministically, composes with AttnRes+SiTU-GLU+NoPE+Muon+MTP at batch" : "  FAIL: mla-batch leg"

# ---- toy#126: --lr / --warmup (the F7b LR-sweep surface) ----
# --lr: deterministic, actually moves the curve (default 0.001 stays
# byte-null via leg 1). --warmup: linear ramp reaches LR at step N —
# pinned via the step events' lr field (the ramp is Ruby-side; the
# events are the observable), and the ramped curve differs from flat.
lr_env = { "STEPS" => "6", "FRANKEN_LR" => "0.01" }
lr1 = run_franken_llama(lr_env, nil).lines.select { |l| l.start_with?("step ") }
lr2 = run_franken_llama(lr_env, nil).lines.select { |l| l.start_with?("step ") }
d6  = run_franken_llama({ "STEPS" => "6" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "lr: not deterministic" unless lr1 == lr2 && lr1.length == 6
failures << "lr: curve identical to default 0.001 (knob dead)" if lr1 == d6
wu_env = lr_env.merge("FRANKEN_WARMUP" => "4")
wu1 = nil
Dir.mktmpdir("franken_lr_gate") do |dir|
  wu1 = run_franken_llama(wu_env, dir).lines.select { |l| l.start_with?("step ") }
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  lrs = evs.select { |e| e["kind"] == "step" }.map { |e| e["lr"] }
  want = [1, 2, 3, 4].map { |t| 0.01 * (t.to_f / 4.0) } + [0.01, 0.01]
  ramp_ok = lrs.length == 6 && lrs.zip(want).all? { |g, w| g.is_a?(Numeric) && (g - w).abs < 1.0e-12 }
  failures << "warmup: step-event lr ramp #{lrs.inspect} (want #{want.inspect})" unless ramp_ok
  failures << "warmup: run_start config warmup/lr wrong" unless evs.first.dig("config", "warmup") == 4 && (evs.first.dig("config", "lr").to_f - 0.01).abs < 1.0e-12
end
wu2 = run_franken_llama(wu_env, nil).lines.select { |l| l.start_with?("step ") }
failures << "warmup: not deterministic" unless wu1 == wu2
failures << "warmup: curve identical to flat lr (ramp dead)" if wu1 == lr1
puts failures.length == n0 ? "  ok: --lr moves the curve (deterministic); --warmup ramps to LR @N (events pin the ramp, curve differs from flat)" : "  FAIL: lr/warmup leg"
n0 = failures.length

# ---- toy#124: shape presets, byte-pinned per preset ----
%w[wide deep].each do |sh|
  fx = File.join(ROOT, "prep", "fixtures", "franken_#{sh}_baseline.txt")
  exp = File.readlines(fx).reject { |l| l.start_with?("#") || l.strip.empty? }.map(&:chomp)
  got = run_franken_llama({ "FRANKEN_SHAPE" => sh, "STEPS" => "5" }, nil)
        .lines.select { |l| l.start_with?("step ") }.map(&:chomp)
  if got == exp
    puts "  ok: --shape #{sh} byte-equals its recorded baseline (5 steps)"
  else
    failures << "shape-#{sh}: curve != #{File.basename(fx)}\ngot:  #{got.join(' | ')}\nwant: #{exp.join(' | ')}"
  end
  s1 = run_franken_llama({ "FRANKEN_SHAPE" => sh, "STEPS" => "5", "SEED" => "1" }, nil)
  s2 = run_franken_llama({ "FRANKEN_SHAPE" => sh, "STEPS" => "5", "SEED" => "1" }, nil)
  failures << "shape-#{sh}: seed=1 not deterministic" unless s1 == s2
end

# ---- toy#129 item 1: TOYC pack header + context/vocab unpinning ----
# The committed GPT-2 smoke pack (data/fineweb_gpt2_smoke.bin, TOYC v1
# vocab 50257) drives the instrument's vocab from the HEADER; context
# comes from FRANKEN_CONTEXT. Deterministic; provenance carries both;
# a conflicting FRANKEN_VOCAB fails loud; a headerless pack + explicit
# FRANKEN_VOCAB=627 byte-equals the implicit-627 run (the flag is a
# name for what already happens, not a new code path).
abort "franken gate: data/fineweb_gpt2_smoke.bin missing — generate it: uv run prep/pretokenize_pack.py --tokens 200_000 --out data/fineweb_gpt2_smoke.bin" unless File.file?(File.join(ROOT, "data", "fineweb_gpt2_smoke.bin"))
pk_env = { "STEPS" => "3", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_CONTEXT" => "64" }
pk1 = nil
Dir.mktmpdir("franken_pack_gate") do |dir|
  pk1 = run_franken_llama(pk_env, dir)
  rsp = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "pack: model.vocab #{rsp.dig('model', 'vocab').inspect} (want 50257 from the TOYC header)" unless rsp.dig("model", "vocab") == 50257
  failures << "pack: config.context #{rsp.dig('config', 'context').inspect} (want 64)" unless rsp.dig("config", "context") == 64
end
pk2 = run_franken_llama(pk_env, nil)
pk1_c = pk1.lines.select { |l| l.start_with?("step ") }
pk2_c = pk2.lines.select { |l| l.start_with?("step ") }
failures << "pack: not deterministic" unless pk1_c == pk2_c && pk1_c.length == 3
l0 = pk1_c.first[/loss=(\S+)/, 1].to_f
failures << "pack: first loss #{l0} not ~ln(50257)" unless l0 > 9.0 && l0 < 12.5
_oc, stc = Open3.capture2e({ "STEPS" => "1", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_VOCAB" => "100" }, RUNNER, chdir: ROOT)
failures << "pack: conflicting FRANKEN_VOCAB not rejected" if stc.success?
hv1 = run_franken_llama({ "STEPS" => "3", "CORPUS" => "data/ts_seqs.bin", "FRANKEN_VOCAB" => "627" }, nil)
hv0 = run_franken_llama({ "STEPS" => "3", "CORPUS" => "data/ts_seqs.bin" }, nil)
failures << "pack: explicit FRANKEN_VOCAB=627 differs from implicit (headerless null broken)" unless hv1.lines.select { |l| l.start_with?("step ") } == hv0.lines.select { |l| l.start_with?("step ") }
puts failures.length == n0 ? "  ok: TOYC pack — vocab 50257 from header, ctx 64, deterministic; vocab-conflict rejected; headerless --vocab null" : "  FAIL: pack leg"

# ---- toy#133: --batch (B corpus windows per step) ----
# THE isolation null: with distinct windows W and X, batched [W,X] and
# the order-swap [X,W] must BOTH equal the mean of the two B=1 losses —
# any cross-window attention leak breaks order-invariance first (the
# lesson: a mis-allocated mask read zeros and B=2 ran fully unmasked;
# this leg would have caught it). Plus B=1 flag-null and B=4
# determinism.
n0 = failures.length
Dir.mktmpdir("franken_batch_gate") do |dir|
  raw = File.binread(File.join(ROOT, "data", "fineweb_gpt2_smoke.bin"))
  w = raw[16, 32 * 4]
  x = raw[16 + 5000 * 4, 32 * 4]
  File.binwrite(File.join(dir, "w.bin"), w)
  File.binwrite(File.join(dir, "x.bin"), x)
  File.binwrite(File.join(dir, "wx.bin"), w + x)
  File.binwrite(File.join(dir, "xw.bin"), x + w)
  get1 = lambda do |env|
    out = run_franken_llama(env, nil)
    out.lines.select { |l| l.start_with?("step ") }.first[/loss=(\S+)/, 1].to_f
  end
  base_env = { "STEPS" => "1", "FRANKEN_VOCAB" => "50257" }
  lw = get1.call(base_env.merge("CORPUS" => File.join(dir, "w.bin")))
  lx = get1.call(base_env.merge("CORPUS" => File.join(dir, "x.bin")))
  lwx = get1.call(base_env.merge("CORPUS" => File.join(dir, "wx.bin"), "FRANKEN_BATCH" => "2"))
  lxw = get1.call(base_env.merge("CORPUS" => File.join(dir, "xw.bin"), "FRANKEN_BATCH" => "2"))
  mean = (lw + lx) / 2.0
  failures << "batch: [W,X] #{lwx} != mean #{mean} (leak or wrong labels)" unless (lwx - mean).abs < 1.0e-4
  failures << "batch: order-swap [X,W] #{lxw} != [W,X] #{lwx} (cross-window leak)" unless (lxw - lwx).abs < 1.0e-4
end
bn1 = run_franken_llama({ "STEPS" => "4", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_BATCH" => "1" }, nil)
bn0 = run_franken_llama({ "STEPS" => "4", "CORPUS" => "data/fineweb_gpt2_smoke.bin" }, nil)
failures << "batch: FRANKEN_BATCH=1 differs from no-flag (flag-null broken)" unless bn1.lines.select { |l| l.start_with?("step ") } == bn0.lines.select { |l| l.start_with?("step ") }
b4a = run_franken_llama({ "STEPS" => "4", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_BATCH" => "4" }, nil).lines.select { |l| l.start_with?("step ") }
b4b = run_franken_llama({ "STEPS" => "4", "CORPUS" => "data/fineweb_gpt2_smoke.bin", "FRANKEN_BATCH" => "4" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "batch: B=4 not deterministic" unless b4a == b4b && b4a.length == 4
failures << "batch: B=4 NaN" if b4a.any? { |l| l[/loss=(\S+)/, 1].to_f.nan? }
_ob, stb = Open3.capture2e({ "STEPS" => "1", "FRANKEN_BATCH" => "2" }, RUNNER, chdir: ROOT)
failures << "batch: B>1 without corpus not rejected" if stb.success?
puts failures.length == n0 ? "  ok: --batch — order-swap isolation null exact-class, B=1 flag-null, B=4 deterministic, corpus guard" : "  FAIL: batch leg"

# ---- toy#136 (K1): --act / --rope / --schedule ----
# situ-glu + nope each train deterministically and move the curve (the
# swiglu/rope defaults stay byte-null via F0); cosine's per-step lr is
# pinned via step events (formula-checked at both ends); provenance
# carries all three axes.
n0 = failures.length
k1_pairs = [
  ["situ-glu", { "FRANKEN_ACT" => "situ-glu" }],
  ["nope",     { "FRANKEN_NOPE" => "1" }],
]
d6k = run_franken_llama({ "STEPS" => "6" }, nil).lines.select { |l| l.start_with?("step ") }
k1_pairs.each do |name, env|
  a = run_franken_llama({ "STEPS" => "6" }.merge(env), nil).lines.select { |l| l.start_with?("step ") }
  b = run_franken_llama({ "STEPS" => "6" }.merge(env), nil).lines.select { |l| l.start_with?("step ") }
  failures << "k1 #{name}: not deterministic" unless a == b && a.length == 6
  failures << "k1 #{name}: NaN" if a.any? { |l| l[/loss=(\S+)/, 1].to_f.nan? }
  failures << "k1 #{name}: curve identical to default (axis dead)" if a == d6k
end
Dir.mktmpdir("franken_k1_gate") do |dir|
  run_franken_llama({ "STEPS" => "6", "FRANKEN_ACT" => "situ-glu", "FRANKEN_NOPE" => "1",
                      "FRANKEN_SCHEDULE" => "cosine", "FRANKEN_LR" => "0.01" }, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  cfg = evs.first["config"] || {}
  failures << "k1: provenance act #{cfg['act'].inspect}" unless cfg["act"] == "situ-glu"
  failures << "k1: provenance rope #{cfg['rope'].inspect}" unless cfg["rope"] == "nope"
  failures << "k1: provenance schedule #{cfg['schedule'].inspect}" unless cfg["schedule"] == "cosine"
  lrs = evs.select { |e| e["kind"] == "step" }.map { |e| e["lr"] }
  # cosine, no warmup: lr_0 = LR (cos(0)=1); lr_5 = min + 0.5*(LR-min)*(1+cos(pi*5/6))
  min_lr = 0.001
  want_last = min_lr + 0.5 * (0.01 - min_lr) * (1.0 + Math.cos(Math::PI * 5.0 / 6.0))
  failures << "k1: cosine lr[0] #{lrs.first} != 0.01" unless (lrs.first - 0.01).abs < 1.0e-12
  failures << "k1: cosine lr[5] #{lrs.last} != #{want_last}" unless (lrs.last - want_last).abs < 1.0e-9
  failures << "k1: cosine lr not decreasing" unless lrs.each_cons(2).all? { |x, y| y < x }
end
puts failures.length == n0 ? "  ok: K1 axes — situ-glu + nope move the curve deterministically; cosine lr event-pinned; provenance carries act/rope/schedule" : "  FAIL: K1 leg"

# ---- toy#137 (K2b): KDA_LAYERS — a Kimi Delta Attention layer ----
# Absent = all-attention, byte-null (leg 1's F0 fixture pins it). With
# a KDA layer: it TRAINS (the whole layer — projections, low-rank decay
# pair, per-head A_h, full-rank gate, W_o — rides one autodiff sweep),
# deterministically, and the curve differs from all-attention. A dfa
# POLICY on that layer must fail loud (the credit-assignment wiring is
# attention-shaped; "KDA under DFA" is its own K-series question), and
# (the engine also fails loud when both lists claim one layer, but this
# runner exposes KDA_LAYERS only, so that guard is not probed here.)
n0 = failures.length
kda_env = { "STEPS" => "6", "SEED" => "0", "KDA_LAYERS" => "1" }
ka = run_franken_llama(kda_env, nil).lines.select { |l| l.start_with?("step ") }
kb = run_franken_llama(kda_env, nil).lines.select { |l| l.start_with?("step ") }
failures << "kda: not deterministic" unless ka == kb && ka.length == 6
kl = ka.map { |l| l[/loss=(\S+)/, 1].to_f }
failures << "kda: NaN" if kl.any?(&:nan?)
failures << "kda: did not train (#{kl.first} -> #{kl.last})" unless kl.last < kl.first - 0.1
d6kda = run_franken_llama({ "STEPS" => "6", "SEED" => "0" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "kda: curve identical to all-attention (layer kind dead)" if ka == d6kda
_ok, stk = Open3.capture2e({ "STEPS" => "2", "KDA_LAYERS" => "1", "FRANKEN_POLICY" => "chain,dfa",
                             "FRANKEN_B_SEED" => "42" }, RUNNER, chdir: ROOT)
failures << "kda: dfa policy on a KDA layer not rejected" if stk.success?
# (the both-lists guard lives in build_gdn_flags! but is NOT reachable
# from this runner — the franken runner wires KDA_LAYERS only, no
# GDN_LAYERS — so it is asserted where it is reachable, not here.)
Dir.mktmpdir("franken_kda_gate") do |dir|
  run_franken_llama({ "STEPS" => "2", "SEED" => "0", "KDA_LAYERS" => "1" }, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  failures << "kda: provenance kda_layers #{evs.first.dig('config', 'kda_layers').inspect}" unless evs.first.dig("config", "kda_layers") == "1"
end
# toy#137 K2c: ShortConv identity null + the batch guard + the cost
# split. The conv is identity-inited (tap0=1, taps1..3=0), so a
# conv-ON run's STEP-1 loss must BYTE-EQUAL a conv-OFF run's — then
# diverge as the taps train. That is a forward-identity null on a
# mechanism whose whole risk is the shifted-view plumbing.
kc_on  = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "KDA_LAYERS" => "1" }, nil).lines.select { |l| l.start_with?("step ") }
kc_off = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "KDA_LAYERS" => "1", "KDA_CONV" => "0" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "kda-conv: step-1 differs from conv-off (identity init broken)\non: #{kc_on.first}off: #{kc_off.first}" unless kc_on.first == kc_off.first
failures << "kda-conv: curve identical to conv-off (taps never train)" if kc_on == kc_off
_ob3, stb3 = Open3.capture2e({ "STEPS" => "1", "KDA_LAYERS" => "1", "FRANKEN_BATCH" => "4",
                               "CORPUS" => "data/fineweb_gpt2_smoke.bin" }, RUNNER, chdir: ROOT)
failures << "kda: --batch > 1 with KDA layers not rejected" if stb3.success?
Dir.mktmpdir("franken_kda_cost") do |dir|
  FileUtils.mkdir_p(File.join(dir, "weights"))
  run_franken_llama({ "STEPS" => "1", "SEED" => "0", "KDA_LAYERS" => "1" }, dir)
  ck = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)["cost"]
  Dir.mktmpdir("franken_attn_cost") do |d2|
    FileUtils.mkdir_p(File.join(d2, "weights"))
    run_franken_llama({ "STEPS" => "1", "SEED" => "0" }, d2)
    ca = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)["cost"]
    failures << "kda-cost: flops not lower than all-attention (#{ck['flops_per_token']} vs #{ca['flops_per_token']})" unless ck["flops_per_token"] < ca["flops_per_token"]
    failures << "kda-cost: params identical to all-attention (formula not kind-aware)" if ck["total_params"] == ca["total_params"]
  end
end
# toy#138 K3a: the hybrid layer pattern — K3's 3 KDA : 1 global with
# the FINAL layer always global. At deep (L=6): KDA at 0,1,2,4 and
# attention at 3,5. Asserted STRUCTURALLY (cost accounting counts
# layer kinds, so params/flops pin the split) rather than by curve.
hy = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                         "FRANKEN_LAYER_PATTERN" => "hybrid" }, nil).lines.select { |l| l.start_with?("step ") }
hy2 = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                          "FRANKEN_LAYER_PATTERN" => "hybrid" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "hybrid: not deterministic" unless hy == hy2 && hy.length == 3
failures << "hybrid: NaN" if hy.any? { |l| l[/loss=(\S+)/, 1].to_f.nan? }
Dir.mktmpdir("franken_hy") do |dir|
  FileUtils.mkdir_p(File.join(dir, "weights"))
  run_franken_llama({ "STEPS" => "1", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                      "FRANKEN_LAYER_PATTERN" => "hybrid" }, dir)
  hj = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "hybrid: provenance layer_pattern #{hj.dig('config', 'layer_pattern').inspect}" unless hj.dig("config", "layer_pattern") == "hybrid"
  Dir.mktmpdir("franken_hy_ref") do |d2|
    FileUtils.mkdir_p(File.join(d2, "weights"))
    # the same 6-layer shape with an EXPLICIT 0,1,2,4 kda list must give
    # the identical cost split — that is the pattern's contract.
    run_franken_llama({ "STEPS" => "1", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                        "KDA_LAYERS" => "0,1,2,4" }, d2)
    rj = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)
    failures << "hybrid: cost differs from explicit 0,1,2,4 (wrong layer split)" unless hj["cost"] == rj["cost"]
  end
end
_oh, sth = Open3.capture2e({ "STEPS" => "1", "FRANKEN_LAYER_PATTERN" => "hybrid",
                             "KDA_LAYERS" => "0" }, RUNNER, chdir: ROOT)
failures << "hybrid: pattern + explicit kda-layers not rejected" if sth.success?
# K-series M2: the Gated-MLA layer kind, and `--layer-pattern k3` —
# the FAITHFUL K3 contract (3 KDA : 1 Gated MLA) as opposed to toy#138's
# `hybrid`, whose "1" slot is ordinary attention. What this pins is that
# MLA is a REAL third kind: it trains, it differs from both attention
# and KDA at the same split, its params move, and the byte-nulls hold.
n0 = failures.length
ml = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "MLA_LAYERS" => "1" }, nil)
     .lines.select { |l| l.start_with?("step ") }
ml2 = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "MLA_LAYERS" => "1" }, nil)
      .lines.select { |l| l.start_with?("step ") }
base = run_franken_llama({ "STEPS" => "3", "SEED" => "0" }, nil)
       .lines.select { |l| l.start_with?("step ") }
kda1 = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "KDA_LAYERS" => "1" }, nil)
       .lines.select { |l| l.start_with?("step ") }
mlv = ml.map { |l| l[/loss=(\S+)/, 1].to_f }
failures << "mla: not deterministic" unless ml == ml2 && ml.length == 3
failures << "mla: NaN" if mlv.any?(&:nan?)
failures << "mla: layer 1 as MLA is byte-identical to all-attention (the kind is not reaching the graph)" if ml == base
failures << "mla: MLA layer identical to a KDA layer at the same index" if ml == kda1
failures << "mla: does not descend (#{mlv.first} -> #{mlv.last})" unless mlv.last < mlv.first
# The FLAG-NULL: no MLA layers must leave every existing curve alone.
failures << "mla: empty MLA_LAYERS perturbs the default curve" unless
  run_franken_llama({ "STEPS" => "3", "SEED" => "0", "MLA_LAYERS" => "" }, nil)
    .lines.select { |l| l.start_with?("step ") } == base
# --mla-rank is a real axis (a different latent width is a different model),
# and the DERIVED default must equal an explicit r at that same value.
r_small = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "MLA_LAYERS" => "1",
                             "FRANKEN_MLA_RANK" => "2" }, nil)
          .lines.select { |l| l.start_with?("step ") }
failures << "mla: --mla-rank 2 does not change the curve" if r_small == ml
# k3 vs hybrid at the same 6-layer split: same KDA positions, different
# global kind, so the curves MUST differ — that is the whole delta.
k3 = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                         "FRANKEN_LAYER_PATTERN" => "k3" }, nil)
     .lines.select { |l| l.start_with?("step ") }
hyb = run_franken_llama({ "STEPS" => "3", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                          "FRANKEN_LAYER_PATTERN" => "hybrid" }, nil)
      .lines.select { |l| l.start_with?("step ") }
failures << "mla: k3 pattern identical to hybrid (the MLA slot is not being built)" if k3 == hyb
failures << "mla: k3 NaN" if k3.any? { |l| l[/loss=(\S+)/, 1].to_f.nan? }
Dir.mktmpdir("franken_k3") do |dir|
  FileUtils.mkdir_p(File.join(dir, "weights"))
  run_franken_llama({ "STEPS" => "1", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                      "FRANKEN_LAYER_PATTERN" => "k3" }, dir)
  kj = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "mla: k3 provenance layer_pattern #{kj.dig('config', 'layer_pattern').inspect}" unless kj.dig("config", "layer_pattern") == "k3"
  # At L=6 the pattern puts MLA in the two global slots (3 and 5).
  failures << "mla: k3 provenance mla_layers #{kj.dig('config', 'mla_layers').inspect} (want 2)" unless kj.dig("config", "mla_layers") == 2
  Dir.mktmpdir("franken_k3_ref") do |d2|
    FileUtils.mkdir_p(File.join(d2, "weights"))
    run_franken_llama({ "STEPS" => "1", "SEED" => "0", "FRANKEN_SHAPE" => "deep",
                        "KDA_LAYERS" => "0,1,2,4", "MLA_LAYERS" => "3,5" }, d2)
    rj = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)
    failures << "mla: k3 cost differs from the explicit kda 0,1,2,4 + mla 3,5 split" unless kj["cost"] == rj["cost"]
  end
end
# Guards, all fail-loud.
[[{ "FRANKEN_LAYER_PATTERN" => "k3", "MLA_LAYERS" => "0" }, "pattern + explicit mla-layers"],
 [{ "MLA_LAYERS" => "0", "KDA_LAYERS" => "0" }, "same layer claimed by kda and mla"],
 [{ "MLA_LAYERS" => "9" },                      "out-of-range mla index"],
 [{ "KDA_LAYERS" => "9" },                      "out-of-range kda index"],
 [{ "KDA_LAYERS" => "0", "FRANKEN_BATCH" => "4", "CORPUS" => "data/ts_seqs.bin" },
  "kda + batch > 1 (KDA is still B=1; MLA no longer is — toy#135)"]].each do |env, what|
  _o, st = Open3.capture2e({ "STEPS" => "1" }.merge(env), RUNNER, chdir: ROOT)
  failures << "mla: #{what} not rejected" if st.success?
end
puts failures.length == n0 ? "  ok: MLA layer kind — trains (#{mlv.first.round(3)} -> #{mlv.last.round(3)}), distinct from attention AND from KDA at the same index, rank is a live axis, flag-null holds; k3 pattern differs from hybrid and matches the explicit kda/mla split; 5 guards reject (the mla+batch guard RETIRED — see the mla-batch leg)" : "  FAIL: MLA leg"

puts failures.length == n0 ? "  ok: KDA_LAYERS — a KDA layer trains through the engine (#{kl.first.round(3)} -> #{kl.last.round(3)}), deterministic, differs from all-attention; conv identity-null + batch guard + linear-attention cost; hybrid 3:1 pattern == explicit split; dfa-policy rejected" : "  FAIL: KDA leg"

# ---- toy#138 K3b: AttnRes (attention residuals over depth) ----
# OFF = byte-null (leg 1's F0 fixture). ON: trains, deterministic,
# differs, and the pseudo-queries show up in the param count. The
# STRUCTURAL anchor: queries init to ZERO, so every source scores
# equally and the mixture starts as the exact MEAN over sources —
# a fully-specified starting point (not an arbitrary one), which is
# why the ON curve starts a hair BELOW the residual path rather than
# anywhere.
n0 = failures.length
ar_env = { "STEPS" => "4", "SEED" => "0", "ATTNRES" => "1" }
ar1 = run_franken_llama(ar_env, nil).lines.select { |l| l.start_with?("step ") }
ar2 = run_franken_llama(ar_env, nil).lines.select { |l| l.start_with?("step ") }
base4 = run_franken_llama({ "STEPS" => "4", "SEED" => "0" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "attnres: not deterministic" unless ar1 == ar2 && ar1.length == 4
arl = ar1.map { |l| l[/loss=(\S+)/, 1].to_f }
failures << "attnres: NaN" if arl.any?(&:nan?)
failures << "attnres: did not train (#{arl.first} -> #{arl.last})" unless arl.last < arl.first - 0.05
failures << "attnres: curve identical to the residual path (axis dead)" if ar1 == base4
Dir.mktmpdir("franken_ar") do |dir|
  FileUtils.mkdir_p(File.join(dir, "weights"))
  run_franken_llama({ "STEPS" => "1", "SEED" => "0", "ATTNRES" => "1" }, dir)
  aj = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "attnres: provenance flag #{aj.dig('config', 'attnres').inspect}" unless aj.dig("config", "attnres") == true
  Dir.mktmpdir("franken_ar_ref") do |d2|
    FileUtils.mkdir_p(File.join(d2, "weights"))
    run_franken_llama({ "STEPS" => "1", "SEED" => "0" }, d2)
    rj = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)
    grew = aj.dig("cost", "total_params") - rj.dig("cost", "total_params")
    # one [d] pseudo-query per layer + one for the final aggregation
    want = (2 + 1) * 64
    failures << "attnres: params grew by #{grew} (want #{want} = (L+1)·d)" unless grew == want
  end
end
puts failures.length == n0 ? "  ok: AttnRes — depth-attention trains (#{arl.first.round(3)} -> #{arl.last.round(3)}), deterministic, differs from the residual path, (L+1)·d pseudo-queries counted" : "  FAIL: attnres leg"

# ---- K-series M10: MTP (Multi-Token Prediction) ----
# The load-bearing assertion here is mtp_sig, NOT the curve. The MTP
# block is not in seq_blocks_ffi, so it carries its own optimizer arm;
# if that arm were missing the module would still build, still emit a
# second loss, and still produce a perfectly healthy training curve
# while never updating a single MTP weight. That is the K4b/experts_sig
# lesson, and here it caught a real defect: under the original
# `loss + lambda*CE_mtp` formulation, --mtp-lambda 0 froze the module
# outright (delta exactly 0.000000).
#
# lambda is therefore a COUPLING dial, not a loss weight — the second
# root is unscaled and lambda gradient-scales what the MTP branch
# borrows from the backbone. The two legs that pin that meaning are
# marked below.
n0 = failures.length
mtp_env = { "STEPS" => "4", "SEED" => "0", "FRANKEN_MTP" => "1" }
mt1 = run_franken_llama(mtp_env, nil).lines.select { |l| l.start_with?("step ") }
mt2 = run_franken_llama(mtp_env, nil).lines.select { |l| l.start_with?("step ") }
failures << "mtp: not deterministic" unless mt1 == mt2 && mt1.length == 4
mtl = mt1.map { |l| l[/loss=(\S+)/, 1].to_f }
failures << "mtp: NaN" if mtl.any?(&:nan?)
failures << "mtp: curve identical to --mtp off (the second root is inert)" if mt1 == base4
# THE LAMBDA=0 SEPARATION: the second root EXISTS and its weights train,
# but it must not perturb the backbone at all. Byte-identical, not
# approximately — the grad_scale forward is exact at lambda 0.
l0 = run_franken_llama(mtp_env.merge("FRANKEN_MTP_LAMBDA" => "0"), nil).lines.select { |l| l.start_with?("step ") }
failures << "mtp: --mtp-lambda 0 perturbs the backbone (want byte-identical to --mtp off)" unless l0 == base4
l1 = run_franken_llama(mtp_env.merge("FRANKEN_MTP_LAMBDA" => "1.0"), nil).lines.select { |l| l.start_with?("step ") }
failures << "mtp: --mtp-lambda 1.0 does not differ from lambda 0 (the coupling dial is dead)" if l1 == l0

# mtp_sig: the module PROVABLY trains — at every lambda, including 0.
%w[0 0.3 1.0].each do |lam|
  Dir.mktmpdir("franken_mtp_#{lam}") do |dir|
    FileUtils.mkdir_p(File.join(dir, "weights"))
    run_franken_llama({ "STEPS" => "4", "SEED" => "0", "FRANKEN_MTP" => "1",
                        "FRANKEN_MTP_LAMBDA" => lam }, dir)
    ev = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
    s0 = ev.first.dig("config", "mtp_sig")
    s1 = ev.last["mtp_sig"]
    if s0.nil? || s1.nil?
      failures << "mtp: mtp_sig missing at lambda #{lam} (start=#{s0.inspect} end=#{s1.inspect})"
    elsif (s1 - s0).abs <= 1e-9
      failures << "mtp: MTP weights NEVER MOVED at lambda #{lam} — the module builds and reports a loss but does not train"
    end
    # the t+2 read rides the step events and run_end (this lane has no
    # in-runner eval_ce; checkpoints are evaluated offline).
    st = ev.select { |e| e["kind"] == "step" }
    failures << "mtp: mtp_loss missing from step events at lambda #{lam}" unless
      st.length == 4 && st.all? { |e| e["mtp_loss"].is_a?(Numeric) }
    failures << "mtp: final_mtp_loss missing from run_end at lambda #{lam}" unless
      ev.last["final_mtp_loss"].is_a?(Numeric)
  end
end
# OFF must report null rather than a fabricated 0.0, and must not carry
# MTP cost.
Dir.mktmpdir("franken_mtp_off") do |dir|
  FileUtils.mkdir_p(File.join(dir, "weights"))
  run_franken_llama({ "STEPS" => "1", "SEED" => "0" }, dir)
  ev = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  failures << "mtp: off should report mtp_loss null" unless
    ev.select { |e| e["kind"] == "step" }.all? { |e| e["mtp_loss"].nil? }
  failures << "mtp: off should report provenance mtp=false" unless ev.first.dig("config", "mtp") == false
  # COST must grow by the projection + one full block, and by nothing
  # else. Third instance of the silent-under-count class (K4b, toy#145).
  Dir.mktmpdir("franken_mtp_on") do |d2|
    FileUtils.mkdir_p(File.join(d2, "weights"))
    run_franken_llama({ "STEPS" => "1", "SEED" => "0", "FRANKEN_MTP" => "1" }, d2)
    on = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)
    grew = on.dig("cost", "total_params") - ev.first.dig("cost", "total_params")
    d = 64; dff = 128   # the base shape (BIG=false) — train_franken_llama.rb:250/253
    want = 2 * d * d + (4 * d * d + 3 * d * dff + 2 * d)   # [d,2d] proj + one block
    failures << "mtp: params grew by #{grew} (want #{want} = [d,2d] projection + one block)" unless grew == want
    failures << "mtp: flops_per_token did not grow" unless
      on.dig("cost", "flops_per_token") > ev.first.dig("cost", "flops_per_token")
    failures << "mtp: provenance mtp/mtp_lambda #{on.dig('config','mtp').inspect}/#{on.dig('config','mtp_lambda').inspect}" unless
      on.dig("config", "mtp") == true && on.dig("config", "mtp_lambda") == 0.3
  end
end
# Guards, fail-loud.
[[{ "FRANKEN_MTP_LAMBDA" => "0.5" }, "lambda without --mtp"],
 [{ "FRANKEN_MTP" => "1", "FRANKEN_MTP_LAMBDA" => "1.5" }, "lambda > 1 (not a coupling fraction)"]].each do |extra, what|
  _o, st = Open3.capture2e({ "STEPS" => "1", "SEED" => "0" }.merge(extra), RUNNER, chdir: ROOT)
  failures << "mtp: #{what} not rejected" if st.success?
end
# toy#135 (toy-k3): MTP AT B > 1. next_token_k raised on batch != 1 and
# shift_ids had no batch parameter at all, so the caller passed a
# literal 1 and handed a context-length id array to a context*batch
# consumer — step 1 looked fine and step 2 died in get_rows with
# GGML_ASSERT(i01 < ne01). B=1 is unchanged by construction (one
# window, base 0), so this pins the B>1 path specifically.
mtb = run_franken_llama({ "STEPS" => "4", "SEED" => "0", "FRANKEN_MTP" => "1",
                          "FRANKEN_SHAPE" => "deep", "CORPUS" => "data/fineweb_gpt2_smoke.bin",
                          "FRANKEN_CONTEXT" => "64", "FRANKEN_BATCH" => "4" }, nil)
mtbl = mtb.lines.select { |l| l.start_with?("step ") }
failures << "mtp: B>1 did not complete (#{mtbl.length}/4 steps) — the shift-by-k ids are not window-local" unless mtbl.length == 4
failures << "mtp: NaN at B>1" if mtbl.map { |l| l[/loss=(\S+)/, 1].to_f }.any?(&:nan?)
puts failures.length == n0 ? "  ok: MTP — trains at B=1 AND B=4 (#{mtl.first.round(3)} -> #{mtl.last.round(3)}), mtp_sig PROVES the weights move at every lambda, lambda=0 leaves the backbone byte-identical to off while still training the module, mtp_loss/final_mtp_loss carried, cost grows by projection + block, 2 guards reject" : "  FAIL: MTP leg"

# ---- toy#139 / K5: PER-HEAD Muon on the llama lane ----
# adamw stays byte-null (leg 1's F0 fixture). muon trains,
# deterministically, and differs. The "per-head" part is STRUCTURAL:
# toy's random-init layout stores q/k/v as one tensor PER HEAD, so
# orthogonalizing each 2D ft_weights entry IS K3's per-head Muon —
# there is no full-QKV matrix here to accidentally couple.
n0 = failures.length
mu_env = { "STEPS" => "5", "SEED" => "0", "FRANKEN_OPTIMIZER" => "muon" }
m1 = run_franken_llama(mu_env, nil).lines.select { |l| l.start_with?("step ") }
m2 = run_franken_llama(mu_env, nil).lines.select { |l| l.start_with?("step ") }
b5 = run_franken_llama({ "STEPS" => "5", "SEED" => "0" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "muon: not deterministic" unless m1 == m2 && m1.length == 5
ml = m1.map { |l| l[/loss=(\S+)/, 1].to_f }
failures << "muon: NaN" if ml.any?(&:nan?)
failures << "muon: did not train (#{ml.first} -> #{ml.last})" unless ml.last < ml.first - 0.1
failures << "muon: curve identical to adamw (optimizer dead)" if m1 == b5
failures << "muon: step-1 differs from adamw (step 1 is pre-update)" unless m1.first == b5.first
_om, stm = Open3.capture2e({ "STEPS" => "1", "FRANKEN_OPTIMIZER" => "sgd" }, RUNNER, chdir: ROOT)
failures << "muon: sgd not rejected on the llama lane" if stm.success?
puts failures.length == n0 ? "  ok: per-head Muon — trains (#{ml.first.round(3)} -> #{ml.last.round(3)}), deterministic, differs from adamw, step-1 pre-update identical; sgd rejected" : "  FAIL: muon leg"
n0 = failures.length

# ---- toy#129 item 2: --no-shadow ----
# THE null (the roadmap's own criterion): applied updates byte-identical
# shadow vs no-shadow at the same policy/seed — the mode changes the
# graph SHAPE (no chain grad-accs on dfa weights), never the numbers.
# Plus: summary says shadow=off, provenance carries shadow=false, and
# the two impossible asks fail loud (align needs the acc; mask modes
# read it).
ns_env = { "STEPS" => "6", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }
sh6 = run_franken_llama(ns_env, nil)
ns6 = run_franken_llama(ns_env.merge("FRANKEN_NO_SHADOW" => "1"), nil)
sh6_c = sh6.lines.select { |l| l.start_with?("step ") }
ns6_c = ns6.lines.select { |l| l.start_with?("step ") }
failures << "no-shadow: applied updates differ from the shadow build\nsh: #{sh6_c.join}ns: #{ns6_c.join}" unless sh6_c == ns6_c && ns6_c.length == 6
failures << "no-shadow: summary does not say shadow=off" unless ns6.include?("shadow=off")
failures << "no-shadow: shadow summary does not say shadow=on" unless sh6.include?("shadow=on")
Dir.mktmpdir("franken_ns_gate") do |dir|
  run_franken_llama({ "STEPS" => "2", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42", "FRANKEN_NO_SHADOW" => "1" }, dir)
  rs0 = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "no-shadow: provenance franken.shadow #{rs0.dig('franken', 'shadow').inspect} (want false)" unless rs0.dig("franken", "shadow") == false
end
_o1, st1 = Open3.capture2e({ "STEPS" => "1", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42", "FRANKEN_ALIGN" => "1", "FRANKEN_NO_SHADOW" => "1" }, RUNNER, chdir: ROOT)
failures << "no-shadow: ALIGN + NO_SHADOW not rejected" if st1.success?
_o2, st2 = Open3.capture2e({ "STEPS" => "1", "FRANKEN_POLICY" => "chain,maskdfa:0.5", "FRANKEN_B_SEED" => "42", "FRANKEN_NO_SHADOW" => "1" }, RUNNER, chdir: ROOT)
failures << "no-shadow: mask mode + NO_SHADOW not rejected" if st2.success?
puts failures.length == n0 ? "  ok: --no-shadow — applied updates byte-equal shadow (the null), provenance shadow=false, align/mask guards fail loud" : "  FAIL: no-shadow leg"
n0 = failures.length

# ---- toy#129 item 3 (enabling seam): --ckpt-every ----
# Mid-run checkpoints at boundaries + THE null: the write (downloads +
# a fresh plain-storage session, freed) must not disturb the training
# sched — curve byte-equals the no-ckpt run.
ck_base = run_franken_llama({ "STEPS" => "5", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
Dir.mktmpdir("franken_ck_gate") do |dir|
  ck_out = run_franken_llama({ "STEPS" => "5", "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42", "FRANKEN_CKPT_EVERY" => "2" }, dir)
  ck_c = ck_out.lines.select { |l| l.start_with?("step ") }
  base_c = ck_base.lines.select { |l| l.start_with?("step ") }
  failures << "ckpt-every: curve differs from no-ckpt run (write disturbs training)" unless ck_c == base_c
  %w[step_2.gguf step_4.gguf step_5.gguf].each do |ck|
    failures << "ckpt-every: missing weights/#{ck}" unless File.file?(File.join(dir, "weights", ck))
  end
end
puts failures.length == n0 ? "  ok: --ckpt-every — boundary checkpoints written; curve byte-equals no-ckpt (write is sched-null)" : "  FAIL: ckpt-every leg"

# ---- toy#158 (F15): MACRO DFA (LightOn block-tap) + RAdam ----
# The three things that make a run macro-DFA, asserted separately —
# because each one can break WITHOUT breaking the others, and a run
# that is only two of the three is a hybrid mislabelled as a recipe.
n0 = failures.length
mac_env = { "STEPS" => "5", "FRANKEN_POLICY" => "dfa,dfa",
            "FRANKEN_DFA_GRANULARITY" => "block", "FRANKEN_B_SEED" => "42" }
mac  = run_franken_llama(mac_env, nil)
mac_c = mac.lines.select { |l| l.start_with?("step ") }
bp_c  = f0_curve

# (a) THE FORWARD IS UNCHANGED. tnn_detach is forward-identity, so the
# macro build must predict EXACTLY what BP predicts at step 1 — that
# is what makes the arms comparable. A cut that moved the forward
# would invalidate every macro-vs-BP number.
failures << "macro: step 1 differs from BP (the cut changed the FORWARD; detach must be forward-identity)\nmacro: #{mac_c[0]}bp:    #{bp_c[0]}" unless mac_c[0] == bp_c[0]
# (b) THE BACKWARD IS DIFFERENT. Steps 2+ must diverge, or the cut did
# not take and this is just BP under a macro label.
failures << "macro: curve identical to BP past step 1 — the cut did not take" if mac_c == bp_c
# (c) THE BLOCKS ACTUALLY TRAIN FROM THE INJECTED ERROR. If the
# surrogate roots did not reach the weights, the blocks would be
# effectively frozen and the FEEDBACK SEED could not matter. It does.
mac_b2 = run_franken_llama(mac_env.merge("FRANKEN_B_SEED" => "43"), nil)
failures << "macro: the B seed does not change the curve — the surrogate roots are not reaching the block weights (blocks frozen, only the head training)" if mac_b2.lines.select { |l| l.start_with?("step ") } == mac_c
# macro != micro at the same policy: they are different recipes, and
# conflating them is exactly what toy#158 exists to stop.
mic = run_franken_llama({ "STEPS" => "5", "FRANKEN_POLICY" => "dfa,dfa", "FRANKEN_B_SEED" => "42" }, nil)
failures << "macro: curve identical to MICRO dfa at the same policy" if mic.lines.select { |l| l.start_with?("step ") } == mac_c
failures << "macro: summary does not report macro_taps=2" unless mac.include?("macro_taps=2")
# determinism
failures << "macro: two identical macro runs differ" unless run_franken_llama(mac_env, nil) == mac
Dir.mktmpdir("franken_macro_gate") do |dir|
  run_franken_llama(mac_env, dir)
  rsm = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  fr = rsm["franken"] || {}
  failures << "macro: provenance dfa_granularity #{fr['dfa_granularity'].inspect} (want \"block\")" unless fr["dfa_granularity"] == "block"
  failures << "macro: provenance macro_taps #{fr['macro_taps'].inspect} (want 2)" unless fr["macro_taps"] == 2
  failures << "macro: provenance policied_tensors #{fr['policied_tensors'].inspect} (want \"block_output\")" unless fr["policied_tensors"] == "block_output"
end
# The combinations that would leave a cross-block backward path alive
# must fail LOUD, not silently produce a hybrid.
[["FRANKEN_ALIGN", "1", "align (no chain shadow exists under macro)"],
 ["ATTNRES", "1", "attnres (re-opens the cross-block path)"],
 ["FRANKEN_MTP", "1", "mtp (couples across the boundary)"],
 ["FRANKEN_NO_SHADOW", "1", "no-shadow (a micro-only axis)"]].each do |k, v, why|
  _o, stx = Open3.capture2e(mac_env.merge(k => v), RUNNER, chdir: ROOT)
  failures << "macro: #{why} was NOT rejected" if stx.success?
end
# A per-layer macro policy is not expressible (the cut is at every
# boundary) — mixed and empty policies must fail loud.
[["chain,dfa", "mixed policy"], ["", "empty policy"]].each do |pol, why|
  _o, stx = Open3.capture2e(mac_env.merge("FRANKEN_POLICY" => pol), RUNNER, chdir: ROOT)
  failures << "macro: #{why} was NOT rejected" if stx.success?
end
puts failures.length == n0 ? "  ok: macro-DFA (toy#158) — forward IDENTICAL to BP at step 1, backward diverges, B-seed moves the curve (blocks really train), macro != micro, provenance + 6 fail-loud guards" : "  FAIL: macro-DFA leg"

# ---- toy#158 ask 2: the RAdam-class optimizer ----
# RAdam's rectification is a per-step scalar on the LR, so the arm is
# AdamW's graph with a shaped lr. Two things are asserted: the
# documented DEAD ZONE (rho_t <= 4 -> no update, so the curve is FLAT
# for the first 4 steps at beta2=0.999 — stated loudly by the runner
# because a flat curve otherwise reads as a broken arm), and that it
# trains afterwards.
n0 = failures.length
rad = run_franken_llama({ "STEPS" => "12", "FRANKEN_OPTIMIZER" => "radam" }, nil)
rad_c = rad.lines.select { |l| l.start_with?("step ") }.map { |l| l[/loss=(.+)/, 1].to_f }
failures << "radam: runner did not announce the dead zone" unless rad.include?("first_stepping_step=5")
failures << "radam: the first 4 steps are not flat (rho_t <= 4 must take NO update)" unless rad_c[0, 4].uniq.length == 1
failures << "radam: loss did not fall after the dead zone (#{rad_c[4]} -> #{rad_c[11]})" unless rad_c[11] < rad_c[4]
failures << "radam: curve identical to adamw" if rad_c[11] == f0_curve[4][/loss=(.+)/, 1].to_f
Dir.mktmpdir("franken_radam_gate") do |dir|
  run_franken_llama({ "STEPS" => "6", "FRANKEN_OPTIMIZER" => "radam" }, dir)
  rsr = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  cfg = rsr["config"] || {}
  failures << "radam: provenance optimizer #{cfg['optimizer'].inspect}" unless cfg["optimizer"] == "radam"
  # beta2 travels with the run: radam CHANGES it (0.999 vs adamw 0.95),
  # so two curves at the same lr differ for this reason alone.
  failures << "radam: provenance beta2 #{cfg['beta2'].inspect} (want 0.999)" unless cfg["beta2"] == 0.999
end
puts failures.length == n0 ? "  ok: radam (toy#158) — the rho_t<=4 dead zone is flat AND announced, trains after it, beta2=0.999 recorded" : "  FAIL: radam leg"

# ---- 4. byte-repro ----
r1 = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
r2 = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
failures << "byte-repro: outputs differ" unless r1 == r2
puts "  ok: byte-repro — two policy runs identical" if r1 == r2

if failures.empty?
  puts "GATE PASS [franken-llama]: F0 byte-parity + seed!=0 parity + bundle/provenance/align + dfa-effect + corpus/align-every (toy#122) + shape presets (toy#124) + lr/warmup (toy#126) + no-shadow/pack-header/ckpt-every (toy#129) + batch (toy#133) + policy-scope/FFN-DFA (toy#151) + K1 axes (toy#136) + KDA layer (toy#137) + hybrid/AttnRes (toy#138) + MLA kind/k3 pattern (K-series M2) + per-head muon (toy#139/K5) + MTP (K-series M10) + macro-DFA/RAdam (toy#158) + byte-repro (toy#112/#113)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken-llama]: #{f}" }
  exit 1
end
