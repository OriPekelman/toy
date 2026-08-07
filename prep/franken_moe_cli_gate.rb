#!/usr/bin/env ruby
# prep/franken_moe_cli_gate.rb — toy#120 gates for the spec-callable MoE
# runner (`toy train franken-moe` → libexec/toy-train-franken-moe-cli).
# Gates the CLI SURFACE itself (the toy#113 lesson), not just the binary.
#
#   1. CROSS-BINARY NULL: dense-chain seed=0 loss values == the twin
#      rig's lane A (same builders, same init stream at seed 0) — the
#      instrument-equivalence anchor.
#   2. SEED SEMANTICS: seed=1 curve differs (init actually varies);
#      per-seed byte-repro (two identical invocations).
#   3. DENSE DFA-EXPERTS: trains (CE < 20% of start @60), differs from
#      chain; align events 4/step with finite cos + positive norms.
#   4. TOP1 aux=0: collapse control — route shares pin to one expert
#      over the tail (the F4 bracket anchor).
#   5. TOP1 aux=0.05 (THE gate alpha, pinned for F4): utilization floor
#      — each expert majority-serves >= 20% of tail steps @120.
#   6. BUNDLE: run_start.franken_moe provenance, route events == steps,
#      run_end, flow.json, TOY_RUN_ID passthrough, parseable JSONL.
#   8. BP-SPINE (toy#121 stretch, vendor-patch 0011): forward-identity
#      null — step-1 loss BYTE-equals the other top1 arms (detach is
#      identity; pre-update forward is the same graph); trains PAST the
#      fully-dfa plateau (the cut works: chain grads reach the spine,
#      the walker never needs mul_mat_id); deterministic; provenance.
#   9. CORPUS (toy#125, the F8 data surface): --corpus streams the
#      packed-i32 corpus at the frozen-vocab contract (627, toy#123) —
#      deterministic, live (differs from the fixed-seq feed), seeded,
#      composes (dense / top1+aux / bp-spine / --shape wide), EOF
#      rotation survives (the toy#122 stuck-window fix).
#   7. BP-ROUTER (toy#121, the F5 core): --moe-policy
#      bp-router-dfa-experts under top1 — curve differs from the
#      fully-DFA arm at the same seed (the p-scale credit path moves
#      Wr), deterministic, aux composes, provenance names the policy.
#      STRUCTURAL only: whether BP router credit escapes the plateau
#      at scale is F5's question, not the gate's (at gate shape it is
#      a mild 2/3-seed improvement — recorded on toy#121).

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")
RIG  = File.join(ROOT, "libexec", "toy-train-franken-moe")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

# The CLI shells out to make for staleness; a stale inherited SPINEL_DIR
# (dev tmux predating the toy#119 guard) would poison it — the gate pins
# the CLI contract, not the caller's env hygiene.
CLEAN = { "SPINEL_DIR" => nil, "SPINEL_SKIP_PIN_CHECK" => nil }

def run_cli(args, extra_env, run_dir)
  env = CLEAN.merge(extra_env)
  env = env.merge("TOY_RUN_ID" => "moe-cli-gate") if run_dir
  argv = [TOY, "train", "franken-moe"] + args
  argv += ["--out", run_dir] if run_dir
  out, st = Open3.capture2e(env, *argv, chdir: ROOT)
  abort "franken_moe_cli_gate: toy train franken-moe exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def losses(out)
  out.lines.select { |l| l.start_with?("step ") }.map { |l| l[/loss=(\S+)/, 1] }
end

failures = []

# ---- 1. cross-binary null vs the rig's lane A ----
rig_out, rig_st = Open3.capture2e(CLEAN.merge("FRANKEN_MOE" => "chain", "STEPS" => "8",
                                              "FRANKEN_SEED" => "1234"), RIG, chdir: ROOT)
abort "franken_moe_cli_gate: rig failed:\n#{rig_out.lines.last(5).join}" unless rig_st.success?
rig_lane_a = rig_out.lines.select { |l| l.start_with?("step ") }
                    .map { |l| l[/lane_a=(\S+)/, 1] }
cli_chain0 = losses(run_cli(%w[--steps 8 --seed 0], {}, nil))
if rig_lane_a.length == 8 && rig_lane_a == cli_chain0
  puts "  ok: dense-chain seed=0 == rig lane A (8 steps, cross-binary null)"
else
  failures << "rig-null: curves differ\nrig: #{rig_lane_a.inspect}\ncli: #{cli_chain0.inspect}"
end

# ---- 2. seed semantics ----
# byte-repro on the STEP LINES (whole stdout carries the run-id counter
# line, which legitimately differs per invocation)
n0 = failures.length
s1a = run_cli(%w[--steps 8 --seed 1], {}, nil)
s1b = run_cli(%w[--steps 8 --seed 1], {}, nil)
failures << "seed: seed=1 step-curve repro failed" unless losses(s1a) == losses(s1b) && losses(s1a).length == 8
failures << "seed: seed=1 curve identical to seed=0 (init not seeded)" if losses(s1a) == cli_chain0
puts failures.length == n0 ? "  ok: seed=1 differs from seed=0, curve-deterministic" : "  FAIL: seed leg"

# ---- 3. dense dfa-experts + align events ----
n0 = failures.length
Dir.mktmpdir("moe_cli_dfa") do |dir|
  out = run_cli(%w[--steps 60 --seed 0 --moe-policy dfa-experts --align-events], {}, dir)
  ls = losses(out).map(&:to_f)
  failures << "dfa-experts: NaN" if ls.any?(&:nan?)
  failures << "dfa-experts: lane CE #{ls.first} -> #{ls.last} (want < #{(ls.first * 0.2).round(3)})" unless ls.last < ls.first * 0.2
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  aligns = evs.select { |e| e["kind"] == "align" }
  failures << "dfa-experts: #{aligns.length} align events (want 240)" unless aligns.length == 240
  bad = aligns.count do |e|
    !e["cos"].is_a?(Numeric) || e["cos"].to_f.abs > 1.0001 ||
      !e["dfa_norm"].is_a?(Numeric) || e["dfa_norm"] <= 0 ||
      !e["bp_norm"].is_a?(Numeric) || e["bp_norm"] < 0 ||
      !%w[up1 down1 up2 down2].include?(e["w"])
  end
  failures << "dfa-experts: #{bad} malformed align events" unless bad == 0
  chain60 = losses(run_cli(%w[--steps 60 --seed 0], {}, nil))
  failures << "dfa-experts: identical to chain (policy no effect)" if losses(out) == chain60
  puts failures.length == n0 ? "  ok: dense dfa-experts trains (#{ls.first.round(3)} -> #{ls.last.round(4)}); 240 align events well-formed" : "  FAIL: dfa-experts leg"
end

# ---- 4 + 5. top1 collapse control / aux utilization floor ----
n0 = failures.length
Dir.mktmpdir("moe_cli_top1") do |dir|
  run_cli(%w[--steps 60 --seed 0 --routing top1], {}, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  shares = evs.select { |e| e["kind"] == "route" }.map { |e| e["shares"][0] }
  failures << "top1-aux0: #{shares.length} route events (want 60)" unless shares.length == 60
  tail = shares.last(15)
  failures << "top1-aux0: no collapse (tail #{tail.uniq.inspect}) — the F4 bracket anchor moved" unless tail.uniq.length == 1 && (tail.first == 0.0 || tail.first == 1.0)
  puts failures.length == n0 ? "  ok: top1 aux=0 collapses (collapse control; tail e0_share=#{tail.first})" : "  FAIL: top1-aux0 leg"
end
n0 = failures.length
Dir.mktmpdir("moe_cli_aux") do |dir|
  out = run_cli(%w[--steps 120 --seed 0 --routing top1 --moe-aux 0.05], {}, dir)
  ls = losses(out).map(&:to_f)
  failures << "top1-aux: NaN" if ls.any?(&:nan?)
  failures << "top1-aux: did not train (#{ls.first} -> #{ls.last})" unless ls.last < ls.first - 0.1
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  routes = evs.select { |e| e["kind"] == "route" }
  shares = routes.map { |e| e["shares"][0] }
  tail = shares.drop(20)
  e0 = tail.count { |r| r > 0.5 }
  e1 = tail.count { |r| r < 0.5 }
  floor = (tail.length * 0.2).floor
  failures << "top1-aux: starvation persists (e0maj #{e0}, e1maj #{e1}, floor #{floor})" unless e0 >= floor && e1 >= floor
  aux_bad = routes.count { |e| !e["aux"].is_a?(Numeric) || e["aux"] < 0 }
  failures << "top1-aux: #{aux_bad} malformed aux values" unless aux_bad == 0
  puts failures.length == n0 ? "  ok: top1 aux=0.05 breaks starvation (e0/e1 majorities #{e0}/#{e1} of #{tail.length})" : "  FAIL: top1-aux leg"
end

# ---- 7. bp-router (toy#121) ----
n0 = failures.length
Dir.mktmpdir("moe_cli_bpr") do |dir|
  out_a = run_cli(%w[--steps 40 --seed 0 --routing top1 --moe-policy bp-router-dfa-experts], {}, dir)
  out_b = run_cli(%w[--steps 40 --seed 0 --routing top1 --moe-policy bp-router-dfa-experts], {}, nil)
  failures << "bp-router: curve repro failed" unless losses(out_a) == losses(out_b)
  ls = losses(out_a).map(&:to_f)
  failures << "bp-router: NaN" if ls.any?(&:nan?)
  fully = losses(run_cli(%w[--steps 40 --seed 0 --routing top1], {}, nil))
  failures << "bp-router: curve identical to fully-dfa (task-BP credit path dead)" if losses(out_a) == fully
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  pol = evs.first && evs.first.dig("franken_moe", "policy")
  failures << "bp-router: provenance policy #{pol.inspect}" unless pol == "bp-router-dfa-experts"
  aux_out = run_cli(%w[--steps 20 --seed 0 --routing top1 --moe-policy bp-router-dfa-experts --moe-aux 0.05], {}, nil)
  failures << "bp-router: aux composition failed" unless losses(aux_out).length == 20 && losses(aux_out).map(&:to_f).none?(&:nan?)
  puts failures.length == n0 ? "  ok: bp-router — differs from fully-dfa (credit path live), deterministic, aux composes, provenance named" : "  FAIL: bp-router leg"
end

# ---- 8. bp-spine (toy#121 stretch; the opaque cut) ----
n0 = failures.length
Dir.mktmpdir("moe_cli_spine") do |dir|
  sp_a = run_cli(%w[--steps 60 --seed 0 --routing top1 --moe-policy bp-spine], {}, dir)
  sp_b = run_cli(%w[--steps 60 --seed 0 --routing top1 --moe-policy bp-spine], {}, nil)
  failures << "bp-spine: curve repro failed" unless losses(sp_a) == losses(sp_b)
  ls = losses(sp_a).map(&:to_f)
  failures << "bp-spine: NaN" if ls.any?(&:nan?)
  # forward-identity null: detach is identity, so the pre-update step-1
  # loss must BYTE-equal the fully-dfa arm's (params don't matter at
  # step 1; same forward graph values).
  fully1 = losses(run_cli(%w[--steps 1 --seed 0 --routing top1], {}, nil)).first
  failures << "bp-spine: step-1 forward differs from fully-dfa (#{losses(sp_a).first} vs #{fully1}) — detach is not identity" unless losses(sp_a).first == fully1
  # the cut's purpose: the BP spine escapes the fully-DFA plateau
  # (fully-dfa sits ~2.6+ at this shape/horizon; bp-spine reaches
  # full-BP-class loss — 0.002-class by step 100; assert a wide margin)
  failures << "bp-spine: did not escape the plateau (step60 #{ls.last})" unless ls.last < 0.5
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  pol = evs.first && evs.first.dig("franken_moe", "policy")
  failures << "bp-spine: provenance policy #{pol.inspect}" unless pol == "bp-spine"
  aux_out = run_cli(%w[--steps 20 --seed 0 --routing top1 --moe-policy bp-spine --moe-aux 0.05], {}, nil)
  failures << "bp-spine: aux composition failed" unless losses(aux_out).length == 20 && losses(aux_out).map(&:to_f).none?(&:nan?)
  puts failures.length == n0 ? "  ok: bp-spine — step-1 byte-equals fully-dfa (detach identity), escapes the plateau (#{ls.first.round(3)} -> #{ls.last.round(4)} @60), deterministic, aux composes" : "  FAIL: bp-spine leg"
end

# ---- toy#124: --shape wide (d256/ff512, same NE/T/vocab) ----
n0 = failures.length
Dir.mktmpdir("moe_cli_wide") do |dir|
  w1 = run_cli(%w[--steps 8 --seed 0 --shape wide --routing top1 --moe-policy bp-spine], {}, dir)
  w2 = run_cli(%w[--steps 8 --seed 0 --shape wide --routing top1 --moe-policy bp-spine], {}, nil)
  failures << "wide: not deterministic" unless losses(w1) == losses(w2)
  wl = losses(w1).map(&:to_f)
  failures << "wide: NaN" if wl.any?(&:nan?)
  failures << "wide: bp-spine did not train (#{wl.first} -> #{wl.last})" unless wl.last < wl.first - 0.5
  wc = losses(run_cli(%w[--steps 8 --seed 0 --shape wide], {}, nil))
  failures << "wide: dense-chain identical to bp-spine (shape/policy dead)" if losses(w1) == wc
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  shp = evs.first && evs.first.dig("model", "shape")
  failures << "wide: provenance shape #{shp.inspect}" unless shp == "wide"
  puts failures.length == n0 ? "  ok: --shape wide — bp-spine trains (#{wl.first.round(3)} -> #{wl.last.round(3)}), deterministic, provenance" : "  FAIL: wide leg"
end

# ---- toy#127: --align-every thinning (the toy#122 thinning, MoE-side) ----
# N=3 over 6 steps -> emissions at steps 1,4 -> 4 weights x 2 = 8 align
# events; step + route events stay per-step (6 each).
n0 = failures.length
Dir.mktmpdir("moe_cli_ae") do |dir|
  run_cli(%w[--steps 6 --seed 0 --moe-policy dfa-experts --align-events --align-every 3], {}, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  aligns = evs.select { |e| e["kind"] == "align" }
  steps_seen = aligns.map { |e| e["step"] }.uniq.sort
  failures << "align-every: #{aligns.length} align events (want 8)" unless aligns.length == 8
  failures << "align-every: wrong steps #{steps_seen.inspect} (want [1,4])" unless steps_seen == [1, 4]
  failures << "align-every: step events thinned too (#{evs.count { |e| e['kind'] == 'step' }})" unless evs.count { |e| e["kind"] == "step" } == 6
  failures << "align-every: route events thinned too (#{evs.count { |e| e['kind'] == 'route' }})" unless evs.count { |e| e["kind"] == "route" } == 6
  puts failures.length == n0 ? "  ok: --align-every thins align emissions (8 @ N=3/6 steps; step/route events untouched)" : "  FAIL: align-every leg"
end

# ---- toy#129 item 2: --no-shadow (dense dfa-experts) ----
# The applied-updates null: dense dfa-experts with --no-shadow (experts
# late-param via the top1 wire, no chain accs) byte-equals the shadow
# build. top1 + --no-shadow fails loud (already shadow-free); provenance
# franken_moe.shadow says true (shadow dfa-experts) / false (no-shadow,
# and every top1 lane).
n0 = failures.length
sh_dfa = losses(run_cli(%w[--steps 8 --seed 0 --moe-policy dfa-experts], {}, nil))
Dir.mktmpdir("moe_cli_ns") do |dir|
  ns_out = run_cli(%w[--steps 8 --seed 0 --moe-policy dfa-experts --no-shadow], {}, dir)
  failures << "no-shadow: applied updates differ from shadow build" unless losses(ns_out) == sh_dfa && sh_dfa.length == 8
  rs0 = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "no-shadow: provenance shadow #{rs0.dig('franken_moe', 'shadow').inspect} (want false)" unless rs0.dig("franken_moe", "shadow") == false
end
Dir.mktmpdir("moe_cli_ns2") do |dir|
  run_cli(%w[--steps 2 --seed 0 --moe-policy dfa-experts --align-events], {}, dir)
  rs1 = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "no-shadow: shadow dfa-experts provenance #{rs1.dig('franken_moe', 'shadow').inspect} (want true)" unless rs1.dig("franken_moe", "shadow") == true
end
Dir.mktmpdir("moe_cli_ns3") do |dir|
  run_cli(%w[--steps 2 --seed 0 --routing top1 --moe-policy bp-spine], {}, dir)
  rs2 = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "no-shadow: top1 provenance shadow #{rs2.dig('franken_moe', 'shadow').inspect} (want false)" unless rs2.dig("franken_moe", "shadow") == false
end
argv_bad = [TOY, "train", "franken-moe", "--steps", "1", "--routing", "top1", "--no-shadow"]
_ob, stb = Open3.capture2e(CLEAN, *argv_bad, chdir: ROOT)
failures << "no-shadow: top1 + --no-shadow not rejected" if stb.success?
puts failures.length == n0 ? "  ok: --no-shadow — dense dfa-experts byte-equals shadow, provenance true/false/false, top1 combo rejected" : "  FAIL: no-shadow leg"

# ---- toy#125: --corpus (the F8 data surface) ----
# The corpus feed runs the frozen-vocab contract (627, toy#123) — the
# vocab-16 fixed-seq embed cannot take the stream's ids — so the leg
# pins the vocab shift in provenance alongside the stream itself:
# deterministic, differs from the fixed-seq feed, seed-sensitive with
# per-seed repro (the standing seed!=0 lesson), composes with dense /
# top1+aux / bp-spine / --shape wide, and survives EOF rotation (the
# toy#122 stuck-window fix; tiny corpus forces the pre-EOF restart).
n0 = failures.length
Dir.mktmpdir("moe_cli_corpus") do |dir|
  c_args = %w[--steps 8 --seed 0 --corpus data/ts_seqs.bin --routing top1 --moe-policy bp-spine]
  c1 = run_cli(c_args, {}, dir)
  c2 = run_cli(c_args, {}, nil)
  failures << "corpus: not deterministic" unless losses(c1) == losses(c2) && losses(c1).length == 8
  failures << "corpus: NaN" if losses(c1).map(&:to_f).any?(&:nan?)
  d8 = losses(run_cli(%w[--steps 8 --seed 0 --routing top1 --moe-policy bp-spine], {}, nil))
  failures << "corpus: curve identical to fixed-seq feed (stream not live)" if losses(c1) == d8
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  vocab = evs.first && evs.first.dig("model", "vocab")
  failures << "corpus: provenance vocab #{vocab.inspect} (want 627 — the frozen-vocab contract)" unless vocab == 627
  s1_args = %w[--steps 8 --seed 1 --corpus data/ts_seqs.bin --routing top1 --moe-policy bp-spine]
  s1a = run_cli(s1_args, {}, nil)
  s1b = run_cli(s1_args, {}, nil)
  failures << "corpus: seed=1 repro failed" unless losses(s1a) == losses(s1b)
  failures << "corpus: seed=1 identical to seed=0 (init not seeded)" if losses(s1a) == losses(c1)
  w_args = %w[--steps 8 --seed 0 --corpus data/ts_seqs.bin --shape wide --routing top1 --moe-policy bp-spine]
  w1 = run_cli(w_args, {}, nil)
  w2 = run_cli(w_args, {}, nil)
  failures << "corpus-wide: not deterministic" unless losses(w1) == losses(w2) && losses(w1).length == 8
  dn = run_cli(%w[--steps 8 --seed 0 --corpus data/ts_seqs.bin], {}, nil)
  ax = run_cli(%w[--steps 8 --seed 0 --corpus data/ts_seqs.bin --routing top1 --moe-aux 0.05], {}, nil)
  comp = losses(dn) + losses(ax)
  failures << "corpus: dense/aux composition failed" unless losses(dn).length == 8 && losses(ax).length == 8 && comp.map(&:to_f).none?(&:nan?)
  tiny = File.join(dir, "tiny_corpus.bin")
  File.binwrite(tiny, File.binread(File.join(ROOT, "data", "ts_seqs.bin"), 160))   # 40 tokens = 10 windows
  r_args = ["--steps", "25", "--seed", "0", "--corpus", tiny, "--routing", "top1", "--moe-policy", "bp-spine"]
  r1 = run_cli(r_args, {}, nil)
  r2 = run_cli(r_args, {}, nil)
  failures << "corpus-rotate: not deterministic across EOF" unless losses(r1) == losses(r2) && losses(r1).length == 25
  failures << "corpus-rotate: NaN past EOF" if losses(r1).map(&:to_f).any?(&:nan?)
  puts failures.length == n0 ? "  ok: --corpus streams at vocab 627 (deterministic, differs from fixed-seq, seed semantics, dense/aux/wide compose, EOF rotation)" : "  FAIL: corpus leg"
end

# ---- toy#128: --experts N (the demonstrator's E axis) ----
# E=2 flag-null (== default byte-exact); E=4: provenance + length-4
# shares; aux bracket — aux=0 collapses to ONE expert, aux=0.2 puts
# every expert >= 0.1 tail mean share (deterministic, so pinnable;
# NOTE aux=0.05 leaves an expert starved at E=4/T=4 — F8c's alpha
# question, recorded on the issue, deliberately NOT gated); bp-spine
# escapes the fully-dfa plateau at E=4; dense dfa-experts wires all
# 8 expert weights (align names up1..down4); composes wide+corpus.
n0 = failures.length
e2 = losses(run_cli(%w[--steps 8 --seed 0 --experts 2], {}, nil))
failures << "experts: --experts 2 differs from default (flag-null broken)" unless e2 == cli_chain0
Dir.mktmpdir("moe_cli_e4") do |dir|
  run_cli(%w[--steps 120 --seed 0 --experts 4 --routing top1], {}, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  rs = evs.first || {}
  failures << "experts: provenance n_experts #{rs.dig('model', 'n_experts').inspect} (want 4)" unless rs.dig("model", "n_experts") == 4
  routes = evs.select { |e| e["kind"] == "route" }
  failures << "experts: shares not length-4 vectors" unless routes.length == 120 && routes.all? { |r| r["shares"].length == 4 }
  tail = routes.drop(20)
  means = [0, 1, 2, 3].map { |e| tail.map { |r| r["shares"][e] }.sum / tail.length }
  failures << "experts: aux=0 did not collapse (#{means.map { |m| m.round(3) }.inspect})" unless means.max > 0.999
end
Dir.mktmpdir("moe_cli_e4aux") do |dir|
  run_cli(%w[--steps 120 --seed 0 --experts 4 --routing top1 --moe-aux 0.2], {}, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  routes = evs.select { |e| e["kind"] == "route" }
  tail = routes.drop(20)
  means = [0, 1, 2, 3].map { |e| tail.map { |r| r["shares"][e] }.sum / tail.length }
  failures << "experts: aux=0.2 floor broken (#{means.map { |m| m.round(3) }.inspect}, want all >= 0.1)" unless means.all? { |m| m >= 0.1 }
end
sp_args = %w[--steps 60 --seed 0 --experts 4 --routing top1 --moe-policy bp-spine]
sp1 = run_cli(sp_args, {}, nil)
sp2 = run_cli(sp_args, {}, nil)
failures << "experts: E=4 bp-spine not deterministic" unless losses(sp1) == losses(sp2)
spl = losses(sp1).map(&:to_f)
failures << "experts: E=4 bp-spine did not escape the plateau (#{spl.last})" unless spl.last < 0.5
fd = losses(run_cli(%w[--steps 60 --seed 0 --experts 4 --routing top1], {}, nil))
failures << "experts: E=4 bp-spine identical to fully-dfa" if losses(sp1) == fd
Dir.mktmpdir("moe_cli_e4dfa") do |dir|
  run_cli(%w[--steps 6 --seed 0 --experts 4 --moe-policy dfa-experts --align-events], {}, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  aligns = evs.select { |e| e["kind"] == "align" }
  names = aligns.map { |e| e["w"] }.uniq.sort
  failures << "experts: #{aligns.length} align events (want 48 = 8 weights x 6)" unless aligns.length == 48
  failures << "experts: align names #{names.inspect}" unless names == %w[down1 down2 down3 down4 up1 up2 up3 up4]
end
wc = run_cli(%w[--steps 8 --seed 0 --experts 4 --shape wide --corpus data/ts_seqs.bin --routing top1 --moe-policy bp-spine], {}, nil)
failures << "experts: wide+corpus composition failed" unless losses(wc).length == 8 && losses(wc).map(&:to_f).none?(&:nan?)
puts failures.length == n0 ? "  ok: --experts — E=2 flag-null; E=4 aux bracket (collapse vs >=0.1 floor @0.2), bp-spine escapes plateau, 8 dfa wires named, wide+corpus compose" : "  FAIL: experts leg"

# ---- toy#129 item 1: TOYC pack + context on the MoE CLI ----
n0 = failures.length
abort "franken gate: data/fineweb_gpt2_smoke.bin missing — generate it: uv run prep/pretokenize_pack.py --tokens 200_000 --out data/fineweb_gpt2_smoke.bin" unless File.file?(File.join(ROOT, "data", "fineweb_gpt2_smoke.bin"))
Dir.mktmpdir("moe_cli_pack") do |dir|
  pk_out = run_cli(%w[--steps 3 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 16 --routing top1 --moe-policy bp-spine], {}, dir)
  pk_out2 = run_cli(%w[--steps 3 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 16 --routing top1 --moe-policy bp-spine], {}, nil)
  failures << "pack: not deterministic" unless losses(pk_out) == losses(pk_out2) && losses(pk_out).length == 3
  rsp = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "pack: model.vocab #{rsp.dig('model', 'vocab').inspect} (want 50257)" unless rsp.dig("model", "vocab") == 50257
  failures << "pack: config.context #{rsp.dig('config', 'context').inspect} (want 16)" unless rsp.dig("config", "context") == 16
end
argv_ctx = [TOY, "train", "franken-moe", "--steps", "1", "--context", "16"]
_oc2, stc2 = Open3.capture2e(CLEAN, *argv_ctx, chdir: ROOT)
failures << "pack: --context without --corpus not rejected" if stc2.success?
puts failures.length == n0 ? "  ok: TOYC pack — vocab 50257 + ctx 16 on the MoE CLI (deterministic, provenance); context-without-corpus rejected" : "  FAIL: pack leg"

# ---- toy#132: --lr/--warmup (toy#126 parity, MoE-side) ----
# --lr moves the curve deterministically; explicit --lr 0.02 byte-
# equals the default (0.02 IS the recipe default — flag-null); warmup
# ramps to LR at step N, pinned via the step events' new lr field.
n0 = failures.length
lr1 = losses(run_cli(%w[--steps 8 --seed 0 --lr 0.05], {}, nil))
lr2 = losses(run_cli(%w[--steps 8 --seed 0 --lr 0.05], {}, nil))
failures << "lr: not deterministic" unless lr1 == lr2 && lr1.length == 8
failures << "lr: curve identical to default 0.02 (knob dead)" if lr1 == cli_chain0
lr_null = losses(run_cli(%w[--steps 8 --seed 0 --lr 0.02], {}, nil))
failures << "lr: explicit 0.02 differs from default (flag-null broken)" unless lr_null == cli_chain0
Dir.mktmpdir("moe_cli_lr") do |dir|
  run_cli(%w[--steps 6 --seed 0 --lr 0.05 --warmup 4], {}, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  lrs = evs.select { |e| e["kind"] == "step" }.map { |e| e["lr"] }
  want = [1, 2, 3, 4].map { |t| 0.05 * (t.to_f / 4.0) } + [0.05, 0.05]
  ramp_ok = lrs.length == 6 && lrs.zip(want).all? { |g, w| g.is_a?(Numeric) && (g - w).abs < 1.0e-12 }
  failures << "warmup: step-event lr ramp #{lrs.inspect} (want #{want.inspect})" unless ramp_ok
  failures << "warmup: run_start config warmup wrong" unless evs.first.dig("config", "warmup") == 4
end
wu1 = losses(run_cli(%w[--steps 6 --seed 0 --lr 0.05 --warmup 4], {}, nil))
flat6 = losses(run_cli(%w[--steps 6 --seed 0 --lr 0.05], {}, nil))
failures << "warmup: curve identical to flat lr (ramp dead)" if wu1 == flat6
puts failures.length == n0 ? "  ok: --lr moves the curve (flag-null at 0.02); --warmup ramps to LR @N (step events pin the ramp)" : "  FAIL: lr/warmup leg"

# ---- toy#130: end-of-run held-out eval (--eval-corpus) ----
# The MoE lane has no GGUF writer, so held-out CE runs IN-RUNNER after
# training (lr=0 windows; Adam moments are dead then). Byte-null on
# training: the step curve must byte-equal the no-eval run; the
# eval_ce line + eval event (before run_end) carry the read.
n0 = failures.length
ev_args = %w[--steps 8 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 16 --routing top1 --moe-policy bp-spine]
no_ev = run_cli(ev_args, {}, nil)
Dir.mktmpdir("moe_cli_eval") do |dir|
  ev1 = run_cli(ev_args + %w[--eval-corpus data/fineweb_gpt2_smoke.bin --eval-tokens 256 --eval-offset 150000], {}, dir)
  ev2 = run_cli(ev_args + %w[--eval-corpus data/fineweb_gpt2_smoke.bin --eval-tokens 256 --eval-offset 150000], {}, nil)
  failures << "eval: training curve differs from no-eval run (eval not byte-null on training)" unless losses(ev1) == losses(no_ev)
  ce1 = ev1.lines.select { |l| l.start_with?("eval_ce:") }
  ce2 = ev2.lines.select { |l| l.start_with?("eval_ce:") }
  failures << "eval: no eval_ce line" unless ce1.length == 1
  failures << "eval: not deterministic\n1: #{ce1.join}2: #{ce2.join}" unless ce1 == ce2
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  evx = evs.find { |e| e["kind"] == "eval" }
  failures << "eval: no eval event" if evx.nil?
  failures << "eval: eval event malformed" unless evx && evx["name"] == "eval-ce" && evx["loss"].is_a?(Numeric) && evx["windows"] == 16
  failures << "eval: run_end not last" unless evs.last && evs.last["kind"] == "run_end"
  puts failures.length == n0 ? "  ok: --eval-corpus — end-of-run held-out CE (#{ce1.first.to_s.strip}); training byte-null; eval event before run_end" : "  FAIL: eval leg"
end

# ---- eval must FREEZE THE WEIGHTS, on every optimizer and under the
# ---- toy#146 ramp (the toy#139/#146 eval-loop regression) ----
# "lr=0 windows" was only ever enforced on the GLOBAL adamw hp vector.
# That silently stopped being the whole story twice:
#   - --optimizer muon/sgd: 2-D weights step through the SGD path, and
#     t_hp_sgd kept the last TRAINING lr, so the held-out windows kept
#     training — the metric training on its own test split. Under sgd,
#     t_hp is unreachable from the graph and uploading it ABORTED.
#   - --lr-schedule ramp-*: the per-layer vectors kept the training lr,
#     so every ramped layer trained through eval.
# Byte-null on the TRAINING curve (the leg above) cannot see any of
# this: the damage happens after the last step event. The assertion has
# to be that run_end's layer_sig is IDENTICAL with and without
# --eval-corpus — i.e. the eval phase moved no weights at all.
n0 = failures.length
fz = %w[--steps 4 --seed 0 --shape deep --experts 4 --context 16
        --corpus data/fineweb_gpt2_smoke.bin --moe-policy dfa-experts --lr 0.05]
fz_ev = %w[--eval-corpus data/fineweb_gpt2_smoke.bin --eval-tokens 64]
[["adamw", []],
 ["muon",  %w[--optimizer muon]],
 ["sgd",   %w[--optimizer sgd]]].each do |oname, oarg|
  [["flat", []],
   ["ramp", %w[--lr-schedule ramp-down --lr-lo 0.005 --lr-hi 0.05]]].each do |rname, rarg|
    Dir.mktmpdir("moe_fz_a") do |da|
      Dir.mktmpdir("moe_fz_b") do |db|
        run_cli(fz + oarg + rarg, {}, da)
        run_cli(fz + oarg + rarg + fz_ev, {}, db)
        la = JSON.parse(File.readlines(File.join(da, "events.jsonl")).last)["layer_sig"]
        lb = JSON.parse(File.readlines(File.join(db, "events.jsonl")).last)["layer_sig"]
        if la.nil? || lb.nil?
          failures << "eval-freeze(#{oname}/#{rname}): no run_end layer_sig (run died)"
        elsif la != lb
          d0 = (0...6).map { |i| ((lb["l#{i}"] || 0) - (la["l#{i}"] || 0)).abs }.max
          failures << "eval-freeze(#{oname}/#{rname}): --eval-corpus MOVED the weights (max |delta| #{d0.round(3)}) — the held-out metric is training on its own split"
        end
      end
    end
  end
end
# The sgd arm above also covers the abort: run_cli aborts the gate on a
# non-zero exit, so reaching this line at all means sgd + --eval-corpus
# completed instead of dying in ggml_backend_tensor_set.
puts failures.length == n0 ? "  ok: --eval-corpus freezes the weights — run_end layer_sig identical with/without eval on adamw/muon/sgd x flat/ramp (sgd no longer aborts)" : "  FAIL: eval-freeze leg"

# ---- toy#149: eval memory must NOT grow with the window count ----
# The eval loop called next_token_guarded_batched per window, which
# allocates a fresh tv x vocab Mat each time. At the F-series shape
# (context 256 x batch 32 = 8192 tokens, vocab 50257) that is 3.3 GB of
# f64 PER WINDOW, so a 64-window eval asked for ~210 GB and Tao's F9m
# muon slice was OOM-killed (rc=137). Peak RSS grew with WINDOW COUNT,
# which is exactly what made shrinking --eval-tokens look like a fix
# rather than a smaller eval set.
#
# The eval now mutates ONE buffer incrementally, the same trick the
# training loop uses (toy#133) and that eval_ce.rb's own header already
# documents. The assertion is the RATIO of peak RSS at 16x the windows:
# pre-fix it was ~3x, post-fix it is 1.00x. Gated loosely at < 1.30 so
# it tests the LEAK, not allocator noise.
#
# Linux-only: /usr/bin/time -f is GNU-specific (BSD/macOS uses -l), and
# the cross-platform gate discipline says pin behaviour, not tooling.
n0 = failures.length
if RUBY_PLATFORM =~ /linux/ && File.executable?("/usr/bin/time")
  Dir.mktmpdir("moe_evalmem") do |dir|
    run_cli(%w[--steps 2 --seed 0 --experts 4 --context 64 --batch 4
               --corpus data/fineweb_gpt2_smoke.bin --moe-policy dfa-experts
               --ckpt-every 2], {}, dir)
    ck = File.join(dir, "weights", "step_2.gguf")
    if !File.file?(ck)
      failures << "eval-mem: no checkpoint to eval"
    else
      peak = lambda do |tokens|
        out, _st = Open3.capture2e(
          CLEAN, "/usr/bin/time", "-f", "%M",
          File.join(ROOT, "bin", "toy"), "train", "franken-moe",
          "--steps", "0", "--seed", "0", "--experts", "4",
          "--context", "64", "--batch", "4",
          "--corpus", "data/fineweb_gpt2_smoke.bin", "--moe-policy", "dfa-experts",
          "--load-ckpt", ck,
          "--eval-corpus", "data/fineweb_gpt2_smoke.bin",
          "--eval-tokens", tokens.to_s, chdir: ROOT)
        [out[/^(\d+)\s*$/, 1].to_i, out]
      end
      small, o_s = peak.call(1024)     # 4 windows
      big,   o_b = peak.call(16384)    # 64 windows
      # the separate --load-ckpt eval must WORK — Tao#149 believed it did
      # not exist for franken-moe checkpoints; it does, and it is the
      # supported way to score them off-line.
      failures << "eval-mem: --load-ckpt eval produced no eval_ce line\n#{o_b.lines.last(4).join}" unless o_b.include?("eval_ce:")
      if small <= 0 || big <= 0
        failures << "eval-mem: could not read peak RSS (small=#{small} big=#{big})"
      else
        ratio = big.to_f / small.to_f
        failures << "eval-mem: peak RSS grew #{ratio.round(2)}x for 16x the windows (#{small}KB -> #{big}KB) — the per-window label allocation is back" unless ratio < 1.30
      end
    end
  end
  puts failures.length == n0 ? "  ok: eval memory is flat in the window count (--load-ckpt eval works; peak RSS ratio < 1.3 at 16x windows)" : "  FAIL: eval-mem leg"
else
  puts "  skip: eval-mem leg (needs Linux + GNU /usr/bin/time)"
end

# ---- toy#133: --batch on the MoE CLI ----
# Same order-swap isolation null (the mask-alloc-after-finalize bug
# read zeros and ran unmasked — this leg pins the fix), plus B=1
# flag-null and determinism.
n0 = failures.length
Dir.mktmpdir("moe_cli_batch") do |dir|
  raw = File.binread(File.join(ROOT, "data", "fineweb_gpt2_smoke.bin"))
  w = raw[16, 32 * 4]
  x = raw[16 + 5000 * 4, 32 * 4]
  File.binwrite(File.join(dir, "w.bin"), w)
  File.binwrite(File.join(dir, "x.bin"), x)
  File.binwrite(File.join(dir, "wx.bin"), w + x)
  File.binwrite(File.join(dir, "xw.bin"), x + w)
  get1 = lambda do |args|
    losses(run_cli(args, {}, nil)).first.to_f
  end
  lw = get1.call(["--steps", "1", "--seed", "0", "--corpus", File.join(dir, "w.bin"), "--context", "32", "--vocab", "50257"])
  lx = get1.call(["--steps", "1", "--seed", "0", "--corpus", File.join(dir, "x.bin"), "--context", "32", "--vocab", "50257"])
  lwx = get1.call(["--steps", "1", "--seed", "0", "--corpus", File.join(dir, "wx.bin"), "--context", "32", "--vocab", "50257", "--batch", "2"])
  lxw = get1.call(["--steps", "1", "--seed", "0", "--corpus", File.join(dir, "xw.bin"), "--context", "32", "--vocab", "50257", "--batch", "2"])
  mean = (lw + lx) / 2.0
  failures << "batch: [W,X] #{lwx} != mean #{mean}" unless (lwx - mean).abs < 1.0e-4
  failures << "batch: order-swap #{lxw} != #{lwx} (cross-window leak)" unless (lxw - lwx).abs < 1.0e-4
end
mb1 = losses(run_cli(%w[--steps 4 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 16 --batch 1 --routing top1 --moe-policy bp-spine], {}, nil))
mb0 = losses(run_cli(%w[--steps 4 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 16 --routing top1 --moe-policy bp-spine], {}, nil))
failures << "batch: --batch 1 differs from no-flag (flag-null broken)" unless mb1 == mb0
mb4a = losses(run_cli(%w[--steps 4 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 16 --batch 4 --routing top1 --moe-policy bp-spine], {}, nil))
mb4b = losses(run_cli(%w[--steps 4 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 16 --batch 4 --routing top1 --moe-policy bp-spine], {}, nil))
failures << "batch: B=4 not deterministic" unless mb4a == mb4b && mb4a.length == 4
puts failures.length == n0 ? "  ok: --batch — order-swap isolation null, flag-null, B=4 deterministic" : "  FAIL: batch leg"

# ---- toy#131: checkpoint writer + --ckpt-every + --load-ckpt ----
# Boundary checkpoints (toy-moe/v1 GGUF, native per-tensor form);
# training curve byte-null vs no-ckpt (the write reads session buffers
# in place — no sched touch); THE round-trip null: eval-after-reload
# byte-equals the training run's own end-of-run eval. Shape-mismatch
# and resume (steps>0) fail loud.
n0 = failures.length
ck_args = %w[--steps 8 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 16 --shape wide --experts 4 --routing top1 --moe-policy bp-spine]
ev_args2 = %w[--eval-corpus data/fineweb_gpt2_smoke.bin --eval-tokens 256 --eval-offset 150000]
no_ck = run_cli(ck_args + ev_args2, {}, nil)
Dir.mktmpdir("moe_cli_ck") do |dir|
  ck_out = run_cli(ck_args + %w[--ckpt-every 4] + ev_args2, {}, dir)
  failures << "ckpt: curve differs from no-ckpt run (write disturbs training)" unless losses(ck_out) == losses(no_ck)
  %w[step_4.gguf step_8.gguf].each do |ck|
    failures << "ckpt: missing weights/#{ck}" unless File.file?(File.join(dir, "weights", ck))
  end
  train_ce = ck_out.lines.select { |l| l.start_with?("eval_ce:") }
  ck8 = File.join(dir, "weights", "step_8.gguf")
  rl_out = run_cli(%w[--steps 0 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 16 --shape wide --experts 4 --routing top1 --moe-policy bp-spine] + ["--load-ckpt", ck8] + ev_args2, {}, nil)
  rl_ce = rl_out.lines.select { |l| l.start_with?("eval_ce:") }
  failures << "ckpt: round-trip eval differs\ntrain: #{train_ce.join}reload: #{rl_ce.join}" unless rl_ce == train_ce && rl_ce.length == 1
  argv_mm = [TOY, "train", "franken-moe", "--steps", "0", "--seed", "0", "--corpus", "data/fineweb_gpt2_smoke.bin", "--context", "16", "--shape", "wide", "--experts", "2", "--routing", "top1", "--moe-policy", "bp-spine", "--load-ckpt", ck8] + ev_args2
  _om, stm = Open3.capture2e(CLEAN, *argv_mm, chdir: ROOT)
  failures << "ckpt: shape mismatch (E=2 vs E=4 checkpoint) not rejected" if stm.success?
  argv_rs = [TOY, "train", "franken-moe", "--steps", "4", "--seed", "0", "--corpus", "data/fineweb_gpt2_smoke.bin", "--context", "16", "--shape", "wide", "--experts", "4", "--routing", "top1", "--moe-policy", "bp-spine", "--load-ckpt", ck8] + ev_args2
  _or, str2 = Open3.capture2e(CLEAN, *argv_rs, chdir: ROOT)
  failures << "ckpt: resume (steps>0 with --load-ckpt) not rejected" if str2.success?
end
puts failures.length == n0 ? "  ok: --ckpt-every/--load-ckpt — boundary ckpts, training byte-null, round-trip eval byte-equal, mismatch+resume rejected" : "  FAIL: ckpt leg"

# ---- toy#136 (K1): --moe-balance qb + --schedule ----
# QB (K3 quantile balancing, aux-FREE): at E=8/120 steps the
# no-balance control collapses to <=3 active experts while QB keeps
# >=4 active (deterministic — pinnable); provenance balance=qb;
# qb+dense and qb+aux fail loud; cosine schedule provenance.
n0 = failures.length
Dir.mktmpdir("moe_cli_qb") do |dir|
  qb_args = %w[--steps 120 --seed 0 --experts 8 --routing top1 --moe-balance qb]
  o1 = run_cli(qb_args, {}, dir)
  o2 = run_cli(qb_args, {}, nil)
  failures << "qb: not deterministic" unless losses(o1) == losses(o2)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  failures << "qb: provenance balance #{evs.first.dig('franken_moe', 'balance').inspect}" unless evs.first.dig("franken_moe", "balance") == "qb"
  tail = evs.select { |e| e["kind"] == "route" }.drop(20)
  qb_active = (0...8).count { |e| (tail.map { |r| r["shares"][e] }.sum / tail.length) > 0.05 }
  failures << "qb: only #{qb_active} active experts (want >= 4)" unless qb_active >= 4
end
Dir.mktmpdir("moe_cli_qbc") do |dir|
  run_cli(%w[--steps 120 --seed 0 --experts 8 --routing top1], {}, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  tail = evs.select { |e| e["kind"] == "route" }.drop(20)
  nb_active = (0...8).count { |e| (tail.map { |r| r["shares"][e] }.sum / tail.length) > 0.05 }
  failures << "qb: no-balance control has #{nb_active} active (want <= 3 — the collapse control moved)" unless nb_active <= 3
end
argv_qd = [TOY, "train", "franken-moe", "--steps", "1", "--moe-balance", "qb"]
_oq, sq = Open3.capture2e(CLEAN, *argv_qd, chdir: ROOT)
failures << "qb: dense + qb not rejected" if sq.success?
argv_qa = [TOY, "train", "franken-moe", "--steps", "1", "--routing", "top1", "--moe-balance", "qb", "--moe-aux", "0.1"]
_oa, sa = Open3.capture2e(CLEAN, *argv_qa, chdir: ROOT)
failures << "qb: qb + aux not rejected" if sa.success?
Dir.mktmpdir("moe_cli_cos") do |dir|
  run_cli(%w[--steps 4 --seed 0 --schedule cosine --lr 0.05], {}, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  failures << "schedule: provenance #{evs.first.dig('config', 'schedule').inspect}" unless evs.first.dig("config", "schedule") == "cosine"
  lrs = evs.select { |e| e["kind"] == "step" }.map { |e| e["lr"] }
  failures << "schedule: lr not decreasing (#{lrs.inspect})" unless lrs.each_cons(2).all? { |x, y| y < x }
end
puts failures.length == n0 ? "  ok: QB balances at E=8 (>=4 active vs <=3 collapse control), deterministic, guards; cosine provenance + decreasing lr" : "  FAIL: qb/schedule leg"

# ---- toy#136 K1.1: --attn-gate (K3 attention output gate) ----
# W_g lands at the TAIL index (9+2E) so the spine 0..8 and the expert
# pairs keep their indices AND their DfaB seeds — the off-path is
# byte-null (leg 1's cross-binary rig anchor pins it). On: the curve
# moves, bp-spine STILL escapes the plateau (W_g is chain-wired into
# the spine), provenance + cost params grow, and the fully-DFA lane
# rejects (no tap for W_g's input).
n0 = failures.length
Dir.mktmpdir("moe_cli_gate") do |dir|
  g_args = %w[--steps 60 --seed 0 --routing top1 --moe-policy bp-spine --attn-gate]
  g1 = run_cli(g_args, {}, dir)
  g2 = run_cli(g_args, {}, nil)
  failures << "attn-gate: not deterministic" unless losses(g1) == losses(g2)
  gl = losses(g1).map(&:to_f)
  failures << "attn-gate: NaN" if gl.any?(&:nan?)
  ungated = losses(run_cli(%w[--steps 60 --seed 0 --routing top1 --moe-policy bp-spine], {}, nil))
  failures << "attn-gate: curve identical to ungated (gate dead)" if losses(g1) == ungated
  failures << "attn-gate: bp-spine no longer escapes the plateau (#{gl.last})" unless gl.last < 0.5
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  failures << "attn-gate: provenance attn_gate #{evs.first.dig('franken_moe', 'attn_gate').inspect}" unless evs.first.dig("franken_moe", "attn_gate") == true
  ug_dir_params = nil
  Dir.mktmpdir("moe_cli_gate_off") do |d2|
    run_cli(%w[--steps 2 --seed 0 --routing top1 --moe-policy bp-spine], {}, d2)
    e2 = File.readlines(File.join(d2, "events.jsonl")).map { |l| JSON.parse(l) }
    ug_dir_params = e2.first.dig("cost", "total_params")
    failures << "attn-gate: off-path provenance attn_gate true" unless e2.first.dig("franken_moe", "attn_gate") == false
  end
  gp = evs.first.dig("cost", "total_params")
  failures << "attn-gate: cost params did not grow (#{ug_dir_params} -> #{gp})" unless gp > ug_dir_params
end
argv_gd = [TOY, "train", "franken-moe", "--steps", "2", "--routing", "top1", "--attn-gate"]
_og, sg = Open3.capture2e(CLEAN, *argv_gd, chdir: ROOT)
failures << "attn-gate: fully-DFA lane + gate not rejected" if sg.success?
puts failures.length == n0 ? "  ok: --attn-gate — tail-index W_g joins the spine (bp-spine still escapes), deterministic, provenance + cost grow, fully-DFA rejected" : "  FAIL: attn-gate leg"

# ---- toy#139: --optimizer adamw|muon|sgd ----
# adamw is the byte-null default (leg 1's cross-binary rig anchor pins
# it; here the EXPLICIT flag must reproduce it). muon and sgd each
# train, deterministically, and differ from adamw and each other.
# Provenance records the optimizer AND the per-param-class routing —
# tao#139 asked for the routing to be auditable from a bundle, because
# "Muon" without "2D-hidden only" is a different (strawman) recipe.
n0 = failures.length
opt_null = losses(run_cli(%w[--steps 8 --seed 0 --optimizer adamw], {}, nil))
failures << "optimizer: explicit adamw differs from the default (flag-null broken)" unless opt_null == cli_chain0
%w[muon sgd].each do |o|
  a = losses(run_cli(["--steps", "8", "--seed", "0", "--optimizer", o], {}, nil))
  b = losses(run_cli(["--steps", "8", "--seed", "0", "--optimizer", o], {}, nil))
  failures << "optimizer #{o}: not deterministic" unless a == b && a.length == 8
  failures << "optimizer #{o}: NaN" if a.map(&:to_f).any?(&:nan?)
  failures << "optimizer #{o}: curve identical to adamw (axis dead)" if a == cli_chain0
  al = a.map(&:to_f)
  failures << "optimizer #{o}: did not train (#{al.first} -> #{al.last})" unless al.last < al.first - 0.01
end
Dir.mktmpdir("moe_cli_opt") do |dir|
  run_cli(%w[--steps 2 --seed 0 --optimizer muon], {}, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  fm = evs.first["franken_moe"] || {}
  failures << "optimizer: provenance #{fm['optimizer'].inspect}" unless fm["optimizer"] == "muon"
  failures << "optimizer: routing not recorded (#{fm['optimizer_routing'].inspect})" unless fm["optimizer_routing"].to_s.include?("muon:2d-hidden")
end
# muon must compose with the DFA lanes — orthogonalizing the DFA
# pseudo-gradient is the whole point of the F9m probe.
%w[dfa-experts].each do |pol|
  dl = losses(run_cli(["--steps", "8", "--seed", "0", "--optimizer", "muon", "--moe-policy", pol], {}, nil))
  failures << "optimizer muon + #{pol}: short/NaN" unless dl.length == 8 && dl.map(&:to_f).none?(&:nan?)
end
sl = losses(run_cli(%w[--steps 8 --seed 0 --optimizer muon --routing top1 --moe-policy bp-spine], {}, nil))
failures << "optimizer muon + bp-spine: short/NaN" unless sl.length == 8 && sl.map(&:to_f).none?(&:nan?)
argv_bo = [TOY, "train", "franken-moe", "--steps", "1", "--optimizer", "rmsprop"]
_ob4, sb4 = Open3.capture2e(CLEAN, *argv_bo, chdir: ROOT)
failures << "optimizer: unknown value not rejected" if sb4.success?
puts failures.length == n0 ? "  ok: --optimizer — adamw flag-null, muon + sgd train deterministically and differ, routing in provenance, DFA lanes compose" : "  FAIL: optimizer leg"

# ---- toy#140 (F10): --donor embedding transfer ----
# NOTE ON THE CRITERION: the ticket asked for "the donor curve starts
# BELOW random at step 0". It does not, and it should not — with a
# TIED embedding and a randomly-initialised spine, the step-1 loss is
# ~ln(V) either way (the logits are embedᵀ·garbage). The transfer
# signal lives in the DESCENT, and there it is unmistakable and
# seed-stable: at 20 steps the donor arm sits ~1.5-2.0 nats below
# scratch on both seeds tested. So the gate asserts THAT.
n0 = failures.length
don_base = %w[--steps 20 --seed 0 --corpus data/fineweb_gpt2_smoke.bin --context 32 --shape wide]
sc = losses(run_cli(don_base, {}, nil)).map(&:to_f)
dn_args = don_base + ["--donor", "data/distilgpt2-f32.gguf"]
dn = losses(run_cli(dn_args, {}, nil)).map(&:to_f)
dn2 = losses(run_cli(dn_args, {}, nil)).map(&:to_f)
failures << "donor: not deterministic" unless dn == dn2 && dn.length == 20
failures << "donor: NaN" if dn.any?(&:nan?)
failures << "donor: no transfer signal — donor #{dn.last.round(3)} vs scratch #{sc.last.round(3)} @20 (want >= 0.5 nats better)" unless dn.last < sc.last - 0.5
fr = losses(run_cli(dn_args + ["--freeze-embed"], {}, nil)).map(&:to_f)
failures << "donor: frozen arm NaN/short" unless fr.length == 20 && fr.none?(&:nan?)
failures << "donor: frozen arm identical to trainable (freeze had no effect)" if fr == dn
failures << "donor: frozen arm lost the transfer signal (#{fr.last.round(3)})" unless fr.last < sc.last - 0.5
Dir.mktmpdir("moe_cli_donor") do |dir|
  run_cli(don_base.take(4) + %w[--corpus data/fineweb_gpt2_smoke.bin --context 32 --shape wide
                                --donor data/distilgpt2-f32.gguf --freeze-embed], {}, dir)
  fm = (File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }.first)["franken_moe"] || {}
  failures << "donor: provenance donor #{fm['donor'].inspect}" unless fm["donor"].to_s.include?("distilgpt2")
  failures << "donor: provenance projection #{fm['donor_projection'].inspect}" unless fm["donor_projection"] == "random-jl,scale-matched"
  failures << "donor: provenance freeze_embed" unless fm["freeze_embed"] == true
end
argv_dm = [TOY, "train", "franken-moe", "--steps", "1", "--donor", "data/distilgpt2-f32.gguf", "--donor-mode", "untied"]
_od, sd = Open3.capture2e(CLEAN, *argv_dm, chdir: ROOT)
failures << "donor: --donor-mode untied not rejected" if sd.success?
argv_fe = [TOY, "train", "franken-moe", "--steps", "1", "--freeze-embed"]
_of, sf = Open3.capture2e(CLEAN, *argv_fe, chdir: ROOT)
failures << "donor: --freeze-embed without --donor not rejected" if sf.success?
puts failures.length == n0 ? "  ok: --donor — transfer signal in the DESCENT (donor #{dn.last.round(2)} vs scratch #{sc.last.round(2)} @20), frozen arm holds it, provenance, untied + bare-freeze rejected" : "  FAIL: donor leg"

# ---- toy#141: --freeze-experts (the R2 inert-experts control) ----
# THE proof obligation is not a curve, it is that the expert tensors
# NEVER MOVE. run_start and run_end both carry experts_sig (the sum of
# squares over every expert up/down tensor), so:
#   frozen   -> the two are BIT-IDENTICAL
#   dfa      -> they differ (the DFA wires do move them)
#   chain    -> they differ (BP moves them too)
# The two-sided form matters: an assertion that only checked "frozen
# doesn't move" would also pass if the signature were broken/constant.
n0 = failures.length
fz_base = %w[--steps 30 --seed 0 --experts 4 --shape wide
             --corpus data/fineweb_gpt2_smoke.bin --context 32]
sigs = {}
[["frozen", %w[--moe-policy dfa-experts --freeze-experts]],
 ["dfa",    %w[--moe-policy dfa-experts]],
 ["chain",  %w[--moe-policy chain]]].each do |name, extra|
  Dir.mktmpdir("moe_cli_fz_" + name) do |dir|
    out = run_cli(fz_base + extra, {}, dir)
    evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
    s0 = evs.first.dig("franken_moe", "experts_sig")
    s1 = evs.last["experts_sig"]
    sigs[name] = [s0, s1]
    failures << "freeze-experts: #{name} missing experts_sig" if s0.nil? || s1.nil?
    failures << "freeze-experts: #{name} provenance experts_frozen wrong" unless evs.first.dig("franken_moe", "experts_frozen") == (name == "frozen")
    failures << "freeze-experts: #{name} NaN" if losses(out).map(&:to_f).any?(&:nan?)
  end
end
failures << "freeze-experts: FROZEN experts MOVED (#{sigs['frozen'].inspect})" unless sigs["frozen"] && sigs["frozen"][0] == sigs["frozen"][1]
failures << "freeze-experts: dfa experts did NOT move — the signature is not measuring anything" unless sigs["dfa"] && sigs["dfa"][0] != sigs["dfa"][1]
failures << "freeze-experts: chain experts did NOT move — the signature is not measuring anything" unless sigs["chain"] && sigs["chain"][0] != sigs["chain"][1]
failures << "freeze-experts: all three arms start from a different init (the control is not matched)" unless sigs["frozen"] && sigs["dfa"] && sigs["chain"] && sigs["frozen"][0] == sigs["dfa"][0] && sigs["dfa"][0] == sigs["chain"][0]
puts failures.length == n0 ? "  ok: --freeze-experts — expert tensors BIT-IDENTICAL start to end (dfa + chain arms provably do move; matched init across all three)" : "  FAIL: freeze-experts leg"

# ---- toy#142 (K4): Stable LatentMoE ----
# Two axes, gated separately because they answer different questions.
# LATENT: the routed experts move into ℓ = d/2 behind W↓/W↑ with the
# stabilising RMSNorm — so the routed parameter count must FALL while
# the model still trains. SHARED: N always-on full-width experts add
# capacity back. Both compose with the DFA lanes (the expert DFA taps
# now read W↓h, not h).
n0 = failures.length
k4 = %w[--steps 12 --seed 0 --shape wide --experts 8]
Dir.mktmpdir("moe_cli_k4p") do |dp|
  run_cli(k4, {}, dp)
  base = JSON.parse(File.readlines(File.join(dp, "events.jsonl")).first)
  Dir.mktmpdir("moe_cli_k4l") do |dl|
    lat_out = run_cli(k4 + %w[--moe-latent], {}, dl)
    lat = JSON.parse(File.readlines(File.join(dl, "events.jsonl")).first)
    ll = losses(lat_out).map(&:to_f)
    failures << "latent: NaN" if ll.any?(&:nan?)
    failures << "latent: did not train (#{ll.first} -> #{ll.last})" unless ll.last < ll.first - 0.1
    failures << "latent: provenance latent_dim #{lat.dig('franken_moe', 'latent_dim').inspect} (want 128 = d/2)" unless lat.dig("franken_moe", "latent_dim") == 128
    failures << "latent: params did not fall (#{base.dig('cost', 'total_params')} -> #{lat.dig('cost', 'total_params')}) — the bottleneck is not real" unless lat.dig("cost", "total_params") < base.dig("cost", "total_params")
    # ...and so must the FLOPS. The expert flops term used d_model where
    # the params terms used ℓ, so every latent run reported plain-width
    # expert flops — the bottleneck showed up in params and vanished in
    # the number the quality-per-FLOP comparisons are made of. Corrected
    # with K4b; pinned here so it cannot drift back.
    failures << "latent: flops_per_token did not fall (#{base.dig('cost', 'flops_per_token')} -> #{lat.dig('cost', 'flops_per_token')}) — the expert flops term is not ℓ-wide" unless lat.dig("cost", "flops_per_token") < base.dig("cost", "flops_per_token")
    l2 = losses(run_cli(k4 + %w[--moe-latent], {}, nil))
    failures << "latent: not deterministic" unless losses(lat_out) == l2
    Dir.mktmpdir("moe_cli_k4s") do |ds|
      sh_out = run_cli(k4 + %w[--moe-latent --moe-shared 2], {}, ds)
      sh = JSON.parse(File.readlines(File.join(ds, "events.jsonl")).first)
      failures << "shared: provenance shared_experts #{sh.dig('franken_moe', 'shared_experts').inspect}" unless sh.dig("franken_moe", "shared_experts") == 2
      failures << "shared: params did not grow vs latent alone" unless sh.dig("cost", "total_params") > lat.dig("cost", "total_params")
      failures << "shared: curve identical to latent alone (shared experts inert)" if losses(sh_out) == losses(lat_out)
      failures << "shared: NaN" if losses(sh_out).map(&:to_f).any?(&:nan?)
    end
  end
end
# composition with the DFA lanes — the expert taps changed shape here
dl2 = losses(run_cli(k4 + %w[--moe-latent --moe-shared 2 --moe-policy dfa-experts], {}, nil))
failures << "latent + dfa-experts: short/NaN" unless dl2.length == 12 && dl2.map(&:to_f).none?(&:nan?)
sp2 = losses(run_cli(k4 + %w[--moe-latent --moe-shared 2 --routing top1 --moe-policy bp-spine], {}, nil))
failures << "latent + bp-spine: short/NaN" unless sp2.length == 12 && sp2.map(&:to_f).none?(&:nan?)
qb2 = losses(run_cli(k4 + %w[--moe-latent --moe-shared 2 --routing top1 --moe-balance qb], {}, nil))
failures << "latent + qb: short/NaN" unless qb2.length == 12 && qb2.map(&:to_f).none?(&:nan?)
puts failures.length == n0 ? "  ok: Stable LatentMoE — latent ℓ=d/2 trains and SHRINKS both routed params AND expert flops, shared experts add capacity back, provenance, composes with dfa-experts + bp-spine + qb" : "  FAIL: K4 leg"

# ---- toy#146: --lr-schedule ramp-up|ramp-down (per-layer LR) ----
# Deliberately built as a GENERAL mechanism, not a DFA one: the ramp is
# applied at apply_step, the single funnel every optimizer step passes
# through, so it reaches every credit lane and every optimizer at once.
# The legs below are mostly about THAT — a per-layer LR that only moved
# the DFA lane would be a much weaker thing than the ticket asks for.
n0 = failures.length
lg = %w[--steps 6 --seed 0 --shape deep --experts 4 --context 16
        --corpus data/fineweb_gpt2_smoke.bin]
ramp = %w[--lr-schedule ramp-up --lr-lo 0.005 --lr-hi 0.05]
down = %w[--lr-schedule ramp-down --lr-lo 0.005 --lr-hi 0.05]
base = losses(run_cli(lg + %w[--moe-policy dfa-experts], {}, nil))
failures << "lr-schedule: explicit uniform differs from the default (byte-null broken)" unless
  losses(run_cli(lg + %w[--moe-policy dfa-experts --lr-schedule uniform], {}, nil)) == base
up_l = losses(run_cli(lg + %w[--moe-policy dfa-experts] + ramp, {}, nil))
dn_l = losses(run_cli(lg + %w[--moe-policy dfa-experts] + down, {}, nil))
failures << "lr-schedule: ramp-up does not change the curve" if up_l == base
failures << "lr-schedule: ramp-down does not change the curve" if dn_l == base
# The two DIRECTIONS must differ from each other — same endpoints, opposite
# assignment. If they matched, the interpolation would be ignoring depth.
failures << "lr-schedule: ramp-up == ramp-down (direction is not reaching the layers)" if up_l == dn_l
# STRUCTURAL direction proof, not a curve comparison. Tao's F9L probe
# could not tell the directions apart because both arms had been driven
# past the stability boundary, where the LOSS saturates near ln(vocab)
# and stops discriminating. Per-layer weight movement still does: under
# ramp-up the LAST block must move more than the FIRST, and under
# ramp-down the reverse. This is the assertion that would actually catch
# a symmetric or ignored layer ordering.
%w[ramp-up ramp-down].each do |dir|
  Dir.mktmpdir("moe_lrdir_#{dir}") do |dd|
    run_cli(lg + %w[--moe-policy dfa-experts --lr-schedule] + [dir] + %w[--lr-lo 0.01 --lr-hi 0.1 --steps 20], {}, dd)
    ev = File.readlines(File.join(dd, "events.jsonl")).map { |l| JSON.parse(l) }
    s0 = ev.first.dig("franken_moe", "layer_sig") || {}
    s1 = ev.last["layer_sig"] || {}
    mv = (0...6).map { |i| ((s1["l#{i}"] || 0) - (s0["l#{i}"] || 0)).abs }
    if mv.any? { |v| v.nil? }
      failures << "lr-schedule: layer_sig missing for #{dir}"
    elsif dir == "ramp-up"
      failures << "lr-schedule: ramp-up did not move the LAST block more than the first (#{mv.first.round(1)} vs #{mv.last.round(1)})" unless mv.last > mv.first
    else
      failures << "lr-schedule: ramp-down did not move the FIRST block more than the last (#{mv.first.round(1)} vs #{mv.last.round(1)})" unless mv.first > mv.last
    end
  end
end
failures << "lr-schedule: not deterministic" unless losses(run_cli(lg + %w[--moe-policy dfa-experts] + ramp, {}, nil)) == up_l
failures << "lr-schedule: NaN" if (up_l + dn_l).map(&:to_f).any?(&:nan?)
Dir.mktmpdir("moe_lrs") do |dir|
  run_cli(lg + %w[--moe-policy dfa-experts] + ramp, {}, dir)
  fm = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)["franken_moe"]
  failures << "lr-schedule: provenance #{fm['lr_schedule'].inspect}" unless fm["lr_schedule"] == "ramp-up"
  # The RESOLVED per-layer LRs are the contract: endpoints exact and
  # monotonic in depth. Asserting the curve moved would not catch an
  # interpolation that is off by a layer or lands on the wrong endpoints.
  pl = fm["lr_per_layer"] || {}
  vals = (0...6).map { |i| pl["l#{i}"] }
  failures << "lr-schedule: per-layer LRs missing (#{pl.inspect})" if vals.any?(&:nil?)
  unless vals.any?(&:nil?)
    failures << "lr-schedule: layer 0 is #{vals.first} (want lo=0.005)" unless (vals.first - 0.005).abs < 1e-9
    failures << "lr-schedule: last layer is #{vals.last} (want hi=0.05)" unless (vals.last - 0.05).abs < 1e-9
    failures << "lr-schedule: not monotonic ascending (#{vals.inspect})" unless vals.each_cons(2).all? { |a, b| b > a }
  end
  Dir.mktmpdir("moe_lrs_d") do |d2|
    run_cli(lg + %w[--moe-policy dfa-experts] + down, {}, d2)
    pd = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first).dig("franken_moe", "lr_per_layer") || {}
    dv = (0...6).map { |i| pd["l#{i}"] }
    failures << "lr-schedule: ramp-down not monotonic descending (#{dv.inspect})" unless
      dv.none?(&:nil?) && dv.each_cons(2).all? { |a, b| b < a }
  end
end
# GENERICITY — the point of building it at apply_step. Every lane and
# every optimizer must see the ramp, not just the DFA lane.
[["chain",     %w[--moe-policy chain]],
 ["block-dfa", %w[--moe-policy dfa-experts --dfa-granularity block]],
 ["muon",      %w[--moe-policy dfa-experts --optimizer muon]],
 ["sgd",       %w[--moe-policy dfa-experts --optimizer sgd]],
 ["top1",      %w[--routing top1 --moe-policy bp-spine]],
 ["latent",    %w[--moe-policy dfa-experts --moe-latent --expert-act situ-glu]]].each do |name, lane|
  b = losses(run_cli(lg + lane, {}, nil))
  r = losses(run_cli(lg + lane + ramp, {}, nil))
  failures << "lr-schedule: no effect on the #{name} lane (the ramp is not generic)" if r.empty? || b == r
end
# Guards, all fail-loud.
[[%w[--lr-schedule ramp-up], "ramp without --lr-lo/--lr-hi"],
 [%w[--lr-lo 0.005 --lr-hi 0.05], "--lr-lo/--lr-hi without a ramp"],
 [%w[--lr-schedule sawtooth --lr-lo 0.005 --lr-hi 0.05], "unknown schedule value"]].each do |extra, what|
  _o, st = Open3.capture2e(CLEAN, TOY, "train", "franken-moe", "--steps", "1",
                           "--shape", "deep", *extra, chdir: ROOT)
  failures << "lr-schedule: #{what} not rejected" if st.success?
end
# A ramp needs depth to interpolate across; at L=1 it is not a ramp.
_o1, st1 = Open3.capture2e(CLEAN, TOY, "train", "franken-moe", "--steps", "1",
                           "--lr-schedule", "ramp-up", "--lr-lo", "0.005", "--lr-hi", "0.05", chdir: ROOT)
failures << "lr-schedule: ramp at L=1 not rejected" if st1.success?

# toy#147: top1 at depth used to abort (the toy#145 regression); the
# per-layer aux root fixed it. Pinned here as the POSITIVE contract, in
# every balance mode, so it cannot silently regress into a crash again.
%w[plain aux qb].each do |mode|
  args = %w[--steps 4 --seed 0 --shape deep --experts 4 --context 16
            --corpus data/fineweb_gpt2_smoke.bin --routing top1]
  args += %w[--moe-aux 0.05] if mode == "aux"
  args += %w[--moe-balance qb] if mode == "qb"
  out = losses(run_cli(args, {}, nil))
  failures << "deep-top1(#{mode}): did not run at L=6" if out.empty?
  failures << "deep-top1(#{mode}): NaN" if out.map(&:to_f).any?(&:nan?)
  failures << "deep-top1(#{mode}): not deterministic" unless losses(run_cli(args, {}, nil)) == out
end
# ...and the per-layer balancing machinery must be LIVE at depth, not
# inert: switching each balance mode on has to change the curve. An
# aux/QB path that quietly balanced only the last block would still
# "run", which is exactly what this leg exists to rule out.
dt = %w[--steps 4 --seed 0 --shape deep --experts 4 --context 16
        --corpus data/fineweb_gpt2_smoke.bin --routing top1]
plain_l = losses(run_cli(dt, {}, nil))
failures << "deep-top1: --moe-aux has no effect at depth" if losses(run_cli(dt + %w[--moe-aux 0.05], {}, nil)) == plain_l
failures << "deep-top1: --moe-balance qb has no effect at depth" if losses(run_cli(dt + %w[--moe-balance qb], {}, nil)) == plain_l
puts failures.length == n0 ? "  ok: --lr-schedule ramp-up/ramp-down — uniform byte-null, both directions move the curve and differ, endpoints exact + monotonic in provenance, and the PER-LAYER WEIGHT MOVEMENT mirrors the LR profile (ramp-up moves the last block most, ramp-down the first) — direction proven structurally, not by curve; GENERIC across chain/block-dfa/muon/sgd/top1/latent; 5 guards reject" : "  FAIL: lr-schedule leg"

# ---- toy#150: --dfa-feedback kolen-pollack (adaptive DFA feedback) ----
# Every DFA experiment through F9m used a FIXED random B. This is the
# adaptive alternative: B stands in for the effective output path P, and
# grad_P L = e . a_out^T, so B <- B - eta*(e a_out^T) - eta*lambda*B —
# Kolen-Pollack with decay, which is exactly ggml's SGD step.
#
# THE LOAD-BEARING ASSERTION IS dfa_b_sig, not the curve. A coupling
# that silently never fires produces a perfectly healthy loss curve;
# only a signature over the feedback matrices can tell "B moved" from
# "B is still at its random init" (the K4b/experts_sig lesson, third
# time it has mattered).
#
# GAINS ARE EXPLICIT HERE, deliberately. The ticket's default (eta tied
# to --lr) DIVERGES at this shape — measured 1000x growth in ||B||^2
# over 20 steps at --lr 0.02 — so the gate pins the MECHANISM at a
# stable operating point and leaves finding the regime to F13. No
# assertion about final loss: that is the experiment's to make.
n0 = failures.length
kg = %w[--steps 20 --seed 0 --shape deep --experts 4 --context 16
        --corpus data/fineweb_gpt2_smoke.bin --moe-policy dfa-experts]
kp = %w[--dfa-feedback kolen-pollack --dfa-feedback-lr 1e-4 --dfa-feedback-decay 0.01]

kbase = losses(run_cli(kg, {}, nil))
failures << "dfa-feedback: explicit `fixed` differs from absent (byte-null broken)" unless
  losses(run_cli(kg + %w[--dfa-feedback fixed], {}, nil)) == kbase
kpl = losses(run_cli(kg + kp, {}, nil))
failures << "dfa-feedback: kolen-pollack did not run" if kpl.length != 20
failures << "dfa-feedback: kolen-pollack does not change the curve (the coupling is inert)" if kpl == kbase
failures << "dfa-feedback: NaN under kolen-pollack" if kpl.map(&:to_f).any?(&:nan?)
failures << "dfa-feedback: not deterministic" unless losses(run_cli(kg + kp, {}, nil)) == kpl

Dir.mktmpdir("moe_kp_fix") do |df|
  Dir.mktmpdir("moe_kp_on") do |dk|
    run_cli(kg + %w[--dfa-feedback fixed], {}, df)
    run_cli(kg + kp, {}, dk)
    ef = File.readlines(File.join(df, "events.jsonl")).map { |l| JSON.parse(l) }
    ek = File.readlines(File.join(dk, "events.jsonl")).map { |l| JSON.parse(l) }
    # FIXED: B must be BIT-IDENTICAL start to end — the frozen control.
    f0 = ef.first.dig("franken_moe", "dfa_b_sig")
    f1 = ef.last["dfa_b_sig"]
    if f0.nil? || f1.nil?
      failures << "dfa-feedback: dfa_b_sig missing (start=#{f0.inspect} end=#{f1.inspect})"
    else
      failures << "dfa-feedback: B MOVED under `fixed` (#{f0} -> #{f1}) — the fixed-B control is broken" unless f0 == f1
    end
    # KOLEN-POLLACK: B must provably move, and stay BOUNDED at these gains.
    k0 = ek.first.dig("franken_moe", "dfa_b_sig")
    k1 = ek.last["dfa_b_sig"]
    if k0.nil? || k1.nil?
      failures << "dfa-feedback: dfa_b_sig missing on the kp arm"
    else
      failures << "dfa-feedback: B did NOT move under kolen-pollack (#{k0} -> #{k1}) — builds, reports a loss, never adapts" unless (k1 - k0).abs > 1e-9
      failures << "dfa-feedback: ||B||^2 grew #{(k1 / k0).round(1)}x at the pinned gains — the coupling is diverging, not tracking" unless k1 / k0 < 2.0
      failures << "dfa-feedback: matched init broken (#{f0} vs #{k0})" unless (f0 - k0).abs < 1e-9
    end
    # The FEEDBACK-TRACKING read: |cos(B, head)| must RISE. This is the
    # one place cos(B, P) is honestly computable — the last layer's down
    # feedback, whose path to the logits IS the tied head.
    c0 = ek.first.dig("franken_moe", "dfa_b_cos_head")
    c1 = ek.last["dfa_b_cos_head"]
    if c0.nil? || c1.nil?
      failures << "dfa-feedback: dfa_b_cos_head missing"
    else
      failures << "dfa-feedback: cos(B, head) did not rise (#{c0} -> #{c1}) — B moves but not toward the forward path" unless c1.abs > c0.abs
    end
    fm = ek.first["franken_moe"]
    failures << "dfa-feedback: provenance mode #{fm['dfa_feedback'].inspect}" unless fm["dfa_feedback"] == "kolen-pollack"
    failures << "dfa-feedback: provenance does not record WHICH rule ran" unless fm["dfa_feedback_rule"] == "path"
    failures << "dfa-feedback: provenance gains #{fm['dfa_feedback_lr'].inspect}/#{fm['dfa_feedback_decay'].inspect}" unless
      (fm["dfa_feedback_lr"] - 1e-4).abs < 1e-12 && (fm["dfa_feedback_decay"] - 0.01).abs < 1e-12
    failures << "dfa-feedback: `fixed` provenance not 'fixed'" unless ef.first.dig("franken_moe", "dfa_feedback") == "fixed"
  end
end
# COMPOSES WITH block-DFA — the granularity F13's arm uses. This leg is
# why the composition exists at all: wiring the coupling inside the
# surrogate-root loop ran cleanly and produced a curve BYTE-IDENTICAL to
# fixed-B, because extend_backward_graph has nothing to attach to before
# tnn_build_backward. The gate caught it; the fix was placement.
#
# Note block-DFA couples b_blks, NOT the expert B's (the experts are
# chain-wired from the surrogate root there), so dfa_b_sig has to cover
# b_blks or this arm reports "B frozen" while its curve provably moves.
kb  = losses(run_cli(kg + %w[--dfa-granularity block], {}, nil))
kbk = losses(run_cli(kg + %w[--dfa-granularity block] + kp, {}, nil))
failures << "dfa-feedback: no effect under --dfa-granularity block (the block feedback matrix is not coupled)" if kbk.empty? || kb == kbk
failures << "dfa-feedback: block arm short/NaN" unless kbk.length == 20 && kbk.map(&:to_f).none?(&:nan?)
Dir.mktmpdir("moe_kp_blk") do |dbk|
  run_cli(kg + %w[--dfa-granularity block] + kp, {}, dbk)
  eb = File.readlines(File.join(dbk, "events.jsonl")).map { |l| JSON.parse(l) }
  b0 = eb.first.dig("franken_moe", "dfa_b_sig")
  b1 = eb.last["dfa_b_sig"]
  if b0.nil? || b1.nil?
    failures << "dfa-feedback: dfa_b_sig missing on the block arm"
  else
    failures << "dfa-feedback: B did NOT move under block-DFA (#{b0} -> #{b1}) — dfa_b_sig must cover b_blks" unless (b1 - b0).abs > 1e-9
    failures << "dfa-feedback: ||B||^2 grew #{(b1 / b0).round(1)}x under block-DFA — diverging, not tracking" unless b1 / b0 < 2.0
  end
  cb0 = eb.first.dig("franken_moe", "dfa_b_cos_head")
  cb1 = eb.last["dfa_b_cos_head"]
  if cb0.nil? || cb1.nil?
    failures << "dfa-feedback: dfa_b_cos_head missing on the block arm"
  else
    failures << "dfa-feedback: cos(B, head) did not rise under block-DFA (#{cb0} -> #{cb1})" unless cb1.abs > cb0.abs
  end
end
# Guards, fail-loud.
[[%w[--dfa-feedback-decay 0.01], "gains without kolen-pollack"],
 [%w[--dfa-feedback adaptive], "unknown --dfa-feedback value"],
 [%w[--dfa-feedback kolen-pollack --dfa-feedback-lr 0], "zero feedback LR (a fixed-B run wearing the kp label)"]].each do |extra, what|
  _o, st = Open3.capture2e(CLEAN, TOY, "train", "franken-moe", "--steps", "1",
                           "--shape", "deep", *extra, chdir: ROOT)
  failures << "dfa-feedback: #{what} not rejected" if st.success?
end
puts failures.length == n0 ? "  ok: --dfa-feedback kolen-pollack — fixed byte-null AND B bit-identical, B provably moves + stays bounded, cos(B,head) RISES, provenance names the rule, composes with block-DFA (b_blks coupled, telemetry follows it), 3 guards" : "  FAIL: dfa-feedback leg"

# ---- toy#148: --lr-control reactive (the loss-reactive LR damper) ----
# toy#146 is the static per-layer SHAPE; this is the dynamic global
# SCALE. Composition is lr_t x lr_mul[l] x ctrl_t, and the legs below
# pin that identity rather than inferring it from a curve.
#
# WHY THE DAMPING CLAIM IS GATED ON WEIGHT MOVEMENT, NOT ON THE LOSS.
# The ticket asks for "the controlled back-half is provably damped".
# Measured at gate shape, back-half loss VARIANCE only falls to ~0.87x
# even with the damper pinned at its floor — because the corpus feed
# rotates a fresh window every step, so most of the loss variance is
# data sampling noise the LR cannot touch. Gating on that would be a
# 13%-margin single-seed assertion, i.e. a flaky gate dressed up as a
# result. Per-block WEIGHT MOVEMENT is the same quantity the damper
# actually controls, it is deterministic, and it falls to ~0.11x. That
# is the F9L/toy#146 lesson applied again: when the loss saturates or
# is noise-dominated, assert on the structure it failed to discriminate.
n0 = failures.length
cg = %w[--steps 40 --seed 0 --shape deep --experts 4 --context 16
        --corpus data/fineweb_gpt2_smoke.bin --moe-policy dfa-experts
        --dfa-granularity block --lr 0.08]
ctl = %w[--lr-control reactive --lr-control-window 3 --lr-control-patience 1
         --lr-control-factor 0.3 --lr-control-recover 1.0 --lr-control-floor 0.02]
# short config for the cheap curve legs
sg = %w[--steps 12 --seed 0 --shape deep --experts 4 --context 16
        --corpus data/fineweb_gpt2_smoke.bin --moe-policy dfa-experts --lr 0.05]
sctl = %w[--lr-control reactive --lr-control-window 2 --lr-control-patience 1
          --lr-control-factor 0.5]

# FLAG-NULL: an explicit `none` must be byte-identical to absent. ctrl
# is a literal 1.0 and lr_t * 1.0 is bit-exact, so this is a property of
# the construction — pinned so a later refactor cannot quietly break it.
cbase = losses(run_cli(sg, {}, nil))
failures << "lr-control: explicit none differs from the default (byte-null broken)" unless
  losses(run_cli(sg + %w[--lr-control none], {}, nil)) == cbase
creact = losses(run_cli(sg + sctl, {}, nil))
failures << "lr-control: reactive does not change the curve" if creact == cbase
failures << "lr-control: not deterministic" unless losses(run_cli(sg + sctl, {}, nil)) == creact
failures << "lr-control: NaN" if creact.map(&:to_f).any?(&:nan?)

# The two 40-step arms carry every structural assertion below.
Dir.mktmpdir("moe_ctl_u") do |du|
  Dir.mktmpdir("moe_ctl_c") do |dc|
    run_cli(cg, {}, du)
    run_cli(cg + ctl, {}, dc)
    eu = File.readlines(File.join(du, "events.jsonl")).map { |l| JSON.parse(l) }
    ec = File.readlines(File.join(dc, "events.jsonl")).map { |l| JSON.parse(l) }
    su = eu.select { |e| e["kind"] == "step" }
    sc = ec.select { |e| e["kind"] == "step" }

    # TELEMETRY: ctrl and lr_eff on every step, in BOTH arms (an
    # uncontrolled run reporting ctrl=1.0 is what makes the two
    # directly comparable).
    failures << "lr-control: ctrl missing from step events" unless
      sc.length == 40 && sc.all? { |e| e.key?("ctrl") && e.key?("lr_eff") }
    failures << "lr-control: uncontrolled arm does not report ctrl=1.0" unless
      su.all? { |e| e["ctrl"] == 1.0 }
    # THE COMPOSITION IDENTITY — this is what proves the damper reaches
    # the optimizer rather than merely being reported.
    failures << "lr-control: lr_eff != lr * ctrl (the damper is not composing)" unless
      sc.all? { |e| (e["lr_eff"] - e["lr"] * e["ctrl"]).abs < 1e-12 }
    # THE DAMPER FIRES, and the floor is respected.
    ctrls = sc.map { |e| e["ctrl"] }
    failures << "lr-control: ctrl never dropped below 1.0 (the damper did not fire)" unless
      ctrls.min < 1.0
    failures << "lr-control: ctrl punched through the floor (#{ctrls.min} < 0.02)" if ctrls.min < 0.02 - 1e-12
    failures << "lr-control: ctrl exceeded 1.0 (#{ctrls.max})" if ctrls.max > 1.0 + 1e-12
    re = ec.last
    failures << "lr-control: run_end lr_ctrl_cuts not positive (#{re['lr_ctrl_cuts'].inspect})" unless
      re["lr_ctrl_cuts"].to_i > 0
    failures << "lr-control: run_end lr_ctrl_final disagrees with the step stream" unless
      re["lr_ctrl_final"] && (re["lr_ctrl_final"] - ctrls.last).abs < 1e-9

    # DAMPED, structurally: the controlled arm moves its weights strictly
    # less. Threshold 0.5 against a measured 0.11 — a real margin, not a
    # coin flip. Deliberately NOT an assertion about the final loss: the
    # gate must not assume F9M's science outcome.
    mv = lambda do |ev|
      s0 = ev.first.dig("franken_moe", "layer_sig") || {}
      s1 = ev.last["layer_sig"] || {}
      (0...6).map { |i| ((s1["l#{i}"] || 0) - (s0["l#{i}"] || 0)).abs }
    end
    mu = mv.call(eu)
    mc = mv.call(ec)
    if mu.sum <= 0
      failures << "lr-control: uncontrolled arm did not move (layer_sig flat) — the comparison is void"
    else
      ratio = mc.sum / mu.sum
      failures << "lr-control: damper did not shrink weight movement (ratio #{ratio.round(3)}, want < 0.5)" unless ratio < 0.5
    end

    # PROVENANCE: mode + every constant, so a bundle says which
    # controller ran without re-reading the runner.
    fm = ec.first["franken_moe"]
    failures << "lr-control: provenance mode #{fm['lr_control'].inspect}" unless fm["lr_control"] == "reactive"
    { "lr_control_window" => 3, "lr_control_patience" => 1, "lr_control_factor" => 0.3,
      "lr_control_recover" => 1.0, "lr_control_floor" => 0.02 }.each do |k, want|
      failures << "lr-control: provenance #{k} = #{fm[k].inspect} (want #{want})" unless
        fm[k] && (fm[k] - want).abs < 1e-12
    end
    # window=3 -> alpha=2/4; recorded so "window" is not left ambiguous.
    failures << "lr-control: provenance lr_control_alpha #{fm['lr_control_alpha'].inspect}" unless
      fm["lr_control_alpha"] && (fm["lr_control_alpha"] - 0.5).abs < 1e-12
    failures << "lr-control: uncontrolled arm provenance is not 'none'" unless
      eu.first.dig("franken_moe", "lr_control") == "none"
  end
end

# GENERICITY, same argument as toy#146: the damper multiplies lr_t at
# the same funnel, so every lane and every optimizer must see it —
# including the toy#146 ramp it is designed to compose with.
[["chain",     %w[--moe-policy chain]],
 ["block-dfa", %w[--moe-policy dfa-experts --dfa-granularity block]],
 ["muon",      %w[--moe-policy dfa-experts --optimizer muon]],
 ["sgd",       %w[--moe-policy dfa-experts --optimizer sgd]],
 ["top1",      %w[--routing top1 --moe-policy bp-spine]],
 ["latent",    %w[--moe-policy dfa-experts --moe-latent --expert-act situ-glu]],
 ["ramp-down", %w[--moe-policy dfa-experts --lr-schedule ramp-down --lr-lo 0.005 --lr-hi 0.05]]].each do |name, lane|
  bl = %w[--steps 12 --seed 0 --shape deep --experts 4 --context 16
          --corpus data/fineweb_gpt2_smoke.bin --lr 0.05] + lane
  b = losses(run_cli(bl, {}, nil))
  r = losses(run_cli(bl + sctl, {}, nil))
  failures << "lr-control: no effect on the #{name} lane (the damper is not generic)" if r.empty? || b == r
end

# Guards, all fail-loud. A factor of 2.0 ("double the LR") silently
# doing the opposite of a damper is the exact class this prevents.
[[%w[--lr-control-window 50], "controller params without --lr-control reactive"],
 [%w[--lr-control none --lr-control-factor 0.5], "controller params with an explicit none"],
 [%w[--lr-control aggressive], "unknown --lr-control value"],
 [%w[--lr-control reactive --lr-control-factor 2.0], "factor >= 1 (not a cut)"],
 [%w[--lr-control reactive --lr-control-factor 0], "factor 0"],
 [%w[--lr-control reactive --lr-control-recover 0.9], "recover < 1 (not a restore)"],
 [%w[--lr-control reactive --lr-control-floor 1.5], "floor > 1"],
 [%w[--lr-control reactive --lr-control-floor 0], "floor 0"],
 [%w[--lr-control reactive --lr-control-window 0], "window 0"],
 [%w[--lr-control reactive --lr-control-patience 0], "patience 0"]].each do |extra, what|
  _o, st = Open3.capture2e(CLEAN, TOY, "train", "franken-moe", "--steps", "1",
                           "--shape", "deep", *extra, chdir: ROOT)
  failures << "lr-control: #{what} not rejected" if st.success?
end
# recipe scoping: franken-moe only.
_o2, st2 = Open3.capture2e(CLEAN, TOY, "train", "franken", "--steps", "1",
                           "--lr-control", "reactive", chdir: ROOT)
failures << "lr-control: accepted on the 'franken' recipe" if st2.success?
puts failures.length == n0 ? "  ok: --lr-control reactive — damper fires, floor held, lr_eff == lr x ctrl, weight movement damped ~9x, generic across lanes, provenance + run_end, guards fail loud" : "  FAIL: lr-control leg"

# ---- toy#145: --shape deep — the DEPTH lever for block-DFA ----
# The tower is now L repeats of (attention + MoE) rather than one of
# each. Two things need pinning: that L=1 is unchanged (covered by every
# other leg in this file plus the rig null), and that L=6 is REAL depth
# — six routed-expert blocks, six feedback matrices, cost and signature
# scaling with them. A "deep" that quietly built one block would look
# healthy on a loss curve and answer nothing.
n0 = failures.length
dp = %w[--steps 4 --seed 0 --shape deep --experts 4 --context 16
        --corpus data/fineweb_gpt2_smoke.bin]
wd = %w[--steps 4 --seed 0 --shape wide --experts 4 --context 16
        --corpus data/fineweb_gpt2_smoke.bin]
deep_l = losses(run_cli(dp, {}, nil))
wide_l = losses(run_cli(wd, {}, nil))
failures << "deep: not deterministic" unless losses(run_cli(dp, {}, nil)) == deep_l
failures << "deep: NaN" if deep_l.map(&:to_f).any?(&:nan?)
failures << "deep: identical to wide (depth is not reaching the graph)" if deep_l == wide_l
Dir.mktmpdir("moe_deep") do |dir|
  run_cli(dp, {}, dir)
  dj = JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)
  failures << "deep: provenance n_layers #{dj.dig('model', 'n_layers').inspect} (want 6)" unless dj.dig("model", "n_layers") == 6
  failures << "deep: provenance shape #{dj.dig('model', 'shape').inspect}" unless dj.dig("model", "shape") == "deep"
  # deep holds wide's WIDTH, so a deep-vs-wide comparison isolates depth.
  failures << "deep: d_model #{dj.dig('model', 'd_model').inspect} (want 256, wide's width)" unless dj.dig("model", "d_model") == 256
  Dir.mktmpdir("moe_deep_w") do |d2|
    run_cli(wd, {}, d2)
    wj = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)
    failures << "deep: provenance n_layers at wide #{wj.dig('model', 'n_layers').inspect} (want 1)" unless wj.dig("model", "n_layers") == 1
    # Cost must scale with depth EXACTLY. Not "dt > k*wt": the tied
    # embedding is shared across layers and at vocab 50257 it dominates
    # the total, so a loose ratio would either pass trivially or fail a
    # correct implementation. Subtract the depth-INDEPENDENT part
    # (embedding + final norm; 2*vocab*d for flops, the tied logits) and
    # assert the remainder is exactly 6x.
    voc = dj.dig("model", "vocab"); dmd = dj.dig("model", "d_model")
    emb = voc * dmd + dmd
    lg  = 2 * voc * dmd
    dt = dj.dig("cost", "total_params"); wt = wj.dig("cost", "total_params")
    failures << "deep: per-layer params not 6x wide's (#{wt - emb} -> #{dt - emb})" unless (dt - emb) == 6 * (wt - emb)
    df = dj.dig("cost", "flops_per_token"); wf = wj.dig("cost", "flops_per_token")
    failures << "deep: per-layer flops not 6x wide's (#{wf - lg} -> #{df - lg})" unless (df - lg) == 6 * (wf - lg)
    # experts_sig must cover EVERY layer's experts — otherwise the
    # freeze proof and the whole depth comparison read one block.
    failures << "deep: experts_sig not larger than wide's (deeper layers are not in the signature)" unless
      dj.dig("franken_moe", "experts_sig") > wj.dig("franken_moe", "experts_sig")
  end
end
# The ticket's target lane: block-DFA at depth. It must train, be
# distinct from matmul-DFA at the same shape, and move the experts.
bd = dp + %w[--moe-policy dfa-experts --dfa-granularity block]
mm = dp + %w[--moe-policy dfa-experts]
bl = losses(run_cli(bd, {}, nil))
failures << "deep: block-DFA NaN" if bl.map(&:to_f).any?(&:nan?)
failures << "deep: block-DFA identical to matmul-DFA at depth" if bl == losses(run_cli(mm, {}, nil))
Dir.mktmpdir("moe_deep_bd") do |dir|
  run_cli(bd + %w[--steps 10], {}, dir)
  ev = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  s0 = ev.first.dig("franken_moe", "experts_sig"); s1 = ev.last["experts_sig"]
  failures << "deep: block-DFA experts did not move at L=6" unless s0 && s1 && s0 != s1
end
# --freeze-experts must still be bit-exact when there are SIX blocks of
# experts to leave alone.
Dir.mktmpdir("moe_deep_fz") do |dir|
  run_cli(dp + %w[--steps 10 --moe-policy dfa-experts --freeze-experts], {}, dir)
  ev = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  failures << "deep: --freeze-experts moved experts_sig at L=6" unless
    ev.first.dig("franken_moe", "experts_sig") == ev.last["experts_sig"]
end
# Composition the ticket asks for.
[%w[--batch 2], %w[--moe-latent], %w[--expert-act situ-glu], %w[--lr 0.005]].each do |extra|
  o = losses(run_cli(dp + %w[--moe-policy dfa-experts] + extra, {}, nil))
  failures << "deep: does not compose with #{extra.join(' ')}" if o.empty? || o.map(&:to_f).any?(&:nan?)
end
puts failures.length == n0 ? "  ok: --shape deep — L=6 x (attention + MoE) at wide's width, distinct from wide, provenance dims, cost AND experts_sig scale with depth; block-DFA spans six blocks (trains, distinct from matmul, experts move), freeze still bit-exact, composes with batch/latent/situ-glu/lr" : "  FAIL: deep leg"

# ---- K4b / M6: --expert-act situ-glu (SiTU-GLU experts) ----
# A GLU needs a SECOND projection per expert, which the 9+2i/10+2i pair
# has no room for, so the E gate matrices are APPENDED at the very tail
# (past the spine tail) and every existing index keeps its meaning. The
# legs below are about exactly that: the new weights must be EXPERT
# weights everywhere it matters, not a fourth thing the bookkeeping
# forgets.
n0 = failures.length
gg = %w[--steps 6 --seed 0 --moe-policy dfa-experts]
base_g = losses(run_cli(gg, {}, nil))
failures << "situ-glu: explicit --expert-act gelu differs from the default (byte-null broken)" unless
  losses(run_cli(gg + %w[--expert-act gelu], {}, nil)) == base_g
glu = losses(run_cli(gg + %w[--expert-act situ-glu], {}, nil))
failures << "situ-glu: curve identical to gelu (the gate branch is not in the graph)" if glu == base_g
failures << "situ-glu: not deterministic" unless losses(run_cli(gg + %w[--expert-act situ-glu], {}, nil)) == glu
failures << "situ-glu: NaN" if glu.map(&:to_f).any?(&:nan?)
# Both other lanes must see it too — top1 dispatches the gate branch
# through the SAME ids2 selection as up.
[%w[--moe-policy chain], %w[--routing top1 --moe-aux 0.05]].each do |lane|
  b = losses(run_cli(%w[--steps 6 --seed 0] + lane, {}, nil))
  g = losses(run_cli(%w[--steps 6 --seed 0] + lane + %w[--expert-act situ-glu], {}, nil))
  failures << "situ-glu: no effect on lane #{lane.join(' ')}" if b == g
end
Dir.mktmpdir("moe_glu") do |dir|
  run_cli(gg + %w[--expert-act situ-glu], {}, dir)
  ev = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  fm = ev.first["franken_moe"]
  failures << "situ-glu: provenance expert_act #{fm['expert_act'].inspect}" unless fm["expert_act"] == "situ-glu"
  # experts_sig MUST include the gate branch — otherwise --freeze-experts
  # could "prove" nothing moved while the gate trained behind it.
  Dir.mktmpdir("moe_glu_ref") do |d2|
    run_cli(gg, {}, d2)
    rf = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)["franken_moe"]
    failures << "situ-glu: experts_sig unchanged by the extra expert matrix (gate branch NOT counted)" if
      rf["experts_sig"] == fm["experts_sig"]
    # cost must grow: three matrices per expert, not two.
    c1 = ev.first["cost"]; c0 = JSON.parse(File.readlines(File.join(d2, "events.jsonl")).first)["cost"]
    failures << "situ-glu: total_params did not grow (#{c0['total_params']} -> #{c1['total_params']})" unless
      c1["total_params"] > c0["total_params"]
    failures << "situ-glu: flops_per_token did not grow" unless c1["flops_per_token"] > c0["flops_per_token"]
  end
end
# TWO-SIDED freeze, the toy#141 shape: frozen must be BIT-IDENTICAL
# (which only holds if the gate branch is both counted AND frozen),
# while dfa and chain provably move from the SAME init.
sigs = {}
%w[frozen dfa chain].each do |arm|
  args = %w[--steps 15 --seed 0 --expert-act situ-glu]
  args += arm == "chain" ? %w[--moe-policy chain] : %w[--moe-policy dfa-experts]
  args += %w[--freeze-experts] if arm == "frozen"
  Dir.mktmpdir("moe_glu_#{arm}") do |dir|
    run_cli(args, {}, dir)
    ev = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
    sigs[arm] = [ev.first["franken_moe"]["experts_sig"], ev.last["experts_sig"]]
  end
end
failures << "situ-glu: --freeze-experts moved experts_sig #{sigs['frozen'].inspect} (the gate branch is training behind the freeze)" unless
  sigs["frozen"][0] == sigs["frozen"][1]
failures << "situ-glu: dfa arm did not move" if sigs["dfa"][0] == sigs["dfa"][1]
failures << "situ-glu: chain arm did not move" if sigs["chain"][0] == sigs["chain"][1]
failures << "situ-glu: arms do not share an init" unless
  sigs["frozen"][0] == sigs["dfa"][0] && sigs["dfa"][0] == sigs["chain"][0]
# Composition with the other K4 axis and with bp-spine.
[%w[--moe-latent], %w[--moe-latent --moe-shared 1]].each do |extra|
  o = losses(run_cli(gg + %w[--expert-act situ-glu] + extra, {}, nil))
  failures << "situ-glu: does not compose with #{extra.join(' ')}" if o.empty? || o.map(&:to_f).any?(&:nan?)
end
sp = losses(run_cli(%w[--steps 6 --seed 0 --routing top1 --moe-policy bp-spine --expert-act situ-glu], {}, nil))
failures << "situ-glu: does not compose with bp-spine" if sp.empty? || sp.map(&:to_f).any?(&:nan?)
_og, stg = Open3.capture2e(CLEAN, TOY, "train", "franken-moe", "--steps", "1",
                           "--expert-act", "swiglu", chdir: ROOT)
failures << "situ-glu: unknown --expert-act value not rejected" if stg.success?
puts failures.length == n0 ? "  ok: --expert-act situ-glu — gelu byte-null, gate branch live on all three lanes, appended at the tail with every existing index intact; counted in experts_sig AND cost (3 matrices/expert), freeze two-sided, composes with latent/shared/bp-spine" : "  FAIL: situ-glu leg"

# ---- toy#143: --dfa-granularity block (the literature recipe) ----
# The ticket's two proof obligations, plus the guards. NOTE what is
# NOT asserted: which arm WINS. At gate scale the arms sit within
# noise of each other and the ordering is not stable across horizons,
# so pinning a winner here would pin noise — that comparison belongs
# to the F9c/F9d fixture, not to a gate.
n0 = failures.length
gb = %w[--steps 40 --seed 0 --shape wide --experts 8 --moe-policy dfa-experts
        --corpus data/fineweb_gpt2_smoke.bin --context 32]
Dir.mktmpdir("moe_cli_blk") do |dir|
  blk_out = run_cli(gb + %w[--dfa-granularity block], {}, dir)
  ev = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  s0 = ev.first.dig("franken_moe", "experts_sig")
  s1 = ev.last["experts_sig"]
  # (1) block-DFA experts must PROVABLY move — the whole point is that
  # they train, unlike the frozen control.
  failures << "block-dfa: experts did NOT move (#{s0} -> #{s1}) — the block loss root is not reaching them" unless s0 && s1 && s0 != s1
  failures << "block-dfa: provenance #{ev.first.dig('franken_moe', 'dfa_granularity').inspect}" unless ev.first.dig("franken_moe", "dfa_granularity") == "block"
  failures << "block-dfa: NaN" if losses(blk_out).map(&:to_f).any?(&:nan?)
  b2 = losses(run_cli(gb + %w[--dfa-granularity block], {}, nil))
  failures << "block-dfa: not deterministic" unless losses(blk_out) == b2
  # (2) it must be a DISTINCT graph from matmul-DFA, not a rename.
  mm = losses(run_cli(gb, {}, nil))
  failures << "block-dfa: curve identical to matmul granularity (same graph)" if losses(blk_out) == mm
  failures << "block-dfa: explicit --dfa-granularity matmul differs from the default (flag-null broken)" unless losses(run_cli(gb + %w[--dfa-granularity matmul], {}, nil)) == mm
end
[["--routing", "top1"], ["--moe-policy", "chain"]].each do |k, v|
  argv = [TOY, "train", "franken-moe", "--steps", "1", "--dfa-granularity", "block", k, v]
  argv += %w[--moe-policy dfa-experts] if k == "--routing"
  _o, st = Open3.capture2e(CLEAN, *argv, chdir: ROOT)
  failures << "block-dfa: #{k} #{v} not rejected" if st.success?
end
argv_bf = [TOY, "train", "franken-moe", "--steps", "1", "--moe-policy", "dfa-experts",
           "--dfa-granularity", "block", "--freeze-experts"]
_obf, sbf = Open3.capture2e(CLEAN, *argv_bf, chdir: ROOT)
failures << "block-dfa: + --freeze-experts not rejected" if sbf.success?
puts failures.length == n0 ? "  ok: --dfa-granularity block — experts provably move via the block loss root, distinct graph from matmul, deterministic, provenance; top1/chain/freeze rejected" : "  FAIL: block-dfa leg"

# ---- 6. bundle structure + run-id passthrough ----
n0 = failures.length
Dir.mktmpdir("moe_cli_bundle") do |dir|
  run_cli(%w[--steps 5 --seed 0 --routing top1 --moe-aux 0.05], {}, dir)
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    evs = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = evs.first || {}
    failures << "bundle: first event not run_start (toy/v1)" unless rs["kind"] == "run_start" && rs["schema"] == "toy/v1"
    failures << "bundle: run_id passthrough lost (got #{rs['run_id'].inspect})" unless rs["run_id"] == "moe-cli-gate"
    fm = rs["franken_moe"]
    if fm.nil?
      failures << "bundle: run_start has no franken_moe object"
    else
      failures << "bundle: wrong provenance #{fm.inspect}" unless fm["routing"] == "top1" && fm["aux_alpha"] == 0.05 && fm["b_seed"] == 1234
    end
    co = rs["cost"]
    if co.nil?
      failures << "bundle: run_start has no cost object (toy#129)"
    else
      failures << "bundle: cost fields not positive (#{co.inspect})" unless %w[total_params active_params flops_per_token].all? { |k| co[k].is_a?(Numeric) && co[k] > 0 }
      failures << "bundle: top1 active_params not < total (#{co.inspect})" unless co["active_params"] < co["total_params"]
    end
    failures << "bundle: #{evs.count { |e| e['kind'] == 'step' }} step events (want 5)" unless evs.count { |e| e["kind"] == "step" } == 5
    failures << "bundle: #{evs.count { |e| e['kind'] == 'route' }} route events (want 5)" unless evs.count { |e| e["kind"] == "route" } == 5
    failures << "bundle: last event not run_end" unless evs.last && evs.last["kind"] == "run_end"
  else
    failures << "bundle: no events.jsonl"
  end
  fj = File.join(dir, "flow.json")
  flow = File.file?(fj) ? (JSON.parse(File.read(fj)) rescue nil) : nil
  failures << "bundle: flow.json missing/invalid" if flow.nil? || flow["format"] != "toy/v1" || !flow["nodes"].is_a?(Array) || flow["nodes"].empty?
  puts failures.length == n0 ? "  ok: bundle — franken_moe provenance + run-id passthrough + 5 step/route + run_end + flow.json" : "  FAIL: bundle leg"
end

if failures.empty?
  puts "GATE PASS [franken-moe-cli]: rig-null + seed semantics + dfa-experts/align + top1 collapse/aux legs + bp-router + bp-spine/detach + shape-wide (toy#124) + corpus/vocab-627 (toy#125) + align-every (toy#127) + experts (toy#128) + no-shadow/pack-header (toy#129) + eval-ce + eval-freeze + eval-mem (toy#130/#149) + lr/warmup (toy#132) + batch (toy#133) + ckpt/load (toy#131) + qb/schedule/attn-gate (toy#136) + optimizer (toy#139) + donor (toy#140) + freeze-experts (toy#141) + latent-moe (toy#142) + block-dfa (toy#143) + situ-glu experts (K4b/M6) + shape-deep (toy#145) + lr-schedule (toy#146) + deep-top1 per-layer routers (toy#147) + lr-control reactive (toy#148) + kolen-pollack feedback (toy#150) + bundle (toy#120/#121)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken-moe-cli]: #{f}" }
  exit 1
end
