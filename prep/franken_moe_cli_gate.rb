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
  puts "GATE PASS [franken-moe-cli]: rig-null + seed semantics + dfa-experts/align + top1 collapse/aux legs + bp-router + bp-spine/detach + bundle (toy#120/#121)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken-moe-cli]: #{f}" }
  exit 1
end
