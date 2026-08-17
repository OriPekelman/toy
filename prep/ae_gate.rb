#!/usr/bin/env ruby
# prep/ae_gate.rb — toy#165 (capstone P1a) gate for the LATENT
# AUTOENCODER lane (libexec/toy-train-ae, `toy train ae`).
#
# Legs:
#   1. DETERMINISM.
#   2. THE PROBE IS BYTE-NULL at sigma 0 — the measurement graph IS the
#      training graph, or none of the arms below compare anything.
#   3. THE LANE CAN LEARN (roomy latent reaches near-ceiling).
#   4. CONTROL-CAN-LOSE — the MANDATORY precondition (tao#19).
#   5. THE MARGIN IS THE READ, AND CLEAN ACCURACY IS NOT.
#   6. THE ALPHABET IS AN AXIS (tao#22) — provenance, and it moves.
#   7. MATCHED-CE STOPPING is byte-null on the training curve.
#   8. FAIL-LOUD.
#   9. CLI.
#
# ── WHAT THIS LANE ANSWERS ──
#
# The diffusion-text-LM capstone needs a per-token continuous latent of
# 4-8 dims (F20/toy#156's DFA-favourable window). Whether text survives
# such a latent is unrun. P1a is the cheapest decisive form and it is ALL
# BP — no DFA, no diffusion; those are P1c and P1b.
#
# Measured while building, 600 steps, context 128, d_model 128, seed 0,
# on data/ae_names (27 distinct bytes):
#
#   latent   clean acc   half-accuracy SNR
#      4       0.993          0.81
#      8       1.000          1.22
#     32       1.000         >=2.0
#
# Two things to read off that table, and the second is the one that
# shapes this gate:
#
#  1. CLEAN RECONSTRUCTION IS VACUOUS. It is 0.99+ at EVERY latent down
#     to 4 — packing 27 codepoints into 4 continuous dims is ample analog
#     capacity. A lane that reported clean accuracy as its headline would
#     have announced "a 4-dim per-token latent carries text" on the
#     strength of a number that cannot distinguish 4 from 32.
#  2. THE NOISE MARGIN DOES discriminate, monotonically, and by a lot:
#     0.81 -> 1.22 -> off the top of the grid. That is why the half-SNR
#     scalar and not clean accuracy is what leg 5 asserts on.
#
# ── WHAT THIS GATE DELIBERATELY DOES NOT ASSERT ──
#
# The ticket proposes gating "noise-margin monotone in d at EVERY SNR".
# That is an empirical prediction, not a determinism property, and at the
# grid's ends it will flake: adjacent dims tie wherever both are near
# ceiling or near floor, and a tie is not a bug. What IS asserted here is
# the far-apart comparison (d=32's margin clears d=4's by a wide fixed
# margin) plus byte-identical re-runs as the hard determinism leg. A
# monotonicity violation in between is a finding to look at, not a gate
# failure — see the toy#165 thread.
#
# ── THE ZEROED CONTROL IS NOT THE GATED ONE, AND THAT MATTERS ──
#
# With a per-position decoder, a zeroed latent leaves the head with only
# its bias, so it lands at the unigram floor BY CONSTRUCTION whether or
# not training worked. Gating on it would be gating on an identity. The
# SHUFFLED control is the one with teeth: each position decodes a real
# latent from the same distribution, just the wrong one, so it can score
# above the floor if the head learned a prior that survives the
# permutation. That is leg 4.
#
# (The head bias is itself a finding from the first smoke run: WITHOUT
# it a zeroed latent gives an all-zero logit vector, argmax breaks the
# tie at class 0, and the control reads ~0.000 — flatteringly far below
# a floor the head was structurally unable to reach. See ae_engine.rb.)

ROOT   = File.expand_path("..", __dir__)
RUNNER = File.join(ROOT, "libexec", "toy-train-ae")
TOY    = File.join(ROOT, "bin", "toy")
PACK   = File.join(ROOT, "data", "ae_names")
PACK_U = File.join(ROOT, "data", "ae_udhr")

require "open3"
require "json"
require "tmpdir"

# The cheap gate cell, calibrated so the legs below have real headroom:
# at 400 steps it reads clean 0.934 / half-SNR 0.79 at latent 4 and
# clean 1.000 / half-SNR >=2.0 at latent 32.
CELL = {
  "AE_TEXT" => PACK, "AE_CONTEXT" => "64", "AE_D_MODEL" => "64",
  "AE_D_FF" => "128", "AE_BLOCKS" => "2", "AE_VAL_BATCHES" => "4",
}.freeze
STEPS_CELL = "400"

# Margins sit far under the measured effects (clean 1.000 vs shuffle
# 0.160 vs floor 0.145; half-SNR 2.0 vs 0.79).
CEILING       = 0.85   # a roomy latent must essentially solve reconstruction
CONTROL_EDGE  = 0.50   # ... and clear the shuffled control by this
FLOOR_SLACK   = 0.10   # the shuffled control must sit within this of the floor
MARGIN_SPREAD = 0.50   # half-SNR(32) - half-SNR(4)

def run_ae(extra_env, run_dir = nil)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(CELL).merge(extra_env)
  env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "ae-gate") if run_dir
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "ae_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

def field(out, prefix, key)
  line = out.lines.find { |l| l.start_with?(prefix) }
  raise "ae_gate: no #{prefix.inspect} line in\n#{out}" unless line
  line[/#{Regexp.escape(key)}=([0-9.eE+-]+)/, 1]
end

# The half-SNR is deliberately NOT a number when the curve never falls to
# half of clean inside the grid — it is reported as ">=<max sigma>". A
# gate that parsed that as its numeric part would be re-introducing the
# clamp the runner refuses to make, so read it as "at least".
def half_snr(out)
  line = out.lines.find { |l| l.start_with?("half_snr:") }
  raise "ae_gate: no half_snr line in\n#{out}" unless line
  v = line.split(":", 2)[1].strip
  v.start_with?(">=") ? [v[2..].split(/\s/).first.to_f, true] : [v.to_f, false]
end

def noise_points(out)
  out.lines.select { |l| l.start_with?("noise:") }.map do |l|
    [l[/sigma=([0-9.eE+-]+)/, 1].to_f, l[/acc=([0-9.eE+-]+)/, 1]]
  end
end

unless %w[.meta.i32 .tok.i32].all? { |s| File.file?(PACK + s) }
  warn "ae_gate: #{PACK} missing — run: ruby prep/fetch_text.rb --all"
  exit 2
end
unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-ae")
  unless build_st.success? && File.executable?(RUNNER)
    warn "ae_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []
# d878143: every leg records the failure count at its START in `n0`.
n0 = 0

# ---- 1. determinism ----
n0 = failures.length
d1 = run_ae({ "STEPS" => "20", "AE_LATENT" => "8" })
d2 = run_ae({ "STEPS" => "20", "AE_LATENT" => "8" })
failures << "determinism: two identical runs differ" unless d1 == d2
d_seed = run_ae({ "STEPS" => "20", "AE_LATENT" => "8", "SEED" => "1" })
failures << "seed inert: --seed did not move the curve — the init is not reaching the weights" if curve(d_seed) == curve(d1)
d_lat = run_ae({ "STEPS" => "20", "AE_LATENT" => "4" })
failures << "latent inert: --latent did not move the curve — the bottleneck is not the swept axis it is reported as" if curve(d_lat) == curve(d1)
puts failures.length == n0 ?
  "  ok: two identical runs are byte-identical; --seed and --latent both move the curve" :
  "  FAIL: determinism"

# ---- 2. the probe is BYTE-NULL at sigma 0 ----
#
# The clean pass and the sigma=0 noise pass are the SAME graph under
# different uploads. If they disagree by so much as a digit, the noise
# arms are measuring something other than the trained model and every
# number below is uninterpretable.
n0 = failures.length
p0 = run_ae({ "STEPS" => "60", "AE_LATENT" => "8" })
clean = field(p0, "val: ", "clean_acc")
zero_sigma = noise_points(p0).find { |s, _| s.zero? }
if zero_sigma.nil?
  failures << "probe byte-null: no sigma=0 point in the noise grid"
else
  failures << "probe byte-null: sigma=0 acc #{zero_sigma[1]} != clean_acc #{clean} — the measurement graph is not the training graph" unless zero_sigma[1] == clean
end
# ... and a sigma that is NOT zero must move it, or the noise tensor is
# not reaching the latent and the whole margin curve is a flat line
# drawn through the clean number.
pts = noise_points(p0)
failures << "probe effect: every sigma gives the same accuracy — the noise upload is not reaching the latent" if pts.map(&:last).uniq.length == 1
puts failures.length == n0 ?
  "  ok: sigma=0 reproduces clean reconstruction EXACTLY (one graph, three uploads) and non-zero sigmas move it" :
  "  FAIL: probe byte-null"

# ---- 3. the lane can learn at all ----
n0 = failures.length
big = run_ae({ "STEPS" => STEPS_CELL, "AE_LATENT" => "32" })
big_clean = field(big, "val: ", "clean_acc").to_f
failures << "ceiling: latent 32 reaches only clean_acc=#{big_clean} (< #{CEILING}) — the lane cannot learn reconstruction, so nothing below means anything" if big_clean < CEILING
puts failures.length == n0 ?
  "  ok: a roomy latent (32) reaches clean_acc=#{big_clean} — the lane learns reconstruction" :
  "  FAIL: ceiling"

# ---- 4. CONTROL-CAN-LOSE — the mandatory precondition (tao#19) ----
n0 = failures.length
shuffle = field(big, "control: ", "shuffle_acc").to_f
zeroed  = field(big, "control: ", "zero_acc").to_f
floor   = field(big, "control: ", "unigram_floor").to_f
failures << "control: shuffled-latent acc #{shuffle} is within #{CONTROL_EDGE} of clean #{big_clean} — the decode is NOT reading the latent, it is reading a prior, and the lane measures nothing" if big_clean - shuffle < CONTROL_EDGE
failures << "control: shuffled-latent acc #{shuffle} sits more than #{FLOOR_SLACK} above the unigram floor #{floor} — the head has learned a prior that survives the permutation" if shuffle > floor + FLOOR_SLACK
failures << "control: zeroed-latent acc #{zeroed} EXCEEDS the unigram floor #{floor} — with no latent the head has only its bias, so beating the floor means something other than the latent is carrying information" if zeroed > floor + FLOOR_SLACK
puts failures.length == n0 ?
  "  ok: CONTROL CAN LOSE — shuffled latent #{shuffle} sits at the unigram floor #{floor} while clean is #{big_clean} (tao#19)" :
  "  FAIL: control-can-lose"

# ---- 5. the margin is the read, and clean accuracy is NOT ----
n0 = failures.length
small = run_ae({ "STEPS" => STEPS_CELL, "AE_LATENT" => "4" })
small_clean = field(small, "val: ", "clean_acc").to_f
sm, sm_open = half_snr(small)
bg, bg_open = half_snr(big)
# The lane's own justification for its headline: if clean accuracy DID
# separate the arms, the noise machinery would be unnecessary. It does
# not — and asserting that keeps the write-up honest if the task ever
# drifts to something where it would.
failures << "vacuity: clean accuracy separates latent 4 (#{small_clean}) from latent 32 (#{big_clean}) by more than #{CONTROL_EDGE} — if clean recon really discriminated, the noise-margin framing would need restating rather than assuming" if (big_clean - small_clean).abs > CONTROL_EDGE
failures << "margin: half-SNR at latent 32 (#{bg}#{bg_open ? '+' : ''}) does not clear latent 4 (#{sm}) by #{MARGIN_SPREAD} — the noise margin is not tracking the bottleneck, which is the lane's entire premise" if bg - sm < MARGIN_SPREAD
# The undefined case must be REPORTED as ">=", never clamped: a clamped
# value reads as a measurement and would make a wide latent look like it
# collapsed exactly where the grid happened to stop.
tight = run_ae({ "STEPS" => STEPS_CELL, "AE_LATENT" => "32", "AE_NOISE_EVAL" => "0,0.1" })
tl = tight.lines.find { |l| l.start_with?("half_snr:") }.to_s
failures << "half-SNR: a grid the curve never crosses reported #{tl.strip.inspect} instead of a >= form — a clamped scalar reads as a measurement (never-mask)" unless tl.include?(">=")
puts failures.length == n0 ?
  "  ok: clean accuracy does NOT discriminate (#{small_clean} vs #{big_clean}) while half-SNR does (#{sm} vs #{bg}#{bg_open ? '+' : ''}), and an uncrossed grid reports >= rather than clamping" :
  "  FAIL: margin"

# ---- 6. THE ALPHABET IS AN AXIS (tao#22) ----
#
# The margin is packing-limited, so a curve without the alphabet it was
# measured at is unscoped: at N=27 the d=4 problem is about as hard as
# N=256 at d=8, and the alphabet alone could manufacture a `go`.
n0 = failures.length
cline = big.lines.find { |l| l.start_with?("corpus:") }.to_s
%w[pack= n_tokens= alphabet= val_alphabet= unigram_floor=].each do |k|
  failures << "provenance: corpus line lacks #{k} (#{cline.strip})" unless cline.include?(k)
end
if File.file?(PACK_U + ".meta.i32")
  udhr = run_ae({ "STEPS" => "5", "AE_LATENT" => "8", "AE_TEXT" => PACK_U })
  a_names = field(big,  "corpus: ", "alphabet").to_i
  a_udhr  = field(udhr, "corpus: ", "alphabet").to_i
  v_names = field(big,  "corpus: ", "val_alphabet").to_i
  v_udhr  = field(udhr, "corpus: ", "val_alphabet").to_i
  failures << "alphabet axis: ae_names and ae_udhr report the same pack alphabet (#{a_names}) — the axis is not real" if a_names == a_udhr
  failures << "alphabet axis: the SCORED windows report the same alphabet on both corpora (#{v_names}) — the effective alphabet is not being measured on the windows" if v_names == v_udhr
  failures << "alphabet axis: ae_udhr's alphabet #{a_udhr} is not larger than ae_names' #{a_names}" unless a_udhr > a_names
else
  failures << "alphabet axis: #{PACK_U} missing — run: ruby prep/fetch_text.rb --all"
end
Dir.mktmpdir("ae_gate") do |dir|
  out = run_ae({ "STEPS" => "5", "AE_LATENT" => "8" }, dir)
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = events.first || {}
    failures << "bundle: first event not run_start" unless rs["kind"] == "run_start"
    failures << "bundle: run_start.name != ae (#{rs['name'].inspect})" unless rs["name"] == "ae"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    md = rs["model"] || {}
    failures << "bundle: model.arch != ae (#{md['arch'].inspect})" unless md["arch"] == "ae"
    failures << "bundle: model.decoder != per_position (#{md['decoder'].inspect}) — a decoder that sees context would be measuring language modelling, not latent capacity" unless md["decoder"] == "per_position"
    failures << "bundle: model.latent_dim missing" unless md["latent_dim"]
    cp = rs["corpus"] || {}
    %w[pack alphabet val_alphabet unigram_floor entropy_bits].each do |k|
      failures << "bundle: corpus.#{k} missing — the margin curve is unscoped without it (tao#22)" unless cp.key?(k)
    end
    nm = events.find { |e| e["name"] == "noise_margin" }
    if nm.nil?
      failures << "bundle: no noise_margin eval event — the lane's actual result is not in the bundle"
    else
      failures << "bundle: noise_margin.sigmas/accs length mismatch" unless nm["sigmas"].is_a?(Array) && nm["accs"].is_a?(Array) && nm["sigmas"].length == nm["accs"].length
      failures << "bundle: noise_margin lacks val_alphabet" unless nm.key?("val_alphabet")
    end
  else
    failures << "bundle: no events.jsonl"
  end
end
puts failures.length == n0 ?
  "  ok: the corpus and its EFFECTIVE alphabet ride stdout and the bundle, and the alphabet axis moves between corpora (tao#22)" :
  "  FAIL: alphabet axis / bundle"

# ---- 7. matched-CE stopping (toy#165 follow-up) ----
#
# The published surface compared cells at matched STEPS and their clean CE
# then spanned SEVEN ORDERS OF MAGNITUDE, which a noise MARGIN is not
# invariant to (it inflates once accuracy saturates and shrinks while
# accuracy is still improving). --target-ce makes the cells comparable.
#
# The load-bearing property is that the probe COSTS NOTHING in numerics:
# ggml's adamw kernel updates m and v even at lr=0, so a mid-training val
# pass on the ordinary eval hp would quietly corrupt Adam's state. The
# probe hp uses beta1=beta2=1.0, making both moment updates the identity.
# Asserted, not asserted-about: a probed run's curve must be BYTE-IDENTICAL.
n0 = failures.length
unprobed = run_ae({ "STEPS" => "120", "AE_LATENT" => "8" })
probed   = run_ae({ "STEPS" => "120", "AE_LATENT" => "8",
                    "AE_TARGET_CE" => "0.0000000001", "AE_EVAL_EVERY" => "10" })
failures << "target-ce byte-null: probing every 10 steps CHANGED the training curve — the probe is perturbing optimizer state (ggml's adamw updates m/v even at lr=0; the probe hp must use beta1=beta2=1.0)" unless curve(probed) == curve(unprobed)
# ... and an unreachable target must run to STEPS and say so, rather than
# reporting an unmatched cell as matched.
cline = probed.lines.find { |l| l.start_with?("converged:") }.to_s
failures << "target-ce: an unreachable target did not report matched=0 (#{cline.strip})" unless cline.include?("matched=0")
failures << "target-ce: an unreachable target did not say NOT REACHED (#{cline.strip})" unless cline.include?("NOT REACHED")
# A reachable target must stop EARLY and land at or under it.
hit = run_ae({ "STEPS" => "4000", "AE_LATENT" => "8",
               "AE_TARGET_CE" => "0.05", "AE_EVAL_EVERY" => "25" })
hline = hit.lines.find { |l| l.start_with?("converged:") }.to_s
if hline.include?("matched=1")
  steps_used = hline[/steps=(\d+)/, 1].to_i
  achieved   = hline[/achieved_ce=([0-9.eE+-]+)/, 1].to_f
  failures << "target-ce: stopped at step #{steps_used} of 4000 — no early stop happened" unless steps_used < 4000
  failures << "target-ce: achieved_ce #{achieved} exceeds the target it claims to have matched" if achieved > 0.05
  # The full-set CE is the authoritative matched quantity; the probe runs
  # on a subset, so the two must at least agree in magnitude or the
  # stopping criterion is measuring something else.
  full = field(hit, "val: ", "loss").to_f
  failures << "target-ce: probe CE #{achieved} and full held-out CE #{full} differ by more than 3x — the subset probe is not tracking the quantity the surface is matched on" if full > 3.0 * achieved || achieved > 3.0 * full
else
  failures << "target-ce: a reachable target (0.05) was not matched (#{hline.strip})"
end
puts failures.length == n0 ?
  "  ok: --target-ce stops early and lands under target, an unreachable one reports matched=0 LOUDLY, and probing is BYTE-NULL on the training curve (beta1=beta2=1 makes the adamw moment update the identity)" :
  "  FAIL: matched-CE stopping"

# ---- 8. fail-loud ----
n0 = failures.length
[
  [{ "AE_TEXT" => "" },                                     "no corpus (there is no synthetic fallback, by design)"],
  [{ "AE_TEXT" => File.join(ROOT, "data", "nope") },        "nonexistent pack"],
  [{ "AE_LATENT" => "64", "AE_D_MODEL" => "64" },           "latent >= d_model (not a bottleneck)"],
  [{ "AE_D_MODEL" => "65", "AE_HEADS" => "4" },             "d_model not a multiple of heads"],
  [{ "AE_VAL_FRAC_PCT" => "0" },                            "val fraction 0 (no held-out set)"],
  [{ "AE_VAL_FRAC_PCT" => "90" },                           "val fraction 90 (train span starved)"],
  [{ "AE_CONTEXT" => "1" },                                 "context 1"],
  [{ "AE_NOISE_EVAL" => "-1" },                             "negative sigma"],
  [{ "AE_TARGET_CE" => "0.05" },                            "--target-ce with no --eval-every (never checked)"],
  [{ "AE_EVAL_EVERY" => "10" },                             "--eval-every with no --target-ce (cost, no effect)"],
  [{ "AE_TARGET_CE" => "0.05", "AE_EVAL_EVERY" => "10", "AE_PROBE_BATCHES" => "0" }, "probe batches 0"],
].each do |env, label|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(CELL).merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did something else)" if st.success?
end
# A pack whose meta disagrees with its tokens would mislabel every
# alphabet-axis reading taken off it.
Dir.mktmpdir("ae_gate_bad") do |dir|
  bad = File.join(dir, "bad")
  File.binwrite(bad + ".meta.i32", [4096, 99].pack("l<*"))
  File.binwrite(bad + ".tok.i32", Array.new(4096) { |i| i % 7 }.pack("l<*"))
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(CELL).merge("AE_TEXT" => bad), RUNNER, chdir: ROOT)
  failures << "fail-loud: a pack whose meta claims 99 symbols but contains 7 was accepted" if st.success?
  failures << "fail-loud: the alphabet-mismatch message does not name both counts" unless out.include?("99") && out.include?("7 distinct")
end
puts failures.length == n0 ?
  "  ok: 12 degenerate configs fail loud, including a pack whose declared alphabet disagrees with its tokens" :
  "  FAIL: fail-loud"

# ---- 9. CLI ----
n0 = failures.length
cli_out, cli_st = Open3.capture2e({ "SPINEL_DIR" => ENV["SPINEL_DIR"].to_s },
  TOY, "train", "ae", "--text", PACK, "--latent", "8", "--context", "64",
  "--d-model", "64", "--d-ff", "128", "--layers", "2", "--val-batches", "4",
  "--steps", "20", chdir: ROOT)
if cli_st.success?
  %w[corpus: noise: half_snr: control:].each do |k|
    failures << "cli: `toy train ae` output lacks the #{k.inspect} line — the lane's result would be reachable only by opening a bundle" unless cli_out.include?(k)
  end
else
  failures << "cli: `toy train ae` exited #{cli_st.exitstatus}:\n#{cli_out.lines.last(6).join}"
end
[
  [%w[ae --latent 8],                                  "--text omitted"],
  [%w[ae --text data/ae_names --device cuda],              "--device cuda on a CPU-only lane"],
  [%w[ae --text data/nope],                                "--text naming a nonexistent pack"],
  [%w[ae --text data/ae_names --vocab 256],                "--vocab on the ae lane (it means an integer pack width on franken)"],
  [%w[ae --text data/ae_names --latent 64 --d-model 64], "--latent >= --d-model"],
  # toy#169/#170 widened --text to the byte-LM lanes, so the rule is no
  # longer "ae/difflm only" — it is that the flag must reach a task that
  # READS it. gtx/ssm consume it only under --task bytelm; without that
  # the runner ignores it, which is the silent no-op this leg guards.
  [%w[gtx --text data/ae_names],                           "--text on gtx WITHOUT --task bytelm (the runner would ignore it)"],
  [%w[mlp --text data/ae_names],                           "--text on a lane that has no text task at all"],
  [%w[ae --text data/ae_names --noise-eval abc],           "--noise-eval that is not a float list"],
].each do |args, label|
  out, st = Open3.capture2e(TOY, "train", *args, "--steps", "2", chdir: ROOT)
  failures << "cli: #{label} exited 0 (the flag silently did nothing)" if st.success?
end
puts failures.length == n0 ?
  "  ok: `toy train ae` carries the four result lines, and 7 misuses are refused — including --vocab, which means an INTEGER on the franken lanes" :
  "  FAIL: cli"

if failures.empty?
  puts "GATE PASS [ae]: per-token latent autoencoder (capstone P1a, all BP) — one graph for training and every probe (sigma=0 reproduces clean EXACTLY), a SHUFFLED control that provably CAN lose at the unigram floor, the noise margin discriminating the bottleneck where clean reconstruction cannot, the effective alphabet reported as the second axis (tao#22), and an uncrossed grid reported as >= rather than clamped (toy#165)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [ae]: #{f}" }
  exit 1
end
