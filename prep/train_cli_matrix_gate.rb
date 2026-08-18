#!/usr/bin/env ruby
# prep/train_cli_matrix_gate.rb — toy#132: the flag x recipe matrix,
# asserted against reality. Every matrix row must REJECT under a wrong
# recipe (exit 2, "only valid with", pre-build — no runner is spawned),
# and representative right-recipe probes must get PAST the matrix (their
# error, when any, comes from a LATER conditional rule instead).
#
# This is the gate leg the toy#132 process note asked for: recipe
# parity became one cell flip in lib/toy/core/cli/train.rb's
# flag_matrix; this gate keeps the table honest.

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")

require "open3"

CLEAN = { "SPINEL_DIR" => nil, "SPINEL_SKIP_PIN_CHECK" => nil }

def probe(args)
  out, st = Open3.capture2e(CLEAN, TOY, "train", *args, chdir: ROOT)
  [out, st]
end

failures = []
# LEG BOOKKEEPING: every leg records the failure count at its START in
# `n0` and summarises with `failures.length == n0`, so each leg reports
# on ITS OWN assertions. Legs used to summarise with `failures.empty?`,
# which made every later leg print FAIL once ANY earlier leg had failed
# — misleading exactly when you are debugging. `n0` is seeded at top
# level so re-assignments inside blocks mutate the outer local.
n0 = 0
n0 = failures.length

# ---- rejection side: one wrong-recipe probe per matrix row ----
REJECT = [
  [%w[from-scratch --model x.gguf],       "--model/--rank"],
  [%w[from-scratch --corpus x.bin],       "--corpus"],
  [%w[from-scratch --init scratch],       "--init"],
  [%w[from-scratch --dfa-b-seed 1],       "--dfa-b-*"],
  [%w[from-scratch --policy chain],       "--policy"],
  [%w[from-scratch --align-every 2],      "--align-every"],
  [%w[from-scratch --lr 0.01],            "--lr/--warmup"],
  [%w[vit-tiny --warmup 2],               "--lr/--warmup"],
  [%w[warm-start --no-shadow],            "--no-shadow"],
  [%w[from-scratch --context 64],         "--context/--vocab"],
  [%w[from-scratch --batch 4],            "--batch"],
  [%w[from-scratch --act situ-glu],       "--act"],
  [%w[franken-moe --act situ-glu],        "--act"],
  [%w[from-scratch --rope nope],          "--rope"],
  [%w[from-scratch --schedule cosine],    "--schedule"],
  [%w[franken --moe-balance qb],          "--moe-balance"],
  [%w[franken --attn-gate],               "--attn-gate"],
  [%w[franken-moe --kda-layers 0],        "--kda-layers"],
  [%w[franken-moe --layer-pattern hybrid], "--layer-pattern"],
  [%w[from-scratch --no-kda-conv],        "--no-kda-conv"],
  [%w[franken-moe --attnres],             "--attnres"],
  [%w[from-scratch --optimizer muon],     "--optimizer"],
  [%w[franken --donor x.gguf],            "--donor"],
  [%w[franken --freeze-experts],          "--freeze-experts"],
  [%w[franken --moe-latent],              "--moe-latent"],
  # toy#158 moved --dfa-granularity onto the franken lane too, so the
  # wrong-recipe probe has to be a recipe that has NO DFA at all.
  [%w[from-scratch --dfa-granularity block], "--dfa-granularity"],
  [%w[franken-moe --optimizer radam],     "--optimizer radam (franken-only)"],
  [%w[franken-moe --mla-layers 1],        "--mla-layers/--mla-rank"],
  [%w[franken --expert-act situ-glu],     "--expert-act"],
  [%w[franken --lr-schedule ramp-up],     "--lr-schedule/--lr-lo/--lr-hi"],
  [%w[franken --lr-control reactive],     "--lr-control/--lr-control-*"],
  [%w[franken --dfa-feedback kolen-pollack], "--dfa-feedback/--dfa-feedback-*"],
  [%w[from-scratch --policy-scope ffn], "--policy-scope"],
  # tao#18 item 1: --policy-scope is DELIBERATELY not accepted on the
  # mlp recipe — the attn|ffn|all meaning stays stable across lanes.
  [%w[mlp --policy-scope ffn],            "--policy-scope"],
  [%w[from-scratch --classes 10],         "--classes/--hidden/--features/--layers"],
  [%w[franken --hidden 64],               "--classes/--hidden/--features/--layers"],
  [%w[from-scratch --task teacher],       "--task/--task-seed/--teacher-dim/--val-batches"],
  [%w[franken-moe --val-batches 4],       "--task/--task-seed/--teacher-dim/--val-batches"],
  [%w[mlp --shape wide],                  "--shape"],
  [%w[from-scratch --ckpt-every 2],       "--ckpt-every"],
  [%w[franken --load-ckpt x.gguf],        "--load-ckpt"],
  [%w[franken --eval-corpus x.bin],       "--eval-corpus"],
  [%w[from-scratch --shape wide],         "--shape"],
  [%w[franken --routing top1],            "--routing"],
  [%w[franken --experts 4],               "--experts"],
  # toy#153 (gnn). --feedback-route is DELIBERATELY a different name
  # from franken-moe's --dfa-feedback: one selects how B is UPDATED,
  # the other how the error is ROUTED. Probing BOTH directions is the
  # point — a lane-local flag leaking into a neighbouring lane is how
  # a swept command line silently records a cell it never ran.
  [%w[from-scratch --graph data/gnn_cora], "--graph"],
  [%w[mlp --graph data/gnn_cora],          "--graph"],
  [%w[franken --nodes 512],                "--nodes/--degree/--homophily/--feat-signal"],
  [%w[mlp --degree 8],                     "--nodes/--degree/--homophily/--feat-signal"],
  [%w[ctr --train-per-class 20],           "--train-per-class"],
  [%w[franken-moe --feedback-route structure], "--feedback-route/--feedback-hops"],
  [%w[mlp --feedback-hops 2],              "--feedback-route/--feedback-hops"],
  [%w[franken --weight-decay 0.01],        "--weight-decay"],
  [%w[gnn --policy-scope ffn],             "--policy-scope"],
  [%w[gnn --val-batches 4],                "--val-batches"],
  [%w[gnn --fm-branch],                    "--base-rate/--lin-scale/--fm-branch"],
  # toy#155 (ssm). --dfa-cut is NOT --dfa-granularity: franken's picks
  # matmul-vs-block in DEPTH, this picks layer-vs-step in TIME. And
  # --align-events is rejected on ssm on purpose — its DFA update lands
  # in the same accumulator a BP run uses, so a cosine would be telemetry
  # that silently means nothing.
  [%w[gnn --seq 32],                       "--seq"],
  [%w[franken --d-inner 32],               "--d-model/--d-inner/--conv-k"],
  [%w[mlp --selection lti],                "--selection"],
  [%w[mlp --dfa-cut step],                 "--dfa-cut"],
  [%w[ctr --cue-span 8],                   "--cue-span/--noise"],
  [%w[ssm --align-events],                 "--align-events"],
  [%w[ssm --policy-scope ffn],             "--policy-scope"],
  [%w[ssm --hidden 32],                    "--hidden"],
  [%w[ssm --features 16],                  "--features"],
  [%w[ssm --graph data/gnn_cora],          "--graph"],
  # toy#157 (lstm). It SHARES the sequence knobs with ssm — same task
  # generator, same cut axis, which is what makes the two lanes one
  # architecture comparison — but NOT the selective-scan-shaped ones.
  # Both directions are probed: a --d-inner/--selection/--dt-init that
  # leaked onto the LSTM lane would name a part it does not have, and
  # --align-events is rejected here for the same reason as on ssm (the
  # DFA update lands in the accumulator a BP run uses, so a cosine
  # against it would be telemetry that silently means nothing).
  [%w[lstm --selection lti],               "--selection"],
  [%w[lstm --d-inner 32],                  "--d-model/--d-inner/--conv-k"],
  [%w[lstm --conv-k 2],                    "--d-model/--d-inner/--conv-k"],
  [%w[lstm --dt-init -4.0],                "--dt-init"],
  [%w[mlp --dt-init -4.0],                 "--dt-init"],
  [%w[lstm --align-events],                "--align-events"],
  [%w[lstm --policy-scope ffn],            "--policy-scope"],
  [%w[lstm --graph data/gnn_cora],         "--graph"],
  [%w[lstm --latent 8],                    "--latent/--time-feat"],
  # toy#156 (diff). --latent is the OUTPUT DIM under test and has its
  # own name rather than riding --classes: this lane regresses epsilon.
  [%w[mlp --latent 8],                     "--latent/--time-feat"],
  [%w[ssm --time-feat 4],                  "--latent/--time-feat"],
  [%w[gnn --modes 4],                      "--modes/--spread/--mode-scale"],
  [%w[ctr --spread 2.0],                   "--modes/--spread/--mode-scale"],
  [%w[franken --diff-steps 50],            "--diff-steps/--beta-lo/--beta-hi"],
  [%w[mlp --beta-hi 0.2],                  "--diff-steps/--beta-lo/--beta-hi"],
  [%w[ssm --eval-n 256],                   "--eval-n"],
  [%w[diff --policy-scope ffn],            "--policy-scope"],
  [%w[diff --dfa-cut step],                "--dfa-cut"],
  [%w[diff --seq 32],                      "--seq/--d-model/--d-inner/--conv-k"],
  [%w[diff --graph data/gnn_cora],         "--graph"],
  # toy#160 (gtx). --entities is NOT --nodes (the gnn lane's graph size)
  # and the pairs-per-step count is NOT --pairs (the ctr lane's feature
  # crosses) — a different meaning gets a different name, and both
  # directions are probed. --dfa-cut IS shared: toy#160 puts attention on
  # the same layer|step axis the recurrent lanes use.
  [%w[mlp --heads 4],                      "--heads/--d-ff/--types/--entities"],
  [%w[ssm --entities 32],                  "--heads/--d-ff/--types/--entities"],
  [%w[lstm --types 4],                     "--heads/--d-ff/--types/--entities"],
  [%w[gnn --d-ff 128],                     "--heads/--d-ff/--types/--entities"],
  [%w[gtx --align-events],                 "--align-events"],
  [%w[gtx --policy-scope ffn],             "--policy-scope"],
  [%w[gtx --selection lti],                "--selection"],
  [%w[gtx --seq 32],                       "--seq"],
  [%w[gtx --dt-init -4.0],                 "--dt-init"],
  [%w[gtx --cue-span 8],                   "--cue-span"],
  [%w[gtx --graph data/gnn_cora],          "--graph"],
  [%w[gtx --hidden 32],                    "--hidden"],
  [%w[gtx --latent 8],                     "--latent"],
  # toy#165 (ae). Two deliberate asymmetries in this block.
  #
  # --latent IS shared with diff, because it is the SAME quantity on both
  # lanes: F20/toy#156's "latent 4" and P1a's bottleneck 4 are the number
  # the capstone compares directly, and two names for one concept is how
  # that comparison gets mis-transcribed. --time-feat is NOT shared, so
  # the old joint row had to split.
  #
  # --vocab is NOT extended to ae even though the P1a spec writes
  # `--vocab byte`. It already means an INTEGER >= 2 on the franken
  # lanes, so honouring the spec would make one flag mean an integer on
  # one recipe and a string on another — the hp-slot-5/6 dual-meaning
  # trap. The ae head is byte-wide by construction, so the knob would
  # have exactly one legal value anyway.
  [%w[mlp --text data/ae_names],           "--text/--noise-eval/--noise-seed/--val-frac-pct"],
  [%w[gtx --noise-eval 0,0.5],             "--text/--noise-eval/--noise-seed/--val-frac-pct"],
  [%w[diff --val-frac-pct 10],             "--text/--noise-eval/--noise-seed/--val-frac-pct"],
  [%w[lstm --noise-seed 99],               "--text/--noise-eval/--noise-seed/--val-frac-pct"],
  [%w[ae --time-feat 8],                   "--time-feat"],
  [%w[ae --vocab 256],                     "--vocab"],
  [%w[ae --policy dfa],                    "--policy"],
  [%w[ae --dfa-cut step],                  "--dfa-cut"],
  # toy#165 follow-up: matched-CE stopping. --target-ce/--eval-every are
  # the ae lane's alone — every other lane's cells are compared at matched
  # steps and nothing has measured what changing that does to them.
  [%w[mlp --target-ce 0.05],               "--target-ce"],
  [%w[diff --eval-every 10],               "--eval-every"],
  [%w[gtx --probe-batches 4],              "--probe-batches"],
  # toy#166 (difflm). The P1b arms are this lane's alone. --policy is
  # refused here by a LATER rule, not the matrix: P1b is all-BP by
  # design and DFA arrives in P1c on the denoiser, so "no --policy" is a
  # statement about the experiment rather than about flag ownership.
  [%w[ae --arm diff-selfcond],             "--arm"],
  [%w[diff --ae-steps 100],                "--ae-steps"],
  [%w[mlp --tsteps 100],                   "--tsteps"],
  [%w[gtx --gen-bytes 1024],               "--gen-bytes"],
  [%w[lstm --judge-steps 100],             "--judge-steps"],
  # toy#168: the stage-2 objective axis is difflm's alone.
  [%w[diff --loss-weight v-param],         "--loss-weight"],
  [%w[ae --minsnr-gamma 5.0],              "--minsnr-gamma"],
  # toy#162: --clip-grad is the lstm lane's FAIR BPTT control, not a
  # general tuning knob — its effect is unmeasured on every other lane,
  # and a knob whose effect nobody measured is worse offered than absent.
  [%w[ssm --clip-grad 1.0],                "--clip-grad"],
  [%w[gtx --clip-grad 1.0],                "--clip-grad"],
  [%w[franken --clip-grad 1.0],            "--clip-grad"],
]
REJECT.each do |args, label|
  out, st = probe(args)
  if st.exitstatus == 2 && out.include?("only valid with")
    next
  end
  failures << "matrix: #{args.join(' ')} — expected exit 2 + 'only valid with' (#{label}); got #{st.exitstatus}:\n#{out.lines.last(2).join}"
end
puts failures.length == n0 ? "  ok: #{REJECT.length} wrong-recipe probes all rejected by the matrix" : "  FAIL: rejection side"

# ---- acceptance side: right-recipe probes get PAST the matrix ----
n0 = failures.length
# franken + --vocab (matrix-valid) trips the LATER --vocab-needs---corpus rule
out, st = probe(%w[franken --vocab 5000])
failures << "matrix: franken --vocab hit the matrix (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("--vocab needs --corpus")
# franken-moe with its full flag set trips the LATER eval-corpus existence rule
out, st = probe(%w[franken-moe --lr 0.01 --warmup 2 --experts 4 --routing top1 --moe-policy bp-spine --eval-corpus /nonexistent.bin])
failures << "matrix: franken-moe full set hit the matrix (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("no such file")
# franken + --ckpt-every without a value trips the parse rule, not the matrix
out, st = probe(%w[franken --ckpt-every])
failures << "matrix: franken --ckpt-every parse error wrong (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("requires a value")
# toy#152: the mlp lane's own flags pass the matrix; --classes 1 trips
# the LATER value rule, and --batch 64 must NOT trip the transformer
# lane's "--batch > 1 needs --corpus" rule (the mlp batch is generated,
# not streamed).
out, st = probe(%w[mlp --batch 64 --classes 1])
failures << "matrix: mlp --batch/--classes hit the wrong rule (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("--classes must be an integer >= 2")
# toy#153: the gnn lane's own flags all pass the matrix; the run then
# stops at the LATER --task token rule, which is lane-local even though
# the flag name is shared with mlp.
out, st = probe(%w[gnn --nodes 512 --degree 8 --feedback-route structure --feedback-hops 2 --weight-decay 0.01 --task blobs])
failures << "matrix: gnn full flag set hit the matrix (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("--task blobs is only valid with recipe 'mlp'")
# toy#155: the ssm lane's own flags all pass the matrix; the run stops at
# the LATER --task token rule.
out, st = probe(%w[ssm --seq 32 --d-model 16 --d-inner 32 --conv-k 2 --selection lti --dfa-cut step --cue-span 4 --noise 0.5 --dt-init -4.0 --task community])
failures << "matrix: ssm full flag set hit the matrix (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("--task community is only valid with recipe 'gnn'")
# toy#156: the diff lane's own flags all pass the matrix; the run stops
# at the LATER --task token rule.
out, st = probe(%w[diff --latent 8 --time-feat 4 --modes 4 --spread 3.0 --mode-scale 0.4 --diff-steps 50 --beta-lo 0.002 --beta-hi 0.2 --eval-n 256 --task cue])
failures << "matrix: diff full flag set hit the matrix (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("--task cue is only valid with recipe 'ssm' or 'lstm'")
# toy#160: the gtx lane's own flags all pass the matrix; the run stops at
# the LATER --task token rule. `relational`/`local` are gtx-local, the
# way `cue`/`mean` belong to the two recurrent lanes.
out, st = probe(%w[gtx --heads 2 --d-ff 64 --types 4 --entities 32 --d-model 32 --features 16 --layers 2 --dfa-cut step --batch 64 --val-batches 2 --task-seed 3 --lr 0.001 --noise 0.2 --task community])
failures << "matrix: gtx full flag set hit the matrix (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("--task community is only valid with recipe 'gnn'")
# tao#24 — gtx's device rule is scoped to a TASK, not to the recipe, and
# this pair is what proves it is scoped rather than merely worded that
# way. Both probes must pass the FLAG MATRIX (which is what this gate is
# about) and then stop at a LATER rule; which later rule they hit is the
# whole point:
#   relational + cuda -> stops at the DEVICE rule (the twin is bytelm-only)
#   bytelm     + cuda -> gets PAST the device rule and stops at --text
# If the device check ever widened back to the recipe, the second probe
# would report the device message instead and fail here.
out, st = probe(%w[gtx --task relational --device cuda])
failures << "matrix: gtx --task relational was rejected (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("requires --task bytelm")
out, st = probe(%w[gtx --task bytelm --device cuda])
failures << "matrix: gtx --task bytelm --device cuda did not reach the --text rule, so the device check is still recipe-scoped rather than task-scoped (tao#24) (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("requires --text")
out, st = probe(%w[mlp --task relational])
failures << "matrix: mlp --task relational not rejected (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("only valid with recipe 'gtx'")
# toy#157: the lstm lane's own flags all pass the matrix, INCLUDING the
# sequence knobs it shares with ssm; the run stops at the LATER --task
# token rule. `cue`/`mean` are shared by the two recurrent lanes on
# purpose — the same generator would otherwise need two names.
out, st = probe(%w[lstm --seq 32 --classes 4 --hidden 32 --features 16 --layers 1 --dfa-cut step --cue-span 4 --noise 0.5 --batch 8 --val-batches 2 --task-seed 3 --lr 0.03 --task community])
failures << "matrix: lstm full flag set hit the matrix (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("--task community is only valid with recipe 'gnn'")
out, st = probe(%w[lstm --task cue --device cuda])
failures << "matrix: lstm --task cue was rejected — cue|mean is SHARED by ssm and lstm (#{out.lines.last(1).join.strip})" unless st.exitstatus == 2 && out.include?("not supported for recipe 'lstm'")
puts failures.length == n0 ? "  ok: right-recipe probes pass the matrix (their errors come from later rules)" : "  FAIL: acceptance side"

if failures.empty?
  puts "GATE PASS [train-cli-matrix]: #{REJECT.length} rejections + 12 acceptance probes (toy#132/#153/#155/#156/#157/#160)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [train-cli-matrix]: #{f}" }
  exit 1
end
