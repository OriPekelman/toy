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
  [%w[franken --dfa-granularity block],   "--dfa-granularity"],
  [%w[franken-moe --mla-layers 1],        "--mla-layers/--mla-rank"],
  [%w[franken --expert-act situ-glu],     "--expert-act"],
  [%w[franken --lr-schedule ramp-up],     "--lr-schedule/--lr-lo/--lr-hi"],
  [%w[from-scratch --ckpt-every 2],       "--ckpt-every"],
  [%w[franken --load-ckpt x.gguf],        "--load-ckpt"],
  [%w[franken --eval-corpus x.bin],       "--eval-corpus"],
  [%w[from-scratch --shape wide],         "--shape"],
  [%w[franken --routing top1],            "--routing"],
  [%w[franken --experts 4],               "--experts"],
]
REJECT.each do |args, label|
  out, st = probe(args)
  if st.exitstatus == 2 && out.include?("only valid with")
    next
  end
  failures << "matrix: #{args.join(' ')} — expected exit 2 + 'only valid with' (#{label}); got #{st.exitstatus}:\n#{out.lines.last(2).join}"
end
puts failures.empty? ? "  ok: #{REJECT.length} wrong-recipe probes all rejected by the matrix" : "  FAIL: rejection side"

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
puts failures.length == n0 ? "  ok: right-recipe probes pass the matrix (their errors come from later rules)" : "  FAIL: acceptance side"

if failures.empty?
  puts "GATE PASS [train-cli-matrix]: #{REJECT.length} rejections + 3 acceptance probes (toy#132)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [train-cli-matrix]: #{f}" }
  exit 1
end
