#!/bin/zsh
# prep/p6_ladder.sh — toy#170 spec P6: ARITHMETIC RANK at a pinned head.
#
# WHAT THIS IS NOT. This is not the output-dim law's rematch. P5 settled
# that in the controlled 27-1024 range and the udhr evidence for it was a
# difficulty/language confound. Symbol inflation raises the ARITHMETIC
# rank — the number of active classes — while the LEARNABLE structure
# stays exactly 65-way, because the m codes for a symbol are drawn
# uniformly and there is nothing in them to learn. So what this measures
# is **B's conditioning under a wide, mostly-noise error vector**, and a
# result here can never be read as "and 50k too": no controlled
# experiment reaches the 50k LEARNABLE regime, since learnable rank
# cannot rise without difficulty rising.
#
# THE HEAD IS PINNED (default 4096) ACROSS EVERY RUNG. That is forced,
# not stylistic: rank <= head always, so a ladder that let the head track
# the rank would move B's width and the rank together and measure
# neither. Pinning at the top rung's requirement makes every lower rung a
# padded head, which is the only way the axis is one axis.
#
# The rungs are what prep/remap_alphabet.rb actually produced, NOT 65*m —
# a symbol occurring fewer than m times cannot use all its codes, so the
# realised rank is read from each run's own provenance.
#
#   ae_shak_a65    rank   65   b = 0.000 bits added
#   ae_shak_a192   rank  192   b = 1.585
#   ae_shak_a380   rank  380   b = 2.585   <- the noise-linearity pre-check
#   ae_shak_a508   rank  508   b = 3.000
#   ae_shak_a1008  rank 1008   b = 4.000
#   ae_shak_a2504  rank 2504   b = 5.320
#
# SEEDS ARE BUDGETED FOR THE DECAY QUESTION, NOT THE SIGN. Measured sd of
# absolute bits recovered grows about 0.032 per bit of added noise
# against a signal that theory holds at ~0.2, so the sign stays cheap
# while a 0.1-bit decay needs n ~ 10 at b=3 and ~ 19 at b=5. The sign is
# nearly free and boring; a decay is the only interesting outcome.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-/srv/data/scratch/p6}
mkdir -p "$OUT"
STEPS=${STEPS:-4000}
BLOCKS=4
SEEDS=${SEEDS:-8}
HEAD=${HEAD:-4096}
RUNGS=${RUNGS:-"ae_shak_a65 ae_shak_a192 ae_shak_a380"}
LR_BP=${LR_BP:-"0.0003 0.001 0.003 0.01"}
# Re-bracketed PER RUNG, never reused: the DFA optimum slid UP between
# rank 65 and 129 in P5, the opposite of what both of us expected, so a
# reused LR would be measuring the wrong cell at every rung above the
# first.
LR_DFA=${LR_DFA:-"0.00001 0.00003 0.0001 0.0003 0.001 0.003"}

cell () {          # cell <label> <policy> <cut> <lr> <seed> <rung>
  local label=$1
  local policy=$2
  local cut=$3
  local lr=$4
  local seed=$5
  local rung=$6
  # One `local` per line — zsh binds every RHS on a `local` line before
  # any assignment takes effect, which once turned a frozen control into
  # a trained one in silence.
  local f="$OUT/${label}_${rung}_v${HEAD}_lr${lr}_s${seed}.txt"
  [[ -s $f ]] && return 0
  env GTX_TASK=bytelm GTX_TEXT=data/$rung GTX_CONTEXT=128 \
      GTX_D_MODEL=128 GTX_HEADS=4 GTX_D_FF=256 GTX_VAL_BATCHES=8 \
      GTX_BLOCKS=$BLOCKS STEPS=$STEPS SEED=$seed GTX_POLICY=$policy \
      GTX_DFA_CUT=$cut GTX_LR=$lr GTX_VOCAB=$HEAD ./libexec/toy-train-gtx \
      > "$f" 2>&1 || echo "  CELL FAILED: $f"
  grep -q "pack=data/$rung " "$f" || echo "  RUNG MISMATCH in $f"
  # Head AND feedback-matrix width, asserted separately: a head that
  # narrows while B stays wide measures the opposite axis and still emits
  # plausible bpb.
  grep -q "vocab=$HEAD " "$f" || echo "  HEAD MISMATCH in $f"
  grep -q "b_dim=$HEAD " "$f" || echo "  B-WIDTH MISMATCH in $f"
  [[ $label == frozen-body ]] && { grep -q "frozen=$BLOCKS " "$f" || echo "  FROZEN ARM NOT FULLY FROZEN in $f" }
  [[ $label == dfa-* ]] && { grep -q "dfa_wired=$BLOCKS " "$f" || echo "  DFA ARM NOT FULLY WIRED in $f" }
  echo "  $label $rung lr=$lr s=$seed -> $(grep -o 'bpb=[0-9.]*' "$f" | head -1) $(grep -o 'alphabet=[0-9]*' "$f" | head -1)"
}

bpb_of () {
  [[ -s $1 ]] || return 0
  grep -o 'bpb=[0-9.]*' "$1" | head -1 | cut -d= -f2
}

best_lr () {       # best_lr <label> <rung> <grid...>
  local label=$1
  local rung=$2
  shift 2
  local grid=($@)
  local blr=""
  local bv=""
  local lr
  for lr in $grid; do
    local v=$(bpb_of "$OUT/${label}_${rung}_v${HEAD}_lr${lr}_s0.txt")
    [[ -z $v ]] && continue
    if [[ -z $bv ]] || (( $(echo "$v < $bv" | bc -l) )); then
      bv=$v; blr=$lr
    fi
  done
  if [[ -z $blr ]]; then
    echo "  NO PHASE-1 CELLS for $label $rung" >&2
    return 1
  fi
  if [[ $blr == $grid[1] || $blr == $grid[-1] ]]; then
    echo "  EDGE OPTIMUM: $label $rung best lr=$blr at grid boundary (bpb=$bv) — EXTEND, this rung is not measured" >&2
  fi
  echo $blr
}

rep () {
  local t=$1
  local n=$2
  local s=$t
  for _ in $(seq 2 $n); do s="$s,$t"; done
  echo $s
}
P_BP=$(rep chain $BLOCKS)
P_DFA=$(rep dfa $BLOCKS)
P_FZ=$(rep frozen $BLOCKS)

for r in ${=RUNGS}; do
  echo "== $r (head $HEAD): per-arm LR sweep (seed 0) =="
  for lr in ${=LR_BP}; do
    cell bp-body     $P_BP layer $lr 0 $r
    cell frozen-body $P_FZ layer $lr 0 $r
  done
  for lr in ${=LR_DFA}; do
    cell dfa-layer $P_DFA layer $lr 0 $r
    cell dfa-step  $P_DFA step  $lr 0 $r
  done
done

for r in ${=RUNGS}; do
  echo "== $r (head $HEAD): seeds at per-arm best LR =="
  B_BP=$(best_lr bp-body     $r ${=LR_BP})
  B_FZ=$(best_lr frozen-body $r ${=LR_BP})
  B_DL=$(best_lr dfa-layer   $r ${=LR_DFA})
  B_DS=$(best_lr dfa-step    $r ${=LR_DFA})
  echo "  best LR: bp=$B_BP frozen=$B_FZ dfa-layer=$B_DL dfa-step=$B_DS"
  s=1
  while [[ $s -lt $SEEDS ]]; do
    cell bp-body     $P_BP  layer $B_BP $s $r
    cell frozen-body $P_FZ  layer $B_FZ $s $r
    cell dfa-layer   $P_DFA layer $B_DL $s $r
    cell dfa-step    $P_DFA step  $B_DS $s $r
    s=$((s + 1))
  done
done
echo "== P6 ladder done: $OUT =="
