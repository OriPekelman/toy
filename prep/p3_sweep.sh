#!/bin/zsh
# prep/p3_sweep.sh — toy#170 (capstone P3) arm sweep on the gtx byte-LM lane.
#
# PER-ARM LR (F22 / toy#160's carried lesson): every arm is measured at
# ITS OWN best LR. An arm scored at another arm's cell is not a negative,
# and this program nearly published "attention is DFA-hostile" from BP's
# LR. Phase 1 sweeps 3 LRs per arm at one seed; phase 2 re-runs each
# arm's winner at 3 seeds.
#
# Args are passed explicitly to the cell function — zsh does NOT
# word-split an unquoted parameter, and the two arms then silently run
# the default and look bit-identical (the toy#141 landmine).
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-/srv/data/scratch/p3}
mkdir -p "$OUT"

STEPS=${STEPS:-4000}
COMMON=(GTX_TASK=bytelm GTX_TEXT=data/ae_shakespeare GTX_CONTEXT=128
        GTX_D_MODEL=128 GTX_HEADS=4 GTX_D_FF=256 GTX_BLOCKS=4
        GTX_VAL_BATCHES=8)

cell () {          # cell <label> <policy> <cut> <lr> <seed>
  local label=$1 policy=$2 cut=$3 lr=$4 seed=$5
  local f="$OUT/${label}_lr${lr}_s${seed}.txt"
  [[ -s $f ]] && { echo "  skip $f"; return 0 }
  env $COMMON STEPS=$STEPS SEED=$seed GTX_POLICY=$policy GTX_DFA_CUT=$cut \
      GTX_LR=$lr ./libexec/toy-train-gtx > "$f" 2>&1 || echo "  CELL FAILED: $f"
  echo "  $label lr=$lr s=$seed -> $(grep -o 'bpb=[0-9.]*' "$f" | head -1)"
}

# 4 blocks, so the policy string has 4 entries.
P_BP=chain,chain,chain,chain
P_DFA=dfa,dfa,dfa,dfa
P_FZ=frozen,frozen,frozen,frozen

echo "== phase 1: per-arm LR sweep (seed 0) =="
for lr in 0.0003 0.001 0.003; do
  cell bp-body    $P_BP  layer $lr 0
  cell dfa-layer  $P_DFA layer $lr 0
  cell dfa-step   $P_DFA step  $lr 0
  cell frozen-body $P_FZ layer $lr 0
done
echo "== phase 1 done: $OUT =="

# Phase 1b: the DFA arms both peaked at the LOW EDGE of the phase-1 grid,
# so their optima were on the boundary. An arm measured at the edge of
# its swept range is not measured (F22 / toy#160), so extend downward
# until the optimum is INTERIOR.
if [[ ${PHASE1B:-0} == 1 ]]; then
  echo "== phase 1b: extend the DFA arms downward =="
  for lr in 0.0001 0.00003; do
    cell dfa-layer $P_DFA layer $lr 0
    cell dfa-step  $P_DFA step  $lr 0
  done
fi

# Phase 2: each arm at ITS OWN best LR, seeds 1 and 2 (seed 0 is already
# on disk from phase 1 and is reused, not re-run).
if [[ ${PHASE2:-0} == 1 ]]; then
  echo "== phase 2: best-LR cells at seeds 1,2 =="
  for s in 1 2; do
    cell bp-body     $P_BP  layer 0.001  $s
    cell dfa-layer   $P_DFA layer 0.0003 $s
    cell dfa-step    $P_DFA step  0.0003 $s
    cell frozen-body $P_FZ  layer 0.001  $s
  done
fi
