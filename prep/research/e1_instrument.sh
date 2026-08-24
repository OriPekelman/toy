#!/bin/zsh
# prep/research/e1_instrument.sh — toy#172/E1 Phase 1.1: read the B-conditioning
# instrument across the P6 rank ladder.
#
# NOT a re-read: the instrument needs the ERROR VECTORS, and the stored
# P6 cells contain only the bpb summary. So these are new cells at the
# same configuration.
#
# MATCHED n ACROSS RUNGS, deliberately. rank(C_E) <= n, so if n moved
# with V the measured rank would fall as width grows for reasons that
# have nothing to do with conditioning — which is exactly E1's `rank`
# verdict manufactured from a sampling artifact. Holding n fixed keeps
# the ceiling constant; the run emits n, v and a `capped` verdict so any
# rung sitting at its ceiling is visible rather than reported.
#
# The DFA arm is the one the B-conditioning question is about, so it runs
# on every rung. bp and frozen run too — they are what says whether a
# conditioning change is specific to the credit rule or a property of the
# task at that width.
set -e
# repo root is TWO levels up from prep/research/ (was one, from prep/).
cd "$(dirname "$0")/../.."
OUT=${OUT:-/srv/data/scratch/e1}
mkdir -p "$OUT"
STEPS=${STEPS:-4000}
BLOCKS=4
HEAD=${HEAD:-4096}
SEEDS=${SEEDS:-3}
IN=${IN:-1024}
RUNGS=${RUNGS:-"ae_shak_a65 ae_shak_a192 ae_shak_a380 ae_shak_a508 ae_shak_a1008 ae_shak_a2504"}

cell () {          # cell <label> <policy> <cut> <lr> <seed> <rung>
  local label=$1
  local policy=$2
  local cut=$3
  local lr=$4
  local seed=$5
  local rung=$6
  local f="$OUT/${label}_${rung}_v${HEAD}_n${IN}_s${seed}.txt"
  [[ -s $f ]] && return 0
  env GTX_TASK=bytelm GTX_TEXT=data/$rung GTX_CONTEXT=128 \
      GTX_D_MODEL=128 GTX_HEADS=4 GTX_D_FF=256 GTX_VAL_BATCHES=8 \
      GTX_BLOCKS=$BLOCKS STEPS=$STEPS SEED=$seed GTX_POLICY=$policy \
      GTX_DFA_CUT=$cut GTX_LR=$lr GTX_VOCAB=$HEAD \
      GTX_INSTRUMENT=1 GTX_INSTRUMENT_N=$IN ./libexec/toy-train-gtx \
      > "$f" 2>&1 || echo "  CELL FAILED: $f"
  grep -q "pack=data/$rung " "$f" || echo "  RUNG MISMATCH in $f"
  grep -q "b_dim=$HEAD " "$f" || echo "  B-WIDTH MISMATCH in $f"
  grep -q "^bcond: " "$f" || echo "  NO bcond LINE in $f"
  echo "  $label $rung s=$seed -> $(grep -o 'stable_rank=[0-9.]*' "$f" | head -1) $(grep -o 'participation_ratio=[0-9.]*' "$f" | head -1) $(grep -o 'capped=[a-zA-Z-]*' "$f" | head -1)"
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

# Per-rung best LRs, carried from the P6 sweep — the DFA optimum slid up
# 3e-4 -> 1e-3 and then CRASHED to 3e-5 at the top rung, so these are not
# interchangeable and must not be tidied into one value.
lr_for () {        # lr_for <arm> <rung>
  local arm=$1
  local rung=$2
  case "$arm:$rung" in
    dfa:ae_shak_a65)   echo 0.0003 ;;
    dfa:ae_shak_a2504) echo 0.00003 ;;
    dfa:*)             echo 0.001 ;;
    frozen:ae_shak_a2504) echo 0.0003 ;;
    *)                 echo 0.001 ;;
  esac
}

for r in ${=RUNGS}; do
  echo "== $r (head $HEAD, n=$IN) =="
  s=0
  while [[ $s -lt $SEEDS ]]; do
    cell dfa-layer   $P_DFA layer $(lr_for dfa $r)    $s $r
    cell bp-body     $P_BP  layer $(lr_for bp $r)     $s $r
    cell frozen-body $P_FZ  layer $(lr_for frozen $r) $s $r
    s=$((s + 1))
  done
done
echo "== E1 instrument sweep done: $OUT =="
