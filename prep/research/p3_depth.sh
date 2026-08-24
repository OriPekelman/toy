#!/bin/zsh
# prep/research/p3_depth.sh — toy#170 (capstone P3) DEPTH sweep on the gtx byte-LM.
#
# The question: P3 measured DFA recovering ~25% of what BP buys on a
# 4-block transformer. Does that fraction HOLD as depth grows? If it
# decays with depth, "DFA on attention" is really a depth story and the
# 25% is an artifact of one scale.
#
# THE FILENAME CARRIES THE DEPTH. `cell` skips a cell whose output file
# already exists, so a name that omits `b${blocks}` would make a depth-2
# run silently inherit depth-4 numbers — the whole sweep would come back
# flat and look like a finding. Encoded, not remembered.
#
# PER-ARM, PER-DEPTH LR: an arm at another arm's cell is not measured
# (F22), and that applies per depth too — the optimum moves with depth.
# The DFA arms get a lower grid because they have an LR ceiling that bp
# and frozen do not (toy#152, confirmed on this lane: both DFA arms
# diverge at 3e-3 while bp/frozen are flat across the same 10x span).
#
# Args are passed explicitly — zsh does NOT word-split an unquoted
# parameter, and the arms then silently run the default and look
# bit-identical (the toy#141 landmine).
set -e
# repo root is TWO levels up from prep/research/ (was one, from prep/).
cd "$(dirname "$0")/../.."
OUT=${OUT:-/srv/data/scratch/p3depth}
mkdir -p "$OUT"
STEPS=${STEPS:-4000}

cell () {          # cell <label> <policy> <cut> <lr> <seed> <blocks>
  local label=$1 policy=$2 cut=$3 lr=$4 seed=$5 blocks=$6
  local f="$OUT/${label}_b${blocks}_lr${lr}_s${seed}.txt"
  [[ -s $f ]] && return 0
  env GTX_TASK=bytelm GTX_TEXT=data/ae_shakespeare GTX_CONTEXT=128 \
      GTX_D_MODEL=128 GTX_HEADS=4 GTX_D_FF=256 GTX_VAL_BATCHES=8 \
      GTX_BLOCKS=$blocks STEPS=$STEPS SEED=$seed GTX_POLICY=$policy \
      GTX_DFA_CUT=$cut GTX_LR=$lr ./libexec/toy-train-gtx > "$f" 2>&1 \
      || echo "  CELL FAILED: $f"
  # Assert the run actually used the depth the filename claims. A policy
  # string of the wrong length would otherwise be absorbed silently.
  grep -q "blocks=$blocks " "$f" || echo "  DEPTH MISMATCH in $f"
  # The frozen arm must report ALL its blocks frozen. A malformed policy
  # string degrades block 0 to `chain` silently, and a control with one
  # trained block is not a control.
  if [[ $label == frozen-body ]]; then
    grep -q "frozen=$blocks " "$f" || echo "  FROZEN ARM NOT FULLY FROZEN in $f"
  fi
  if [[ $label == dfa-* ]]; then
    grep -q "dfa_wired=$blocks " "$f" || echo "  DFA ARM NOT FULLY WIRED in $f"
  fi
  echo "  $label b=$blocks lr=$lr s=$seed -> $(grep -o 'bpb=[0-9.]*' "$f" | head -1)"
}

rep () {           # rep <token> <n>  — "chain" 4 -> chain,chain,chain,chain
  # NOT `local t=$1 n=$2 s=$t`: zsh evaluates every RHS on a `local`
  # line BEFORE any of its assignments bind, so `s` would get the OLD
  # (empty) `t` and every policy string would come out ",chain,chain".
  # That is not cosmetic — an empty leading field parses as `chain`, so
  # the FROZEN arm silently runs with block 0 trainable and the control
  # stops being a control. The dfa arms fail loud on it; frozen does not.
  local t=$1
  local n=$2
  local s=$t
  for _ in $(seq 2 $n); do s="$s,$t"; done
  echo $s
}

for b in ${=DEPTHS:-2 4 8}; do
  P_BP=$(rep chain $b); P_DFA=$(rep dfa $b); P_FZ=$(rep frozen $b)
  echo "== depth $b: per-arm LR sweep (seed 0) =="
  for lr in 0.0003 0.001 0.003; do
    cell bp-body     $P_BP layer $lr 0 $b
    cell frozen-body $P_FZ layer $lr 0 $b
  done
  for lr in 0.00003 0.0001 0.0003 0.001; do
    cell dfa-layer $P_DFA layer $lr 0 $b
    cell dfa-step  $P_DFA step  $lr 0 $b
  done
done
echo "== depth sweep phase 1 done: $OUT =="
