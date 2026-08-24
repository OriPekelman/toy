#!/bin/zsh
# prep/research/p4_alphabet.sh — toy#170 spec P4: the ALPHABET arm on the gtx
# byte-LM lane.
#
# The question: P3 measured DFA recovering ~25% of what BP buys at
# alphabet 65. The output-dim law predicts that fraction DECAYS as the
# active output rank grows. This varies only `--text` across the three
# P1a packs (27 / 65 / 201 distinct bytes) with the nominal head fixed
# at 256, so only the ACTIVE rank moves.
#
# THE FILENAME CARRIES THE CORPUS, for the same reason the depth sweep's
# carries the depth: `cell` skips when the output file exists, so a name
# that omits the corpus would make udhr silently inherit shakespeare's
# numbers and the alphabet curve would come back flat — which is the
# predicted result, and therefore the most dangerous possible bug here.
#
# Depth is FIXED at 4 (the P3 anchor) so the alphabet is the only axis.
set -e
# repo root is TWO levels up from prep/research/ (was one, from prep/).
cd "$(dirname "$0")/../.."
OUT=${OUT:-/srv/data/scratch/p4}
mkdir -p "$OUT"
STEPS=${STEPS:-4000}
BLOCKS=4

cell () {          # cell <label> <policy> <cut> <lr> <seed> <corpus>
  local label=$1 policy=$2 cut=$3 lr=$4 seed=$5 corpus=$6
  local f="$OUT/${label}_${corpus}_lr${lr}_s${seed}.txt"
  [[ -s $f ]] && return 0
  env GTX_TASK=bytelm GTX_TEXT=data/$corpus GTX_CONTEXT=128 \
      GTX_D_MODEL=128 GTX_HEADS=4 GTX_D_FF=256 GTX_VAL_BATCHES=8 \
      GTX_BLOCKS=$BLOCKS STEPS=$STEPS SEED=$seed GTX_POLICY=$policy \
      GTX_DFA_CUT=$cut GTX_LR=$lr ./libexec/toy-train-gtx > "$f" 2>&1 \
      || echo "  CELL FAILED: $f"
  # The cell must be the arm its label claims. A malformed policy string
  # degrades block 0 to `chain` SILENTLY on the frozen arm (the dfa arms
  # fail loud on it, frozen does not) — a control that trains one block
  # is not a control, and it cost a full depth-sweep re-run to catch.
  grep -q "pack=data/$corpus " "$f" || echo "  CORPUS MISMATCH in $f"
  [[ $label == frozen-body ]] && { grep -q "frozen=$BLOCKS " "$f" || echo "  FROZEN ARM NOT FULLY FROZEN in $f" }
  [[ $label == dfa-* ]] && { grep -q "dfa_wired=$BLOCKS " "$f" || echo "  DFA ARM NOT FULLY WIRED in $f" }
  echo "  $label $corpus lr=$lr s=$seed -> $(grep -o 'bpb=[0-9.]*' "$f" | head -1) $(grep -o 'alphabet=[0-9]*' "$f" | head -1)"
}

rep () {           # rep <token> <n>
  # Separate `local` lines: zsh evaluates every RHS on a `local` line
  # BEFORE any of its assignments bind (see prep/research/p3_depth.sh).
  local t=$1
  local n=$2
  local s=$t
  for _ in $(seq 2 $n); do s="$s,$t"; done
  echo $s
}
P_BP=$(rep chain $BLOCKS); P_DFA=$(rep dfa $BLOCKS); P_FZ=$(rep frozen $BLOCKS)

for c in ${=CORPORA:-ae_names ae_udhr}; do
  echo "== $c: per-arm LR sweep (seed 0) =="
  for lr in 0.0003 0.001 0.003; do
    cell bp-body     $P_BP layer $lr 0 $c
    cell frozen-body $P_FZ layer $lr 0 $c
  done
  # The DFA grid runs LOWER: a bigger alphabet may push the optimum
  # below 3e-4 via the toy#152 ceiling, and an optimum on the edge of
  # the grid is not measured — that trap moved P3's headline 32%->25%.
  for lr in 0.00001 0.00003 0.0001 0.0003 0.001; do
    cell dfa-layer $P_DFA layer $lr 0 $c
    cell dfa-step  $P_DFA step  $lr 0 $c
  done
done
echo "== P4 phase 1 done: $OUT =="
