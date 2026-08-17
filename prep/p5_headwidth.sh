#!/bin/zsh
# prep/p5_headwidth.sh — toy#170 spec P5, the OTHER half: the NOMINAL
# head width.
#
# P4 and prep/p5_remap.sh both move how many symbols the corpus ACTUALLY
# uses while the head stays 256. But the output-dim law is a claim about
# the FEEDBACK MATRIX: B is [d_model, vocab] and its inv_sqrt_fan scale
# is 1/sqrt(vocab). Both were pinned at 256 on every byte-LM cell this
# lane has ever produced, P3 and P4 included. So those sweeps measured
# the EFFECTIVE RANK of the error; this one measures the WIDTH OF B.
#
# The two are the same number only when the head is sized to the corpus,
# which is precisely the confound the pair of sweeps exists to split:
#
#   p5_remap.sh    effective rank 27..192   B fixed at 256
#   p5_headwidth.sh   B 65..1024            effective rank fixed at 65
#
# THE CORPUS NEVER CHANGES HERE. One pack (ae_shak_a65 — shakespeare
# relabelled to dense ids, the same task as the P3/P4 anchor up to a
# permutation), so difficulty, redundancy, genre and language count are
# not merely controlled, they are IDENTICAL across every cell. The extra
# classes above 65 are dead: they never appear as a label. What moves is
# the width of B, the width of the head and the embedding, and B's scale.
#
# THE LIMIT, WRITTEN IN: dead classes are not live classes. The error
# vector p - y does carry real mass in those coordinates (the softmax
# puts it there), so B routes genuine signal through the full width — but
# this is a measurement of the DFA mechanism's output dimension, not a
# claim about what a real 1024-symbol corpus would do. The remap sweep is
# the one that moves real symbols; this one moves the matrix.
#
# It also reaches PAST 256 with no tokenizer: widths 512 and 1024 are one
# env var, where a BPE route to the same span would introduce a new
# tokenization and a new per-underlying-byte metric alongside the axis.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-/srv/data/scratch/p5hw}
mkdir -p "$OUT"
STEPS=${STEPS:-4000}
BLOCKS=4
SEEDS=${SEEDS:-5}
TEXT=${TEXT:-ae_shak_a65}
WIDTHS=${WIDTHS:-"65 128 256 512 1024"}
# 0.01 is in the grid for the same reason it is in p5_remap.sh: on this
# corpus bp's optimum sits at 0.003, which was the old top edge. An
# optimum on a boundary is not an optimum, and this is the arm that forms
# the denominator of every recovery number.
LR_BP=${LR_BP:-"0.0003 0.001 0.003 0.01"}
LR_DFA=${LR_DFA:-"0.00001 0.00003 0.0001 0.0003 0.001 0.003"}

cell () {          # cell <label> <policy> <cut> <lr> <seed> <width>
  local label=$1
  local policy=$2
  local cut=$3
  local lr=$4
  local seed=$5
  local w=$6
  # One `local` per line: zsh binds every RHS on a `local` line before
  # any assignment takes effect, which once turned a frozen control into
  # a trained one in silence.
  local f="$OUT/${label}_v${w}_lr${lr}_s${seed}.txt"
  [[ -s $f ]] && return 0
  env GTX_TASK=bytelm GTX_TEXT=data/$TEXT GTX_CONTEXT=128 \
      GTX_D_MODEL=128 GTX_HEADS=4 GTX_D_FF=256 GTX_VAL_BATCHES=8 \
      GTX_BLOCKS=$BLOCKS STEPS=$STEPS SEED=$seed GTX_POLICY=$policy \
      GTX_DFA_CUT=$cut GTX_LR=$lr GTX_VOCAB=$w ./libexec/toy-train-gtx \
      > "$f" 2>&1 || echo "  CELL FAILED: $f"
  # THE WIDTH IS ASSERTED ON BOTH NUMBERS. vocab and b_dim are printed
  # separately by the engine precisely so a head that narrows while B
  # stays 256 wide cannot pass as this experiment — that failure would
  # measure the opposite axis and still produce plausible bpb.
  grep -q "vocab=$w " "$f" || echo "  WIDTH MISMATCH (head) in $f"
  grep -q "b_dim=$w " "$f" || echo "  WIDTH MISMATCH (feedback matrix B) in $f"
  grep -q "pack=data/$TEXT " "$f" || echo "  CORPUS MISMATCH in $f"
  [[ $label == frozen-body ]] && { grep -q "frozen=$BLOCKS " "$f" || echo "  FROZEN ARM NOT FULLY FROZEN in $f" }
  [[ $label == dfa-* ]] && { grep -q "dfa_wired=$BLOCKS " "$f" || echo "  DFA ARM NOT FULLY WIRED in $f" }
  echo "  $label v=$w lr=$lr s=$seed -> $(grep -o 'bpb=[0-9.]*' "$f" | head -1)"
}

bpb_of () {
  [[ -s $1 ]] || return 0
  grep -o 'bpb=[0-9.]*' "$1" | head -1 | cut -d= -f2
}

best_lr () {       # best_lr <label> <width> <grid...>
  local label=$1
  local w=$2
  shift 2
  local grid=($@)
  local blr=""
  local bv=""
  local lr
  for lr in $grid; do
    local v=$(bpb_of "$OUT/${label}_v${w}_lr${lr}_s0.txt")
    [[ -z $v ]] && continue
    if [[ -z $bv ]] || (( $(echo "$v < $bv" | bc -l) )); then
      bv=$v; blr=$lr
    fi
  done
  if [[ -z $blr ]]; then
    echo "  NO PHASE-1 CELLS for $label v=$w" >&2
    return 1
  fi
  if [[ $blr == $grid[1] || $blr == $grid[-1] ]]; then
    echo "  EDGE OPTIMUM: $label v=$w best lr=$blr at grid boundary (bpb=$bv) — EXTEND, this cell is not measured" >&2
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

for w in ${=WIDTHS}; do
  echo "== head width $w: per-arm LR sweep (seed 0) =="
  for lr in ${=LR_BP}; do
    cell bp-body     $P_BP layer $lr 0 $w
    cell frozen-body $P_FZ layer $lr 0 $w
  done
  for lr in ${=LR_DFA}; do
    cell dfa-layer $P_DFA layer $lr 0 $w
    cell dfa-step  $P_DFA step  $lr 0 $w
  done
done

for w in ${=WIDTHS}; do
  echo "== head width $w: seeds at per-arm best LR =="
  B_BP=$(best_lr bp-body     $w ${=LR_BP})
  B_FZ=$(best_lr frozen-body $w ${=LR_BP})
  B_DL=$(best_lr dfa-layer   $w ${=LR_DFA})
  B_DS=$(best_lr dfa-step    $w ${=LR_DFA})
  echo "  best LR: bp=$B_BP frozen=$B_FZ dfa-layer=$B_DL dfa-step=$B_DS"
  s=1
  while [[ $s -lt $SEEDS ]]; do
    cell bp-body     $P_BP  layer $B_BP $s $w
    cell frozen-body $P_FZ  layer $B_FZ $s $w
    cell dfa-layer   $P_DFA layer $B_DL $s $w
    cell dfa-step    $P_DFA step  $B_DS $s $w
    s=$((s + 1))
  done
done
echo "== P5 head-width done: $OUT =="
