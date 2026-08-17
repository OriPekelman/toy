#!/bin/zsh
# prep/p5_remap.sh — toy#170 spec P5: the CONTROLLED alphabet axis.
#
# P4 moved the alphabet by moving the CORPUS, so alphabet travelled with
# difficulty, redundancy, genre and language count. Its one separated
# point (udhr, 201) is also the one carrying every confound. This holds
# ONE corpus fixed — shakespeare — and moves only the symbol set, via
# prep/remap_alphabet.rb:
#
#   ae_shak_a27    deflate: top 26 symbols + one OTHER class
#   ae_shakespeare 65, the P3/P4 anchor      <- REUSED from /srv/data/scratch/p4
#   ae_shak_a129   inflate x2  (entropy-preserving: +log2(2) bits, every arm)
#   ae_shak_a192   inflate x3  (entropy-preserving: +log2(3) bits, every arm)
#
# THE INFLATION ARMS ARE THE CONTROL. Each symbol gets m private codes
# chosen uniformly per occurrence, so the sequence structure is IDENTICAL
# and exactly log2(m) bits of irreducible noise are added to EVERY ARM
# EQUALLY. An ideal predictor's loss rises by log2(m) on bp, dfa and
# frozen alike, so `frozen - dfa` (the absolute bits recovered, which is
# now the primary statistic) is unchanged IN THEORY. If it moves, the
# output rank moved it, and no difficulty story is available.
#
# a27 is NOT entropy-preserving — merging destroys information — so it is
# read against its OWN bp/frozen anchors and never differenced against
# the inflation arms.
#
# Every pack is one token per original byte (n_tokens identical across
# all four), so bpb is per-underlying-byte throughout and the inflated
# arms differ from the anchor by a known additive log2(m).
#
# THE FILENAME CARRIES THE CORPUS for the reason p4_alphabet.sh says:
# `cell` skips when the file exists, so a name omitting the corpus makes
# every later pack inherit the first one's numbers — a flat curve, which
# is a predicted outcome and therefore the most dangerous possible bug.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-/srv/data/scratch/p5}
mkdir -p "$OUT"
STEPS=${STEPS:-4000}
BLOCKS=4
SEEDS=${SEEDS:-5}                      # n>=5: a 3-seed median is not reportable
# ae_shak_a65 is the PIPELINE'S OWN CONTROL: a pure relabeling of
# shakespeare into dense ids, so it is the same task as the P3/P4 anchor
# up to a permutation of the embedding and head rows. Measured through
# this exact harness, it answers the question the other three points
# cannot — whether the repacking itself moves the result.
CORPORA=${CORPORA:-"ae_shak_a27 ae_shak_a65 ae_shak_a129 ae_shak_a192"}
# 0.01 is in the grid because ae_shak_a65's bp optimum landed ON 0.003,
# the old top edge, while raw shakespeare's — the SAME TASK — sat
# interior at 0.001. An edge optimum is not an optimum, and this one is
# on the arm that forms the denominator of every recovery number.
LR_BP=${LR_BP:-"0.0003 0.001 0.003 0.01"}
# The DFA grid runs LOWER: toy#152's ceiling pushes the optimum down as
# the rank grows, and an optimum on the edge of a grid is not measured —
# that trap moved P3's headline from 32% to 25%.
# 0.003 is in the grid because a129's dfa-step optimum landed ON 0.001,
# the old top edge — the opposite direction from the prediction that a
# bigger rank pushes the optimum DOWN. An edge optimum is not an optimum,
# it is the smallest value the grid was allowed to look at.
LR_DFA=${LR_DFA:-"0.00001 0.00003 0.0001 0.0003 0.001 0.003"}

cell () {          # cell <label> <policy> <cut> <lr> <seed> <corpus>
  local label=$1
  local policy=$2
  local cut=$3
  local lr=$4
  local seed=$5
  local corpus=$6
  # Separate `local` lines throughout: zsh evaluates every RHS on a
  # `local` line BEFORE any assignment binds, so `local a=$1 b=$a` gives
  # b the OLD a. That emitted ",frozen,frozen,frozen", an empty leading
  # field parses as `chain`, and the frozen control silently trained
  # block 0 — a tidy false finding that cost a full depth-sweep re-run.
  local f="$OUT/${label}_${corpus}_lr${lr}_s${seed}.txt"
  [[ -s $f ]] && return 0
  env GTX_TASK=bytelm GTX_TEXT=data/$corpus GTX_CONTEXT=128 \
      GTX_D_MODEL=128 GTX_HEADS=4 GTX_D_FF=256 GTX_VAL_BATCHES=8 \
      GTX_BLOCKS=$BLOCKS STEPS=$STEPS SEED=$seed GTX_POLICY=$policy \
      GTX_DFA_CUT=$cut GTX_LR=$lr ./libexec/toy-train-gtx > "$f" 2>&1 \
      || echo "  CELL FAILED: $f"
  # The cell must be the arm AND the corpus its filename claims.
  grep -q "pack=data/$corpus " "$f" || echo "  CORPUS MISMATCH in $f"
  [[ $label == frozen-body ]] && { grep -q "frozen=$BLOCKS " "$f" || echo "  FROZEN ARM NOT FULLY FROZEN in $f" }
  [[ $label == dfa-* ]] && { grep -q "dfa_wired=$BLOCKS " "$f" || echo "  DFA ARM NOT FULLY WIRED in $f" }
  echo "  $label $corpus lr=$lr s=$seed -> $(grep -o 'bpb=[0-9.]*' "$f" | head -1) $(grep -o 'alphabet=[0-9]*' "$f" | head -1)"
}

bpb_of () {        # bpb_of <file> -> bpb, or nothing
  [[ -s $1 ]] || return 0
  grep -o 'bpb=[0-9.]*' "$1" | head -1 | cut -d= -f2
}

# Pick the LR with the lowest seed-0 bpb, and REFUSE a grid edge. An
# optimum on the boundary is not bracketed, so it is not an optimum —
# it is the smallest value the grid was allowed to look at.
best_lr () {       # best_lr <label> <corpus> <grid...> -> lr on stdout
  local label=$1
  local corpus=$2
  shift 2
  local grid=($@)
  local blr=""
  local bv=""
  local lr
  for lr in $grid; do
    local v=$(bpb_of "$OUT/${label}_${corpus}_lr${lr}_s0.txt")
    [[ -z $v ]] && continue
    if [[ -z $bv ]] || (( $(echo "$v < $bv" | bc -l) )); then
      bv=$v; blr=$lr
    fi
  done
  if [[ -z $blr ]]; then
    echo "  NO PHASE-1 CELLS for $label $corpus" >&2
    return 1
  fi
  if [[ $blr == $grid[1] || $blr == $grid[-1] ]]; then
    echo "  EDGE OPTIMUM: $label $corpus best lr=$blr at grid boundary (bpb=$bv) — EXTEND THE GRID, this cell is not measured" >&2
  fi
  echo $blr
}

rep () {           # rep <token> <n>
  local t=$1
  local n=$2
  local s=$t
  for _ in $(seq 2 $n); do s="$s,$t"; done
  echo $s
}
P_BP=$(rep chain $BLOCKS)
P_DFA=$(rep dfa $BLOCKS)
P_FZ=$(rep frozen $BLOCKS)

# ---- phase 1: per-arm LR, seed 0 ----
for c in ${=CORPORA}; do
  echo "== $c: per-arm LR sweep (seed 0) =="
  for lr in ${=LR_BP}; do
    cell bp-body     $P_BP layer $lr 0 $c
    cell frozen-body $P_FZ layer $lr 0 $c
  done
  for lr in ${=LR_DFA}; do
    cell dfa-layer $P_DFA layer $lr 0 $c
    cell dfa-step  $P_DFA step  $lr 0 $c
  done
done

# ---- phase 2: seeds 1..SEEDS-1 at each arm's own best LR ----
for c in ${=CORPORA}; do
  echo "== $c: seeds at per-arm best LR =="
  B_BP=$(best_lr bp-body     $c ${=LR_BP})
  B_FZ=$(best_lr frozen-body $c ${=LR_BP})
  B_DL=$(best_lr dfa-layer   $c ${=LR_DFA})
  B_DS=$(best_lr dfa-step    $c ${=LR_DFA})
  echo "  best LR: bp=$B_BP frozen=$B_FZ dfa-layer=$B_DL dfa-step=$B_DS"
  s=1
  while [[ $s -lt $SEEDS ]]; do
    cell bp-body     $P_BP  layer $B_BP $s $c
    cell frozen-body $P_FZ  layer $B_FZ $s $c
    cell dfa-layer   $P_DFA layer $B_DL $s $c
    cell dfa-step    $P_DFA step  $B_DS $s $c
    s=$((s + 1))
  done
done
echo "== P5 done: $OUT =="
