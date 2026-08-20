#!/bin/zsh
# prep/g1_ast.sh — toy#173 (G1): masked AST-node-type prediction on the
# GNN lane.
#
# NO ENGINE CHANGE. train_gnn takes GNN_GRAPH as a bundle prefix and
# reads n_classes from the pack's own meta, so this rides F17's engine
# byte-for-byte. The only thing that differs from the Cora cells is the
# pack — which is what makes the verdict clean: a `no-go-dfa` cannot be
# blamed on a new code path, because there is not one.
#
# THE CONTROL MUST BE ABLE TO LOSE, and on a tree that is not free:
# structure is highly informative and a frozen random GNN is already a
# neighbourhood aggregator. Measured before any arm is quoted — bp 0.114
# vs frozen 0.037 at chance 0.013, so the cell discriminates. The gate on
# every rung below is `bp > frozen`, per seed.
#
# LAYERS=1 HIDDEN=64 IS F17'S CORA CELL, NOT A CHOICE. I first ran
# LAYERS=2 HIDDEN=32 and the precondition FAILED (frozen .400 vs bp
# .069): an AST is a TREE, and two rounds of neighbourhood averaging
# collapse node representations, which a trained body optimises into
# and a random one does not. At 1 layer bp .483 > frozen .453. The
# transfer question is meaningless if the architecture moves, so this
# matches F17's cell exactly.
#
# SIZED AGAINST A MEASUREMENT, not a guess: this pack is 17.6k nodes
# (6.5x Cora), and at hidden=64 a step costs ~2.1s — a 3000-step sweep
# would have run 57 HOURS. At hidden=32 it is 0.92s/step, and 600 steps
# already separates the arms (bp .114 vs frozen .037 at 400).
#
# F17's finding is what transfers or does not: on Cora, dfa-structure
# BEAT bp (.736 vs .672) as a regulariser. Cora is homophilous; an AST is
# a heterophilous tree. That is the open question G1 exists to answer.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-/srv/data/scratch/g1}
mkdir -p "$OUT"
GRAPH=${GRAPH:-data/ast_code}
STEPS=${STEPS:-600}
SEEDS=${SEEDS:-5}
HIDDEN=${HIDDEN:-64}
LAYERS=${LAYERS:-1}
# Per-arm grids. DFA has tolerated LESS learning rate than BP on every
# lane in this programme (toy#152's ceiling, and the gtx optimum that
# crashed 33x at the top rung), so its grid runs lower and an optimum on
# a boundary is not an optimum.
# 0.3/1.0 are here because FROZEN's optimum landed ON 0.1, the old
# top edge. That matters more than a normal edge optimum: frozen is
# the CONTROL every margin is stated against, so an unbracketed
# frozen makes every number in the sweep provisional. Bracketed it
# reads .455 / .471 / .460 / .452 — a real interior peak at 0.1.
LR_BP=${LR_BP:-"0.003 0.01 0.03 0.1 0.3 1.0"}
LR_DFA=${LR_DFA:-"0.001 0.003 0.01 0.03"}

cell () {          # cell <label> <policy> <route> <lr> <seed>
  local label=$1
  local policy=$2
  local route=$3
  local lr=$4
  local seed=$5
  # One `local` per line: zsh binds every RHS on a `local` line before
  # any assignment takes effect, which once turned a frozen control into
  # a trained one in silence.
  local f="$OUT/${label}_lr${lr}_s${seed}.txt"
  [[ -s $f ]] && return 0
  env GNN_GRAPH=$GRAPH GNN_POLICY=$policy GNN_FEEDBACK_ROUTE=$route \
      GNN_HIDDEN=$HIDDEN GNN_LAYERS=$LAYERS STEPS=$STEPS SEED=$seed \
      GNN_LR=$lr ./libexec/toy-train-gnn > "$f" 2>&1 || echo "  CELL FAILED: $f"
  grep -q "graph: " "$f" || echo "  NO GRAPH LINE in $f"
  echo "  $label lr=$lr s=$seed -> $(grep -oE 'acc=[0-9.]+' "$f" | tail -1)"
}

acc_of () {
  [[ -s $1 ]] || return 0
  grep -oE 'acc=[0-9.]+' "$1" | tail -1 | cut -d= -f2
}

best_lr () {       # best_lr <label> <grid...>
  local label=$1
  shift
  local grid=($@)
  local blr=""
  local bv=""
  local lr
  for lr in $grid; do
    local v=$(acc_of "$OUT/${label}_lr${lr}_s0.txt")
    [[ -z $v ]] && continue
    if [[ -z $bv ]] || (( $(echo "$v > $bv" | bc -l) )); then
      bv=$v; blr=$lr
    fi
  done
  if [[ -z $blr ]]; then echo "  NO PHASE-1 CELLS for $label" >&2; return 1; fi
  if [[ $blr == $grid[1] || $blr == $grid[-1] ]]; then
    echo "  EDGE OPTIMUM: $label best lr=$blr at grid boundary (acc=$bv) — EXTEND, this arm is not measured" >&2
  fi
  echo $blr
}

P_BP=$(printf 'chain,%.0s' {1..$LAYERS}); P_BP=${P_BP%,}
P_DFA=$(printf 'dfa,%.0s' {1..$LAYERS}); P_DFA=${P_DFA%,}
P_FZ=$(printf 'frozen,%.0s' {1..$LAYERS}); P_FZ=${P_FZ%,}

echo "== phase 1: per-arm LR (seed 0) =="
for lr in ${=LR_BP}; do
  cell bp     $P_BP direct    $lr 0
  cell frozen $P_FZ direct    $lr 0
done
for lr in ${=LR_DFA}; do
  cell dfa-structure $P_DFA structure $lr 0
  cell dfa-direct    $P_DFA direct    $lr 0
done

echo "== phase 2: seeds at each arm's own best LR =="
B_BP=$(best_lr bp ${=LR_BP})
B_FZ=$(best_lr frozen ${=LR_BP})
B_DS=$(best_lr dfa-structure ${=LR_DFA})
B_DD=$(best_lr dfa-direct ${=LR_DFA})
echo "  best LR: bp=$B_BP frozen=$B_FZ dfa-structure=$B_DS dfa-direct=$B_DD"
s=1
while [[ $s -lt $SEEDS ]]; do
  cell bp            $P_BP  direct    $B_BP $s
  cell frozen        $P_FZ  direct    $B_FZ $s
  cell dfa-structure $P_DFA structure $B_DS $s
  cell dfa-direct    $P_DFA direct    $B_DD $s
  s=$((s + 1))
done
echo "== G1 done: $OUT =="
