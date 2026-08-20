#!/bin/zsh
# prep/g1_leafsep.sh — toy#173 (G1) read 3: LEAF SEPARABILITY.
#
# The dimension read says structure is small (76, saturating) while the
# leaf vocabulary is open (1091 at 400 functions, growing ~n^0.79). So
# `go` turns on whether the leaves FACTOR: can leaf TYPE be predicted
# from context, and does predicting it interfere with predicting
# structure?
#
# Two packs, identical corpus and identical engine:
#   full   — leaves as three types (IDENTIFIER / NUMBER / STRING)
#   merged — the same three collapsed into ONE leaf class
#
# If structural accuracy is unchanged between them, leaf-type prediction
# is separable from structural prediction and a G2 could factor leaves to
# their own mechanism. If collapsing them MOVES structural accuracy, the
# two are entangled and the small-output claim does not survive the
# leaves — which is the honest `no-go` on separability.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-/srv/data/scratch/g1leaf}
mkdir -p "$OUT"
STEPS=${STEPS:-3000}
SEEDS=${SEEDS:-5}
LR=${LR:-0.03}

python3 prep/build_ast_pack.py --out data/ast_leafmerged --merge-leaves >/dev/null 2>&1 || {
  echo "build_ast_pack.py has no --merge-leaves yet"; exit 2; }

for pack in ast_code ast_leafmerged; do
  s=0
  while [[ $s -lt $SEEDS ]]; do
    f="$OUT/bp_${pack}_s${s}.txt"
    if [[ ! -s $f ]]; then
      env GNN_GRAPH=data/$pack GNN_POLICY=chain,chain GNN_HIDDEN=64 GNN_LAYERS=2 \
          STEPS=$STEPS SEED=$s GNN_LR=$LR ./libexec/toy-train-gnn > "$f" 2>&1 || echo "  FAILED $f"
    fi
    echo "  $pack s=$s -> $(grep -oE 'acc=[0-9.]+' "$f" | tail -1)"
    s=$((s + 1))
  done
done
echo "== leaf-separability done: $OUT =="
