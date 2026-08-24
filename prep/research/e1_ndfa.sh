#!/bin/zsh
# prep/research/e1_ndfa.sh — toy#172/E1 Phase 1.2: the nDFA error-side
# preconditioner across the P6 rank ladder.
#
# WHAT IS AND IS NOT MEASURED HERE.
#
# nDFA folds lambda(C_E + lambda I)^-1 into the DFA feedback matrix B, so
# it exists ONLY on the dfa arm — the runner REFUSES it on a policy with
# no `dfa` block, precisely so it cannot be run as a silent no-op. That
# has a consequence for E1's stated discriminator ("nDFA must move the
# DFA arm while leaving bp and frozen unmoved"): bp and frozen are
# unmoved BY CONSTRUCTION, not by measurement, because there is no B on
# those arms to precondition. They are still run — they are the
# excess-over-noise reference the dfa curve is stated against — but the
# "does it move all three" branch of the verdict is answered
# structurally, and the report says so rather than presenting a
# tautology as evidence.
#
# THE REUSED CELLS. GTX_NDFA=0 is byte-identical to the pre-flag runner
# (verified by diffing a real run against a saved copy of the binary, and
# gated in prep/gtx_gate.rb leg 16), so the nDFA=off arms are the
# EXISTING P6 cells — same device, same binary, same seeds. Set
# BASE=/srv/data/scratch/p6. Nothing is re-run to produce a number that
# is already bit-reproducible; the gate is what makes that legitimate.
#
# PER-RUNG LR, carried from P6. The DFA optimum slid 3e-4 -> 1e-3 and
# then CRASHED to 3e-5 at the top rung, so a single LR would measure the
# wrong cell at four of six rungs. PHASE=bracket re-brackets the nDFA arm
# itself at the top rung: gain=preserve holds ||B||_F fixed so the
# first-order LR coupling is already removed, but "an arm measured at
# ANOTHER arm's cell is not a negative" is this program's most expensive
# rule and it applies to nDFA too.
set -e
# repo root is TWO levels up from prep/research/ (was one, from prep/).
cd "$(dirname "$0")/../.."
OUT=${OUT:-/srv/data/scratch/e1ndfa}
BASE=${BASE:-/srv/data/scratch/p6}
mkdir -p "$OUT"
STEPS=${STEPS:-4000}
BLOCKS=4
HEAD=${HEAD:-4096}
SEEDS=${SEEDS:-3}
EVERY=${EVERY:-500}
M=${M:-256}
GAIN=${GAIN:-preserve}
PHASE=${PHASE:-sweep}
RUNGS=${RUNGS:-"ae_shak_a65 ae_shak_a508 ae_shak_a1008 ae_shak_a2504"}
# THE RIDGE IS READ AGAINST THE INSTRUMENT, not picked round. P's
# eigenvalues are lambda/(lambda + s_i), so the grid has to straddle
# lambda_max(C_E) or every cell is either the identity or a projector.
# Phase 1.1 measured lambda_max(C_E) = lambda_max(G)/n at 0.121 (rung 65)
# down to 0.0079 (rung 2504) — it FALLS with width, which is the same
# anisotropy result seen from the other side. 1e-2 sits above it at every
# rung, 1e-4 well below at every rung.
LAMBDAS=${LAMBDAS:-"0.01 0.001 0.0001"}
# The large-lambda control. P -> I as lambda grows and the correction
# underflows against B's own magnitude in f64, so this arm must come out
# BYTE-IDENTICAL to the nDFA=off cell. It is gated, and it is also run
# live here at one seed per rung: a gate on a 40-step toy config and a
# 4000-step ladder cell are not the same evidence.
LAM_INF=${LAM_INF:-1e30}
BRACKET_LR=${BRACKET_LR:-"0.00001 0.00003 0.0001 0.0003 0.001"}
BRACKET_RUNG=${BRACKET_RUNG:-ae_shak_a2504}
BRACKET_LAM=${BRACKET_LAM:-0.01}

rep () {
  local t=$1
  local n=$2
  local s=$t
  for _ in $(seq 2 $n); do s="$s,$t"; done
  echo $s
}
P_DFA=$(rep dfa $BLOCKS)

# One `local` per line — zsh binds every RHS on a `local` line before any
# assignment takes effect, which once turned a frozen control into a
# trained one in silence (toy#170).
lr_for () {        # lr_for <rung>
  local rung=$1
  # LR_OVERRIDE exists for exactly one job: running seeds at nDFA's OWN
  # bracketed optimum once PHASE=bracket has found that it differs from
  # the plain arm's. "An arm measured at ANOTHER arm's cell is not a
  # negative" applies to nDFA too, and at rank 65 its optimum really did
  # move (3e-4 -> 1e-4).
  if [[ -n $LR_OVERRIDE ]]; then
    echo $LR_OVERRIDE
    return 0
  fi
  case "$rung" in
    ae_shak_a65)   echo 0.0003 ;;
    ae_shak_a2504) echo 0.00003 ;;
    *)             echo 0.001 ;;
  esac
}

cell () {          # cell <rung> <lambda> <lr> <seed>
  local rung=$1
  local lam=$2
  local lr=$3
  local seed=$4
  local f="$OUT/dfa-ndfa_${rung}_v${HEAD}_lam${lam}_e${EVERY}_m${M}_${GAIN}_lr${lr}_s${seed}.txt"
  [[ -s $f ]] && return 0
  env GTX_TASK=bytelm GTX_TEXT=data/$rung GTX_CONTEXT=128 \
      GTX_D_MODEL=128 GTX_HEADS=4 GTX_D_FF=256 GTX_VAL_BATCHES=8 \
      GTX_BLOCKS=$BLOCKS STEPS=$STEPS SEED=$seed GTX_POLICY=$P_DFA \
      GTX_DFA_CUT=layer GTX_LR=$lr GTX_VOCAB=$HEAD \
      GTX_NDFA=1 GTX_NDFA_LAMBDA=$lam GTX_NDFA_EVERY=$EVERY \
      GTX_NDFA_M=$M GTX_NDFA_GAIN=$GAIN ./libexec/toy-train-gtx \
      > "$f" 2>&1 || echo "  CELL FAILED: $f"
  # Every assertion the P6 harness makes, plus the two this arm adds: the
  # preconditioner must have actually refreshed, and its error norm must
  # be a number.
  grep -q "pack=data/$rung " "$f" || echo "  RUNG MISMATCH in $f"
  grep -q "b_dim=$HEAD " "$f"     || echo "  B-WIDTH MISMATCH in $f"
  grep -q "dfa_wired=$BLOCKS " "$f" || echo "  DFA ARM NOT FULLY WIRED in $f"
  grep -q "^ndfa: on=1 " "$f"     || echo "  NO ndfa LINE in $f"
  grep -q "err_post=null" "$f"    && echo "  NON-FINITE PRECONDITIONED ERROR in $f"
  echo "  ndfa $rung lam=$lam lr=$lr s=$seed -> $(grep -o 'bpb=[0-9.]*' "$f" | head -1) $(grep -o 'err_post=[0-9.e+-]*' "$f" | head -1) $(grep -o 'b_shrink=[0-9.e+-]*' "$f" | head -1) $(grep -o 'refreshes=[0-9]*' "$f" | head -1)"
}

if [[ $PHASE == bracket ]]; then
  echo "== nDFA LR BRACKET: $BRACKET_RUNG lambda=$BRACKET_LAM (seed 0) =="
  echo "   plain-DFA best LR at this rung: $(lr_for $BRACKET_RUNG)"
  for lr in ${=BRACKET_LR}; do
    cell $BRACKET_RUNG $BRACKET_LAM $lr 0
  done
  echo "== bracket done: $OUT =="
  exit 0
fi

for r in ${=RUNGS}; do
  echo "== $r (head $HEAD, every $EVERY, m $M, gain $GAIN) =="
  LR=$(lr_for $r)
  echo "   lr=$LR (P6's per-rung dfa optimum)"
  s=0
  while [[ $s -lt $SEEDS ]]; do
    for lam in ${=LAMBDAS}; do
      cell $r $lam $LR $s
    done
    s=$((s + 1))
  done
  # The large-lambda control, seed 0 only: it is a byte identity, so a
  # second seed measures the same identity twice.
  cell $r $LAM_INF $LR 0
done
echo "== E1 nDFA sweep done: $OUT (nDFA=off arms are reused from $BASE) =="
