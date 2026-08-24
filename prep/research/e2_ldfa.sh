#!/bin/zsh
# prep/research/e2_ldfa.sh — toy#172/E2: LDFA, adaptive low-rank feedback, across
# the P6 rank ladder.
#
# THE HYPOTHESIS IS A CONTRAST, NOT A CURVE. LDFA (Hanut & Kadmon 2026)
# predicts that a FIXED low-rank feedback path still degrades with output
# width (spurious fixed points) while an ADAPTIVE one — P tracking the
# error's top-r subspace by Oja's rule — flattens and keeps recovery
# positive. So `dfa-fixed-lowrank` vs `dfa-adaptive-lowrank` AT EACH RUNG
# is the whole measurement, and `dfa-fixed-wide` is the P6 negative it is
# stated against.
#
# THIS IS A CLEAN B-TEST, unlike E1 Phase 1.2's. nDFA's discriminator
# ("moves dfa, leaves bp and frozen alone") was true BY CONSTRUCTION — B
# exists only on the dfa arm — and carried no evidential weight. Here both
# arms modify B, both collect the same error samples on the same steps,
# and the ONLY difference is whether the Oja update is applied. The
# runner gates that: at eta=0 the two arms are byte-identical.
#
# ── EVERYTHING IS RE-RUN ON THE CUDA TWIN, INCLUDING THE BASELINE ──
#
# P6's cells are CPU (verified here: a stored P6 cell reproduces
# BIT-FOR-BIT against the pre-E2 binary, so reuse would have been
# legitimate on that device). But this sweep runs on the tao#24 twin — a
# ladder cell is 18 s there against 103 s on CPU — and cross-device cells
# are NOT numerically comparable. The E-series rule is one device
# throughout, so `dfa-fixed-wide`, `bp` and `frozen` are all re-measured
# here rather than quoted across devices.
#
# ── PER-ARM, PER-RUNG LR, AND IT IS NOT A COURTESY ──
#
# The DFA optimum slid 3e-4 -> 1e-3 across the P6 ladder and then CRASHED
# to 3e-5 at the top rung. "An arm measured at ANOTHER arm's cell is not a
# negative" is this program's most expensive rule; PHASE=bracket runs the
# grid for every arm x rung and prints an EDGE OPTIMUM warning, because an
# optimum on a grid boundary is not an optimum.
#
# ── THE SCALE CONTROL ──
#
# Every low-rank cell asserts `scale_ratio` = 1 to 1e-9. A rank-r Q.P has
# a different Frobenius norm from the full-width B it replaces, so an
# unnormalised arm would make "low rank hurts" indistinguishable from "the
# updates got smaller". The arms differ in RANK and not in SCALE, and the
# harness refuses to file a cell where that is not true of the run itself.
#
# ── ETA IS A PER-LADDER CONSTANT AND IT HAD TO BE RE-CHOSEN ──
#
# Oja's rule steps by eta * C_E, and E1 Phase 1.1 measured lambda_max(C_E)
# FALLING 0.121 -> 0.0079 across this ladder. So a fixed eta converges
# ~15x slower at the top rung than at the base, and the FIRST sweep here
# (eta 0.05) showed exactly that: captured energy ran 76x the random
# baseline at rung 65 and 1.1x at rung 2504. A null contrast read off that
# would have been a CONVERGENCE FAILURE reported as a hypothesis test.
#
# eta is therefore chosen against `p_energy` — a MECHANISM diagnostic, not
# the outcome — so the choice is not tuning on the result. Measured at
# r=64, seed 0, over {0.05, 0.2, 1.0, 2.0, 5.0}: energy rises monotonically
# with eta at every rung; eta 5.0 DIVERGES at every rung and eta 2.0
# diverges at rung 65 (the runner aborts, it does not renormalise); eta 1.0
# is the largest value that holds everywhere, and it lands the adaptation
# at 4-184x the random baseline across the whole ladder. That is the
# default. Both settings are reported — the ladder was run twice.
set -e
# repo root is TWO levels up from prep/research/ (was one, from prep/).
cd "$(dirname "$0")/../.."
OUT=${OUT:-/srv/data/scratch/e2ldfa}
mkdir -p "$OUT"
STEPS=${STEPS:-4000}
BLOCKS=4
HEAD=${HEAD:-4096}
SEEDS=${SEEDS:-3}
EVERY=${EVERY:-500}
M=${M:-128}
ETA=${ETA:-1.0}
PHASE=${PHASE:-sweep}
RUNGS=${RUNGS:-"ae_shak_a65 ae_shak_a508 ae_shak_a1008 ae_shak_a2504"}
# r straddles dout = GTX_D_MODEL = 128 deliberately. rank(Q.P) <=
# min(dout, r), so 16 and 64 are genuine rank reductions while 256 buys
# the SAME matrix rank as full width and only confines the row space to
# P's span. That is still a real intervention under `oja` (the span is the
# error's dominant subspace) and it is close to a no-op under `none` — a
# prediction the ladder tests rather than assumes.
RANKS=${RANKS:-"16 64 256"}
# E1 Phase 1.1 measured the error covariance's stable rank at ~128 at
# V=2504, so LDFA's claim that required rank tracks TASK dimension rather
# than output width predicts r=256 ample and r=16 not.
RUNNER=${RUNNER:-./libexec/toy-train-gtx-cuda}
LR_DFA=${LR_DFA:-"0.00001 0.00003 0.0001 0.0003 0.001 0.003"}
# The bp/frozen grid runs LOWER than P6's because it had to: at the top
# rung the FROZEN control's optimum landed on 3e-4, the boundary of P6's
# {3e-4 .. 1e-2}. Extending down to 1e-5 put it back in the interior at
# the same value — but the frozen control is the denominator of RECOVERY,
# so leaving it on an edge would have put an unmeasured arm in the
# headline.
LR_BP=${LR_BP:-"0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01"}

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

# One `local` per line — zsh binds every RHS on a `local` line before any
# assignment takes effect, which once turned a frozen control into a
# trained one in silence (toy#170).
arm_env () {       # arm_env <label> -> echoes the extra env for that arm
  local label=$1
  case "$label" in
    bp-body)     echo "GTX_POLICY=$P_BP" ;;
    frozen-body) echo "GTX_POLICY=$P_FZ" ;;
    dfa-wide)    echo "GTX_POLICY=$P_DFA" ;;
    dfa-fix-*)   echo "GTX_POLICY=$P_DFA GTX_DFA_RANK=${label#dfa-fix-} GTX_LDFA_EVERY=$EVERY GTX_LDFA_M=$M" ;;
    dfa-oja-*)   echo "GTX_POLICY=$P_DFA GTX_DFA_RANK=${label#dfa-oja-} GTX_DFA_ADAPT=oja GTX_LDFA_ETA=$ETA GTX_LDFA_EVERY=$EVERY GTX_LDFA_M=$M" ;;
    *)           echo "BAD_ARM=$label" ;;
  esac
}

cell () {          # cell <label> <rung> <lr> <seed>
  local label=$1
  local rung=$2
  local lr=$3
  local seed=$4
  local f="$OUT/${label}_${rung}_v${HEAD}_lr${lr}_s${seed}.txt"
  [[ -s $f ]] && return 0
  local extra
  extra=$(arm_env $label)
  # zsh does NOT word-split a scalar into separate assignments for `env`
  # (toy#141: `env $e` with e="A=1 B=2" silently dropped BOTH and two arms
  # ran the default looking bit-identical). ${=extra} forces the split,
  # and the per-cell greps below are what would catch it if it ever did
  # not.
  env GTX_TASK=bytelm GTX_TEXT=data/$rung GTX_CONTEXT=128 \
      GTX_D_MODEL=128 GTX_HEADS=4 GTX_D_FF=256 GTX_VAL_BATCHES=8 \
      GTX_BLOCKS=$BLOCKS STEPS=$STEPS SEED=$seed \
      GTX_DFA_CUT=layer GTX_LR=$lr GTX_VOCAB=$HEAD \
      ${=extra} $RUNNER > "$f" 2>&1 || echo "  CELL FAILED: $f"
  grep -q "pack=data/$rung " "$f" || echo "  RUNG MISMATCH in $f"
  grep -q "b_dim=$HEAD " "$f"     || echo "  B-WIDTH MISMATCH in $f"
  case "$label" in
    frozen-body) grep -q "frozen=$BLOCKS " "$f" || echo "  FROZEN ARM NOT FULLY FROZEN in $f" ;;
    dfa-*)       grep -q "dfa_wired=$BLOCKS " "$f" || echo "  DFA ARM NOT FULLY WIRED in $f" ;;
  esac
  case "$label" in
    dfa-fix-*|dfa-oja-*)
      # THE SCALE GATE, per cell. Not a global claim about the code: a
      # cell whose B_eff came out at a different scale from the
      # full-width B is confounded by a gain and must not be filed.
      grep -q "^ldfa: " "$f" || echo "  NO ldfa LINE in $f"
      local sr
      sr=$(grep -o 'scale_ratio=[0-9.e+-]*' "$f" | head -1 | cut -d= -f2)
      [[ -n $sr ]] && (( $(echo "($sr - 1.0) < 0.000000001 && (1.0 - $sr) < 0.000000001" | bc -l) )) \
        || echo "  SCALE MISMATCH ($sr) in $f — this cell differs from the wide arm in SCALE as well as RANK"
      ;;
  esac
  case "$label" in
    dfa-oja-*) grep -q "adapt=oja " "$f" || echo "  OJA ARM NOT ADAPTIVE in $f" ;;
    dfa-fix-*) grep -q "adapt=none " "$f" || echo "  FIXED ARM NOT FIXED in $f" ;;
  esac
  echo "  $label $rung lr=$lr s=$seed -> $(grep -o 'bpb=[0-9.]*' "$f" | head -1) $(grep -o 'p_energy=[0-9.e+-]*' "$f" | head -1) $(grep -o 'rank_eff=[0-9]*' "$f" | head -1)"
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
    echo "  NO BRACKET CELLS for $label $rung" >&2
    return 1
  fi
  if [[ $blr == $grid[1] || $blr == $grid[-1] ]]; then
    echo "  EDGE OPTIMUM: $label $rung best lr=$blr at grid boundary (bpb=$bv) — EXTEND, this rung is not measured" >&2
  fi
  echo $blr
}

ARMS=(bp-body frozen-body dfa-wide)
for r in ${=RANKS}; do
  ARMS+=(dfa-fix-$r dfa-oja-$r)
done

if [[ $PHASE == bracket ]]; then
  for rung in ${=RUNGS}; do
    echo "== BRACKET $rung (head $HEAD) =="
    for a in $ARMS; do
      case "$a" in
        bp-body|frozen-body) GRID=$LR_BP ;;
        *)                   GRID=$LR_DFA ;;
      esac
      for lr in ${=GRID}; do
        cell $a $rung $lr 0
      done
    done
  done
  echo "== bracket done: $OUT =="
  exit 0
fi

for rung in ${=RUNGS}; do
  echo "== $rung (head $HEAD, every $EVERY, m $M, eta $ETA) =="
  for a in $ARMS; do
    case "$a" in
      bp-body|frozen-body) GRID=$LR_BP ;;
      *)                   GRID=$LR_DFA ;;
    esac
    LR=$(best_lr $a $rung ${=GRID}) || continue
    echo "   $a lr=$LR (its OWN bracketed optimum at this rung)"
    s=0
    while [[ $s -lt $SEEDS ]]; do
      cell $a $rung $LR $s
      s=$((s + 1))
    done
  done
done
echo "== E2 LDFA sweep done: $OUT =="
