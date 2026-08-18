#!/bin/zsh
# prep/gate_profile.sh — per-leg wall time for the battery, done properly.
#
# An earlier ad-hoc version of this got two things wrong and both mattered:
#
#   * it fed a HAND-TYPED list of gate names, two of which (`gate-eval`,
#     `gate-infer`) do not exist. `make` errored in ~0s and the script
#     recorded that as a 0-second pass.
#   * it redirected output to /dev/null and NEVER CHECKED THE EXIT CODE,
#     so a failing leg and an instant leg looked identical.
#
# So the list is derived from the Makefile the same way GATES is, and the
# exit code is recorded next to every time. A leg that reports 0s AND
# rc!=0 is a phantom, not a fast gate.
#
# Runs SERIALLY by default. The ad-hoc version ran 4-way concurrent and
# its per-leg times summed to less than half the measured serial battery
# — concurrency makes each leg's own wall time unrepresentative, so a sum
# over concurrent measurements is not a serial estimate.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-/tmp/gate_profile.txt}
JOBS=${JOBS:-1}

ALL=$(grep -oE '^gate-[a-z0-9-]+' Makefile | sort -u)
case "$(uname)" in
  Darwin) LEGS=$(echo "$ALL" | grep -v -- '-cuda$' | grep -v '^gate-cuda$') ;;
  *)      LEGS=$(echo "$ALL" | grep -v -- '-metal$' | grep -v '^gate-metal$') ;;
esac

: > "$OUT"
echo "# leg profile: $(echo "$LEGS" | wc -l) legs, JOBS=$JOBS, $(date '+%F %T')" >> "$OUT"

one () {
  local g=$1
  local s=$(date +%s)
  local out
  out=$(make -s "$g" 2>&1) && local rc=0 || local rc=$?
  local e=$(date +%s)
  local pass=$(print -r -- "$out" | grep -c 'GATE PASS' || true)
  printf "%5ds  rc=%-3d pass=%-3d %s\n" $((e-s)) $rc $pass "$g" >> "$OUT"
}

if [[ $JOBS -le 1 ]]; then
  for g in ${=LEGS}; do one $g; done
else
  print -l ${=LEGS} | xargs -P $JOBS -I{} zsh -c "cd $(pwd); source $0 >/dev/null 2>&1; one {}" 2>/dev/null || \
    for g in ${=LEGS}; do one $g; done
fi

echo "PROFILE DONE" >> "$OUT"
