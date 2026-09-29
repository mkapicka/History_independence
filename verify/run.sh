#!/bin/bash
# =============================================================================
# run.sh
#
# Runs the golden-master harness over every solver directory and writes the
# results under verify/<tag>/. Compare two tags with a plain diff:
#
#   ./verify/run.sh baseline          # before a change
#   ./verify/run.sh after             # after it
#   diff -r verify/baseline verify/after && echo "no number changed"
#
# Each directory has its own Julia environment, so each runs under its own
# --project. Directories can be named to run a subset:
#
#   ./verify/run.sh after hi hiinf
#
# Marek Kapicka, 2026
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

TAG="${1:?usage: run.sh <tag> [dirs...]}"
shift
DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then
    DIRS=(hi hi_htm hiinf hiinf_htm hd hd_htm hdinf hdinf_htm)
fi

mkdir -p "verify/$TAG"
fail=0
for d in "${DIRS[@]}"; do
    out="verify/$TAG/$d.txt"
    if julia --startup-file=no --project="$d" -t 2 verify/golden.jl "$d" "$out" \
         > "verify/$TAG/$d.log" 2>&1; then
        :
    else
        echo "  FAILED $d  (see verify/$TAG/$d.log)"
        fail=1
    fi
done
[ $fail -eq 0 ] && echo "  all ${#DIRS[@]} directories captured under verify/$TAG/"
exit $fail
