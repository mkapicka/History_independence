#!/bin/bash
# =============================================================================
# run.sh
#
# Runs the golden-master harness over every solver directory and writes the
# results under verify/<tag>/. Compare two tags with a plain diff:
#
#   ./verify/run.sh baseline          # before a change
#   ./verify/run.sh after             # after it
#   diff -r verify/baseline verify/after && echo "nothing changed"
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

# Pin the Julia version. The PATH default is whatever juliaup points at (1.13.0
# at the time of writing), and anything that writes a Manifest under it rewrites
# the file to manifest_format 2.1, which the cluster's newest Julia (1.12.6)
# cannot read. Pinning here keeps both the solves and any Pkg operation on the
# version the environments are resolved for.
JULIA="${JULIA:-julia +1.12.6}"

TAG="${1:?usage: run.sh <tag> [dirs...] [key=value ...]}"
shift
DIRS=()
OVERRIDES=()
for a in "$@"; do
    case "$a" in
        *=*) OVERRIDES+=("$a") ;;
        *)   DIRS+=("$a") ;;
    esac
done
if [ ${#DIRS[@]} -eq 0 ]; then
    DIRS=(hi hi_htm hiinf hiinf_htm hd hd_htm hdinf hdinf_htm)
fi

mkdir -p "verify/$TAG"
fail=0
for d in "${DIRS[@]}"; do
    out="verify/$TAG/$d.txt"
    if $JULIA --startup-file=no --project="$d" -t 2 verify/golden.jl "$d" "$out" \
         ${OVERRIDES[@]+"${OVERRIDES[@]}"} > "verify/$TAG/$d.log" 2>&1; then
        :
    else
        echo "  FAILED $d  (see verify/$TAG/$d.log)"
        fail=1
    fi
done
[ $fail -eq 0 ] && echo "  all ${#DIRS[@]} directories captured under verify/$TAG/"
exit $fail
