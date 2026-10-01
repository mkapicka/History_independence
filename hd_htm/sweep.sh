#!/bin/bash
# =============================================================================
# sweep.sh -- MetaCentrum (PBS Pro) array job for sweep_mu2
#
# One mu2 point per array task, so the sweep runs N_POINTS-wide in parallel.
# No code change is needed: sweep_mu2.jl already accepts key=value arguments and
# writes a result file whose name encodes mu2, so tasks never collide.
#
# Submit from the directory holding the hd_htm code (PBS_O_WORKDIR is that
# directory, which is why no username or /storage path is hardcoded):
#
#     cd ~/hd_htm && qsub sweep.sh          # the full 10-point sweep
#     cd ~/hd_htm && qsub -J 1-2 sweep.sh   # 2 tasks, to test the pipeline first
#
# PBS Pro rejects a one-element array range -- `qsub -J 1-1` fails with
# "illegal -J value" -- so use 1-2 for a trial. Alternatively comment out the
# `#PBS -J` line below and plain `qsub sweep.sh` runs a single task at index 1.
#
# Monitor / cancel:
#     qstat -u $USER          # queued and running
#     qstat -x -u $USER       # including finished
#     qdel <jobid>[]          # kill the whole array
#
# Per-task logs land in sweep_mu2.o<jobid>.<index> and .e<jobid>.<index>.
# =============================================================================

#PBS -N sweep_mu2
#PBS -l select=1:ncpus=8:mem=128gb:scratch_local=10gb
#PBS -l walltime=168:00:00
#PBS -J 1-20

# -----------------------------------------------------------------------------
# Configuration. NOTE: -J above cannot read shell variables, so if you change
# N_POINTS you must change the -J range to match.
#
# WALLTIME. PBS routes to a queue by the walltime requested (2h, 4h, 1d, 2d,
# ...), and kills the job hard at the limit -- results are copied back only
# after julia returns, so a timed-out task produces nothing at all. Measured
# per-point cost scales roughly 97x from the nZ=5/nEps=5/nH=101 settings to
# nZ=15/nEps=11/nH=151, i.e. about 4 hours per point, so 24h leaves a ~6x
# margin. Drop back to 4:00:00 (queue q_4h, shorter waits) if you return to the
# lighter shock grids. Note mu2=0 is ~100x cheaper than every other point,
# because the s2 grid collapses to [0.0] there -- do not size from task 1.
# -----------------------------------------------------------------------------
N_POINTS=20                 # must equal the -J range above
MU2_MIN=0.0
MU2_MAX=0.98

# Model settings passed straight through to sweep_mu2. Anything the REPL call
# accepts works here.
#
# alpha=0 puts all weight on the mu2 block, i.e. the one-root tax
# theta_k = theta0*mu2^k. It is set EXPLICITLY because HD_SETTINGS uses
# alpha=:paper, which derives alpha = (rho-mu1)/(mu2-mu1) = rho/mu2 at mu1 = 0 --
# that exceeds 1 for every mu2 below rho = 0.958 and gives a signed mixture whose
# lambda solve fails. If you want the paper mixture instead, drop alpha below and
# set MU2_MIN=0.958 so the roots bracket rho.
# A bash ARRAY, not a string: expanded as "${MODEL_ARGS[@]}" each element stays
# one argument no matter what it contains. A plain string would rely on
# unquoted word-splitting, which bash does and zsh does not -- so the same line
# behaves differently depending on the shell that runs it.
# lambdaMin=1e-5 lowers the lambda bracket floor below sweep_mu2.jl's own
# LAMBDA_MIN_SWEEP = 1e-3. That constant was sized on a saver-only sweep,
# where the budget-clearing lambda bottoms out near 0.007 at mu2 = 0.98.
# Two things push it lower here: mu2 runs to 0.99, where theta0 collapses
# to 0.030 and the hours exponent (1-tau)*theta0 to 0.025; and
# hand-to-mouth households consume their income rather than smoothing, so
# the level of the tax function that balances the budget differs. A root
# below the bracket makes the solver fall back to the best endpoint
# rather than fail, so the floor is worth setting generously.
# pSS/pHH are the asset-market-access chain, calibrated to Kaplan, Violante
# and Weidner (2014) Table 4 (see references/KVW2014_WealthyHandToMouth/ and
# paper/notes/DOT_AccessChainAnnualization.tex). They are passed EXPLICITLY
# rather than left to HD_SETTINGS so the submitted job records the chain it
# ran; pSS=1 pHH=0 reproduces the hd sweep with no hand-to-mouth agents.
MODEL_ARGS=(J=79 pSS=0.8882970895 pHH=0.7589294614 mu1=0 alpha=0 nZ=15 nEps=11 nKappa=5 tau=0.181 nS1=1 nS2=151  nA=151 labor_grid_size=151 s_grid_method=:quantile collect_distributions=false qSav=0.983574309039048 qBorr=0.983574309039048 qGov=0.983574309039048 bbar=-0.17378794875598913 beta=0.983385123227395 aMax=50.0 age0_real=20 stats_age_lo=3 stats_age_hi=39 lambdaMin=1e-5)

# nS1=1 because mu1=0 collapses the s1 grid to the single point [0.0] regardless
# of nS1 -- carrying 7 would multiply the state space sevenfold for nothing.

# ncpus=8 is deliberate: measured thread scaling on this solver is 3.93x at 8
# threads and 4.04x at 10, so the serial fraction (the sequential lambda Brent
# loop, and a distribution phase that threads only over nKappa) caps the return
# well before large core counts. More array tasks beat more cores per task, and
# a smaller request also queues sooner.

set -o pipefail
trap 'clean_scratch' TERM EXIT      # always release scratch, including on failure

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
# PIN THE VERSION. Bare `module add julia` loads julia/1.7.0 here, which cannot
# read this project's Manifest.toml (julia_version = "1.12.6",
# manifest_format = "2.0"): it reports every package as "required but does not
# seem to be installed", including JLD2. 1.12.6 matches the Manifest exactly, so
# the pinned environment is used as recorded rather than re-resolved.
module add julia/1.12.6 || {
    echo "ERROR: could not load julia/1.12.6 -- check 'module avail julia/'" >&2
    exit 1
}
echo "julia: $(julia --version)"

# The depot MUST be the one you ran Pkg.instantiate() into on the frontend.
# $HOME is not reliable here: in a batch job it can resolve differently from the
# login shell (or be unset), and Julia then reports every package as "required
# but does not seem to be installed". PBS exports PBS_O_HOME, the HOME of the
# submitting environment, which is the right value -- fall back to $HOME only if
# it is missing. Set JULIA_DEPOT_OVERRIDE below to a literal path
# (e.g. /storage/praha1/home/YOURUSER/.julia) if both are wrong.
JULIA_DEPOT_OVERRIDE=""
export JULIA_DEPOT_PATH=${JULIA_DEPOT_OVERRIDE:-${PBS_O_HOME:-$HOME}/.julia}

echo "depot: $JULIA_DEPOT_PATH"
test -d "$JULIA_DEPOT_PATH/packages" || {
    echo "ERROR: no package depot at $JULIA_DEPOT_PATH" >&2
    echo "       On the FRONTEND (compute nodes have no network), run:" >&2
    echo "         cd ~/hd_htm && module add julia" >&2
    echo "         export JULIA_DEPOT_PATH=\$HOME/.julia" >&2
    echo "         julia --project=. -e 'using Pkg; Pkg.instantiate()'" >&2
    exit 1
}

# -----------------------------------------------------------------------------
# Stage the code onto the node-local scratch disk
# -----------------------------------------------------------------------------
test -n "$SCRATCHDIR" || { echo "ERROR: SCRATCHDIR is not set" >&2; exit 1; }
# The Manifest records the shared package BewleyCommon as a dev dependency at
# the relative path ../common, so scratch must reproduce that layout: the code
# goes in $SCRATCHDIR/run and the package (its Project.toml and src/) in
# $SCRATCHDIR/common. Copying everything flat into $SCRATCHDIR makes the path
# resolve ABOVE the scratch directory and every task dies at load time.
mkdir -p "$SCRATCHDIR/run" "$SCRATCHDIR/common" || exit 2
cp -r "$PBS_O_WORKDIR"/*.jl "$PBS_O_WORKDIR"/Project.toml "$PBS_O_WORKDIR"/Manifest.toml \
      "$SCRATCHDIR/run"/ || { echo "ERROR: could not copy the code to scratch" >&2; exit 2; }
cp -r "$PBS_O_WORKDIR"/../common/Project.toml "$PBS_O_WORKDIR"/../common/src \
      "$SCRATCHDIR/common"/ || { echo "ERROR: could not copy common/ to scratch" >&2; exit 2; }
cd "$SCRATCHDIR/run" || exit 2

# -----------------------------------------------------------------------------
# This task's mu2 point
# -----------------------------------------------------------------------------
# Default to index 1 when this is NOT an array job, so the script also works as
# a plain `qsub sweep.sh` with the -J directive commented out. Without the
# default an unset PBS_ARRAY_INDEX makes awk read it as 0 and return a mu2 one
# step BELOW mu2_min, silently solving the wrong model.
IDX=${PBS_ARRAY_INDEX:-1}

MU2=$(awk -v i="$IDX" -v n="$N_POINTS" -v lo="$MU2_MIN" -v hi="$MU2_MAX" \
      'BEGIN { printf "%.10f", (n > 1) ? lo + (i - 1) * (hi - lo) / (n - 1) : lo }')

echo "task $IDX/$N_POINTS  mu2=$MU2  threads=$PBS_NCPUS  host=$(hostname)"
echo "args: ${MODEL_ARGS[*]}"

# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------
julia --project=. -t "$PBS_NCPUS" sweep_mu2.jl \
      mu2_min="$MU2" mu2_max="$MU2" n_mu2=1 "${MODEL_ARGS[@]}" \
    || { echo "ERROR: julia exited nonzero for mu2=$MU2" >&2; exit 3; }

# -----------------------------------------------------------------------------
# Bring the results back. Each task writes its own filename (it encodes mu2), so
# tasks finishing concurrently cannot overwrite one another.
# -----------------------------------------------------------------------------
mkdir -p "$PBS_O_WORKDIR/results"
# $SCRATCHDIR/run, not $SCRATCHDIR: julia runs with that as its working
# directory (see the staging block above), so sweep_mu2.jl's relative
# "results/" lands there.
cp -r "$SCRATCHDIR"/run/results/* "$PBS_O_WORKDIR/results/" \
    || { echo "ERROR: could not copy results back" >&2; exit 4; }

echo "task $IDX done"
