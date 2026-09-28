# =============================================================================
# join_results.jl
#
# Merges the per-point sweep files and reports the moments in the units
# the calibration targets.
# For the history-dependent tax model with hand-to-mouth agents.
#
# Marek Kapicka, 2026
# =============================================================================

using JLD2, Printf

# =============================================================================
# join_results.jl -- merge the per-point sweep files and report the moments in
# the units the calibration targets.
#
# HAND-TO-MOUTH VARIANT, for sweeps written by this directory (FINITE horizon).
# They differ from `hd` sweeps in two ways, and guard 0 below refuses to merge
# across either: the asset-market-access chain, and the fact that `hd` sweeps
# record LEVELS only -- `hd/sweep_mu2.jl` stores no `meanLaborIncome`, so its
# files cannot be converted to the ratios the calibration targets. That defect
# is fixed in this directory's `sweep_mu2.jl`.
#
# `hd/join_results.jl` is a fourteen-line stub that concatenates three level
# series with no guards at all. This is the full version, ported from
# `hdinf_htm`.
#
# RATIOS, NOT LEVELS. `meanAssets` / `medianAssets` in a sweep file are LEVELS.
# The calibration targets mean and median assets RELATIVE to mean labor income,
# and the two differ by a denominator that varies across mu2, so a level series
# is not a rescaled ratio series. Sweeps written before that was recorded carry
# no denominator at all and cannot be converted after the fact -- this script
# says so rather than silently plotting the wrong quantity.
# =============================================================================

const N_EXPECTED = nothing        # -J range in sweep.sh; set to `nothing` to skip the check

# Which files to merge. Edit MATCH to pick a sweep; the per-task files written
# by sweep.sh are all `sweep_mu2_n=1_...`, and the name now carries nS1, nS2,
# pSS and pHH, so any of them can be selected here. RESULTS_DIR and MATCH can
# also be overridden from the command line:
#     julia --project=. join_results.jl results "nS2=101"
const RESULTS_DIR = length(ARGS) >= 1 ? ARGS[1] : "results"
const MATCH       = length(ARGS) >= 2 ? ARGS[2] : "nS2=151"

files = sort(filter(f -> occursin(MATCH, f) && startswith(f, "sweep_mu2_n=1_"),
                    readdir(RESULTS_DIR)))
s = [JLD2.load(joinpath(RESULTS_DIR, f))["sweep"] for f in files]
isempty(s) && error("no sweep files matching \"$(MATCH)\" in $(RESULTS_DIR)/")

# --- guards, cheapest first --------------------------------------------------
# 0. One access chain across the merged files. Checked FIRST because it is the
#    only mismatch that yields a plausible-looking plot of two different models
#    instead of an error.
all(x -> hasproperty(x, :pSS), s) || error("""
    at least one sweep file records no access chain (pSS/pHH). Those are
    `hd` sweeps, written before the hand-to-mouth extension, and cannot be
    distinguished from a piH = 0 run of this directory -- re-run them here, or
    read them with hd/join_results.jl.""")
let chains = unique([(round(x.pSS, digits = 12), round(x.pHH, digits = 12)) for x in s])
    length(chains) == 1 || error(
        "the sweep files mix $(length(chains)) access chains $(chains); those " *
        "are different models. Separate them by pSS/pHH -- the filenames " *
        "carry both.")
end
# 1. Completeness. A partial sweep silently truncates every series below, and
#    indexing it with a hardcoded 1:N throws far from the cause.
if N_EXPECTED !== nothing && length(s) != N_EXPECTED
    @warn "incomplete sweep" have = length(s) expected = N_EXPECTED files = length(files)
end

# 2. Did any point THROW? This must be checked before `converged`: a point that
#    threw never produced an equilibrium, so its `converged` entry is false for
#    a reason that has nothing to do with the government budget.
errs = vcat((x.errors for x in s)...)
nthrew = count(!isnothing, errs)
if nthrew > 0
    for (i, e) in enumerate(errs)
        e === nothing || @printf("  point %d threw: %s\n", i, first(e, 100))
    end
    error("$(nthrew) of $(length(errs)) points threw; fix those before reading the sweep")
end

# 3. Did lambda balance the budget?
conv = vcat((x.converged for x in s)...)
all(conv) || @warn "some points did not converge" nconverged = count(conv) n = length(conv) resid = vcat((x.govBudgetResidual for x in s)...)[.!conv]

mu2 = vcat((x.mu2 for x in s)...)
W   = vcat((x.overallValueFunction for x in s)...)


# --- the series --------------------------------------------------------------
hasratio = hasproperty(first(s), :meanAssetsToMeanLaborIncome)
hasratio || error("""
    these sweep files predate the ratio fields and store LEVELS only.
    `meanLaborIncome` was not recorded, so the ratio cannot be recovered --
    re-run the sweep with this directory's sweep_mu2.jl.""")

mu2       = vcat((x.mu2 for x in s)...)
W         = vcat((x.overallValueFunction for x in s)...)
AmeanToY  = vcat((x.meanAssetsToMeanLaborIncome   for x in s)...)
AmedToY   = vcat((x.medianAssetsToMeanLaborIncome for x in s)...)
Ymean     = vcat((x.meanLaborIncome for x in s)...)
# Entry-age cross-section, if the sweep recorded it. Same denominator as the
# window ratios above, so the four are directly comparable.
haslo     = hasproperty(first(s), :meanAssetsAtStatsAgeLoToMeanLaborIncome)
AmeanLo   = haslo ? vcat((x.meanAssetsAtStatsAgeLoToMeanLaborIncome   for x in s)...) : fill(NaN, length(mu2))
AmedLo    = haslo ? vcat((x.medianAssetsAtStatsAgeLoToMeanLaborIncome for x in s)...) : fill(NaN, length(mu2))
# Real age the window opens at, read from the settings the run actually used.
st        = first(s).settings
loage     = (haskey(st, :age0_real) && haskey(st, :stats_age_lo)) ?
            st.age0_real + st.stats_age_lo - 1 : 0
Amean     = vcat((x.meanAssets   for x in s)...)   # levels, kept for reference
Amed      = vcat((x.medianAssets for x in s)...)
# Realized hand-to-mouth mass per point, which must equal piH everywhere: the
# access chain starts stationary, so nothing should move it.
HtM       = hasproperty(first(s), :shareHandToMouth) ?
            vcat((x.shareHandToMouth for x in s)...) : fill(NaN, length(mu2))

ord = sortperm(mu2)
mu2, W, AmeanToY, AmedToY, Ymean, Amean, Amed, AmeanLo, AmedLo, HtM, conv =
    mu2[ord], W[ord], AmeanToY[ord], AmedToY[ord], Ymean[ord],
    Amean[ord], Amed[ord], AmeanLo[ord], AmedLo[ord], HtM[ord], conv[ord]

haslo || @warn "entry-age ratios not recorded in these files; the two age-$(loage) columns will read NaN"

piH = first(s).piH
@printf("\naccess chain: pSS = %.6f, pHH = %.6f  ->  piH = %.6f%s\n",
        first(s).pSS, first(s).pHH, piH,
        piH == 0.0 ? "   [NO HtM AGENTS: equivalent to an hd sweep]" : "")
# The tolerance is RELATIVE and deliberately loose. The realized share is not
# exact: `massTol` in the forward pass skips cells below 1e-14, and that
# truncation accumulates over the state space without cancelling between the
# two access states. Measured, the deviation scales with the number of cells --
# 1.4e-12 at 43,050 cells, 7.8e-10 at the 7,524,330 of a production grid
# (nA=151, nS2=151, nZ=15, nEps=11). An ABSOLUTE 1e-10, which this used, fires
# on that arithmetic noise. 1e-6 relative sits four orders above the noise and
# four below anything economically meaningful, so it still catches a genuine
# loss of access mass.
let bad = findall(x -> isfinite(x) && abs(x - piH) > 1e-6 * piH, HtM)
    isempty(bad) || @warn("realized hand-to-mouth share departs from piH; the " *
                          "forward pass did not preserve access mass",
                          points = bad, piH = piH, realized = HtM[bad])
end

@printf("\n%4s %8s %13s %13s %13s %13s %11s %12s %5s\n",
        "i", "mu2", "meanA/meanY", "medA/meanY",
        "meanA$(loage)/Y", "medA$(loage)/Y", "meanY", "valueFn", "conv")
for i in eachindex(mu2)
    @printf("%4d %8.4f %13.8f %13.8f %13.8f %13.8f %11.8f %12.8f %5s\n",
            i, mu2[i], AmeanToY[i], AmedToY[i], AmeanLo[i], AmedLo[i],
            Ymean[i], W[i], conv[i] ? "yes" : "NO")
end
@printf("\ncolumns 3-6 all divide by mean labor income over model ages %s-%s (real %s-%s)\n",
        get(st, :stats_age_lo, "?"), get(st, :stats_age_hi, "?"),
        loage, haskey(st, :age0_real) && haskey(st, :stats_age_hi) ?
               st.age0_real + st.stats_age_hi - 1 : "?")
println()
