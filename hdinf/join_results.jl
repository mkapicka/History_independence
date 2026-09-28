using JLD2, Printf

# =============================================================================
# join_results.jl -- merge the per-point sweep files and report the moments in
# the units the calibration targets.
#
# RATIOS, NOT LEVELS. `meanAssets` / `medianAssets` in a sweep file are LEVELS.
# The calibration targets mean and median assets RELATIVE to mean labor income,
# and the two differ by a denominator that varies across mu2, so a level series
# is not a rescaled ratio series. Sweeps written before that was recorded carry
# no denominator at all and cannot be converted after the fact -- this script
# says so rather than silently plotting the wrong quantity.
# =============================================================================

const N_EXPECTED = nothing        # -J range in sweep.sh; set to `nothing` to skip the check

files = sort(filter(f -> occursin("nS2=151", f) && startswith(f, "sweep_mu2_n=1_"),
                    readdir("results")))
s = [JLD2.load(joinpath("results", f))["sweep"] for f in files]
isempty(s) && error("no sweep files matched in results/")

# --- guards, cheapest first --------------------------------------------------
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


# # --- the series --------------------------------------------------------------
# hasratio = hasproperty(first(s), :meanAssetsToMeanLaborIncome)
# hasratio || error("""
#     these sweep files predate the ratio fields and store LEVELS only.
#     `meanLaborIncome` was not recorded, so the ratio cannot be recovered --
#     re-run the sweep with the current sweep_mu2.jl.""")

# mu2       = vcat((x.mu2 for x in s)...)
# W         = vcat((x.overallValueFunction for x in s)...)
# AmeanToY  = vcat((x.meanAssetsToMeanLaborIncome   for x in s)...)
# AmedToY   = vcat((x.medianAssetsToMeanLaborIncome for x in s)...)
# Ymean     = vcat((x.meanLaborIncome for x in s)...)
# # Entry-age cross-section, if the sweep recorded it. Same denominator as the
# # window ratios above, so the four are directly comparable.
# haslo     = hasproperty(first(s), :meanAssetsAtStatsAgeLoToMeanLaborIncome)
# AmeanLo   = haslo ? vcat((x.meanAssetsAtStatsAgeLoToMeanLaborIncome   for x in s)...) : fill(NaN, length(mu2))
# AmedLo    = haslo ? vcat((x.medianAssetsAtStatsAgeLoToMeanLaborIncome for x in s)...) : fill(NaN, length(mu2))
# # Real age the window opens at, read from the settings the run actually used.
# st        = first(s).settings
# loage     = (haskey(st, :age0_real) && haskey(st, :stats_age_lo)) ?
#             st.age0_real + st.stats_age_lo - 1 : 0
# Amean     = vcat((x.meanAssets   for x in s)...)   # levels, kept for reference
# Amed      = vcat((x.medianAssets for x in s)...)

# ord = sortperm(mu2)
# mu2, W, AmeanToY, AmedToY, Ymean, Amean, Amed, AmeanLo, AmedLo, conv =
#     mu2[ord], W[ord], AmeanToY[ord], AmedToY[ord], Ymean[ord],
#     Amean[ord], Amed[ord], AmeanLo[ord], AmedLo[ord], conv[ord]

# haslo || @warn "entry-age ratios not recorded in these files; the two age-$(loage) columns will read NaN"

# @printf("\n%4s %8s %13s %13s %13s %13s %11s %12s %5s\n",
#         "i", "mu2", "meanA/meanY", "medA/meanY",
#         "meanA$(loage)/Y", "medA$(loage)/Y", "meanY", "valueFn", "conv")
# for i in eachindex(mu2)
#     @printf("%4d %8.4f %13.8f %13.8f %13.8f %13.8f %11.8f %12.8f %5s\n",
#             i, mu2[i], AmeanToY[i], AmedToY[i], AmeanLo[i], AmedLo[i],
#             Ymean[i], W[i], conv[i] ? "yes" : "NO")
# end
# @printf("\ncolumns 3-6 all divide by mean labor income over model ages %s-%s (real %s-%s)\n",
#         get(st, :stats_age_lo, "?"), get(st, :stats_age_hi, "?"),
#         loage, haskey(st, :age0_real) && haskey(st, :stats_age_hi) ?
#                st.age0_real + st.stats_age_hi - 1 : "?")
# println()
