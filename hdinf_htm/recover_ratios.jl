# =============================================================================
# recover_ratios.jl
#
# Recovers the ratio statistics from sweep files that stored levels only.
# For the infinite-horizon history-dependent tax model with hand-to-mouth agents.
#
# Marek Kapicka, 2026
# =============================================================================

using JLD2, Printf

# =============================================================================
# recover_ratios.jl -- add the ratio fields to sweep files that stored LEVELS.
#
# WHY THIS EXISTS. Sweeps written before `meanLaborIncome` was recorded carry
# `meanAssets` and `medianAssets` as levels only, and `join_results.jl` refuses
# them because the calibration targets RATIOS to mean labor income and the
# denominator moves across mu2. The obvious fix -- re-run the sweep -- costs
# what the sweep cost: 28.25 hours for one point at the production grids.
#
# WHAT THIS DOES INSTEAD. Each sweep file records `lambda`, and `settings`
# records every parameter the run overrode. So the equilibrium can be rebuilt
# with ONE solve at the stored lambda, skipping the lambda root-find that the
# original sweep paid for over and over. The saving is the number of lambda
# evaluations that point used.
#
# THE CROSS-CHECK IS THE POINT. `meanAssets` and `medianAssets` ARE stored, so
# the rebuilt equilibrium must reproduce them. If it does, the recovered
# denominator belongs to the same equilibrium and the ratios are exact. If it
# does not, some parameter that mattered is NOT in `settings` -- it fell back to
# the current HD_SETTINGS, which has moved -- and that point must be re-run. The
# check is per point and printed, never assumed.
#
# maxAge is the field to watch: it is NOT in the stored `settings` tuple, so it
# comes from whatever HD_SETTINGS says now.
#
# IN THIS DIRECTORY, so is the access chain for foreign files. `sweep_mu2.jl`
# here forces pSS/pHH into `settings`, so anything this directory wrote rebuilds
# exactly. A sweep written by `hdinf` records no chain, falls back to
# HD_SETTINGS -- which defaults to piH = 1/3 -- and is then a DIFFERENT model.
# The level cross-check catches that, but do not recover `hdinf` files here in
# the first place: use hdinf/recover_ratios.jl.
#
# USAGE
#   julia --project=. recover_ratios.jl [SRC_DIR] [LIMIT]
#
#   julia --project=. recover_ratios.jl results 1     # one point first, ALWAYS
#   julia --project=. recover_ratios.jl results       # then the rest
#
# Recovered files are written to SRC_DIR * "_ratios"; nothing is overwritten.
# `join_results.jl` then reads that directory.
# =============================================================================

include(joinpath(@__DIR__, "run_history_dependent_tax.jl"))
# `government_residual_at_lambda` is internal to the solver module and not
# exported, so it is reached by qualified name. That is deliberate on the
# solver's side: this script is the only caller that wants an equilibrium at a
# GIVEN lambda rather than at the budget-balancing one.
const HDT = HistoryDependentTaxInfinite

const SRC   = length(ARGS) >= 1 ? ARGS[1] : "results"
const LIMIT = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : typemax(Int)
# Relative agreement demanded of the re-solved levels. The solve is
# deterministic, so a genuine match is at round-off; 1e-8 is loose enough to
# survive a thread-count change in the reduction order and tight enough that a
# different equilibrium cannot slip through.
const LEVEL_TOL = 1e-8

reldiff(a, b) = abs(b) > 0 ? abs(a - b) / abs(b) : abs(a - b)

function recover_file(path::AbstractString, outdir::AbstractString)
    s = JLD2.load(path)["sweep"]
    name = basename(path)

    if hasproperty(s, :meanAssetsToMeanLaborIncome)
        @printf("%-58s already has ratios; copied\n", name)
        mkpath(outdir); jldsave(joinpath(outdir, name); sweep = s)
        return true
    end

    n = length(s.mu2)
    meanLaborIncome = fill(NaN, n)
    r_meanA = fill(NaN, n); r_medA = fill(NaN, n)
    r_meanLo = fill(NaN, n); r_medLo = fill(NaN, n)
    ok = true

    for i in 1:n
        s.errors[i] === nothing || (@printf("%-58s point %d threw in the original sweep; skipped\n", name, i); ok = false; continue)
        # Everything the original run overrode, replayed. `settings` also
        # carries nS1/nS2/labor_grid_size/lambdaMin, so it is passed whole.
        p = make_history_dependent_params(; s.settings..., mu1 = s.mu1,
                                          mu2 = s.mu2[i], verbose = false)
        t0 = time()
        _, eq = redirect_stdout(devnull) do
            HDT.government_residual_at_lambda(s.lambda[i], p)
        end
        st = eq.statistics
        dmean = reldiff(st.meanAssets, s.meanAssets[i])
        dmed  = reldiff(st.medianAssets, s.medianAssets[i])
        pass  = dmean <= LEVEL_TOL && dmed <= LEVEL_TOL
        ok &= pass

        meanLaborIncome[i] = st.meanLaborIncome
        r_meanA[i] = st.meanAssetsToMeanLaborIncome
        r_medA[i]  = st.medianAssetsToMeanLaborIncome
        r_meanLo[i] = hasproperty(st, :meanAssetsAtStatsAgeLoToMeanLaborIncome) ?
                      st.meanAssetsAtStatsAgeLoToMeanLaborIncome : NaN
        r_medLo[i]  = hasproperty(st, :medianAssetsAtStatsAgeLoToMeanLaborIncome) ?
                      st.medianAssetsAtStatsAgeLoToMeanLaborIncome : NaN

        @printf("%-58s mu2=%.4f  meanY=%.8f  meanA/Y=%.8f  levels %s (mean %.1e, median %.1e)  %.0fs\n",
                name, s.mu2[i], st.meanLaborIncome, st.meanAssetsToMeanLaborIncome,
                pass ? "MATCH" : "*** MISMATCH ***", dmean, dmed, time() - t0)
        flush(stdout)
    end

    out = merge(s, (; meanLaborIncome,
                      meanAssetsToMeanLaborIncome = r_meanA,
                      medianAssetsToMeanLaborIncome = r_medA,
                      meanAssetsAtStatsAgeLoToMeanLaborIncome = r_meanLo,
                      medianAssetsAtStatsAgeLoToMeanLaborIncome = r_medLo,
                      recoveredFromLevels = true))
    mkpath(outdir); jldsave(joinpath(outdir, name); sweep = out)
    return ok
end

function main()
    files = sort(filter(f -> endswith(f, ".jld2"), readdir(SRC)))
    isempty(files) && error("no .jld2 files in $(SRC)")
    outdir = SRC * "_ratios"
    todo = min(LIMIT, length(files))
    @printf("recovering %d of %d file(s) from %s -> %s\n\n", todo, length(files), SRC, outdir)
    allok = true
    for f in files[1:todo]
        allok &= recover_file(joinpath(SRC, f), outdir)
    end
    println()
    if allok
        @printf("ALL LEVELS MATCHED. Point join_results.jl at \"%s\".\n", outdir)
    else
        println("AT LEAST ONE POINT MISMATCHED: the rebuilt parameters are not the ones")
        println("the sweep ran with. Check maxAge (not stored in `settings`) against the")
        println("HD_SETTINGS in force when the sweep ran; those points need a re-run.")
    end
end

main()
