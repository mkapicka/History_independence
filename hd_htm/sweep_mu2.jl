# =============================================================================
# sweep_mu2.jl
#
# Sweep the second history-dependence root mu2 over a grid, holding mu1 = 0,
# and collect for each grid point
#
#   eq.welfare.overallValueFunction   (ex-ante welfare, kappa-averaged)
#   eq.statistics.meanAssets
#   eq.statistics.medianAssets
#
# in vectors indexed by the grid point. The default grid is 10 points on
# [0, 0.98]; mu1 = 0 throughout, so alpha loads the mu1 block onto a stock
# that is identically zero and theta0 falls with mu2 through
#
#   theta0 = 1 / ( alpha/(1 - beta*mu1) + (1-alpha)/(1 - beta*mu2) ),
#
# from theta0 = 1 at mu2 = 0 down to theta0 = 0.1118 at mu2 = 0.98 (alpha = 0.5,
# beta = 0.96). The first grid point mu2 = 0 is the history-independent limit
# and reproduces the hi solver, so it doubles as a consistency check.
#
# -----------------------------------------------------------------------------
# Cost and accuracy notes
# -----------------------------------------------------------------------------
# * COST. With mu1 = 0 the s1-grid collapses to a single point (build_s_grid
#   returns [0.0] whenever mu == 0, regardless of nS1), so the state space is
#   nS1 = 1 rather than 7 and each solve is roughly a seventh of the cost of
#   the two-stock baseline. Still budget tens of minutes for the full sweep at
#   nA = 101, nS2 = 7, labor_grid_size = 101, and run with `julia -t auto`.
#
# * S-GRID RESOLUTION DEGRADES IN mu2. The s2-grid spans
#   [mu2*m_lo/(1-mu2), mu2*m_hi/(1-mu2)], so its width grows like mu2/(1-mu2):
#   a factor 0.43 at mu2 = 0.3 but 49 at mu2 = 0.98. At fixed nS2 = 7 the
#   spacing at the top of the grid is therefore about 114 times coarser than at
#   mu2 = 0.3. Treat the high-mu2 end as indicative, not converged, and re-run
#   the last few points with a larger nS2 before reading anything off them.
#   The reported sClampedMassShare is the diagnostic for the grid *bounds*, not
#   for this spacing problem — it can be negligible while the interpolation
#   error is not.
#
# * THE LAMBDA BRACKET MUST WIDEN WITH mu2. The budget-clearing lambda falls
#   steeply in mu2 — theta0 falls, so the income scale lambda*exp(pow*m)*h^pow
#   needs a much smaller level factor to balance the budget — and it leaves the
#   HD_SETTINGS bracket [0.20, 2.50] well before the top of the grid. At
#   mu2 = 0.98 the root is around 0.007, two orders of magnitude below
#   lambdaMin. This sweep therefore overrides lambdaMin to LAMBDA_MIN_SWEEP =
#   1e-3 by default; pass lambdaMin = ... to change it. This matters because
#   the solver does NOT throw when the bracket fails: it returns the best
#   evaluated equilibrium, which sits at the bracket endpoint with a large
#   budget residual and asset statistics that are pure artifact (at mu2 = 0.98
#   with the stock bracket, lambda pins at 0.200000 with residual -0.6 and a
#   median-assets figure an order of magnitude off).
#
# * CONVERGENCE IS CHECKED, NOT ASSUMED. Because of the above, every point is
#   flagged `converged[i] = abs(govBudgetResidual[i]) <= tolGovBudget`. Read
#   only the converged points; the summary marks the others and warns.
#
# * FAILURES ARE RECORDED, NOT FATAL. Each point runs inside a try/catch; a
#   point that throws records NaN in every output vector and its message in
#   `errors`, and the sweep continues.
#
# Usage:
#   include("sweep_mu2.jl")
#   sweep = sweep_mu2()                                  # 10 points on [0, 0.98]
#   sweep = sweep_mu2(mu2_min = 0.2, mu2_max = 0.9, n_mu2 = 20)   # set the grid
#   sweep = sweep_mu2(nS2 = 15)                          # set the stock grids
#   sweep = sweep_mu2(range(0, 0.9, length = 5))         # or pass a grid directly
#   save_mu2_sweep(sweep)
#
# As a script the arguments are positional, mu2_min mu2_max n_mu2 nS1 nS2:
#   julia -t auto sweep_mu2.jl                 # defaults
#   julia -t auto sweep_mu2.jl 0.2 0.9 20
#   julia -t auto sweep_mu2.jl 0.2 0.9 20 7 15
# =============================================================================

using JLD2
using Printf

include("run_history_dependent_tax.jl")   # defines run_history_dependent_tax
                                          # and loads the solver + HD_SETTINGS

"""
    MU2_MIN, MU2_MAX, N_MU2

Defaults for the mu2 grid: `N_MU2 = 10` points on `[MU2_MIN, MU2_MAX] =
[0, 0.98]`. Override any of the three at the call site, e.g.
`sweep_mu2(mu2_min = 0.2, mu2_max = 0.9, n_mu2 = 20)`.
"""
const MU2_MIN = 0.0
const MU2_MAX = 0.98
const N_MU2   = 10

"""
    build_mu2_grid(; mu2_min = MU2_MIN, mu2_max = MU2_MAX, n_mu2 = N_MU2)

Equally spaced mu2 grid on `[mu2_min, mu2_max]` with `n_mu2` points. Both ends
must lie in [0, 1), the interval must be non-empty, and `n_mu2 = 1` returns the
single point `mu2_min` (`range` with `length = 1` requires equal endpoints, so
that case is handled separately).
"""
function build_mu2_grid(; mu2_min = MU2_MIN, mu2_max = MU2_MAX, n_mu2 = N_MU2)
    lo, hi, n = Float64(mu2_min), Float64(mu2_max), Int(n_mu2)
    n >= 1              || error("n_mu2 must be at least 1 (got $n)")
    lo <= hi            || error("need mu2_min <= mu2_max (got $lo > $hi)")
    0.0 <= lo < 1.0     || error("mu2_min must be in [0, 1) (got $lo)")
    0.0 <= hi < 1.0     || error("mu2_max must be in [0, 1) (got $hi)")
    n == 1 && return [lo]
    return collect(range(lo, hi, length = n))
end

"""
    MU2_GRID

The default grid, `build_mu2_grid()`: 10 points on [0, 0.98], step 0.108889.
"""
const MU2_GRID = build_mu2_grid()

"""
    LAMBDA_MIN_SWEEP

Lower end of the lambda bracket used by the sweep, replacing the HD_SETTINGS
value of 0.20. The budget-clearing lambda falls steeply in mu2 and reaches
about 0.007 at mu2 = 0.98, so the stock bracket does not contain the root over
the upper half of the grid.
"""
const LAMBDA_MIN_SWEEP = 1e-3

"""
    sweep_mu2(mu2_grid = nothing;
              mu2_min = MU2_MIN, mu2_max = MU2_MAX, n_mu2 = N_MU2,
              nS1 = HD_SETTINGS.nS1, nS2 = HD_SETTINGS.nS2,
              mu1 = 0.0, labor_grid_size = 101,
              lambdaMin = LAMBDA_MIN_SWEEP, full_output = false, kwargs...)

Solve the history-dependent model once per mu2 grid point and return a
NamedTuple of vectors indexed by the grid point.

The grid is set at the call site in either of two ways. Give the endpoints and
the length — `sweep_mu2(mu2_min = 0.2, mu2_max = 0.9, n_mu2 = 20)` — or pass an
explicit grid as the positional argument, `sweep_mu2(range(0, 0.9, length = 5))`
or any vector. A positional grid wins; `mu2_min`/`mu2_max`/`n_mu2` are then
ignored. The grid actually used is returned as `sweep.mu2`.

`nS1` and `nS2` set the past-income stock grids, defaulting to `HD_SETTINGS`.
Raise `nS2` to check convergence at the top of the mu2 grid, where the s2-grid
is widest and therefore coarsest at fixed `nS2`. Raising `nS1` does nothing at
`mu1 = 0`: `build_s_grid` returns the single point `[0.0]` whenever `mu == 0`,
regardless of `nS1`, so the s1 dimension is 1 and the cost is unaffected.

The result is

    (; mu2, overallValueFunction, meanAssets, medianAssets,
       theta0, lambda, govBudgetResidual, sClampedMassShare, converged,
       elapsedSeconds, errors, mu1, settings)

Each solve is `run_history_dependent_tax(; mu1 = mu1, mu2 = mu2_grid[i],
labor_grid_size = labor_grid_size, lambdaMin = lambdaMin, kwargs...)`, so any
further `kwargs` override `HD_SETTINGS` exactly as they do there. `theta0` and
`lambda` are carried along because they are what changes across the sweep:
theta0 is implied by mu2 through the promise-keeping restriction, and lambda is
re-solved for budget balance at each point, so neither is comparable across
points by assumption.

`converged[i]` is `abs(govBudgetResidual[i]) <= tolGovBudget`. It is not
redundant: when the lambda bracket contains no sign change the solver returns
the best evaluated equilibrium rather than throwing, so a non-converged point
arrives looking like a normal result. Use only the converged points.

Set `full_output = true` to let each run print its own options block and
equilibrium summary; by default that output is suppressed and only the sweep's
own progress line per point and closing table are printed.

Failed points hold NaN in every numeric vector, with the exception message in
`errors[i]` (`nothing` when the point succeeded).
"""
function sweep_mu2(mu2_grid = nothing;
                   mu2_min = MU2_MIN,
                   mu2_max = MU2_MAX,
                   n_mu2 = N_MU2,
                   nS1 = HD_SETTINGS.nS1,
                   nS2 = HD_SETTINGS.nS2,
                   mu1 = 0.0,
                   labor_grid_size = 101,
                   lambdaMin = LAMBDA_MIN_SWEEP,
                   full_output::Bool = false,
                   kwargs...)
    mu2vec = mu2_grid === nothing ?
        build_mu2_grid(; mu2_min, mu2_max, n_mu2) : collect(Float64, mu2_grid)
    n = length(mu2vec)
    n > 0                        || error("the mu2 grid is empty")
    all(0.0 .<= mu2vec .< 1.0)   || error("every mu2 must be in [0, 1)")
    nS1 >= 1                     || error("nS1 must be at least 1 (got $nS1)")
    nS2 >= 1                     || error("nS2 must be at least 1 (got $nS2)")

    overallValueFunction = fill(NaN, n)
    meanAssets           = fill(NaN, n)
    medianAssets         = fill(NaN, n)
    # RATIOS, NOT JUST LEVELS. The calibration targets mean and median assets
    # RELATIVE to mean labor income, and the denominator moves across mu2, so a
    # level series is not a rescaled ratio series. A sweep that stores only
    # levels cannot be converted after the fact -- which is exactly what
    # happened to the `hdinf` sweeps and cost a re-solve per point.
    meanLaborIncome      = fill(NaN, n)
    meanAssetsToMeanLaborIncome   = fill(NaN, n)
    medianAssetsToMeanLaborIncome = fill(NaN, n)
    # The cross-section AS IT ENTERS the calibration window, at model age
    # stats_age_lo. Same denominator as the two above.
    meanAssetsAtStatsAgeLoToMeanLaborIncome   = fill(NaN, n)
    medianAssetsAtStatsAgeLoToMeanLaborIncome = fill(NaN, n)
    # Realized hand-to-mouth mass; must equal piH at every point.
    shareHandToMouth     = fill(NaN, n)
    theta0               = fill(NaN, n)
    lambda               = fill(NaN, n)
    govBudgetResidual    = fill(NaN, n)
    sClampedMassShare    = fill(NaN, n)
    elapsedSeconds       = fill(NaN, n)
    converged            = falses(n)
    errors               = Vector{Union{Nothing, String}}(nothing, n)
    # Scalars, not per-point: the access chain is a property of the sweep.
    # Seeded from HD_SETTINGS so a sweep in which EVERY point threw still
    # records the chain it meant to run.
    pSS_used = Ref(Float64(HD_SETTINGS.pSS))
    pHH_used = Ref(Float64(HD_SETTINGS.pHH))
    piH_used = Ref(last(HistoryDependentTax.access_stationary_distribution(
                            HD_SETTINGS.pSS, HD_SETTINGS.pHH)))
    J_used    = Ref(Int(HD_SETTINGS.J))
    age0_used = Ref(Int(HD_SETTINGS.age0_real))
    lo_used   = Ref(Int(HD_SETTINGS.stats_age_lo))
    hi_used   = Ref(Int(HD_SETTINGS.stats_age_hi))

    @printf("\n=== mu2 sweep: %d points on [%.4f, %.4f], mu1 = %.4f, nS1 = %d, nS2 = %d, lambdaMin = %.1e ===\n",
            n, first(mu2vec), last(mu2vec), mu1, nS1, nS2, lambdaMin)
    if mu1 == 0.0 && nS1 > 1
        @printf("note: mu1 = 0 collapses the s1 grid to one point; nS1 = %d has no effect.\n",
                nS1)
    end
    @printf("%4s  %8s  %8s  %10s  %12s  %12s  %10s  %8s\n",
            "i", "mu2", "theta0", "lambda", "valueFn", "meanAssets",
            "medianAss", "sec")

    for i in 1:n
        t0 = time()
        try
            # The solve prints its own options block and summary; suppress that
            # unless the caller asked for it, so the sweep's table stays legible.
            result = if full_output
                run_history_dependent_tax(; mu1 = mu1, mu2 = mu2vec[i],
                                          nS1 = nS1, nS2 = nS2,
                                          labor_grid_size = labor_grid_size,
                                          lambdaMin = lambdaMin, kwargs...)
            else
                redirect_stdout(devnull) do
                    run_history_dependent_tax(; mu1 = mu1, mu2 = mu2vec[i],
                                              nS1 = nS1, nS2 = nS2,
                                              labor_grid_size = labor_grid_size,
                                              lambdaMin = lambdaMin, kwargs...)
                end
            end

            eq = result.eq
            overallValueFunction[i] = eq.welfare.overallValueFunction
            meanAssets[i]           = eq.statistics.meanAssets
            medianAssets[i]         = eq.statistics.medianAssets
            meanLaborIncome[i]      = eq.statistics.meanLaborIncome
            meanAssetsToMeanLaborIncome[i]   =
                eq.statistics.meanAssetsToMeanLaborIncome
            medianAssetsToMeanLaborIncome[i] =
                eq.statistics.medianAssetsToMeanLaborIncome
            meanAssetsAtStatsAgeLoToMeanLaborIncome[i] =
                eq.statistics.meanAssetsAtStatsAgeLoToMeanLaborIncome
            medianAssetsAtStatsAgeLoToMeanLaborIncome[i] =
                eq.statistics.medianAssetsAtStatsAgeLoToMeanLaborIncome
            shareHandToMouth[i]     = eq.statistics.shareHandToMouth
            # The access chain in force, read off the params the solve actually
            # used, so an override is recorded rather than the default.
            pSS_used[]              = result.params.pSS
            pHH_used[]              = result.params.pHH
            piH_used[]              = result.params.piH
            J_used[]                = result.params.J
            age0_used[]             = result.params.age0_real
            lo_used[]               = result.params.stats_age_lo
            hi_used[]               = result.params.stats_age_hi
            theta0[i]               = result.params.theta0
            lambda[i]               = eq.lambda
            govBudgetResidual[i]    = eq.govBudgetResidual
            sClampedMassShare[i]    = eq.sClampedMassShare
            converged[i]            = abs(eq.govBudgetResidual) <=
                                      result.params.tolGovBudget
        catch err
            errors[i] = sprint(showerror, err)
        end
        elapsedSeconds[i] = time() - t0

        if errors[i] === nothing
            @printf("%4d  %8.4f  %8.4f  %10.6f  %12.8f  %12.8f  %10.8f  %8.1f%s\n",
                    i, mu2vec[i], theta0[i], lambda[i], overallValueFunction[i],
                    meanAssets[i], medianAssets[i], elapsedSeconds[i],
                    converged[i] ? "" : "   NOT CONVERGED")
        else
            @printf("%4d  %8.4f  %8s  %10s  %12s  %12s  %10s  %8.1f   FAILED: %s\n",
                    i, mu2vec[i], "-", "-", "-", "-", "-", elapsedSeconds[i],
                    first(errors[i], 60))
        end
        flush(stdout)
    end

    sweep = (; mu2 = mu2vec,
             overallValueFunction, meanAssets, medianAssets,
             meanLaborIncome,
             meanAssetsToMeanLaborIncome, medianAssetsToMeanLaborIncome,
             meanAssetsAtStatsAgeLoToMeanLaborIncome,
             medianAssetsAtStatsAgeLoToMeanLaborIncome,
             shareHandToMouth,
             theta0, lambda, govBudgetResidual, sClampedMassShare, converged,
             elapsedSeconds, errors,
             mu1 = Float64(mu1),
             pSS = pSS_used[], pHH = pHH_used[], piH = piH_used[],
             # pSS/pHH are forced into `settings` as well as carried as
             # scalars: `settings` is what a recovery script would splat back
             # to rebuild the run, and a chain that came from HD_SETTINGS
             # rather than from `kwargs` would otherwise be absent there.
             # The calibration window is forced in for the same reason as the
             # access chain: `join_results.jl` labels its columns with it, and
             # a window that came from HD_SETTINGS rather than from `kwargs`
             # would otherwise print as "model ages ?-? (real ?-?)".
             settings = (; nS1, nS2, labor_grid_size, lambdaMin, kwargs...,
                           pSS = pSS_used[], pHH = pHH_used[],
                           J = J_used[], age0_real = age0_used[],
                           stats_age_lo = lo_used[], stats_age_hi = hi_used[]))

    print_mu2_sweep_summary(sweep)
    return sweep
end

"""
    sweep_nS(sweep, key)

`sweep.settings[key]` as a display string, or `"?"` for a sweep saved before
`nS1`/`nS2` were recorded. Keeps the summary and the filename working on
results loaded from disk rather than throwing a `FieldError` on them.
"""
sweep_nS(sweep, key::Symbol) = string(get(sweep.settings, key, "?"))

"""
    print_mu2_sweep_summary(sweep)

Print the three requested series against mu2, plus the diagnostics that decide
whether a point is trustworthy: the government-budget residual and the share of
mass whose s' was clamped onto the s-grid.
"""
function print_mu2_sweep_summary(sweep)
    n = length(sweep.mu2)
    nfail = count(!isnothing, sweep.errors)
    nconv = count(sweep.converged)

    @printf("\n=== mu2 sweep summary (mu1 = %.4f, nS1 = %s, nS2 = %s, %d points, %d converged, %d threw) ===\n",
            sweep.mu1, sweep_nS(sweep, :nS1), sweep_nS(sweep, :nS2),
            n, nconv, nfail)
    if haskey(sweep, :pSS)
        @printf("access chain: pSS = %.6f, pHH = %.6f  ->  piH = %.6f%s\n",
                sweep.pSS, sweep.pHH, sweep.piH,
                sweep.piH == 0.0 ? "   [NO HtM AGENTS: this is the hd sweep]" : "")
    end
    @printf("%4s  %8s  %8s  %14s  %14s  %14s  %11s  %11s  %5s\n",
            "i", "mu2", "theta0", "valueFn", "meanAssets", "medianAssets",
            "govResid", "sClamped", "conv")
    for i in 1:n
        if sweep.errors[i] === nothing
            @printf("%4d  %8.4f  %8.4f  %14.8f  %14.8f  %14.8f  %11.2e  %11.2e  %5s\n",
                    i, sweep.mu2[i], sweep.theta0[i],
                    sweep.overallValueFunction[i], sweep.meanAssets[i],
                    sweep.medianAssets[i], sweep.govBudgetResidual[i],
                    sweep.sClampedMassShare[i], sweep.converged[i] ? "yes" : "NO")
        else
            @printf("%4d  %8.4f  %8s  %14s  %14s  %14s  %11s  %11s  %5s\n",
                    i, sweep.mu2[i], "-", "FAILED", "-", "-", "-", "-", "NO")
        end
    end
    @printf("total solve time = %.1f seconds\n", sum(sweep.elapsedSeconds))

    unconverged = findall(i -> sweep.errors[i] === nothing && !sweep.converged[i], 1:n)
    if !isempty(unconverged)
        @printf("WARNING: the lambda solve did not clear the government budget at points %s.\n",
                string(unconverged))
        @printf("         Those rows are bracket-endpoint fallbacks, not equilibria: discard\n")
        @printf("         them or re-run with a lower lambdaMin (currently %.1e).\n",
                sweep.settings.lambdaMin)
    end

    material = findall(x -> !isnan(x) && x > 1e-6, sweep.sClampedMassShare)
    if !isempty(material)
        @printf("WARNING: s' clamped mass share exceeds 1e-6 at points %s; ",
                string(material))
        @printf("widen the s-grids or raise nS2 there.\n")
    end
    for i in 1:n
        sweep.errors[i] === nothing && continue
        @printf("point %d (mu2 = %.4f) failed: %s\n", i, sweep.mu2[i],
                sweep.errors[i])
    end
    println()
end

"""
    save_mu2_sweep(sweep; dir = joinpath(@__DIR__, "results"))

Save a `sweep_mu2()` result to `dir` under a filename built from the grid, e.g.
`sweep_mu2_n=10_mu2=0.000-0.980_mu1=0.000_nS1=7_nS2=7.jld2`. The stock grid
sizes are part of the name so an nS2 convergence re-run does not overwrite the
baseline. Overwrites an existing file of the same name. Returns the saved path.
Reload with `load_mu2_sweep(path)`.
"""
function save_mu2_sweep(sweep; dir = joinpath(@__DIR__, "results"))
    mkpath(dir)
    # pSS/pHH are in the name: two sweeps differing only in the access chain
    # are different models and must not overwrite each other.
    name = @sprintf("sweep_mu2_n=%d_mu2=%.3f-%.3f_mu1=%.3f_nS1=%s_nS2=%s_pSS=%.4f_pHH=%.4f.jld2",
                    length(sweep.mu2), first(sweep.mu2), last(sweep.mu2),
                    sweep.mu1, sweep_nS(sweep, :nS1), sweep_nS(sweep, :nS2),
                    get(sweep, :pSS, NaN), get(sweep, :pHH, NaN))
    path = joinpath(dir, name)
    jldsave(path; sweep = sweep)
    return path
end

"""
    load_mu2_sweep(path)

Load a sweep saved by `save_mu2_sweep`. Returns the sweep NamedTuple.
"""
load_mu2_sweep(path::AbstractString) = JLD2.load(path)["sweep"]

"""
    merge_mu2_sweeps(dir = joinpath(@__DIR__, "results"); match = "", prefix = "sweep_mu2_n=1_")

Combine the one-point files written by an array job into a single sweep, sorted
by mu2, with the same fields as a `sweep_mu2` result.

Each array task writes its own file because the filename encodes mu2, so a
20-task job leaves 20 files rather than one sweep. `match` filters those by a
substring of the filename -- pass `"nS2=151"` to pick one resolution out of a
`results/` directory holding several runs, which is the usual case after a few
rounds of experimenting.

Prints a warning, rather than failing, when the parts disagree on mu1 or on the
solver settings: that combination is almost always a mistake (files from two
different configurations), but it is occasionally deliberate.
"""
function merge_mu2_sweeps(dir = joinpath(@__DIR__, "results");
                          match::AbstractString = "",
                          prefix::AbstractString = "sweep_mu2_n=1_")
    isdir(dir) || error("no such directory: $dir")
    files = sort(filter(readdir(dir)) do f
        startswith(f, prefix) && endswith(f, ".jld2") && occursin(match, f)
    end)
    isempty(files) &&
        error("no files matching \"$prefix*$match*.jld2\" in $dir")
    parts = [load_mu2_sweep(joinpath(dir, f)) for f in files]

    length(unique(p.mu1 for p in parts)) == 1 ||
        @warn "parts disagree on mu1; merging anyway" mu1s = unique(p.mu1 for p in parts)
    length(unique(string(p.settings) for p in parts)) == 1 ||
        @warn "parts disagree on solver settings; merging anyway (use `match` to select one run)"

    cat_field(f) = reduce(vcat, (getproperty(p, f) for p in parts))
    vec_fields = (:mu2, :overallValueFunction, :meanAssets, :medianAssets, :theta0,
                  :lambda, :govBudgetResidual, :sClampedMassShare, :converged,
                  :elapsedSeconds, :errors)
    merged = NamedTuple{vec_fields}(cat_field.(vec_fields))

    order = sortperm(merged.mu2)
    sorted = NamedTuple{vec_fields}(getproperty(merged, f)[order] for f in vec_fields)

    return (; sorted..., mu1 = first(parts).mu1, settings = first(parts).settings,
            nFiles = length(files), files = files[order])
end

# Script entry point. `julia -t auto sweep_mu2.jl` runs the defaults;
# `julia -t auto sweep_mu2.jl <mu2_min> <mu2_max> <n_mu2> <nS1> <nS2>` sets
# them, and a leading subset works too (one argument sets mu2_min only, and so
# on). From the REPL, include this file and call `sweep = sweep_mu2()`.
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    # Two argument styles, chosen by whether any token contains '='.
    #
    #   key=value  any keyword sweep_mu2 accepts, same as the REPL call:
    #                julia sweep_mu2.jl mu2_min=0.96 n_mu2=4 alpha=0.5 nS2=101
    #   positional mu2_min mu2_max n_mu2 nS1 nS2, a leading subset allowed:
    #                julia sweep_mu2.jl 0.98 0.98 1 7 21
    #
    # The positional form is kept because existing batch scripts use it.
    sweep = if any(contains('='), ARGS)
        sweep_mu2(; parse_cli_kwargs(ARGS)...)
    else
        length(ARGS) <= 5 ||
            error("usage: julia sweep_mu2.jl [mu2_min [mu2_max [n_mu2 [nS1 [nS2]]]]]\n" *
                  "   or: julia sweep_mu2.jl key=value ...")
        sweep_mu2(;
            mu2_min = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : MU2_MIN,
            mu2_max = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : MU2_MAX,
            n_mu2   = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : N_MU2,
            nS1     = length(ARGS) >= 4 ? parse(Int, ARGS[4])     : HD_SETTINGS.nS1,
            nS2     = length(ARGS) >= 5 ? parse(Int, ARGS[5])     : HD_SETTINGS.nS2)
    end
    @printf("saved to %s\n", save_mu2_sweep(sweep))
end
