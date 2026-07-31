# =============================================================================
# calibrate_history_independent_tax.jl
#
# Calibrate the three financial / borrowing-limit parameters
#
#     qSav   (gross savings price,  a' >= 0)
#     qBorr  (gross borrowing price, a' <  0)
#     bbar   (borrowing-limit scale, bbar <= 0)
#
# so that the stationary cross-section produced by
# `solve_history_independent_tax` matches three data moments:
#
#     (i)   mean assets / mean labor income             = 0.588
#           (or median / mean labor income = 0.043 when asset_moment = :median)
#     (ii)  true borrowing limit / mean labor income     = 0.180
#     (iii) share of households with negative liquid      = 0.260
#           assets
#
# These map onto the statistics returned by `finalize_statistics` as
#
#     (i)   eq.statistics.meanAssetsToMeanLaborIncome   (default), or
#           eq.statistics.medianAssetsToMeanLaborIncome (asset_moment = :median)
#     (ii)  eq.statistics.meanBorrowingLimitToMeanLaborIncome, which averages
#           -bbar*exp(kappa+rho*z) over ages j = 0,...,J-1; the terminal age is
#           excluded because a' >= 0 is imposed there and no limit is defined
#     (iii) eq.statistics.shareNegativeLiquidAssets
#
# The `solve_*` file is used unmodified; this file only wraps it.
#
# -----------------------------------------------------------------------------
# Identification logic
# -----------------------------------------------------------------------------
# The mapping (qSav, qBorr, bbar) -> (moment_i, moment_ii, moment_iii) is
# coupled, but it has a strong near-triangular structure that we exploit:
#
#   * bbar  is the *only* parameter entering the "true" borrowing limit
#     numerator  -bbar * E[exp(kappa + rho*z)]  (see `true_borrowing_limit`
#     in simulate_kappa!, averaged over ages j = 0,...,J-1).  It therefore
#     pins down moment (ii) almost mechanically, with only a second-order
#     feedback through mean labor income.  -> bbar solves (ii).
#
#   * qSav  governs the return to saving and hence the right tail / median of
#     the asset distribution.  -> qSav solves (i).
#
#   * qBorr governs the cost of borrowing and hence how many households choose
#     a' < 0.  -> qBorr solves (iii).
#
# We solve this with an outer block-Gauss-Seidel / nested-bisection scheme:
# each instrument is moved by a 1-D bracketed root finder (Roots.Brent, already
# a dependency of the solver) holding the others fixed, and we sweep the three
# blocks until the joint residual is below tolerance.  Each residual evaluation
# rebuilds HIParams (because the asset grid's lower bound depends on bbar) and
# calls the unmodified solver with verbose output suppressed.
#
# References for this calibration strategy in heterogeneous-agent models:
#   * Kaplan, Moll & Violante (2018, AER) -- two-asset HANK; liquid-asset
#     targets (median liquid wealth, share of hand-to-mouth / negative liquid
#     positions) calibrated to SCF.
#   * Guvenen, Karahan, Ozkan & Song (2021, Ecta) -- moment-matching of
#     earnings-driven wealth statistics.
#   * Standard SMM/just-identified GMM logic: 3 instruments, 3 moments.
# =============================================================================

using Printf

include("solve_history_independent_tax.jl")
include("model_settings.jl")

# -----------------------------------------------------------------------------
# Calibration targets and configuration
# -----------------------------------------------------------------------------
"""
    CalibrationTargets

Data moments the calibration matches. Field names mirror the moment they map to
in `eq.statistics`.

`asset_moment` selects which asset statistic block (i) targets via `qSav`:
`:mean` (default) matches `meanAssetsToMeanLaborIncome`; `:median` matches
`medianAssetsToMeanLaborIncome` instead. Only the selected one is targeted; the
other is reported but not matched.
"""
Base.@kwdef struct CalibrationTargets
    medianAssetsToMeanLaborIncome::Float64       = 0.043   # (i), asset_moment = :median
    meanAssetsToMeanLaborIncome::Float64         = 0.588   # (i), asset_moment = :mean
    trueBorrowingLimitToMeanLaborIncome::Float64 = 0.180   # (ii)
    shareNegativeLiquidAssets::Float64           = 0.260   # (iii)
    asset_moment::Symbol                         = :mean   # :median or :mean
end

# Targeted asset ratio (moment i) under the selected asset_moment.
asset_ratio(m, t::CalibrationTargets) =
    t.asset_moment === :median ? m.medianAssetsToMeanLaborIncome :
                                 m.meanAssetsToMeanLaborIncome
asset_target(t::CalibrationTargets) =
    t.asset_moment === :median ? t.medianAssetsToMeanLaborIncome :
                                 t.meanAssetsToMeanLaborIncome
asset_label(t::CalibrationTargets) =
    t.asset_moment === :median ? "median assets / mean labor income" :
                                 "mean assets / mean labor income"
asset_label_short(t::CalibrationTargets) =
    t.asset_moment === :median ? "median/LI" : "mean/LI"

"""
    CalibrationConfig

Solver controls for the calibration loop.

Brackets are on the *economically meaningful* ranges of the instruments:
  * `qSav`  in (0, qSav_max]; higher q => cheaper saving => more saving.
  * `qBorr` in [qBorr_min, qBorr_max]; higher q => cheaper borrowing => more
    households with a' < 0.  We keep qBorr <= qSav is NOT imposed by the model,
    but the default bracket allows a wide spread.
  * `bbar`  in [bbar_min, 0); more negative bbar => looser limit => larger
    true-borrowing-limit moment.

`outer_max_sweeps` block-coordinate sweeps, each refining one instrument with a
1-D Brent root find of `inner_maxevals` evaluations and `inner_xtol` bracket
tolerance. `moment_tol` is the stopping criterion on the max absolute moment
residual.

Efficiency controls (all reduce the number of full model solves, which
dominate run time):
  * `lambda_warm_start` / `lambda_warm_width`: solve each inner model with the
    government-budget lambda bracketed within `+/- lambda_warm_width` (relative)
    of the previously found lambda, falling back to the full default bracket if
    that narrow solve does not converge.
  * `shrink_brackets` / `bracket_shrink`: from sweep 2 onward, root-find each
    instrument on a window of width `bracket_shrink * (full range)` centered on
    its current value, falling back to the full bracket if no sign change is
    found there.
  * `final_resolve`: if `true` (default), re-solve once more at the calibrated
    point with the caller's verbosity, printing the full solver log
    ("=== History-independent tax finite-horizon solver ===") after the
    calibration; if `false`, reuse the cached equilibrium (one solve cheaper,
    no solver log).
"""
Base.@kwdef struct CalibrationConfig
    # Instrument brackets
    qSav_min::Float64  = 0.900
    qSav_max::Float64  = 1.040
    qBorr_min::Float64 = 0.700
    qBorr_max::Float64 = 1.040
    bbar_min::Float64  = -0.80
    bbar_max::Float64  = -0.01

    # Inner 1-D root finder (Roots.Brent)
    inner_xtol::Float64 = 1e-5
    inner_maxevals::Int = 40

    # Outer block-coordinate loop
    outer_max_sweeps::Int = 12
    moment_tol::Float64   = 5e-4

    # Efficiency controls
    lambda_warm_start::Bool  = true
    lambda_warm_width::Float64 = 0.10
    shrink_brackets::Bool    = true
    bracket_shrink::Float64  = 0.15
    final_resolve::Bool      = true

    # Pass-through solver verbosity (the *inner* model solves are silenced
    # regardless; this only controls the calibration's own logging).
    verbose::Bool = true
end

# -----------------------------------------------------------------------------
# Model evaluation at a candidate (qSav, qBorr, bbar)
# -----------------------------------------------------------------------------
"""
    moments_from(eq)

The four calibration statistics of an equilibrium, named as the targets are.
"""
moments_from(eq) = (;
    medianAssetsToMeanLaborIncome       = eq.statistics.medianAssetsToMeanLaborIncome,
    meanAssetsToMeanLaborIncome         = eq.statistics.meanAssetsToMeanLaborIncome,
    trueBorrowingLimitToMeanLaborIncome = eq.statistics.meanBorrowingLimitToMeanLaborIncome,
    shareNegativeLiquidAssets           = eq.statistics.shareNegativeLiquidAssets,
)

# Signed residual moment - target for each block. Block (i) uses whichever
# asset ratio (median or mean) `targets.asset_moment` selects.
resid_i(m, t)   = asset_ratio(m, t)                     - asset_target(t)
resid_ii(m, t)  = m.trueBorrowingLimitToMeanLaborIncome - t.trueBorrowingLimitToMeanLaborIncome
resid_iii(m, t) = m.shareNegativeLiquidAssets           - t.shareNegativeLiquidAssets

max_abs_resid(m, t) = max(abs(resid_i(m, t)),
                          abs(resid_ii(m, t)),
                          abs(resid_iii(m, t)))

# -----------------------------------------------------------------------------
# A robust 1-D bracketed solver wrapper
# -----------------------------------------------------------------------------
# `f` is monotone in the relevant region but may be flat (step-like) because
# the asset choice lives on a discrete grid. We therefore:
#   1. probe the supplied bracket endpoints,
#   2. if they do not straddle zero, expand/scan the interval,
#   3. fall back to returning the endpoint with the smallest |residual| when no
#      sign change exists (the moment is then as close as the grid allows).
"""
    solve_scalar(f, lo, hi; xtol, maxevals, nscan)

Find x in [lo, hi] with f(x) ~ 0. Returns `(x, fx, bracketed::Bool)`.
"""
function solve_scalar(f, lo::Float64, hi::Float64;
                      xtol::Float64 = 1e-5, maxevals::Int = 40,
                      nscan::Int = 9)
    flo = f(lo)
    fhi = f(hi)

    if abs(flo) <= xtol
        return lo, flo, true
    elseif abs(fhi) <= xtol
        return hi, fhi, true
    end

    a, fa, b, fb = lo, flo, hi, fhi
    if sign(flo) == sign(fhi)
        # Scan the interior for a sign change.
        xs = collect(range(lo, hi, length = nscan))
        fs = similar(xs)
        fs[1] = flo
        fs[end] = fhi
        for k in 2:(nscan - 1)
            fs[k] = f(xs[k])
        end
        bracket = nothing
        for k in 1:(nscan - 1)
            if isfinite(fs[k]) && isfinite(fs[k + 1]) &&
               sign(fs[k]) != sign(fs[k + 1])
                bracket = (k, k + 1)
                break
            end
        end
        if bracket === nothing
            # No sign change: return the closest scanned point.
            idx = argmin(abs.(fs))
            return xs[idx], fs[idx], false
        end
        klo, khi = bracket
        a, fa, b, fb = xs[klo], fs[klo], xs[khi], fs[khi]
    end

    x = Roots.find_zero(f, (a, b), Roots.Brent();
                        xatol = xtol, maxevals = max(maxevals, 12))
    return x, f(x), true
end

# -----------------------------------------------------------------------------
# Main calibration routine
# -----------------------------------------------------------------------------
"""
    calibrate_history_independent_tax(; targets, config, base_kwargs...)

Calibrate `(qSav, qBorr, bbar)` to `targets`. Returns a NamedTuple

    (; qSav, qBorr, bbar, eq, moments, residuals, converged, sweeps,
       nSolves, elapsedSeconds, targets, params)

where `eq` is the equilibrium at the calibrated parameters (by default
re-solved once with the caller's verbosity, printing the full solver log
after the calibration; set `config.final_resolve = false` to reuse the
cached equilibrium instead), `params` are the calibrated `HIParams`,
`moments`/`residuals` report the achieved fit, and `nSolves` counts full
model solves.

`base_kwargs` are forwarded to `make_history_independent_params` for every
evaluation, so any non-calibrated setting can be overridden here. As a
convenience, any `CalibrationTargets` field name passed as a bare keyword
(e.g. `asset_moment = :median`, `meanAssetsToMeanLaborIncome = 0.6`) is
intercepted and used to build the targets instead of being forwarded to the
model; an explicitly supplied `targets` takes precedence over such keywords.
The calibrated instruments (`qSav`, `qBorr`, `bbar`) and `verbose` cannot be
passed through `base_kwargs`.
"""
function calibrate_history_independent_tax(;
        targets::Union{CalibrationTargets,Nothing} = nothing,
        config::CalibrationConfig = CalibrationConfig(),
        kwargs...)

    start_time = time()

    # Split off CalibrationTargets fields passed as bare keywords
    # (convenience: `asset_moment = :mean` instead of building the struct).
    kw = Dict{Symbol,Any}(pairs(kwargs))
    target_kw = Dict{Symbol,Any}()
    for k in collect(keys(kw))
        if k in fieldnames(CalibrationTargets)
            target_kw[k] = pop!(kw, k)
        end
    end
    if targets === nothing
        targets = CalibrationTargets(; target_kw...)
    elseif !isempty(target_kw)
        @warn "Both `targets` and target keyword(s) $(sort!(collect(keys(target_kw)))) " *
              "were supplied; using `targets` and ignoring the keywords."
    end
    base_kwargs = (; kw...)

    targets.asset_moment in (:median, :mean) ||
        error("targets.asset_moment must be :median or :mean")

    # Calibrated instruments (and inner-solver verbosity) are controlled here;
    # passing them through base_kwargs would silently conflict.
    for k in (:qSav, :qBorr, :bbar, :verbose)
        haskey(base_kwargs, k) &&
            error("`$(k)` cannot be passed via base_kwargs; it is set by the calibration")
    end

    # Initial guesses from SETTINGS (sensible center of each bracket if absent).
    qSav  = clamp(get(SETTINGS, :qSav, 0.99),  config.qSav_min,  config.qSav_max)
    qBorr = clamp(get(SETTINGS, :qBorr, 0.90), config.qBorr_min, config.qBorr_max)
    bbar  = clamp(get(SETTINGS, :bbar, -0.20), config.bbar_min,  config.bbar_max)

    # ------------------------------------------------------------------
    # Cached, lambda-warm-started model evaluation.
    #
    # Every moment evaluation is a full model solve, so we (a) memoize on
    # the instrument triple -- Brent re-probes bracket endpoints, and the
    # sweep-end / final evaluations repeat points already solved -- and
    # (b) bracket the government-budget lambda near the previously found
    # root, falling back to the full default bracket if that fails.
    # ------------------------------------------------------------------
    cache = Dict{NTuple{3,Float64},Any}()
    n_solves = Ref(0)
    lambda_warm = Ref(NaN)

    function eval_point(qS::Float64, qB::Float64, bb::Float64)
        key = (qS, qB, bb)
        haskey(cache, key) && return cache[key]

        solve_at = function (; lambda_kwargs...)
            p = make_history_independent_params(;
                base_kwargs..., qSav = qS, qBorr = qB, bbar = bb,
                verbose = false, lambda_kwargs...,
                collect_distributions = false)
            n_solves[] += 1
            return solve_history_independent_tax(p)
        end

        local eq
        if config.lambda_warm_start && isfinite(lambda_warm[])
            w = config.lambda_warm_width
            # nLambdaSearch = 5 caps the cost of the solver's internal
            # no-sign-change fallback if the warm bracket misses the root.
            eq, _ = solve_at(; lambdaMin = lambda_warm[] * (1.0 - w),
                               lambdaMax = lambda_warm[] * (1.0 + w),
                               nLambdaSearch = 5)
            if !(hasproperty(eq, :converged) && eq.converged === true)
                eq, _ = solve_at()   # retry on the full default bracket
            end
        else
            eq, _ = solve_at()
        end

        lambda_warm[] = eq.lambda
        moments = moments_from(eq)
        cache[key] = (moments, eq)
        return moments, eq
    end

    # Root-find one instrument: from sweep 2 onward, search a narrow window
    # around the current value first (the solution moves little between
    # sweeps), falling back to the full bracket if no sign change is found.
    function solve_block(f, x_now::Float64, lo_full::Float64, hi_full::Float64,
                         first_sweep::Bool)
        if !first_sweep && config.shrink_brackets
            w = max(config.bracket_shrink * (hi_full - lo_full), 10.0 * config.inner_xtol)
            lo = max(lo_full, x_now - w)
            hi = min(hi_full, x_now + w)
            x, fx, br = solve_scalar(f, lo, hi;
                                     xtol = config.inner_xtol,
                                     maxevals = config.inner_maxevals)
            br && return x, fx, br
        end
        return solve_scalar(f, lo_full, hi_full;
                            xtol = config.inner_xtol,
                            maxevals = config.inner_maxevals)
    end

    if config.verbose
        println("\n=== Calibration targets ===")
        @printf("%-40s = %.8f\n", asset_label(targets), asset_target(targets))
        @printf("true borrowing limit / mean labor income = %.8f\n",
                targets.trueBorrowingLimitToMeanLaborIncome)
        @printf("share negative liquid assets             = %.8f\n",
                targets.shareNegativeLiquidAssets)
        println()
        println("=== Calibration search ===")
        @printf("start:   qSav=%.6f qBorr=%.6f bbar=%.6f\n", qSav, qBorr, bbar)
        println()
        @printf("%4s   %-10s  %-10s  %-11s %-10s  %-10s  %-9s %s\n",
                "eval", "qSav", "qBorr", "bbar",
                asset_label_short(targets), "trueBL/LI", "neg share", "max")
        flush(stdout)
    end

    print_row = function (label, m, gap, note::String = "")
        @printf("%4s  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f %.0e%s\n",
                label, qSav, qBorr, bbar,
                asset_ratio(m, targets),
                m.trueBorrowingLimitToMeanLaborIncome,
                m.shareNegativeLiquidAssets,
                gap, note)
        flush(stdout)
    end

    # Baseline at the starting point (row 1); exit early if already on target.
    moments, _ = eval_point(qSav, qBorr, bbar)
    converged = max_abs_resid(moments, targets) <= config.moment_tol
    sweep = 0
    config.verbose && print_row("1", moments, max_abs_resid(moments, targets))

    while !converged && sweep < config.outer_max_sweeps
        sweep += 1
        first_sweep = sweep == 1

        # --- Block (ii): bbar -> true borrowing limit / mean labor income ---
        # More negative bbar raises the numerator, so the residual is
        # *increasing* in (-bbar), i.e. decreasing in bbar.
        f_ii = b -> resid_ii(first(eval_point(qSav, qBorr, b)), targets)
        bbar, _, br_ii = solve_block(f_ii, bbar, config.bbar_min, config.bbar_max,
                                     first_sweep)

        # --- Block (i): qSav -> asset ratio (median or mean per asset_moment) ---
        # Higher qSav => cheaper saving => higher (median and mean) assets =>
        # residual increasing in qSav.
        f_i = q -> resid_i(first(eval_point(q, qBorr, bbar)), targets)
        qSav, _, br_i = solve_block(f_i, qSav, config.qSav_min, config.qSav_max,
                                    first_sweep)

        # --- Block (iii): qBorr -> share with negative liquid assets ---
        # Higher qBorr => cheaper borrowing => larger negative-asset share =>
        # residual increasing in qBorr.
        f_iii = q -> resid_iii(first(eval_point(qSav, q, bbar)), targets)
        qBorr, _, br_iii = solve_block(f_iii, qBorr, config.qBorr_min, config.qBorr_max,
                                       first_sweep)

        # Cache hit: the qBorr block just evaluated this exact triple.
        moments, _ = eval_point(qSav, qBorr, bbar)
        gap = max_abs_resid(moments, targets)

        config.verbose && print_row(string(sweep + 1), moments, gap,
                                 (br_i && br_ii && br_iii) ? "" : "  [unbracketed]")

        converged = gap <= config.moment_tol
    end

    # Equilibrium at the calibrated point: reuse the cached solve unless the
    # caller asked for a fresh, fully-logged final solve.
    p_final = make_history_independent_params(;
        base_kwargs..., qSav = qSav, qBorr = qBorr, bbar = bbar)
    if config.final_resolve
        eq, _ = solve_history_independent_tax(p_final)
        n_solves[] += 1
    else
        _, eq = eval_point(qSav, qBorr, bbar)   # cache hit
    end
    moments_final = moments_from(eq)
    residuals = (;
        assetsToMeanLaborIncome             = resid_i(moments_final, targets),
        trueBorrowingLimitToMeanLaborIncome = resid_ii(moments_final, targets),
        shareNegativeLiquidAssets           = resid_iii(moments_final, targets),
    )

    result = (;
        qSav = qSav, qBorr = qBorr, bbar = bbar,
        eq = eq, moments = moments_final, residuals = residuals,
        converged = converged, sweeps = sweep, nSolves = n_solves[],
        elapsedSeconds = time() - start_time,
        targets = targets, params = p_final,
    )

    config.verbose && print_calibration_result(result)
    return result
end

"""
    print_calibration_result(result)

Report the calibrated instruments, the achieved fit, and the welfare check.
Callers that follow this with `print_equilibrium_summary` should pass
`show_welfare = false` so the welfare block is not printed twice.
"""
function print_calibration_result(result)
    t = result.targets
    m = result.moments
    r = result.residuals
    targeted, untargeted = t.asset_moment === :median ?
        ("median A / mean Y", "mean A / mean Y") :
        ("mean A / mean Y", "median A / mean Y")

    println("\n=== Calibration result ===")
    @printf("converged                = %s (after %d sweep(s), maxgap=%.2e)\n",
            result.converged, result.sweeps, max_abs_resid(m, t))
    @printf("qSav                     = %.8f\n", result.qSav)
    @printf("qBorr                    = %.8f\n", result.qBorr)
    @printf("bbar                     = %.8f\n", result.bbar)
    @printf("%-24s = %.8f  (target %.6f, resid % .2e)\n",
            targeted, asset_ratio(m, t), asset_target(t),
            r.assetsToMeanLaborIncome)
    @printf("true borr lim / mean Y   = %.8f  (target %.6f, resid % .2e)\n",
            m.trueBorrowingLimitToMeanLaborIncome,
            t.trueBorrowingLimitToMeanLaborIncome,
            r.trueBorrowingLimitToMeanLaborIncome)
    @printf("share negative liquid A  = %.8f  (target %.6f, resid % .2e)\n",
            m.shareNegativeLiquidAssets, t.shareNegativeLiquidAssets,
            r.shareNegativeLiquidAssets)
    @printf("%-24s = %.8f  (not targeted)\n", untargeted,
            t.asset_moment === :median ? m.meanAssetsToMeanLaborIncome :
                                         m.medianAssetsToMeanLaborIncome)
    hasproperty(result.eq, :welfare) && print_welfare_summary(result.eq.welfare)
    @printf("model solves             = %d\n", result.nSolves)
    @printf("calibration time         = %.3f seconds\n", result.elapsedSeconds)
    flush(stdout)
    return nothing
end

# -----------------------------------------------------------------------------
# Script entry point (mirrors run_history_independent_tax.jl)
# -----------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    result = calibrate_history_independent_tax()
    print_equilibrium_summary(result.eq, result.params;
                              title = "Final calibrated equilibrium",
                              show_welfare = false)
end
