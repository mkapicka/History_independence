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
#     (i)   median assets / mean labor income            = 0.043
#     (ii)  true borrowing limit / mean labor income     = 0.180
#     (iii) share of households with negative liquid      = 0.260
#           assets
#
# These map onto the statistics returned by `finalize_statistics` as
#
#     (i)   eq.statistics.medianAssetsToMeanLaborIncome
#     (ii)  eq.statistics.meanBorrowingLimitToMeanLaborIncome
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
#     in simulate_kappa!).  It therefore pins down moment (ii) almost
#     mechanically, with only a second-order feedback through mean labor
#     income.  -> bbar solves (ii).
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
using Statistics

include("solve_history_independent_tax.jl")
include("plot_history_independent_tax.jl")
include("model_settings.jl")

# -----------------------------------------------------------------------------
# Calibration targets and configuration
# -----------------------------------------------------------------------------
"""
    CalibrationTargets

Data moments the calibration matches. Field names mirror the moment they map to
in `eq.statistics`.

`asset_moment` selects which asset statistic block (i) targets via `qSav`:
`:median` (default) matches `medianAssetsToMeanLaborIncome`; `:mean` matches
`meanAssetsToMeanLaborIncome` instead. Only the selected one is targeted; the
other is reported but not matched.
"""
Base.@kwdef struct CalibrationTargets
    medianAssetsToMeanLaborIncome::Float64       = 0.043   # (i), asset_moment = :median
    meanAssetsToMeanLaborIncome::Float64         = 0.588   # (i), asset_moment = :mean
    trueBorrowingLimitToMeanLaborIncome::Float64 = 0.180   # (ii)
    shareNegativeLiquidAssets::Float64           = 0.260   # (iii)
    asset_moment::Symbol                         = :median # :median or :mean
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
    qSav_max::Float64  = 0.999
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
    evaluate_moments(qSav, qBorr, bbar; base_kwargs...)

Rebuild `HIParams` with the candidate instruments (all other settings inherited
from `SETTINGS` / `base_kwargs`), solve the model silently, and return
`(moments, eq)` where `moments` is a NamedTuple of the three calibration
statistics.

Rebuilding (rather than mutating) is required because the asset grid's lower
bound `amin = min_{kappa,z} bbar*exp(kappa+rho*z)` depends on `bbar`.
"""
function evaluate_moments(qSav::Float64, qBorr::Float64, bbar::Float64;
                          base_kwargs...)
    p = make_history_independent_params(;
        qSav = qSav,
        qBorr = qBorr,
        bbar = bbar,
        verbose = false,        # silence the inner solver
        base_kwargs...,
    )
    eq, _ = solve_history_independent_tax(p)
    s = eq.statistics
    moments = (;
        medianAssetsToMeanLaborIncome       = s.medianAssetsToMeanLaborIncome,
        meanAssetsToMeanLaborIncome         = s.meanAssetsToMeanLaborIncome,
        trueBorrowingLimitToMeanLaborIncome = s.meanBorrowingLimitToMeanLaborIncome,
        shareNegativeLiquidAssets           = s.shareNegativeLiquidAssets,
    )
    return moments, eq
end

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
(e.g. `asset_moment = :mean`, `meanAssetsToMeanLaborIncome = 0.6`) is
intercepted and used to build the targets instead of being forwarded to the
model; an explicitly supplied `targets` takes precedence over such keywords.
The calibrated instruments (`qSav`, `qBorr`, `bbar`) and `verbose` cannot be
passed through `base_kwargs`.
"""
function calibrate_history_independent_tax(;
        targets::Union{CalibrationTargets,Nothing} = nothing,
        config::CalibrationConfig = CalibrationConfig(),
        kwargs...)

    cfg = config
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
    qSav  = clamp(get(SETTINGS, :qSav, 0.99),  cfg.qSav_min,  cfg.qSav_max)
    qBorr = clamp(get(SETTINGS, :qBorr, 0.90), cfg.qBorr_min, cfg.qBorr_max)
    bbar  = clamp(get(SETTINGS, :bbar, -0.20), cfg.bbar_min,  cfg.bbar_max)

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
                verbose = false, lambda_kwargs...)
            n_solves[] += 1
            return solve_history_independent_tax(p)
        end

        local eq
        if cfg.lambda_warm_start && isfinite(lambda_warm[])
            w = cfg.lambda_warm_width
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
        s = eq.statistics
        moments = (;
            medianAssetsToMeanLaborIncome       = s.medianAssetsToMeanLaborIncome,
            meanAssetsToMeanLaborIncome         = s.meanAssetsToMeanLaborIncome,
            trueBorrowingLimitToMeanLaborIncome = s.meanBorrowingLimitToMeanLaborIncome,
            shareNegativeLiquidAssets           = s.shareNegativeLiquidAssets,
        )
        cache[key] = (moments, eq)
        return moments, eq
    end

    # Root-find one instrument: from sweep 2 onward, search a narrow window
    # around the current value first (the solution moves little between
    # sweeps), falling back to the full bracket if no sign change is found.
    function solve_block(f, x_now::Float64, lo_full::Float64, hi_full::Float64,
                         first_sweep::Bool)
        if !first_sweep && cfg.shrink_brackets
            w = max(cfg.bracket_shrink * (hi_full - lo_full), 10.0 * cfg.inner_xtol)
            lo = max(lo_full, x_now - w)
            hi = min(hi_full, x_now + w)
            x, fx, br = solve_scalar(f, lo, hi;
                                     xtol = cfg.inner_xtol,
                                     maxevals = cfg.inner_maxevals)
            br && return x, fx, br
        end
        return solve_scalar(f, lo_full, hi_full;
                            xtol = cfg.inner_xtol,
                            maxevals = cfg.inner_maxevals)
    end

    if cfg.verbose
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
    converged = max_abs_resid(moments, targets) <= cfg.moment_tol
    sweep = 0
    cfg.verbose && print_row("1", moments, max_abs_resid(moments, targets))

    while !converged && sweep < cfg.outer_max_sweeps
        sweep += 1
        first_sweep = sweep == 1

        # --- Block (ii): bbar -> true borrowing limit / mean labor income ---
        # More negative bbar raises the numerator, so the residual is
        # *increasing* in (-bbar), i.e. decreasing in bbar.
        f_ii = b -> resid_ii(first(eval_point(qSav, qBorr, b)), targets)
        bbar, _, br_ii = solve_block(f_ii, bbar, cfg.bbar_min, cfg.bbar_max,
                                     first_sweep)

        # --- Block (i): qSav -> asset ratio (median or mean per asset_moment) ---
        # Higher qSav => cheaper saving => higher (median and mean) assets =>
        # residual increasing in qSav.
        f_i = q -> resid_i(first(eval_point(q, qBorr, bbar)), targets)
        qSav, _, br_i = solve_block(f_i, qSav, cfg.qSav_min, cfg.qSav_max,
                                    first_sweep)

        # --- Block (iii): qBorr -> share with negative liquid assets ---
        # Higher qBorr => cheaper borrowing => larger negative-asset share =>
        # residual increasing in qBorr.
        f_iii = q -> resid_iii(first(eval_point(qSav, q, bbar)), targets)
        qBorr, _, br_iii = solve_block(f_iii, qBorr, cfg.qBorr_min, cfg.qBorr_max,
                                       first_sweep)

        # Cache hit: the qBorr block just evaluated this exact triple.
        moments, _ = eval_point(qSav, qBorr, bbar)
        gap = max_abs_resid(moments, targets)

        cfg.verbose && print_row(string(sweep + 1), moments, gap,
                                 (br_i && br_ii && br_iii) ? "" : "  [unbracketed]")

        converged = gap <= cfg.moment_tol
    end

    # Equilibrium at the calibrated point: reuse the cached solve unless the
    # caller asked for a fresh, fully-logged final solve.
    p_final = make_history_independent_params(;
        base_kwargs..., qSav = qSav, qBorr = qBorr, bbar = bbar)
    if cfg.final_resolve
        eq, _ = solve_history_independent_tax(p_final)
        n_solves[] += 1
    else
        _, eq = eval_point(qSav, qBorr, bbar)   # cache hit
    end
    moments_final = (;
        medianAssetsToMeanLaborIncome       = eq.statistics.medianAssetsToMeanLaborIncome,
        meanAssetsToMeanLaborIncome         = eq.statistics.meanAssetsToMeanLaborIncome,
        trueBorrowingLimitToMeanLaborIncome = eq.statistics.meanBorrowingLimitToMeanLaborIncome,
        shareNegativeLiquidAssets           = eq.statistics.shareNegativeLiquidAssets,
    )
    residuals = (;
        assetsToMeanLaborIncome             = resid_i(moments_final, targets),
        trueBorrowingLimitToMeanLaborIncome = resid_ii(moments_final, targets),
        shareNegativeLiquidAssets           = resid_iii(moments_final, targets),
    )

    elapsed = time() - start_time

    if cfg.verbose
        println("\n=== Calibration result ===")
        @printf("converged                = %s (after %d sweep(s), maxgap=%.2e)\n",
                converged, sweep, max_abs_resid(moments_final, targets))
        @printf("qSav                     = %.8f\n", qSav)
        @printf("qBorr                    = %.8f\n", qBorr)
        @printf("bbar                     = %.8f\n", bbar)
        @printf("%-24s = %.8f  (target %.6f, resid % .2e)\n",
                targets.asset_moment === :median ? "median A / mean Y" :
                                                   "mean A / mean Y",
                asset_ratio(moments_final, targets),
                asset_target(targets),
                residuals.assetsToMeanLaborIncome)
        @printf("true borr lim / mean Y   = %.8f  (target %.6f, resid % .2e)\n",
                moments_final.trueBorrowingLimitToMeanLaborIncome,
                targets.trueBorrowingLimitToMeanLaborIncome,
                residuals.trueBorrowingLimitToMeanLaborIncome)
        @printf("share negative liquid A  = %.8f  (target %.6f, resid % .2e)\n",
                moments_final.shareNegativeLiquidAssets,
                targets.shareNegativeLiquidAssets,
                residuals.shareNegativeLiquidAssets)
        @printf("%-24s = %.8f  (not targeted)\n",
                targets.asset_moment === :median ? "mean A / mean Y" :
                                                   "median A / mean Y",
                targets.asset_moment === :median ?
                    moments_final.meanAssetsToMeanLaborIncome :
                    moments_final.medianAssetsToMeanLaborIncome)
        @printf("model solves             = %d\n", n_solves[])
        @printf("calibration time         = %.3f seconds\n", elapsed)
        flush(stdout)
    end

    return (;
        qSav = qSav, qBorr = qBorr, bbar = bbar,
        eq = eq, moments = moments_final, residuals = residuals,
        converged = converged, sweeps = sweep, nSolves = n_solves[],
        elapsedSeconds = elapsed,
        targets = targets, params = p_final,
    )
end

# -----------------------------------------------------------------------------
# Script entry point (mirrors run_history_independent_tax.jl)
# -----------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    result = calibrate_history_independent_tax()

    eq = result.eq
    asset_grid_path = save_asset_grid_figure(result.params)
    figure_paths = save_unconditional_distribution_figures(eq)

    @printf("\n=== Final calibrated equilibrium ===\n")
    @printf("qSav                       = %.8f\n", result.qSav)
    @printf("qBorr                      = %.8f\n", result.qBorr)
    @printf("bbar                       = %.8f\n", result.bbar)
    @printf("lambda                     = %.8f\n", eq.lambda)
    @printf("government budget residual = %.8e\n", eq.govBudgetResidual)
    @printf("mean output                = %.8f\n", mean(eq.Y))
    @printf("mean consumption           = %.8f\n", mean(eq.C))

    s = eq.statistics
    t = result.targets
    @printf("\n=== Calibrated moments ===\n")
    @printf("%-40s = %.8f (target %.3f)\n",
            asset_label(t), asset_ratio(s, t), asset_target(t))
    @printf("true borrowing limit / mean labor income = %.8f (target %.3f)\n",
            s.meanBorrowingLimitToMeanLaborIncome,
            t.trueBorrowingLimitToMeanLaborIncome)
    @printf("share negative liquid assets             = %.8f (target %.3f)\n",
            s.shareNegativeLiquidAssets, t.shareNegativeLiquidAssets)
    @printf("%-40s = %.8f (not targeted)\n",
            t.asset_moment === :median ? "mean assets / mean labor income" :
                                         "median assets / mean labor income",
            t.asset_moment === :median ? s.meanAssetsToMeanLaborIncome :
                                         s.medianAssetsToMeanLaborIncome)

    @printf("\n=== Figures saved ===\n")
    @printf("asset grid                = %s\n", display_path(asset_grid_path))
    @printf("assets                    = %s\n", display_path(figure_paths.assets))
    @printf("hours worked              = %s\n", display_path(figure_paths.hours))
    @printf("consumption               = %s\n", display_path(figure_paths.consumption))
end
