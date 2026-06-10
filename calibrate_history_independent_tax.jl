using Printf
using Roots

include("solve_history_independent_tax.jl")
include("model_settings.jl")

const CALIBRATION_TARGETS = (;
    median_assets_to_mean_labor_income = 0.043,
    true_borrowing_limit_to_mean_labor_income = 0.180,
    share_negative_liquid_assets = 0.260,
)

const CALIBRATION_SETTINGS = (;
    qSav_bounds = (0.900, 0.9995),
    qBorr_bounds = (0.700, 0.9995),
    bbar_bounds = (-0.600, -0.010),
    moment_tol = 5e-4,
    max_sweeps = 8,
    scalar_xtol = 1e-4,
    scalar_maxevals = 18,
    scalar_scan_points = 9,
)

mutable struct CalibrationCache
    evaluations::Dict{Tuple{Float64,Float64,Float64},Any}
end

CalibrationCache() = CalibrationCache(Dict{Tuple{Float64,Float64,Float64},Any}())

cache_key(qSav::Float64, qBorr::Float64, bbar::Float64) =
    (round(qSav; digits = 8), round(qBorr; digits = 8), round(bbar; digits = 8))

function calibration_moments(eq)
    s = eq.statistics
    return (;
        median_assets_to_mean_labor_income =
            s.medianAssetsToMeanLaborIncome,
        true_borrowing_limit_to_mean_labor_income =
            s.meanBorrowingLimitToMeanLaborIncome,
        share_negative_liquid_assets =
            s.shareNegativeLiquidAssets,
    )
end

function calibration_residuals(moments, targets)
    return (;
        median_assets_to_mean_labor_income =
            moments.median_assets_to_mean_labor_income -
            targets.median_assets_to_mean_labor_income,
        true_borrowing_limit_to_mean_labor_income =
            moments.true_borrowing_limit_to_mean_labor_income -
            targets.true_borrowing_limit_to_mean_labor_income,
        share_negative_liquid_assets =
            moments.share_negative_liquid_assets -
            targets.share_negative_liquid_assets,
    )
end

max_abs_residual(residuals) = maximum(abs, (
    residuals.median_assets_to_mean_labor_income,
    residuals.true_borrowing_limit_to_mean_labor_income,
    residuals.share_negative_liquid_assets,
))

function evaluate_calibration!(cache::CalibrationCache, qSav::Float64,
                               qBorr::Float64, bbar::Float64,
                               targets; model_kwargs...)
    key = cache_key(qSav, qBorr, bbar)
    if haskey(cache.evaluations, key)
        return cache.evaluations[key]
    end

    p = make_history_independent_params(;
        model_kwargs...,
        qSav = qSav,
        qBorr = qBorr,
        bbar = bbar,
        verbose = false,
        printEveryLambda = 0,
    )
    eq, sol = solve_history_independent_tax(p)
    moments = calibration_moments(eq)
    residuals = calibration_residuals(moments, targets)
    result = (;
        qSav = qSav,
        qBorr = qBorr,
        bbar = bbar,
        moments = moments,
        residuals = residuals,
        max_abs_residual = max_abs_residual(residuals),
        eq = eq,
        sol = sol,
        params = p,
    )
    cache.evaluations[key] = result
    return result
end

function best_cached_result(cache::CalibrationCache)
    isempty(cache.evaluations) && error("calibration cache is empty")
    best = first(values(cache.evaluations))
    for candidate in values(cache.evaluations)
        if candidate.max_abs_residual < best.max_abs_residual
            best = candidate
        end
    end
    return best
end

function solve_scalar_closest(f, lo::Float64, hi::Float64;
                              xtol::Float64, maxevals::Int,
                              scan_points::Int)
    lo < hi || error("scalar bracket must satisfy lo < hi")
    scan_points >= 2 || error("scan_points must be at least 2")

    xs = collect(range(lo, hi; length = scan_points))
    fs = [f(x) for x in xs]
    best = argmin(abs.(fs))

    for i in 1:(length(xs) - 1)
        if !isfinite(fs[i]) || !isfinite(fs[i + 1])
            continue
        elseif fs[i] == 0.0
            return xs[i], fs[i], true
        elseif sign(fs[i]) != sign(fs[i + 1])
            root = Roots.find_zero(f, (xs[i], xs[i + 1]), Roots.Brent();
                                   xatol = xtol,
                                   maxevals = max(maxevals, 12))
            return root, f(root), true
        end
    end

    return xs[best], fs[best], false
end

function print_calibration_header(targets, settings)
    println("\n=== Calibrating qSav, qBorr, and bbar ===")
    @printf("targets: median assets / mean labor income        = %.6f\n",
            targets.median_assets_to_mean_labor_income)
    @printf("         true borrowing limit / mean labor income = %.6f\n",
            targets.true_borrowing_limit_to_mean_labor_income)
    @printf("         share negative liquid assets             = %.6f\n",
            targets.share_negative_liquid_assets)
    @printf("bounds:  qSav [% .4f, % .4f], qBorr [% .4f, % .4f], bbar [% .4f, % .4f]\n",
            settings.qSav_bounds[1], settings.qSav_bounds[2],
            settings.qBorr_bounds[1], settings.qBorr_bounds[2],
            settings.bbar_bounds[1], settings.bbar_bounds[2])
    println()
end

function print_calibration_row(label::AbstractString, result, targets)
    r = result.residuals
    @printf("%-8s qSav=% .6f qBorr=% .6f bbar=% .6f | ",
            label, result.qSav, result.qBorr, result.bbar)
    @printf("median=% .6f (%+.2e), borrow=% .6f (%+.2e), neg=% .6f (%+.2e) | max=% .2e\n",
            result.moments.median_assets_to_mean_labor_income,
            r.median_assets_to_mean_labor_income,
            result.moments.true_borrowing_limit_to_mean_labor_income,
            r.true_borrowing_limit_to_mean_labor_income,
            result.moments.share_negative_liquid_assets,
            r.share_negative_liquid_assets,
            result.max_abs_residual)
end

function calibrate_history_independent_tax(;
        targets = CALIBRATION_TARGETS,
        settings = CALIBRATION_SETTINGS,
        verbose::Bool = true,
        model_kwargs...)

    qSav = clamp(get(SETTINGS, :qSav, 0.99),
                 settings.qSav_bounds[1], settings.qSav_bounds[2])
    qBorr = clamp(get(SETTINGS, :qBorr, 0.90),
                  settings.qBorr_bounds[1], settings.qBorr_bounds[2])
    bbar = clamp(get(SETTINGS, :bbar, -0.20),
                 settings.bbar_bounds[1], settings.bbar_bounds[2])
    cache = CalibrationCache()

    if verbose
        print_calibration_header(targets, settings)
    end

    current = evaluate_calibration!(cache, qSav, qBorr, bbar, targets;
                                    model_kwargs...)
    verbose && print_calibration_row("start", current, targets)

    converged = current.max_abs_residual <= settings.moment_tol
    sweep = 0

    for iter in 1:settings.max_sweeps
        sweep = iter

        f_bbar(b) = evaluate_calibration!(
            cache, qSav, qBorr, b, targets; model_kwargs...
        ).residuals.true_borrowing_limit_to_mean_labor_income
        bbar, _, br_bbar = solve_scalar_closest(
            f_bbar, settings.bbar_bounds[1], settings.bbar_bounds[2];
            xtol = settings.scalar_xtol,
            maxevals = settings.scalar_maxevals,
            scan_points = settings.scalar_scan_points,
        )

        f_qSav(q) = evaluate_calibration!(
            cache, q, qBorr, bbar, targets; model_kwargs...
        ).residuals.median_assets_to_mean_labor_income
        qSav, _, br_qSav = solve_scalar_closest(
            f_qSav, settings.qSav_bounds[1], settings.qSav_bounds[2];
            xtol = settings.scalar_xtol,
            maxevals = settings.scalar_maxevals,
            scan_points = settings.scalar_scan_points,
        )

        f_qBorr(q) = evaluate_calibration!(
            cache, qSav, q, bbar, targets; model_kwargs...
        ).residuals.share_negative_liquid_assets
        qBorr, _, br_qBorr = solve_scalar_closest(
            f_qBorr, settings.qBorr_bounds[1], settings.qBorr_bounds[2];
            xtol = settings.scalar_xtol,
            maxevals = settings.scalar_maxevals,
            scan_points = settings.scalar_scan_points,
        )

        current = evaluate_calibration!(
            cache, qSav, qBorr, bbar, targets; model_kwargs...
        )

        if verbose
            label = @sprintf("sweep %d", iter)
            print_calibration_row(label, current, targets)
            if !(br_bbar && br_qSav && br_qBorr)
                println("         warning: at least one scalar update was unbracketed; used closest scanned value")
            end
            flush(stdout)
        end

        if current.max_abs_residual <= settings.moment_tol
            converged = true
            break
        end
    end

    current = evaluate_calibration!(cache, qSav, qBorr, bbar, targets;
                                    model_kwargs...)
    best = best_cached_result(cache)
    final = converged ? current : best

    if verbose
        println("\n=== Calibration Result ===")
        @printf("converged                 = %s\n", converged)
        @printf("sweeps                    = %d\n", sweep)
        @printf("model evaluations         = %d\n", length(cache.evaluations))
        if !converged && final !== current
            println("returned                  = best evaluated point")
        end
        @printf("qSav                      = %.8f\n", final.qSav)
        @printf("qBorr                     = %.8f\n", final.qBorr)
        @printf("bbar                      = %.8f\n", final.bbar)
        @printf("lambda                    = %.8f\n", final.eq.lambda)
        print_calibration_row("final", final, targets)
    end

    return merge(final, (;
        converged = converged,
        sweeps = sweep,
        targets = targets,
        settings = settings,
        model_evaluations = length(cache.evaluations),
    ))
end

if abspath(PROGRAM_FILE) == @__FILE__
    calibrate_history_independent_tax()
end
