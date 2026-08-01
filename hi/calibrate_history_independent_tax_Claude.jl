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

using Dates
using Printf

# Includes flow one direction: solve <- settings <- run <- calibrate.
include("run_history_independent_tax.jl")

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

# Moment (i) reads the same field name out of the achieved moments and out of
# the targets, so one field selector serves both.
asset_field(t::CalibrationTargets) =
    t.asset_moment === :median ? :medianAssetsToMeanLaborIncome :
                                 :meanAssetsToMeanLaborIncome
asset_ratio(m, t::CalibrationTargets)  = getproperty(m, asset_field(t))
asset_target(t::CalibrationTargets)    = getproperty(t, asset_field(t))
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

Earlier versions carried three speed controls, all since removed:
  * warm-started lambda brackets: measured to do nothing. Brent needs 5
    aggregate solves per equilibrium whether it starts from [0.20, 2.50] or
    from lambda*(1 +- 0.10), because its convergence is insensitive to the
    initial bracket width.
  * sweep-to-sweep bracket shrinking: this one did save work. The search takes
    144 model solves without it against 127 with it, about 13% more. Restore it
    if that matters; the calibrated instruments agree to six significant
    figures either way.
  * reuse of the cached final equilibrium: saved one solve in ~130.
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

Find x in [lo, hi] with f(x) ~ 0. Returns `(x, bracketed::Bool)`. The residual
at `x` is deliberately not returned: every caller ignored it, and computing it
costs an extra model solve whenever Brent's last probe was not at the root.
"""
function solve_scalar(f, lo::Float64, hi::Float64;
                      xtol::Float64 = 1e-5, maxevals::Int = 40,
                      nscan::Int = 9)
    flo = f(lo)
    fhi = f(hi)

    if abs(flo) <= xtol
        return lo, true
    elseif abs(fhi) <= xtol
        return hi, true
    end

    a, b = lo, hi
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
            return xs[argmin(abs.(fs))], false
        end
        klo, khi = bracket
        a, b = xs[klo], xs[khi]
    end

    x = Roots.find_zero(f, (a, b), Roots.Brent();
                        xatol = xtol, maxevals = max(maxevals, 12))
    return x, true
end

# -----------------------------------------------------------------------------
# Main calibration routine
# -----------------------------------------------------------------------------
# The three blocks, in sweep order. `i` indexes the instrument vector
# x = [qSav, qBorr, bbar]; `lo`/`hi` name the bracket fields of
# CalibrationConfig; `resid` is the moment residual that instrument zeroes.
#   bbar  -> (ii): more negative bbar raises -bbar*E[exp(kappa+rho z)], so the
#            residual is increasing in -bbar, i.e. decreasing in bbar.
#   qSav  -> (i):  higher qSav => cheaper saving => higher assets.
#   qBorr -> (iii):higher qBorr => cheaper borrowing => larger negative share.
const BLOCKS = (
    (i = 3, lo = :bbar_min,  hi = :bbar_max,  resid = resid_ii),
    (i = 1, lo = :qSav_min,  hi = :qSav_max,  resid = resid_i),
    (i = 2, lo = :qBorr_min, hi = :qBorr_max, resid = resid_iii),
)

"""
    calibrate_history_independent_tax(; targets, config, base_kwargs...)

Calibrate `(qSav, qBorr, bbar)` to `targets`. Returns a NamedTuple

    (; qSav, qBorr, bbar, eq, moments, residuals, converged, sweeps,
       nSolves, elapsedSeconds, targets, params)

where `eq` is the equilibrium at the calibrated parameters, re-solved once with
the caller's verbosity so the full solver log follows the calibration,
`params` are the calibrated `HIParams`, `moments`/`residuals` report the
achieved fit, and `nSolves` counts full model solves.

`targets` must be a `CalibrationTargets`; build one explicitly to retarget, as
in `targets = CalibrationTargets(asset_moment = :median)`.

`base_kwargs` are forwarded to `make_history_independent_params` for every
evaluation, so any non-calibrated setting can be overridden here. The
calibrated instruments (`qSav`, `qBorr`, `bbar`) and `verbose` cannot be passed
that way.
"""
function calibrate_history_independent_tax(;
        targets::CalibrationTargets = CalibrationTargets(),
        config::CalibrationConfig = CalibrationConfig(),
        base_kwargs...)

    start_time = time()

    targets.asset_moment in (:median, :mean) ||
        error("targets.asset_moment must be :median or :mean")

    # Calibrated instruments (and inner-solver verbosity) are controlled here;
    # passing them through base_kwargs would silently conflict.
    for k in (:qSav, :qBorr, :bbar, :verbose)
        haskey(base_kwargs, k) &&
            error("`$(k)` cannot be passed via base_kwargs; it is set by the calibration")
    end

    # Instrument vector, ordered as BLOCKS indexes it.
    x = [clamp(SETTINGS.qSav,  config.qSav_min,  config.qSav_max),
         clamp(SETTINGS.qBorr, config.qBorr_min, config.qBorr_max),
         clamp(SETTINGS.bbar,  config.bbar_min,  config.bbar_max)]

    # Every moment evaluation is a full model solve, so memoize on the
    # instrument triple: Brent re-probes bracket endpoints, and the sweep-end
    # evaluation repeats the point the last block just solved.
    cache = Dict{NTuple{3,Float64},Any}()
    n_solves = Ref(0)

    function eval_point(qS::Float64, qB::Float64, bb::Float64)
        key = (qS, qB, bb)
        haskey(cache, key) && return cache[key]

        p = make_history_independent_params(;
            base_kwargs..., qSav = qS, qBorr = qB, bbar = bb,
            verbose = false, collect_distributions = false)
        n_solves[] += 1
        eq = solve_history_independent_tax(p)

        cache[key] = (moments_from(eq), eq)
        return cache[key]
    end

    moments_at(x) = first(eval_point(x[1], x[2], x[3]))

    if config.verbose
        println("\n=== Calibration targets ===")
        @printf("%-40s = %.8f\n", asset_label(targets), asset_target(targets))
        @printf("true borrowing limit / mean labor income = %.8f\n",
                targets.trueBorrowingLimitToMeanLaborIncome)
        @printf("share negative liquid assets             = %.8f\n",
                targets.shareNegativeLiquidAssets)
        println()
        println("=== Calibration search ===")
        @printf("start:   qSav=%.6f qBorr=%.6f bbar=%.6f\n", x[1], x[2], x[3])
        println()
        @printf("%4s   %-10s  %-10s  %-11s %-10s  %-10s  %-9s %s\n",
                "eval", "qSav", "qBorr", "bbar",
                asset_label_short(targets), "trueBL/LI", "neg share", "max")
        flush(stdout)
    end

    function print_row(label, x, m, gap, note::String = "")
        @printf("%4s  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f %.0e%s\n",
                label, x[1], x[2], x[3],
                asset_ratio(m, targets),
                m.trueBorrowingLimitToMeanLaborIncome,
                m.shareNegativeLiquidAssets,
                gap, note)
        flush(stdout)
    end

    # Baseline at the starting point (row 1); exit early if already on target.
    moments = moments_at(x)
    converged = max_abs_resid(moments, targets) <= config.moment_tol
    sweep = 0
    config.verbose && print_row("1", x, moments, max_abs_resid(moments, targets))

    while !converged && sweep < config.outer_max_sweeps
        sweep += 1
        bracketed = true

        for blk in BLOCKS
            probe = copy(x)
            f = function (z)
                probe[blk.i] = z
                return blk.resid(moments_at(probe), targets)
            end
            x[blk.i], br = solve_scalar(f, getfield(config, blk.lo),
                                        getfield(config, blk.hi);
                                        xtol = config.inner_xtol,
                                        maxevals = config.inner_maxevals)
            bracketed &= br
        end

        # Cache hit: the last block just evaluated this exact triple.
        moments = moments_at(x)
        gap = max_abs_resid(moments, targets)
        config.verbose && print_row(string(sweep + 1), x, moments, gap,
                                    bracketed ? "" : "  [unbracketed]")
        converged = gap <= config.moment_tol
    end

    # Equilibrium at the calibrated point, re-solved with the caller's
    # verbosity so the full solver log follows the search.
    p_final = make_history_independent_params(;
        base_kwargs..., qSav = x[1], qBorr = x[2], bbar = x[3])
    eq = solve_history_independent_tax(p_final)
    n_solves[] += 1

    moments_final = moments_from(eq)
    residuals = (;
        assetsToMeanLaborIncome             = resid_i(moments_final, targets),
        trueBorrowingLimitToMeanLaborIncome = resid_ii(moments_final, targets),
        shareNegativeLiquidAssets           = resid_iii(moments_final, targets),
    )

    result = (;
        qSav = x[1], qBorr = x[2], bbar = x[3],
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
"""
    with_tee(f, path)

Run `f()` with everything written to `stdout` also appended to the file at
`path`. Returns `f()`'s value. Used by the script entry point so calibration
transcripts are archived automatically instead of copy-pasted from the console.
"""
function with_tee(f, path::AbstractString)
    mkpath(dirname(path))
    io = open(path, "w")
    original = stdout
    rd, wr = redirect_stdout()
    copier = @async while !eof(rd)
        chunk = readavailable(rd)
        write(original, chunk)
        write(io, chunk)
        flush(original)
        flush(io)
    end
    try
        return f()
    finally
        redirect_stdout(original)
        close(wr)
        wait(copier)
        close(io)
    end
end

function calibration_log_path(targets::CalibrationTargets;
                              log_dir = joinpath(@__DIR__, "calibration_results"))
    s = SETTINGS
    stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")
    return joinpath(log_dir,
        "calib_$(targets.asset_moment)_J$(s.J)_nA$(s.nA)_nZ$(s.nZ)" *
        "_nEps$(s.nEps)_nKappa$(s.nKappa)_$(stamp).txt")
end

if abspath(PROGRAM_FILE) == @__FILE__
    targets = CalibrationTargets()
    log_path = calibration_log_path(targets)
    result = with_tee(log_path) do
        r = calibrate_history_independent_tax(targets = targets)
        print_equilibrium_summary(r.eq, r.params;
                                  title = "Final calibrated equilibrium",
                                  show_welfare = false)
        r
    end
    println("\ntranscript saved to ", log_path)
end
