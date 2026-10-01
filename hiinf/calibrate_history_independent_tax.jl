# =============================================================================
# calibrate_history_independent_tax.jl
#
# Calibrate the three financial / borrowing-limit parameters
#
#     qSav   (gross savings price,  a' >= 0)
#     qBorr  (gross borrowing price, a' <  0)
#     bbar   (borrowing-limit scale, bbar <= 0)
#
# so that the cross-section produced by solve_history_independent_tax matches
# three data moments, read off eq.statistics:
#
#     (i)   mean (or median) assets / mean labor income
#     (ii)  true borrowing limit / mean labor income
#     (iii) share of households with negative liquid assets
#
# The solver is used unmodified; this file only wraps it.
#
# -----------------------------------------------------------------------------
# IDENTIFICATION
# -----------------------------------------------------------------------------
# The mapping (qSav, qBorr, bbar) -> (i, ii, iii) is coupled but near
# triangular, which the search exploits:
#
#   bbar  enters the borrowing-limit numerator alone, so it pins (ii) almost
#         mechanically, with only second-order feedback through labor income.
#   qSav  governs the return to saving, hence the asset distribution -> (i).
#   qBorr governs the cost of borrowing, hence the negative share -> (iii).
#
# Each instrument is moved by a 1-D bracketed root find holding the others
# fixed, sweeping the three blocks until the joint residual clears tolerance.
# Every residual evaluation rebuilds HIParams, because the asset grid's lower
# bound depends on bbar, and calls the solver with output suppressed.
#
# Marek Kapicka, 2026
# =============================================================================

using Dates
using Printf

# Includes flow one direction: solve <- settings <- run <- calibrate.
include("run_history_independent_tax.jl")

# -----------------------------------------------------------------------------
# Calibration parameters: targets and search controls
# -----------------------------------------------------------------------------
"""
    CalibrationParams

Everything the calibration routine needs beyond the model itself: the data
moments to match, and the search controls. Target field names mirror the moment
they map to in `eq.statistics`.

`asset_moment` selects which asset statistic block (i) targets via `qSav`:
`:mean` (default) matches `meanAssetsToMeanLaborIncome`; `:median` matches
`medianAssetsToMeanLaborIncome` instead. Only the selected one is targeted; the
other is reported but not matched.

Brackets are on the *economically meaningful* ranges of the instruments:
  * `qSav`  in (0, qSav_max]; higher q => cheaper saving => more saving.
  * `qBorr` in [qBorr_min, qBorr_max]; higher q => cheaper borrowing => more
    households with a' < 0.  qBorr <= qSav is NOT imposed by the model,
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
Base.@kwdef struct CalibrationParams
    # Targets. Only the asset_moment-selected ratio among the first two is
    # targeted; the other is reported but left free. All three come from
    # Kaplan and Violante (2014); see NOTES.md for the table and the figures.
    medianAssetsToMeanLaborIncome::Float64       = 0.0498  # (i), asset_moment = :median
    meanAssetsToMeanLaborIncome::Float64         = 0.588   # (i), asset_moment = :mean
    trueBorrowingLimitToMeanLaborIncome::Float64 = 0.185   # (ii)
    shareNegativeLiquidAssets::Float64           = 0.260   # (iii)
    asset_moment::Symbol                         = :mean   # :median or :mean

    # Initial guesses for the instruments. These default to the SETTINGS values,
    # which is where the starting point used to come from -- and the ONLY place
    # it could come from, since qSav/qBorr/bbar are rejected in base_kwargs as
    # calibrated instruments. Overriding them here changes the starting point
    # without touching model_settings.jl:
    #
    #   calibrate_history_independent_tax(calib = CalibrationParams(bbar_init = -0.17707415))
    #
    # bbar_init is the one worth setting deliberately. Block (ii) is EXACTLY
    # linear in bbar -- the true limit is -bbar * E[exp(kappa + rho*z)] and bbar
    # enters nothing else in that moment -- so a single solve pins it down:
    # at bbar = -0.2 the windowed ratio is 0.20895201 against a target of 0.185,
    # giving -0.2 * 0.185/0.20895201 = -0.17707415. Measured at nZ=15, nEps=11,
    # nKappa=5, nA=151; it moves little with the grid, and as a starting point
    # it only needs to be close. The old default of -0.2 was 14% away.
    # qSav_init/qBorr_init are calibrated values carried over from a previous
    # run, not the SETTINGS defaults (0.99 / 0.97): starting the search at a
    # near-root saves sweeps, and qSav is the instrument that costs the most
    # solves. They were calibrated against the ALL-AGES statistic, so with the
    # ages 3-40 window the mean asset ratio is 4.3% lower and the qSav root sits
    # somewhat above this -- still a far better start than 0.99.
    qSav_init::Float64  = 0.980681209802701
    qBorr_init::Float64 = 0.9895282395052278
    bbar_init::Float64  = -0.17707415

    # Instrument brackets
    qSav_min::Float64  = 0.970
    qSav_max::Float64  = 0.997
    qBorr_min::Float64 = 0.960
    qBorr_max::Float64 = 1.040
    bbar_min::Float64  = -0.20
    bbar_max::Float64  = -0.15

    # Inner 1-D root finder (Roots.Brent)
    inner_xtol::Float64 = 1e-5
    inner_maxevals::Int = 40

    # Outer block-coordinate loop
    outer_max_sweeps::Int = 12
    moment_tol::Float64   = 5e-4

    # The inner model solves are silenced regardless; this controls only the
    # calibration's own logging.
    verbose::Bool = true
end

# One selector serves both the achieved moments and the targets, which share
# the field name.
asset_field(c::CalibrationParams) =
    c.asset_moment === :median ? :medianAssetsToMeanLaborIncome :
                                 :meanAssetsToMeanLaborIncome
asset_ratio(m, c::CalibrationParams)  = getproperty(m, asset_field(c))
asset_target(c::CalibrationParams)    = getproperty(c, asset_field(c))
asset_label(c::CalibrationParams) =
    c.asset_moment === :median ? "median assets / mean labor income" :
                                 "mean assets / mean labor income"
asset_label_short(c::CalibrationParams) =
    c.asset_moment === :median ? "median/LI" : "mean/LI"

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
# asset ratio (median or mean) `asset_moment` selects.
resid_i(m, t)   = asset_ratio(m, t)                     - asset_target(t)
resid_ii(m, t)  = m.trueBorrowingLimitToMeanLaborIncome - t.trueBorrowingLimitToMeanLaborIncome
resid_iii(m, t) = m.shareNegativeLiquidAssets           - t.shareNegativeLiquidAssets

max_abs_resid(m, t) = max(abs(resid_i(m, t)),
                          abs(resid_ii(m, t)),
                          abs(resid_iii(m, t)))

# -----------------------------------------------------------------------------
# A robust 1-D bracketed solver wrapper
# -----------------------------------------------------------------------------
# `f` is monotone in the relevant region but steps, because the asset choice
# lives on a discrete grid. So: probe the bracket endpoints, scan the interval
# if they do not straddle zero, and fall back to the endpoint with the smallest
# residual when no sign change exists.
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
# The three blocks, in sweep order. `i` indexes x = [qSav, qBorr, bbar],
# `lo`/`hi` name the bracket fields, `resid` is the moment that instrument
# zeroes: bbar -> (ii), qSav -> (i), qBorr -> (iii).
const BLOCKS = (
    (i = 3, lo = :bbar_min,  hi = :bbar_max,  resid = resid_ii),
    (i = 1, lo = :qSav_min,  hi = :qSav_max,  resid = resid_i),
    (i = 2, lo = :qBorr_min, hi = :qBorr_max, resid = resid_iii),
)

"""
    calibrate_history_independent_tax(; calib, base_kwargs...)

Calibrate `(qSav, qBorr, bbar)` to the targets in `calib`. Returns a NamedTuple

    (; qSav, qBorr, bbar, eq, moments, residuals, converged, sweeps,
       nSolves, elapsedSeconds, calib, params)

where `eq` is the equilibrium at the calibrated parameters, re-solved once with
the caller's verbosity so the full solver log follows the calibration,
`params` are the calibrated `HIParams`, `moments`/`residuals` report the
achieved fit, and `nSolves` counts full model solves.

`calib` must be a `CalibrationParams`; build one explicitly to retarget or to
change the search, as in `calib = CalibrationParams(asset_moment = :median,
moment_tol = 1e-4)`.

`base_kwargs` are forwarded to `make_history_independent_params` for every
evaluation, so any non-calibrated setting can be overridden here. The
calibrated instruments (`qSav`, `qBorr`, `bbar`) and `verbose` cannot be passed
that way.
"""
function calibrate_history_independent_tax(;
        calib::CalibrationParams = CalibrationParams(),
        base_kwargs...)

    start_time = time()

    calib.asset_moment in (:median, :mean) ||
        error("calib.asset_moment must be :median or :mean")

    # Calibrated instruments (and inner-solver verbosity) are controlled here;
    # passing them through base_kwargs would silently conflict.
    for k in (:qSav, :qBorr, :bbar, :verbose)
        haskey(base_kwargs, k) &&
            error("`$(k)` cannot be passed via base_kwargs; it is set by the calibration")
    end

    # Instrument vector, ordered as BLOCKS indexes it. The starting point comes
    # from calib, not SETTINGS, so it can be set per call; the defaults keep the
    # SETTINGS values for qSav/qBorr.
    x = [clamp(calib.qSav_init,  calib.qSav_min,  calib.qSav_max),
         clamp(calib.qBorr_init, calib.qBorr_min, calib.qBorr_max),
         clamp(calib.bbar_init,  calib.bbar_min,  calib.bbar_max)]

    # Every evaluation is a full model solve, so memoize on the instrument
    # triple: Brent re-probes endpoints and the sweep end repeats a point.
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

    if calib.verbose
        println("\n=== Calibration targets ===")
        @printf("%-40s = %.8f\n", asset_label(calib), asset_target(calib))
        @printf("true borrowing limit / mean labor income = %.8f\n",
                calib.trueBorrowingLimitToMeanLaborIncome)
        @printf("share negative liquid assets             = %.8f\n",
                calib.shareNegativeLiquidAssets)
        println()
        println("=== Calibration search ===")
        @printf("start:   qSav=%.6f qBorr=%.6f bbar=%.6f\n", x[1], x[2], x[3])
        println()
        @printf("%4s   %-10s  %-10s  %-11s %-10s  %-10s  %-9s %s\n",
                "eval", "qSav", "qBorr", "bbar",
                asset_label_short(calib), "trueBL/LI", "neg share", "max")
        flush(stdout)
    end

    function print_row(label, x, m, gap, note::String = "")
        @printf("%4s  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f %.0e%s\n",
                label, x[1], x[2], x[3],
                asset_ratio(m, calib),
                m.trueBorrowingLimitToMeanLaborIncome,
                m.shareNegativeLiquidAssets,
                gap, note)
        flush(stdout)
    end

    # Baseline at the starting point (row 1); exit early if already on target.
    moments = moments_at(x)
    converged = max_abs_resid(moments, calib) <= calib.moment_tol
    sweep = 0
    calib.verbose && print_row("1", x, moments, max_abs_resid(moments, calib))

    while !converged && sweep < calib.outer_max_sweeps
        sweep += 1
        bracketed = true

        for blk in BLOCKS
            probe = copy(x)
            f = function (z)
                probe[blk.i] = z
                return blk.resid(moments_at(probe), calib)
            end
            x[blk.i], br = solve_scalar(f, getfield(calib, blk.lo),
                                        getfield(calib, blk.hi);
                                        xtol = calib.inner_xtol,
                                        maxevals = calib.inner_maxevals)
            bracketed &= br
        end

        # Cache hit: the last block just evaluated this exact triple.
        moments = moments_at(x)
        gap = max_abs_resid(moments, calib)
        calib.verbose && print_row(string(sweep + 1), x, moments, gap,
                                    bracketed ? "" : "  [unbracketed]")
        converged = gap <= calib.moment_tol
    end

    # Re-solved at the calibrated point with the caller's verbosity.
    p_final = make_history_independent_params(;
        base_kwargs..., qSav = x[1], qBorr = x[2], bbar = x[3])
    eq = solve_history_independent_tax(p_final)
    n_solves[] += 1

    moments_final = moments_from(eq)
    residuals = (;
        assetsToMeanLaborIncome             = resid_i(moments_final, calib),
        trueBorrowingLimitToMeanLaborIncome = resid_ii(moments_final, calib),
        shareNegativeLiquidAssets           = resid_iii(moments_final, calib),
    )

    result = (;
        qSav = x[1], qBorr = x[2], bbar = x[3],
        eq = eq, moments = moments_final, residuals = residuals,
        converged = converged, sweeps = sweep, nSolves = n_solves[],
        elapsedSeconds = time() - start_time,
        calib = calib, params = p_final,
    )

    calib.verbose && print_calibration_result(result)
    return result
end

"""
    print_calibration_result(result)

Report the calibrated instruments, the achieved fit, and the welfare check.
Callers that follow this with `print_equilibrium_summary` should pass
`show_welfare = false` so the welfare block is not printed twice.
"""
function print_calibration_result(result)
    t = result.calib
    m = result.moments
    r = result.residuals
    targeted, untargeted = t.asset_moment === :median ?
        ("median A / mean Y", "mean A / mean Y") :
        ("mean A / mean Y", "median A / mean Y")

    println("\n=== Calibration result ===")
    @printf("converged                = %s (after %d sweep(s), maxgap=%.2e)\n",
            result.converged, result.sweeps, max_abs_resid(m, t))
    @printf("qSav                     = %s\n", result.qSav)
    @printf("qBorr                    = %s\n", result.qBorr)
    @printf("bbar                     = %s\n", result.bbar)
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

function calibration_log_path(calib::CalibrationParams;
                              log_dir = joinpath(@__DIR__, "calibration_results"))
    s = SETTINGS
    stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")
    return joinpath(log_dir,
        "calib_$(calib.asset_moment)_J$(s.J)_nA$(s.nA)_nZ$(s.nZ)" *
        "_nEps$(s.nEps)_nKappa$(s.nKappa)_$(stamp).txt")
end

if abspath(PROGRAM_FILE) == @__FILE__
    calib = CalibrationParams()
    log_path = calibration_log_path(calib)
    result = with_tee(log_path) do
        r = calibrate_history_independent_tax(calib = calib)
        print_equilibrium_summary(r.eq, r.params;
                                  title = "Final calibrated equilibrium",
                                  show_welfare = false)
        r
    end
    println("\ntranscript saved to ", log_path)
end
