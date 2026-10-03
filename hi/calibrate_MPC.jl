# =============================================================================
# calibrate_MPC.jl
#
# A TWO-INSTRUMENT, TWO-TARGET calibration with the three prices GIVEN, matching
# the AVERAGE IMPACT MPC instead of a wealth-to-income ratio:
#
#       qSav, qBorr, qGov  fixed by the caller,
#       beta               calibrated to the mean MPC,
#       bbar               calibrated to the borrowing limit.
#
# This is `calibrate_beta.jl` with its first target replaced. Everything else --
# the instruments, the block-coordinate sweep, the stall guard, the reuse of
# `solve_scalar` and the price treatment -- is the same, so the two files can be
# compared line for line and differ only where the target does.
#
# -----------------------------------------------------------------------------
# IDENTIFICATION
# -----------------------------------------------------------------------------
# Two blocks, swept in this order:
#
#   * bbar -> the borrowing-limit moment. Unchanged from `calibrate_beta.jl`:
#     bbar is the only parameter in the numerator -bbar * E[exp(kappa + rho z)],
#     so one solve very nearly pins it down. Swept first for that reason.
#
#   * beta -> the mean MPC. NOTE THE SIGN, WHICH IS THE OPPOSITE OF THE ASSET
#     BLOCK. A higher beta is more patience, so more saving, so more wealth, so
#     households sit further out on the FLAT part of the concave consumption
#     function: the mean MPC is DECREASING in beta, where the asset moment is
#     increasing in it. The root finder only needs a sign change and does not
#     care, but every bracket diagnostic below reads the other way round, and a
#     bracket copied from `calibrate_beta.jl` will report its endpoints
#     inverted.
#
#   * The borrowing limit moves the MPC too, and strongly: a looser limit lets
#     constrained households borrow rather than cut consumption. The two blocks
#     are therefore more entangled here than in the asset calibration, where
#     bbar is nearly exactly identified by its own moment. Expect more sweeps.
#
# -----------------------------------------------------------------------------
# WHAT THE TARGET IS, AND WHAT IT IS NOT
# -----------------------------------------------------------------------------
# The default target, 0.408, is the ANNUAL MPC that Kaplan and Violante (2022),
# Annual Review of Economics 14:747-75, Table 1 column 5 report for the one-asset
# model calibrated to MEAN LIQUID WEALTH -- the same calibration this directory
# runs. Their full Table 1 "Annual MPC (%)" row, for picking a different one:
#
#     column 2  baseline, mean net worth 4.1 x earnings   14.6
#     column 3  mean net worth including the top 10%       8.5
#     column 4  median net worth                          13.6
#     column 5  MEAN LIQUID WEALTH                        40.8   <- default here
#     column 6  median liquid wealth                      77.4
#     column 7  share of hand-to-mouth households         58.7
#
# THESE ARE MODEL OUTPUTS, NOT DATA MOMENTS. Column 1 of their Table 1 is the
# data and its MPC entries are NA: the paper calibrates to wealth and reports the
# MPC that follows. Targeting 0.408 therefore asks this model to reproduce the
# MPC of THEIR model under a comparable calibration, which is a model-comparison
# exercise, not an estimation. Set `meanMPC` to an empirical estimate if that is
# what is wanted.
#
# Two further mismatches against their number, both worth knowing before reading
# a residual as success:
#
#   * PERIOD LENGTH. Their baseline is quarterly and the 40.8 is the annualized
#     figure; this model is annual by construction, with one period per year of
#     real age. The two annual MPCs are built differently.
#
#   * LABOR SUPPLY. Their income process is exogenous, so c(b + x) >= c(b) and
#     every MPC is weakly positive. Hours are a choice here, so a windfall cuts
#     labor income and the MPC can be NEGATIVE; at the shipped calibration about
#     19% of mass has one. The mean over all cells is then a near-cancellation
#     of two large groups and is not the same object as theirs. The result
#     prints the positive/negative split and the mean over responders, which is
#     the more nearly comparable statistic.
#
# -----------------------------------------------------------------------------
# THE WINDFALL DRIFTS DURING THE SEARCH
# -----------------------------------------------------------------------------
# `mpc_shock` is a fixed number in model asset units, derived once from a dollar
# windfall and a mean labor income. The calibration moves beta, which moves mean
# labor income, so the windfall's size RELATIVE TO INCOME drifts as the search
# proceeds -- the target is nominally "the MPC out of $500", but the model's $500
# is not held fixed. The result reports the realized ratio at the solution so the
# drift is visible; re-derive `mpc_shock` and re-run if it has moved materially.
# Holding the dollar value fixed instead would mean recomputing `mpc_shock` from
# mean labor income at every evaluation, which makes the target a fixed point in
# its own right and is deliberately NOT done here.
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   include("calibrate_MPC.jl")
#
#   r = calibrate_MPC(nZ = 15, nEps = 11, nKappa = 5, nA = 151)
#
#   # a different target, and a tighter inner solve:
#   r = calibrate_MPC(calib = MPCCalibration(meanMPC = 0.25, inner_xtol = 1e-7))
#
#   r.beta, r.bbar, r.moments.meanMPC
#
# Returns `(; beta, bbar, qSav, qBorr, qGov, eq, moments, residuals, converged,
#            stalled, sweeps, nSolves, elapsedSeconds, calib, params)`.
# Marek Kapicka, 2026
#
# =============================================================================

# Reuses `solve_scalar`, `with_tee` and the solver entry points from the
# three-instrument file rather than copying them. Guarded so including several
# calibration files in one session does not redefine everything.
isdefined(@__MODULE__, :solve_scalar) ||
    include("calibrate_twoprice.jl")

using Dates
using Printf

"""
    MPCCalibration(; kwargs...)

Targets, brackets and search controls for the MPC calibration.

Only two moments are targeted. The wealth ratios and the negative-asset share
are reported by the result but not aimed at: with the prices given and both
instruments spoken for, nothing is left to move them.
"""
Base.@kwdef struct MPCCalibration
    # See the header for what this number is and what it is not. 0.408 is
    # Kaplan-Violante (2022) Table 1 column 5, the annual MPC of their one-asset
    # model calibrated to mean liquid wealth.
    meanMPC::Float64 = 0.408
    # Unchanged from calibrate_beta.jl: Kaplan and Violante (2014), Table III.
    # Ignored when `bbar_fixed` is set, since there is then no instrument left
    # to hit it with.
    trueBorrowingLimitToMeanLaborIncome::Float64 = 0.185

    # Hold the borrowing scale EXOGENOUS instead of calibrating it. `nothing`
    # calibrates bbar to the limit target, as calibrate_beta.jl does;
    # `bbar_fixed = 0.0` imposes a zero borrowing limit, so assets are
    # nonnegative and the model is a pure buffer-stock economy. Any other number
    # pins bbar there.
    #
    # Fixing it drops the bbar BLOCK, not just the instrument: with one
    # instrument there is one target, and the limit moment becomes a reported
    # output rather than something aimed at. Convergence is then judged on the
    # MPC residual alone.
    bbar_fixed::Union{Nothing,Float64} = nothing

    # Starting points. bbar_init is the same back-out as in calibrate_beta.jl,
    # since its block is identical. beta_init is the SETTINGS value.
    beta_init::Float64 = 0.960
    bbar_init::Float64 = -0.17707415

    # Brackets. A plain search bracket: the finite horizon has no divergence
    # boundary, so beta_max is not a ceiling the economics imposes and may sit
    # above qSav. The only hard limit is the model's own 0 <= beta < 1.
    #
    # The floor is LOWER than in calibrate_beta.jl. A high MPC target needs an
    # IMPATIENT household, so this search runs towards the bottom of the bracket
    # where the asset calibration runs towards the top, and 0.900 is not low
    # enough to bracket the liquid-wealth MPC targets.
    beta_min::Float64 = 0.800
    beta_max::Float64 = 0.995
    bbar_min::Float64 = -0.80
    bbar_max::Float64 = -0.01

    # Inner 1-D root finder (Roots.Brent, via solve_scalar)
    inner_xtol::Float64 = 1e-5
    inner_maxevals::Int = 40

    # Outer block-coordinate loop. More sweeps than the asset calibration
    # allows, because bbar moves the MPC as well as its own moment, so the two
    # blocks are not nearly orthogonal the way they are there.
    outer_max_sweeps::Int = 20
    # Looser than the asset calibration's 5e-4. The MPC is an ARC over a
    # discrete asset grid, so it is a step function of beta at the grid scale:
    # asking for more than about 1e-3 chases discretization, not economics.
    moment_tol::Float64 = 1e-3

    verbose::Bool = true
end

"""
    mpc_moments_from(eq)

The calibration statistics, named as the targets are. `moments_from` in the
three-instrument file carries only the four wealth moments, so this adds the MPC
block rather than widening a function three other drivers depend on.
"""
mpc_moments_from(eq) = (;
    meanMPC                             = eq.statistics.meanMPC,
    trueBorrowingLimitToMeanLaborIncome = eq.statistics.meanBorrowingLimitToMeanLaborIncome,
    # Reported, never targeted.
    meanAssetsToMeanLaborIncome         = eq.statistics.meanAssetsToMeanLaborIncome,
    medianAssetsToMeanLaborIncome       = eq.statistics.medianAssetsToMeanLaborIncome,
    shareNegativeLiquidAssets           = eq.statistics.shareNegativeLiquidAssets,
    meanMPCConditionalOnPositive        = eq.statistics.meanMPCConditionalOnPositive,
    shareMPCPositive                    = eq.statistics.shareMPCPositive,
    shareMPCNegative                    = eq.statistics.shareMPCNegative,
    mpcShockToMeanLaborIncome           = eq.statistics.mpcShockToMeanLaborIncome,
    shareMPCExtrapolated                = eq.statistics.shareMPCExtrapolated,
)

# Signed residuals.
mc_resid_beta(m, t) = m.meanMPC - t.meanMPC
mc_resid_bbar(m, t) = m.trueBorrowingLimitToMeanLaborIncome -
                      t.trueBorrowingLimitToMeanLaborIncome
mc_max_abs_resid(m, t) = t.bbar_fixed === nothing ?
    max(abs(mc_resid_beta(m, t)), abs(mc_resid_bbar(m, t))) :
    abs(mc_resid_beta(m, t))

"""
    calibrate_MPC(; calib, base_kwargs...)

Calibrate `(beta, bbar)` at GIVEN `qSav`, `qBorr` and `qGov`, matching the mean
impact MPC and the true borrowing limit. The wealth ratios and the share of
households with negative liquid assets are reported but not targeted.

`base_kwargs` are forwarded to `make_history_independent_params` on every
evaluation, so the prices are set there. `beta`, `bbar` and `verbose` cannot be
passed that way: the first two are this calibration's own instruments.
"""
function calibrate_MPC(;
        calib::MPCCalibration = MPCCalibration(),
        base_kwargs...)

    start_time = time()

    for k in (:beta, :bbar, :verbose)
        haskey(base_kwargs, k) &&
            error("`$(k)` cannot be passed via base_kwargs; it is set by the calibration")
    end
    0.0 < calib.meanMPC < 1.0 ||
        error("meanMPC target must lie strictly in (0, 1), got $(calib.meanMPC)")

    price_in_force(k) = haskey(base_kwargs, k) ? Float64(base_kwargs[k]) :
                                                 Float64(getproperty(SETTINGS, k))
    qSav_used  = price_in_force(:qSav)
    qBorr_used = price_in_force(:qBorr)
    qGov_used  = price_in_force(:qGov)
    J_used = haskey(base_kwargs, :J) ? Int(base_kwargs[:J]) : SETTINGS.J

    beta_lo = calib.beta_min
    beta_hi = calib.beta_max
    beta_lo < beta_hi || error(
        "empty beta bracket: beta_min = $(beta_lo) is not below beta_max = $(beta_hi)")
    beta_hi < 1.0 || error("beta_max must be below 1, got $(beta_hi)")

    x = [clamp(calib.beta_init, beta_lo,        beta_hi),
         calib.bbar_fixed === nothing ?
             clamp(calib.bbar_init, calib.bbar_min, calib.bbar_max) :
             Float64(calib.bbar_fixed)]
    calib.bbar_fixed === nothing || calib.bbar_fixed <= 0.0 ||
        error("bbar_fixed must be <= 0 (0 means a zero borrowing limit), " *
              "got $(calib.bbar_fixed)")

    cache = Dict{NTuple{2,Float64},Any}()
    n_solves = Ref(0)

    function eval_point(bt::Float64, bb::Float64)
        key = (bt, bb)
        haskey(cache, key) && return cache[key]
        # collect_distributions stays FALSE: meanMPC is a running mass-weighted
        # sum and is always accumulated. Only medianMPC needs the
        # per-observation vector, and nothing here targets it.
        p = make_history_independent_params(;
            base_kwargs..., beta = bt, bbar = bb,
            verbose = false, collect_distributions = false)
        n_solves[] += 1
        eq = solve_hi(p)
        cache[key] = (mpc_moments_from(eq), eq)
        return cache[key]
    end

    moments_at(x) = first(eval_point(x[1], x[2]))

    # Sweep order: the near-exact block first. `i` indexes x = [beta, bbar].
    beta_block = (i = 1, lo = beta_lo, hi = beta_hi,
                  resid = mc_resid_beta, name = "beta", target = calib.meanMPC)
    bbar_block = (i = 2, lo = calib.bbar_min, hi = calib.bbar_max,
                  resid = mc_resid_bbar, name = "bbar",
                  target = calib.trueBorrowingLimitToMeanLaborIncome)
    blocks = calib.bbar_fixed === nothing ? (bbar_block, beta_block) : (beta_block,)

    if calib.verbose
        println("\n=== MPC calibration targets ===")
        @printf("%-40s = %.8f\n", "average impact MPC", calib.meanMPC)
        if calib.bbar_fixed === nothing
            @printf("%-40s = %.8f\n", "true borrowing limit / mean labor income",
                    calib.trueBorrowingLimitToMeanLaborIncome)
        else
            @printf("bbar                                     : FIXED at %s (not calibrated)%s\n",
                    calib.bbar_fixed,
                    calib.bbar_fixed == 0.0 ? "  -- zero borrowing limit, assets >= 0" : "")
        end
        println("wealth ratios, negative share           : NOT TARGETED (reported only)")
        @printf("prices (GIVEN, not calibrated)           : qSav=%.6f qBorr=%.6f qGov=%.6f\n",
                qSav_used, qBorr_used, qGov_used)
        @printf("beta bracket                             : [%.6f, %.6f]  (MPC DECREASES in beta)\n",
                beta_lo, beta_hi)
        @printf("horizon                                  : J = %d (%d periods, j = 0..J)\n",
                J_used, J_used + 1)
        println()
        println("=== Calibration search ===")
        @printf("start:   beta=%.6f bbar=%.6f\n\n", x[1], x[2])
        @printf("%4s   %-10s  %-11s %-10s  %-10s  %-9s %s\n",
                "eval", "beta", "bbar", "meanMPC", "trueBL/LI", "mean a/LI", "max")
        flush(stdout)
    end

    function print_row(label, x, m, gap, note::String = "")
        @printf("%4s  %.8f  %.8f  %.8f  %.8f  %.8f %.0e%s\n",
                label, x[1], x[2], m.meanMPC,
                m.trueBorrowingLimitToMeanLaborIncome,
                m.meanAssetsToMeanLaborIncome, gap, note)
        flush(stdout)
    end

    function report_stall(blk, x)
        probe = copy(x)
        probe[blk.i] = blk.lo
        rlo = blk.resid(moments_at(probe), calib)
        probe[blk.i] = blk.hi
        rhi = blk.resid(moments_at(probe), calib)
        @printf("\nSTALLED: the %s block found no sign change in [%.6f, %.6f].\n",
                blk.name, blk.lo, blk.hi)
        @printf("  %s = %.6f  ->  residual % .6e\n", blk.name, blk.lo, rlo)
        @printf("  %s = %.6f  ->  residual % .6e\n", blk.name, blk.hi, rhi)
        @printf("  target = %.8f; attainable on this bracket: [%.8f, %.8f]\n",
                blk.target, blk.target + min(rlo, rhi), blk.target + max(rlo, rhi))
        if sign(rlo) == sign(rhi)
            println("  The target lies OUTSIDE that range. Widen the bracket, or the")
            println("  target is not attainable at these prices and this grid.")
            # The sign is the mirror of the asset calibration: an unattainably
            # HIGH MPC needs a LOWER beta_min, not a higher beta_max.
            if blk.name == "beta"
                rlo < 0.0 && println(
                    "  The MPC is below target even at beta_min: lower beta_min. " *
                    "The MPC DECREASES in beta, so impatience is what raises it.")
                rhi > 0.0 && println(
                    "  The MPC is above target even at beta_max: raise beta_max " *
                    "(it may sit above qSav in a finite horizon).")
            end
        end
        flush(stdout)
        return nothing
    end

    moments = moments_at(x)
    converged = mc_max_abs_resid(moments, calib) <= calib.moment_tol
    sweep = 0
    stalled = false
    calib.verbose && print_row("1", x, moments, mc_max_abs_resid(moments, calib))

    while !converged && !stalled && sweep < calib.outer_max_sweeps
        sweep += 1
        x_prev = copy(x)
        unbracketed = Any[]

        for blk in blocks
            probe = copy(x)
            f = function (z)
                probe[blk.i] = z
                return blk.resid(moments_at(probe), calib)
            end
            x[blk.i], br = solve_scalar(f, blk.lo, blk.hi;
                                        xtol = calib.inner_xtol,
                                        maxevals = calib.inner_maxevals)
            br || push!(unbracketed, blk)
        end

        moments = moments_at(x)
        gap = mc_max_abs_resid(moments, calib)
        calib.verbose && print_row(string(sweep + 1), x, moments, gap,
                                   isempty(unbracketed) ? "" : "  [unbracketed]")
        converged = gap <= calib.moment_tol

        if !converged && !isempty(unbracketed) && x == x_prev
            stalled = true
            calib.verbose && foreach(blk -> report_stall(blk, x), unbracketed)
        end
    end

    p_final = make_history_independent_params(;
        collect_distributions = false, base_kwargs..., beta = x[1], bbar = x[2])
    eq = solve_hi(p_final)
    n_solves[] += 1

    moments_final = mpc_moments_from(eq)
    residuals = (;
        meanMPC = mc_resid_beta(moments_final, calib),
        # NaN rather than a number when bbar is fixed: there is no target, so a
        # residual would invite being read as a miss.
        trueBorrowingLimitToMeanLaborIncome =
            calib.bbar_fixed === nothing ? mc_resid_bbar(moments_final, calib) : NaN,
    )

    result = (;
        beta = x[1], bbar = x[2],
        qSav = p_final.qSav, qBorr = p_final.qBorr, qGov = p_final.qGov,
        beta_bracket = (beta_lo, beta_hi),
        eq = eq, moments = moments_final, residuals = residuals,
        converged = converged, stalled = stalled, sweeps = sweep,
        nSolves = n_solves[], elapsedSeconds = time() - start_time,
        calib = calib, params = p_final,
    )

    calib.verbose && print_MPC_calibration_result(result)
    return result
end

"""
    print_MPC_calibration_result(result)

Report the calibrated discount factor and borrowing scale, the achieved fit, and
the wealth ratios the calibration did NOT target -- which are the interesting
output here, since matching an MPC and matching wealth are the tension the whole
Kaplan-Violante review is about.
"""
function print_MPC_calibration_result(result)
    t = result.calib
    m = result.moments
    r = result.residuals
    beta_lo, beta_hi = result.beta_bracket

    println("\n=== MPC calibration result ===")
    @printf("converged                = %s%s (%d sweeps, %d solves, %.1f s)\n",
            result.converged, result.stalled ? " [STALLED]" : "",
            result.sweeps, result.nSolves, result.elapsedSeconds)
    @printf("beta                       = %s\n", result.beta)
    @printf("bbar                       = %s\n", result.bbar)
    @printf("qSav / qBorr / qGov (given) = %.6f / %.6f / %.6f\n",
            result.qSav, result.qBorr, result.qGov)
    if result.beta <= beta_lo + 10 * eps(beta_lo)
        @printf("WARNING: beta is AT the bracket floor %.6f.\n", beta_lo)
        println("         A higher MPC needs MORE impatience, so lower beta_min and re-run.")
    elseif result.beta >= beta_hi - 10 * eps(beta_hi)
        @printf("WARNING: beta is AT beta_max = %.6f.\n", beta_hi)
        println("         A lower MPC needs MORE patience; raise beta_max and re-run.")
    end
    println()
    @printf("%-42s %12s %12s %11s\n", "moment", "model", "target", "residual")
    @printf("%-42s %12.8f %12.8f %11.2e\n", "average impact MPC",
            m.meanMPC, t.meanMPC, r.meanMPC)
    # With bbar fixed there is no instrument aimed at the limit, so printing it
    # as a target with a residual of -0.185 would report a failure that is not
    # one. It becomes a reported output, like the wealth ratios below.
    if t.bbar_fixed === nothing
        @printf("%-42s %12.8f %12.8f %11.2e\n", "true borrowing limit / mean labor income",
                m.trueBorrowingLimitToMeanLaborIncome,
                t.trueBorrowingLimitToMeanLaborIncome,
                r.trueBorrowingLimitToMeanLaborIncome)
    else
        @printf("%-42s %12.8f %12s %11s\n", "true borrowing limit / mean labor income",
                m.trueBorrowingLimitToMeanLaborIncome, "bbar fixed", "--")
    end
    println()
    # The wealth the MPC target implies. This is the point of the exercise: a
    # one-asset model that hits a high MPC does so by holding little wealth, and
    # how little is the number worth reading.
    @printf("%-42s %12.8f %12s %11s\n", "mean assets / mean labor income",
            m.meanAssetsToMeanLaborIncome, "not targeted", "--")
    @printf("%-42s %12.8f %12s %11s\n", "median assets / mean labor income",
            m.medianAssetsToMeanLaborIncome, "not targeted", "--")
    @printf("%-42s %12.8f %12s %11s\n", "share negative liquid assets",
            m.shareNegativeLiquidAssets, "not targeted", "--")
    println()
    # The MPC distribution behind the mean. With endogenous hours the mean is a
    # cancellation of two large groups, so the mean alone understates both.
    @printf("%-42s %12.8f\n", "  mean MPC | responders (mpc > 0)",
            m.meanMPCConditionalOnPositive)
    @printf("%-42s %12.6f / %.6f\n", "  share mpc positive / negative",
            m.shareMPCPositive, m.shareMPCNegative)
    # The drift described in the header: if this has moved away from the dollar
    # ratio mpc_shock was derived on, the target is no longer the MPC out of the
    # dollar amount intended.
    @printf("%-42s %12.6f   (re-derive mpc_shock if this moved)\n",
            "  windfall / mean labor income", m.mpcShockToMeanLaborIncome)
    if m.shareMPCExtrapolated > 1e-8
        @printf("%-42s %12.8f   [raise aMax]\n",
                "  share extrapolated above the grid", m.shareMPCExtrapolated)
    end
    println()
    one_price = result.qSav == result.qBorr == result.qGov
    gap = result.eq.outputPV - result.eq.consumptionPV -
          result.params.G / (1.0 - result.params.qGov)
    @printf("PV(Y) - PV(C) - PV(G)    = %.3e   %s\n", gap,
            one_price ? "(exact identity when q = qGov)" :
                        "(NOT the government budget: qSav/qBorr/qGov differ)")
    @printf("government budget residual = %.3e\n", result.eq.govBudgetResidual)
    flush(stdout)
    return nothing
end

# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
function MPC_calibration_log_path(calib::MPCCalibration;
                                  log_dir = joinpath(@__DIR__, "calibration_results"))
    s = SETTINGS
    stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")
    # The target is in the name: two runs differing only in the MPC aimed at are
    # different exercises and must not land in the same transcript.
    return joinpath(log_dir,
        "calib_MPC$(round(calib.meanMPC, digits = 4))_J$(s.J)_nA$(s.nA)_nZ$(s.nZ)" *
        "_nEps$(s.nEps)_nKappa$(s.nKappa)_$(stamp).txt")
end

if abspath(PROGRAM_FILE) == @__FILE__
    calib = MPCCalibration()
    log_path = MPC_calibration_log_path(calib)
    result = with_tee(log_path) do
        r = calibrate_MPC(calib = calib)
        print_equilibrium_summary(r.eq, r.params;
                                  title = "Final calibrated equilibrium",
                                  show_welfare = false)
        r
    end
    println("\ntranscript saved to ", log_path)
end
