# =============================================================================
# calibrate_beta.jl
#
# A TWO-INSTRUMENT, TWO-TARGET calibration in which the three prices are GIVEN
# and the discount factor does the work:
#
#       qSav, qBorr, qGov  fixed by the caller,
#       beta               calibrated to the asset moment,
#       bbar               calibrated to the borrowing limit.
#
# The share of households with negative liquid assets is NOT targeted -- with
# the prices fixed there is no qBorr left to move it.
#
# -----------------------------------------------------------------------------
# IDENTIFICATION
# -----------------------------------------------------------------------------
# Two blocks, swept in this order:
#
#   * bbar -> the borrowing-limit moment. bbar is the only parameter in the
#     numerator -bbar * E[exp(kappa + rho z)], so this block is very nearly
#     exact: one solve pins it down up to the second-order feedback through mean
#     labor income. It is swept first for that reason.
#
#   * beta -> the asset moment. A HIGHER beta is more patience and therefore
#     MORE saving, so the asset moment is INCREASING in beta. Note the sign: it
#     is the opposite of the q block in the one-price file, where a higher q is
#     a lower return and less saving.
#
# THE UPPER BRACKET IS NOT A FREE CHOICE. Asset demand diverges as beta*(1/qSav)
# approaches one, that is as beta approaches qSav from below: the precautionary
# motive stops being offset by impatience and the stationary distribution walks
# off the top of the grid. This is the same boundary the one-price file runs
# into from the other side, where q_min = 0.962 sits just above beta = 0.960.
# `beta_max` IS that ceiling, set directly. The only thing enforced is that it
# lies strictly below the qSav actually in force, so a beta_max carried over
# from a run at a different price is an error rather than a silent divergence.
# If the calibrated beta comes back sitting on beta_max, the asset target is not
# attainable at the given prices; that is information about the prices, not a
# failure of the search.
#
# HOW FAR BELOW qSav TO SET IT. The cross-section relaxes toward its stationary
# distribution on a timescale
#
#   1/(1 - beta/qSav) = qSav / (qSav - beta),
#
# so the horizon the ceiling asks for is qSav/(qSav - beta_max) periods, against
# a forward pass of maxAge. A beta_max within 0.002 of qSav wants roughly 500
# periods and at maxAge = 600 the cross-section provably has not settled -- that
# is `warn_if_unsettled`'s drift warning made structural rather than marginal.
# Keeping three relaxation times inside the horizon needs
#
#   qSav - beta_max  >=  3 * qSav / maxAge,
#
# which is beta_max <= 0.9851 at the SETTINGS qSav = 0.99, and beta_max <= 0.9787
# at qSav = 1/1.0167. The search header prints the implied relaxation time and
# flags it when it is short, so this is judged per run rather than guessed.
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   include("calibrate_beta.jl")
#
#   # prices at their SETTINGS values (qSav 0.99, qBorr 0.97, qGov 0.99):
#   r = calibrate_beta(nZ = 15, nEps = 11,
#                                              nKappa = 5, nA = 151)
#
#   # one price everywhere, so the government budget telescopes exactly:
#   r = calibrate_beta(nZ = 15, nEps = 11, nKappa = 5,
#                                              nA = 151, qSav = 0.99,
#                                              qBorr = 0.99, qGov = 0.99)
#
#   # median instead of mean, and a tighter inner solve:
#   r = calibrate_beta(
#           calib = BetaCalibration(asset_moment = :median,
#                                   inner_xtol = 1e-7))
#
#   r.beta, r.bbar, r.moments.shareNegativeLiquidAssets
#
# Returns `(; beta, bbar, qSav, qBorr, qGov, eq, moments, residuals, converged,
#            stalled, sweeps, nSolves, elapsedSeconds, calib, params)`.
# Marek Kapicka, 2026
#
# Passages on the choice of instrument, the beta ceiling and the access
# chain are in NOTES.md.
#
# =============================================================================

# Reuses `solve_scalar`, `moments_from` and the solver entry points from the
# three-instrument file rather than copying them. Guarded so including several
# calibration files in one session does not redefine everything.
isdefined(@__MODULE__, :solve_scalar) ||
    include("calibrate_twoprice.jl")

using Dates
using Printf

"""
    BetaCalibration(; kwargs...)

Targets, brackets and search controls for the beta calibration.

Only two moments are targeted. `shareNegativeLiquidAssets` has deliberately no
field here: it is reported by the result but not aimed at, because the prices
are inputs and nothing is left to move it.
"""
Base.@kwdef struct BetaCalibration
    # Targets. Kaplan and Violante (2014), Table III, on a 2001 SCF cross-section
    # of households aged 22-59 with the top 5% by net worth dropped: mean net
    # LIQUID wealth over mean earnings-plus-benefits is 31,001/52,745 = 0.588,
    # and the median counterpart is 2,629/52,745 = 0.0498, which is the figure
    # carried by paper/Bewley.tex and by the three-instrument file. (Both read
    # 2,269 / 0.043 until 2026-09-23; that median was a transposition of the
    # 2,629 printed in the table, whose every other figure matched.)
    #
    # Do NOT put 62,442/52,745 = 1.184 here. That is median NET WORTH (it is
    # labelled `tgt_networthmedian_to_ymean` in code/old/.../parameters.jl and
    # sits commented out in paper/Bewley.tex). This is a one-asset liquid-wealth
    # model whose borrowing-limit target, 0.185, is itself a liquid object;
    # asking it for a net-worth median drives the search into the divergence
    # boundary and it will stall against the bracket cap.
    medianAssetsToMeanLaborIncome::Float64       = 0.0498
    meanAssetsToMeanLaborIncome::Float64         = 0.588
    # Ignored when `bbar_fixed` is set: there is then no instrument left to hit
    # it with.
    trueBorrowingLimitToMeanLaborIncome::Float64 = 0.185

    # Hold the borrowing scale EXOGENOUS instead of calibrating it. `nothing`
    # calibrates bbar to the limit target; `bbar_fixed = 0.0` imposes a ZERO
    # borrowing limit, so assets are nonnegative and the model is a pure
    # buffer-stock economy. Any other number pins bbar there.
    #
    # Fixing it drops the bbar BLOCK, not just the instrument: with one
    # instrument there is one target, so the limit moment becomes a reported
    # output and convergence is judged on the asset residual alone.
    bbar_fixed::Union{Nothing,Float64} = nothing
    asset_moment::Symbol                         = :mean   # :mean or :median

    # Starting points. bbar_init is close to its root by construction: the
    # borrowing-limit block is linear in bbar, and -0.2 * 0.185/0.20895 was the
    # one-solve back-out at the production grids (retaken when the target moved
    # from 0.180 to 0.185; at 0.180 it read -0.17228836). beta_init is the
    # SETTINGS value, which is the natural neutral start.
    beta_init::Float64 = 0.960
    bbar_init::Float64 = -0.17707415

    # Brackets. beta_max is the ceiling, used exactly as given. It must sit
    # strictly below the qSav in force; the header's relaxation-time rule says
    # how far below. The default is sized for the SETTINGS qSav of 0.99, where
    # it asks for 198 periods against maxAge = 600. A lower qSav needs a lower
    # beta_max and says so rather than being capped silently.
    beta_min::Float64 = 0.900
    beta_max::Float64 = 0.985
    bbar_min::Float64 = -0.80
    bbar_max::Float64 = -0.01

    # Inner 1-D root finder (Roots.Brent, via solve_scalar)
    inner_xtol::Float64 = 1e-5
    inner_maxevals::Int = 40

    # Outer block-coordinate loop. The sweep budget is a backstop only: a block
    # that cannot bracket its target is caught by the stall guard, which stops
    # on the first repeated instrument vector rather than burning the budget.
    outer_max_sweeps::Int = 12
    moment_tol::Float64   = 5e-4

    verbose::Bool = true
end

# Field selectors, mirroring the other two calibration files but dispatching on
# this struct. Duplicated rather than shared because the originals are typed on
# ::CalibrationParams and ::OnePriceCalibration.
bc_asset_field(c::BetaCalibration) =
    c.asset_moment === :median ? :medianAssetsToMeanLaborIncome :
                                 :meanAssetsToMeanLaborIncome
bc_asset_ratio(m, c::BetaCalibration)  = getproperty(m, bc_asset_field(c))
bc_asset_target(c::BetaCalibration)    = getproperty(c, bc_asset_field(c))
bc_asset_label(c::BetaCalibration) =
    c.asset_moment === :median ? "median assets / mean labor income" :
                                 "mean assets / mean labor income"
bc_asset_label_short(c::BetaCalibration) =
    c.asset_moment === :median ? "median/LI" : "mean/LI"

# Signed residuals. There is no third block: the negative-asset share is
# reported by `moments_from` and printed, but never driven to a target.
bc_resid_beta(m, t) = bc_asset_ratio(m, t) - bc_asset_target(t)
bc_resid_bbar(m, t) = m.trueBorrowingLimitToMeanLaborIncome -
                      t.trueBorrowingLimitToMeanLaborIncome
bc_max_abs_resid(m, t) = t.bbar_fixed === nothing ?
    max(abs(bc_resid_beta(m, t)), abs(bc_resid_bbar(m, t))) :
    abs(bc_resid_beta(m, t))

"""
    calibrate_beta(; calib, base_kwargs...)

Calibrate `(beta, bbar)` at GIVEN `qSav`, `qBorr` and `qGov`, matching the asset
moment and the true borrowing limit. The share of households with negative
liquid assets is reported but not targeted.

`base_kwargs` are forwarded to `make_history_independent_params` on every
evaluation, so the prices are set there -- `qSav`, `qBorr` and `qGov` are inputs
to this calibration, unlike in the other two files where they are instruments.
`beta`, `bbar` and `verbose` cannot be passed that way: the first two are this
calibration's own instruments and passing them would silently conflict.
"""
function calibrate_beta(;
        calib::BetaCalibration = BetaCalibration(),
        base_kwargs...)

    start_time = time()

    calib.asset_moment in (:median, :mean) ||
        error("calib.asset_moment must be :median or :mean")
    for k in (:beta, :bbar, :verbose)
        haskey(base_kwargs, k) &&
            error("`$(k)` cannot be passed via base_kwargs; it is set by the calibration")
    end

    # The prices in force. They are not instruments here, so they come either
    # from the caller or from SETTINGS, and qSav sets the divergence boundary.
    price_in_force(k) = haskey(base_kwargs, k) ? Float64(base_kwargs[k]) :
                                                 Float64(getproperty(SETTINGS, k))
    qSav_used  = price_in_force(:qSav)
    qBorr_used = price_in_force(:qBorr)
    qGov_used  = price_in_force(:qGov)
    # The forward-pass length, needed to judge whether the ceiling can settle.
    maxAge_used = haskey(base_kwargs, :maxAge) ? Int(base_kwargs[:maxAge]) :
                                                 SETTINGS.maxAge

    # The bracket, used as given. asset demand diverges as beta -> qSav, so the
    # ceiling has to sit below qSav; how far below is the caller's call, and the
    # relaxation time it implies is reported with the search header.
    beta_lo = calib.beta_min
    beta_hi = calib.beta_max
    beta_lo < beta_hi || error(
        "empty beta bracket: beta_min = $(beta_lo) is not below beta_max = $(beta_hi)")
    beta_hi < 1.0 || error("beta_max must be below 1, got $(beta_hi)")
    beta_hi < qSav_used || error(
        "beta_max = $(beta_hi) is not below qSav = $(qSav_used): asset demand " *
        "diverges as beta -> qSav. Lower beta_max -- see the header for the " *
        "relaxation-time rule, which gives beta_max <= " *
        "$(round(qSav_used * (1 - 3 / maxAge_used), digits = 4)) here.")
    relax_at_ceiling = qSav_used / (qSav_used - beta_hi)

    x = [clamp(calib.beta_init, beta_lo,          beta_hi),
         calib.bbar_fixed === nothing ?
             clamp(calib.bbar_init, calib.bbar_min, calib.bbar_max) :
             Float64(calib.bbar_fixed)]
    calib.bbar_fixed === nothing || calib.bbar_fixed <= 0.0 ||
        error("bbar_fixed must be <= 0 (0 means a zero borrowing limit), " *
              "got $(calib.bbar_fixed)")

    # Memoized on the instrument PAIR: Brent re-probes bracket endpoints, and
    # the sweep-end evaluation repeats the point the last block just solved.
    cache = Dict{NTuple{2,Float64},Any}()
    n_solves = Ref(0)

    function eval_point(bt::Float64, bb::Float64)
        key = (bt, bb)
        haskey(cache, key) && return cache[key]
        # The single line that defines this file: beta is the instrument, the
        # prices are whatever the caller passed.
        p = make_history_independent_params(;
            base_kwargs..., beta = bt, bbar = bb,
            verbose = false, collect_distributions = false)
        n_solves[] += 1
        eq = solve_hi(p)
        cache[key] = (moments_from(eq), eq)
        return cache[key]
    end

    moments_at(x) = first(eval_point(x[1], x[2]))

    # Sweep order: the near-exact block first. `i` indexes x = [beta, bbar].
    # Built here rather than as a const, because the beta bounds are only known
    # once qSav is.
    beta_block = (i = 1, lo = beta_lo, hi = beta_hi,
                  resid = bc_resid_beta, name = "beta", target = bc_asset_target(calib))
    bbar_block = (i = 2, lo = calib.bbar_min, hi = calib.bbar_max,
                  resid = bc_resid_bbar, name = "bbar",
                  target = calib.trueBorrowingLimitToMeanLaborIncome)
    blocks = calib.bbar_fixed === nothing ? (bbar_block, beta_block) : (beta_block,)

    if calib.verbose
        println("\n=== Beta calibration targets ===")
        @printf("%-40s = %.8f\n", bc_asset_label(calib), bc_asset_target(calib))
        if calib.bbar_fixed === nothing
            @printf("%-40s = %.8f\n", "true borrowing limit / mean labor income",
                    calib.trueBorrowingLimitToMeanLaborIncome)
        else
            @printf("bbar                                     : FIXED at %s (not calibrated)%s\n",
                    calib.bbar_fixed,
                    calib.bbar_fixed == 0.0 ? "  -- zero borrowing limit, assets >= 0" : "")
        end
        println("share negative liquid assets             : NOT TARGETED (reported only)")
        @printf("prices (GIVEN, not calibrated)           : qSav=%.6f qBorr=%.6f qGov=%.6f\n",
                qSav_used, qBorr_used, qGov_used)
        if qSav_used == qBorr_used == qGov_used
            println("                                         : one price everywhere, so PV(T) = G telescopes exactly")
        else
            println("                                         : prices differ, so PV(Y-C) = G is NOT the government budget (see header)")
        end
        @printf("beta bracket                             : [%.6f, %.6f]  (beta_max is %.6f below qSav)\n",
                beta_lo, beta_hi, qSav_used - beta_hi)
        @printf("relaxation time at the ceiling           : %.0f periods against maxAge = %d%s\n",
                relax_at_ceiling, maxAge_used,
                3 * relax_at_ceiling > maxAge_used ?
                    "   [under 3 relaxation times: expect a drift warning]" : "")
        println()
        println("=== Calibration search ===")
        @printf("start:   beta=%.6f bbar=%.6f\n\n", x[1], x[2])
        @printf("%4s   %-10s  %-11s %-10s  %-10s  %-9s %s\n",
                "eval", "beta", "bbar", bc_asset_label_short(calib),
                "trueBL/LI", "neg share", "max")
        flush(stdout)
    end

    function print_row(label, x, m, gap, note::String = "")
        @printf("%4s  %.8f  %.8f  %.8f  %.8f  %.8f %.0e%s\n",
                label, x[1], x[2], bc_asset_ratio(m, calib),
                m.trueBorrowingLimitToMeanLaborIncome,
                m.shareNegativeLiquidAssets, gap, note)
        flush(stdout)
    end

    # Diagnostic for a block that found no sign change. Both endpoints are
    # already in the cache from `solve_scalar`'s scan, so this costs no solves.
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
            blk.name == "beta" && rhi < 0.0 && println(
                "  beta is at beta_max; a higher asset moment needs a higher " *
                "beta_max (kept below qSav), or a higher qSav.")
        end
        flush(stdout)
        return nothing
    end

    moments = moments_at(x)
    converged = bc_max_abs_resid(moments, calib) <= calib.moment_tol
    sweep = 0
    stalled = false
    calib.verbose && print_row("1", x, moments, bc_max_abs_resid(moments, calib))

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

        moments = moments_at(x)          # cache hit from the last block
        gap = bc_max_abs_resid(moments, calib)
        calib.verbose && print_row(string(sweep + 1), x, moments, gap,
                                   isempty(unbracketed) ? "" : "  [unbracketed]")
        converged = gap <= calib.moment_tol

        # Stall guard. A block that cannot bracket its target returns the
        # endpoint with the smallest |residual|; if the whole instrument vector
        # then repeats, every later sweep is a cache hit that prints the same
        # row. Stop and say which block failed and why, instead of running the
        # sweep budget out in silence.
        if !converged && !isempty(unbracketed) && x == x_prev
            stalled = true
            calib.verbose && foreach(blk -> report_stall(blk, x), unbracketed)
        end
    end

    # Re-solved at the calibrated point with the caller's verbosity.
    # collect_distributions defaults to FALSE for the calibration, including
    # this final re-solve: nothing the calibration targets reads the
    # per-observation vectors -- the asset histogram behind the median is a
    # separate field that is always accumulated -- while they are the
    # allocation that OOM-kills large jobs. It sits BEFORE the splat, so
    # `base_kwargs` can still turn it back on when the calibrated equilibrium
    # is wanted for plotting.
    p_final = make_history_independent_params(;
        collect_distributions = false, base_kwargs..., beta = x[1], bbar = x[2])
    eq = solve_hi(p_final)
    n_solves[] += 1

    moments_final = moments_from(eq)
    residuals = (;
        assetsToMeanLaborIncome             = bc_resid_beta(moments_final, calib),
        # NaN rather than a number when bbar is fixed: with no target, a
        # residual would invite being read as a miss.
        trueBorrowingLimitToMeanLaborIncome =
            calib.bbar_fixed === nothing ? bc_resid_bbar(moments_final, calib) : NaN,
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

    calib.verbose && print_beta_calibration_result(result)
    return result
end

"""
    print_beta_calibration_result(result)

Report the calibrated discount factor and borrowing scale, the achieved fit, the
untargeted borrowing share, and the government-budget residual. The budget check
is only an identity when qSav = qBorr = qGov, so it is labelled accordingly.
"""
function print_beta_calibration_result(result)
    t = result.calib
    m = result.moments
    r = result.residuals
    beta_lo, beta_hi = result.beta_bracket

    println("\n=== Beta calibration result ===")
    @printf("converged                = %s%s (%d sweeps, %d solves, %.1f s)\n",
            result.converged, result.stalled ? " [STALLED]" : "",
            result.sweeps, result.nSolves, result.elapsedSeconds)
    @printf("beta                       = %s\n", result.beta)
    @printf("bbar                       = %s\n", result.bbar)
    @printf("qSav / qBorr / qGov (given) = %.6f / %.6f / %.6f\n",
            result.qSav, result.qBorr, result.qGov)
    # A beta sitting on the ceiling means the asset target was not attainable at
    # these prices, which is worth saying out loud rather than leaving in the
    # eighth decimal of the residual.
    if result.beta >= beta_hi - 10 * eps(beta_hi)
        @printf("WARNING: beta is AT beta_max = %.6f (qSav = %.6f).\n",
                beta_hi, result.qSav)
        println("         The asset target is not attainable at these prices; beta cannot")
        println("         be raised further without asset demand diverging.")
    elseif result.beta <= beta_lo + 10 * eps(beta_lo)
        @printf("WARNING: beta is AT the bracket floor %.6f.\n", beta_lo)
    end
    println()
    @printf("%-40s %12s %12s %11s\n", "moment", "model", "target", "residual")
    @printf("%-40s %12.8f %12.8f %11.2e\n", bc_asset_label(t),
            bc_asset_ratio(m, t), bc_asset_target(t), r.assetsToMeanLaborIncome)
    # With bbar fixed nothing aims at the limit, so scoring it would report a
    # failure that is not one.
    if t.bbar_fixed === nothing
        @printf("%-40s %12.8f %12.8f %11.2e\n", "true borrowing limit / mean labor income",
                m.trueBorrowingLimitToMeanLaborIncome,
                t.trueBorrowingLimitToMeanLaborIncome,
                r.trueBorrowingLimitToMeanLaborIncome)
    else
        @printf("%-40s %12.8f %12s %11s\n", "true borrowing limit / mean labor income",
                m.trueBorrowingLimitToMeanLaborIncome, "bbar fixed", "--")
    end
    @printf("%-40s %12.8f %12s %11s\n", "share negative liquid assets",
            m.shareNegativeLiquidAssets, "not targeted", "--")
    @printf("%-40s %12.8f %12s %11s\n",
            t.asset_moment === :median ? "mean assets / mean labor income" :
                                         "median assets / mean labor income",
            t.asset_moment === :median ? m.meanAssetsToMeanLaborIncome :
                                         m.medianAssetsToMeanLaborIncome,
            "not targeted", "--")
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
# Deliberately does NOT reuse `calibration_log_path` from the three-instrument
# file: that one interpolates `SETTINGS.J`, which the infinite-horizon SETTINGS
# does not define, so it throws when run as a script.
function beta_calibration_log_path(calib::BetaCalibration;
                                   log_dir = joinpath(@__DIR__, "calibration_results"))
    s = SETTINGS
    stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")
    return joinpath(log_dir,
        "calib_beta_$(calib.asset_moment)_nA$(s.nA)_nZ$(s.nZ)" *
        "_nEps$(s.nEps)_nKappa$(s.nKappa)_$(stamp).txt")
end

if abspath(PROGRAM_FILE) == @__FILE__
    calib = BetaCalibration()
    log_path = beta_calibration_log_path(calib)
    result = with_tee(log_path) do
        r = calibrate_beta(calib = calib)
        print_equilibrium_summary(r.eq, r.params;
                                  title = "Final calibrated equilibrium",
                                  show_welfare = false)
        r
    end
    println("\ntranscript saved to ", log_path)
end
