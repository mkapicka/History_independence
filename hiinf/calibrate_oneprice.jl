# =============================================================================
# calibrate_oneprice.jl
#
# A TWO-INSTRUMENT, TWO-TARGET calibration in which households face a single
# intertemporal price:
#
#       qSav = qBorr = qGov = q,
#
# with q calibrated to the asset moment and bbar to the borrowing limit. The
# share of households with negative liquid assets is NOT targeted -- with one
# price there is no qBorr left to hit it.
#
# -----------------------------------------------------------------------------
# WHY ONE PRICE
# -----------------------------------------------------------------------------
# In the three-instrument calibration the household's saving and borrowing
# prices both differ from the rate the government discounts at, and that wedge
# does not net out. Substituting the household budget into the resource
# constraint and discounting at qGov,
#
#   PV(Y - C) = PV(T) - A_1 - sum_j qGov^(j-1) (qGov - q_j) A_(j+1),
#
# so the asset terms telescope to the initial position A_1 ONLY when q_j = qGov
# at every state. Otherwise the solver, which drives PV(Y - C) to G, is not
# imposing PV(T) = G: it is imposing PV(T) = G plus a transfer to households
# that no agent in the model finances. Measured at the three-instrument
# calibration (qSav = 0.9807, qBorr = 0.9895, qGov = 0.99), that residual claim
# was worth 0.45 in present value, about 0.5 percent of lifetime output -- the
# same order as the welfare gains from history dependence the project measures,
# and five orders above the 1e-8 budget tolerance.
#
# Setting q = qGov removes it identically. The telescoping is then exact, and
# with a_0 = 0 the condition the solver imposes IS the government budget:
#
#   PV(Y - C) = G   <=>   PV(T) = G.
#
# The price is calibrated rather than fixed at 0.99 so the asset moment is still
# matched: with only one instrument left for the asset distribution, q has to do
# the job qSav did before.
#
# -----------------------------------------------------------------------------
# WHAT IS GIVEN UP
# -----------------------------------------------------------------------------
# The borrowing share is no longer matched, and should be reported rather than
# assumed. One price means no spread between lending and borrowing, so the model
# has nothing left to generate the observed mass of households at negative
# liquid wealth beyond what the borrowing limit and the shock process imply. In
# the three-instrument calibration that moment was 0.260 by construction; here
# it comes out wherever it comes out, and a large miss is informative about the
# one-price assumption rather than a failure of the search.
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
#   * q    -> the asset moment. q is the price of one unit of next-period
#     assets, so the gross return is 1/q and a HIGHER q means a LOWER return and
#     LESS saving. Measured at reduced grids with the borrowing limit held
#     fixed: q = 0.99 gave mean assets / mean labor income = 0.296, q = 0.97
#     gave 1.184, q = 0.95 gave 4.273. The moment is therefore DECREASING in q,
#     which is the opposite of what a comment in the three-instrument file
#     asserts; the numbers above are what this bracket is set from.
#
# The q bracket stops well above beta = 0.96. As q falls towards beta the return
# 1/q approaches 1/beta, the precautionary motive stops being offset by
# impatience, and asset demand diverges; at q = 0.93 the lambda solve failed
# outright and mean assets came back non-monotone. Keeping q_min at 0.962 stays
# inside the region where the asset moment is monotone in q.
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   include("calibrate_oneprice.jl")
#   r = calibrate_oneprice(nZ = 15, nEps = 11,
#                                                  nKappa = 5, nA = 151)
#   r.q, r.bbar, r.moments.shareNegativeLiquidAssets
#
#   # median instead of mean, and a tighter inner solve:
#   r = calibrate_oneprice(
#           calib = OnePriceCalibration(asset_moment = :median,
#                                       inner_xtol = 1e-7))
#
# Returns `(; q, bbar, eq, moments, residuals, converged, sweeps, nSolves,
#            elapsedSeconds, calib, params)`.
# Marek Kapicka, 2026
# =============================================================================

# Reuses `solve_scalar`, `moments_from` and the solver entry points from the
# three-instrument file rather than copying them. Guarded so including both in
# one session does not redefine everything.
isdefined(@__MODULE__, :solve_scalar) ||
    include("calibrate_twoprice.jl")

using Printf

"""
    OnePriceCalibration(; kwargs...)

Targets, brackets and search controls for the one-price calibration.

Only two moments are targeted. `shareNegativeLiquidAssets` has deliberately no
field here: it is reported by the result but not aimed at, because with
`qSav = qBorr` there is no instrument left to move it.
"""
Base.@kwdef struct OnePriceCalibration
    # Targets. Kaplan and Violante (2014), Table III: mean net liquid wealth over
    # mean earnings-plus-benefits, 31,001/52,745 = 0.588, on a 2001 SCF
    # cross-section of households aged 22-59 with the top 5% by net worth
    # dropped. The median counterpart from the same table is 2,629/52,745 =
    # 0.0498. The other calibration files carried 0.043 here, a transposition
    # of the 2,629; they were brought into line on 2026-09-23.
    medianAssetsToMeanLaborIncome::Float64       = 0.0498
    meanAssetsToMeanLaborIncome::Float64         = 0.588
    trueBorrowingLimitToMeanLaborIncome::Float64 = 0.185
    asset_moment::Symbol                         = :mean   # :mean or :median

    # Starting points. bbar_init is close to its root by construction: the
    # borrowing-limit block is linear in bbar, and -0.2 * 0.185/0.20895 was the
    # one-solve back-out at the production grids (retaken when the target moved
    # from 0.180 to 0.185; at 0.180 it read -0.17228836).
    q_init::Float64    = 0.985
    bbar_init::Float64 = -0.17707415

    # Brackets. See the identification note in the header for why q_min sits
    # above beta rather than at the 0.900 the three-instrument file uses.
    q_min::Float64    = 0.950
    q_max::Float64    = 0.999
    bbar_min::Float64 = -0.80
    bbar_max::Float64 = -0.01

    # Inner 1-D root finder (Roots.Brent, via solve_scalar)
    inner_xtol::Float64 = 1e-5
    inner_maxevals::Int = 40

    # Outer block-coordinate loop
    outer_max_sweeps::Int = 12
    moment_tol::Float64   = 5e-4

    verbose::Bool = true
end

# Field selectors, mirroring the three-instrument file but dispatching on this
# struct. Duplicated rather than shared because the originals are typed on
# ::CalibrationParams.
op_asset_field(c::OnePriceCalibration) =
    c.asset_moment === :median ? :medianAssetsToMeanLaborIncome :
                                 :meanAssetsToMeanLaborIncome
op_asset_ratio(m, c::OnePriceCalibration)  = getproperty(m, op_asset_field(c))
op_asset_target(c::OnePriceCalibration)    = getproperty(c, op_asset_field(c))
op_asset_label(c::OnePriceCalibration) =
    c.asset_moment === :median ? "median assets / mean labor income" :
                                 "mean assets / mean labor income"
op_asset_label_short(c::OnePriceCalibration) =
    c.asset_moment === :median ? "median/LI" : "mean/LI"

# Signed residuals. There is no third block: the negative-asset share is
# reported by `moments_from` and printed, but never driven to a target.
op_resid_q(m, t)    = op_asset_ratio(m, t) - op_asset_target(t)
op_resid_bbar(m, t) = m.trueBorrowingLimitToMeanLaborIncome -
                      t.trueBorrowingLimitToMeanLaborIncome
op_max_abs_resid(m, t) = max(abs(op_resid_q(m, t)), abs(op_resid_bbar(m, t)))

# Sweep order: the near-exact block first. `i` indexes x = [q, bbar].
const OP_BLOCKS = (
    (i = 2, lo = :bbar_min, hi = :bbar_max, resid = op_resid_bbar),
    (i = 1, lo = :q_min,    hi = :q_max,    resid = op_resid_q),
)

"""
    calibrate_oneprice(; calib, base_kwargs...)

Calibrate `(q, bbar)` with `qSav = qBorr = qGov = q`, matching the asset moment
and the borrowing limit. The share of households with negative liquid assets is
reported but not targeted.

`base_kwargs` are forwarded to `make_history_independent_params` on every
evaluation, so any other setting can be overridden. `qSav`, `qBorr`, `qGov`,
`bbar` and `verbose` cannot be passed that way -- the first four are the
calibration's own instruments and passing them would silently conflict.
"""
function calibrate_oneprice(;
        calib::OnePriceCalibration = OnePriceCalibration(),
        base_kwargs...)

    start_time = time()

    calib.asset_moment in (:median, :mean) ||
        error("calib.asset_moment must be :median or :mean")
    # qGov joins the reserved list here: it is tied to q, which is exactly what
    # distinguishes this calibration from the three-instrument one.
    for k in (:qSav, :qBorr, :qGov, :bbar, :verbose)
        haskey(base_kwargs, k) &&
            error("`$(k)` cannot be passed via base_kwargs; it is set by the calibration")
    end

    x = [clamp(calib.q_init,    calib.q_min,    calib.q_max),
         clamp(calib.bbar_init, calib.bbar_min, calib.bbar_max)]

    # Memoized on the instrument PAIR: Brent re-probes bracket endpoints, and
    # the sweep-end evaluation repeats the point the last block just solved.
    cache = Dict{NTuple{2,Float64},Any}()
    n_solves = Ref(0)

    function eval_point(q::Float64, bb::Float64)
        key = (q, bb)
        haskey(cache, key) && return cache[key]
        # The single line that defines this file: one price everywhere.
        p = make_history_independent_params(;
            base_kwargs..., qSav = q, qBorr = q, qGov = q, bbar = bb,
            verbose = false, collect_distributions = false)
        n_solves[] += 1
        eq = solve_hi(p)
        cache[key] = (moments_from(eq), eq)
        return cache[key]
    end

    moments_at(x) = first(eval_point(x[1], x[2]))

    if calib.verbose
        println("\n=== One-price calibration targets ===")
        @printf("%-40s = %.8f\n", op_asset_label(calib), op_asset_target(calib))
        @printf("%-40s = %.8f\n", "true borrowing limit / mean labor income",
                calib.trueBorrowingLimitToMeanLaborIncome)
        println("share negative liquid assets             : NOT TARGETED (reported only)")
        println("prices                                   : qSav = qBorr = qGov = q")
        println()
        println("=== Calibration search ===")
        @printf("start:   q=%.6f bbar=%.6f\n\n", x[1], x[2])
        @printf("%4s   %-10s  %-11s %-10s  %-10s  %-9s %s\n",
                "eval", "q", "bbar", op_asset_label_short(calib),
                "trueBL/LI", "neg share", "max")
        flush(stdout)
    end

    function print_row(label, x, m, gap, note::String = "")
        @printf("%4s  %.8f  %.8f  %.8f  %.8f  %.8f %.0e%s\n",
                label, x[1], x[2], op_asset_ratio(m, calib),
                m.trueBorrowingLimitToMeanLaborIncome,
                m.shareNegativeLiquidAssets, gap, note)
        flush(stdout)
    end

    moments = moments_at(x)
    converged = op_max_abs_resid(moments, calib) <= calib.moment_tol
    sweep = 0
    calib.verbose && print_row("1", x, moments, op_max_abs_resid(moments, calib))

    while !converged && sweep < calib.outer_max_sweeps
        sweep += 1
        bracketed = true
        for blk in OP_BLOCKS
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
        moments = moments_at(x)          # cache hit from the last block
        gap = op_max_abs_resid(moments, calib)
        calib.verbose && print_row(string(sweep + 1), x, moments, gap,
                                   bracketed ? "" : "  [unbracketed]")
        converged = gap <= calib.moment_tol
    end

    # Re-solved at the calibrated point with the caller's verbosity.
    p_final = make_history_independent_params(;
        base_kwargs..., qSav = x[1], qBorr = x[1], qGov = x[1], bbar = x[2])
    eq = solve_hi(p_final)
    n_solves[] += 1

    moments_final = moments_from(eq)
    residuals = (;
        assetsToMeanLaborIncome             = op_resid_q(moments_final, calib),
        trueBorrowingLimitToMeanLaborIncome = op_resid_bbar(moments_final, calib),
    )

    result = (;
        q = x[1], bbar = x[2],
        eq = eq, moments = moments_final, residuals = residuals,
        converged = converged, sweeps = sweep, nSolves = n_solves[],
        elapsedSeconds = time() - start_time,
        calib = calib, params = p_final,
    )

    calib.verbose && print_oneprice_calibration_result(result)
    return result
end

"""
    print_oneprice_calibration_result(result)

Report the calibrated price and borrowing scale, the achieved fit, and the
untargeted borrowing share. Also reports PV(Y) - PV(C) - G, which is the point
of the one-price restriction: with q = qGov the resource constraint and the
government budget are the same equation, so this should be at solver tolerance
rather than the ~0.5 percent of lifetime output the three-price calibration
leaves unaccounted.
"""
function print_oneprice_calibration_result(result)
    t = result.calib
    m = result.moments
    r = result.residuals

    println("\n=== One-price calibration result ===")
    @printf("converged                = %s (%d sweeps, %d solves, %.1f s)\n",
            result.converged, result.sweeps, result.nSolves, result.elapsedSeconds)
    @printf("q  (= qSav = qBorr = qGov) = %s\n", result.q)
    @printf("bbar                       = %s\n", result.bbar)
    println()
    @printf("%-40s %12s %12s %11s\n", "moment", "model", "target", "residual")
    @printf("%-40s %12.8f %12.8f %11.2e\n", op_asset_label(t),
            op_asset_ratio(m, t), op_asset_target(t), r.assetsToMeanLaborIncome)
    @printf("%-40s %12.8f %12.8f %11.2e\n", "true borrowing limit / mean labor income",
            m.trueBorrowingLimitToMeanLaborIncome,
            t.trueBorrowingLimitToMeanLaborIncome,
            r.trueBorrowingLimitToMeanLaborIncome)
    @printf("%-40s %12.8f %12s %11s\n", "share negative liquid assets",
            m.shareNegativeLiquidAssets, "not targeted", "--")
    println()
    # With one price the telescoping in the resource constraint is exact, so
    # this is the government budget itself and should sit at solver tolerance.
    gap = result.eq.outputPV - result.eq.consumptionPV -
          result.params.G / (1.0 - result.params.qGov)
    @printf("PV(Y) - PV(C) - PV(G)    = %.3e   (exact identity when q = qGov)\n", gap)
    @printf("government budget residual = %.3e\n", result.eq.govBudgetResidual)
    flush(stdout)
    return nothing
end
