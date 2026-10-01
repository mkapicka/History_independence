# =============================================================================
# solve_history_dependent_tax.jl
#
# Marek Kapicka, 2026
#
# Finite-horizon Bewley economy with a history-dependent tax system
# (Section 1 of Bewley.tex). Budget constraint:
#
#   c + q(a') a' <= lambda * exp( pow * (z + eps + kappa
#                                        + alpha*s1 + (1-alpha)*s2) ) * h^pow + a,
#
# pow = (1 - tau) * theta0, and past-income stocks
#
#   s1' = mu1 * (z + eps + kappa + ln h + s1),
#   s2' = mu2 * (z + eps + kappa + ln h + s2),
#
# with theta0 implied by the finite-horizon promise-keeping restriction
#   sum_{s=0}^{J} beta^s theta_s = 1,  theta_s = theta0*(alpha*mu1^s + (1-alpha)*mu2^s),
# as build_theta imposes in the no-savings code. alpha is a free setting: a
# number, or :paper for (rho - mu1)/(mu2 - mu1). theta0 is derived and cannot
# be set. alpha lies in [0, 1] iff mu1 <= rho <= mu2; outside that the mixture
# is signed, which warns but is not rejected.
#
# mu1 = mu2 = 0 gives alpha = 1, theta0 = 1 and s1 = s2 = 0, reproducing the
# history-independent model exactly (use nS1 = nS2 = 1).
#
# -----------------------------------------------------------------------------
# Wrapped in its own module so it can be loaded beside the history-independent
# code in one session; the shared infrastructure comes from the BewleyCommon
# package at ../common. Exported API:
#
#   HDParams, HD_SETTINGS, make_history_dependent_params,
#   solve_history_dependent_tax, print_hd_equilibrium_summary,
#   check_history_independent_limit
#
# Usage:
#   include("solve_history_dependent_tax.jl")   # also loads the HD settings file
#   using .HistoryDependentTax
#   p  = make_history_dependent_params()        # HD_SETTINGS + overrides
#   eq = solve_history_dependent_tax(p)
#
# Solution method: hours move s' and are therefore intertemporal, so (a', h)
# are chosen jointly on grids against a continuation value bilinearly
# interpolated in (s1', s2'), with the matching bilinear Young (1990) lottery
# for the distribution. Infeasible states carry the finite sentinel
# VINFEASIBLE, since -Inf would give 0 * Inf = NaN in the interpolation. s'
# outside the grid is clamped and the clamped share is reported.
# =============================================================================

module HistoryDependentTaxInfinite

using LinearAlgebra
using Printf
using Statistics
using FastGaussQuadrature
using QuantEcon
using Roots
using StatsBase

# -----------------------------------------------------------------------------
# Shared infrastructure. `using` rather than include, so the methods land in
# this module's scope. It must precede the HDParams declaration below, which
# subtypes AbstractBewleyParams.
# -----------------------------------------------------------------------------
using BewleyCommon
using BewleyCommon: BlockScratch, S_GRID_UNIFORM_BLEND, finalize_welfare

export HDParams, HD_SETTINGS, make_history_dependent_params,
       solve_history_dependent_tax, print_hd_equilibrium_summary,
       check_history_independent_limit


include("params.jl")
include("statistics.jl")
include("report.jl")

"""
    s_stock_moments(mu; tau, eta, rho, sigma_omega, sigma_epsilon, sigma_kappa)

Mean and variance of the ERGODIC distribution of the past-income stock, in
closed form. Returns `(mean, var)` as scalars -- unlike the finite-horizon hd
version, which returns one value per age, because an infinitely lived agent's
stock has a single stationary distribution.

`s' = mu*(x + s)` with `x = kappa + z + eps + ln h` is an AR(1) in s with
coefficient mu and innovation `mu*x`, so unwinding gives
`s_t = mu * sum_{k>=0} mu^k x_{t-k}` and

    E[s]   = mu*E[x] / (1 - mu)
    Var[s] = (mu/(1-mu))^2 * sigma_kappa^2      # permanent: loads on every lag
           + mu^2/(1-mu^2) * sigma_epsilon^2    # iid
           + sigma_omega^2 * sum_n c_n^2,       # AR(1)
             c_n = mu * sum_{k=0}^{n} mu^k rho^(n-k)

The AR(1) term is summed numerically rather than in closed form so that
`mu == rho` needs no special case; the series is truncated once
`max(mu, rho)^n` is below 1e-14.

Hours are constant here. The infinite-horizon normalization makes
`Theta = sum_{s>=0} beta^s theta_s = 1` by construction, so the no-savings
hours rule collapses to `ln h = log(1-tau)/(1+eta)`, independent of age -- a
simplification the finite-horizon version does not enjoy.

Both the mean and the variance diverge as mu -> 1: the ergodic variance carries
`1/(1-mu^2)`, so a unit-root stock has no stationary distribution and the model
is only defined for mu strictly below one. The finite horizon regularises this
implicitly; here it is real.

APPROXIMATE FOR THIS MODEL, deliberately -- Bewley households face assets, a
borrowing limit and a discrete hours grid, so their hours differ from the rule
above. Acceptable because these moments only decide where grid points are
PLACED, never what the model is.
"""
function s_stock_moments(mu::Real; tau, eta, rho,
                         sigma_omega, sigma_epsilon, sigma_kappa)
    mu = Float64(mu)
    mu == 0.0 && return 0.0, 0.0
    mu < 1.0 || error("ergodic s-moments need mu < 1 (got $mu)")

    m_omega = -0.5 * sigma_omega^2
    m_eps   = -0.5 * sigma_epsilon^2
    m_kappa = -0.5 * sigma_kappa^2
    E_z     = m_omega / (1.0 - rho)
    E_lnh   = log(1.0 - tau) / (1.0 + eta)       # Theta = 1 in infinite horizon
    E_x     = m_kappa + E_z + m_eps + E_lnh

    mean_s = mu * E_x / (1.0 - mu)

    # AR(1) block, summed until the terms vanish.
    nmax = ceil(Int, log(1e-14) / log(max(mu, rho, 1e-12)))
    ar_sum = 0.0
    for n in 0:nmax
        c = 0.0
        for k in 0:n
            c += mu^k * rho^(n - k)
        end
        c *= mu
        ar_sum += c^2
    end

    var_s = (mu / (1.0 - mu))^2 * sigma_kappa^2 +
            mu^2 / (1.0 - mu^2) * sigma_epsilon^2 +
            sigma_omega^2 * ar_sum
    return mean_s, var_s
end

"""
    build_s_grid(mu, nS, kappa_grid, z_grid, eps_grid, s_hours_floor, hMax)

Grid for one past-income stock, spanning the INFINITE-horizon support.

With m = kappa + z + eps + ln h, the recursion s' = mu*(m + s) unwinds to
`s = mu * sum_{k>=0} mu^k m`, so s is bounded by `scale * m` with

    scale = mu / (1 - mu).

That is the fixed point of the s-map, and therefore self-invariant: from any
point inside the bounds, s' stays inside, so nothing is ever clamped. The hd
solver tightens this to the J-period reachable range, which is 3.2x narrower at
J = 39 with mu near one -- but an infinitely lived agent genuinely reaches the
full support, so no tightening is available here. The two coincide once J is
large: at J = 99 and mu = 0.85 both give 5.6667 to seven digits.

`scale` diverges as mu -> 1, matching the ergodic variance in
`s_stock_moments`: a unit-root stock has no stationary distribution.

`ln h` is bounded below using s_hours_floor (log(hMin) would blow the grid up).
Bounds are extended to include 0, which is still the birth value of the stock.
Returns `[0.0]` when mu = 0.

`method` controls the SPACING of the points inside those bounds:

  * `:linear`   equally spaced;
  * `:quantile` placed at quantiles of the ERGODIC distribution of the stock,
    normal in closed form from `s_stock_moments`, passed in via `moment_args`.

Quantile spacing matters more here than in the finite-horizon solver. There the
stock is still spreading out from s = 0 when life ends; here it has reached its
ergodic distribution, which is far more concentrated relative to the support the
bounds must cover, so equal spacing wastes proportionally more points.

Endpoints are pinned to `s_lo`/`s_hi` and one interior node is snapped to
exactly 0.0, which the birth condition still looks up.
"""
function build_s_grid(mu::Real, nS::Int, kappa_grid, z_grid, eps_grid,
                      s_hours_floor::Real, hMax::Real;
                      method::Symbol = :linear, moment_args = nothing)
    mu = Float64(mu)
    s_hours_floor = Float64(s_hours_floor)
    hMax = Float64(hMax)
    method in (:linear, :quantile) ||
        error("s grid method must be :linear or :quantile (got :$method)")
    mu == 0.0 && return [0.0]
    m_lo = minimum(kappa_grid) + minimum(z_grid) + minimum(eps_grid) +
           log(s_hours_floor)
    m_hi = maximum(kappa_grid) + maximum(z_grid) + maximum(eps_grid) + log(hMax)
    # Fixed point of the s-map: self-invariant, so s' from any interior point
    # stays interior and nothing is clamped.
    scale = mu / (1.0 - mu)
    s_lo = min(scale * m_lo, 0.0)
    s_hi = max(scale * m_hi, 0.0)
    s_hi > s_lo || error("degenerate s-grid bounds [$s_lo, $s_hi]")

    n = max(nS, 2)
    method === :linear && return collect(range(s_lo, s_hi, length = n))

    moment_args === nothing &&
        error("build_s_grid with method = :quantile needs moment_args")
    mean_s, var_s = s_stock_moments(mu; moment_args...)

    # The ergodic distribution is a SINGLE normal, not the age mixture the
    # finite-horizon solver has to average over -- so no atom at s = 0 and no
    # mixture to build. That also removes the duplicate-node failure mode
    # documented in hd, where quantile levels landing inside the age-0 point
    # mass all inverted to exactly zero.
    F_stock(x::Float64) =
        var_s <= 0.0 ? (x >= mean_s ? 1.0 : 0.0) :
                       normal_cdf((x - mean_s) / sqrt(var_s))

    # Blended with a uniform, as in importance sampling. Pure quantile spacing
    # packs points so tightly around the mean that the outermost cells become
    # enormous and mass placed on the extreme node leaves the grid. Blending
    # caps the widest cell while keeping most of the concentration. See
    # NOTES.md and the tuning table at S_GRID_UNIFORM_BLEND.
    w = S_GRID_UNIFORM_BLEND
    F(x::Float64) = (1.0 - w) * F_stock(x) +
                    w * clamp((x - s_lo) / (s_hi - s_lo), 0.0, 1.0)

    # All ages degenerate (mu so small the stock never moves): nothing to
    # resolve, so fall back rather than invert a step function.
    if var_s <= 0.0
        @warn "s-stock distribution is degenerate; falling back to :linear spacing" mu nS
        return collect(range(s_lo, s_hi, length = n))
    end

    # Midpoint quantile levels avoid p = 0 and p = 1, whose exact quantiles are
    # the infinite tails of the normal rather than anything reachable.
    grid = Vector{Float64}(undef, n)
    for i in 1:n
        p = (i - 0.5) / n
        lo, hi = s_lo, s_hi
        for _ in 1:100                       # bisection; F is nondecreasing
            mid = 0.5 * (lo + hi)
            F(mid) < p ? (lo = mid) : (hi = mid)
        end
        grid[i] = 0.5 * (lo + hi)
    end

    # Local repair of coincident nodes: a steep enough F can return two nodes
    # within rounding of each other. Drop the duplicates and refill by
    # bisecting the widest gaps, rather than discarding the quantile grid.
    tol = 1e-10 * (s_hi - s_lo)
    sort!(grid)
    kept = Float64[grid[1]]
    for x in @view grid[2:end]
        x - kept[end] > tol && push!(kept, x)
    end
    while length(kept) < n                       # refill: split the widest gap
        gaps = diff(kept)
        i = argmax(gaps)
        insert!(kept, i + 1, 0.5 * (kept[i] + kept[i+1]))
    end
    grid = kept

    # Span the reachable range exactly, so nothing can clamp.
    grid[1] = s_lo
    grid[n] = s_hi

    # Put a node exactly at the initial condition. The node nearest zero always
    # has its neighbours straddling it, so this preserves strict ordering.
    if n >= 3
        interior = 2:(n-1)
        i0 = interior[argmin(abs.(@view grid[interior]))]
        grid[i0] = 0.0
    end

    # Post-repair this must hold; if it does not, the construction above is
    # wrong and should be fixed rather than silently downgraded.
    (issorted(grid) && all(diff(grid) .> 0.0)) ||
        error("quantile s-grid is not strictly increasing after repair " *
              "(mu = $mu, nS = $nS); this is a bug in build_s_grid")
    return grid
end


# -----------------------------------------------------------------------------
# Income bases, feasibility, prices
# -----------------------------------------------------------------------------
function precompute_income_bases(kappa::Float64, p::HDParams)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    tax_base = Matrix{Float64}(undef, nZ, nE)      # exp(pow * log wage)
    wage_base = Matrix{Float64}(undef, nZ, nE)     # exp(log wage)
    @inbounds for iz in 1:nZ, ie in 1:nE
        log_wage = kappa + p.z_grid[iz] + p.eps_grid[ie]
        tax_base[iz, ie] = exp(p.pow * log_wage)
        wage_base[iz, ie] = exp(log_wage)
    end
    return tax_base, wage_base
end

# NOTE: `first_nonnegative_asset_index` lived here. It existed only to build the
# hd solver's terminal-age constraint a' >= 0, which has no counterpart in an
# infinite horizon -- the borrowing limit binds at every age. It was computed
# once per lambda, threaded through `simulate_kappa!`, and never read.

# -----------------------------------------------------------------------------
# Hand-to-mouth asset rule
# -----------------------------------------------------------------------------

"""
    HTMTransition

The exogenous hand-to-mouth asset rule, precomputed once per `HDParams`:
`a' = a/qSav` for `a >= 0` and `a' = a` for `a < 0` (psmodel.tex).

`a/qSav` lands between asset grid nodes by construction, so the continuation
value and the forward transition BOTH read it through the same Young lottery
`(left, right, weight)` stored here. Using two different placements would break
the value-function/simulation welfare cross-check, which is the guard that this
is implemented consistently.

`cash` is `a - q(a')a'`: exactly `0` for an unclipped rollover and
`(1-qBorr)*a < 0` for a debtor, matching the budget in psmodel.tex. `clipped`
flags that `a/qSav` ran past the top grid node for some `a`, in which case `a'`
is held at `aMax` and the excess is consumed -- the rule has no fixed point
above zero, so only the grid stops it.
"""
struct HTMTransition
    next_assets::Vector{Float64}
    cash::Vector{Float64}
    left::Vector{Int}
    right::Vector{Int}
    weight::Vector{Float64}
    clipped::Bool
end

function htm_transition(p::HDParams)
    nA = length(p.a_grid)
    a_top = last(p.a_grid)
    next_assets = Vector{Float64}(undef, nA)
    cash = Vector{Float64}(undef, nA)
    left = Vector{Int}(undef, nA)
    right = Vector{Int}(undef, nA)
    weight = Vector{Float64}(undef, nA)
    clipped = false
    for ia in 1:nA
        a = p.a_grid[ia]
        if a >= 0.0
            ap = a / p.qSav
            if ap > a_top
                clipped = true
                ap = a_top
                cash[ia] = a - p.qSav * ap
            else
                # Set to zero rather than evaluating a - qSav*(a/qSav), which
                # is the same number up to a rounding error that would leak
                # into consumption at every positive-asset HtM state.
                cash[ia] = 0.0
            end
        else
            ap = a
            cash[ia] = a - p.qBorr * ap
        end
        next_assets[ia] = ap
        l, r, w = grid_lookup_weights(p.a_grid, ap)
        left[ia] = l; right[ia] = r; weight[ia] = w
    end
    return HTMTransition(next_assets, cash, left, right, weight, clipped)
end

# -----------------------------------------------------------------------------
# Backward induction for one kappa: joint (a', h) grid choice with bilinear
# interpolation of the continuation value in (s1', s2')
#
# Parallelism lives here, over the nEps*nS1*nS2 blocks of (eps, s1, s2) at
# each (age, z), not over kappa, which is typically 3 and would cap the solver
# at 3 cores. Blocks write disjoint slices and only read the shared EVz, so the
# result is bit-for-bit identical to a serial run.
# -----------------------------------------------------------------------------

"""
    solve_block!(...)

Solve one (eps, s1, s2) block at a given (age, z): fill `EVh` by bilinear
interpolation of the continuation in (s1', s2'), then choose (a', h) jointly
on the grids for every current asset level.

The hours scan is accelerated by Topkis monotonicity
(`exploit_hours_monotonicity`), documented at the scan itself. A tangent-line
pre-filter on `log` was tried here and removed: profiling puts `log` at about
half of runtime, but the bound costs a multiply, an add, and a branch per
candidate, which measured 7% SLOWER than simply calling `log`.
"""
function solve_block!(Vcur, policyAIndex, policyH, sc::BlockScratch,
                      EVz, cash, ie::Int, is1::Int, is2::Int, iz::Int,
                      ia_first::Int, has_continuation::Bool,
                      m_base::Float64, coeff::Float64, p::HDParams)
    nA = size(cash, 1)
    nH = length(p.h_grid)
    dis = p.h_grid_disutility
    hpow = p.h_income_power
    lnh = p.log_h_grid
    beta = p.beta
    util_weight = 1.0 - beta
    inc = sc.inc
    EVh = sc.EVh
    ih_ub = sc.ih_ub

    @inbounds begin
        for ih in 1:nH
            inc[ih] = coeff * hpow[ih]
        end
        inc_max = inc[nH]

        if has_continuation
            l1v, h1v, w1v = sc.l1v, sc.h1v, sc.w1v
            l2v, h2v, w2v = sc.l2v, sc.h2v, sc.w2v
            for ih in 1:nH
                s1n = p.mu1 * (m_base + lnh[ih] + p.s1_grid[is1])
                s2n = p.mu2 * (m_base + lnh[ih] + p.s2_grid[is2])
                l1v[ih], h1v[ih], w1v[ih] = grid_lookup_weights(p.s1_grid, s1n)
                l2v[ih], h2v[ih], w2v[ih] = grid_lookup_weights(p.s2_grid, s2n)
            end
            for ih in 1:nH
                l1, h1, w1 = l1v[ih], h1v[ih], w1v[ih]
                l2, h2, w2 = l2v[ih], h2v[ih], w2v[ih]
                w11 = (1.0 - w1) * (1.0 - w2)
                w12 = (1.0 - w1) * w2
                w21 = w1 * (1.0 - w2)
                w22 = w1 * w2
                for iap in ia_first:nA
                    EVh[ih, iap] =
                        w11 * EVz[iap, l1, l2] +
                        w12 * EVz[iap, l1, h2] +
                        w21 * EVz[iap, h1, l2] +
                        w22 * EVz[iap, h1, h2]
                end
            end
        end

        # Optimal-hours monotonicity (Topkis): for fixed a' (so a fixed EVh
        # column), the objective has decreasing differences in (h, cash)
        # because d^2 ln(cash + inc(h)) / d inc d cash < 0 and inc is
        # increasing in h, while the continuation term does not depend on
        # cash. Since cash[iap, ia] is strictly increasing in ia, the
        # (largest) maximizing hours index is nonincreasing in ia for each
        # iap. Scanning ia in ascending order, the previous optimum at the
        # same iap is therefore a valid upper bound for the hours scan.
        exploit = p.exploit_hours_monotonicity
        exploit && fill!(ih_ub, nH)

        for ia in 1:nA
            best_val = VINFEASIBLE
            best_iap = ia_first
            best_ih = nH

            for iap in ia_first:nA
                cash_v = cash[iap, ia]
                # cash is decreasing in a' (q > 0 on both sides of zero and
                # continuous there), so once maximal hours cannot deliver
                # c > 0, no larger a' can.
                if cash_v + inc_max <= 0.0
                    break
                end
                ih0 = cash_v > 0.0 ? 1 : searchsortedfirst(inc, -cash_v)
                ub = exploit ? max(ih_ub[iap], ih0) : nH

                local_best = VINFEASIBLE
                local_ih = ub
                for ih in ih0:ub
                    c = cash_v + inc[ih]
                    c <= 0.0 && continue
                    val = util_weight * (log(c) - dis[ih]) + beta * EVh[ih, iap]
                    # ">=" selects the LARGEST maximizer, as the monotone
                    # bound requires.
                    if val >= local_best
                        local_best = val
                        local_ih = ih
                    end
                end
                exploit && (ih_ub[iap] = local_ih)

                if local_best > best_val
                    best_val = local_best
                    best_iap = iap
                    best_ih = local_ih
                end
            end

            Vcur[ia, is1, is2, iz, ie] = best_val
            policyAIndex[ia, is1, is2, iz, ie] = Int32(best_iap)
            policyH[ia, is1, is2, iz, ie] = p.h_grid[best_ih]
        end
    end
    return nothing
end


"""
    solve_block_htm!(...)

The hand-to-mouth counterpart of `solve_block!` for one (eps, s1, s2) block.

`a'` is exogenous, so the joint (a', h) search collapses to a scan over hours
alone -- but it is STILL a maximization, and this is the one place where the
history-dependent extension differs from `hiinf_htm`. There the hand-to-mouth
hours choice is static and its payoff is precomputed once per lambda. Here
hours move the past-income stocks through s' = mu*(log wage + log h + s), so
the household trades current leisure against future tax liabilities exactly as
a saver does, and the scan has to be redone whenever the continuation changes.

The continuation is TRILINEAR: bilinear in (s1', s2') as for the saver, and
linear in a' as well, because a'= a/qSav falls between asset grid nodes. Both
weights come from `htm`, the same object the forward pass uses.

`sc.EVh` is reused as (h, a) here rather than (h, a'); the saver's block has
finished with it by the time this runs.
"""
function solve_block_htm!(VcurH, policyHtmH, sc::BlockScratch, EVzH,
                          htm::HTMTransition, ie::Int, is1::Int, is2::Int,
                          iz::Int, m_base::Float64, coeff::Float64, p::HDParams)
    nA = length(p.a_grid)
    nH = length(p.h_grid)
    dis = p.h_grid_disutility
    hpow = p.h_income_power
    lnh = p.log_h_grid
    beta = p.beta
    util_weight = 1.0 - beta
    inc = sc.inc
    EVh = sc.EVh

    @inbounds begin
        for ih in 1:nH
            inc[ih] = coeff * hpow[ih]
        end

        for ih in 1:nH
            s1n = p.mu1 * (m_base + lnh[ih] + p.s1_grid[is1])
            s2n = p.mu2 * (m_base + lnh[ih] + p.s2_grid[is2])
            l1, h1, w1 = grid_lookup_weights(p.s1_grid, s1n)
            l2, h2, w2 = grid_lookup_weights(p.s2_grid, s2n)
            w11 = (1.0 - w1) * (1.0 - w2)
            w12 = (1.0 - w1) * w2
            w21 = w1 * (1.0 - w2)
            w22 = w1 * w2
            for ia in 1:nA
                al = htm.left[ia]
                ev = w11 * EVzH[al, l1, l2] + w12 * EVzH[al, l1, h2] +
                     w21 * EVzH[al, h1, l2] + w22 * EVzH[al, h1, h2]
                wa = htm.weight[ia]
                if wa > 0.0
                    ar = htm.right[ia]
                    evr = w11 * EVzH[ar, l1, l2] + w12 * EVzH[ar, l1, h2] +
                          w21 * EVzH[ar, h1, l2] + w22 * EVzH[ar, h1, h2]
                    ev = (1.0 - wa) * ev + wa * evr
                end
                EVh[ih, ia] = ev
            end
        end

        # No Topkis bound here. The saver's monotonicity argument runs in `ia`
        # through cash[iap, ia]; the hand-to-mouth household's cash is 0 at
        # every nonnegative a and (1-qBorr)*a below, so there is no comparable
        # ordering to exploit and the scan is over the full hours grid.
        for ia in 1:nA
            cash_v = htm.cash[ia]
            best_val = VINFEASIBLE
            best_ih = nH
            ih0 = cash_v > 0.0 ? 1 : searchsortedfirst(inc, -cash_v)
            for ih in ih0:nH
                c = cash_v + inc[ih]
                c <= 0.0 && continue
                val = util_weight * (log(c) - dis[ih]) + beta * EVh[ih, ia]
                if val > best_val
                    best_val = val
                    best_ih = ih
                end
            end
            VcurH[ia, is1, is2, iz, ie] = best_val
            policyHtmH[ia, is1, is2, iz, ie] = p.h_grid[best_ih]
        end
    end
    return nothing
end

"""
    evaluate_block_htm!(...)

Policy evaluation for the hand-to-mouth block: apply the stored hours and form
the same trilinear continuation, with no scan. The Howard counterpart of
`solve_block_htm!`, and it must mirror it in how the continuation is built for
the same reason `evaluate_block!` must mirror `solve_block!`.
"""
function evaluate_block_htm!(VcurH, policyHtmH, EVzH, htm::HTMTransition,
                             ie::Int, is1::Int, is2::Int, iz::Int,
                             m_base::Float64, coeff::Float64, p::HDParams)
    nA = length(p.a_grid)
    beta = p.beta
    util_weight = 1.0 - beta

    @inbounds for ia in 1:nA
        h = policyHtmH[ia, is1, is2, iz, ie]
        c = coeff * h^p.pow + htm.cash[ia]
        if c <= 0.0
            VcurH[ia, is1, is2, iz, ie] = VINFEASIBLE
            continue
        end
        u = log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta)

        s1n = p.mu1 * (m_base + log(h) + p.s1_grid[is1])
        s2n = p.mu2 * (m_base + log(h) + p.s2_grid[is2])
        l1, h1, w1 = grid_lookup_weights(p.s1_grid, s1n)
        l2, h2, w2 = grid_lookup_weights(p.s2_grid, s2n)
        al = htm.left[ia]
        ev = (1.0 - w1) * ((1.0 - w2) * EVzH[al, l1, l2] + w2 * EVzH[al, l1, h2]) +
             w1 * ((1.0 - w2) * EVzH[al, h1, l2] + w2 * EVzH[al, h1, h2])
        wa = htm.weight[ia]
        if wa > 0.0
            ar = htm.right[ia]
            evr = (1.0 - w1) * ((1.0 - w2) * EVzH[ar, l1, l2] + w2 * EVzH[ar, l1, h2]) +
                  w1 * ((1.0 - w2) * EVzH[ar, h1, l2] + w2 * EVzH[ar, h1, h2])
            ev = (1.0 - wa) * ev + wa * evr
        end
        VcurH[ia, is1, is2, iz, ie] = util_weight * u + beta * ev
    end
    return nothing
end

"""
    solve_value_function_for_kappa(lambda, kappa, first_ap, q_by_ap, tax_base, htm, p)

Solve the agent's STATIONARY problem by value-function iteration, returning
age-independent policies. This is the central difference from the hd solver:
there, backward induction produces a different policy at each of J+1 ages and
stores them all; here the fixed point is a single policy, which is what removes
the `nAge` dimension from the policy arrays and with it ~99% of the memory.

Iteration is Howard's method (modified policy iteration) when
`p.howardSteps > 0`: one maximizing sweep, then `howardSteps` cheap evaluation
sweeps holding the policy fixed. This matters because plain VFI contracts at
beta = 0.96, needing ln(tol)/ln(beta) ~ 451 sweeps for tol = 1e-8 -- 4.5x the
100 age sweeps the J = 99 finite model does. Howard typically converges in
20-40 maximizations, so the expensive work falls BELOW the finite-horizon cost.
Set `howardSteps = 0` for plain VFI, which is slower but a useful cross-check:
both must reach the same fixed point.

Convergence is measured in the sup norm on V between successive maximizing
sweeps. Returns `(policyAIndex, policyH, V, welfare_value_function, iters, gap)`.
"""
function solve_value_function_for_kappa(lambda::Float64, kappa::Float64,
                                        first_ap::Vector{Int},
                                        q_by_ap::Vector{Float64},
                                        tax_base::Matrix{Float64},
                                        htm::HTMTransition, p::HDParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nS1 = length(p.s1_grid)
    nS2 = length(p.s2_grid)
    nH = length(p.h_grid)

    # One value function per access state. Everything below is doubled; the
    # saver's own block is untouched.
    VnextS = zeros(nA, nS1, nS2, nZ, nE)             # V_0 = 0
    VcurS = similar(VnextS)
    VnextH = zeros(nA, nS1, nS2, nZ, nE)
    VcurH = similar(VnextH)
    VbarS = Array{Float64}(undef, nA, nS1, nS2, nZ)  # sum over eps'
    VbarH = Array{Float64}(undef, nA, nS1, nS2, nZ)
    EVzS = Array{Float64}(undef, nA, nS1, nS2)       # sum over z' given z
    EVzH = Array{Float64}(undef, nA, nS1, nS2)
    # The access-mixed continuations the two problems actually face. The access
    # shock is independent of (z', eps'), so the mixing is done AFTER the
    # expectation and the saver's block sees a drop-in replacement for EVz.
    EVmixS = Array{Float64}(undef, nA, nS1, nS2)
    EVmixH = Array{Float64}(undef, nA, nS1, nS2)

    # No age dimension: this is the memory win over the finite-horizon solver.
    policyAIndex = Array{Int32}(undef, nA, nS1, nS2, nZ, nE)
    policyH = Array{Float64}(undef, nA, nS1, nS2, nZ, nE)
    policyHtmH = Array{Float64}(undef, nA, nS1, nS2, nZ, nE)

    cash = Matrix{Float64}(undef, nA, nA)            # cash[iap, ia]
    @inbounds for ia in 1:nA, iap in 1:nA
        cash[iap, ia] = p.a_grid[ia] - q_by_ap[iap] * p.a_grid[iap]
    end

    scratch = [BlockScratch(nH, nA) for _ in 1:Threads.maxthreadid()]
    blocks = [(ie, is1, is2) for is2 in 1:nS2 for is1 in 1:nS1 for ie in 1:nE]
    nBlocks = length(blocks)

    # One sweep of the Bellman operator: `maximize = true` re-optimizes and
    # rewrites the policy, `false` just applies the stored one (Howard).
    function sweep!(maximize::Bool)
        fill!(VbarS, 0.0)
        fill!(VbarH, 0.0)
        @inbounds for ie in 1:nE
            VbarS .+= p.Peps[ie] .* view(VnextS, :, :, :, :, ie)
            VbarH .+= p.Peps[ie] .* view(VnextH, :, :, :, :, ie)
        end
        for iz in 1:nZ
            ia_first = first_ap[iz]          # no terminal age, so always this
            fill!(EVzS, 0.0)
            fill!(EVzH, 0.0)
            @inbounds for izp in 1:nZ
                pz = p.Pz[iz, izp]
                pz == 0.0 && continue
                EVzS .+= pz .* view(VbarS, :, :, :, izp)
                EVzH .+= pz .* view(VbarH, :, :, :, izp)
            end
            # A state with no feasible choice carries the FINITE sentinel, so
            # these products are 0.0 when the weight is zero. With -Inf they
            # would be NaN, and the pSS = 1 / pHH = 0 corner -- the one that
            # has to reproduce hdinf exactly -- is where the weight is zero.
            @inbounds for i in eachindex(EVzS)
                evs = EVzS[i]
                evh = EVzH[i]
                EVmixS[i] = p.pSS * evs + (1.0 - p.pSS) * evh
                EVmixH[i] = p.pHH * evh + (1.0 - p.pHH) * evs
            end
            Threads.@threads :static for ib in 1:nBlocks
                ie, is1, is2 = blocks[ib]
                m_base = kappa + p.z_grid[iz] + p.eps_grid[ie]
                coeff = lambda * tax_base[iz, ie] * p.s_factor[is1, is2]
                if maximize
                    sc = scratch[Threads.threadid()]
                    solve_block!(VcurS, policyAIndex, policyH, sc, EVmixS, cash,
                                 ie, is1, is2, iz, ia_first, true,
                                 m_base, coeff, p)
                    solve_block_htm!(VcurH, policyHtmH, sc, EVmixH, htm,
                                     ie, is1, is2, iz, m_base, coeff, p)
                else
                    evaluate_block!(VcurS, policyAIndex, policyH, EVmixS, cash,
                                    ie, is1, is2, iz, m_base, coeff, p)
                    evaluate_block_htm!(VcurH, policyHtmH, EVmixH, htm,
                                        ie, is1, is2, iz, m_base, coeff, p)
                end
            end
        end
        return nothing
    end

    # The swap lives here rather than inside `sweep!` so the closure only ever
    # mutates the arrays; assigning to a captured variable would box both and
    # make every access inside the sweep type-unstable.
    iters = 0
    gap = Inf
    for outer_iter in 1:p.maxIterV
        sweep!(true)                                  # maximize: Vnext -> Vcur
        # Over BOTH value functions: V^H can still be moving after V^S has
        # settled, since its only dynamics run through the access chain and
        # the past-income stocks.
        gap = max(maximum(abs, VcurS .- VnextS), maximum(abs, VcurH .- VnextH))
        VnextS, VcurS = VcurS, VnextS
        VnextH, VcurH = VcurH, VnextH
        iters += 1
        gap <= p.tolV && break
        for _ in 1:p.howardSteps                      # evaluate at fixed policy
            sweep!(false)
            VnextS, VcurS = VcurS, VnextS
            VnextH, VcurH = VcurH, VnextH
        end
    end
    gap <= p.tolV || @warn "value function did not converge" kappa iters gap tolV = p.tolV

    welfare_value_function = expected_initial_value(VnextS, VnextH, kappa, p)
    return policyAIndex, policyH, policyHtmH, VnextS, VnextH,
           welfare_value_function, iters, gap
end

# Newborns draw their access state from the STATIONARY distribution (piS, piH),
# independently of (a0, s0, z, eps), so the birth value is the piS/piH mix of
# the two value functions at the same (a0, s0) placement. `simulate_kappa!`
# seeds the distribution the same way; if the two disagree, the welfare
# cross-check in `finalize_welfare` reports it.
"""
    initial_asset_weights(kappa, p)

Grid placement of the initial asset holding, as `(left, right, right_weight)`
from the same Young lottery used for a'. Both the birth value function and the
simulated initial distribution go through THIS function; if they disagree the
value-function/simulation welfare cross-check breaks, which is the guard wanted.
"""
function initial_asset_weights(kappa::Float64, p::HDParams)
    a0 = p.a0_scales_with_kappa ? p.a0 * exp(kappa) : p.a0
    return grid_lookup_weights(p.a_grid, clamp(a0, first(p.a_grid), last(p.a_grid)))
end

function expected_initial_value(V0S, V0H, kappa::Float64, p::HDParams)
    ia0l, ia0r, ia0w = initial_asset_weights(kappa, p)
    l1, h1, w1 = grid_lookup_weights(p.s1_grid, 0.0)
    l2, h2, w2 = grid_lookup_weights(p.s2_grid, 0.0)
    expected_value = 0.0
    @inbounds for iz in eachindex(p.z_grid), ie in eachindex(p.eps_grid)
        prob = p.z0_probs[iz] * p.Peps[ie]
        prob == 0.0 && continue
        bilS(ia) = (1.0 - w1) * ((1.0 - w2) * V0S[ia, l1, l2, iz, ie] +
                                 w2 * V0S[ia, l1, h2, iz, ie]) +
                   w1 * ((1.0 - w2) * V0S[ia, h1, l2, iz, ie] +
                         w2 * V0S[ia, h1, h2, iz, ie])
        bilH(ia) = (1.0 - w1) * ((1.0 - w2) * V0H[ia, l1, l2, iz, ie] +
                                 w2 * V0H[ia, l1, h2, iz, ie]) +
                   w1 * ((1.0 - w2) * V0H[ia, h1, l2, iz, ie] +
                         w2 * V0H[ia, h1, h2, iz, ie])
        vS = ia0w > 0.0 ? (1.0 - ia0w) * bilS(ia0l) + ia0w * bilS(ia0r) : bilS(ia0l)
        vH = ia0w > 0.0 ? (1.0 - ia0w) * bilH(ia0l) + ia0w * bilH(ia0r) : bilH(ia0l)
        expected_value += prob * (p.piS * vS + p.piH * vH)
    end
    return expected_value
end

# -----------------------------------------------------------------------------
# Distribution iteration for one kappa (bilinear Young lottery in s')
# -----------------------------------------------------------------------------
"""
    simulate_kappa!(...)

Forward pass for one kappa, over `(a, s1, s2, z, eps, access)` with
`access = 1` for savers and `access = 2` for hand-to-mouth, seeded at the
stationary `(piS, piH)`.

The access chain is independent of `(z', eps')`, of the asset choice and of the
past-income stocks, so the transition factorizes: the `(a', s1', s2', z')` mass
is built exactly as in `hdinf` and then split `pSS / 1-pSS` (from S) or
`1-pHH / pHH` (from H). The one structural addition is that a hand-to-mouth
household's `a' = a/qSav` is off the asset grid, so it lands on two asset nodes
through the same Young lottery the value function used, where a saver lands on
one.
"""
function simulate_kappa!(C, H, Y, A, stats::StatsAccumulator,
                         stats_all::StatsAccumulator,
                         stats_lo::StatsAccumulator,
                         policyAIndex, policyH, policyHtmH, kappa::Float64,
                         pkappa::Float64,
                         first_ap::Vector{Int},
                         q_by_ap::Vector{Float64},
                         tax_base::Matrix{Float64},
                         wage_base::Matrix{Float64},
                         htm::HTMTransition,
                         p::HDParams, lambda::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nS1 = length(p.s1_grid)
    nS2 = length(p.s2_grid)
    nAge = p.maxAge
    h_upper = hours_upper_bound(p)
    a_top = last(p.a_grid)
    upper_tol = upper_bound_level_tol(a_top)

    # Sixth axis: 1 = saver (S), 2 = hand-to-mouth (H).
    dist = zeros(nA, nS1, nS2, nZ, nE, 2)
    dist_noeps = zeros(nA, nS1, nS2, nZ, 2)

    ia0l, ia0r, ia0w = initial_asset_weights(kappa, p)
    l1, h1, w1 = grid_lookup_weights(p.s1_grid, 0.0)
    l2, h2, w2 = grid_lookup_weights(p.s2_grid, 0.0)
    @inbounds for iz in 1:nZ, ie in 1:nE
        m0 = p.z0_probs[iz] * p.Peps[ie]
        m0 == 0.0 && continue
        for (iacc, share) in ((1, p.piS), (2, p.piH)),
            (ia0, aw) in ((ia0l, 1.0 - ia0w), (ia0r, ia0w))
            aw == 0.0 && continue
            m0s = m0 * share * aw
            dist[ia0, l1, l2, iz, ie, iacc] += m0s * (1.0 - w1) * (1.0 - w2)
            dist[ia0, l1, h2, iz, ie, iacc] += m0s * (1.0 - w1) * w2
            dist[ia0, h1, l2, iz, ie, iacc] += m0s * w1 * (1.0 - w2)
            dist[ia0, h1, h2, iz, ie, iacc] += m0s * w1 * w2
        end
    end

    # Flow utility per age, kept separately so the discounted sum can be closed
    # analytically past the settled age. Accumulating the discounted total
    # directly would silently truncate: breaking at age Jc drops a tail worth
    # beta^Jc of lifetime utility, which is 2.2e-3 at Jc = 150 -- far above the
    # 1e-17 agreement this check is supposed to demonstrate.
    u_by_age = zeros(nAge)
    clamped_mass = 0.0
    converged_age = 0           # diagnostic only; 0 means never settled by nAge
    final_drift = NaN           # drift at the last age, reported not tested
    s1_lo = p.s1_grid[1]; s1_hi = p.s1_grid[end]
    s2_lo = p.s2_grid[1]; s2_hi = p.s2_grid[end]

    @inbounds for age in 1:nAge
        fill!(dist_noeps, 0.0)
        # Three age coverages, exactly as in the history-independent solver:
        # `stats` is the calibration window, `stats_all` every simulated age,
        # and `stats_lo` the single age at which the window opens. The
        # aggregates C/H/Y/A below are NOT gated -- the government budget and
        # the welfare integral need the whole path.
        in_stats_window = p.stats_age_lo <= age <= p.stats_age_hi
        at_stats_age_lo = age == p.stats_age_lo

        for ie in 1:nE, iz in 1:nZ
            # Infinite horizon: no terminal age, so the borrowing limit binds
            # at every age and there is no a' >= 0 special case to exclude.
            lower_idx = first_ap[iz]
            true_borrowing_limit = -borrowing_limit(kappa, iz, p)
            effective_borrowing_limit = -p.a_grid[lower_idx]
            m_base = kappa + p.z_grid[iz] + p.eps_grid[ie]

            for is2 in 1:nS2, is1 in 1:nS1
                coeff = lambda * tax_base[iz, ie] * p.s_factor[is1, is2]
                for ia in 1:nA, iacc in 1:2
                    mass = dist[ia, is1, is2, iz, ie, iacc]
                    mass <= p.massTol && continue
                    is_htm = iacc == 2

                    a = p.a_grid[ia]
                    if is_htm
                        # Exogenous rule, and the borrowing limit is not a
                        # constraint on someone who does not choose, so
                        # `at_borrowing_constraint` is false.
                        ap = htm.next_assets[ia]
                        cash = htm.cash[ia]
                        h = policyHtmH[ia, is1, is2, iz, ie]
                        next_left = htm.left[ia]
                        next_right = htm.right[ia]
                        next_w = htm.weight[ia]
                        at_borrowing_constraint = false
                        at_asset_upper = ap >= a_top - upper_tol
                    else
                        iap = Int(policyAIndex[ia, is1, is2, iz, ie])
                        ap = p.a_grid[iap]
                        cash = a - q_by_ap[iap] * ap
                        h = policyH[ia, is1, is2, iz, ie]
                        next_left = iap
                        next_right = iap
                        next_w = 0.0
                        at_borrowing_constraint = iap == lower_idx
                        at_asset_upper = iap == nA
                    end
                    c = coeff * h^p.pow + cash
                    c > 0.0 || error(
                        "negative consumption on a positive-mass state " *
                        "(age=$age, ia=$ia, access=$(is_htm ? "H" : "S")): " *
                        "widen grids or check feasibility")
                    y = wage_base[iz, ie] * h
                    u = log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta)
                    u_by_age[age] += mass * u

                    weighted_mass = pkappa * mass
                    C[age] += weighted_mass * c
                    H[age] += weighted_mass * h
                    Y[age] += weighted_mass * y
                    A[age] += weighted_mass * ap

                    accumulate_stats!(stats_all, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      at_borrowing_constraint, at_asset_upper,
                                      h_upper, is_htm, false)
                    in_stats_window &&
                        accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                                          true_borrowing_limit, effective_borrowing_limit,
                                          at_borrowing_constraint, at_asset_upper,
                                          h_upper, is_htm, p.collect_distributions)
                    at_stats_age_lo &&
                        accumulate_stats!(stats_lo, weighted_mass, ia, a, ap, h, c, y,
                                          true_borrowing_limit, effective_borrowing_limit,
                                          at_borrowing_constraint, at_asset_upper,
                                          h_upper, is_htm, false)

                    if age < nAge
                        s1n = p.mu1 * (m_base + log(h) + p.s1_grid[is1])
                        s2n = p.mu2 * (m_base + log(h) + p.s2_grid[is2])
                        if (nS1 > 1 && (s1n < s1_lo || s1n > s1_hi)) ||
                           (nS2 > 1 && (s2n < s2_lo || s2n > s2_hi))
                            clamped_mass += weighted_mass
                        end
                        c1l, c1h, cw1 = grid_lookup_weights(p.s1_grid, s1n)
                        c2l, c2h, cw2 = grid_lookup_weights(p.s2_grid, s2n)
                        v11 = (1.0 - cw1) * (1.0 - cw2)
                        v12 = (1.0 - cw1) * cw2
                        v21 = cw1 * (1.0 - cw2)
                        v22 = cw1 * cw2
                        # The access split. Independent of everything else, so
                        # it multiplies the (a', s', z') mass rather than
                        # changing how it is built.
                        stay_w = is_htm ? p.pHH : p.pSS
                        switch_w = 1.0 - stay_w
                        iacc_switch = 3 - iacc
                        for izp in 1:nZ
                            pz = p.Pz[iz, izp]
                            pz == 0.0 && continue
                            base = mass * pz
                            # A saver lands on one asset node (next_w = 0, so
                            # the second leg is skipped); a hand-to-mouth
                            # household lands on the two the Young lottery
                            # straddles.
                            for (iapn, aw) in ((next_left, 1.0 - next_w),
                                               (next_right, next_w))
                                aw == 0.0 && continue
                                basea = base * aw
                                for (iaccp, accw) in ((iacc, stay_w),
                                                      (iacc_switch, switch_w))
                                    accw == 0.0 && continue
                                    b = basea * accw
                                    dist_noeps[iapn, c1l, c2l, izp, iaccp] += b * v11
                                    dist_noeps[iapn, c1l, c2h, izp, iaccp] += b * v12
                                    dist_noeps[iapn, c1h, c2l, izp, iaccp] += b * v21
                                    dist_noeps[iapn, c1h, c2h, izp, iaccp] += b * v22
                                end
                            end
                        end
                    end
                end
            end
        end

        for iep in 1:nE
            view(dist, :, :, :, :, iep, :) .= p.Peps[iep] .* dist_noeps
        end

        # Record where the cross-section settles, but DO NOT stop here; see the
        # long note in `hdinf` on the four things an early break corrupted.
        if age > 1
            drift = abs(Y[age] - Y[age-1]) + abs(C[age] - C[age-1])
            converged_age == 0 && drift <= p.tolDist && (converged_age = age)
            age == nAge && (final_drift = drift)
        end
    end

    # Discounted lifetime utility. The path now runs to maxAge, so the closed
    # form covers only ages beyond it: flow utility is constant at
    # u_by_age[nAge] from there on, giving u_inf * beta^nAge.
    welfare_simulation = 0.0
    for age in 1:nAge
        welfare_simulation += (1.0 - p.beta) * p.beta^(age - 1) * u_by_age[age]
    end
    welfare_simulation += p.beta^nAge * u_by_age[nAge]

    return welfare_simulation, clamped_mass, converged_age, final_drift
end

# -----------------------------------------------------------------------------
# Aggregates at a given lambda and the government-budget residual. Policies
# are solved kappa-by-kappa (each solve is internally threaded over blocks);
# the forward distribution is then threaded over kappa.
# -----------------------------------------------------------------------------
function solve_aggregates_for_lambda(lambda::Float64, p::HDParams)
    nAge = p.maxAge
    nKappa = length(p.kappa_grid)
    C = zeros(nAge); H = zeros(nAge); Y = zeros(nAge); A = zeros(nAge)
    stats_acc = StatsAccumulator(length(p.a_grid))
    stats_all_acc = StatsAccumulator(length(p.a_grid))
    stats_lo_acc = StatsAccumulator(length(p.a_grid))

    C_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    H_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    Y_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    A_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    stats_by_kappa = Vector{StatsAccumulator}(undef, nKappa)
    stats_all_by_kappa = Vector{StatsAccumulator}(undef, nKappa)
    stats_lo_by_kappa = Vector{StatsAccumulator}(undef, nKappa)
    welfare_value_function_by_kappa = Vector{Float64}(undef, nKappa)
    welfare_simulation_by_kappa = similar(welfare_value_function_by_kappa)
    clamped_by_kappa = zeros(nKappa)

    q_by_ap = asset_prices(p)
    # kappa-independent, so built once and shared read-only.
    htm = htm_transition(p)

    # Phase 1: the stationary policies. kappa runs SERIALLY because the value
    # function solve is itself threaded over the nEps*nS1*nS2 blocks, which
    # offers far more parallelism than the nKappa values ever could.
    # The arrays are 5-D, not 6-D: no age index in an infinite horizon.
    policyA_by_kappa = Vector{Array{Int32,5}}(undef, nKappa)
    policyH_by_kappa = Vector{Array{Float64,5}}(undef, nKappa)
    policyHtmH_by_kappa = Vector{Array{Float64,5}}(undef, nKappa)
    first_ap_by_kappa = Vector{Vector{Int}}(undef, nKappa)
    vIters = zeros(Int, nKappa)
    vGap = fill(NaN, nKappa)
    converged_by_kappa = zeros(Int, nKappa)
    drift_by_kappa = fill(NaN, nKappa)
    tax_base_by_kappa = Vector{Matrix{Float64}}(undef, nKappa)
    wage_base_by_kappa = Vector{Matrix{Float64}}(undef, nKappa)
    for ik in 1:nKappa
        kappa = p.kappa_grid[ik]
        first_ap_by_kappa[ik] = first_feasible_asset_indices(kappa, p)
        tax_base_by_kappa[ik], wage_base_by_kappa[ik] =
            precompute_income_bases(kappa, p)
        policyA_by_kappa[ik], policyH_by_kappa[ik], policyHtmH_by_kappa[ik],
        _, _, welfare_value_function_by_kappa[ik], vIters[ik], vGap[ik] =
            solve_value_function_for_kappa(lambda, kappa, first_ap_by_kappa[ik],
                                           q_by_ap, tax_base_by_kappa[ik], htm, p)
    end

    # Phase 2: the forward distribution, which is independent across kappa.
    Threads.@threads :static for ik in 1:nKappa
        stats_local = StatsAccumulator(length(p.a_grid))
        stats_all_local = StatsAccumulator(length(p.a_grid))
        stats_lo_local = StatsAccumulator(length(p.a_grid))
        welfare_simulation, clamped, converged, drift = simulate_kappa!(
            C_by_kappa[ik], H_by_kappa[ik], Y_by_kappa[ik], A_by_kappa[ik],
            stats_local, stats_all_local, stats_lo_local,
            policyA_by_kappa[ik], policyH_by_kappa[ik], policyHtmH_by_kappa[ik],
            p.kappa_grid[ik], p.Pkappa[ik],
            first_ap_by_kappa[ik], q_by_ap,
            tax_base_by_kappa[ik], wage_base_by_kappa[ik], htm, p, lambda,
        )
        stats_by_kappa[ik] = stats_local
        stats_all_by_kappa[ik] = stats_all_local
        stats_lo_by_kappa[ik] = stats_lo_local
        welfare_simulation_by_kappa[ik] = welfare_simulation
        clamped_by_kappa[ik] = clamped
        converged_by_kappa[ik] = converged
        drift_by_kappa[ik] = drift
    end

    for ik in 1:nKappa
        C .+= C_by_kappa[ik]
        H .+= H_by_kappa[ik]
        Y .+= Y_by_kappa[ik]
        A .+= A_by_kappa[ik]
        merge_stats!(stats_acc, stats_by_kappa[ik])
        merge_stats!(stats_all_acc, stats_all_by_kappa[ik])
        merge_stats!(stats_lo_acc, stats_lo_by_kappa[ik])
    end

    stats = finalize_statistics(stats_acc, p)
    stats_all = finalize_statistics(stats_all_acc, p)
    # The entry-age cross-section, reduced with the same machinery so it cannot
    # drift from the windowed one. Only the two asset RATIOS are carried over,
    # and both divide by the WINDOW's mean labor income -- the same denominator
    # `medianAssetsToMeanLaborIncome` uses, matching the history-independent
    # solver field for field.
    stats_lo = finalize_statistics(stats_lo_acc, p)
    stats = merge(stats, (;
        meanAssetsAtStatsAgeLoToMeanLaborIncome =
            safe_ratio(stats_lo.meanAssets, stats.meanLaborIncome),
        medianAssetsAtStatsAgeLoToMeanLaborIncome =
            safe_ratio(stats_lo.medianAssets, stats.meanLaborIncome),
    ))
    welfare = finalize_welfare(
        welfare_value_function_by_kappa, welfare_simulation_by_kappa, p,
    )
    # Unchanged denominator: the clamped share is a property of the whole
    # forward pass, not of the calibration window.
    clamped_share = stats_all_acc.total_mass > 0.0 ?
                    sum(clamped_by_kappa) / stats_all_acc.total_mass : 0.0
    # Every kappa now runs the full maxAge, so the PV tail opens at maxAge for
    # all of them and settledAge is no longer a per-kappa quantity.
    diagnostics = (; vIters = vIters, vGap = vGap,
                   htmRolloverClipped = htm.clipped, settledAge = p.maxAge,
                   convergedAge = maximum(converged_by_kappa),
                   convergedAgeByKappa = converged_by_kappa,
                   finalDrift = drift_by_kappa)
    return (; C = C, H = H, Y = Y, A = A), stats, stats_all, welfare, clamped_share, diagnostics
end

function government_residual_at_lambda(lambda::Float64, p::HDParams)
    aggs, stats, stats_all, welfare, clamped_share, diag =
        solve_aggregates_for_lambda(lambda, p)

    # The budget is STILL a present value over ages: the agent is infinitely
    # lived but the cohort's aggregates vary over its life, so this is not the
    # stationary condition a steady-state Bewley model would use.
    #
    # The path is iterated to maxAge for every kappa, after which Y_j - C_j is
    # constant and the remaining terms sum in closed form. Jc is maxAge now; it
    # used to be maximum(settled_by_kappa), which made the loop below read the
    # band where kappas had dropped out one at a time and the aggregates were
    # partial sums across kappa:
    #
    #   sum_{j>Jc} qGov^j (Y-C)_inf = (Y-C)_inf * qGov^(Jc+1) / (1 - qGov).
    #
    # Iterating instead would need ln(1e-6)/ln(0.99) ~ 1375 ages for the same
    # accuracy at qGov = 0.99.
    Jc = diag.settledAge
    lhs = 0.0
    for j in 1:Jc
        lhs += p.qGov^(j - 1) * (aggs.Y[j] - aggs.C[j])
    end
    tail_flow = aggs.Y[Jc] - aggs.C[Jc]
    lhs += tail_flow * p.qGov^Jc / (1.0 - p.qGov)
    lhs *= (1.0 - p.qGov)
    rhs = p.G                       # (1 - qGov^inf) * G = G
    residual = lhs - rhs

    eq = (;
        lambda = lambda,
        govBudgetResidual = residual,
        govBudgetLHS = lhs,
        govBudgetRHS = rhs,
        C = aggs.C,
        H = aggs.H,
        Y = aggs.Y,
        A = aggs.A,
        consumptionPV = discounted_sum_with_tail(aggs.C, p.qGov),
        outputPV = discounted_sum_with_tail(aggs.Y, p.qGov),
        statistics = stats,
        statisticsAllAges = stats_all,
        welfare = welfare,
        sClampedMassShare = clamped_share,
        parameters = p,
    )
    return residual, eq
end

# -----------------------------------------------------------------------------
# Equilibrium solver: Brent on the government-budget residual in lambda
# -----------------------------------------------------------------------------
"""
    solve_history_dependent_tax(p::HDParams)

Solve for the tax level `lambda` clearing the government budget constraint.
Residuals are cached per lambda, a grid fallback over `nLambdaSearch` values
handles brackets without a sign change, and the best evaluated equilibrium is
kept as a fallback. Returns the equilibrium NamedTuple.
"""
function solve_history_dependent_tax(p::HDParams)
    start_time = time()

    if p.verbose
        println("\n=== History-dependent tax finite-horizon solver ===")
        print_solver_options(p)
        flush(stdout)
    end

    eval_cache = Dict{Float64,Float64}()
    best_abs_residual = Ref(Inf)
    best_eq = Ref{Any}(nothing)
    last_lambda = Ref(NaN)
    last_eq = Ref{Any}(nothing)

    function evaluate_residual(lambda::Float64)
        key = Float64(lambda)
        haskey(eval_cache, key) && return eval_cache[key]
        residual, eq = government_residual_at_lambda(key, p)
        eval_cache[key] = residual
        last_lambda[] = key
        last_eq[] = eq
        if isfinite(residual) && abs(residual) < best_abs_residual[]
            best_abs_residual[] = abs(residual)
            best_eq[] = eq
        end
        if p.verbose
            @printf("lambda = %.8f: residual = %.8e\n", key, residual)
            flush(stdout)
        end
        return residual
    end

    function full_equilibrium(lambda::Float64)
        key = Float64(lambda)
        if last_eq[] !== nothing && last_lambda[] == key
            return last_eq[]
        elseif best_eq[] !== nothing && best_eq[].lambda == key
            return best_eq[]
        end
        _, eq = government_residual_at_lambda(key, p)
        return eq
    end

    r_low = evaluate_residual(p.lambdaMin)
    r_high = evaluate_residual(p.lambdaMax)
    lambda_low, lambda_high = p.lambdaMin, p.lambdaMax

    if !isfinite(r_low) || !isfinite(r_high) || sign(r_low) == sign(r_high)
        if p.verbose
            @printf("\nNo sign change on requested bracket. Searching %d lambda values.\n",
                    p.nLambdaSearch)
            flush(stdout)
        end
        grid = collect(range(p.lambdaMin, p.lambdaMax, length = p.nLambdaSearch))
        residuals = [evaluate_residual(Float64(l)) for l in grid]
        bracket = find_bracket(grid, residuals)
        if bracket === nothing
            best_eq[] === nothing &&
                error("Could not evaluate any finite government residual")
            return attach_elapsed(best_eq[], start_time, p;
                                  converged = false, bracketWarning = true)
        end
        i_low, i_high = bracket
        lambda_low, lambda_high = grid[i_low], grid[i_high]
        r_low, r_high = residuals[i_low], residuals[i_high]
    end

    local eq
    try
        lambda_root = Roots.find_zero(
            evaluate_residual, (lambda_low, lambda_high), Roots.Brent();
            xatol = p.tolLambda,
            maxevals = max(p.maxIterLambda, 20),
        )
        eq = full_equilibrium(Float64(lambda_root))
        r_root = eq.govBudgetResidual
        converged = isfinite(r_root) && abs(r_root) <= p.tolGovBudget
        eq = attach_elapsed(eq, start_time, p;
                            converged = converged,
                            bracketWarning = false,
                            rootResidualWarning = !converged)
    catch err
        best_eq[] === nothing && rethrow(err)
        eq = attach_elapsed(best_eq[], start_time, p;
                            converged = false,
                            bracketWarning = false,
                            rootSolverWarning = true)
    end
    if p.verbose && eq.sClampedMassShare > 1e-6
        @printf("WARNING: %.4e of mass had s' clamped to the s-grid bounds; consider widening nS1/nS2 or the s bounds.\n",
                eq.sClampedMassShare)
    end
    return eq
end

function attach_elapsed(eq, start_time::Float64, p::HDParams; kwargs...)
    elapsed = time() - start_time
    eq_with_elapsed = merge(eq, (; kwargs..., elapsedSeconds = elapsed))
    if p.verbose
        print_lambda_warnings(eq_with_elapsed)
        @printf("total solve time          = %.3f seconds\n", elapsed)
        flush(stdout)
    end
    return eq_with_elapsed
end

# -----------------------------------------------------------------------------
# Self-contained consistency check (no external solver required)
# -----------------------------------------------------------------------------
"""
    check_history_independent_limit(; kwargs...)

Solve the model at the history-independent limit `mu1 = mu2 = 0` (then
`theta0 = 1`, `pow = 1 - tau`, and the s-grids collapse to a single point, so
the past-income stocks are identically zero and the model reduces exactly to
Section 1.1). This uses ONLY this module -- no external solver.

Two internal consistency conditions are checked and reported:
  * the value-function welfare and the simulation welfare agree (the standard
    cross-check that backward induction and the forward distribution are
    mutually consistent), and
  * the single s-grid point is exactly 0 and no mass is clamped.

Returns the equilibrium NamedTuple. `kwargs` override `HD_SETTINGS`.
"""
function check_history_independent_limit(; kwargs...)
    p = make_history_dependent_params(; kwargs...,
                                      mu1 = 0.0, mu2 = 0.0, nS1 = 1, nS2 = 1)
    eq = solve_history_dependent_tax(p)

    welfare_gap = abs(eq.welfare.overallDifference)
    s1_zero = length(p.s1_grid) == 1 && p.s1_grid[1] == 0.0
    s2_zero = length(p.s2_grid) == 1 && p.s2_grid[1] == 0.0

    println("\n=== History-independent limit (mu1 = mu2 = 0) ===")
    @printf("%-42s %14.8f\n", "theta0 (should be 1)", p.theta0)
    @printf("%-42s %14.8f\n", "pow = 1 - tau", p.pow)
    @printf("%-42s %14s\n", "s1, s2 single point at 0",
            string(s1_zero && s2_zero))
    @printf("%-42s %14.3e\n", "s' clamped mass share", eq.sClampedMassShare)
    @printf("%-42s %14.8f\n", "lambda", eq.lambda)
    @printf("%-42s %14.8f\n", "mean assets / mean labor income",
            eq.statistics.meanAssetsToMeanLaborIncome)
    @printf("%-42s %14.8f\n", "median assets / mean labor income",
            eq.statistics.medianAssetsToMeanLaborIncome)
    @printf("%-42s %14.8f\n", "share negative liquid assets",
            eq.statistics.shareNegativeLiquidAssets)
    @printf("%-42s %14.8f\n", "welfare (value function)",
            eq.welfare.overallValueFunction)
    @printf("%-42s %14.3e\n", "welfare VF vs simulation gap", welfare_gap)
    return eq
end

# Load user-editable settings (defines HD_SETTINGS and
# make_history_dependent_params inside this module).
include("model_settings.jl")

end # module HistoryDependentTaxInfinite
