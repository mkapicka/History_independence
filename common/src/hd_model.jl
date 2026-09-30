# =============================================================================
# hd_model.jl
#
# Machinery shared by the history-dependent solvers: the past-income
# stock grids s1, s2 and their moments, and the joint (a', h) block solve. Each
# block names the directories it came from, because the hd family does not
# agree on all of them.
#
# Nothing here is exported. Each of these names is defined with a DIFFERENT
# implementation in the sibling directories that do not share this version, and
# a directory that both `using`s an exported name and defines its own gets a
# Julia error. So the directories that want these import them by name:
#
#     using BewleyCommon: build_s_grid, solve_block!
#
# which also documents at the top of each solver exactly what it takes from the
# package. This follows build_labor_grid, unexported for the same reason.
#
# Marek Kapicka, 2026
# =============================================================================

# ---- s_stock_moments, build_s_grid: hd and hd_htm, the finite-horizon pair.

"""
    s_stock_moments(mu, J; alpha, mu1, mu2, theta0, beta, rho, tau, eta,
                    sigma_omega, sigma_epsilon, sigma_kappa)

Mean and variance of the past-income stock at each age `0:J`, in closed form.

Everything in the no-savings model is jointly normal and hours are closed-form,
so the stock is EXACTLY normal at each age and its moments need no simulation.
Unwinding `s' = mu*(m + ln h + s)` from `s_0 = 0` with `m = kappa + z + eps`
gives `s_j = sum_{k=0}^{j-1} a_k * x_k` with `a_k = mu^(j-k)` and
`x_k = kappa + z_k + eps_k + ln h_k`, hence

    E[s_j]   = sum_k a_k * (m_kappa + E[z_k] + m_eps + E_lnh[k]),
               E[z_k] = m_omega * (1 - rho^(k+1)) / (1 - rho)
    Var[s_j] = (sum_k a_k)^2 * sigma_kappa^2        # permanent, loads on every k
             + (sum_k a_k^2) * sigma_epsilon^2      # iid
             + sigma_omega^2 * sum_i c_i^2,         # AR(1)
               c_i = a_i + rho*c_{i+1},  c_j = 0

the last by the backward recursion, which needs no special case at rho = mu.
Hours are `ln h_k = (log(1-tau) + log(Theta_k))/(1+eta)` with
`Theta_j = sum_{s=0}^{J-j} beta^s theta_s` and
`theta_k = theta0*(alpha*mu1^k + (1-alpha)*mu2^k)`. Under log utility and the
HSV tax, hours are independent of productivity, so they move the mean of the
stock but not its variance.

Verified against a 200,000-path Monte Carlo of the recursion: mean and variance
agree to MC error at every age, across the baseline roots and two other pairs.

APPROXIMATE FOR THIS MODEL, deliberately. The Bewley households face assets, a
borrowing limit, a discrete hours grid and the `hMax` cap, so their hours differ
from the closed-form rule above. That is acceptable because these moments only
decide where grid points are PLACED: a grid that spans the reachable range
cannot bias the solution, only how efficiently it resolves it.
"""
function s_stock_moments(mu::Real, J::Int; alpha, mu1, mu2, theta0, beta, rho,
                         tau, eta, sigma_omega, sigma_epsilon, sigma_kappa)
    mu = Float64(mu)
    ages = 0:J
    theta = [theta0 * (alpha * mu1^k + (1.0 - alpha) * mu2^k) for k in ages]
    Theta = reverse(cumsum([beta^s * theta[s+1] for s in ages]))
    all(Theta .> 0.0) ||
        error("Theta_j must be positive at every age to place quantile s-grids")
    E_lnh = [(log(1.0 - tau) + log(Theta[k+1])) / (1.0 + eta) for k in ages]

    m_omega = -0.5 * sigma_omega^2
    m_eps   = -0.5 * sigma_epsilon^2
    m_kappa = -0.5 * sigma_kappa^2
    E_z(k)  = m_omega * (1.0 - rho^(k + 1)) / (1.0 - rho)

    means = zeros(J + 1)
    vars  = zeros(J + 1)
    for j in ages
        j == 0 && continue                          # s_0 = 0 exactly
        a = [mu^(j - k) for k in 0:(j-1)]
        means[j+1] = sum(a[k+1] * (m_kappa + E_z(k) + m_eps + E_lnh[k+1])
                         for k in 0:(j-1))
        c = zeros(j)
        cnext = 0.0
        for i in (j-1):-1:0
            c[i+1] = a[i+1] + rho * cnext
            cnext  = c[i+1]
        end
        vars[j+1] = sum(a)^2 * sigma_kappa^2 +
                    sum(a .^ 2) * sigma_epsilon^2 +
                    sigma_omega^2 * sum(c .^ 2)
    end
    return means, vars
end

"""
    build_s_grid(mu, nS, J, kappa_grid, z_grid, eps_grid, s_hours_floor, hMax)

Linear grid for one past-income stock, spanning the range REACHABLE in a
(J+1)-period life rather than the infinite-horizon support.

With m = kappa + z + eps + ln h, the recursion s' = mu*(m + s) started from
s_0 = 0 unwinds to

    s_j = sum_{k=0}^{j-1} mu^(j-k) * m_k,

so the stock is bounded by `scale * m`, where

    scale = sum_{k=1}^{J+1} mu^k = mu*(1 - mu^(J+1))/(1 - mu).

The sum runs to J+1, not J: the backward induction and the distribution both
evaluate s' at the terminal age, producing one more update than the J ages of
life would suggest. Bounding at J instead leaves a small but nonzero clamped
mass (6.5e-06 at the baseline roots, 3.3e-04 at mu = 0.999), which is exactly
the bias this grid is supposed to avoid.

This replaces the earlier bound `mu/(1-mu)`, which is the same sum taken to
infinity. The difference is large exactly where the grid hurts: at J = 39 the
infinite bound is 80.30 against a reachable 31.35 at mu = 0.9877, and 999.00
against 39.23 at mu = 0.999. Points spent outside the reachable set are pure
waste, so tightening the bound raises resolution at fixed nS with no
approximation -- the truncated states cannot occur.

`ln h` is bounded below using s_hours_floor (log(hMin) would blow the grid up).
Bounds are extended to include the initial value 0. Returns `[0.0]` when mu = 0.

`scale` stays finite at mu = 1, where it equals J+1, so unit-root stocks are
representable by this formula (the `mu < 1` check in the constructor is a
separate restriction).

`method` controls the SPACING of the points inside those bounds:

  * `:linear`   equally spaced (the default, and what this file always did);
  * `:quantile` placed at quantiles of the age-pooled distribution of the stock,
    computed in closed form by `s_stock_moments` and passed in through
    `moment_args`. The distribution is concentrated well inside the reachable
    range, so equal spacing wastes points in tails the model rarely visits.

The quantile grid still spans the full reachable range: the two endpoints are
pinned to `s_lo` and `s_hi`, so no reachable state is truncated and the clamped
mass stays at zero. One interior node is snapped to exactly 0.0, mirroring
`asset_grid_with_zero`, because the initial condition `s_0 = 0` is looked up on
this grid. Snapping preserves strict ordering: the node nearest zero always has
its neighbours straddling it.
"""
function build_s_grid(mu::Real, nS::Int, J::Int, kappa_grid, z_grid, eps_grid,
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
    # Summed directly rather than via mu*(1-mu^(J+1))/(1-mu) so mu = 1 needs no
    # special case; J is at most a few hundred, so the loop is free.
    scale = sum(mu^k for k in 1:(J + 1))
    s_lo = min(scale * m_lo, 0.0)
    s_hi = max(scale * m_hi, 0.0)
    s_hi > s_lo || error("degenerate s-grid bounds [$s_lo, $s_hi]")

    n = max(nS, 2)
    method === :linear && return collect(range(s_lo, s_hi, length = n))

    moment_args === nothing &&
        error("build_s_grid with method = :quantile needs moment_args")
    means, vars = s_stock_moments(mu, J; moment_args...)

    # Age-pooled CDF, equal weight per age, over ages 1..J only. Age 0 is
    # excluded: it would enter as a point mass at s = 0, and at large nS two or
    # more quantile levels land inside that jump, duplicate the node and fail
    # the strict-monotonicity check. Excluding it costs nothing, since the
    # snap-to-zero node below already represents the initial condition. See
    # NOTES.md.
    function F_stock(x::Float64)
        acc = 0.0
        for j in 1:J
            v = vars[j+1]
            acc += v <= 0.0 ? (x >= means[j+1] ? 1.0 : 0.0) :
                              normal_cdf((x - means[j+1]) / sqrt(v))
        end
        return acc / J
    end

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
    if all(<=(0.0), @view vars[2:end])
        @warn "s-stock distribution is degenerate; falling back to :linear spacing" mu nS J
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
              "(mu = $mu, nS = $nS, J = $J); this is a bug in build_s_grid")
    return grid
end

# ---- BlockScratch: all four hd directories agree on both the struct and
# its constructor, so they move together; solve_block! needs the type.

# Per-thread scratch. EVh starts zeroed because the terminal age is solved
# first and reads it as the (zero) continuation without writing it.
struct BlockScratch
    inc::Vector{Float64}
    l1v::Vector{Int}
    h1v::Vector{Int}
    w1v::Vector{Float64}
    l2v::Vector{Int}
    h2v::Vector{Int}
    w2v::Vector{Float64}
    ih_ub::Vector{Int}          # monotone upper bound on optimal ih, per a'
    EVh::Matrix{Float64}        # EV at (h, a')
end

function BlockScratch(nH::Int, nA::Int)
    return BlockScratch(
        Vector{Float64}(undef, nH),
        Vector{Int}(undef, nH), Vector{Int}(undef, nH), Vector{Float64}(undef, nH),
        Vector{Int}(undef, nH), Vector{Int}(undef, nH), Vector{Float64}(undef, nH),
        Vector{Int}(undef, nA), zeros(nH, nA),
    )
end

# ---- solve_block!: hd and hd_htm. The infinite-horizon pair has its own
# version, which is why this one is not exported.

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
                      age::Int, ia_first::Int, has_continuation::Bool,
                      m_base::Float64, coeff::Float64, p::AbstractBewleyParams)
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
            policyAIndex[ia, is1, is2, iz, ie, age] = Int32(best_iap)
            policyH[ia, is1, is2, iz, ie, age] = p.h_grid[best_ih]
        end
    end
    return nothing
end

# ---- initial_asset_weights: hd, hd_htm and hdinf. hdinf_htm differs.

"""
    initial_asset_weights(kappa, p)

Grid placement of the initial asset holding, as `(left, right, right_weight)`
from the same Young lottery used for a'. Both the birth value function and the
simulated initial distribution go through THIS function; if they disagree the
value-function/simulation welfare cross-check breaks, which is the guard wanted.
"""
function initial_asset_weights(kappa::Float64, p::AbstractBewleyParams)
    a0 = p.a0_scales_with_kappa ? p.a0 * exp(kappa) : p.a0
    return grid_lookup_weights(p.a_grid, clamp(a0, first(p.a_grid), last(p.a_grid)))
end
