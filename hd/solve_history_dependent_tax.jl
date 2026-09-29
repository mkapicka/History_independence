# =============================================================================
# solve_history_dependent_tax.jl  --  STANDALONE
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
# Self-contained: it does not include or call the history-independent code. All
# shared infrastructure is replicated inside the module `HistoryDependentTax`,
# so the two codebases can be loaded in the same session. Exported API:
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

module HistoryDependentTax

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

export HDParams, HD_SETTINGS, make_history_dependent_params,
       solve_history_dependent_tax, print_hd_equilibrium_summary,
       check_history_independent_limit

const VINFEASIBLE = -1.0e18

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
"""
    HDParams(; kwargs...)

Parameters for the history-dependent tax model. Model and solver keywords are
REQUIRED (no defaults): construct via `make_history_dependent_params`, which
fills them from `HD_SETTINGS` -- the single source of truth in
model_settings.jl. `alpha` is a free setting: pass a number, or `:paper` for the
paper mixture `(rho - mu1)/(mu2 - mu1)` (1 when the roots coincide). `theta0` is
DERIVED and cannot be set: it follows from the finite-horizon restriction
`sum_{s=0}^{J} beta^s theta_s = 1`, so it depends on `(alpha, mu1, mu2, beta, J)`.
The asset choice is always grid search and hours are always chosen on the labor
grid (the static labor FOC is invalid because hours move s').
"""
struct HDParams <: AbstractBewleyParams
    # preferences and tax
    beta::Float64
    eta::Float64
    phi::Float64
    tau::Float64
    theta0::Float64
    alpha::Float64
    mu1::Float64
    mu2::Float64
    pow::Float64                     # (1 - tau) * theta0

    # horizon and shocks
    J::Int

    # Model age is the 1-based array index: model age 1 is j = 0, at real age
    # age0_real. Statistics are averaged over model ages stats_age_lo to
    # stats_age_hi inclusive; `stats_age_hi = 0` resolves to J+1. The
    # equilibrium reports both the window and all ages, as `hi` does.
    age0_real::Int
    stats_age_lo::Int
    stats_age_hi::Int

    # Initial assets at model age 1. a0_scales_with_kappa multiplies it by
    # exp(kappa), as wages and the borrowing limit already scale with the
    # permanent type. Placed by the same Young lottery used for a'.
    a0::Float64
    a0_scales_with_kappa::Bool
    z_grid::Vector{Float64}
    Pz::Matrix{Float64}
    z0_probs::Vector{Float64}
    eps_grid::Vector{Float64}
    Peps::Vector{Float64}
    kappa_grid::Vector{Float64}
    Pkappa::Vector{Float64}
    z_discretization_method::Symbol
    tauchen_width::Float64
    rho::Float64

    # asset grid
    bbar::Float64
    aMax::Float64
    nA::Int
    a_grid::Vector{Float64}
    asset_grid_method::Symbol
    asset_grid_curvature_borrow::Float64
    asset_grid_curvature_save::Float64
    asset_grid_borrow_share::Float64
    asset_grid_zero_share::Float64
    asset_grid_zero_width::Float64

    # financial and government
    qBorr::Float64
    qSav::Float64
    qGov::Float64
    G::Float64

    # labor
    hMin::Float64
    hMax::Float64
    h_grid::Vector{Float64}
    h_grid_disutility::Vector{Float64}
    log_h_grid::Vector{Float64}
    h_income_power::Vector{Float64}  # h_grid .^ pow
    labor_grid_spacing::Symbol       # :log or :uniform

    # past-income stocks
    s1_grid::Vector{Float64}
    s2_grid::Vector{Float64}
    s_factor::Matrix{Float64}        # exp(pow*(alpha*s1 + (1-alpha)*s2))
    s_hours_floor::Float64
    s_grid_method::Symbol            # :linear or :quantile

    # lambda solver
    lambdaMin::Float64
    lambdaMax::Float64
    nLambdaSearch::Int
    maxIterLambda::Int
    tolLambda::Float64
    tolGovBudget::Float64

    # output and solver behavior
    verbose::Bool
    massTol::Float64
    collect_distributions::Bool
    exploit_hours_monotonicity::Bool
end

function HDParams(;
    # No defaults here: HD_SETTINGS is the single source of truth, applied
    # through make_history_dependent_params. A direct HDParams() call missing a
    # keyword raises UndefKeywordError rather than solving a different model.
    beta,
    eta,
    phi,
    tau,
    alpha,
    mu1,
    mu2,
    J,
    age0_real,
    stats_age_lo,
    stats_age_hi,
    a0,
    a0_scales_with_kappa,
    rho,
    sigma_omega,
    sigma_epsilon,
    sigma_kappa,
    nZ,
    nEps,
    nKappa,
    z_discretization_method,
    bbar,
    aMax,
    nA,
    asset_grid_method,
    asset_grid_curvature_borrow,
    asset_grid_curvature_save,
    asset_grid_borrow_share,
    asset_grid_zero_share,
    asset_grid_zero_width,
    qBorr,
    qSav,
    qGov,
    G,
    hMin,
    hMax,
    labor_grid_size,
    labor_grid_spacing,
    exploit_hours_monotonicity,
    nS1,
    nS2,
    s_hours_floor,
    s_grid_method,
    lambdaMin,
    lambdaMax,
    nLambdaSearch,
    maxIterLambda,
    tolLambda,
    tolGovBudget,
    verbose,
    massTol,
    collect_distributions,
    # Defaults survive ONLY where nothing is duplicated: derived formulas,
    # empty-grid sentinels meaning "build the grid", and optional knobs that
    # HD_SETTINGS deliberately omits.
    omega_mean = -0.5 * sigma_omega^2,
    epsilon_mean = -0.5 * sigma_epsilon^2,
    kappa_mean = -0.5 * sigma_kappa^2,
    tauchen_width = 3.0,
    z_initial = 0.0,
    a_grid = Float64[],
    h_grid = Float64[],
)
    z_discretization_method in (:rouwenhorst, :tauchen) ||
        error("z_discretization_method must be :rouwenhorst or :tauchen")

    z_grid, Pz, z0_probs = build_markov_shock(
        "z", nZ, rho, omega_mean, sigma_omega, z_initial,
        tauchen_width, z_discretization_method,
    )
    eps_grid, Peps = build_iid_normal_shock("eps", nEps, epsilon_mean, sigma_epsilon)
    kappa_grid, Pkappa = build_iid_normal_shock("kappa", nKappa, kappa_mean, sigma_kappa)

    0.0 <= beta < 1.0 || error("beta must satisfy 0 <= beta < 1")
    tau < 1.0 || error("tau must be less than one")
    0.0 <= mu1 < 1.0 || error("mu1 must be in [0, 1)")
    0.0 <= mu2 < 1.0 || error("mu2 must be in [0, 1)")
    nS1 >= 1 || error("nS1 must be at least 1")
    nS2 >= 1 || error("nS2 must be at least 1")
    bbar <= 0.0 || error("Use bbar <= 0. For a borrowing limit B > 0, pass bbar = -B.")
    hMin > 0.0 || error("hMin must be positive")
    hMax > hMin || error("hMax must exceed hMin")
    0.0 < s_hours_floor < hMax || error("s_hours_floor must lie in (0, hMax)")
    labor_grid_size >= 2 || error("labor_grid_size must be at least 2")
    labor_grid_spacing in (:log, :uniform) ||
        error("labor_grid_spacing must be :log or :uniform")
    s_grid_method in (:linear, :quantile) ||
        error("s_grid_method must be :linear or :quantile")
    asset_grid_method in (:nonuniform, :linear) ||
        error("asset_grid_method must be :nonuniform or :linear")
    0.0 <= asset_grid_zero_share < 1.0 ||
        error("asset_grid_zero_share must be in [0, 1)")
    asset_grid_zero_width >= 0.0 || error("asset_grid_zero_width must be nonnegative")

    # The mixture weight is a free setting: a number, used as given, or
    # :paper for (rho - mu1)/(mu2 - mu1), which is what build_theta uses in the
    # no-savings code and stays consistent when the roots move. Equal roots are
    # the one-root case, where alpha is irrelevant and set to 1.
    #
    # alpha lies in [0, 1] iff the roots bracket rho. Outside that range the
    # mixture is signed, which warns rather than errors; the denom and pow
    # checks below catch the degenerate cases.
    alpha = if alpha === :paper
        isapprox(mu1, mu2) ? 1.0 : (rho - mu1) / (mu2 - mu1)
    elseif alpha isa Real
        Float64(alpha)
    else
        error("alpha must be a number or :paper (got $(repr(alpha)))")
    end
    if !(0.0 <= alpha <= 1.0)
        @warn "alpha is outside [0, 1]: the two root blocks carry opposite signs" alpha rho mu1 mu2
    end

    # Promise keeping pins down theta0 through the FINITE-horizon
    # normalization,
    #
    #   sum_{s=0}^{J} beta^s theta_s = 1,   theta_s = theta0*M_s,
    #   M_s = alpha*mu1^s + (1-alpha)*mu2^s,
    #
    # written as an explicit sum so it matches build_theta in the no-savings
    # code term for term. The infinite-horizon version understates theta0; see
    # NOTES.md. mu = 0 contributes M_0 = 1 and M_s = 0 for s >= 1, so the
    # mu1 = mu2 = 0 limit still gives theta0 = 1 exactly.
    M = [alpha * mu1^s + (1.0 - alpha) * mu2^s for s in 0:J]
    denom = sum(beta^s * M[s+1] for s in 0:J)
    denom > 0.0 || error("invalid (alpha, mu1, mu2): theta0 denominator <= 0")
    theta0 = 1.0 / denom
    pow = (1.0 - tau) * theta0
    pow > 0.0 || error("(1 - tau) * theta0 must be positive")

    if isempty(a_grid)
        amin = minimum(bbar * exp(kappa + rho * z) for kappa in kappa_grid for z in z_grid)
        a_grid = asset_grid_with_zero(amin, aMax, nA;
                                      method = asset_grid_method,
                                      curvature_borrow = asset_grid_curvature_borrow,
                                      curvature_save = asset_grid_curvature_save,
                                      borrow_share = asset_grid_borrow_share,
                                      zero_share = asset_grid_zero_share,
                                      zero_width = asset_grid_zero_width)
    else
        a_grid = sort(collect(Float64.(a_grid)))
        nA = length(a_grid)
        any(iszero, a_grid) || error("a_grid must include 0.0 exactly for the initial condition")
    end
    minimum(a_grid) <= 0.0 <= maximum(a_grid) || error("a_grid must contain 0")
    maximum(a_grid) <= aMax + 1e-12 || error("a_grid has points above aMax")

    h_grid = BewleyCommon.build_labor_grid(hMin, hMax, labor_grid_size, h_grid;
                              spacing = labor_grid_spacing)
    h_grid_disutility = phi .* (h_grid .^ (1.0 + eta)) ./ (1.0 + eta)
    log_h_grid = log.(h_grid)
    h_income_power = h_grid .^ pow

    # Effective hours floor for the s-grid bounds: with hMin >= s_hours_floor
    # every realizable s' lies inside the grid (no clamping from low hours).
    s_floor = max(Float64(hMin), Float64(s_hours_floor))
    # Shared inputs to the closed-form stock moments; ignored by :linear.
    moment_args = (; alpha, mu1, mu2, theta0, beta, rho, tau, eta,
                   sigma_omega, sigma_epsilon, sigma_kappa)
    s1_grid = build_s_grid(mu1, nS1, J, kappa_grid, z_grid, eps_grid,
                           s_floor, hMax; method = s_grid_method, moment_args)
    s2_grid = build_s_grid(mu2, nS2, J, kappa_grid, z_grid, eps_grid,
                           s_floor, hMax; method = s_grid_method, moment_args)
    s_factor = Matrix{Float64}(undef, length(s1_grid), length(s2_grid))
    for i1 in eachindex(s1_grid), i2 in eachindex(s2_grid)
        s_factor[i1, i2] =
            exp(pow * (alpha * s1_grid[i1] + (1.0 - alpha) * s2_grid[i2]))
    end

    return HDParams(
        beta, eta, phi, tau, theta0, alpha, mu1, mu2, pow,
        J, age0_real, stats_age_lo, stats_age_hi,
        Float64(a0), a0_scales_with_kappa,
        z_grid, Pz, z0_probs, eps_grid, Peps, kappa_grid, Pkappa,
        z_discretization_method, tauchen_width, rho,
        bbar, aMax, nA, a_grid,
        asset_grid_method, asset_grid_curvature_borrow,
        asset_grid_curvature_save, asset_grid_borrow_share,
        asset_grid_zero_share, asset_grid_zero_width,
        qBorr, qSav, qGov, G,
        hMin, hMax, h_grid, h_grid_disutility, log_h_grid, h_income_power,
        labor_grid_spacing,
        s1_grid, s2_grid, s_factor, Float64(s_hours_floor), s_grid_method,
        lambdaMin, lambdaMax, nLambdaSearch, maxIterLambda,
        tolLambda, tolGovBudget,
        verbose, massTol, collect_distributions, exploit_hours_monotonicity,
    )
end

"""
    S_GRID_UNIFORM_BLEND

Weight on a uniform when placing `:quantile` s-grid nodes, so that the outermost
cells stay bounded instead of spanning most of the reachable range. Tuned on the
baseline roots (mu1 = 0.6061, mu2 = 0.9877, J = 39), welfare error against an
nS2 = 401 reference, as a multiple of the `:linear` error at the same nS2
(higher is better, and below 1.0 means worse than equal spacing):

    blend w     nS2=7    nS2=15    nS2=31    nS2=61   clamped at nS2=61
      0.00      0.90x     0.80x     2.41x    11.16x        3.7e-04
      0.05      0.91x     0.96x     4.52x    43.01x        4e-05
      0.15      1.14x     1.56x    11.96x    61.94x        9e-09
      0.30      1.35x     3.75x    12.64x    74.96x        3e-14
      0.50       --        --        --      27.57x        0

Improvement is monotone in w up to 0.30 and then reverses -- by w = 0.50 enough
points have been pulled back into the tails that the gain at nS2 = 61 falls from
75x to 28x. Pure quantile spacing (w = 0) is worse than equal spacing at
nS2 <= 15, for the reason documented at the blend itself. w is a constant rather
than a setting because it trades one kind of grid error against another with no
economic content.
"""
const S_GRID_UNIFORM_BLEND = 0.30


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


# -----------------------------------------------------------------------------
# Statistics accumulator (same fields and semantics as the history-independent
# solver, so downstream statistics are directly comparable)
# -----------------------------------------------------------------------------
mutable struct StatsAccumulator
    asset_mass::Vector{Float64}
    distribution_weights::Vector{Float64}
    hours_values::Vector{Float64}
    consumption_values::Vector{Float64}
    total_mass::Float64
    sum_current_assets::Float64
    sum_labor_income::Float64
    sum_borrowing_limit::Float64
    sum_effective_borrowing_limit::Float64
    borrowing_limit_mass::Float64
    negative_asset_mass::Float64
    zero_asset_mass::Float64
    borrowing_constraint_mass::Float64
    upper_bound_mass::Float64
    hours_upper_bound_mass::Float64
    max_next_assets::Float64
    max_hours::Float64
    max_material_next_assets::Float64
    max_material_hours::Float64
end

function StatsAccumulator(nA::Int)
    nA > 0 || error("nA must be positive")
    return StatsAccumulator(
        zeros(nA), Float64[], Float64[], Float64[],
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        -Inf, -Inf, -Inf, -Inf,
    )
end

function merge_stats!(dest::StatsAccumulator, src::StatsAccumulator)
    length(dest.asset_mass) == length(src.asset_mass) ||
        error("Cannot merge statistics with different asset-grid sizes")
    dest.asset_mass .+= src.asset_mass
    append!(dest.distribution_weights, src.distribution_weights)
    append!(dest.hours_values, src.hours_values)
    append!(dest.consumption_values, src.consumption_values)
    dest.total_mass += src.total_mass
    dest.sum_current_assets += src.sum_current_assets
    dest.sum_labor_income += src.sum_labor_income
    dest.sum_borrowing_limit += src.sum_borrowing_limit
    dest.sum_effective_borrowing_limit += src.sum_effective_borrowing_limit
    dest.borrowing_limit_mass += src.borrowing_limit_mass
    dest.negative_asset_mass += src.negative_asset_mass
    dest.zero_asset_mass += src.zero_asset_mass
    dest.borrowing_constraint_mass += src.borrowing_constraint_mass
    dest.upper_bound_mass += src.upper_bound_mass
    dest.hours_upper_bound_mass += src.hours_upper_bound_mass
    dest.max_next_assets = max(dest.max_next_assets, src.max_next_assets)
    dest.max_hours = max(dest.max_hours, src.max_hours)
    dest.max_material_next_assets =
        max(dest.max_material_next_assets, src.max_material_next_assets)
    dest.max_material_hours = max(dest.max_material_hours, src.max_material_hours)
    return dest
end

upper_bound_share_tol() = 1e-8
function finalize_statistics(stats::StatsAccumulator, p::HDParams)
    total_mass = stats.total_mass
    mean_assets = stats.sum_current_assets / total_mass
    mean_labor_income = stats.sum_labor_income / total_mass
    # Mid-cumulative interpolation rather than StatsBase's weighted-quantile
    # convention, which is biased low on a coarse nonuniform grid holding a
    # discretized continuous distribution. See `interpolated_weighted_quantile`
    # in common/grids.jl, and NOTES.md for the measured comparison.
    median_assets = interpolated_weighted_quantile(p.a_grid, stats.asset_mass, 0.5)
    # Both limits exist only at ages j = 0,...,J-1, so they are averaged over
    # the mass of those ages rather than over the whole population.
    mean_borrowing_limit = safe_ratio(stats.sum_borrowing_limit,
                                      stats.borrowing_limit_mass)
    mean_effective_borrowing_limit = safe_ratio(stats.sum_effective_borrowing_limit,
                                                stats.borrowing_limit_mass)
    share_at_effective_borrowing_constraint = stats.borrowing_constraint_mass / total_mass
    share_at_asset_upper_bound = stats.upper_bound_mass / total_mass
    share_at_hours_upper_bound = stats.hours_upper_bound_mass / total_mass
    max_next_assets = isfinite(stats.max_next_assets) ? stats.max_next_assets : NaN
    max_hours = isfinite(stats.max_hours) ? stats.max_hours : NaN
    max_material_next_assets =
        isfinite(stats.max_material_next_assets) ? stats.max_material_next_assets : NaN
    max_material_hours = isfinite(stats.max_material_hours) ? stats.max_material_hours : NaN
    asset_upper = asset_upper_bound(p)
    hours_upper = hours_upper_bound(p)
    asset_upper_bound_slack = asset_upper - max_material_next_assets
    hours_upper_bound_slack = hours_upper - max_material_hours
    asset_upper_bound_binding = share_at_asset_upper_bound > upper_bound_share_tol()
    hours_upper_bound_binding = share_at_hours_upper_bound > upper_bound_share_tol()
    distributions = (;
        assetGrid = p.a_grid,
        assetMass = copy(stats.asset_mass),
        assetMassTotal = sum(stats.asset_mass),
        hours = copy(stats.hours_values),
        consumption = copy(stats.consumption_values),
        weights = copy(stats.distribution_weights),
        observationWeightTotal = sum(stats.distribution_weights),
    )

    return (;
        totalMass = total_mass,
        meanAssets = mean_assets,
        medianAssets = median_assets,
        meanLaborIncome = mean_labor_income,
        meanBorrowingLimit = mean_borrowing_limit,
        meanEffectiveGridBorrowingLimit = mean_effective_borrowing_limit,
        meanAssetsToMeanLaborIncome = safe_ratio(mean_assets, mean_labor_income),
        medianAssetsToMeanLaborIncome = safe_ratio(median_assets, mean_labor_income),
        meanBorrowingLimitToMeanLaborIncome =
            safe_ratio(mean_borrowing_limit, mean_labor_income),
        meanEffectiveGridBorrowingLimitToMeanLaborIncome =
            safe_ratio(mean_effective_borrowing_limit, mean_labor_income),
        shareNegativeLiquidAssets = stats.negative_asset_mass / total_mass,
        shareAtEffectiveBorrowingConstraint = share_at_effective_borrowing_constraint,
        shareZeroAssets = stats.zero_asset_mass / total_mass,
        shareAtAssetUpperBound = share_at_asset_upper_bound,
        shareAtHoursUpperBound = share_at_hours_upper_bound,
        assetUpperBound = asset_upper,
        hoursUpperBound = hours_upper,
        maxNextAssets = max_next_assets,
        maxHours = max_hours,
        maxMaterialNextAssets = max_material_next_assets,
        maxMaterialHours = max_material_hours,
        assetUpperBoundSlack = asset_upper_bound_slack,
        hoursUpperBoundSlack = hours_upper_bound_slack,
        assetUpperBoundBinding = asset_upper_bound_binding,
        hoursUpperBoundBinding = hours_upper_bound_binding,
        upperBoundsBinding = asset_upper_bound_binding || hours_upper_bound_binding,
        unconditionalDistributions = distributions,
    )
end

function finalize_welfare(value_function_by_kappa::Vector{Float64},
                          simulation_by_kappa::Vector{Float64},
                          p::HDParams)
    difference_by_kappa = simulation_by_kappa .- value_function_by_kappa
    overall_value_function = dot(p.Pkappa, value_function_by_kappa)
    overall_simulation = dot(p.Pkappa, simulation_by_kappa)
    overall_difference = overall_simulation - overall_value_function
    return (;
        kappaGrid = p.kappa_grid,
        kappaProbabilities = p.Pkappa,
        valueFunctionByKappa = value_function_by_kappa,
        simulationByKappa = simulation_by_kappa,
        differenceByKappa = difference_by_kappa,
        overallValueFunction = overall_value_function,
        overallSimulation = overall_simulation,
        overallDifference = overall_difference,
        maxAbsDifferenceByKappa = maximum(abs.(difference_by_kappa)),
    )
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

function first_nonnegative_asset_index(p::HDParams)
    idx = searchsortedfirst(p.a_grid, -1e-12)
    while idx <= length(p.a_grid) && p.a_grid[idx] < -1e-12
        idx += 1
    end
    idx <= length(p.a_grid) || error("a_grid must contain a nonnegative asset point")
    return idx
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
            policyAIndex[ia, is1, is2, iz, ie, age] = Int32(best_iap)
            policyH[ia, is1, is2, iz, ie, age] = p.h_grid[best_ih]
        end
    end
    return nothing
end

function solve_policies_for_kappa(lambda::Float64, kappa::Float64,
                                  first_ap::Vector{Int},
                                  terminal_first_ap::Int,
                                  q_by_ap::Vector{Float64},
                                  tax_base::Matrix{Float64}, p::HDParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nS1 = length(p.s1_grid)
    nS2 = length(p.s2_grid)
    nAge = p.J + 1
    nH = length(p.h_grid)

    Vnext = zeros(nA, nS1, nS2, nZ, nE)     # terminal continuation V_{J+2} = 0
    Vcur = similar(Vnext)
    Vbar = Array{Float64}(undef, nA, nS1, nS2, nZ)   # sum over eps'
    EVz = Array{Float64}(undef, nA, nS1, nS2)        # sum over z' given z

    policyAIndex = Array{Int32}(undef, nA, nS1, nS2, nZ, nE, nAge)
    policyH = Array{Float64}(undef, nA, nS1, nS2, nZ, nE, nAge)

    cash = Matrix{Float64}(undef, nA, nA)            # cash[iap, ia]
    @inbounds for ia in 1:nA, iap in 1:nA
        cash[iap, ia] = p.a_grid[ia] - q_by_ap[iap] * p.a_grid[iap]
    end

    # One scratch set per thread; :static scheduling keeps threadid() stable
    # for the duration of each loop. Sized by maxthreadid(), not nthreads():
    # the interactive threadpool carries ids above the default pool's count.
    scratch = [BlockScratch(nH, nA) for _ in 1:Threads.maxthreadid()]
    blocks = [(ie, is1, is2) for is2 in 1:nS2 for is1 in 1:nS1 for ie in 1:nE]
    nBlocks = length(blocks)

    for age in nAge:-1:1
        has_continuation = age < nAge

        if has_continuation
            fill!(Vbar, 0.0)
            @inbounds for ie in 1:nE
                Vbar .+= p.Peps[ie] .* view(Vnext, :, :, :, :, ie)
            end
        end

        for iz in 1:nZ
            ia_first = age == nAge ? terminal_first_ap : first_ap[iz]

            if has_continuation
                fill!(EVz, 0.0)
                @inbounds for izp in 1:nZ
                    pz = p.Pz[iz, izp]
                    pz == 0.0 && continue
                    EVz .+= pz .* view(Vbar, :, :, :, izp)
                end
            end

            Threads.@threads :static for ib in 1:nBlocks
                ie, is1, is2 = blocks[ib]
                sc = scratch[Threads.threadid()]
                m_base = kappa + p.z_grid[iz] + p.eps_grid[ie]
                coeff = lambda * tax_base[iz, ie] * p.s_factor[is1, is2]
                solve_block!(Vcur, policyAIndex, policyH, sc, EVz, cash,
                             ie, is1, is2, iz, age, ia_first,
                             has_continuation, m_base, coeff, p)
            end
        end

        Vnext, Vcur = Vcur, Vnext
    end

    welfare_value_function = expected_initial_value(Vnext, kappa, p)
    return policyAIndex, policyH, welfare_value_function
end

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

function expected_initial_value(V0, kappa::Float64, p::HDParams)
    al, ar, aw = initial_asset_weights(kappa, p)
    l1, h1, w1 = grid_lookup_weights(p.s1_grid, 0.0)
    l2, h2, w2 = grid_lookup_weights(p.s2_grid, 0.0)
    expected_value = 0.0
    @inbounds for iz in eachindex(p.z_grid), ie in eachindex(p.eps_grid)
        prob = p.z0_probs[iz] * p.Peps[ie]
        prob == 0.0 && continue
        bil(ia) = (1.0 - w1) * ((1.0 - w2) * V0[ia, l1, l2, iz, ie] +
                                w2 * V0[ia, l1, h2, iz, ie]) +
                  w1 * ((1.0 - w2) * V0[ia, h1, l2, iz, ie] +
                        w2 * V0[ia, h1, h2, iz, ie])
        v = aw > 0.0 ? (1.0 - aw) * bil(al) + aw * bil(ar) : bil(al)
        expected_value += prob * v
    end
    return expected_value
end

# -----------------------------------------------------------------------------
# Distribution iteration for one kappa (bilinear Young lottery in s')
# -----------------------------------------------------------------------------
"""
    accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                      true_borrowing_limit, effective_borrowing_limit,
                      at_borrowing_constraint, at_asset_upper, h_upper,
                      binding_age, collect)

Add one (age, state) observation to a statistics accumulator.

Factored out of the forward pass so the several age windows below are fed from
ONE body and cannot drift apart -- the same arrangement `hdinf` and `hi` use.
`collect` is passed rather than read from `p` so an accumulator can skip the
distribution vectors, which are a cross-sectional object.

`binding_age` is the finite-horizon wrinkle: at the terminal age `a' >= 0`
replaces the borrowing limit, so that age contributes no mass to the limit
averages.
"""
@inline function accumulate_stats!(stats::StatsAccumulator, weighted_mass::Float64,
                                   ia::Int, a::Float64, ap::Float64, h::Float64,
                                   c::Float64, y::Float64,
                                   true_borrowing_limit::Float64,
                                   effective_borrowing_limit::Float64,
                                   at_borrowing_constraint::Bool,
                                   at_asset_upper::Bool, h_upper::Float64,
                                   binding_age::Bool, collect::Bool)
    @inbounds begin
        stats.asset_mass[ia] += weighted_mass
        if collect
            push!(stats.distribution_weights, weighted_mass)
            push!(stats.hours_values, h)
            push!(stats.consumption_values, c)
        end
        stats.total_mass += weighted_mass
        stats.sum_current_assets += weighted_mass * a
        stats.sum_labor_income += weighted_mass * y
        stats.sum_borrowing_limit += weighted_mass * true_borrowing_limit
        stats.sum_effective_borrowing_limit += weighted_mass * effective_borrowing_limit
        if binding_age
            stats.borrowing_limit_mass += weighted_mass
        end
        stats.max_next_assets = max(stats.max_next_assets, ap)
        stats.max_hours = max(stats.max_hours, h)
        if weighted_mass > upper_bound_share_tol()
            stats.max_material_next_assets = max(stats.max_material_next_assets, ap)
            stats.max_material_hours = max(stats.max_material_hours, h)
        end
        if a < -1e-10
            stats.negative_asset_mass += weighted_mass
        end
        if abs(a) <= 1e-10
            stats.zero_asset_mass += weighted_mass
        end
        if at_borrowing_constraint
            stats.borrowing_constraint_mass += weighted_mass
        end
        if at_asset_upper
            stats.upper_bound_mass += weighted_mass
        end
        if h >= h_upper - upper_bound_level_tol(h_upper)
            stats.hours_upper_bound_mass += weighted_mass
        end
    end
    return nothing
end

function simulate_kappa!(C, H, Y, A, stats::StatsAccumulator,
                         stats_all::StatsAccumulator,
                         stats_lo::StatsAccumulator,
                         policyAIndex, policyH, kappa::Float64,
                         pkappa::Float64,
                         first_ap::Vector{Int}, terminal_first_ap::Int,
                         q_by_ap::Vector{Float64},
                         tax_base::Matrix{Float64},
                         wage_base::Matrix{Float64},
                         p::HDParams, lambda::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nS1 = length(p.s1_grid)
    nS2 = length(p.s2_grid)
    nAge = p.J + 1
    h_upper = hours_upper_bound(p)

    dist = zeros(nA, nS1, nS2, nZ, nE)
    dist_noeps = zeros(nA, nS1, nS2, nZ)

    ia0l, ia0r, ia0w = initial_asset_weights(kappa, p)
    l1, h1, w1 = grid_lookup_weights(p.s1_grid, 0.0)
    l2, h2, w2 = grid_lookup_weights(p.s2_grid, 0.0)
    @inbounds for iz in 1:nZ, ie in 1:nE
        m0 = p.z0_probs[iz] * p.Peps[ie]
        m0 == 0.0 && continue
        for (ia0, aw) in ((ia0l, 1.0 - ia0w), (ia0r, ia0w))
            aw == 0.0 && continue
            m0a = m0 * aw
            dist[ia0, l1, l2, iz, ie] += m0a * (1.0 - w1) * (1.0 - w2)
            dist[ia0, l1, h2, iz, ie] += m0a * (1.0 - w1) * w2
            dist[ia0, h1, l2, iz, ie] += m0a * w1 * (1.0 - w2)
            dist[ia0, h1, h2, iz, ie] += m0a * w1 * w2
        end
    end

    welfare_simulation = 0.0
    clamped_mass = 0.0
    s1_lo = p.s1_grid[1]; s1_hi = p.s1_grid[end]
    s2_lo = p.s2_grid[1]; s2_hi = p.s2_grid[end]

    @inbounds for age in 1:nAge
        in_stats_window = p.stats_age_lo <= age <= p.stats_age_hi
        at_stats_age_lo = age == p.stats_age_lo
        fill!(dist_noeps, 0.0)
        utility_weight = (1.0 - p.beta) * p.beta^(age - 1)

        for ie in 1:nE, iz in 1:nZ
            # A borrowing limit only exists before the terminal age, where
            # a' >= 0 is imposed instead. Its mass is accumulated separately
            # so the reported means average over ages j = 0,...,J-1 only.
            binding_age = age < nAge
            lower_idx = binding_age ? first_ap[iz] : terminal_first_ap
            true_borrowing_limit = binding_age ?
                -borrowing_limit(kappa, iz, p) : 0.0
            effective_borrowing_limit = binding_age ? -p.a_grid[lower_idx] : 0.0
            m_base = kappa + p.z_grid[iz] + p.eps_grid[ie]

            for is2 in 1:nS2, is1 in 1:nS1
                coeff = lambda * tax_base[iz, ie] * p.s_factor[is1, is2]
                for ia in 1:nA
                    mass = dist[ia, is1, is2, iz, ie]
                    mass <= p.massTol && continue

                    a = p.a_grid[ia]
                    iap = Int(policyAIndex[ia, is1, is2, iz, ie, age])
                    ap = p.a_grid[iap]
                    h = policyH[ia, is1, is2, iz, ie, age]
                    c = coeff * h^p.pow + a - q_by_ap[iap] * ap
                    c > 0.0 || error(
                        "negative consumption on a positive-mass state " *
                        "(age=$age, ia=$ia): widen grids or check feasibility")
                    y = wage_base[iz, ie] * h
                    u = log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta)
                    welfare_simulation += utility_weight * mass * u

                    weighted_mass = pkappa * mass
                    C[age] += weighted_mass * c
                    H[age] += weighted_mass * h
                    Y[age] += weighted_mass * y
                    A[age] += weighted_mass * ap

                    # Three age coverages from one body: `stats` is the
                    # calibration window, `stats_all` every age, and `stats_lo`
                    # the single age at which the window opens. Each keeps its
                    # own total_mass, the denominator every share and mean
                    # divides by, so the gate wraps the whole call.
                    accumulate_stats!(stats_all, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      iap == lower_idx, iap == nA, h_upper,
                                      binding_age, false)
                    in_stats_window &&
                        accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                                          true_borrowing_limit, effective_borrowing_limit,
                                          iap == lower_idx, iap == nA, h_upper,
                                          binding_age, p.collect_distributions)
                    at_stats_age_lo &&
                        accumulate_stats!(stats_lo, weighted_mass, ia, a, ap, h, c, y,
                                          true_borrowing_limit, effective_borrowing_limit,
                                          iap == lower_idx, iap == nA, h_upper,
                                          binding_age, false)

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
                        for izp in 1:nZ
                            pz = p.Pz[iz, izp]
                            pz == 0.0 && continue
                            base = mass * pz
                            dist_noeps[iap, c1l, c2l, izp] += base * v11
                            dist_noeps[iap, c1l, c2h, izp] += base * v12
                            dist_noeps[iap, c1h, c2l, izp] += base * v21
                            dist_noeps[iap, c1h, c2h, izp] += base * v22
                        end
                    end
                end
            end
        end

        if age < nAge
            for iep in 1:nE
                view(dist, :, :, :, :, iep) .= p.Peps[iep] .* dist_noeps
            end
        end
    end

    return welfare_simulation, clamped_mass
end

# -----------------------------------------------------------------------------
# Aggregates at a given lambda and the government-budget residual. Policies
# are solved kappa-by-kappa (each solve is internally threaded over blocks);
# the forward distribution is then threaded over kappa.
# -----------------------------------------------------------------------------
function solve_aggregates_for_lambda(lambda::Float64, p::HDParams)
    nAge = p.J + 1
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
    terminal_first_ap = first_nonnegative_asset_index(p)

    # Phase 1: policies. kappa runs SERIALLY because solve_policies_for_kappa
    # is itself threaded over the nEps*nS1*nS2 blocks, which offers far more
    # parallelism than the nKappa (typically 3) values ever could.
    policyA_by_kappa = Vector{Array{Int32,6}}(undef, nKappa)
    policyH_by_kappa = Vector{Array{Float64,6}}(undef, nKappa)
    first_ap_by_kappa = Vector{Vector{Int}}(undef, nKappa)
    tax_base_by_kappa = Vector{Matrix{Float64}}(undef, nKappa)
    wage_base_by_kappa = Vector{Matrix{Float64}}(undef, nKappa)
    for ik in 1:nKappa
        kappa = p.kappa_grid[ik]
        first_ap_by_kappa[ik] = first_feasible_asset_indices(kappa, p)
        tax_base_by_kappa[ik], wage_base_by_kappa[ik] =
            precompute_income_bases(kappa, p)
        policyA_by_kappa[ik], policyH_by_kappa[ik],
        welfare_value_function_by_kappa[ik] =
            solve_policies_for_kappa(lambda, kappa, first_ap_by_kappa[ik],
                                     terminal_first_ap, q_by_ap,
                                     tax_base_by_kappa[ik], p)
    end

    # Phase 2: the forward distribution, which is independent across kappa.
    Threads.@threads :static for ik in 1:nKappa
        stats_local = StatsAccumulator(length(p.a_grid))
        stats_all_local = StatsAccumulator(length(p.a_grid))
        stats_lo_local = StatsAccumulator(length(p.a_grid))
        welfare_simulation, clamped = simulate_kappa!(
            C_by_kappa[ik], H_by_kappa[ik], Y_by_kappa[ik], A_by_kappa[ik],
            stats_local, stats_all_local, stats_lo_local,
            policyA_by_kappa[ik], policyH_by_kappa[ik],
            p.kappa_grid[ik], p.Pkappa[ik],
            first_ap_by_kappa[ik], terminal_first_ap, q_by_ap,
            tax_base_by_kappa[ik], wage_base_by_kappa[ik], p, lambda,
        )
        stats_by_kappa[ik] = stats_local
        stats_all_by_kappa[ik] = stats_all_local
        stats_lo_by_kappa[ik] = stats_lo_local
        welfare_simulation_by_kappa[ik] = welfare_simulation
        clamped_by_kappa[ik] = clamped
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
    # drift from the windowed one. Both ratios divide by the WINDOW's mean
    # labor income, matching `hdinf` and `hi` field for field.
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
    clamped_share = stats_acc.total_mass > 0.0 ?
                    sum(clamped_by_kappa) / stats_acc.total_mass : 0.0
    return (; C = C, H = H, Y = Y, A = A), stats, stats_all, welfare, clamped_share
end

function government_residual_at_lambda(lambda::Float64, p::HDParams)
    aggs, stats, stats_all, welfare, clamped_share = solve_aggregates_for_lambda(lambda, p)
    nAge = p.J + 1
    lhs = 0.0
    for j in 1:nAge
        lhs += p.qGov^(j - 1) * (aggs.Y[j] - aggs.C[j])
    end
    lhs *= (1.0 - p.qGov)
    rhs = (1.0 - p.qGov^nAge) * p.G
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
        consumptionPV = discounted_sum(aggs.C, p.qGov),
        outputPV = discounted_sum(aggs.Y, p.qGov),
        statistics = stats,
        # Same quantities over every age instead of the calibration
        # window. Reported for comparison; nothing is calibrated on it.
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

# -----------------------------------------------------------------------------
# Printing
# -----------------------------------------------------------------------------
function print_solver_options(p::HDParams)
    println("Options:")
    @printf("  age dimension J               = %d\n", p.J)
    @printf("  shock grid dimension nZ       = %d\n", length(p.z_grid))
    @printf("  shock grid dimension nEps     = %d\n", length(p.eps_grid))
    @printf("  shock grid dimension nKappa   = %d\n", length(p.kappa_grid))
    @printf("  z_discretization_method       = :%s  (alternatives: :rouwenhorst, :tauchen)\n",
            String(p.z_discretization_method))
    @printf("  asset grid dimension nA       = %d\n", length(p.a_grid))
    @printf("  asset grid bounds             = [%.6f, %.6f]\n",
            minimum(p.a_grid), maximum(p.a_grid))
    @printf("  asset grid method             = :%s  (alternatives: :nonuniform, :linear)\n",
            String(p.asset_grid_method))
    # Calibrated inputs print with every digit (shortest representation that
    # round-trips to the same Float64) so they can be copied back verbatim.
    @printf("  bbar                          = %s\n", p.bbar)
    @printf("  labor grid size               = %d on [%.4f, %.4f]\n",
            length(p.h_grid), p.hMin, p.hMax)
    @printf("  labor grid spacing            = :%s  (alternatives: :log, :uniform)\n",
            String(p.labor_grid_spacing))
    @printf("  exploit_hours_monotonicity    = %s\n",
            string(p.exploit_hours_monotonicity))
    @printf("  labor history 1 dimension nS1 = %d\n", length(p.s1_grid))
    @printf("  labor history 1 bounds        = [%.4f, %.4f]\n",
            p.s1_grid[1], p.s1_grid[end])
    @printf("  labor history 1 grid method   = :%s  (alternatives: :linear, :quantile)\n",
            String(p.s_grid_method))
    @printf("  labor history 2 dimension nS2 = %d\n", length(p.s2_grid))
    @printf("  labor history 2 bounds        = [%.4f, %.4f]\n",
            p.s2_grid[1], p.s2_grid[end])
    @printf("  labor history 2 grid method   = :%s  (alternatives: :linear, :quantile)\n",
            String(p.s_grid_method))
    @printf("  qSav                          = %s\n", p.qSav)
    @printf("  qBorr                         = %s\n", p.qBorr)
    @printf("  theta0 (implied)              = %.6f\n", p.theta0)
    @printf("  alpha                         = %.6f\n", p.alpha)
    @printf("  mu1, mu2                      = %.6f, %.6f\n", p.mu1, p.mu2)
    @printf("  hours exponent (1-tau)*theta0 = %.6f\n", p.pow)
    @printf("  s_hours_floor                 = %.4f\n", p.s_hours_floor)
    @printf("  terminal_borrowing            = :zero\n")
    @printf("  lambda_solver                 = :brent  (Roots.jl; fallback: grid search over %d values)\n",
            p.nLambdaSearch)
    @printf("  lambda_bracket                = [%.6f, %.6f], tol = %.2e\n",
            p.lambdaMin, p.lambdaMax, p.tolGovBudget)
    @printf("  collect_distributions         = %s\n", string(p.collect_distributions))
    println()
end

function print_hd_equilibrium_summary(eq, p::HDParams;
                                      title = "Final history-dependent equilibrium")
    @printf("\n=== %s ===\n", title)
    @printf("lambda                     = %.8f\n", eq.lambda)
    @printf("government budget residual = %.8e\n", eq.govBudgetResidual)
    @printf("PV output                  = %.8f\n", eq.outputPV)
    @printf("PV consumption             = %.8f\n", eq.consumptionPV)
    @printf("mean output                = %.8f\n", mean(eq.Y))
    @printf("mean consumption           = %.8f\n", mean(eq.C))
    @printf("terminal assets            = %.8f\n", eq.A[end])
    # Printed with every digit (shortest representation that round-trips to
    # the same Float64), so calibrated values can be copied back verbatim.
    @printf("qSav                       = %s\n", p.qSav)
    @printf("qBorr                      = %s\n", p.qBorr)
    @printf("bbar                       = %s\n", p.bbar)
    @printf("theta0 (implied)           = %.8f\n", p.theta0)
    @printf("s' clamped mass share      = %.3e\n", eq.sClampedMassShare)
    if hasproperty(eq, :elapsedSeconds)
        @printf("solve time                 = %.3f seconds\n", eq.elapsedSeconds)
    end
    print_aggregate_statistics(eq.statistics, p;
        label = @sprintf("model ages %d-%d, real %d-%d  [CALIBRATION WINDOW]",
                         p.stats_age_lo, p.stats_age_hi,
                         p.age0_real + p.stats_age_lo - 1,
                         p.age0_real + p.stats_age_hi - 1))
    # The same statistics over the whole life. Nothing is calibrated on these;
    # they are printed so the effect of restricting the moments to the
    # working-age window is visible rather than implied.
    if hasproperty(eq, :statisticsAllAges)
        print_aggregate_statistics(eq.statisticsAllAges, p;
            label = @sprintf("model ages 1-%d, real %d-%d  [ALL AGES]",
                             p.J + 1, p.age0_real, p.age0_real + p.J))
    end
    print_welfare_summary(eq.welfare)
    print_upper_bound_warning(eq.statistics)
    return nothing
end

function print_aggregate_statistics(s, p::HDParams; label::AbstractString = "")
    @printf("\n=== Aggregate statistics%s ===\n",
            isempty(label) ? "" : ": " * label)
    @printf("mean assets / mean labor income          = %.8f\n",
            s.meanAssetsToMeanLaborIncome)
    @printf("median assets / mean labor income        = %.8f\n",
            s.medianAssetsToMeanLaborIncome)
    @printf("true borrowing limit / mean labor income = %.8f\n",
            s.meanBorrowingLimitToMeanLaborIncome)
    @printf("grid borrowing limit / mean labor income = %.8f\n",
            s.meanEffectiveGridBorrowingLimitToMeanLaborIncome)
    @printf("share negative liquid assets             = %.8f\n",
            s.shareNegativeLiquidAssets)
    @printf("share at effective grid borrowing bound  = %.8f\n",
            s.shareAtEffectiveBorrowingConstraint)
    @printf("share with zero assets                   = %.8f\n", s.shareZeroAssets)
    @printf("share at upper asset bound               = %.8f\n", s.shareAtAssetUpperBound)
    @printf("share at hours upper bound               = %.8f\n", s.shareAtHoursUpperBound)
    # Only the windowed statistics carry the entry-age block; the all-ages
    # block is printed through this same function and has no such age. Same
    # guard, same two lines, same wording as the other three solvers.
    if hasproperty(s, :meanAssetsAtStatsAgeLoToMeanLaborIncome)
        lo_real = p.age0_real + p.stats_age_lo - 1
        @printf("mean assets at age %-2d / mean labor income = %.8f\n",
                lo_real, s.meanAssetsAtStatsAgeLoToMeanLaborIncome)
        @printf("median assets at age %-2d / mean labor inc. = %.8f\n",
                lo_real, s.medianAssetsAtStatsAgeLoToMeanLaborIncome)
    end
    return nothing
end

function print_upper_bound_warning(s)
    s.upperBoundsBinding || return nothing
    @printf("\n=== Upper-bound warning ===\n")
    if s.assetUpperBoundBinding
        @printf("asset upper bound binding: bound = %.8f, material max a' = %.8f, slack = %.8e\n",
                s.assetUpperBound, s.maxMaterialNextAssets, s.assetUpperBoundSlack)
    end
    if s.hoursUpperBoundBinding
        @printf("hours upper bound binding: bound = %.8f, material max h = %.8f, slack = %.8e\n",
                s.hoursUpperBound, s.maxMaterialHours, s.hoursUpperBoundSlack)
    end
    return nothing
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

eq_flag(eq, field::Symbol) =
    hasproperty(eq, field) && getproperty(eq, field) === true

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

end # module HistoryDependentTax
