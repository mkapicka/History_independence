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
using BewleyCommon: BlockScratch, build_s_grid, finalize_welfare,
    initial_asset_weights, s_stock_moments, solve_block!

export HDParams, HD_SETTINGS, make_history_dependent_params,
       solve_history_dependent_tax, print_hd_equilibrium_summary,
       check_history_independent_limit


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
    # ---------------------------------------------------------------------
    # PREFERENCES AND TAX
    # ---------------------------------------------------------------------
    beta::Float64
    eta::Float64
    phi::Float64
    tau::Float64
    theta0::Float64
    alpha::Float64
    mu1::Float64
    mu2::Float64
    pow::Float64                     # (1 - tau) * theta0

    # Asset-market access. Households are in one of two exogenous states:
    # savers (S), who choose a' freely subject to the borrowing limit, and
    # hand-to-mouth (H), who have no access to the asset market and whose
    # assets follow a' = a/qSav for a >= 0 and a' = a for a < 0. `pSS` and
    # `pHH` are `s` and `h` in psmodel.tex. piS/piH are derived.
    #
    # THIS DIRECTORY COMBINES BOTH COMPLICATIONS. Like `hdinf_htm`, hours move
    # the past-income stocks, so the hand-to-mouth hours choice is DYNAMIC and
    # `solve_block_htm!` must maximize rather than look up a static payoff.
    # Like `hi_htm`, the horizon is finite, so at the terminal age `a' >= 0` is
    # imposed and the rollover is clamped: a' = max(htm rule, 0). A terminal
    # hand-to-mouth debtor therefore settles up, consuming y + a rather than
    # y + (1-qBorr)a. psmodel.tex is infinite-horizon and says nothing about
    # that last point; it is a choice made here, matching `hi_htm`.
    pSS::Float64                     # Pr(S' = S | S)
    pHH::Float64                     # Pr(H' = H | H)
    piS::Float64
    piH::Float64

    # ---------------------------------------------------------------------
    # HORIZON, AGES AND SHOCKS
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # ASSET GRID AND ASSET CHOICE
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # PRICES AND GOVERNMENT
    # ---------------------------------------------------------------------
    qBorr::Float64
    qSav::Float64
    qGov::Float64
    G::Float64

    # ---------------------------------------------------------------------
    # LABOR
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # LAMBDA SOLVER
    # ---------------------------------------------------------------------
    lambdaMin::Float64
    lambdaMax::Float64
    nLambdaSearch::Int
    maxIterLambda::Int
    tolLambda::Float64
    tolGovBudget::Float64

    # ---------------------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------------------- and solver behavior
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
    pSS,
    pHH,
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
    0.0 <= pSS <= 1.0 || error("pSS must satisfy 0 <= pSS <= 1")
    0.0 <= pHH <= 1.0 || error("pHH must satisfy 0 <= pHH <= 1")
    piS, piH = access_stationary_distribution(pSS, pHH)
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
        Float64(pSS), Float64(pHH), piS, piH,
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


# -----------------------------------------------------------------------------
# Statistics accumulator (same fields and semantics as the history-independent
# solver, so downstream statistics are directly comparable)
# -----------------------------------------------------------------------------
mutable struct StatsAccumulator <: AbstractStatsAccumulator
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
    htm_mass::Float64
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
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
    dest.htm_mass += src.htm_mass
    dest.upper_bound_mass += src.upper_bound_mass
    dest.hours_upper_bound_mass += src.hours_upper_bound_mass
    dest.max_next_assets = max(dest.max_next_assets, src.max_next_assets)
    dest.max_hours = max(dest.max_hours, src.max_hours)
    dest.max_material_next_assets =
        max(dest.max_material_next_assets, src.max_material_next_assets)
    dest.max_material_hours = max(dest.max_material_hours, src.max_material_hours)
    return dest
end

function finalize_statistics(stats::StatsAccumulator, p::HDParams)
    # The unrestricted maxima, beside the material ones core_statistics
    # publishes. These are taken over every cell; the material pair is taken
    # only over cells carrying more than UPPER_BOUND_SHARE_TOL of mass, so a
    # state sitting at the grid edge with negligible mass shows up here and not
    # there. Only the hd family publishes both.
    max_next_assets = isfinite(stats.max_next_assets) ? stats.max_next_assets : NaN
    max_hours = isfinite(stats.max_hours) ? stats.max_hours : NaN
    return merge(core_statistics(stats, p),
                 (; maxNextAssets = max_next_assets, maxHours = max_hours,
                    shareHandToMouth = stats.htm_mass / stats.total_mass))
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

# -----------------------------------------------------------------------------
# Backward induction for one kappa: joint (a', h) grid choice with bilinear
# interpolation of the continuation value in (s1', s2')
#
# Parallelism lives here, over the nEps*nS1*nS2 blocks of (eps, s1, s2) at
# each (age, z), not over kappa, which is typically 3 and would cap the solver
# at 3 cores. Blocks write disjoint slices and only read the shared EVz, so the
# result is bit-for-bit identical to a serial run.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Hand-to-mouth block
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
`(1-qBorr)*a < 0` for a debtor. `clipped` flags that `a/qSav` ran past the top
grid node for some `a`, in which case `a'` is held at `aMax` and the excess is
consumed -- the rule has no fixed point above zero, so only the grid stops it.

TWO VARIANTS. `terminal = true` additionally clamps `a' >= 0`, because the last
age imposes that on savers and a hand-to-mouth debtor would otherwise die owing
money. A terminal debtor then settles up: `a' = 0` and `cash = a`. This mirrors
`hi_htm`; psmodel.tex is infinite-horizon and does not cover it.
"""
struct HTMTransition
    next_assets::Vector{Float64}
    cash::Vector{Float64}
    left::Vector{Int}
    right::Vector{Int}
    weight::Vector{Float64}
    clipped::Bool
end

function htm_transition(p::HDParams; terminal::Bool = false)
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
        elseif terminal
            ap = 0.0
            cash[ia] = a                      # settle up: c = y + a
        else
            ap = a
            cash[ia] = a - p.qBorr * ap       # (1 - qBorr) * a < 0
        end
        next_assets[ia] = ap
        l, r, w = grid_lookup_weights(p.a_grid, ap)
        left[ia] = l; right[ia] = r; weight[ia] = w
    end

    return HTMTransition(next_assets, cash, left, right, weight, clipped)
end

"""
    solve_block_htm!(...)

The hand-to-mouth counterpart of `solve_block!` for one (eps, s1, s2) block at
one age.

`a'` is exogenous, so the joint (a', h) search collapses to a scan over hours
alone -- but it is STILL a maximization, which is what separates this directory
and `hdinf_htm` from `hi_htm`. There, hand-to-mouth hours solve a static
within-period problem and the payoff is precomputed once. Here hours move the
past-income stocks through s' = mu*(log wage + log h + s), so the household
trades current leisure against future tax liabilities exactly as a saver does,
and the scan must be redone at every age.

The continuation is TRILINEAR: bilinear in (s1', s2') as for the saver, and
linear in a' as well, because a' = a/qSav falls between asset grid nodes. Both
weights come from `htm`, the same object the forward pass uses.

`sc.EVh` is reused as (h, a) here rather than (h, a'); the saver's block has
finished with it by the time this runs.
"""
function solve_block_htm!(VcurH, policyHtmH, sc::BlockScratch, EVzH,
                          htm::HTMTransition, ie::Int, is1::Int, is2::Int,
                          iz::Int, age::Int, has_continuation::Bool,
                          m_base::Float64, coeff::Float64, p::HDParams)
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

        if has_continuation
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
                val = util_weight * (log(c) - dis[ih])
                has_continuation && (val += beta * EVh[ih, ia])
                if val > best_val
                    best_val = val
                    best_ih = ih
                end
            end
            VcurH[ia, is1, is2, iz, ie] = best_val
            policyHtmH[ia, is1, is2, iz, ie, age] = p.h_grid[best_ih]
        end
    end
    return nothing
end

function solve_policies_for_kappa(lambda::Float64, kappa::Float64,
                                  first_ap::Vector{Int},
                                  terminal_first_ap::Int,
                                  q_by_ap::Vector{Float64},
                                  tax_base::Matrix{Float64},
                                  htm::HTMTransition, htm_terminal::HTMTransition,
                                  p::HDParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nS1 = length(p.s1_grid)
    nS2 = length(p.s2_grid)
    nAge = p.J + 1
    nH = length(p.h_grid)

    # One value function per access state. The saver's own block is untouched;
    # it just receives the MIXED continuation in place of the plain one.
    VnextS = zeros(nA, nS1, nS2, nZ, nE)    # terminal continuation V_{J+2} = 0
    VcurS = similar(VnextS)
    VnextH = zeros(nA, nS1, nS2, nZ, nE)
    VcurH = similar(VnextH)
    VbarS = Array{Float64}(undef, nA, nS1, nS2, nZ)  # sum over eps'
    VbarH = Array{Float64}(undef, nA, nS1, nS2, nZ)
    EVzS = Array{Float64}(undef, nA, nS1, nS2)       # sum over z' given z
    EVzH = Array{Float64}(undef, nA, nS1, nS2)
    EVmixS = Array{Float64}(undef, nA, nS1, nS2)
    EVmixH = Array{Float64}(undef, nA, nS1, nS2)

    policyAIndex = Array{Int32}(undef, nA, nS1, nS2, nZ, nE, nAge)
    policyH = Array{Float64}(undef, nA, nS1, nS2, nZ, nE, nAge)
    policyHtmH = Array{Float64}(undef, nA, nS1, nS2, nZ, nE, nAge)

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
            fill!(VbarS, 0.0)
            fill!(VbarH, 0.0)
            @inbounds for ie in 1:nE
                VbarS .+= p.Peps[ie] .* view(VnextS, :, :, :, :, ie)
                VbarH .+= p.Peps[ie] .* view(VnextH, :, :, :, :, ie)
            end
        end

        for iz in 1:nZ
            ia_first = age == nAge ? terminal_first_ap : first_ap[iz]

            if has_continuation
                fill!(EVzS, 0.0)
                fill!(EVzH, 0.0)
                @inbounds for izp in 1:nZ
                    pz = p.Pz[iz, izp]
                    pz == 0.0 && continue
                    EVzS .+= pz .* view(VbarS, :, :, :, izp)
                    EVzH .+= pz .* view(VbarH, :, :, :, izp)
                end
                # The access shock is independent of (z', eps'), so the mixing
                # is done AFTER the expectation and the saver's block sees a
                # drop-in replacement for EVz. A state with no feasible choice
                # carries the FINITE sentinel, so these products are 0.0 when
                # the weight is zero; with -Inf the pSS = 1 corner -- the one
                # that has to reproduce `hd` exactly -- would be NaN.
                @inbounds for i in eachindex(EVzS)
                    evs = EVzS[i]
                    evh = EVzH[i]
                    EVmixS[i] = p.pSS * evs + (1.0 - p.pSS) * evh
                    EVmixH[i] = p.pHH * evh + (1.0 - p.pHH) * evs
                end
            end

            Threads.@threads :static for ib in 1:nBlocks
                ie, is1, is2 = blocks[ib]
                sc = scratch[Threads.threadid()]
                m_base = kappa + p.z_grid[iz] + p.eps_grid[ie]
                coeff = lambda * tax_base[iz, ie] * p.s_factor[is1, is2]
                solve_block!(VcurS, policyAIndex, policyH, sc, EVmixS, cash,
                             ie, is1, is2, iz, age, ia_first,
                             has_continuation, m_base, coeff, p)
                solve_block_htm!(VcurH, policyHtmH, sc, EVmixH,
                                 age == nAge ? htm_terminal : htm,
                                 ie, is1, is2, iz, age, has_continuation,
                                 m_base, coeff, p)
            end
        end

        VnextS, VcurS = VcurS, VnextS
        VnextH, VcurH = VcurH, VnextH
    end

    welfare_value_function = expected_initial_value(VnextS, VnextH, kappa, p)
    return policyAIndex, policyH, policyHtmH, welfare_value_function
end

# Newborns draw their access state from the STATIONARY distribution (piS, piH),
# independently of (a0, s0, z, eps), so the birth value is the piS/piH mix of
# the two value functions at the same placement. `simulate_kappa!` seeds the
# distribution the same way; if the two disagree, the welfare cross-check in
# `finalize_welfare` reports it.
function expected_initial_value(V0S, V0H, kappa::Float64, p::HDParams)
    al, ar, aw = initial_asset_weights(kappa, p)
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
        vS = aw > 0.0 ? (1.0 - aw) * bilS(al) + aw * bilS(ar) : bilS(al)
        vH = aw > 0.0 ? (1.0 - aw) * bilH(al) + aw * bilH(ar) : bilH(al)
        expected_value += prob * (p.piS * vS + p.piH * vH)
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
                      binding_age, is_htm, collect)

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
                                   binding_age::Bool, is_htm::Bool,
                                   collect::Bool)
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
        if weighted_mass > UPPER_BOUND_SHARE_TOL
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
        if is_htm
            stats.htm_mass += weighted_mass
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
                         policyAIndex, policyH, policyHtmH, kappa::Float64,
                         pkappa::Float64,
                         first_ap::Vector{Int}, terminal_first_ap::Int,
                         q_by_ap::Vector{Float64},
                         tax_base::Matrix{Float64},
                         wage_base::Matrix{Float64},
                         htm::HTMTransition, htm_terminal::HTMTransition,
                         p::HDParams, lambda::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nS1 = length(p.s1_grid)
    nS2 = length(p.s2_grid)
    nAge = p.J + 1
    h_upper = hours_upper_bound(p)

    # Sixth axis: 1 = saver (S), 2 = hand-to-mouth (H).
    dist = zeros(nA, nS1, nS2, nZ, nE, 2)
    dist_noeps = zeros(nA, nS1, nS2, nZ, 2)
    a_top = last(p.a_grid)

    ia0l, ia0r, ia0w = initial_asset_weights(kappa, p)
    l1, h1, w1 = grid_lookup_weights(p.s1_grid, 0.0)
    l2, h2, w2 = grid_lookup_weights(p.s2_grid, 0.0)
    @inbounds for iz in 1:nZ, ie in 1:nE
        m0 = p.z0_probs[iz] * p.Peps[ie]
        m0 == 0.0 && continue
        # Newborns draw the access state from the stationary distribution, so
        # the HtM share is constant over the life cycle.
        for (ia0, aw) in ((ia0l, 1.0 - ia0w), (ia0r, ia0w)),
            (iacc, share) in ((1, p.piS), (2, p.piH))
            aw == 0.0 && continue
            m0a = m0 * aw * share
            dist[ia0, l1, l2, iz, ie, iacc] += m0a * (1.0 - w1) * (1.0 - w2)
            dist[ia0, l1, h2, iz, ie, iacc] += m0a * (1.0 - w1) * w2
            dist[ia0, h1, l2, iz, ie, iacc] += m0a * w1 * (1.0 - w2)
            dist[ia0, h1, h2, iz, ie, iacc] += m0a * w1 * w2
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
                for ia in 1:nA, iacc in 1:2
                    mass = dist[ia, is1, is2, iz, ie, iacc]
                    mass <= p.massTol && continue
                    is_htm = iacc == 2

                    a = p.a_grid[ia]
                    if is_htm
                        # Exogenous rule, clamped to a' >= 0 at the terminal
                        # age. The borrowing limit is not a constraint on
                        # someone who does not choose, so at_borrowing_
                        # constraint is false. `cash` is taken from `hm` rather
                        # than recomputed: a - q(a')a' agrees analytically but
                        # carries rounding noise on the positive branch, where
                        # it must be exactly zero.
                        hm = age == nAge ? htm_terminal : htm
                        ap = hm.next_assets[ia]
                        cash = hm.cash[ia]
                        h = policyHtmH[ia, is1, is2, iz, ie, age]
                        at_cons = false
                        at_upper = ap >= a_top - upper_bound_level_tol(a_top)
                        nl, nr, nw = hm.left[ia], hm.right[ia], hm.weight[ia]
                    else
                        iap = Int(policyAIndex[ia, is1, is2, iz, ie, age])
                        ap = p.a_grid[iap]
                        cash = a - q_by_ap[iap] * ap
                        h = policyH[ia, is1, is2, iz, ie, age]
                        at_cons = iap == lower_idx
                        at_upper = iap == nA
                        nl = nr = iap; nw = 0.0
                    end
                    c = coeff * h^p.pow + cash
                    if !(c > 0.0)
                        acc_label = is_htm ? "H" : "S"
                        error("negative consumption on a positive-mass state " *
                              "(age=$age, ia=$ia, access=$acc_label): " *
                              "widen grids or check feasibility")
                    end
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
                                      at_cons, at_upper, h_upper,
                                      binding_age, is_htm, false)
                    in_stats_window &&
                        accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                                          true_borrowing_limit, effective_borrowing_limit,
                                          at_cons, at_upper, h_upper,
                                          binding_age, is_htm, p.collect_distributions)
                    at_stats_age_lo &&
                        accumulate_stats!(stats_lo, weighted_mass, ia, a, ap, h, c, y,
                                          true_borrowing_limit, effective_borrowing_limit,
                                          at_cons, at_upper, h_upper,
                                          binding_age, is_htm, false)

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
                        # The access chain is independent of (z', eps'), of a'
                        # and of the s-stocks, so the (a', s', z') mass is built
                        # exactly as in `hd` and then split across the two
                        # access states. A saver lands on one asset node
                        # (nw = 0, second leg skipped); a hand-to-mouth
                        # household on the two its Young lottery straddles.
                        stay_w = is_htm ? p.pHH : p.pSS
                        switch_w = 1.0 - stay_w
                        iacc_sw = 3 - iacc
                        for izp in 1:nZ
                            pz = p.Pz[iz, izp]
                            pz == 0.0 && continue
                            base = mass * pz
                            for (iapn, aw) in ((nl, 1.0 - nw), (nr, nw))
                                aw == 0.0 && continue
                                basea = base * aw
                                for (iaccp, accw) in ((iacc, stay_w),
                                                      (iacc_sw, switch_w))
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

        if age < nAge
            for iep in 1:nE
                view(dist, :, :, :, :, iep, :) .= p.Peps[iep] .* dist_noeps
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
    # kappa-independent, so built once and shared read-only across the threads.
    htm = htm_transition(p)
    htm_terminal = htm_transition(p; terminal = true)

    # Phase 1: policies. kappa runs SERIALLY because solve_policies_for_kappa
    # is itself threaded over the nEps*nS1*nS2 blocks, which offers far more
    # parallelism than the nKappa (typically 3) values ever could.
    policyA_by_kappa = Vector{Array{Int32,6}}(undef, nKappa)
    policyH_by_kappa = Vector{Array{Float64,6}}(undef, nKappa)
    policyHtmH_by_kappa = Vector{Array{Float64,6}}(undef, nKappa)
    first_ap_by_kappa = Vector{Vector{Int}}(undef, nKappa)
    tax_base_by_kappa = Vector{Matrix{Float64}}(undef, nKappa)
    wage_base_by_kappa = Vector{Matrix{Float64}}(undef, nKappa)
    for ik in 1:nKappa
        kappa = p.kappa_grid[ik]
        first_ap_by_kappa[ik] = first_feasible_asset_indices(kappa, p)
        tax_base_by_kappa[ik], wage_base_by_kappa[ik] =
            precompute_income_bases(kappa, p)
        policyA_by_kappa[ik], policyH_by_kappa[ik], policyHtmH_by_kappa[ik],
        welfare_value_function_by_kappa[ik] =
            solve_policies_for_kappa(lambda, kappa, first_ap_by_kappa[ik],
                                     terminal_first_ap, q_by_ap,
                                     tax_base_by_kappa[ik],
                                     htm, htm_terminal, p)
    end

    # Phase 2: the forward distribution, which is independent across kappa.
    Threads.@threads :static for ik in 1:nKappa
        stats_local = StatsAccumulator(length(p.a_grid))
        stats_all_local = StatsAccumulator(length(p.a_grid))
        stats_lo_local = StatsAccumulator(length(p.a_grid))
        welfare_simulation, clamped = simulate_kappa!(
            C_by_kappa[ik], H_by_kappa[ik], Y_by_kappa[ik], A_by_kappa[ik],
            stats_local, stats_all_local, stats_lo_local,
            policyA_by_kappa[ik], policyH_by_kappa[ik], policyHtmH_by_kappa[ik],
            p.kappa_grid[ik], p.Pkappa[ik],
            first_ap_by_kappa[ik], terminal_first_ap, q_by_ap,
            tax_base_by_kappa[ik], wage_base_by_kappa[ik],
            htm, htm_terminal, p, lambda,
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
    # ---------------------------------------------------------------------
    # LABOR
    # --------------------------------------------------------------------- income, matching `hdinf` and `hi` field for field.
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
    @printf("pSS / pHH                  = %.8f / %.8f   (piH = %.8f)%s\n",
            p.pSS, p.pHH, p.piH,
            p.piH == 0.0 ? "   [NO HtM: this is the hd model]" : "")
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

end # module HistoryDependentTax
