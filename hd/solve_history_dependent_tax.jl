# =============================================================================
# solve_history_dependent_tax.jl  --  STANDALONE
#
# Finite-horizon Bewley economy with a HISTORY-DEPENDENT tax system
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
# with theta0 implied by the promise-keeping restriction
#   theta0 * ( alpha/(1 - beta*mu1) + (1-alpha)/(1 - beta*mu2) ) = 1.
#
# mu1 = mu2 = 0 implies theta0 = 1 and s1 = s2 = 0, reproducing the
# history-independent model of Section 1.1 exactly (use nS1 = nS2 = 1).
#
# -----------------------------------------------------------------------------
# This file is SELF-CONTAINED: it does not include or call the
# history-independent code. All shared infrastructure (shock discretization,
# asset/labor grids, statistics, the lambda solver) is replicated here inside
# the module `HistoryDependentTax`, so the two codebases can evolve
# independently and be loaded in the same session without name collisions.
# Only the HD-specific API is exported:
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
# Solution method: hours affect s' and are therefore intertemporal, so (a', h)
# are chosen JOINTLY on grids against a continuation value BILINEARLY
# interpolated in (s1', s2'); the distribution uses the matching bilinear
# Young (1990) lottery; infeasible states carry the finite sentinel
# VINFEASIBLE (not -Inf, which would create 0 * Inf = NaN in the
# interpolation); s' outside the grid is clamped and the clamped mass share is
# reported in eq.sClampedMassShare with a warning when material.
# =============================================================================

module HistoryDependentTax

using LinearAlgebra
using Printf
using Statistics
using FastGaussQuadrature
using QuantEcon
using Roots
using StatsBase

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
model_settings.jl. `theta0` is computed from the restriction
`theta0*(alpha/(1-beta*mu1) + (1-alpha)/(1-beta*mu2)) = 1` and cannot be set
directly. The asset choice is always grid search and hours are always chosen
on the labor grid (the static labor FOC is invalid because hours move s').
"""
struct HDParams
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
    # Model and solver parameters carry NO defaults here: `HD_SETTINGS` in
    # model_settings.jl is the single source of truth, applied through
    # `make_history_dependent_params`. A direct `HDParams()` call missing a
    # keyword fails fast with an UndefKeywordError instead of silently
    # solving a different model.
    beta,
    eta,
    phi,
    tau,
    alpha,
    mu1,
    mu2,
    J,
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
    0.0 <= alpha <= 1.0 || error("alpha must be in [0, 1]")
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
    asset_grid_method in (:nonuniform, :linear) ||
        error("asset_grid_method must be :nonuniform or :linear")
    0.0 <= asset_grid_zero_share < 1.0 ||
        error("asset_grid_zero_share must be in [0, 1)")
    asset_grid_zero_width >= 0.0 || error("asset_grid_zero_width must be nonnegative")

    # Promise-keeping restriction pins down theta0.
    denom = alpha / (1.0 - beta * mu1) + (1.0 - alpha) / (1.0 - beta * mu2)
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

    h_grid = build_labor_grid(hMin, hMax, labor_grid_size, h_grid;
                              spacing = labor_grid_spacing)
    h_grid_disutility = phi .* (h_grid .^ (1.0 + eta)) ./ (1.0 + eta)
    log_h_grid = log.(h_grid)
    h_income_power = h_grid .^ pow

    # Effective hours floor for the s-grid bounds: with hMin >= s_hours_floor
    # every realizable s' lies inside the grid (no clamping from low hours).
    s_floor = max(Float64(hMin), Float64(s_hours_floor))
    s1_grid = build_s_grid(mu1, nS1, kappa_grid, z_grid, eps_grid,
                           s_floor, hMax)
    s2_grid = build_s_grid(mu2, nS2, kappa_grid, z_grid, eps_grid,
                           s_floor, hMax)
    s_factor = Matrix{Float64}(undef, length(s1_grid), length(s2_grid))
    for i1 in eachindex(s1_grid), i2 in eachindex(s2_grid)
        s_factor[i1, i2] =
            exp(pow * (alpha * s1_grid[i1] + (1.0 - alpha) * s2_grid[i2]))
    end

    return HDParams(
        beta, eta, phi, tau, theta0, alpha, mu1, mu2, pow,
        J, z_grid, Pz, z0_probs, eps_grid, Peps, kappa_grid, Pkappa,
        z_discretization_method, tauchen_width, rho,
        bbar, aMax, nA, a_grid,
        asset_grid_method, asset_grid_curvature_borrow,
        asset_grid_curvature_save, asset_grid_borrow_share,
        asset_grid_zero_share, asset_grid_zero_width,
        qBorr, qSav, qGov, G,
        hMin, hMax, h_grid, h_grid_disutility, log_h_grid, h_income_power,
        labor_grid_spacing,
        s1_grid, s2_grid, s_factor, Float64(s_hours_floor),
        lambdaMin, lambdaMax, nLambdaSearch, maxIterLambda,
        tolLambda, tolGovBudget,
        verbose, massTol, collect_distributions, exploit_hours_monotonicity,
    )
end

"""
    build_s_grid(mu, nS, kappa_grid, z_grid, eps_grid, s_hours_floor, hMax)

Linear grid for one past-income stock. With m = kappa + z + eps + ln h and
mu in [0, 1), the recursion s' = mu*(m + s) keeps s in
[mu*m_lo/(1-mu), mu*m_hi/(1-mu)]; ln h is bounded below using s_hours_floor
(log(hMin) would blow the grid up). Bounds are extended to include the
initial value 0. Returns `[0.0]` when mu = 0.
"""
function build_s_grid(mu::Real, nS::Int, kappa_grid, z_grid, eps_grid,
                      s_hours_floor::Real, hMax::Real)
    mu = Float64(mu)
    s_hours_floor = Float64(s_hours_floor)
    hMax = Float64(hMax)
    mu == 0.0 && return [0.0]
    m_lo = minimum(kappa_grid) + minimum(z_grid) + minimum(eps_grid) +
           log(s_hours_floor)
    m_hi = maximum(kappa_grid) + maximum(z_grid) + maximum(eps_grid) + log(hMax)
    s_lo = min(mu * m_lo / (1.0 - mu), 0.0)
    s_hi = max(mu * m_hi / (1.0 - mu), 0.0)
    s_hi > s_lo || error("degenerate s-grid bounds [$s_lo, $s_hi]")
    return collect(range(s_lo, s_hi, length = max(nS, 2)))
end

"""
    grid_lookup_weights(grid, x)

Clamped linear-interpolation lookup: `(lo, hi, w)` with
`x ~ (1-w)*grid[lo] + w*grid[hi]`; out-of-bounds clamps to the nearest
endpoint with `lo == hi`; a single-point grid returns `(1, 1, 0.0)`.
"""
function grid_lookup_weights(grid::Vector{Float64}, x::Float64)
    n = length(grid)
    if n == 1 || x <= grid[1]
        return 1, 1, 0.0
    elseif x >= grid[n]
        return n, n, 0.0
    end
    hi = searchsortedfirst(grid, x)
    lo = hi - 1
    w = (x - grid[lo]) / (grid[hi] - grid[lo])
    return lo, hi, w
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
upper_bound_level_tol(bound::Real) = 1e-8 * max(1.0, abs(Float64(bound)))
asset_upper_bound(p::HDParams) = maximum(p.a_grid)
hours_upper_bound(p::HDParams) = p.hMax
safe_ratio(num::Real, den::Real) =
    abs(den) > eps(Float64) ? Float64(num) / Float64(den) : NaN

function finalize_statistics(stats::StatsAccumulator, p::HDParams)
    total_mass = stats.total_mass
    mean_assets = stats.sum_current_assets / total_mass
    mean_labor_income = stats.sum_labor_income / total_mass
    median_assets = StatsBase.median(p.a_grid, StatsBase.weights(stats.asset_mass))
    # Both limits exist only at ages j = 0,...,J-1, so they are averaged over
    # the mass of those ages rather than over the whole population (matching
    # the history-independent solver).
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

asset_price(ap::Real, p::HDParams) = ap < 0.0 ? p.qBorr : p.qSav
asset_prices(p::HDParams) = [asset_price(ap, p) for ap in p.a_grid]

function first_feasible_asset_indices(kappa::Float64, p::HDParams)
    nZ = length(p.z_grid)
    idx = Vector{Int}(undef, nZ)
    for iz in 1:nZ
        lower = p.bbar * exp(kappa + p.rho * p.z_grid[iz])
        idx[iz] = searchsortedfirst(p.a_grid, lower - 1e-12)
        if idx[iz] > length(p.a_grid)
            error("No feasible next-period asset for kappa=$kappa, z=$(p.z_grid[iz])")
        end
    end
    return idx
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
# -----------------------------------------------------------------------------
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
    dis = p.h_grid_disutility
    hpow = p.h_income_power
    lnh = p.log_h_grid
    beta = p.beta
    util_weight = 1.0 - beta

    Vnext = zeros(nA, nS1, nS2, nZ, nE)     # terminal continuation V_{J+2} = 0
    Vcur = similar(Vnext)
    Vbar = Array{Float64}(undef, nA, nS1, nS2, nZ)   # sum over eps'
    EVz = Array{Float64}(undef, nA, nS1, nS2)        # sum over z' given z
    EVh = Matrix{Float64}(undef, nH, nA)             # EV at (h, a')

    policyAIndex = Array{Int32}(undef, nA, nS1, nS2, nZ, nE, nAge)
    policyH = Array{Float64}(undef, nA, nS1, nS2, nZ, nE, nAge)

    cash = Matrix{Float64}(undef, nA, nA)            # cash[iap, ia]
    @inbounds for ia in 1:nA, iap in 1:nA
        cash[iap, ia] = p.a_grid[ia] - q_by_ap[iap] * p.a_grid[iap]
    end

    inc = Vector{Float64}(undef, nH)
    l1v = Vector{Int}(undef, nH); h1v = Vector{Int}(undef, nH)
    w1v = Vector{Float64}(undef, nH)
    l2v = Vector{Int}(undef, nH); h2v = Vector{Int}(undef, nH)
    w2v = Vector{Float64}(undef, nH)
    ih_ub = Vector{Int}(undef, nA)   # monotone upper bound on optimal ih, per a'

    @inbounds for age in nAge:-1:1
        has_continuation = age < nAge
        has_continuation || fill!(EVh, 0.0)   # terminal: continuation is zero

        if has_continuation
            fill!(Vbar, 0.0)
            for ie in 1:nE
                Vbar .+= p.Peps[ie] .* view(Vnext, :, :, :, :, ie)
            end
        end

        for iz in 1:nZ
            ia_first = age == nAge ? terminal_first_ap : first_ap[iz]

            if has_continuation
                fill!(EVz, 0.0)
                for izp in 1:nZ
                    pz = p.Pz[iz, izp]
                    pz == 0.0 && continue
                    EVz .+= pz .* view(Vbar, :, :, :, izp)
                end
            end

            for ie in 1:nE
                m_base = kappa + p.z_grid[iz] + p.eps_grid[ie]

                for is2 in 1:nS2, is1 in 1:nS1
                    coeff = lambda * tax_base[iz, ie] * p.s_factor[is1, is2]
                    for ih in 1:nH
                        inc[ih] = coeff * hpow[ih]
                    end
                    inc_max = inc[nH]

                    if has_continuation
                        for ih in 1:nH
                            s1n = p.mu1 * (m_base + lnh[ih] + p.s1_grid[is1])
                            s2n = p.mu2 * (m_base + lnh[ih] + p.s2_grid[is2])
                            l1v[ih], h1v[ih], w1v[ih] =
                                grid_lookup_weights(p.s1_grid, s1n)
                            l2v[ih], h2v[ih], w2v[ih] =
                                grid_lookup_weights(p.s2_grid, s2n)
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

                    # Optimal-hours monotonicity (Topkis): for fixed a' (so a
                    # fixed EVh column), the objective has decreasing
                    # differences in (h, cash) because
                    # d^2 ln(cash + inc(h)) / d inc d cash < 0 and inc is
                    # increasing in h, while the continuation term does not
                    # depend on cash. Since cash[iap, ia] is strictly
                    # increasing in ia, the (largest) maximizing hours index
                    # is nonincreasing in ia for each iap. Scanning ia in
                    # ascending order, the previous optimum at the same iap is
                    # therefore a valid upper bound for the hours scan.
                    exploit = p.exploit_hours_monotonicity
                    exploit && fill!(ih_ub, nH)

                    for ia in 1:nA
                        best_val = VINFEASIBLE
                        best_iap = ia_first
                        best_ih = nH

                        for iap in ia_first:nA
                            cash_v = cash[iap, ia]
                            # cash is decreasing in a' (q > 0 on both sides of
                            # zero and continuous there), so once maximal hours
                            # cannot deliver c > 0, no larger a' can.
                            if cash_v + inc_max <= 0.0
                                break
                            end
                            ih0 = cash_v > 0.0 ? 1 :
                                  searchsortedfirst(inc, -cash_v)
                            ub = exploit ? max(ih_ub[iap], ih0) : nH

                            local_best = VINFEASIBLE
                            local_ih = ub
                            for ih in ih0:ub
                                c = cash_v + inc[ih]
                                c <= 0.0 && continue
                                val = util_weight * (log(c) - dis[ih]) +
                                      beta * EVh[ih, iap]
                                # ">=" selects the LARGEST maximizer, as the
                                # monotone bound requires.
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
            end
        end

        Vnext, Vcur = Vcur, Vnext
    end

    welfare_value_function = expected_initial_value(Vnext, p)
    return policyAIndex, policyH, welfare_value_function
end

function expected_initial_value(V0, p::HDParams)
    ia0 = nearest_index(p.a_grid, 0.0)
    l1, h1, w1 = grid_lookup_weights(p.s1_grid, 0.0)
    l2, h2, w2 = grid_lookup_weights(p.s2_grid, 0.0)
    expected_value = 0.0
    @inbounds for iz in eachindex(p.z_grid), ie in eachindex(p.eps_grid)
        prob = p.z0_probs[iz] * p.Peps[ie]
        prob == 0.0 && continue
        v = (1.0 - w1) * ((1.0 - w2) * V0[ia0, l1, l2, iz, ie] +
                          w2 * V0[ia0, l1, h2, iz, ie]) +
            w1 * ((1.0 - w2) * V0[ia0, h1, l2, iz, ie] +
                  w2 * V0[ia0, h1, h2, iz, ie])
        expected_value += prob * v
    end
    return expected_value
end

# -----------------------------------------------------------------------------
# Distribution iteration for one kappa (bilinear Young lottery in s')
# -----------------------------------------------------------------------------
function simulate_kappa!(C, H, Y, A, stats::StatsAccumulator,
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

    ia0 = nearest_index(p.a_grid, 0.0)
    l1, h1, w1 = grid_lookup_weights(p.s1_grid, 0.0)
    l2, h2, w2 = grid_lookup_weights(p.s2_grid, 0.0)
    @inbounds for iz in 1:nZ, ie in 1:nE
        m0 = p.z0_probs[iz] * p.Peps[ie]
        m0 == 0.0 && continue
        dist[ia0, l1, l2, iz, ie] += m0 * (1.0 - w1) * (1.0 - w2)
        dist[ia0, l1, h2, iz, ie] += m0 * (1.0 - w1) * w2
        dist[ia0, h1, l2, iz, ie] += m0 * w1 * (1.0 - w2)
        dist[ia0, h1, h2, iz, ie] += m0 * w1 * w2
    end

    welfare_simulation = 0.0
    clamped_mass = 0.0
    s1_lo = p.s1_grid[1]; s1_hi = p.s1_grid[end]
    s2_lo = p.s2_grid[1]; s2_hi = p.s2_grid[end]

    @inbounds for age in 1:nAge
        fill!(dist_noeps, 0.0)
        utility_weight = (1.0 - p.beta) * p.beta^(age - 1)

        for ie in 1:nE, iz in 1:nZ
            # A borrowing limit only exists before the terminal age, where
            # a' >= 0 is imposed instead. Its mass is accumulated separately
            # so the reported means average over ages j = 0,...,J-1 only.
            binding_age = age < nAge
            lower_idx = binding_age ? first_ap[iz] : terminal_first_ap
            true_borrowing_limit = binding_age ?
                -p.bbar * exp(kappa + p.rho * p.z_grid[iz]) : 0.0
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

                    stats.asset_mass[ia] += weighted_mass
                    if p.collect_distributions
                        push!(stats.distribution_weights, weighted_mass)
                        push!(stats.hours_values, h)
                        push!(stats.consumption_values, c)
                    end
                    stats.total_mass += weighted_mass
                    stats.sum_current_assets += weighted_mass * a
                    stats.sum_labor_income += weighted_mass * y
                    stats.sum_borrowing_limit += weighted_mass * true_borrowing_limit
                    stats.sum_effective_borrowing_limit +=
                        weighted_mass * effective_borrowing_limit
                    if binding_age
                        stats.borrowing_limit_mass += weighted_mass
                    end
                    stats.max_next_assets = max(stats.max_next_assets, ap)
                    stats.max_hours = max(stats.max_hours, h)
                    if weighted_mass > upper_bound_share_tol()
                        stats.max_material_next_assets =
                            max(stats.max_material_next_assets, ap)
                        stats.max_material_hours = max(stats.max_material_hours, h)
                    end
                    if a < -1e-10
                        stats.negative_asset_mass += weighted_mass
                    end
                    if abs(a) <= 1e-10
                        stats.zero_asset_mass += weighted_mass
                    end
                    if iap == lower_idx
                        stats.borrowing_constraint_mass += weighted_mass
                    end
                    if iap == nA
                        stats.upper_bound_mass += weighted_mass
                    end
                    if h >= h_upper - upper_bound_level_tol(h_upper)
                        stats.hours_upper_bound_mass += weighted_mass
                    end

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
# Aggregates at a given lambda (threaded over kappa) and government residual
# -----------------------------------------------------------------------------
function solve_aggregates_for_lambda(lambda::Float64, p::HDParams)
    nAge = p.J + 1
    nKappa = length(p.kappa_grid)
    C = zeros(nAge); H = zeros(nAge); Y = zeros(nAge); A = zeros(nAge)
    stats_acc = StatsAccumulator(length(p.a_grid))

    C_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    H_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    Y_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    A_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    stats_by_kappa = Vector{StatsAccumulator}(undef, nKappa)
    welfare_value_function_by_kappa = Vector{Float64}(undef, nKappa)
    welfare_simulation_by_kappa = similar(welfare_value_function_by_kappa)
    clamped_by_kappa = zeros(nKappa)

    q_by_ap = asset_prices(p)
    terminal_first_ap = first_nonnegative_asset_index(p)

    Threads.@threads :static for ik in 1:nKappa
        kappa = p.kappa_grid[ik]
        pkappa = p.Pkappa[ik]
        first_ap = first_feasible_asset_indices(kappa, p)
        tax_base, wage_base = precompute_income_bases(kappa, p)
        policyAIndex, policyH, welfare_value_function =
            solve_policies_for_kappa(lambda, kappa, first_ap,
                                     terminal_first_ap, q_by_ap, tax_base, p)
        stats_local = StatsAccumulator(length(p.a_grid))
        welfare_simulation, clamped = simulate_kappa!(
            C_by_kappa[ik], H_by_kappa[ik], Y_by_kappa[ik], A_by_kappa[ik],
            stats_local, policyAIndex, policyH, kappa, pkappa,
            first_ap, terminal_first_ap, q_by_ap, tax_base, wage_base,
            p, lambda,
        )
        stats_by_kappa[ik] = stats_local
        welfare_value_function_by_kappa[ik] = welfare_value_function
        welfare_simulation_by_kappa[ik] = welfare_simulation
        clamped_by_kappa[ik] = clamped
    end

    for ik in 1:nKappa
        C .+= C_by_kappa[ik]
        H .+= H_by_kappa[ik]
        Y .+= Y_by_kappa[ik]
        A .+= A_by_kappa[ik]
        merge_stats!(stats_acc, stats_by_kappa[ik])
    end

    stats = finalize_statistics(stats_acc, p)
    welfare = finalize_welfare(
        welfare_value_function_by_kappa, welfare_simulation_by_kappa, p,
    )
    clamped_share = stats_acc.total_mass > 0.0 ?
                    sum(clamped_by_kappa) / stats_acc.total_mass : 0.0
    return (; C = C, H = H, Y = Y, A = A), stats, welfare, clamped_share
end

function government_residual_at_lambda(lambda::Float64, p::HDParams)
    aggs, stats, welfare, clamped_share = solve_aggregates_for_lambda(lambda, p)
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
    @printf("  bbar                          = %.6f\n", p.bbar)
    @printf("  labor grid size               = %d on [%.4f, %.4f]\n",
            length(p.h_grid), p.hMin, p.hMax)
    @printf("  labor grid spacing            = :%s  (alternatives: :log, :uniform)\n",
            String(p.labor_grid_spacing))
    @printf("  exploit_hours_monotonicity    = %s\n",
            string(p.exploit_hours_monotonicity))
    @printf("  labor history 1 dimension nS1 = %d\n", length(p.s1_grid))
    @printf("  labor history 1 bounds        = [%.4f, %.4f]\n",
            p.s1_grid[1], p.s1_grid[end])
    @printf("  labor history 1 grid method   = :linear\n")
    @printf("  labor history 2 dimension nS2 = %d\n", length(p.s2_grid))
    @printf("  labor history 2 bounds        = [%.4f, %.4f]\n",
            p.s2_grid[1], p.s2_grid[end])
    @printf("  labor history 2 grid method   = :linear\n")
    @printf("  qSav                          = %.6f\n", p.qSav)
    @printf("  qBorr                         = %.6f\n", p.qBorr)
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
    @printf("theta0 (implied)           = %.8f\n", p.theta0)
    @printf("s' clamped mass share      = %.3e\n", eq.sClampedMassShare)
    if hasproperty(eq, :elapsedSeconds)
        @printf("solve time                 = %.3f seconds\n", eq.elapsedSeconds)
    end
    print_aggregate_statistics(eq.statistics)
    print_welfare_summary(eq.welfare)
    print_upper_bound_warning(eq.statistics)
    return nothing
end

function print_aggregate_statistics(s)
    @printf("\n=== Aggregate statistics ===\n")
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
    return nothing
end

function print_welfare_summary(w)
    @printf("\n=== Welfare ===\n")
    @printf("overall value function utility = %.10f\n", w.overallValueFunction)
    @printf("overall simulation utility     = %.10f\n", w.overallSimulation)
    @printf("overall difference             = %.8e\n", w.overallDifference)
    @printf("kappa      prob        value function  simulation     difference\n")
    for ik in eachindex(w.kappaGrid)
        @printf("% .6f  %.8f  % .10f  % .10f  % .8e\n",
                w.kappaGrid[ik], w.kappaProbabilities[ik],
                w.valueFunctionByKappa[ik], w.simulationByKappa[ik],
                w.differenceByKappa[ik])
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

function print_lambda_warnings(eq)
    has_warning = (hasproperty(eq, :converged) && eq.converged === false) ||
                  eq_flag(eq, :bracketWarning) ||
                  eq_flag(eq, :rootResidualWarning) ||
                  eq_flag(eq, :rootSolverWarning)
    has_warning || return nothing

    println("WARNING: lambda solver returned an approximate solution.")
    if eq_flag(eq, :bracketWarning)
        @printf("  no sign change was found; using best grid-search lambda %.8f with residual %.8e\n",
                eq.lambda, eq.govBudgetResidual)
    end
    if eq_flag(eq, :rootResidualWarning)
        @printf("  root residual %.8e exceeds tolerance %.8e\n",
                abs(eq.govBudgetResidual), eq.parameters.tolGovBudget)
    end
    if eq_flag(eq, :rootSolverWarning)
        @printf("  Brent solver failed; using best evaluated lambda %.8f with residual %.8e\n",
                eq.lambda, eq.govBudgetResidual)
    end
    flush(stdout)
end

# -----------------------------------------------------------------------------
# Shock discretization (replicated infrastructure)
# -----------------------------------------------------------------------------
function build_markov_shock(name::String, n::Int, rho::Float64,
                            innovation_mean::Float64, innovation_sd::Float64,
                            initial_state::Float64, tauchen_width::Float64,
                            method::Symbol)
    grid, P = quantecon_ar1(n, rho, innovation_mean, innovation_sd;
                            method = method, width = tauchen_width)
    initial_probs = ar1_conditional_probabilities(initial_state, grid, rho,
                                                  innovation_mean, innovation_sd)
    initial_probs = normalize_probabilities(initial_probs, "$(name)0_probs")
    validate_transition(P, length(grid))
    length(initial_probs) == length(grid) ||
        error("$(name)0_probs length must match $(name)_grid")
    return grid, P, initial_probs
end

function build_iid_normal_shock(name::String, n::Int, mean::Float64, sd::Float64)
    grid, probs = normal_gauss_hermite(n, mean, sd)
    probs = normalize_probabilities(probs, "P$(name)")
    length(probs) == length(grid) || error("P$(name) length must match $(name)_grid")
    return grid, probs
end

function quantecon_ar1(n::Int, rho::Float64, innovation_mean::Float64,
                       innovation_sd::Float64; method::Symbol = :rouwenhorst,
                       width::Float64 = 3.0)
    n >= 1 || error("n must be positive")
    innovation_sd >= 0.0 || error("innovation_sd must be nonnegative")
    method in (:rouwenhorst, :tauchen) ||
        error("method must be :rouwenhorst or :tauchen")
    abs(rho) < 1.0 || error("rho must satisfy |rho| < 1")

    unconditional_mean = innovation_mean / (1.0 - rho)
    if n == 1 || innovation_sd == 0.0
        return [unconditional_mean], ones(1, 1)
    end

    mc = method == :rouwenhorst ?
         QuantEcon.rouwenhorst(n, rho, innovation_sd, innovation_mean) :
         QuantEcon.tauchen(n, rho, innovation_sd, innovation_mean, width)
    grid = collect(Float64.(mc.state_values))
    P = Matrix{Float64}(mc.p)
    return grid, P
end

function ar1_conditional_probabilities(current_z::Real, grid::AbstractVector{<:Real},
                                       rho::Float64, innovation_mean::Float64,
                                       innovation_sd::Float64)
    grid = collect(Float64.(grid))
    n = length(grid)
    n >= 1 || error("grid must be nonempty")
    if n == 1 || innovation_sd == 0.0
        p = zeros(n)
        p[nearest_index(grid, innovation_mean + rho * current_z)] = 1.0
        return p
    end

    mean_next = innovation_mean + rho * current_z
    cutoffs = [(grid[i] + grid[i + 1]) / 2.0 for i in 1:(n - 1)]
    probs = Vector{Float64}(undef, n)
    probs[1] = normal_cdf((cutoffs[1] - mean_next) / innovation_sd)
    for j in 2:(n - 1)
        upper = (cutoffs[j] - mean_next) / innovation_sd
        lower = (cutoffs[j - 1] - mean_next) / innovation_sd
        probs[j] = normal_cdf(upper) - normal_cdf(lower)
    end
    probs[n] = 1.0 - normal_cdf((cutoffs[end] - mean_next) / innovation_sd)
    return normalize_probabilities(probs, "AR(1) conditional probabilities")
end

function normal_gauss_hermite(n::Int, mean::Float64, sd::Float64)
    n >= 1 || error("n must be positive")
    sd >= 0.0 || error("sd must be nonnegative")
    if n == 1 || sd == 0.0
        return [mean], [1.0]
    end
    nodes, weights = gausshermite(n; normalize = true)
    probs = normalize_probabilities(collect(weights), "Gauss-Hermite weights")
    grid = mean .+ sd .* nodes
    return collect(grid), probs
end

function normal_cdf(x::Real)
    z = Float64(x)
    if z < -8.0
        return 0.0
    elseif z > 8.0
        return 1.0
    end
    return QuantEcon.std_norm_cdf(z)
end

function validate_transition(P::Matrix{Float64}, n::Int)
    size(P) == (n, n) || error("transition matrix must be $n x $n")
    for i in 1:n
        s = sum(P[i, :])
        abs(s - 1.0) <= 1e-8 || error("transition row $i sums to $s, not 1")
        all(P[i, :] .>= -1e-14) || error("transition row $i has negative probabilities")
    end
end

function normalize_probabilities(p::Vector{Float64}, name::String)
    all(p .>= -1e-14) || error("$name has negative probabilities")
    s = sum(p)
    s > 0.0 || error("$name sums to zero")
    p ./= s
    return p
end

# -----------------------------------------------------------------------------
# Asset and labor grids (replicated infrastructure)
# -----------------------------------------------------------------------------
function asset_grid_with_zero(amin::Float64, amax::Float64, nA::Int;
                              method::Symbol = :nonuniform,
                              curvature_borrow::Float64 = 1.8,
                              curvature_save::Float64 = 2.5,
                              borrow_share::Float64 = 0.35,
                              zero_share::Float64 = 0.0,
                              zero_width::Float64 = 0.0)
    amin <= 0.0 <= amax || error("asset grid bounds must contain 0")
    nA >= 3 || error("nA must be at least 3")
    method in (:nonuniform, :linear) || error("unknown asset grid method")

    if method == :linear
        return linear_asset_grid(amin, amax, nA)
    end

    curvature_borrow > 0.0 || error("curvature_borrow must be positive")
    curvature_save > 0.0 || error("curvature_save must be positive")
    0.0 < borrow_share < 1.0 || error("borrow_share must be in (0, 1)")
    0.0 <= zero_share < 1.0 || error("zero_share must be in [0, 1)")
    zero_width >= 0.0 || error("zero_width must be nonnegative")

    if abs(amin) <= 1e-14
        return nonnegative_asset_grid(amax, nA; curvature_save = curvature_save)
    end

    if zero_share <= 0.0 || zero_width <= 0.0
        return two_region_asset_grid(
            amin, amax, nA;
            borrow_share = borrow_share,
            curvature_borrow = curvature_borrow,
            curvature_save = curvature_save,
        )
    end

    return zero_band_asset_grid(
        amin, amax, nA;
        borrow_share = borrow_share,
        curvature_borrow = curvature_borrow,
        curvature_save = curvature_save,
        zero_share = zero_share,
        zero_width = zero_width,
    )
end

function linear_asset_grid(amin::Float64, amax::Float64, nA::Int)
    grid = collect(range(amin, amax, length = nA))
    grid[nearest_index(grid, 0.0)] = 0.0
    sort!(grid)
    return grid
end

function nonnegative_asset_grid(amax::Float64, nA::Int; curvature_save::Float64)
    x = collect(range(0.0, 1.0, length = nA))
    grid = amax .* (x .^ curvature_save)
    grid[1] = 0.0
    grid[end] = amax
    return grid
end

function two_region_asset_grid(amin::Float64, amax::Float64, nA::Int;
                               borrow_share::Float64,
                               curvature_borrow::Float64,
                               curvature_save::Float64)
    n_borrow = clamp(round(Int, borrow_share * (nA - 1)), 1, nA - 2)
    n_save = nA - n_borrow

    xb = collect(range(0.0, 1.0, length = n_borrow + 1))
    borrow_grid = amin .+ (0.0 - amin) .* (1.0 .- (1.0 .- xb) .^ curvature_borrow)

    xs = collect(range(0.0, 1.0, length = n_save))
    save_grid = amax .* (xs .^ curvature_save)

    grid = vcat(borrow_grid[1:end-1], save_grid)
    grid[nearest_index(grid, 0.0)] = 0.0
    grid[1] = amin
    grid[end] = amax
    sort!(grid)
    return grid
end

function zero_band_asset_grid(amin::Float64, amax::Float64, nA::Int;
                              borrow_share::Float64,
                              curvature_borrow::Float64,
                              curvature_save::Float64,
                              zero_share::Float64,
                              zero_width::Float64)
    zero_low = max(amin, -zero_width)
    zero_high = min(amax, zero_width)
    zero_low < 0.0 < zero_high || error("zero_width must create a band around zero")

    n_zero = clamp(round(Int, zero_share * nA), 3, nA - 2)
    n_outer = nA - n_zero
    n_borrow = clamp(round(Int, borrow_share * n_outer), 1, n_outer - 1)
    n_save = n_outer - n_borrow

    xb = collect(range(0.0, 1.0, length = n_borrow + 1))
    borrow_grid = amin .+ (zero_low - amin) .* (1.0 .- (1.0 .- xb) .^ curvature_borrow)

    xz = collect(range(0.0, 1.0, length = n_zero))
    zero_grid = zero_low .+ (zero_high - zero_low) .* xz

    xs = collect(range(0.0, 1.0, length = n_save + 1))
    save_grid = zero_high .+ (amax - zero_high) .* (xs .^ curvature_save)

    grid = vcat(borrow_grid[1:end-1], zero_grid, save_grid[2:end])
    grid[nearest_index(grid, 0.0)] = 0.0
    grid[1] = amin
    grid[end] = amax
    sort!(grid)
    return grid
end

function build_labor_grid(hMin::Float64, hMax::Float64, labor_grid_size::Int,
                          h_grid; spacing::Symbol = :log)
    if isempty(h_grid)
        if spacing == :log
            # Geometric spacing: uniform in ln h. This matches the model's
            # structure -- s' depends on ln h, and income (h^pow) and
            # disutility (h^(1+eta)) are power functions, so RELATIVE hours
            # resolution is what matters. A uniform grid wastes points at
            # high h and is catastrophically coarse in ln h near hMin.
            return exp.(collect(range(log(hMin), log(hMax),
                                      length = labor_grid_size)))
        end
        return collect(range(hMin, hMax, length = labor_grid_size))
    end
    grid = sort(unique(collect(Float64.(h_grid))))
    all(h -> hMin - 1e-12 <= h <= hMax + 1e-12, grid) ||
        error("h_grid entries must lie inside [hMin, hMax]")
    grid = sort(unique(vcat(hMin, grid, hMax)))
    return grid
end

function nearest_index(x::AbstractVector{<:Real}, value::Real)
    return argmin(abs.(x .- value))
end

function find_bracket(grid, residuals)
    for i in 1:(length(grid) - 1)
        if isfinite(residuals[i]) && isfinite(residuals[i + 1])
            if residuals[i] == 0.0
                return (i, i)
            elseif sign(residuals[i]) != sign(residuals[i + 1])
                return (i, i + 1)
            end
        end
    end
    return nothing
end

function discounted_sum(x::AbstractVector{<:Real}, q::Real)
    total = 0.0
    for (j, val) in enumerate(x)
        total += q^(j - 1) * val
    end
    return total
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
