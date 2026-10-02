# =============================================================================
# solve.jl
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
#   solve_hd, print_hd_equilibrium_summary,
#   check_history_independent_limit
#
# Usage:
#   include("solve.jl")   # also loads the HD settings file
#   using .HistoryDependentTax
#   p  = make_history_dependent_params()        # HD_SETTINGS + overrides
#   eq = solve_hd(p)
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
       solve_hd, print_hd_equilibrium_summary,
       check_history_independent_limit


include("params.jl")
include("statistics.jl")
include("report.jl")


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
    solve_hd(p::HDParams)

Solve for the tax level `lambda` clearing the government budget constraint.
Residuals are cached per lambda, a grid fallback over `nLambdaSearch` values
handles brackets without a sign change, and the best evaluated equilibrium is
kept as a fallback. Returns the equilibrium NamedTuple.
"""
function solve_hd(p::HDParams)
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
    eq = solve_hd(p)

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
