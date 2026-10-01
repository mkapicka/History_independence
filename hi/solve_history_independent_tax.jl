# =============================================================================
# solve_history_independent_tax.jl
#
# Solver for the finite-horizon Bewley model with a history-independent HSV
# tax. Backward induction gives the policies, a forward pass gives the
# cross-section and its statistics, and a Brent solve sets the tax level lambda
# so that the government budget clears.
#
# Marek Kapicka, 2026
# =============================================================================

using LinearAlgebra
using Printf
using Statistics
using FastGaussQuadrature
using QuantEcon
using Roots
using StatsBase

# -----------------------------------------------------------------------------
# Shared infrastructure: the BewleyCommon package at ../common. The second line
# imports the unexported, family-specific functions this solver takes from it.
# -----------------------------------------------------------------------------
using BewleyCommon
using BewleyCommon: asset_choice_lower_bound, finalize_welfare,
    precompute_income_bases, print_equilibrium_summary,
    solve_policy_age_interpolated!

include("params.jl")
include("statistics.jl")
include("report.jl")

"""
    solve_history_independent_tax(p::HIParams)

Solve for `lambda`, household policies, age distributions, and aggregates.
Returns `eq`, a named tuple of equilibrium objects. Household policies are
attached as `eq.solutions` when `p.store_solutions = true` (otherwise
`eq.solutions === nothing`). Build `p` with `make_history_independent_params()`.
"""
function solve_history_independent_tax(p::HIParams)
    start_time = time()

    if p.verbose
        println("\n=== History-independent tax finite-horizon solver ===")
        print_solver_options(p)
        flush(stdout)
    end

    # Brent only needs scalar residuals, so cache every residual but retain just
    # one full equilibrium -- the smallest-residual one seen. Keeping every `eq`
    # would hold the distribution arrays of each trial lambda in memory.
    eval_cache = Dict{Float64,Float64}()
    best = Ref{Any}(nothing)   # (; lambda, residual, eq)

    function solve_at_lambda(lambda::Float64)
        residual, eq = government_residual_at_lambda(lambda, p)
        eval_cache[lambda] = residual
        incumbent = best[]
        if isfinite(residual) &&
           (incumbent === nothing || abs(residual) < abs(incumbent.residual))
            best[] = (; lambda = lambda, residual = residual, eq = eq)
        end
        return residual, eq
    end

    function evaluate_lambda_residual(lambda::Float64)
        haskey(eval_cache, lambda) && return eval_cache[lambda]
        return first(solve_at_lambda(lambda))
    end

    function evaluate_lambda_full(lambda::Float64)
        incumbent = best[]
        incumbent !== nothing && incumbent.lambda == lambda &&
            return incumbent.residual, incumbent.eq
        return solve_at_lambda(lambda)
    end

    r_low = evaluate_lambda_residual(p.lambdaMin)
    r_high = evaluate_lambda_residual(p.lambdaMax)

    if p.verbose
        @printf("lambda = %.8f: residual = %.8e\n", p.lambdaMin, r_low)
        @printf("lambda = %.8f: residual = %.8e\n", p.lambdaMax, r_high)
        flush(stdout)
    end

    lambda_low = p.lambdaMin
    lambda_high = p.lambdaMax

    if !isfinite(r_low) || !isfinite(r_high) || sign(r_low) == sign(r_high)
        if p.verbose
            @printf("\nNo sign change on requested bracket. Searching %d lambda values.\n",
                    p.nLambdaSearch)
            flush(stdout)
        end
        grid = collect(range(p.lambdaMin, p.lambdaMax, length = p.nLambdaSearch))
        residuals = fill(NaN, length(grid))
        for (i, lambda) in enumerate(grid)
            residuals[i] = evaluate_lambda_residual(Float64(lambda))
            if p.verbose
                @printf("lambda search %d/%d: lambda=%.8f residual=%.8e\n",
                        i, length(grid), lambda, residuals[i])
                flush(stdout)
            end
        end
        bracket = find_bracket(grid, residuals)
        if bracket === nothing
            best[] === nothing &&
                error("Could not evaluate any finite government residual")
            @warn "lambda solver: no sign change found; using best grid-search " *
                  "lambda $(best[].lambda) with residual $(best[].residual)"
            return attach_elapsed(best[].eq, start_time, p; converged = false)
        end
        i_low, i_high = bracket
        lambda_low, lambda_high = grid[i_low], grid[i_high]
        r_low, r_high = residuals[i_low], residuals[i_high]
        if p.verbose
            @printf("Using lambda bracket [%.8f, %.8f].\n", lambda_low, lambda_high)
            flush(stdout)
        end
    end

    root_eval_count = Ref(0)
    function residual_only(lambda::Float64)
        key = Float64(lambda)
        was_cached = haskey(eval_cache, key)
        residual = evaluate_lambda_residual(lambda)
        if !was_cached
            root_eval_count[] += 1
            if p.verbose && p.printEveryLambda > 0 &&
               (root_eval_count[] == 1 ||
                root_eval_count[] % p.printEveryLambda == 0)
                @printf("lambda eval %d: lambda=%.8f, residual=%.8e\n",
                        root_eval_count[], lambda, residual)
                flush(stdout)
            end
        end
        return residual
    end

    try
        lambda_root = Roots.find_zero(
            residual_only, (lambda_low, lambda_high), Roots.Brent();
            xatol = p.tolLambda,
            maxevals = max(p.maxIterLambda, 20),
        )
        r_root, eq_root = evaluate_lambda_full(Float64(lambda_root))
        if p.verbose
            @printf("lambda root: lambda=%.8f, residual=%.8e\n", lambda_root, r_root)
            flush(stdout)
        end
        converged = isfinite(r_root) && abs(r_root) <= p.tolGovBudget
        converged ||
            @warn "lambda solver: root residual $(r_root) exceeds tolerance $(p.tolGovBudget)"
        return attach_elapsed(eq_root, start_time, p; converged = converged)
    catch err
        best[] === nothing && rethrow(err)
        @warn "lambda solver: Brent failed; using best evaluated lambda " *
              "$(best[].lambda) with residual $(best[].residual)" exception = err
        return attach_elapsed(best[].eq, start_time, p; converged = false)
    end
end

# Failure modes (no bracket, Brent throw, residual above tolerance) are reported
# with @warn at the point of failure, so eq carries only `converged` beyond the
# equilibrium objects themselves.
function attach_elapsed(eq, start_time::Float64, p::HIParams; converged::Bool)
    elapsed = time() - start_time
    eq_out = merge(eq, (; converged = converged, elapsedSeconds = elapsed))
    if p.verbose
        print_upper_bound_warning(eq_out.statistics)
        @printf("total solve time          = %.3f seconds\n", elapsed)
        flush(stdout)
    end
    return eq_out
end

function government_residual_at_lambda(lambda::Float64, p::HIParams)
    aggs, stats, stats_all, welfare, solutions = solve_aggregates_for_lambda(lambda, p)
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
        solutions = solutions,
        parameters = p,
    )
    return residual, eq
end

function solve_policies_for_kappa(lambda::Float64, kappa::Float64,
                                  first_ap::Vector{Int},
                                  terminal_first_ap::Int,
                                  q_by_ap::Vector{Float64},
                                  tax_base::Matrix{Float64}, p::HIParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nAge = p.J + 1

    Vnext = zeros(nA, nZ, nE)
    Vcur = similar(Vnext)
    EV = zeros(nA, nZ)
    policyAIndex = Array{Int32}(undef, nA, nZ, nE, nAge)
    policyA = Array{Float64}(undef, nA, nZ, nE, nAge)
    policyH = Array{Float64}(undef, nA, nZ, nE, nAge)

    flow_u, flow_h = p.asset_choice_method == :grid_search ?
                     precompute_flow_payoffs(lambda, first_ap, q_by_ap, tax_base, p) :
                     (nothing, nothing)

    beta = p.beta
    util_weight = 1.0 - beta

    for age in nAge:-1:1
        compute_expected_value!(EV, Vnext, p)

        if p.asset_choice_method == :grid_search
            solve_policy_age_grid_search!(
                Vcur, policyAIndex, policyA, policyH, flow_u, flow_h, EV,
                age, first_ap, terminal_first_ap, p, util_weight, beta,
            )
        else
            solve_policy_age_interpolated!(
                Vcur, policyAIndex, policyA, policyH, EV, age, lambda, kappa,
                tax_base, p, util_weight, beta,
            )
        end

        Vnext, Vcur = Vcur, Vnext
    end

    welfare_value_function = expected_initial_value(Vnext, kappa, p)
    return policyAIndex, policyA, policyH, welfare_value_function
end

"""
    initial_asset_weights(kappa, p)

Grid placement of the initial asset holding for a household of type `kappa`, as
`(left, right, right_weight)` from the same Young lottery used for a'.

Both the birth value function and the simulated initial distribution must go
through THIS function. If they disagree the value-function/simulation
cross-check in `finalize_welfare` breaks -- which is exactly the guard wanted,
since the two are otherwise independent computations of the same object.
"""
function initial_asset_weights(kappa::Float64, p::HIParams)
    a0 = p.a0_scales_with_kappa ? p.a0 * exp(kappa) : p.a0
    return asset_transition_weights(clamp(a0, first(p.a_grid), last(p.a_grid)), p)
end

function expected_initial_value(V0, kappa::Float64, p::HIParams)
    il, ir, w = initial_asset_weights(kappa, p)
    expected_value = 0.0
    @inbounds for iz in eachindex(p.z_grid), ie in eachindex(p.eps_grid)
        prob = p.z0_probs[iz] * p.Peps[ie]
        expected_value += prob * ((1.0 - w) * V0[il, iz, ie] + w * V0[ir, iz, ie])
    end
    return expected_value
end

function solve_policy_age_grid_search!(Vcur, policyAIndex, policyA, policyH,
                                       flow_u, flow_h, EV, age::Int,
                                       first_ap::Vector{Int},
                                       terminal_first_ap::Int, p::HIParams,
                                       util_weight::Float64, beta::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)

    @inbounds for ia in 1:nA
        for iz in 1:nZ
            ia_first = age == p.J + 1 ? terminal_first_ap : first_ap[iz]
            for ie in 1:nE
                best_val = -Inf
                best_iap = ia_first
                best_h = p.hMin

                for iap in ia_first:nA
                    u = flow_u[iap, ia, iz, ie]
                    if isfinite(u)
                        val = util_weight * u + beta * EV[iap, iz]
                        if val > best_val
                            best_val = val
                            best_iap = iap
                            best_h = flow_h[iap, ia, iz, ie]
                        end
                    end
                end

                Vcur[ia, iz, ie] = best_val
                policyAIndex[ia, iz, ie, age] = Int32(best_iap)
                policyA[ia, iz, ie, age] = p.a_grid[best_iap]
                policyH[ia, iz, ie, age] = best_h
            end
        end
    end
    return nothing
end

# =============================================================================
# THE IMPACT MPC
# =============================================================================
# Kaplan and Violante (2022), their equation (2): for a household in state
# (b, y) when an unanticipated windfall x arrives,
#
#     m_0(x; b, y) = [ c(b + x, y) - c(b, y) ] / x,
#
# averaged under the distribution as in their equation (D.7). Here the windfall
# lands on current assets and the average is taken with the forward-pass mass
# over two age coverages: ages j = 0,...,J and the statistics window.
#
# The level is not comparable to theirs. KV report a quarterly MPC and this
# model is annual, and hours are endogenous here, so c(a + x) is net of a labor
# supply response. The formula is theirs; the consumption function is this
# model's.
#
# c(a + x) is read off the consumption policy by linear interpolation in a, as
# `coninterp_mpc` does in Discrete_HA. The reported MPC is an ARC, not a
# derivative, so it converges in the asset grid rather than in the windfall.
# :grid_search biases it up at coarse grids, by roughly 3% at the production
# nA = 151; raise nA or switch to :interpolate when the MPC is the object of
# interest. See NOTES.md for the measured convergence.
@inline function interpolate_consumption(con::AbstractVector{Float64},
                                        a_target::Float64, p::HIParams)
    grid = p.a_grid
    nA = length(grid)
    # Off-grid handling follows `extend_interp` in Discrete_HA's solve_EGP.m:
    # continue the last segment's slope above the top node rather than clamp,
    # which would report slope 0. The flag is still returned as a diagnostic.
    if a_target >= grid[nA]
        slope = (con[nA] - con[nA-1]) / (grid[nA] - grid[nA-1])
        return con[nA] + slope * (a_target - grid[nA]), true
    elseif a_target <= grid[1]
        # Their rule below the grid: consume the whole shortfall, slope 1.
        # Unreachable for a positive windfall, but kept so a negative
        # `mpc_shock` behaves as it does in their code.
        return con[1] + (a_target - grid[1]), false
    end
    # The grid is sorted, so one searchsorted gives the bracket (i-1, i).
    i = searchsortedfirst(grid, a_target)
    i <= 1 && return con[1], false
    lo, hi = grid[i-1], grid[i]
    w = (a_target - lo) / (hi - lo)
    return (1.0 - w) * con[i-1] + w * con[i], false
end

# Consumption over the whole asset grid at one age: the perturbed state a + x
# sits at a different asset index, so the policy must be available away from the
# cell being visited. Filled once per age into a reused buffer.
function fill_consumption_policy!(con::Array{Float64,3}, age::Int,
                                  policyAIndex, policyA, policyH,
                                  q_by_ap::Vector{Float64},
                                  tax_base::Matrix{Float64},
                                  p::HIParams, lambda::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    on_grid = p.asset_choice_method == :grid_search
    @inbounds for ia in 1:nA
        a = p.a_grid[ia]
        for iz in 1:nZ, ie in 1:nE
            if on_grid
                iap = Int(policyAIndex[ia, iz, ie, age])
                ap = p.a_grid[iap]
                q_ap = q_by_ap[iap]
            else
                ap = policyA[ia, iz, ie, age]
                q_ap = asset_price(ap, p)
            end
            h = policyH[ia, iz, ie, age]
            con[ia, iz, ie] =
                lambda * tax_base[iz, ie] * h^(1.0 - p.tau) + a - q_ap * ap
        end
    end
    return nothing
end

function simulate_kappa!(C, H, Y, A, stats::HIStatsAccumulator,
                         stats_all::HIStatsAccumulator,
                         stats_lo::HIStatsAccumulator,
                         policyAIndex, policyA, policyH, kappa, pkappa,
                         first_ap::Vector{Int}, terminal_first_ap::Int,
                         q_by_ap::Vector{Float64},
                         tax_base::Matrix{Float64}, wage_base::Matrix{Float64},
                         p::HIParams, lambda::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nAge = p.J + 1
    dist = zeros(nA, nZ, nE)
    dist_next = similar(dist)
    ia0_left, ia0_right, ia0_w = initial_asset_weights(kappa, p)
    h_upper = hours_upper_bound(p)
    welfare_simulation = 0.0
    # Consumption over the whole asset grid at the age being visited, refilled
    # once per age; the MPC reads c(a + mpc_shock) off it.
    con_age = Array{Float64}(undef, nA, nZ, nE)
    mpc_shock = p.mpc_shock

    @inbounds for iz in 1:nZ, ie in 1:nE
        prob = p.z0_probs[iz] * p.Peps[ie]
        dist[ia0_left, iz, ie] += (1.0 - ia0_w) * prob
        dist[ia0_right, iz, ie] += ia0_w * prob
    end

    @inbounds for age in 1:nAge
        in_stats_window = p.stats_age_lo <= age <= p.stats_age_hi
        at_stats_age_lo = age == p.stats_age_lo
        fill!(dist_next, 0.0)
        utility_weight = (1.0 - p.beta) * p.beta^(age - 1)
        fill_consumption_policy!(con_age, age, policyAIndex, policyA, policyH,
                                 q_by_ap, tax_base, p, lambda)

        for ia in 1:nA
            a = p.a_grid[ia]
            for iz in 1:nZ, ie in 1:nE
                mass = dist[ia, iz, ie]
                if mass <= p.massTol
                    continue
                end

                if p.asset_choice_method == :grid_search
                    iap = Int(policyAIndex[ia, iz, ie, age])
                    ap = p.a_grid[iap]
                    q_ap = q_by_ap[iap]
                    lower_idx = age == nAge ? terminal_first_ap : first_ap[iz]
                    lower_ap = p.a_grid[lower_idx]
                    at_borrowing_constraint = iap == lower_idx
                    at_asset_upper = iap == nA
                    next_left = iap
                    next_right = iap
                    next_right_weight = 0.0
                else
                    ap = policyA[ia, iz, ie, age]
                    q_ap = asset_price(ap, p)
                    lower_ap = asset_choice_lower_bound(age, kappa, iz, p)
                    at_borrowing_constraint =
                        abs(ap - lower_ap) <= asset_choice_bound_tol(lower_ap, p)
                    at_asset_upper =
                        ap >= asset_upper_bound(p) - asset_choice_bound_tol(asset_upper_bound(p), p)
                    next_left, next_right, next_right_weight = asset_transition_weights(ap, p)
                end
                h = policyH[ia, iz, ie, age]
                c = lambda * tax_base[iz, ie] * h^(1.0 - p.tau) + a - q_ap * ap
                y = wage_base[iz, ie] * h
                u = log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta)
                # A borrowing limit only exists before the terminal age, where
                # a' >= 0 is imposed instead. Its mass is accumulated separately
                # so the reported means average over ages j = 0,...,J-1 only.
                binding_age = age < nAge
                true_borrowing_limit = binding_age ?
                                       -borrowing_limit(kappa, iz, p) : 0.0
                effective_borrowing_limit = binding_age ? -lower_ap : 0.0
                welfare_simulation += utility_weight * mass * u

                # Both legs come from the same interpolant: mixing the exact
                # c(a) with an interpolated c(a + x) would put the
                # interpolation error straight into the numerator.
                c_here, _ = interpolate_consumption(view(con_age, :, iz, ie),
                                                    a, p)
                c_up, mpc_extrapolated = interpolate_consumption(
                    view(con_age, :, iz, ie), a + mpc_shock, p)
                mpc = (c_up - c_here) / mpc_shock

                weighted_mass = pkappa * mass
                C[age] += weighted_mass * c
                H[age] += weighted_mass * h
                Y[age] += weighted_mass * y
                A[age] += weighted_mass * ap

                # Three age coverages from one body: the calibration window,
                # every age, and the single entry age. Each keeps its own
                # total_mass, so the gate wraps the whole call.
                accumulate_stats!(stats_all, weighted_mass, ia, a, ap, h, c, y,
                                  true_borrowing_limit, effective_borrowing_limit,
                                  at_borrowing_constraint, at_asset_upper,
                                  h_upper, binding_age, mpc, mpc_extrapolated, false, p)
                in_stats_window &&
                    accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      at_borrowing_constraint, at_asset_upper,
                                      h_upper, binding_age, mpc, mpc_extrapolated,
                                      p.collect_distributions, p)
                at_stats_age_lo &&
                    accumulate_stats!(stats_lo, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      at_borrowing_constraint, at_asset_upper,
                                      h_upper, binding_age, mpc, mpc_extrapolated, false, p)

                if age < nAge
                    for izp in 1:nZ
                        zprob = p.Pz[iz, izp]
                        if zprob == 0.0
                            continue
                        end
                        for iep in 1:nE
                            next_mass = mass * zprob * p.Peps[iep]
                            dist_next[next_left, izp, iep] +=
                                (1.0 - next_right_weight) * next_mass
                            if next_right != next_left && next_right_weight > 0.0
                                dist_next[next_right, izp, iep] +=
                                    next_right_weight * next_mass
                            end
                        end
                    end
                end
            end
        end

        dist, dist_next = dist_next, dist
    end
    return welfare_simulation
end

function solve_aggregates_for_lambda(lambda::Float64, p::HIParams)
    nAge = p.J + 1
    nKappa = length(p.kappa_grid)
    C = zeros(nAge)
    H = zeros(nAge)
    Y = zeros(nAge)
    A = zeros(nAge)
    stats_acc = HIStatsAccumulator(length(p.a_grid))
    stats_all_acc = HIStatsAccumulator(length(p.a_grid))
    stats_lo_acc = HIStatsAccumulator(length(p.a_grid))

    C_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    H_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    Y_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    A_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    stats_by_kappa = Vector{HIStatsAccumulator}(undef, nKappa)
    stats_all_by_kappa = Vector{HIStatsAccumulator}(undef, nKappa)
    stats_lo_by_kappa = Vector{HIStatsAccumulator}(undef, nKappa)
    saved_policyAIndex = p.store_solutions ?
                         Vector{Array{Int32,4}}(undef, nKappa) : Array{Int32,4}[]
    saved_policyA = p.store_solutions ?
                    Vector{Array{Float64,4}}(undef, nKappa) : Array{Float64,4}[]
    saved_policyH = p.store_solutions ?
                    Vector{Array{Float64,4}}(undef, nKappa) : Array{Float64,4}[]
    q_by_ap = asset_prices(p)
    terminal_first_ap = first_nonnegative_asset_index(p)
    welfare_value_function_by_kappa = Vector{Float64}(undef, length(p.kappa_grid))
    welfare_simulation_by_kappa = similar(welfare_value_function_by_kappa)

    # Each kappa owns local arrays and a local stats accumulator; the reduction
    # after this loop avoids races on aggregate sums and distribution vectors.
    Threads.@threads :static for ik in 1:nKappa
        kappa = p.kappa_grid[ik]
        pkappa = p.Pkappa[ik]
        first_ap = first_feasible_asset_indices(kappa, p)
        tax_base, wage_base = precompute_income_bases(kappa, p)
        policyAIndex, policyA, policyH, welfare_value_function = solve_policies_for_kappa(
            lambda, kappa, first_ap, terminal_first_ap, q_by_ap, tax_base, p,
        )
        C_local = C_by_kappa[ik]
        H_local = H_by_kappa[ik]
        Y_local = Y_by_kappa[ik]
        A_local = A_by_kappa[ik]
        stats_local = HIStatsAccumulator(length(p.a_grid))
        stats_all_local = HIStatsAccumulator(length(p.a_grid))
        stats_lo_local = HIStatsAccumulator(length(p.a_grid))
        welfare_simulation = simulate_kappa!(
            C_local, H_local, Y_local, A_local, stats_local,
            stats_all_local, stats_lo_local,
            policyAIndex, policyA, policyH,
            kappa, pkappa, first_ap, terminal_first_ap, q_by_ap,
            tax_base, wage_base, p, lambda,
        )
        stats_by_kappa[ik] = stats_local
        stats_all_by_kappa[ik] = stats_all_local
        stats_lo_by_kappa[ik] = stats_lo_local
        welfare_value_function_by_kappa[ik] = welfare_value_function
        welfare_simulation_by_kappa[ik] = welfare_simulation

        if p.store_solutions
            saved_policyAIndex[ik] = policyAIndex
            saved_policyA[ik] = policyA
            saved_policyH[ik] = policyH
        end
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

    solutions = p.store_solutions ? (;
        lambda = lambda,
        policyAIndex_by_kappa = saved_policyAIndex,
        policyA_by_kappa = saved_policyA,
        policyH_by_kappa = saved_policyH,
        a_grid = p.a_grid,
        z_grid = p.z_grid,
        eps_grid = p.eps_grid,
        kappa_grid = p.kappa_grid,
    ) : nothing

    stats = finalize_statistics(stats_acc, p)
    stats_all = finalize_statistics(stats_all_acc, p)
    # The entry-age cross-section, reduced with the same machinery. Only the
    # two asset ratios are carried over, both divided by the WINDOW's mean
    # labor income so they are comparable to the calibration targets.
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
    return (; C = C, H = H, Y = Y, A = A), stats, stats_all, welfare, solutions
end

"""
    finalize_statistics(stats, p)

The published statistics for one accumulated group: the shared core, plus
the impact-MPC block.
"""
finalize_statistics(stats::HIStatsAccumulator, p::HIParams) =
    merge(core_statistics(stats, p), mpc_statistics(stats, p))

ar1_grid(n::Int, rho::Float64, innovation_mean::Float64, innovation_sd::Float64,
         method::Symbol, tauchen_width::Float64) =
    first(quantecon_ar1(n, rho, innovation_mean, innovation_sd;
                        method = method, width = tauchen_width))

gauss_hermite_grid(n::Int, mean::Float64, sd::Float64) =
    first(normal_gauss_hermite(n, mean, sd))

# The builders below already scale their weights, but a second pass costs
# nothing and leaves each probability vector summing to one in floating point.
gauss_hermite_probs(n::Int, mean::Float64, sd::Float64) =
    normalize_probabilities(last(normal_gauss_hermite(n, mean, sd)), "Gauss-Hermite weights")

ar1_initial_probabilities(z_initial::Float64, z_grid::Vector{Float64}, rho::Float64,
                          innovation_mean::Float64, innovation_sd::Float64) =
    normalize_probabilities(
        ar1_conditional_probabilities(z_initial, z_grid, rho, innovation_mean, innovation_sd),
        "z0_probs")

