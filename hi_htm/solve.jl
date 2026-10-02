# =============================================================================
# solve.jl
#
# Solver: backward induction for the policies, a forward pass for the
# cross-section and its statistics, and a Brent solve for the tax level.
# For the history-independent tax model with hand-to-mouth agents.
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
    solve_hi(p::HIParams)

Solve for `lambda`, household policies, age distributions, and aggregates.
Returns `eq`, a named tuple of equilibrium objects. Household policies are
attached as `eq.solutions` when `p.store_solutions = true` (otherwise
`eq.solutions === nothing`). Build `p` with `make_history_independent_params()`.
"""
function solve_hi(p::HIParams)
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

# -----------------------------------------------------------------------------
# Hand-to-mouth block
# -----------------------------------------------------------------------------

"""
    HTMTransition

The exogenous hand-to-mouth asset rule, precomputed once per `HIParams`:
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
money. A terminal debtor then settles up: `a' = 0` and `cash = a`, so
consumption is `y + a` rather than `y + (1-qBorr)a`. psmodel.tex is
infinite-horizon and says nothing about this; it is a choice made here.
"""
struct HTMTransition
    next_assets::Vector{Float64}
    cash::Vector{Float64}
    left::Vector{Int}
    right::Vector{Int}
    weight::Vector{Float64}
    clipped::Bool
end

function htm_transition(p::HIParams; terminal::Bool = false)
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
        l, r, w = asset_transition_weights(ap, p)
        left[ia] = l; right[ia] = r; weight[ia] = w
    end

    return HTMTransition(next_assets, cash, left, right, weight, clipped)
end

"""
    precompute_htm_payoffs(lambda, tax_base, htm, p)

Flow utility and hours for the hand-to-mouth state, over `(a, z, eps)`.

The HtM choice is STATIC: `a'` is exogenous and nothing else links periods, so
hours solve the same within-period first-order condition the saver uses, at
`cash = a - q(a')a'`, with no continuation term. That is why this is computed
once per `(lambda, kappa)` and reused at every age, and why the HtM half of the
backward recursion never maximizes -- it only evaluates. (The history-DEPENDENT
extension in `hdinf_htm` has no such shortcut: there hours move the past-income
stocks, so its HtM block must maximize on every sweep.)
"""
function precompute_htm_payoffs(lambda::Float64, tax_base::Matrix{Float64},
                                htm::HTMTransition, p::HIParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    flow_u = fill(-Inf, nA, nZ, nE)
    flow_h = fill(NaN, nA, nZ, nE)

    @inbounds for ia in 1:nA
        cash = htm.cash[ia]
        for iz in 1:nZ, ie in 1:nE
            u, h = optimal_labor_foc(cash, lambda * tax_base[iz, ie], p)
            if isfinite(u)
                flow_u[ia, iz, ie] = u
                flow_h[ia, iz, ie] = h
            end
        end
    end
    return flow_u, flow_h
end

"""
    update_htm_values!(VcurH, policyHtmH, flow_u_H, flow_h_H, EV_H, htm, age, p,
                       util_weight, beta)

One age of the hand-to-mouth recursion. No maximization: the static flow payoff
is looked up and the continuation is read at the exogenous `a'` through the
stored Young lottery.
"""
function update_htm_values!(VcurH, policyHtmH, flow_u_H, flow_h_H, EV_H,
                            htm::HTMTransition, age::Int, p::HIParams,
                            util_weight::Float64, beta::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    @inbounds for ia in 1:nA
        il = htm.left[ia]
        ir = htm.right[ia]
        w = htm.weight[ia]
        for iz in 1:nZ
            continuation = (1.0 - w) * EV_H[il, iz] + w * EV_H[ir, iz]
            for ie in 1:nE
                u = flow_u_H[ia, iz, ie]
                VcurH[ia, iz, ie] = isfinite(u) ?
                    util_weight * u + beta * continuation : VINFEASIBLE
                policyHtmH[ia, iz, ie, age] = flow_h_H[ia, iz, ie]
            end
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
                                  p::HIParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nAge = p.J + 1

    # One value function per access state. The saver's own block is untouched;
    # it just receives the MIXED continuation in place of the plain one.
    VnextS = zeros(nA, nZ, nE)
    VcurS = similar(VnextS)
    VnextH = zeros(nA, nZ, nE)
    VcurH = similar(VnextH)
    EVS_raw = zeros(nA, nZ)
    EVH_raw = zeros(nA, nZ)
    EVmixS = zeros(nA, nZ)
    EVmixH = zeros(nA, nZ)
    policyAIndex = Array{Int32}(undef, nA, nZ, nE, nAge)
    policyA = Array{Float64}(undef, nA, nZ, nE, nAge)
    policyH = Array{Float64}(undef, nA, nZ, nE, nAge)
    policyHtmH = Array{Float64}(undef, nA, nZ, nE, nAge)

    flow_u, flow_h = p.asset_choice_method == :grid_search ?
                     precompute_flow_payoffs(lambda, first_ap, q_by_ap, tax_base, p) :
                     (nothing, nothing)
    # Static, so computed once and reused at every age -- except the terminal
    # age, whose asset rule differs and therefore whose cash does too.
    flow_u_H, flow_h_H = precompute_htm_payoffs(lambda, tax_base, htm, p)
    flow_u_HT, flow_h_HT = precompute_htm_payoffs(lambda, tax_base, htm_terminal, p)

    beta = p.beta
    util_weight = 1.0 - beta

    for age in nAge:-1:1
        compute_expected_value!(EVS_raw, VnextS, p)
        compute_expected_value!(EVH_raw, VnextH, p)
        # The access shock is independent of (z', eps'), so the mixing is done
        # AFTER the expectation and the saver's solver sees a drop-in
        # replacement for EV. A state with no feasible choice carries the
        # FINITE sentinel, so these products are 0.0 when the weight is zero;
        # with -Inf the pSS = 1 corner would be NaN.
        @inbounds for i in eachindex(EVS_raw)
            evs = EVS_raw[i]
            evh = EVH_raw[i]
            EVmixS[i] = p.pSS * evs + (1.0 - p.pSS) * evh
            EVmixH[i] = p.pHH * evh + (1.0 - p.pHH) * evs
        end

        if p.asset_choice_method == :grid_search
            solve_policy_age_grid_search!(
                VcurS, policyAIndex, policyA, policyH, flow_u, flow_h, EVmixS,
                age, first_ap, terminal_first_ap, p, util_weight, beta,
            )
        else
            solve_policy_age_interpolated!(
                VcurS, policyAIndex, policyA, policyH, EVmixS, age, lambda, kappa,
                tax_base, p, util_weight, beta,
            )
        end

        is_terminal = age == nAge
        update_htm_values!(VcurH, policyHtmH,
                           is_terminal ? flow_u_HT : flow_u_H,
                           is_terminal ? flow_h_HT : flow_h_H,
                           EVmixH, is_terminal ? htm_terminal : htm,
                           age, p, util_weight, beta)

        VnextS, VcurS = VcurS, VnextS
        VnextH, VcurH = VcurH, VnextH
    end

    welfare_value_function = expected_initial_value(VnextS, VnextH, kappa, p)
    return policyAIndex, policyA, policyH, policyHtmH, welfare_value_function
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

# Newborns draw their access state from the STATIONARY distribution (piS, piH),
# independently of (a0, z, eps), so the birth value is the piS/piH mix of the
# two value functions at the same asset placement. `simulate_kappa!` seeds the
# distribution the same way; if the two disagree, the welfare cross-check in
# `finalize_welfare` reports it.
function expected_initial_value(V0S, V0H, kappa::Float64, p::HIParams)
    il, ir, w = initial_asset_weights(kappa, p)
    expected_value = 0.0
    @inbounds for iz in eachindex(p.z_grid), ie in eachindex(p.eps_grid)
        prob = p.z0_probs[iz] * p.Peps[ie]
        vS = (1.0 - w) * V0S[il, iz, ie] + w * V0S[ir, iz, ie]
        vH = (1.0 - w) * V0H[il, iz, ie] + w * V0H[ir, iz, ie]
        expected_value += prob * (p.piS * vS + p.piH * vH)
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
                best_val = VINFEASIBLE
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

function simulate_kappa!(C, H, Y, A, stats::HIStatsAccumulator,
                         stats_all::HIStatsAccumulator,
                         stats_lo::HIStatsAccumulator,
                         policyAIndex, policyA, policyH, policyHtmH,
                         kappa, pkappa,
                         first_ap::Vector{Int}, terminal_first_ap::Int,
                         q_by_ap::Vector{Float64},
                         tax_base::Matrix{Float64}, wage_base::Matrix{Float64},
                         htm::HTMTransition, htm_terminal::HTMTransition,
                         p::HIParams, lambda::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nAge = p.J + 1
    dist = zeros(nA, nZ, nE, 2)   # 4th axis: 1 = saver, 2 = hand-to-mouth
    dist_next = similar(dist)
    ia0_left, ia0_right, ia0_w = initial_asset_weights(kappa, p)
    h_upper = hours_upper_bound(p)
    a_top = last(p.a_grid)
    welfare_simulation = 0.0

    # Newborns draw the access state from the stationary distribution, so the
    # HtM share is constant over the life cycle. `expected_initial_value` mixes
    # the birth value functions with the same weights.
    @inbounds for iz in 1:nZ, ie in 1:nE
        prob = p.z0_probs[iz] * p.Peps[ie]
        for (iacc, share) in ((1, p.piS), (2, p.piH))
            dist[ia0_left, iz, ie, iacc] += (1.0 - ia0_w) * prob * share
            dist[ia0_right, iz, ie, iacc] += ia0_w * prob * share
        end
    end

    @inbounds for age in 1:nAge
        in_stats_window = p.stats_age_lo <= age <= p.stats_age_hi
        at_stats_age_lo = age == p.stats_age_lo
        fill!(dist_next, 0.0)
        utility_weight = (1.0 - p.beta) * p.beta^(age - 1)

        for ia in 1:nA
            a = p.a_grid[ia]
            for iz in 1:nZ, ie in 1:nE, iacc in 1:2
                mass = dist[ia, iz, ie, iacc]
                if mass <= p.massTol
                    continue
                end
                is_htm = iacc == 2

                if is_htm
                    # Exogenous rule, clamped to a' >= 0 at the terminal age.
                    # The borrowing limit is not a constraint on someone who
                    # does not choose, so at_borrowing_constraint is false.
                    hm = age == nAge ? htm_terminal : htm
                    ap = hm.next_assets[ia]
                    q_ap = asset_price(ap, p)
                    lower_idx = age == nAge ? terminal_first_ap : first_ap[iz]
                    lower_ap = p.a_grid[lower_idx]
                    at_borrowing_constraint = false
                    at_asset_upper = ap >= a_top - upper_bound_level_tol(a_top)
                    next_left = hm.left[ia]
                    next_right = hm.right[ia]
                    next_right_weight = hm.weight[ia]
                    # The SAME cash the value function's flow payoff used;
                    # a - q_ap*ap agrees analytically but carries rounding
                    # noise on the positive branch, where it must be 0.
                    cash = hm.cash[ia]
                elseif p.asset_choice_method == :grid_search
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
                    cash = a - q_ap * ap
                else
                    ap = policyA[ia, iz, ie, age]
                    q_ap = asset_price(ap, p)
                    lower_ap = asset_choice_lower_bound(age, kappa, iz, p)
                    at_borrowing_constraint =
                        abs(ap - lower_ap) <= asset_choice_bound_tol(lower_ap, p)
                    at_asset_upper =
                        ap >= asset_upper_bound(p) - asset_choice_bound_tol(asset_upper_bound(p), p)
                    next_left, next_right, next_right_weight = asset_transition_weights(ap, p)
                    cash = a - q_ap * ap
                end
                h = is_htm ? policyHtmH[ia, iz, ie, age] : policyH[ia, iz, ie, age]
                if !(h > 0.0)
                    acc_label = is_htm ? "H" : "S"
                    error("infeasible policy on a positive-mass state " *
                          "(age=$age, ia=$ia, iz=$iz, ie=$ie, " *
                          "access=$acc_label): no feasible choice was found")
                end
                c = lambda * tax_base[iz, ie] * h^(1.0 - p.tau) + cash
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
                                  h_upper, binding_age, is_htm, false, p)
                in_stats_window &&
                    accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      at_borrowing_constraint, at_asset_upper,
                                      h_upper, binding_age, is_htm, p.collect_distributions, p)
                at_stats_age_lo &&
                    accumulate_stats!(stats_lo, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      at_borrowing_constraint, at_asset_upper,
                                      h_upper, binding_age, is_htm, false, p)

                if age < nAge
                    # The access chain is independent of (z', eps') and of a',
                    # so the (a', z', eps') mass is built exactly as in `hi`
                    # and then split across the two access states.
                    stay_weight = is_htm ? p.pHH : p.pSS
                    switch_weight = 1.0 - stay_weight
                    iacc_switch = 3 - iacc
                    for izp in 1:nZ
                        zprob = p.Pz[iz, izp]
                        if zprob == 0.0
                            continue
                        end
                        for iep in 1:nE
                            next_mass = mass * zprob * p.Peps[iep]
                            left_mass = (1.0 - next_right_weight) * next_mass
                            dist_next[next_left, izp, iep, iacc] +=
                                stay_weight * left_mass
                            dist_next[next_left, izp, iep, iacc_switch] +=
                                switch_weight * left_mass
                            if next_right != next_left && next_right_weight > 0.0
                                right_mass = next_right_weight * next_mass
                                dist_next[next_right, izp, iep, iacc] +=
                                    stay_weight * right_mass
                                dist_next[next_right, izp, iep, iacc_switch] +=
                                    switch_weight * right_mass
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
    # kappa-independent, so built once and shared read-only across threads.
    htm = htm_transition(p)
    htm_terminal = htm_transition(p; terminal = true)
    welfare_value_function_by_kappa = Vector{Float64}(undef, length(p.kappa_grid))
    welfare_simulation_by_kappa = similar(welfare_value_function_by_kappa)

    # Each kappa owns local arrays and a local stats accumulator; the reduction
    # after this loop avoids races on aggregate sums and distribution vectors.
    Threads.@threads :static for ik in 1:nKappa
        kappa = p.kappa_grid[ik]
        pkappa = p.Pkappa[ik]
        first_ap = first_feasible_asset_indices(kappa, p)
        tax_base, wage_base = precompute_income_bases(kappa, p)
        policyAIndex, policyA, policyH, policyHtmH, welfare_value_function =
            solve_policies_for_kappa(
                lambda, kappa, first_ap, terminal_first_ap, q_by_ap, tax_base,
                htm, htm_terminal, p,
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
            policyAIndex, policyA, policyH, policyHtmH,
            kappa, pkappa, first_ap, terminal_first_ap, q_by_ap,
            tax_base, wage_base, htm, htm_terminal, p, lambda,
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
    # ---------------------------------------------------------------------
    # LABOR
    # --------------------------------------------------------------------- income so they are comparable to the calibration targets.
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
the realized hand-to-mouth share.
"""
finalize_statistics(stats::HIStatsAccumulator, p::HIParams) =
    merge(core_statistics(stats, p),
          (; shareHandToMouth = stats.htm_mass / stats.total_mass))

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

