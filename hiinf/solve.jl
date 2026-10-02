# =============================================================================
# solve.jl
#
# Solver: backward induction for the policies, a forward pass for the
# cross-section and its statistics, and a Brent solve for the tax level.
# For the infinite-horizon history-independent tax model.
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
using BewleyCommon: finalize_welfare, precompute_income_bases

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
        println("\n=== History-independent tax infinite-horizon solver ===")
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
    warn_if_unsettled(eq_out, p; converged = converged)
    if p.verbose
        print_upper_bound_warning(eq_out.statistics)
        @printf("total solve time          = %.3f seconds\n", elapsed)
        flush(stdout)
    end
    return eq_out
end

function government_residual_at_lambda(lambda::Float64, p::HIParams)
    aggs, stats, stats_all, welfare, solutions, diag =
        solve_aggregates_for_lambda(lambda, p)

    # The budget is STILL a present value over ages: the agent is infinitely
    # lived, but the cohort's aggregates vary over its life, so this is not the
    # stationary condition a steady-state Bewley model would use.
    #
    # The path is iterated to maxAge for every kappa, after which Y_j - C_j is
    # constant and the remaining terms sum in closed form:
    #   sum_{j>Jc} qGov^j (Y-C)_inf = (Y-C)_inf * qGov^(Jc+1) / (1 - qGov).
    # Jc is maxAge now. It used to be maximum(settled_by_kappa), which made the
    # loop below read the band where kappas had dropped out one at a time and
    # the aggregates were partial sums.
    Jc = diag.settledAge
    lhs = 0.0
    for j in 1:Jc
        lhs += p.qGov^(j - 1) * (aggs.Y[j] - aggs.C[j])
    end
    lhs += (aggs.Y[Jc] - aggs.C[Jc]) * p.qGov^Jc / (1.0 - p.qGov)
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
        # Same quantities over ages 1..maxAge instead of the calibration
        # window. Reported for comparison; nothing is calibrated on it.
        statisticsAllAges = stats_all,
        welfare = welfare,
        solutions = solutions,
        diagnostics = diag,
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
    # No age dimension on the policies: this is the memory difference from the
    # finite-horizon solver, and the reason the stationary problem is cheaper.
    Vnext = zeros(nA, nZ, nE)
    Vcur = similar(Vnext)
    EV = zeros(nA, nZ)
    policyAIndex = Array{Int32}(undef, nA, nZ, nE)
    policyA = Array{Float64}(undef, nA, nZ, nE)
    policyH = Array{Float64}(undef, nA, nZ, nE)

    flow_u, flow_h = p.asset_choice_method == :grid_search ?
                     precompute_flow_payoffs(lambda, first_ap, q_by_ap, tax_base, p) :
                     (nothing, nothing)

    beta = p.beta
    util_weight = 1.0 - beta

    # One application of the Bellman operator. `maximize = true` re-optimizes
    # and rewrites the policy; `false` applies the stored one (Howard).
    function sweep!(maximize::Bool)
        compute_expected_value!(EV, Vnext, p)
        if p.asset_choice_method == :grid_search
            if maximize
                solve_policy_age_grid_search!(
                    Vcur, policyAIndex, policyA, policyH, flow_u, flow_h, EV,
                    first_ap, p, util_weight, beta,
                )
            else
                evaluate_policy_grid_search!(
                    Vcur, policyAIndex, flow_u, EV, p, util_weight, beta,
                )
            end
        else
            # The interpolated branch has no cheap evaluation step (the flow
            # payoff is not precomputed), so it always maximizes; howardSteps
            # is ignored there and plain VFI is what runs.
            solve_policy_age_interpolated!(
                Vcur, policyAIndex, policyA, policyH, EV, lambda, kappa,
                tax_base, p, util_weight, beta,
            )
        end
        return nothing
    end

    # Swapping outside the closure keeps Vnext/Vcur unboxed: assigning to a
    # captured variable inside would make every access type-unstable.
    howard = p.asset_choice_method == :grid_search ? p.howardSteps : 0
    iters = 0
    gap = Inf
    for _ in 1:p.maxIterV
        sweep!(true)
        # Measure the sup norm over FINITE entries only, before swapping.
        # States with no feasible a' carry -Inf in both sweeps, and
        # -Inf - (-Inf) = NaN would poison the comparison: `gap <= tolV` is
        # then false forever and the iteration silently runs to maxIterV with
        # an unconverged policy. Those states never change and carry no mass,
        # so excluding them is the right measure, not a workaround.
        gap = 0.0
        @inbounds for i in eachindex(Vcur)
            (isfinite(Vcur[i]) && isfinite(Vnext[i])) || continue
            d = abs(Vcur[i] - Vnext[i])
            d > gap && (gap = d)
        end
        Vnext, Vcur = Vcur, Vnext
        iters += 1
        gap <= p.tolV && break
        for _ in 1:howard
            sweep!(false)
            Vnext, Vcur = Vcur, Vnext
        end
    end
    gap <= p.tolV ||
        @warn "value function did not converge" kappa iters gap tolV = p.tolV

    welfare_value_function = expected_initial_value(Vnext, kappa, p)
    return policyAIndex, policyA, policyH, welfare_value_function, iters, gap
end

"""
    initial_asset_weights(kappa, p)

Grid placement of the initial asset holding for a household of type `kappa`, as
`(left, right, right_weight)` from the same Young lottery used for a'.

Both the birth value function and the simulated initial distribution must go
through THIS function. If they disagree, the value-function/simulation
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
                                       flow_u, flow_h, EV,
                                       first_ap::Vector{Int}, p::HIParams,
                                       util_weight::Float64, beta::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)

    @inbounds for ia in 1:nA
        for iz in 1:nZ
            ia_first = first_ap[iz]
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
                policyAIndex[ia, iz, ie] = Int32(best_iap)
                policyA[ia, iz, ie] = p.a_grid[best_iap]
                policyH[ia, iz, ie] = best_h
            end
        end
    end
    return nothing
end

function solve_policy_age_interpolated!(Vcur, policyAIndex, policyA, policyH,
                                        EV, lambda::Float64,
                                        kappa::Float64,
                                        tax_base::Matrix{Float64}, p::HIParams,
                                        util_weight::Float64, beta::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)

    @inbounds for ia in 1:nA
        a = p.a_grid[ia]
        for iz in 1:nZ
            lower = borrowing_limit(kappa, iz, p)
            for ie in 1:nE
                income_coeff = lambda * tax_base[iz, ie]
                best_val, best_ap, best_iap, best_h = interpolated_asset_choice(
                    a, lower, income_coeff, EV, iz, p, util_weight, beta,
                )
                # Same reason as the grid-search branch: keep the value finite
                # so the next sweep's comparisons stay well defined. Here the
                # search seeds `best_val` from an evaluation rather than a
                # constant, so the guard sits at the assignment.
                Vcur[ia, iz, ie] = isfinite(best_val) ? best_val : VINFEASIBLE
                policyAIndex[ia, iz, ie] = Int32(best_iap)
                policyA[ia, iz, ie] = best_ap
                policyH[ia, iz, ie] = best_h
            end
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
    nAge = p.maxAge
    dist = zeros(nA, nZ, nE)
    dist_next = similar(dist)
    ia0_left, ia0_right, ia0_w = initial_asset_weights(kappa, p)
    h_upper = hours_upper_bound(p)
    # Flow utility per age, so the discounted sum can be closed analytically
    # past the settled age. Accumulating the discounted total directly would
    # silently truncate: stopping at age Jc drops a tail worth beta^Jc of
    # lifetime utility, 2.2e-3 at Jc = 150.
    u_by_age = zeros(nAge)
    converged_age = 0           # diagnostic only; 0 means never settled by nAge
    final_drift = NaN           # drift at the last age, reported not tested

    @inbounds for iz in 1:nZ, ie in 1:nE
        prob = p.z0_probs[iz] * p.Peps[ie]
        dist[ia0_left, iz, ie] += (1.0 - ia0_w) * prob
        dist[ia0_right, iz, ie] += ia0_w * prob
    end

    @inbounds for age in 1:nAge
        fill!(dist_next, 0.0)
        # MODEL AGE now equals the array index: index 1 is model age 1, the
        # first simulated period, at real age age0_real. The aggregates
        # C/H/Y/A and u_by_age are NOT gated -- the government budget and the
        # welfare integral need the whole path, and only the cross-sectional
        # statistics are meant to mimic a data cross-section.
        in_stats_window = p.stats_age_lo <= age <= p.stats_age_hi
        # A third window, one age wide: the cross-section AS IT ENTERS the
        # calibration window. With age0_real = 18 and stats_age_lo = 5 that is
        # real age 22, where households hold four years of accumulation rather
        # than the a0 = 0 they are born with -- which is exactly the quantity
        # the statistics window can no longer show.
        at_stats_age_lo = age == p.stats_age_lo

        for ia in 1:nA
            a = p.a_grid[ia]
            for iz in 1:nZ, ie in 1:nE
                mass = dist[ia, iz, ie]
                if mass <= p.massTol
                    continue
                end

                if p.asset_choice_method == :grid_search
                    iap = Int(policyAIndex[ia, iz, ie])
                    ap = p.a_grid[iap]
                    q_ap = q_by_ap[iap]
                    lower_idx = first_ap[iz]
                    lower_ap = p.a_grid[lower_idx]
                    at_borrowing_constraint = iap == lower_idx
                    at_asset_upper = iap == nA
                    next_left = iap
                    next_right = iap
                    next_right_weight = 0.0
                else
                    ap = policyA[ia, iz, ie]
                    q_ap = asset_price(ap, p)
                    lower_ap = borrowing_limit(kappa, iz, p)
                    at_borrowing_constraint =
                        abs(ap - lower_ap) <= asset_choice_bound_tol(lower_ap, p)
                    at_asset_upper =
                        ap >= asset_upper_bound(p) - asset_choice_bound_tol(asset_upper_bound(p), p)
                    next_left, next_right, next_right_weight = asset_transition_weights(ap, p)
                end
                h = policyH[ia, iz, ie]
                # A state with no feasible a' keeps the `best_h = hMin`
                # fallback and an infinite value; if the distribution ever
                # reaches one, consumption is non-positive and `log(c)` throws a
                # bare DomainError with no context. Report the state instead.
                h > 0.0 || error(
                    "infeasible policy on a positive-mass state " *
                    "(age=$age, ia=$ia, iz=$iz, ie=$ie, a=$a, h=$h): " *
                    "no feasible next-period asset was found when solving")
                c = lambda * tax_base[iz, ie] * h^(1.0 - p.tau) + a - q_ap * ap
                c > 0.0 || error(
                    "non-positive consumption on a positive-mass state " *
                    "(age=$age, ia=$ia, iz=$iz, ie=$ie, a=$a, ap=$ap, h=$h, c=$c)")
                y = wage_base[iz, ie] * h
                u = log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta)
                # Infinite horizon: the borrowing limit binds at every age, so
                # all mass counts towards the reported means -- the finite
                # solver has to exclude its terminal age, where a' >= 0 stands
                # in for the limit.
                true_borrowing_limit = -borrowing_limit(kappa, iz, p)
                effective_borrowing_limit = -lower_ap
                u_by_age[age] += mass * u

                weighted_mass = pkappa * mass
                C[age] += weighted_mass * c
                H[age] += weighted_mass * h
                Y[age] += weighted_mass * y
                A[age] += weighted_mass * ap

                # Two accumulators, two age windows. `stats` covers the
                # calibration window [stats_age_lo, stats_age_hi] and is what
                # the moments are matched on; `stats_all` covers every age of
                # the forward pass. Both are fed from the same helper so they
                # cannot drift apart, and each keeps its own total_mass -- the
                # denominator every share and mean divides by, which is why the
                # window has to gate the whole block rather than parts of it.
                accumulate_stats!(stats_all, weighted_mass, ia, a, ap, h, c, y,
                                  true_borrowing_limit, effective_borrowing_limit,
                                  at_borrowing_constraint, at_asset_upper,
                                  h_upper, false, p)
                in_stats_window &&
                    accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      at_borrowing_constraint, at_asset_upper,
                                      h_upper, p.collect_distributions, p)
                at_stats_age_lo &&
                    accumulate_stats!(stats_lo, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      at_borrowing_constraint, at_asset_upper,
                                      h_upper, false, p)

                begin
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

        # Record where the cross-section settles, but do NOT stop here.
        # Breaking out corrupts the age profiles, the government budget, the
        # PV sums and the statistics accumulator at once; see NOTES.md. Every
        # kappa runs the full maxAge, and `converged_age` is kept only as a
        # diagnostic so an undersized maxAge is visible.
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

    return welfare_simulation, converged_age, final_drift
end

function solve_aggregates_for_lambda(lambda::Float64, p::HIParams)
    nAge = p.maxAge
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
                         Vector{Array{Int32,3}}(undef, nKappa) : Array{Int32,3}[]
    saved_policyA = p.store_solutions ?
                    Vector{Array{Float64,3}}(undef, nKappa) : Array{Float64,3}[]
    saved_policyH = p.store_solutions ?
                    Vector{Array{Float64,3}}(undef, nKappa) : Array{Float64,3}[]
    q_by_ap = asset_prices(p)
    terminal_first_ap = first_nonnegative_asset_index(p)
    welfare_value_function_by_kappa = Vector{Float64}(undef, length(p.kappa_grid))
    welfare_simulation_by_kappa = similar(welfare_value_function_by_kappa)
    vIters_by_kappa = zeros(Int, nKappa)
    vGap_by_kappa = fill(NaN, nKappa)
    converged_by_kappa = zeros(Int, nKappa)
    drift_by_kappa = fill(NaN, nKappa)

    # Each kappa owns local arrays and a local stats accumulator; the reduction
    # after this loop avoids races on aggregate sums and distribution vectors.
    Threads.@threads :static for ik in 1:nKappa
        kappa = p.kappa_grid[ik]
        pkappa = p.Pkappa[ik]
        first_ap = first_feasible_asset_indices(kappa, p)
        tax_base, wage_base = precompute_income_bases(kappa, p)
        policyAIndex, policyA, policyH, welfare_value_function, vIters, vGap =
            solve_policies_for_kappa(
            lambda, kappa, first_ap, terminal_first_ap, q_by_ap, tax_base, p,
        )
        C_local = C_by_kappa[ik]
        H_local = H_by_kappa[ik]
        Y_local = Y_by_kappa[ik]
        A_local = A_by_kappa[ik]
        stats_local = HIStatsAccumulator(length(p.a_grid))
        stats_all_local = HIStatsAccumulator(length(p.a_grid))
        stats_lo_local = HIStatsAccumulator(length(p.a_grid))
        welfare_simulation, converged, final_drift = simulate_kappa!(
            C_local, H_local, Y_local, A_local, stats_local, stats_all_local,
            stats_lo_local,
            policyAIndex, policyA, policyH,
            kappa, pkappa, first_ap, terminal_first_ap, q_by_ap,
            tax_base, wage_base, p, lambda,
        )
        stats_by_kappa[ik] = stats_local
        stats_all_by_kappa[ik] = stats_all_local
        stats_lo_by_kappa[ik] = stats_lo_local
        vIters_by_kappa[ik] = vIters
        vGap_by_kappa[ik] = vGap
        converged_by_kappa[ik] = converged
        drift_by_kappa[ik] = final_drift
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
    # Every kappa runs the full maxAge, so the PV tail opens at maxAge for all
    # of them. `convergedAge` is diagnostic: 0 means the cross-section had not
    # settled by maxAge, so the closed-form tail is unearned and maxAge should
    # be raised.
    # Report the ACHIEVED drift rather than a pass/fail on tolDist. Failing an
    # absolute 1e-10 test says little on its own: what the closed-form tail
    # needs is that (Y - C) has stopped moving, and a drift of 3e-10 and one of
    # 3e-4 are worlds apart while both "fail". Printing the number lets the
    # magnitude be judged. convergedAge = 0 means that kappa never got under
    # tolDist at any age.
    # The settling check is NOT emitted here. This routine runs once per lambda
    # probe, and the lambda root-finder deliberately visits degenerate corners:
    # observed warnings carried lambda = 0.01 (= lambdaMin) at qSav = 0.911 with
    # Y[end] = 0.155 against a normal 0.927. Those probes say nothing about the
    # answer and drowned the one solve that matters. The drift is stored in
    # `diagnostics.finalDrift` instead and judged once, on the FINAL
    # equilibrium, in `attach_elapsed`.
    diagnostics = (; vIters = vIters_by_kappa, vGap = vGap_by_kappa,
                   settledAge = p.maxAge,
                   convergedAge = maximum(converged_by_kappa),
                   convergedAgeByKappa = converged_by_kappa,
                   finalDrift = drift_by_kappa)
    return (; C = C, H = H, Y = Y, A = A), stats, stats_all, welfare, solutions, diagnostics
end

"""
    finalize_statistics(stats, p)

The published statistics for one accumulated group: the shared core, and
nothing else.
"""
finalize_statistics(stats::HIStatsAccumulator, p::HIParams) =
    core_statistics(stats, p)

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

