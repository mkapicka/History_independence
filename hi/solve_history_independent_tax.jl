using LinearAlgebra
using Printf
using Statistics
using FastGaussQuadrature
using QuantEcon
using Roots
using StatsBase

"""
    HIParams(; kwargs...)

Parameters for the finite-horizon Bewley model with the history-independent
tax function from `Bewley.tex`.

The `z_grid` and `Pz` objects are generated with QuantEcon.jl's Rouwenhorst
method by default for

    z' = omega_mean + rho*z + innovation,  innovation ~ N(0, sigma_omega^2),

and the iid `epsilon` and `kappa` shocks are generated with Gauss-Hermite
normal quadrature. Set `z_discretization_method = :tauchen` to use
QuantEcon.jl's Tauchen method instead.

Households solve, for ages j = 0,...,J,

    max (1-beta)*(log(c) - phi*h^(1+eta)/(1+eta)) + beta*E[V_{j+1}]

subject to

    c + q(a')*a' = lambda*exp((1-tau)*(kappa + z + epsilon))*h^(1-tau) + a,
    bbar*exp(kappa + rho*z) <= a' <= aMax.

The convention is `bbar <= 0` for borrowing. If `bbar == 0`, assets are
nonnegative. In the terminal decision period `J`, the borrowing constraint is
set to zero so agents cannot end with negative assets. `lambda` is chosen so
that

    (1-qGov)*sum_j qGov^j*(Y_j-C_j) = (1-qGov^(J+1))*G.

Set `asset_choice_method = :grid_search` to restrict `a'` to the asset grid,
or `:interpolate` to optimize continuous `a'` with linear interpolation of
the continuation value.

Every model parameter is a field with no hard-coded default, so `SETTINGS` in
`model_settings.jl` is the single source of truth. Construct with
`make_history_independent_params(; overrides...)`, or with `hi_params(; ...)`
when supplying a complete parameter set explicitly. Fields whose default is an
expression are derived from the fields above them.
"""
Base.@kwdef struct HIParams
    # Preferences and tax.
    beta::Float64
    eta::Float64
    phi::Float64
    tau::Float64

    # Ages j = 0,...,J are stored in length-(J+1) vectors at index j+1.
    J::Int

    # Shocks. The *_mean defaults normalize each shock to exp-mean one.
    rho::Float64
    sigma_omega::Float64
    sigma_epsilon::Float64
    sigma_kappa::Float64
    omega_mean::Float64 = -0.5 * sigma_omega^2
    epsilon_mean::Float64 = -0.5 * sigma_epsilon^2
    kappa_mean::Float64 = -0.5 * sigma_kappa^2
    nZ::Int
    nEps::Int
    nKappa::Int
    z_discretization_method::Symbol
    tauchen_width::Float64
    z_initial::Float64
    # ar1_grid and ar1_transition each rediscretize the AR(1). Both are O(nZ^2)
    # and run once per HIParams, so staging the shared result is not worth it.
    z_grid::Vector{Float64} =
        ar1_grid(nZ, rho, omega_mean, sigma_omega, z_discretization_method, tauchen_width)
    Pz::Matrix{Float64} =
        ar1_transition(nZ, rho, omega_mean, sigma_omega, z_discretization_method, tauchen_width)
    z0_probs::Vector{Float64} =
        ar1_initial_probabilities(z_initial, z_grid, rho, omega_mean, sigma_omega)
    eps_grid::Vector{Float64} = gauss_hermite_grid(nEps, epsilon_mean, sigma_epsilon)
    Peps::Vector{Float64} = gauss_hermite_probs(nEps, epsilon_mean, sigma_epsilon)
    kappa_grid::Vector{Float64} = gauss_hermite_grid(nKappa, kappa_mean, sigma_kappa)
    Pkappa::Vector{Float64} = gauss_hermite_probs(nKappa, kappa_mean, sigma_kappa)

    # Asset grid.
    bbar::Float64
    aMax::Float64
    nA::Int
    asset_grid_method::Symbol
    asset_grid_curvature_borrow::Float64
    asset_grid_curvature_save::Float64
    asset_grid_borrow_share::Float64
    asset_grid_zero_share::Float64
    asset_grid_zero_width::Float64
    a_grid::Vector{Float64} = default_asset_grid(
        bbar, aMax, nA, rho, kappa_grid, z_grid;
        method = asset_grid_method,
        curvature_borrow = asset_grid_curvature_borrow,
        curvature_save = asset_grid_curvature_save,
        borrow_share = asset_grid_borrow_share,
        zero_share = asset_grid_zero_share,
        zero_width = asset_grid_zero_width,
    )

    # Asset choice.
    asset_choice_method::Symbol
    asset_choice_tol::Float64
    asset_choice_max_iter::Int

    # Financial and government.
    qBorr::Float64
    qSav::Float64
    qGov::Float64
    G::Float64

    # Labor.
    hMin::Float64
    hMax::Float64
    labor_grid_size::Int
    labor_solver::Symbol
    h_grid::Vector{Float64} = build_labor_grid(hMin, hMax, labor_grid_size)
    h_grid_income_power::Vector{Float64} = h_grid .^ (1.0 - tau)
    h_grid_disutility::Vector{Float64} = phi .* (h_grid .^ (1.0 + eta)) ./ (1.0 + eta)

    # lambda solver.
    lambdaMin::Float64
    lambdaMax::Float64
    nLambdaSearch::Int
    maxIterLambda::Int = 60
    tolLambda::Float64
    tolGovBudget::Float64

    # Output.
    verbose::Bool
    printEveryLambda::Int
    massTol::Float64 = 1e-14
    store_solutions::Bool
    collect_distributions::Bool
end

"""
    hi_params(; kwargs...)

Build a validated `HIParams` from a complete set of parameters. A nonempty
`a_grid` or `h_grid` replaces the generated grid and resets `nA` /
`labor_grid_size` to match. `make_history_independent_params` supplies
`SETTINGS` for everything the caller does not override.
"""
function hi_params(; kwargs...)
    opts = NamedTuple(kwargs)

    if haskey(opts, :a_grid) && !isempty(opts.a_grid)
        grid = sort(collect(Float64.(opts.a_grid)))
        opts = merge(opts, (; a_grid = grid, nA = length(grid)))
    elseif haskey(opts, :a_grid)
        opts = Base.structdiff(opts, (; a_grid = nothing))
    end

    if haskey(opts, :h_grid) && !isempty(opts.h_grid)
        grid = normalize_labor_grid(opts.h_grid, opts.hMin, opts.hMax)
        opts = merge(opts, (; h_grid = grid, labor_grid_size = length(grid)))
    elseif haskey(opts, :h_grid)
        opts = Base.structdiff(opts, (; h_grid = nothing))
    end

    return validate(HIParams(; opts...))
end

"""
    validate(p::HIParams)

Check the parameter combinations that the individual grid builders cannot see.
Returns `p` so it can wrap a construction.
"""
function validate(p::HIParams)
    0.0 <= p.beta < 1.0 || error("beta must satisfy 0 <= beta < 1")
    p.tau < 1.0 || error("tau must be less than one for h^(1-tau)")
    p.labor_solver in (:brent, :hybrid_newton, :grid) ||
        error("labor_solver must be :brent, :hybrid_newton, or :grid")
    p.asset_choice_method in (:grid_search, :interpolate) ||
        error("asset_choice_method must be :grid_search or :interpolate")
    p.asset_choice_tol > 0.0 || error("asset_choice_tol must be positive")
    p.asset_choice_max_iter > 0 || error("asset_choice_max_iter must be positive")
    p.printEveryLambda >= 0 || error("printEveryLambda must be nonnegative")
    any(iszero, p.a_grid) ||
        error("a_grid must include 0.0 exactly for the initial condition")
    maximum(p.a_grid) <= p.aMax + 1e-12 || error("a_grid has points above aMax")
    length(p.z0_probs) == length(p.z_grid) || error("z0_probs length must match z_grid")
    length(p.Peps) == length(p.eps_grid) || error("Peps length must match eps_grid")
    length(p.Pkappa) == length(p.kappa_grid) ||
        error("Pkappa length must match kappa_grid")
    return p
end

Base.@kwdef mutable struct HIStatsAccumulator
    asset_mass::Vector{Float64}
    distribution_weights::Vector{Float64} = Float64[]
    hours_values::Vector{Float64} = Float64[]
    consumption_values::Vector{Float64} = Float64[]
    total_mass::Float64 = 0.0
    sum_current_assets::Float64 = 0.0
    sum_labor_income::Float64 = 0.0
    sum_borrowing_limit::Float64 = 0.0
    sum_effective_borrowing_limit::Float64 = 0.0
    borrowing_limit_mass::Float64 = 0.0
    negative_asset_mass::Float64 = 0.0
    zero_asset_mass::Float64 = 0.0
    borrowing_constraint_mass::Float64 = 0.0
    upper_bound_mass::Float64 = 0.0
    hours_upper_bound_mass::Float64 = 0.0
    max_next_assets::Float64 = -Inf
    max_hours::Float64 = -Inf
    max_material_next_assets::Float64 = -Inf
    max_material_hours::Float64 = -Inf
end

function HIStatsAccumulator(nA::Int)
    nA > 0 || error("nA must be positive")
    return HIStatsAccumulator(asset_mass = zeros(nA))
end

# How each field combines when per-kappa accumulators are reduced.
const STATS_APPEND_FIELDS =
    (:distribution_weights, :hours_values, :consumption_values)
const STATS_SUM_FIELDS =
    (:total_mass, :sum_current_assets, :sum_labor_income, :sum_borrowing_limit,
     :sum_effective_borrowing_limit, :borrowing_limit_mass, :negative_asset_mass,
     :zero_asset_mass, :borrowing_constraint_mass, :upper_bound_mass,
     :hours_upper_bound_mass)
const STATS_MAX_FIELDS =
    (:max_next_assets, :max_hours, :max_material_next_assets, :max_material_hours)

function merge_stats!(dest::HIStatsAccumulator, src::HIStatsAccumulator)
    length(dest.asset_mass) == length(src.asset_mass) ||
        error("Cannot merge statistics with different asset-grid sizes")

    dest.asset_mass .+= src.asset_mass
    for f in STATS_APPEND_FIELDS
        append!(getfield(dest, f), getfield(src, f))
    end
    for f in STATS_SUM_FIELDS
        setfield!(dest, f, getfield(dest, f) + getfield(src, f))
    end
    for f in STATS_MAX_FIELDS
        setfield!(dest, f, max(getfield(dest, f), getfield(src, f)))
    end
    return dest
end

"""
    solve_history_independent_tax(p::HIParams)

Solve for `lambda`, household policies, age distributions, and aggregates.
Returns `(eq, sol)`, where `eq` is a named tuple of equilibrium objects and
`sol` contains policies only when `p.store_solutions = true`. Build `p` with
`make_history_independent_params()`.
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
    best = Ref{Any}(nothing)   # (; lambda, residual, eq, sol)

    function solve_at_lambda(lambda::Float64)
        residual, eq, sol = government_residual_at_lambda(lambda, p)
        eval_cache[lambda] = residual
        incumbent = best[]
        if isfinite(residual) &&
           (incumbent === nothing || abs(residual) < abs(incumbent.residual))
            best[] = (; lambda = lambda, residual = residual, eq = eq, sol = sol)
        end
        return residual, eq, sol
    end

    function evaluate_lambda_residual(lambda::Float64)
        haskey(eval_cache, lambda) && return eval_cache[lambda]
        return first(solve_at_lambda(lambda))
    end

    function evaluate_lambda_full(lambda::Float64)
        incumbent = best[]
        incumbent !== nothing && incumbent.lambda == lambda &&
            return incumbent.residual, incumbent.eq, incumbent.sol
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
            finite_idx = findall(isfinite, residuals)
            if isempty(finite_idx)
                error("Could not evaluate any finite government residual")
            end
            eq = attach_elapsed(best[].eq, start_time, p;
                                converged = false,
                                bracketWarning = true)
            return eq, best[].sol
        end
        i_low, i_high = bracket
        lambda_low, lambda_high = grid[i_low], grid[i_high]
        r_low, r_high = residuals[i_low], residuals[i_high]
        if p.verbose
            @printf("Using lambda bracket [%.8f, %.8f].\n", lambda_low, lambda_high)
            flush(stdout)
        end
    end

    if r_low == 0.0 || lambda_low == lambda_high
        _, eq_low, sol_low = evaluate_lambda_full(Float64(lambda_low))
        eq = attach_elapsed(eq_low, start_time, p;
                            converged = isfinite(r_low) && abs(r_low) <= p.tolGovBudget,
                            bracketWarning = false)
        return eq, sol_low
    elseif r_high == 0.0
        _, eq_high, sol_high = evaluate_lambda_full(Float64(lambda_high))
        eq = attach_elapsed(eq_high, start_time, p;
                            converged = isfinite(r_high) && abs(r_high) <= p.tolGovBudget,
                            bracketWarning = false)
        return eq, sol_high
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
        r_root, eq_root, sol_root = evaluate_lambda_full(Float64(lambda_root))
        if p.verbose
            @printf("lambda root: lambda=%.8f, residual=%.8e\n", lambda_root, r_root)
            flush(stdout)
        end
        converged = isfinite(r_root) && abs(r_root) <= p.tolGovBudget
        eq = attach_elapsed(eq_root, start_time, p;
                            converged = converged,
                            bracketWarning = false,
                            rootResidualWarning = !converged)
        return eq, sol_root
    catch err
        if best[] === nothing
            rethrow(err)
        end
        eq = attach_elapsed(best[].eq, start_time, p;
                            converged = false,
                            bracketWarning = false,
                            rootSolverWarning = true,
                            rootSolverError = sprint(showerror, err))
        return eq, best[].sol
    end
end

function print_solver_options(p::HIParams)
    println("Options:")
    @printf("  age dimension J             = %d\n", p.J)
    @printf("  shock grid dimension nZ     = %d\n", length(p.z_grid))
    @printf("  shock grid dimension nEps   = %d\n", length(p.eps_grid))
    @printf("  shock grid dimension nKappa = %d\n", length(p.kappa_grid))
    @printf("  z_discretization_method     = :%s  (alternatives: :rouwenhorst, :tauchen)\n",
            String(p.z_discretization_method))
    @printf("  tauchen_width               = %.3f  (used when z_discretization_method = :tauchen)\n",
            p.tauchen_width)
    @printf("  asset grid dimension nA     = %d\n", length(p.a_grid))
    @printf("  asset_grid_method           = :%s  (alternatives: :nonuniform, :linear)\n",
            String(p.asset_grid_method))
    if p.asset_grid_method == :nonuniform
        @printf("  asset_grid_borrow_share     = %.3f\n", p.asset_grid_borrow_share)
        @printf("  asset_grid_curvatures       = borrow %.3f, save %.3f\n",
                p.asset_grid_curvature_borrow, p.asset_grid_curvature_save)
        @printf("  asset_grid_zero_band        = share %.3f, width %.3f\n",
                p.asset_grid_zero_share, p.asset_grid_zero_width)
    end
    @printf("  asset_grid_bounds           = [%.6f, %.6f], bbar = %.6f\n",
            minimum(p.a_grid), maximum(p.a_grid), p.bbar)
    @printf("  asset_choice_method         = :%s  (alternatives: :grid_search, :interpolate)\n",
            String(p.asset_choice_method))
    if p.asset_choice_method == :interpolate
        @printf("  asset_choice_optimizer      = golden search with linear continuation interpolation, tol = %.2e, max_iter = %d\n",
                p.asset_choice_tol, p.asset_choice_max_iter)
    end
    @printf("  labor_solver                = :%s  (alternatives: :brent, :hybrid_newton, :grid)\n",
            String(p.labor_solver))
    @printf("  labor_bounds                = [%.2e, %.4f]\n", p.hMin, p.hMax)
    if p.labor_solver == :grid
        @printf("  labor_grid_size             = %d\n", length(p.h_grid))
    end
    @printf("  terminal_borrowing          = :zero\n")
    @printf("  lambda_solver               = :brent  (Roots.jl; fallback: grid search over %d values)\n",
            p.nLambdaSearch)
    @printf("  lambda_bracket              = [%.6f, %.6f], tol = %.2e\n",
            p.lambdaMin, p.lambdaMax, p.tolGovBudget)
    @printf("  collect_distributions       = %s\n", string(p.collect_distributions))
    println()
end

"""
    print_equilibrium_summary(eq, p; title, show_statistics, show_welfare)

Print the equilibrium, aggregate statistics, and welfare decomposition. Binding
upper bounds are reported by the solver itself (see `print_upper_bound_warning`,
called from `attach_elapsed`), so they are not repeated here.
"""
function print_equilibrium_summary(eq, p::HIParams;
                                   title = "Final history-independent equilibrium",
                                   show_statistics::Bool = true,
                                   show_welfare::Bool = true)
    @printf("\n=== %s ===\n", title)
    @printf("lambda                     = %.8f\n", eq.lambda)
    @printf("government budget residual = %.8e\n", eq.govBudgetResidual)
    @printf("PV output                  = %.8f\n", eq.outputPV)
    @printf("PV consumption             = %.8f\n", eq.consumptionPV)
    @printf("mean output                = %.8f\n", mean(eq.Y))
    @printf("mean consumption           = %.8f\n", mean(eq.C))
    @printf("terminal assets            = %.8f\n", eq.A[end])
    if hasproperty(eq, :elapsedSeconds)
        @printf("solve time                 = %.3f seconds\n", eq.elapsedSeconds)
    end

    if show_statistics && hasproperty(eq, :statistics)
        print_aggregate_statistics(eq.statistics, p)
    end
    if show_welfare && hasproperty(eq, :welfare)
        print_welfare_summary(eq.welfare)
    end
    return nothing
end

function print_aggregate_statistics(s, p::HIParams)
    on_grid = p.asset_choice_method == :grid_search
    limit_label = on_grid ? "grid borrowing limit / mean labor income" :
                            "choice borrowing limit / mean labor income"
    bound_label = on_grid ? "share at effective grid borrowing bound" :
                            "share at borrowing bound"

    @printf("\n=== Aggregate statistics ===\n")
    @printf("mean assets / mean labor income          = %.8f\n",
            s.meanAssetsToMeanLaborIncome)
    @printf("median assets / mean labor income        = %.8f\n",
            s.medianAssetsToMeanLaborIncome)
    @printf("true borrowing limit / mean labor income = %.8f\n",
            s.meanBorrowingLimitToMeanLaborIncome)
    @printf("%-40s = %.8f\n", limit_label,
            s.meanEffectiveGridBorrowingLimitToMeanLaborIncome)
    @printf("share negative liquid assets             = %.8f\n",
            s.shareNegativeLiquidAssets)
    @printf("%-40s = %.8f\n", bound_label, s.shareAtEffectiveBorrowingConstraint)
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

function attach_elapsed(eq, start_time::Float64, p::HIParams; kwargs...)
    elapsed = time() - start_time
    eq_with_elapsed = merge(eq, (; kwargs..., elapsedSeconds = elapsed))
    if p.verbose
        print_lambda_warnings(eq_with_elapsed)
        print_upper_bound_warning(eq_with_elapsed.statistics)
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
        if hasproperty(eq, :rootSolverError)
            @printf("  solver error: %s\n", eq.rootSolverError)
        end
    end
    flush(stdout)
end

function print_upper_bound_warning(s)
    s.upperBoundsBinding || return nothing

    println("WARNING: upper bound is binding.")
    if s.assetUpperBoundBinding
        @printf("  asset upper bound       = BINDING (share = %.8e, bound = %.8f, material max a' = %.8f, slack = %.8e)\n",
                s.shareAtAssetUpperBound, s.assetUpperBound,
                s.maxMaterialNextAssets, s.assetUpperBoundSlack)
    end
    if s.hoursUpperBoundBinding
        @printf("  hours upper bound       = BINDING (share = %.8e, bound = %.8f, material max h = %.8f, slack = %.8e)\n",
                s.shareAtHoursUpperBound, s.hoursUpperBound,
                s.maxMaterialHours, s.hoursUpperBoundSlack)
    end
    flush(stdout)
end

function government_residual_at_lambda(lambda::Float64, p::HIParams)
    aggs, stats, welfare, sol = solve_aggregates_for_lambda(lambda, p)
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
        parameters = p,
    )
    return residual, eq, sol
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

    welfare_value_function = expected_initial_value(Vnext, p)
    return policyAIndex, policyA, policyH, welfare_value_function
end

function expected_initial_value(V0, p::HIParams)
    ia0 = nearest_index(p.a_grid, 0.0)
    expected_value = 0.0
    @inbounds for iz in eachindex(p.z_grid), ie in eachindex(p.eps_grid)
        expected_value += p.z0_probs[iz] * p.Peps[ie] * V0[ia0, iz, ie]
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

function solve_policy_age_interpolated!(Vcur, policyAIndex, policyA, policyH,
                                        EV, age::Int, lambda::Float64,
                                        kappa::Float64,
                                        tax_base::Matrix{Float64}, p::HIParams,
                                        util_weight::Float64, beta::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)

    @inbounds for ia in 1:nA
        a = p.a_grid[ia]
        for iz in 1:nZ
            lower = asset_choice_lower_bound(age, kappa, iz, p)
            for ie in 1:nE
                income_coeff = lambda * tax_base[iz, ie]
                best_val, best_ap, best_iap, best_h = interpolated_asset_choice(
                    a, lower, income_coeff, EV, iz, p, util_weight, beta,
                )
                Vcur[ia, iz, ie] = best_val
                policyAIndex[ia, iz, ie, age] = Int32(best_iap)
                policyA[ia, iz, ie, age] = best_ap
                policyH[ia, iz, ie, age] = best_h
            end
        end
    end
    return nothing
end

function interpolated_asset_choice(a::Float64, lower::Float64,
                                   income_coeff::Float64, EV, iz::Int,
                                   p::HIParams, util_weight::Float64,
                                   beta::Float64)
    upper = asset_upper_bound(p)
    lower = min(max(lower, p.a_grid[1]), upper)

    best_val, best_h = interpolated_choice_value(
        lower, a, income_coeff, EV, iz, p, util_weight, beta,
    )
    best_ap = lower

    val_upper, h_upper = interpolated_choice_value(
        upper, a, income_coeff, EV, iz, p, util_weight, beta,
    )
    if val_upper > best_val
        best_val = val_upper
        best_ap = upper
        best_h = h_upper
    end

    if lower < 0.0 < upper
        val_zero, h_zero = interpolated_choice_value(
            0.0, a, income_coeff, EV, iz, p, util_weight, beta,
        )
        if val_zero > best_val
            best_val = val_zero
            best_ap = 0.0
            best_h = h_zero
        end
    end

    if lower < 0.0
        segment_hi = min(0.0, upper)
        best_val, best_ap, best_h = update_with_asset_segment_max(
            best_val, best_ap, best_h, lower, segment_hi,
            a, income_coeff, EV, iz, p, util_weight, beta,
        )
    end

    if upper > 0.0
        segment_lo = max(0.0, lower)
        best_val, best_ap, best_h = update_with_asset_segment_max(
            best_val, best_ap, best_h, segment_lo, upper,
            a, income_coeff, EV, iz, p, util_weight, beta,
        )
    end

    return best_val, best_ap, nearest_asset_index(best_ap, p), best_h
end

function update_with_asset_segment_max(best_val::Float64, best_ap::Float64,
                                       best_h::Float64, lo::Float64, hi::Float64,
                                       a::Float64, income_coeff::Float64, EV,
                                       iz::Int, p::HIParams,
                                       util_weight::Float64, beta::Float64)
    if hi - lo <= p.asset_choice_tol * max(1.0, abs(hi))
        return best_val, best_ap, best_h
    end

    ap, val, h = maximize_asset_segment(
        lo, hi, a, income_coeff, EV, iz, p, util_weight, beta,
    )
    if val > best_val
        return val, ap, h
    end
    return best_val, best_ap, best_h
end

function maximize_asset_segment(lo::Float64, hi::Float64, a::Float64,
                                income_coeff::Float64, EV, iz::Int,
                                p::HIParams, util_weight::Float64,
                                beta::Float64)
    invphi = (sqrt(5.0) - 1.0) / 2.0
    c = hi - invphi * (hi - lo)
    d = lo + invphi * (hi - lo)
    vc, hc = interpolated_choice_value(c, a, income_coeff, EV, iz, p, util_weight, beta)
    vd, hd = interpolated_choice_value(d, a, income_coeff, EV, iz, p, util_weight, beta)

    for _ in 1:p.asset_choice_max_iter
        if hi - lo <= p.asset_choice_tol * max(1.0, abs(0.5 * (lo + hi)))
            break
        end

        if vc < vd
            lo = c
            c = d
            vc = vd
            hc = hd
            d = lo + invphi * (hi - lo)
            vd, hd = interpolated_choice_value(
                d, a, income_coeff, EV, iz, p, util_weight, beta,
            )
        else
            hi = d
            d = c
            vd = vc
            hd = hc
            c = hi - invphi * (hi - lo)
            vc, hc = interpolated_choice_value(
                c, a, income_coeff, EV, iz, p, util_weight, beta,
            )
        end
    end

    if vc >= vd
        return c, vc, hc
    end
    return d, vd, hd
end

function interpolated_choice_value(ap::Float64, a::Float64,
                                   income_coeff::Float64, EV, iz::Int,
                                   p::HIParams, util_weight::Float64,
                                   beta::Float64)
    cash = a - asset_price(ap, p) * ap
    u, h = optimal_labor_foc(cash, income_coeff, p)
    if !isfinite(u)
        return -Inf, p.hMin
    end
    continuation = interpolate_asset_value(ap, p.a_grid, EV, iz)
    return util_weight * u + beta * continuation, h
end

function interpolate_asset_value(ap::Float64, a_grid::Vector{Float64}, values, iz::Int)
    nA = length(a_grid)
    if ap <= a_grid[1]
        return values[1, iz]
    elseif ap >= a_grid[nA]
        return values[nA, iz]
    end

    hi = searchsortedfirst(a_grid, ap)
    if hi <= nA && a_grid[hi] == ap
        return values[hi, iz]
    end
    lo = hi - 1
    weight_hi = (ap - a_grid[lo]) / (a_grid[hi] - a_grid[lo])
    return (1.0 - weight_hi) * values[lo, iz] + weight_hi * values[hi, iz]
end

function precompute_flow_payoffs(lambda::Float64, first_ap::Vector{Int},
                                 q_by_ap::Vector{Float64},
                                 tax_base::Matrix{Float64}, p::HIParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    flow_u = fill(-Inf, nA, nA, nZ, nE)
    flow_h = Array{Float64}(undef, nA, nA, nZ, nE)
    cash = Array{Float64}(undef, nA, nA)

    @inbounds for ia in 1:nA, iap in 1:nA
        cash[iap, ia] = p.a_grid[ia] - q_by_ap[iap] * p.a_grid[iap]
    end

    @inbounds for ia in 1:nA
        for iz in 1:nZ
            ia_first = first_ap[iz]
            for ie in 1:nE
                income_coeff = lambda * tax_base[iz, ie]
                for iap in ia_first:nA
                    cash_iap_ia = cash[iap, ia]

                    u, h = optimal_labor_foc(cash_iap_ia, income_coeff, p)
                    if isfinite(u)
                        flow_u[iap, ia, iz, ie] = u
                        flow_h[iap, ia, iz, ie] = h
                    end
                end
            end
        end
    end

    return flow_u, flow_h
end

function simulate_kappa!(C, H, Y, A, stats::HIStatsAccumulator,
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
    ia0 = nearest_index(p.a_grid, 0.0)
    h_upper = hours_upper_bound(p)
    welfare_simulation = 0.0

    @inbounds for iz in 1:nZ, ie in 1:nE
        dist[ia0, iz, ie] = p.z0_probs[iz] * p.Peps[ie]
    end

    @inbounds for age in 1:nAge
        fill!(dist_next, 0.0)
        utility_weight = (1.0 - p.beta) * p.beta^(age - 1)

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
                                       -p.bbar * exp(kappa + p.rho * p.z_grid[iz]) : 0.0
                effective_borrowing_limit = binding_age ? -lower_ap : 0.0
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

    C_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    H_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    Y_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    A_by_kappa = [zeros(nAge) for _ in 1:nKappa]
    stats_by_kappa = Vector{HIStatsAccumulator}(undef, nKappa)
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
        welfare_simulation = simulate_kappa!(
            C_local, H_local, Y_local, A_local, stats_local,
            policyAIndex, policyA, policyH,
            kappa, pkappa, first_ap, terminal_first_ap, q_by_ap,
            tax_base, wage_base, p, lambda,
        )
        stats_by_kappa[ik] = stats_local
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
    end

    sol = p.store_solutions ? (;
        lambda = lambda,
        policyAIndex_by_kappa = saved_policyAIndex,
        policyA_by_kappa = saved_policyA,
        policyH_by_kappa = saved_policyH,
        a_grid = p.a_grid,
        z_grid = p.z_grid,
        eps_grid = p.eps_grid,
        kappa_grid = p.kappa_grid,
    ) : (; lambda = lambda)

    stats = finalize_statistics(stats_acc, p)
    welfare = finalize_welfare(
        welfare_value_function_by_kappa, welfare_simulation_by_kappa, p,
    )
    return (; C = C, H = H, Y = Y, A = A), stats, welfare, sol
end

function finalize_welfare(value_function_by_kappa::Vector{Float64},
                          simulation_by_kappa::Vector{Float64},
                          p::HIParams)
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

function finalize_statistics(stats::HIStatsAccumulator, p::HIParams)
    total_mass = stats.total_mass
    mean_assets = stats.sum_current_assets / total_mass
    mean_labor_income = stats.sum_labor_income / total_mass
    median_assets = StatsBase.median(p.a_grid, StatsBase.weights(stats.asset_mass))
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
        meanBorrowingLimitToMeanLaborIncome = safe_ratio(mean_borrowing_limit, mean_labor_income),
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

upper_bound_share_tol() = 1e-8
upper_bound_level_tol(bound::Real) = 1e-8 * max(1.0, abs(Float64(bound)))
asset_upper_bound(p::HIParams) = maximum(p.a_grid)
hours_upper_bound(p::HIParams) = p.hMax

safe_ratio(num::Real, den::Real) = abs(den) > eps(Float64) ? Float64(num) / Float64(den) : NaN

function compute_expected_value!(EV, Vnext, p::HIParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    fill!(EV, 0.0)

    @inbounds for ia in 1:nA
        for iz in 1:nZ
            total = 0.0
            for izp in 1:nZ
                pe_z = p.Pz[iz, izp]
                if pe_z == 0.0
                    continue
                end
                eps_total = 0.0
                for iep in 1:nE
                    eps_total += p.Peps[iep] * Vnext[ia, izp, iep]
                end
                total += pe_z * eps_total
            end
            EV[ia, iz] = total
        end
    end
    return EV
end

function optimal_labor_foc(cash::Float64, income_coeff::Float64, p::HIParams)
    tau = p.tau
    h_low = p.hMin
    h_high = p.hMax

    if income_coeff <= 0.0
        if cash <= 0.0
            return -Inf, NaN
        end
        h = h_low
        c = cash
        return log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta), h
    end

    if cash + income_coeff * h_high^(1.0 - tau) <= 0.0
        return -Inf, NaN
    end

    if cash + income_coeff * h_low^(1.0 - tau) <= 0.0
        h_low = ((-cash / income_coeff) * (1.0 + 1e-12))^(1.0 / (1.0 - tau))
        h_low = min(max(h_low, p.hMin), h_high)
        if cash + income_coeff * h_low^(1.0 - tau) <= 0.0
            h_low = nextfloat(h_low)
        end
    end

    if p.labor_solver == :grid
        return optimal_labor_grid(h_low, h_high, cash, income_coeff, p)
    end

    d_low = labor_foc_residual(h_low, cash, income_coeff, p)
    d_high = labor_foc_residual(h_high, cash, income_coeff, p)

    if d_low <= 0.0
        h = h_low
    elseif d_high >= 0.0
        h = h_high
    else
        h = solve_labor_root(h_low, h_high, cash, income_coeff, p)
    end

    c = cash + income_coeff * h^(1.0 - tau)
    if c <= 0.0
        return -Inf, NaN
    end
    u = log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta)
    return u, h
end

function solve_labor_root(h_low::Float64, h_high::Float64, cash::Float64,
                          income_coeff::Float64, p::HIParams)
    if p.labor_solver == :brent
        f(h) = labor_foc_residual(h, cash, income_coeff, p)
        return Roots.find_zero(f, (h_low, h_high), Roots.Brent())
    elseif p.labor_solver == :hybrid_newton
        return labor_root_hybrid_newton(h_low, h_high, cash, income_coeff, p)
    end
    error("Unknown labor_solver = $(p.labor_solver)")
end

function optimal_labor_grid(h_low::Float64, h_high::Float64, cash::Float64,
                            income_coeff::Float64, p::HIParams)
    first_h = searchsortedfirst(p.h_grid, h_low - 1e-12)
    best_u = -Inf
    best_h = NaN

    @inbounds for ih in first_h:length(p.h_grid)
        h = p.h_grid[ih]
        if h > h_high + 1e-12
            break
        end

        c = cash + income_coeff * p.h_grid_income_power[ih]
        if c <= 0.0
            continue
        end

        u = log(c) - p.h_grid_disutility[ih]
        if u > best_u
            best_u = u
            best_h = h
        end
    end

    if !isfinite(best_u)
        return -Inf, NaN
    end
    return best_u, best_h
end

function labor_root_hybrid_newton(h_low::Float64, h_high::Float64, cash::Float64,
                                  income_coeff::Float64, p::HIParams)
    lo = h_low
    hi = h_high
    h = 0.5 * (lo + hi)

    for _ in 1:50
        f = labor_foc_residual(h, cash, income_coeff, p)
        if abs(f) <= 1e-12
            return h
        end

        if f > 0.0
            lo = h
        else
            hi = h
        end

        fp = labor_foc_residual_derivative(h, cash, income_coeff, p)
        h_newton = h - f / fp
        if isfinite(h_newton) && lo < h_newton < hi
            h = h_newton
        else
            h = 0.5 * (lo + hi)
        end

        if hi - lo <= 1e-12 * max(1.0, abs(h))
            return 0.5 * (lo + hi)
        end
    end

    return 0.5 * (lo + hi)
end

function labor_foc_residual(h::Float64, cash::Float64, income_coeff::Float64, p::HIParams)
    c = cash + income_coeff * h^(1.0 - p.tau)
    if c <= 0.0
        return Inf
    end
    return income_coeff * (1.0 - p.tau) - p.phi * h^(p.eta + p.tau) * c
end

function labor_foc_residual_derivative(h::Float64, cash::Float64,
                                       income_coeff::Float64, p::HIParams)
    c = cash + income_coeff * h^(1.0 - p.tau)
    if c <= 0.0
        return -Inf
    end
    return -p.phi * (
        (p.eta + p.tau) * h^(p.eta + p.tau - 1.0) * c +
        income_coeff * (1.0 - p.tau) * h^(p.eta)
    )
end

function first_feasible_asset_indices(kappa::Float64, p::HIParams)
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

function first_nonnegative_asset_index(p::HIParams)
    idx = searchsortedfirst(p.a_grid, -1e-12)
    while idx <= length(p.a_grid) && p.a_grid[idx] < -1e-12
        idx += 1
    end
    idx <= length(p.a_grid) || error("a_grid must contain a nonnegative asset point")
    return idx
end

function build_labor_grid(hMin::Float64, hMax::Float64, labor_grid_size::Int)
    hMin > 0.0 || error("hMin must be positive")
    hMax > hMin || error("hMax must exceed hMin")
    labor_grid_size >= 2 || error("labor_grid_size must be at least 2")
    return collect(range(hMin, hMax, length = labor_grid_size))
end

function normalize_labor_grid(h_grid, hMin::Float64, hMax::Float64)
    grid = sort(unique(collect(Float64.(h_grid))))
    all(h -> hMin - 1e-12 <= h <= hMax + 1e-12, grid) ||
        error("h_grid entries must lie inside [hMin, hMax]")
    return sort(unique(vcat(hMin, grid, hMax)))
end

asset_price(ap::Real, p::HIParams) = ap < 0.0 ? p.qBorr : p.qSav
asset_prices(p::HIParams) = [asset_price(ap, p) for ap in p.a_grid]

function asset_choice_lower_bound(age::Int, kappa::Float64, iz::Int, p::HIParams)
    if age == p.J + 1
        return 0.0
    end
    return p.bbar * exp(kappa + p.rho * p.z_grid[iz])
end

asset_choice_bound_tol(bound::Real, p::HIParams) =
    max(1e-10, p.asset_choice_tol * max(1.0, abs(Float64(bound))))

function asset_transition_weights(ap::Float64, p::HIParams)
    grid = p.a_grid
    nA = length(grid)

    if ap <= grid[1]
        return 1, 1, 0.0
    elseif ap >= grid[nA]
        return nA, nA, 0.0
    end

    hi = searchsortedfirst(grid, ap)
    if hi <= nA && abs(ap - grid[hi]) <= asset_choice_bound_tol(grid[hi], p)
        return hi, hi, 0.0
    end

    lo = hi - 1
    weight_hi = (ap - grid[lo]) / (grid[hi] - grid[lo])
    return lo, hi, weight_hi
end

function nearest_asset_index(ap::Float64, p::HIParams)
    lo, hi, weight_hi = asset_transition_weights(ap, p)
    return weight_hi <= 0.5 ? lo : hi
end

function precompute_income_bases(kappa::Float64, p::HIParams)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    tax_base = Matrix{Float64}(undef, nZ, nE)
    wage_base = Matrix{Float64}(undef, nZ, nE)
    @inbounds for iz in 1:nZ, ie in 1:nE
        log_wage = kappa + p.z_grid[iz] + p.eps_grid[ie]
        tax_base[iz, ie] = exp((1.0 - p.tau) * log_wage)
        wage_base[iz, ie] = exp(log_wage)
    end
    return tax_base, wage_base
end

ar1_grid(n::Int, rho::Float64, innovation_mean::Float64, innovation_sd::Float64,
         method::Symbol, tauchen_width::Float64) =
    first(quantecon_ar1(n, rho, innovation_mean, innovation_sd;
                        method = method, width = tauchen_width))

function ar1_transition(n::Int, rho::Float64, innovation_mean::Float64,
                        innovation_sd::Float64, method::Symbol,
                        tauchen_width::Float64)
    P = last(quantecon_ar1(n, rho, innovation_mean, innovation_sd;
                           method = method, width = tauchen_width))
    validate_transition(P, size(P, 1))
    return P
end

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
        grid = [unconditional_mean]
        return grid, ones(1, 1)
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

"""
    default_asset_grid(bbar, aMax, nA, rho, kappa_grid, z_grid; grid options...)

Asset grid spanning the loosest borrowing limit `min_{kappa,z} bbar*exp(kappa +
rho*z)` up to `aMax`.
"""
function default_asset_grid(bbar::Float64, aMax::Float64, nA::Int, rho::Float64,
                            kappa_grid::Vector{Float64}, z_grid::Vector{Float64};
                            kwargs...)
    bbar <= 0.0 ||
        error("Use bbar <= 0. For a borrowing limit B > 0, pass bbar = -B.")
    amin = minimum(bbar * exp(kappa + rho * z) for kappa in kappa_grid for z in z_grid)
    return asset_grid_with_zero(amin, aMax, nA; kwargs...)
end

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
