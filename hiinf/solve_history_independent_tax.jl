# =============================================================================
# solve_history_independent_tax.jl
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

"""
    VINFEASIBLE

Value assigned to a state with no feasible choice. FINITE on purpose.

The finite-horizon solver uses `-Inf` here, which is harmless when the value
function is only ever swept backwards. Under value-function iteration it is not:
with `-Inf` the comparison `val > best_val` is `-Inf > -Inf`, which is false, so
no choice is ever selected and the maximizer silently returns its
initialization -- `best_iap = ia_first`, the LOOSEST borrowing limit at that z.
The forward pass then puts mass on an asset the next low-z draw cannot service,
and consumption goes negative.

A finite sentinel makes every comparison well defined, so an infeasible state
takes the least-bad option instead of an arbitrary index, and `-1e18` dominates
any attainable utility so such states are never chosen when anything else is
available. This is the same device `hd` uses (there to avoid `0 * Inf = NaN` in
the s-interpolation), which is why the infinite-horizon hd solver never hit
this.
"""
const VINFEASIBLE = -1.0e18

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

    # INFINITE HORIZON: no J. The agent's problem is stationary in (a, z, eps),
    # so V and the policies carry no age index -- that is the whole difference
    # from the hi solver. `maxAge` sets the length of the FORWARD pass, which
    # still runs age by age from the birth condition because aggregates vary
    # over the life cycle and the government budget is a present value over
    # them. Every kappa runs all maxAge ages; the PV tail past maxAge is closed
    # in the form, and `tolDist` only reports whether the path had settled by
    # then (see the convergedAge diagnostic).
    maxAge::Int                      # length of the forward pass
    tolV::Float64                    # sup-norm tolerance on the value function
    maxIterV::Int                    # cap on maximizing sweeps
    howardSteps::Int                 # policy-evaluation sweeps between maximizations
    tolDist::Float64                 # drift below this => report path settled
    # The WARNING threshold, relative to Y. tolDist is an absolute bound on a
    # sum of two aggregate differences, which is uninterpretable without the
    # scale: a drift of 1.5e-09 against Y = 0.927 fails an absolute 1e-10 test
    # while being nine orders below the quantity it measures, and lambda and W
    # were shown invariant to 15 digits across a doubled maxAge in exactly that
    # case. A genuinely unsettled path (maxAge = 40) instead sits at 1e-04 to
    # 1e-03 of Y, so a relative threshold separates the two cleanly. Set to Inf
    # to silence the warning entirely; tolDist still drives convergedAge.
    tolDriftRel::Float64             # warn when drift/|Y[end]| exceeds this
    # Initial asset holdings at model age 0. a0 = 0.0 is the original condition
    # (everyone born with nothing) and remains the default, so results are
    # unchanged unless it is set. a0_scales_with_kappa multiplies a0 by
    # exp(kappa), matching how wages and the borrowing limit
    # (-bbar*exp(kappa + rho*z)) already scale with the permanent type: a flat
    # a0 would otherwise leave the lowest-kappa household starting relatively
    # far richer than the highest. a0 is placed on the grid by the same Young
    # lottery used for a', not snapped to the nearest node, so it stays exact
    # between grid points.
    a0::Float64                      # initial assets (level, or scale if below)
    a0_scales_with_kappa::Bool       # use a0 * exp(kappa) instead of a0
    # MODEL AGE is the 1-based period index: model age 1 is the first simulated
    # period, at real age `age0_real`. Real age = age0_real + model age - 1.
    # The model is born AT age0_real, so nothing before it is simulated.
    #
    # Cross-sectional statistics are averaged over model ages stats_age_lo to
    # stats_age_hi inclusive, not over the whole forward pass. The calibration
    # target is Kaplan-Violante (2014) Table 2: mean net liquid wealth over mean
    # earnings-plus-benefits, 31,001/52,745 = 0.588, computed on a 2001 SCF
    # cross-section of households aged 22-59 with the top 5% by net worth
    # dropped. With age0_real = 22 that window is model ages 1-38.
    # Averaging over all maxAge ages instead made the statistic depend on
    # maxAge (0.32034 at 600 against 0.32141 at 1200) and mixed in ages the
    # data moment excludes.
    age0_real::Int                   # real age at model age 1 (birth)
    stats_age_lo::Int                # first MODEL age in the statistics window
    stats_age_hi::Int                # last  MODEL age in the statistics window

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

    # ---------------------------------------------------------------------
    # LAMBDA SOLVER
    # ---------------------------------------------------------------------.
    lambdaMin::Float64
    lambdaMax::Float64
    nLambdaSearch::Int
    maxIterLambda::Int
    tolLambda::Float64
    tolGovBudget::Float64

    # Output.
    verbose::Bool
    printEveryLambda::Int
    massTol::Float64
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
    p.maxAge >= 2 || error("maxAge must be at least 2")
    p.tolV > 0.0 || error("tolV must be positive")
    p.maxIterV >= 1 || error("maxIterV must be at least 1")
    p.howardSteps >= 0 || error("howardSteps must be nonnegative (0 = plain VFI)")
    p.tolDist > 0.0 || error("tolDist must be positive")
    p.tolDriftRel > 0.0 || error("tolDriftRel must be positive (Inf silences the warning)")
    1 <= p.stats_age_lo <= p.stats_age_hi ||
        error("need 1 <= stats_age_lo <= stats_age_hi, got $(p.stats_age_lo), $(p.stats_age_hi)")
    p.stats_age_hi <= p.maxAge ||
        error("stats_age_hi = $(p.stats_age_hi) exceeds the last model age " *
              "$(p.maxAge); widen maxAge or narrow the statistics window")
    p.asset_choice_max_iter > 0 || error("asset_choice_max_iter must be positive")
    p.printEveryLambda >= 0 || error("printEveryLambda must be nonnegative")
    # The original requirement was that a_grid contain 0.0 exactly, because the
    # initial condition was hard-coded to zero. What actually matters is that
    # the initial holding lies inside the grid and is feasible for every type.
    if p.a0 == 0.0 && !p.a0_scales_with_kappa
        any(iszero, p.a_grid) ||
            error("a_grid must include 0.0 exactly when a0 = 0")
    end
    for kappa in p.kappa_grid
        a0k = p.a0_scales_with_kappa ? p.a0 * exp(kappa) : p.a0
        first(p.a_grid) - 1e-12 <= a0k <= last(p.a_grid) + 1e-12 ||
            error("initial assets a0 = $a0k (kappa = $kappa) fall outside " *
                  "a_grid [$(first(p.a_grid)), $(last(p.a_grid))]")
        for z in p.z_grid
            # bbar is NEGATIVE, so this IS the lower bound on assets; the
            # positive -bbar*exp(.) reported as trueBorrowingLimit is its
            # magnitude, not the bound.
            limit = p.bbar * exp(kappa + p.rho * z)
            a0k >= limit - 1e-12 ||
                error("initial assets a0 = $a0k (kappa = $kappa, z = $z) are " *
                      "below the borrowing limit $limit")
        end
    end
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
const STATS_MAX_FIELDS = (:max_material_next_assets, :max_material_hours)

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
Returns `eq`, a named tuple of equilibrium objects. Household policies are
attached as `eq.solutions` when `p.store_solutions = true` (otherwise
`eq.solutions === nothing`). Build `p` with `make_history_independent_params()`.
"""
function solve_history_independent_tax(p::HIParams)
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

function print_solver_options(p::HIParams)
    println("Options:")
    @printf("  horizon                     = infinite (VFI fixed point)\n")
    @printf("  maxAge (forward pass length)= %d\n", p.maxAge)
    @printf("  statistics window (model age)= %d-%d (real age %d-%d)\n",
            p.stats_age_lo, p.stats_age_hi,
            p.age0_real + p.stats_age_lo - 1, p.age0_real + p.stats_age_hi - 1)
    @printf("  tolV, maxIterV, howardSteps = %.1e, %d, %d\n",
            p.tolV, p.maxIterV, p.howardSteps)
    @printf("  tolDist (settling report)   = %.1e\n", p.tolDist)
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
    @printf("  asset_grid_bounds           = [%.6f, %.6f]\n",
            minimum(p.a_grid), maximum(p.a_grid))
    # Calibrated inputs print with every digit (shortest representation that
    # round-trips to the same Float64) so they can be copied back verbatim.
    @printf("  bbar                        = %s\n", p.bbar)
    @printf("  qSav                        = %-20s  (price of saving,    a' >= 0)\n", p.qSav)
    @printf("  qBorr                       = %-20s  (price of borrowing, a' < 0)\n", p.qBorr)
    @printf("  qGov                        = %-20.6f  (government discount price)\n", p.qGov)
    @printf("  G                           = %-20.6f  (government spending)\n", p.G)
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
    # There is no terminal age here, so A[end] is the SETTLED level the profile
    # is carried forward at, not a terminal condition. Before the profiles were
    # padded this printed 0.0 -- the untouched tail of `zeros(maxAge)`.
    @printf("settled assets             = %.8f\n", eq.A[end])
    # Printed with every digit (shortest representation that round-trips to
    # the same Float64), so calibrated values can be copied back verbatim.
    @printf("qSav                       = %s\n", p.qSav)
    @printf("qBorr                      = %s\n", p.qBorr)
    @printf("bbar                       = %s\n", p.bbar)
    if hasproperty(eq, :elapsedSeconds)
        @printf("solve time                 = %.3f seconds\n", eq.elapsedSeconds)
    end

    if show_statistics && hasproperty(eq, :statistics)
        print_aggregate_statistics(eq.statistics, p;
            label = @sprintf("ages %d-%d, real %d-%d  [CALIBRATION WINDOW]",
                             p.stats_age_lo, p.stats_age_hi,
                             p.age0_real + p.stats_age_lo - 1,
                             p.age0_real + p.stats_age_hi - 1))
        # The same statistics over the whole forward pass. Nothing is
        # calibrated on these; they are printed so the effect of restricting
        # the moments to the working-age window is visible rather than implied.
        if hasproperty(eq, :statisticsAllAges)
            print_aggregate_statistics(eq.statisticsAllAges, p;
                label = @sprintf("ages 1-%d, real %d-%d  [ALL AGES]",
                                 p.maxAge, p.age0_real,
                                 p.age0_real + p.maxAge - 1))
        end
    end
    if show_welfare && hasproperty(eq, :welfare)
        print_welfare_summary(eq.welfare)
    end
    return nothing
end

function print_aggregate_statistics(s, p::HIParams; label::AbstractString = "")
    on_grid = p.asset_choice_method == :grid_search
    limit_label = on_grid ? "grid borrowing limit / mean labor income" :
                            "choice borrowing limit / mean labor income"
    bound_label = on_grid ? "share at effective grid borrowing bound" :
                            "share at borrowing bound"

    @printf("\n=== Aggregate statistics%s ===\n",
            isempty(label) ? "" : ": " * label)
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
    # Only the windowed statistics carry the entry-age block; `stats_all` is
    # printed through this same function and has no such age.
    if hasproperty(s, :meanAssetsAtStatsAgeLoToMeanLaborIncome)
        lo_real = p.age0_real + p.stats_age_lo - 1
        @printf("mean assets at age %-2d / mean labor income = %.8f\n",
                lo_real, s.meanAssetsAtStatsAgeLoToMeanLaborIncome)
        @printf("median assets at age %-2d / mean labor inc. = %.8f\n",
                lo_real, s.medianAssetsAtStatsAgeLoToMeanLaborIncome)
    end
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

"""
    warn_if_unsettled(eq, p)

Report the cross-section drift of the RETURNED equilibrium, once.

The closed-form PV tail assumes Y_j - C_j has stopped moving past `maxAge`.
Checking that inside the per-lambda solve produced one warning per probe, and
the root-finder visits corners (lambda = lambdaMin, qSav near its bracket) where
the economy is degenerate and legitimately unsettled -- true but useless. Only
the equilibrium actually returned has to be clean, so the test lives here, on
the single funnel every return path passes through.

The drift is reported RELATIVE to Y: an absolute bound on a sum of aggregate
differences is uninterpretable without its scale. A converged solve was measured
at 1.7e-09 of Y with lambda and W invariant to 15 digits across a doubled
maxAge; a genuinely unsettled path runs 1e-04 and worse. `tolDriftRel` sits
between them, and `Inf` silences this entirely.
"""
function warn_if_unsettled(eq, p::HIParams; converged::Bool = true)
    # A solve whose lambda never converged is not an equilibrium, so its
    # settling behaviour is not informative -- and the lambda failure is already
    # reported by the solver. Stacking a second warning on top buries the one
    # that matters. Observed: a calibration probe at qSav = 0.911 (floor 0.900)
    # where no lambda balances the budget, bottoming out at lambda = 0.018
    # against an equilibrium ~1.01, warned twice for one underlying problem.
    converged || return nothing
    hasproperty(eq, :diagnostics) || return nothing
    d = eq.diagnostics.finalDrift
    yscale = abs(eq.Y[end]) > 0 ? abs(eq.Y[end]) : 1.0
    any(x -> !(x / yscale <= p.tolDriftRel), d) || return nothing
    lines = join((@sprintf("kappa %d (% .4f): drift %.3e (%.1e of Y)  convergedAge %s",
                           ik, p.kappa_grid[ik], d[ik], d[ik] / yscale,
                           eq.diagnostics.convergedAgeByKappa[ik] == 0 ? "never" :
                           string(eq.diagnostics.convergedAgeByKappa[ik]))
                  for ik in eachindex(d)), "\n")
    @warn("RETURNED equilibrium: cross-section drift at maxAge exceeds tolDriftRel " *
          "for at least one kappa; the closed-form PV tail assumes the path has " *
          "settled.\n" *
          @sprintf("maxAge = %d, tolDriftRel = %.1e, Y[end] = %.6f\n",
                   p.maxAge, p.tolDriftRel, eq.Y[end]) *
          @sprintf("qSav = %.8f, qBorr = %.8f, bbar = %.8f, lambda = %.8f\n",
                   p.qSav, p.qBorr, p.bbar, eq.lambda) * lines *
          "\nThis is a CONVERGED equilibrium being returned, not a lambda probe. " *
          "During a calibration it is still one instrument triple among many; " *
          "check qSav/qBorr/bbar above against the calibrated values. " *
          "Judge by the relative column; raise maxAge if it is not many orders below Y.")
    return nothing
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
    evaluate_policy_grid_search!(Vcur, policyAIndex, flow_u, EV, p, util_weight, beta)

Policy evaluation: apply the stored asset choice without searching over it.
This is the cheap half of Howard's method -- one lookup per state instead of a
scan over `ia_first:nA` -- so it costs roughly 1/nA of a maximizing sweep.

Only the grid-search branch has this, because it is the branch whose flow
payoff `flow_u` is precomputed; reconstructing the payoff would cost as much as
re-optimizing and defeat the purpose.
"""
function evaluate_policy_grid_search!(Vcur, policyAIndex, flow_u, EV,
                                      p::HIParams, util_weight::Float64,
                                      beta::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    @inbounds for ia in 1:nA, iz in 1:nZ, ie in 1:nE
        iap = Int(policyAIndex[ia, iz, ie])
        u = flow_u[iap, ia, iz, ie]
        # A state with no feasible choice carries the `ia_first` fallback
        # policy, whose flow payoff is -Inf. Writing that into V would undo the
        # finite sentinel: the next maximizing sweep would see EV = -Inf, fail
        # every comparison, and fall back again -- so the state must keep the
        # same finite value the maximizer gave it.
        Vcur[ia, iz, ie] = isfinite(u) ?
            util_weight * u + beta * EV[iap, iz] : VINFEASIBLE
    end
    return nothing
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
            lower = asset_choice_lower_bound(kappa, iz, p)
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

"""
    accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                      true_borrowing_limit, effective_borrowing_limit,
                      at_borrowing_constraint, at_asset_upper, h_upper,
                      collect::Bool, p)

Add one (age, state) observation to a statistics accumulator.

Factored out because the forward pass now feeds TWO accumulators -- one over the
calibration age window and one over every age -- and a copied block would let
the two definitions drift. `collect` is passed rather than read from `p` so the
all-ages accumulator can skip the distribution vectors: those are a
cross-sectional object, the cross-section is the window, and pushing three
Float64 per (age, state) across all `maxAge` ages is the allocation that
OOM-killed the 128 GiB cluster jobs.

Marked `@inline`: this is the innermost loop of the forward pass, called once
per accumulator per positive-mass state per age.
"""
@inline function accumulate_stats!(stats::HIStatsAccumulator, weighted_mass::Float64,
                                   ia::Int, a::Float64, ap::Float64, h::Float64,
                                   c::Float64, y::Float64,
                                   true_borrowing_limit::Float64,
                                   effective_borrowing_limit::Float64,
                                   at_borrowing_constraint::Bool,
                                   at_asset_upper::Bool, h_upper::Float64,
                                   collect::Bool, p::HIParams)
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
        # Every age has a borrowing limit here, so all mass counts towards the
        # limit averages; the finite solver excludes its terminal age, where
        # a' >= 0 replaces the limit.
        stats.borrowing_limit_mass += weighted_mass
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
                    lower_ap = asset_choice_lower_bound(kappa, iz, p)
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
                true_borrowing_limit = -p.bbar * exp(kappa + p.rho * p.z_grid[iz])
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

        # Record where the cross-section settles, but DO NOT stop here. An
        # earlier version broke out of the loop at this point, which was a
        # speed optimization that corrupted everything downstream of the
        # profiles:
        #
        #   * C/H/Y/A keep their `zeros(maxAge)` initialization past the break,
        #     so the age profiles fell off a cliff to exactly 0.0 -- array
        #     initialization presented as model output;
        #   * each kappa settles at its own age and the aggregates are summed
        #     ACROSS kappa, so ages between the earliest and latest settled age
        #     held partial sums over only the kappas still running. The
        #     government budget read exactly that band, since it sums to Jc;
        #   * `discounted_sum` walks the whole maxAge-long array, so
        #     consumptionPV and outputPV were summing that zero tail;
        #   * worst, the `stats` accumulator also stopped, so each kappa
        #     contributed settled_age ages of mass and meanAssets came out
        #     weighted by settled age rather than by Pkappa. total_mass would
        #     be an integer if every kappa ran the same number of ages; it came
        #     out at 417.166667, which is the tell.
        #
        # Running every kappa the full maxAge fixes all four at once and costs
        # only the ages past settlement in the forward pass, which is small
        # next to the VFI. `converged_age` is kept purely as a diagnostic, so
        # an undersized maxAge is visible rather than silent.
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
    # Every kappa now runs the full maxAge, so the PV tail opens at maxAge for
    # all of them and `settledAge` is no longer a per-kappa quantity.
    # `convergedAge` is diagnostic: 0 for a kappa means the cross-section had
    # NOT settled by maxAge, in which case the closed-form tail rests on an
    # assumption the path has not yet earned and maxAge should be raised.
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
    )
end

function finalize_statistics(stats::HIStatsAccumulator, p::HIParams)
    total_mass = stats.total_mass
    mean_assets = stats.sum_current_assets / total_mass
    mean_labor_income = stats.sum_labor_income / total_mass
    # Mid-cumulative interpolation rather than StatsBase's weighted-quantile
    # convention, which is biased low on a coarse nonuniform grid holding a
    # discretized continuous distribution. See `interpolated_weighted_quantile`
    # in common/grids.jl for the measured comparison against a known median.
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

# Infinite horizon: the borrowing limit binds at every age. The finite solver
# replaces it with a' >= 0 at the terminal age; there is no terminal age here.
function asset_choice_lower_bound(kappa::Float64, iz::Int, p::HIParams)
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

# -----------------------------------------------------------------------------
# Shared infrastructure. Included rather than imported, so the methods land in
# THIS module's scope exactly as when they were written out inline here.
# -----------------------------------------------------------------------------
include(joinpath(@__DIR__, "..", "common", "grids.jl"))


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


function find_bracket(grid, residuals)
    for i in 1:(length(grid) - 1)
        # sign(0.0) is 0.0, so an exact zero at either end also brackets.
        if isfinite(residuals[i]) && isfinite(residuals[i + 1]) &&
           sign(residuals[i]) != sign(residuals[i + 1])
            return (i, i + 1)
        end
    end
    return nothing
end


"""
    discounted_sum_with_tail(x, q)

Present value of `x` over an INFINITE horizon, with the path constant at
`x[end]` past the last element:

    sum_{j=1}^{n} q^(j-1) x_j  +  x_n * q^n / (1 - q).

`discounted_sum` alone stops dead at `maxAge`, which drops `q^maxAge` of the
total -- 0.24% at qGov = 0.99 and maxAge = 600, and measurably so: outputPV
read 88.75 at maxAge 600 against 88.96 at 1200, a gap of 0.234% where q^600
predicts 0.2405%. The government budget never had this problem because it
already closed its own tail; this is the same closure applied to the reported
present values, and it makes them independent of maxAge.
"""
function discounted_sum_with_tail(x::AbstractVector{<:Real}, q::Real)
    isempty(x) && return 0.0
    0 <= q < 1 || error("discounted_sum_with_tail needs 0 <= q < 1, got q = $q")
    n = length(x)
    return discounted_sum(x, q) + x[n] * q^n / (1 - q)
end
