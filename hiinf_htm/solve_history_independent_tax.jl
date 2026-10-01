# =============================================================================
# solve_history_independent_tax.jl
#
# Solver: backward induction for the policies, a forward pass for the
# cross-section and its statistics, and a Brent solve for the tax level.
# For the infinite-horizon history-independent tax model with hand-to-mouth agents.
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

Households are in one of two exogenous asset-market-access states. SAVERS (S)
solve

    V^S(a,x) = max (1-beta)*(log(c) - phi*h^(1+eta)/(1+eta))
                   + beta*E[ pSS*V^S(a',x') + (1-pSS)*V^H(a',x') | x ]

subject to

    c + q(a')*a' = lambda*exp((1-tau)*(kappa + z + epsilon))*h^(1-tau) + a,
    bbar*exp(kappa + rho*z) <= a' <= aMax,

which is the original `hiinf` problem with a mixed continuation value.
HAND-TO-MOUTH (H) households solve the same problem with a' imposed rather than
chosen,

    a' = a/qSav for a >= 0,   a' = a for a < 0,
    c   = lambda*exp((1-tau)*(kappa + z + epsilon))*h^(1-tau) + 1_{a<0}*(1-qBorr)*a,

so only hours are chosen, statically. The access state follows a two-state
Markov chain with Pr(S'=S|S) = pSS and Pr(H'=H|H) = pHH, is independent of
(z, epsilon) and of the asset choice, and starts at its stationary
distribution, so the HtM share is constant at piH over the life cycle.

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
Base.@kwdef struct HIParams <: AbstractBewleyParams
    # Preferences and tax.
    beta::Float64
    eta::Float64
    phi::Float64
    tau::Float64

    # Asset-market access. Households are in one of two exogenous states:
    # savers (S), who choose a' freely subject to the borrowing limit, and
    # hand-to-mouth (H), who have no access to the asset market and whose
    # assets follow a' = a/qSav for a >= 0 and a' = a for a < 0 -- positive
    # balances are illiquid and roll over at the going return, negative ones
    # are serviced first, so debt neither grows nor is repaid. `pSS` and `pHH`
    # are `s` and `h` in psmodel.tex. piS/piH are derived; do not set them.
    pSS::Float64                     # Pr(S' = S | S)
    pHH::Float64                     # Pr(H' = H | H)
    piS::Float64 = first(access_stationary_distribution(pSS, pHH))
    piH::Float64 = last(access_stationary_distribution(pSS, pHH))

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
    # Share of all-ages mass at the top asset node above which the
    # hand-to-mouth rollover clipping is reported. Inf silences it; see
    # `warn_if_htm_rollover_clipped` and the note in model_settings.jl.
    htmClipWarnShare::Float64
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
    h_grid::Vector{Float64} = uniform_labor_grid(hMin, hMax, labor_grid_size)
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
    0.0 <= p.pSS <= 1.0 || error("pSS must satisfy 0 <= pSS <= 1")
    0.0 <= p.pHH <= 1.0 || error("pHH must satisfy 0 <= pHH <= 1")
    # access_stationary_distribution already rejects pSS = pHH = 1; re-check so
    # the failure is reported by validate when piS/piH are passed explicitly.
    p.pSS + p.pHH < 2.0 ||
        error("pSS + pHH must be below 2 for a unique access distribution")
    piS_check, piH_check = access_stationary_distribution(p.pSS, p.pHH)
    abs(p.piS - piS_check) <= 1e-12 && abs(p.piH - piH_check) <= 1e-12 ||
        error("piS/piH do not match the stationary distribution implied by " *
              "pSS = $(p.pSS), pHH = $(p.pHH); leave them unset")
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

Base.@kwdef mutable struct HIStatsAccumulator <: AbstractStatsAccumulator
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
    htm_mass::Float64 = 0.0
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
     :zero_asset_mass, :borrowing_constraint_mass, :htm_mass, :upper_bound_mass,
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
    @printf("  beta                        = %-20s  (discount factor)\n", p.beta)
    @printf("  eta                         = %-20s  (labor disutility curvature)\n", p.eta)
    @printf("  phi                         = %-20s  (labor disutility weight)\n", p.phi)
    @printf("  tau                         = %-20s  (HSV tax progressivity)\n", p.tau)
    @printf("  a0                          = %-20s  (initial assets at model age 1)\n", p.a0)
    @printf("  rho                         = %-20s  (AR(1) persistence of z)\n", p.rho)
    @printf("  sigma_omega                 = %-20s  (s.d. of the persistent innovation)\n", p.sigma_omega)
    @printf("  sigma_epsilon               = %-20s  (s.d. of the transitory shock)\n", p.sigma_epsilon)
    @printf("  sigma_kappa                 = %-20s  (s.d. of the fixed effect)\n", p.sigma_kappa)
    @printf("  z_initial                   = %-20s  (z at model age 1)\n", p.z_initial)
    @printf("  bbar                        = %-20s  (borrowing limit scale)\n", p.bbar)
    @printf("  pSS (stay saver)            = %-20s  (s in psmodel.tex)\n", p.pSS)
    @printf("  pHH (stay hand-to-mouth)    = %-20s  (h in psmodel.tex)\n", p.pHH)
    @printf("  access shares (piS, piH)    = (%.8f, %.8f)%s\n",
            p.piS, p.piH,
            p.piH == 0.0 ? "   [NO HtM AGENTS: this is the hiinf model]" : "")
    @printf("  qSav                        = %-20s  (price of saving,    a' >= 0)\n", p.qSav)
    @printf("  qBorr                       = %-20s  (price of borrowing, a' < 0)\n", p.qBorr)
    @printf("  qGov                        = %-20s  (government discount price)\n", p.qGov)
    @printf("  G                           = %-20s  (government spending)\n", p.G)
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
    # The inputs -- beta, bbar and the three prices -- are NOT repeated here.
    # They are printed once, with every digit, in the options panel at the top of
    # the run, so this panel carries only what the solve produced.
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

# Failure modes (no bracket, Brent throw, residual above tolerance) are reported
# with @warn at the point of failure, so eq carries only `converged` beyond the
# equilibrium objects themselves.
function attach_elapsed(eq, start_time::Float64, p::HIParams; converged::Bool)
    elapsed = time() - start_time
    eq_out = merge(eq, (; converged = converged, elapsedSeconds = elapsed))
    warn_if_unsettled(eq_out, p; converged = converged)
    warn_if_htm_rollover_clipped(eq_out, p)
    if p.verbose
        print_upper_bound_warning(eq_out.statistics)
        @printf("total solve time          = %.3f seconds\n", elapsed)
        flush(stdout)
    end
    return eq_out
end

"""
    warn_if_htm_rollover_clipped(eq, p)

Report that the hand-to-mouth rollover rule `a' = a/qSav` ran off the top of
the asset grid for some node, and whether any mass is actually there.

The rule is NOT a fixed point: a household that stays hand-to-mouth with `a > 0`
sees its assets grow at the gross return `1/qSav` forever while consuming only
labor income, so the only thing that stops it is `aMax`. Whether that matters
is an empirical question about `pHH` and `maxAge`, not something the grid can
decide, so this reports the level rather than raising an error. `clipped` alone
is nearly always true (the top node maps above itself); the share at the upper
bound is what to judge.

`htmClipWarnShare` sets the threshold and `Inf` silences it. Silencing does NOT
make the clipping harmless: measured at maxAge = 1500, qSav = 0.98357431, the
share reaches 0.489, and raising aMax from 100 to 400 only moves it to 0.209
while the settled asset level tracks the bound (97.3 -> 313.2). The process has
no interior stationary distribution, so the bound, not the model, decides where
the mass ends up.
"""
function warn_if_htm_rollover_clipped(eq, p::HIParams)
    p.piH > 0.0 || return nothing
    hasproperty(eq, :diagnostics) || return nothing
    eq.diagnostics.htmRolloverClipped || return nothing
    share = eq.statisticsAllAges.shareAtAssetUpperBound
    share > p.htmClipWarnShare || return nothing
    # @sprintf needs a single literal format, so the prose is concatenated
    # around it rather than inside it -- the same shape warn_if_unsettled uses.
    @warn("hand-to-mouth rollover a' = a/qSav is capped at the top asset grid " *
          "node and a nonzero share of the all-ages mass sits there. HtM " *
          "balances grow at the gross return with no offsetting decumulation, " *
          "so this share rises with pHH and with maxAge. Raise aMax or lower " *
          "pHH.\n" *
          @sprintf("aMax = %.6f, share at upper bound = %.3e, 1/qSav = %.8f, pHH = %.4f, maxAge = %d",
                   last(p.a_grid), share, 1.0 / p.qSav, p.pHH, p.maxAge))
    return nothing
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

# -----------------------------------------------------------------------------
# Hand-to-mouth block. Defined ahead of the solver because `HTMTransition`
# annotates its signatures, and Julia evaluates those at method definition.
# -----------------------------------------------------------------------------
"""
    HTMTransition

Everything about the hand-to-mouth asset rule that does not depend on `kappa`,
`lambda`, or the shocks, precomputed once per `HIParams`.

The rule is exogenous: `a' = a/qSav` for `a >= 0` and `a' = a` for `a < 0`
(psmodel.tex). It is off the asset grid at every node -- `a/qSav` lands between
nodes by construction -- so the continuation value and the forward transition
BOTH go through the same Young lottery `(left, right, weight)` stored here.
Using two different placements would break the value-function/simulation
welfare cross-check in `finalize_welfare`, which is the guard that this is
implemented consistently.

`cash` is `a - q(a')a'`, the resources left for consumption before labor
income: exactly `0` for an unclipped saver-balance rollover and
`(1-qBorr)*a < 0` for a debtor, matching the budget constraint in psmodel.tex.
`clipped` flags that `a/qSav` exceeded the top grid node for at least one `a`,
in which case `a'` is held at `aMax` and the excess `a - qSav*aMax` is
consumed.
"""
struct HTMTransition
    next_assets::Vector{Float64}
    cash::Vector{Float64}
    left::Vector{Int}
    right::Vector{Int}
    weight::Vector{Float64}
    clipped::Bool
end

function htm_transition(p::HIParams)
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
        else
            ap = a
            cash[ia] = a - p.qBorr * ap
        end
        next_assets[ia] = ap
        l, r, w = asset_transition_weights(ap, p)
        left[ia] = l
        right[ia] = r
        weight[ia] = w
    end

    return HTMTransition(next_assets, cash, left, right, weight, clipped)
end

"""
    precompute_htm_payoffs(lambda, tax_base, htm, p)

Flow utility and hours for the hand-to-mouth state, over `(a, z, eps)`.

The HtM choice is STATIC: `a'` is exogenous, so hours solve the same
within-period first-order condition the saver uses, at `cash = a - q(a')a'`,
with no continuation term. That is why this is computed once per `lambda` and
`kappa` rather than inside the sweep, and why the HtM half of the Bellman
operator never maximizes -- it only evaluates.
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
            income_coeff = lambda * tax_base[iz, ie]
            u, h = optimal_labor_foc(cash, income_coeff, p)
            if isfinite(u)
                flow_u[ia, iz, ie] = u
                flow_h[ia, iz, ie] = h
            end
        end
    end
    return flow_u, flow_h
end

"""
    update_htm_values!(VcurH, flow_u_H, EV_H, htm, p, util_weight, beta)

One application of the hand-to-mouth Bellman operator. No maximization: the
static flow payoff is looked up and the continuation is read at the exogenous
`a'` through the stored Young lottery.
"""
function update_htm_values!(VcurH, flow_u_H, EV_H, htm::HTMTransition,
                            p::HIParams, util_weight::Float64, beta::Float64)
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
            end
        end
    end
    return nothing
end

"""
    solve_policies_for_kappa(lambda, kappa, first_ap, terminal_first_ap,
                             q_by_ap, tax_base, htm, p)

Value-function iteration on the PAIR `(V^S, V^H)`.

The two problems are linked only through the continuation. Because the access
shock is independent of `(z', eps')`, the mixing can be done AFTER the
expectation rather than inside it:

    EV_S = pSS*E[V^S] + (1-pSS)*E[V^H],
    EV_H = pHH*E[V^H] + (1-pHH)*E[V^S],

so `compute_expected_value!` is called twice, unchanged, and the saver block is
the `hiinf` block verbatim with `EV_S` in place of `EV`. That is the whole
change on the saver side.

Howard evaluation still applies to the saver's asset policy only. The HtM half
has no policy to store -- hours are static and `a'` is exogenous -- so
`update_htm_values!` runs on both maximizing and evaluating sweeps.
"""
function solve_policies_for_kappa(lambda::Float64, kappa::Float64,
                                  first_ap::Vector{Int},
                                  terminal_first_ap::Int,
                                  q_by_ap::Vector{Float64},
                                  tax_base::Matrix{Float64},
                                  htm::HTMTransition, p::HIParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    # No age dimension on the policies: this is the memory difference from the
    # finite-horizon solver, and the reason the stationary problem is cheaper.
    VnextS = zeros(nA, nZ, nE)
    VcurS = similar(VnextS)
    VnextH = zeros(nA, nZ, nE)
    VcurH = similar(VnextH)
    EVS_raw = zeros(nA, nZ)
    EVH_raw = zeros(nA, nZ)
    EV_S = zeros(nA, nZ)
    EV_H = zeros(nA, nZ)
    policyAIndex = Array{Int32}(undef, nA, nZ, nE)
    policyA = Array{Float64}(undef, nA, nZ, nE)
    policyH = Array{Float64}(undef, nA, nZ, nE)

    flow_u, flow_h = p.asset_choice_method == :grid_search ?
                     precompute_flow_payoffs(lambda, first_ap, q_by_ap, tax_base, p) :
                     (nothing, nothing)
    flow_u_H, flow_h_H = precompute_htm_payoffs(lambda, tax_base, htm, p)

    beta = p.beta
    util_weight = 1.0 - beta

    # One application of the Bellman operator. `maximize = true` re-optimizes
    # and rewrites the SAVER policy; `false` applies the stored one (Howard).
    function sweep!(maximize::Bool)
        compute_expected_value!(EVS_raw, VnextS, p)
        compute_expected_value!(EVH_raw, VnextH, p)
        # A state with no feasible choice carries the FINITE sentinel, so these
        # products are 0.0 when the weight is zero. With -Inf they would be
        # NaN, and the pSS = 1 / pHH = 0 corner -- the one that has to
        # reproduce hiinf exactly -- is precisely where the weight is zero.
        @inbounds for i in eachindex(EV_S)
            evs = EVS_raw[i]
            evh = EVH_raw[i]
            EV_S[i] = p.pSS * evs + (1.0 - p.pSS) * evh
            EV_H[i] = p.pHH * evh + (1.0 - p.pHH) * evs
        end

        if p.asset_choice_method == :grid_search
            if maximize
                solve_policy_age_grid_search!(
                    VcurS, policyAIndex, policyA, policyH, flow_u, flow_h, EV_S,
                    first_ap, p, util_weight, beta,
                )
            else
                evaluate_policy_grid_search!(
                    VcurS, policyAIndex, flow_u, EV_S, p, util_weight, beta,
                )
            end
        else
            # The interpolated branch has no cheap evaluation step (the flow
            # payoff is not precomputed), so it always maximizes; howardSteps
            # is ignored there and plain VFI is what runs.
            solve_policy_age_interpolated!(
                VcurS, policyAIndex, policyA, policyH, EV_S, lambda, kappa,
                tax_base, p, util_weight, beta,
            )
        end

        update_htm_values!(VcurH, flow_u_H, EV_H, htm, p, util_weight, beta)
        return nothing
    end

    # Swapping outside the closure keeps Vnext/Vcur unboxed: assigning to a
    # captured variable inside would make every access type-unstable.
    howard = p.asset_choice_method == :grid_search ? p.howardSteps : 0
    iters = 0
    gap = Inf
    for _ in 1:p.maxIterV
        sweep!(true)
        # Measure the sup norm over FINITE entries only, before swapping, and
        # over BOTH value functions: V^H can still be moving after V^S has
        # settled, since its only dynamics come through the access chain.
        # States with no feasible a' carry -Inf in both sweeps, and
        # -Inf - (-Inf) = NaN would poison the comparison: `gap <= tolV` is
        # then false forever and the iteration silently runs to maxIterV with
        # an unconverged policy. Those states never change and carry no mass,
        # so excluding them is the right measure, not a workaround.
        gap = 0.0
        @inbounds for i in eachindex(VcurS)
            if isfinite(VcurS[i]) && isfinite(VnextS[i])
                d = abs(VcurS[i] - VnextS[i])
                d > gap && (gap = d)
            end
            if isfinite(VcurH[i]) && isfinite(VnextH[i])
                d = abs(VcurH[i] - VnextH[i])
                d > gap && (gap = d)
            end
        end
        VnextS, VcurS = VcurS, VnextS
        VnextH, VcurH = VcurH, VnextH
        iters += 1
        gap <= p.tolV && break
        for _ in 1:howard
            sweep!(false)
            VnextS, VcurS = VcurS, VnextS
            VnextH, VcurH = VcurH, VnextH
        end
    end
    gap <= p.tolV ||
        @warn "value function did not converge" kappa iters gap tolV = p.tolV

    welfare_value_function = expected_initial_value(VnextS, VnextH, kappa, p)
    return policyAIndex, policyA, policyH, flow_h_H, welfare_value_function,
           iters, gap
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

# Newborns draw their access state from the STATIONARY distribution (piS, piH),
# independently of (a0, z, eps), so the birth value is the piS/piH mix of the
# two value functions at the same asset placement. The simulated counterpart in
# `simulate_kappa!` seeds the distribution the same way; if the two ever
# disagree, the welfare cross-check in `finalize_welfare` reports it.
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

"""
    accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                      true_borrowing_limit, effective_borrowing_limit,
                      at_borrowing_constraint, at_asset_upper, h_upper,
                      is_htm, collect::Bool, p)

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
                                   is_htm::Bool, collect::Bool, p::HIParams)
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

"""
    simulate_kappa!(...)

Forward pass for one `kappa`. The cross-section is now over
`(a, z, eps, access)` with `access = 1` for savers and `access = 2` for
hand-to-mouth, seeded at the stationary `(piS, piH)`.

The access chain is independent of `(z', eps')` and of the asset choice, so the
transition factorizes: the `(a', z', eps')` mass is built exactly as before and
then split `pSS / 1-pSS` (from S) or `1-pHH / pHH` (from H) across the two
access states. Nothing about the saver's own transition changes.
"""
function simulate_kappa!(C, H, Y, A, stats::HIStatsAccumulator,
                         stats_all::HIStatsAccumulator,
                         stats_lo::HIStatsAccumulator,
                         policyAIndex, policyA, policyH, policyH_htm,
                         kappa, pkappa,
                         first_ap::Vector{Int}, terminal_first_ap::Int,
                         q_by_ap::Vector{Float64},
                         tax_base::Matrix{Float64}, wage_base::Matrix{Float64},
                         htm::HTMTransition, p::HIParams, lambda::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    nAge = p.maxAge
    # Fourth axis: 1 = saver (S), 2 = hand-to-mouth (H).
    dist = zeros(nA, nZ, nE, 2)
    dist_next = similar(dist)
    ia0_left, ia0_right, ia0_w = initial_asset_weights(kappa, p)
    h_upper = hours_upper_bound(p)
    a_top = last(p.a_grid)
    upper_tol = asset_choice_bound_tol(a_top, p)
    # Flow utility per age, so the discounted sum can be closed analytically
    # past the settled age. Accumulating the discounted total directly would
    # silently truncate: stopping at age Jc drops a tail worth beta^Jc of
    # lifetime utility, 2.2e-3 at Jc = 150.
    u_by_age = zeros(nAge)
    converged_age = 0           # diagnostic only; 0 means never settled by nAge
    final_drift = NaN           # drift at the last age, reported not tested

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
            for iz in 1:nZ, ie in 1:nE, iacc in 1:2
                mass = dist[ia, iz, ie, iacc]
                if mass <= p.massTol
                    continue
                end
                is_htm = iacc == 2

                lower_idx = first_ap[iz]
                if is_htm
                    # Exogenous rule: a' = a/qSav for a >= 0, a' = a for a < 0,
                    # with the same Young lottery the value function used. The
                    # borrowing limit is not a constraint here -- HtM agents do
                    # not choose -- so `at_borrowing_constraint` is false and
                    # only the reported limit levels are carried through.
                    ap = htm.next_assets[ia]
                    q_ap = asset_price(ap, p)
                    cash = htm.cash[ia]
                    lower_ap = p.a_grid[lower_idx]
                    at_borrowing_constraint = false
                    at_asset_upper = ap >= a_top - upper_tol
                    next_left = htm.left[ia]
                    next_right = htm.right[ia]
                    next_right_weight = htm.weight[ia]
                    h = policyH_htm[ia, iz, ie]
                elseif p.asset_choice_method == :grid_search
                    iap = Int(policyAIndex[ia, iz, ie])
                    ap = p.a_grid[iap]
                    q_ap = q_by_ap[iap]
                    cash = a - q_ap * ap
                    lower_ap = p.a_grid[lower_idx]
                    at_borrowing_constraint = iap == lower_idx
                    at_asset_upper = iap == nA
                    next_left = iap
                    next_right = iap
                    next_right_weight = 0.0
                    h = policyH[ia, iz, ie]
                else
                    ap = policyA[ia, iz, ie]
                    q_ap = asset_price(ap, p)
                    cash = a - q_ap * ap
                    lower_ap = borrowing_limit(kappa, iz, p)
                    at_borrowing_constraint =
                        abs(ap - lower_ap) <= asset_choice_bound_tol(lower_ap, p)
                    at_asset_upper =
                        ap >= asset_upper_bound(p) - asset_choice_bound_tol(asset_upper_bound(p), p)
                    next_left, next_right, next_right_weight = asset_transition_weights(ap, p)
                    h = policyH[ia, iz, ie]
                end
                # A state with no feasible a' keeps the `best_h = hMin`
                # fallback and an infinite value; if the distribution ever
                # reaches one, consumption is non-positive and `log(c)` throws a
                # bare DomainError with no context. Report the state instead.
                # For HtM the fallback is `NaN`, which fails this test too.
                h > 0.0 || error(
                    "infeasible policy on a positive-mass state " *
                    "(age=$age, ia=$ia, iz=$iz, ie=$ie, access=$(is_htm ? "H" : "S"), " *
                    "a=$a, h=$h): no feasible next-period asset was found when solving")
                c = lambda * tax_base[iz, ie] * h^(1.0 - p.tau) + cash
                c > 0.0 || error(
                    "non-positive consumption on a positive-mass state " *
                    "(age=$age, ia=$ia, iz=$iz, ie=$ie, access=$(is_htm ? "H" : "S"), " *
                    "a=$a, ap=$ap, h=$h, c=$c)")
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
                                  h_upper, is_htm, false, p)
                in_stats_window &&
                    accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      at_borrowing_constraint, at_asset_upper,
                                      h_upper, is_htm, p.collect_distributions, p)
                at_stats_age_lo &&
                    accumulate_stats!(stats_lo, weighted_mass, ia, a, ap, h, c, y,
                                      true_borrowing_limit, effective_borrowing_limit,
                                      at_borrowing_constraint, at_asset_upper,
                                      h_upper, is_htm, false, p)

                # The access chain is independent of (z', eps') and of a', so
                # the (a', z', eps') mass is built exactly as in hiinf and then
                # split across the two access states.
                begin
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
    saved_policyHtmH = p.store_solutions ?
                       Vector{Array{Float64,3}}(undef, nKappa) : Array{Float64,3}[]
    q_by_ap = asset_prices(p)
    # kappa-independent, so built once and shared read-only across the threads.
    htm = htm_transition(p)
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
        policyAIndex, policyA, policyH, policyH_htm, welfare_value_function,
        vIters, vGap = solve_policies_for_kappa(
            lambda, kappa, first_ap, terminal_first_ap, q_by_ap, tax_base, htm, p,
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
            policyAIndex, policyA, policyH, policyH_htm,
            kappa, pkappa, first_ap, terminal_first_ap, q_by_ap,
            tax_base, wage_base, htm, p, lambda,
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
            saved_policyHtmH[ik] = policyH_htm
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
        # Hours in the hand-to-mouth state. There is no HtM asset policy to
        # save: a' = htm.next_assets[ia] for every (z, eps).
        policyHtmH_by_kappa = saved_policyHtmH,
        htmNextAssets = htm.next_assets,
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
                   htmRolloverClipped = htm.clipped,
                   settledAge = p.maxAge,
                   convergedAge = maximum(converged_by_kappa),
                   convergedAgeByKappa = converged_by_kappa,
                   finalDrift = drift_by_kappa)
    return (; C = C, H = H, Y = Y, A = A), stats, stats_all, welfare, solutions, diagnostics
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

