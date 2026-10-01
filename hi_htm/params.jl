# =============================================================================
# params.jl
#
# The parameter object for this directory: its declaration, its constructor and
# its validation. Split out of the solver so that solve_history_independent_tax.jl is the solver
# and nothing else -- the same separation BewleyCommon made for shared code, and
# the one Discrete_HA draws between +setup and +solver.
#
# model_settings.jl remains the single source of truth for VALUES. The struct
# carries no defaults of its own, so a value is set there or passed explicitly;
# this file says what the values are called, how the grids are built from them,
# and what combinations are refused.
#
# Included by solve_history_independent_tax.jl, after its `using` block and before anything that
# needs the type. It is not a module: the definitions land in the same scope they
# did when they lived in the solver.
#
# Marek Kapicka, 2026
# =============================================================================

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
    #
    # THE FINITE-HORIZON WRINKLE. psmodel.tex is written for an infinitely
    # lived agent, so it never says what a hand-to-mouth household does at the
    # LAST age, where `a' >= 0` is imposed so nobody dies in debt. The rule
    # a' = a would violate that for a debtor, so at the terminal age only the
    # rollover is clamped: a' = max(htm rule, 0). A terminal hand-to-mouth
    # debtor therefore settles up, consuming y + a rather than y + (1-q)a.
    # That is a modelling choice this directory makes, not one the tex dictates.
    pSS::Float64                     # Pr(S' = S | S)
    pHH::Float64                     # Pr(H' = H | H)
    piS::Float64 = first(access_stationary_distribution(pSS, pHH))
    piH::Float64 = last(access_stationary_distribution(pSS, pHH))

    # Ages j = 0,...,J are stored in length-(J+1) vectors at index j+1.
    J::Int

    # Model age is the 1-based array index: model age 1 is j = 0, at real age
    # age0_real. Statistics are averaged over model ages stats_age_lo to
    # stats_age_hi inclusive; `stats_age_hi = 0` is resolved to J+1 by
    # `hi_params`. The equilibrium reports both the window and all ages.
    age0_real::Int
    stats_age_lo::Int
    stats_age_hi::Int

    # Initial assets at model age 1. a0_scales_with_kappa multiplies a0 by
    # exp(kappa), as wages and the borrowing limit already scale with the
    # permanent type. Placed on the grid by the same Young lottery used for a',
    # not snapped to the nearest node.
    a0::Float64
    a0_scales_with_kappa::Bool

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

    # `stats_age_hi = 0` means "through the last age". Resolved here rather
    # than as a struct default because it depends on J, which is itself a
    # setting: a hard-coded number would silently mismatch whenever J moved.
    if !haskey(opts, :stats_age_hi) || opts.stats_age_hi <= 0
        opts = merge(opts, (; stats_age_hi = opts.J + 1))
    end

    return validate(HIParams(; opts...))
end

"""
    validate(p::HIParams)

Check the parameter combinations that the individual grid builders cannot see.
Returns `p` so it can wrap a construction.
"""
function validate(p::HIParams)
    0.0 <= p.pSS <= 1.0 || error("pSS must satisfy 0 <= pSS <= 1")
    0.0 <= p.pHH <= 1.0 || error("pHH must satisfy 0 <= pHH <= 1")
    p.pSS + p.pHH < 2.0 ||
        error("pSS + pHH must be below 2 for a unique access distribution")
    let (piS_check, piH_check) = access_stationary_distribution(p.pSS, p.pHH)
        abs(p.piS - piS_check) <= 1e-12 && abs(p.piH - piH_check) <= 1e-12 ||
            error("piS/piH do not match the stationary distribution implied " *
                  "by pSS = $(p.pSS), pHH = $(p.pHH); leave them unset")
    end
    # The original requirement was that a_grid contain 0.0 exactly, because
    # the initial holding was hard-coded to zero. What actually matters is that
    # it lies inside the grid and is feasible for every type.
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
            limit = p.bbar * exp(kappa + p.rho * z)
            a0k >= limit - 1e-12 ||
                error("initial assets a0 = $a0k (kappa = $kappa, z = $z) are " *
                      "below the borrowing limit $limit")
        end
    end
    1 <= p.stats_age_lo <= p.stats_age_hi ||
        error("need 1 <= stats_age_lo <= stats_age_hi, got $(p.stats_age_lo), $(p.stats_age_hi)")
    p.stats_age_hi <= p.J + 1 ||
        error("stats_age_hi = $(p.stats_age_hi) exceeds the last model age " *
              "$(p.J + 1); widen J or narrow the statistics window")
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
