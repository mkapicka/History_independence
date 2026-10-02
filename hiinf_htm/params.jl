# =============================================================================
# params.jl
#
# The parameter object for this directory: its declaration, its constructor and
# its validation. Split out of the solver so that solve.jl is the solver
# and nothing else -- the same separation BewleyCommon made for shared code, and
# the one Discrete_HA draws between +setup and +solver.
#
# model_settings.jl remains the single source of truth for VALUES. The struct
# carries no defaults of its own, so a value is set there or passed explicitly;
# this file says what the values are called, how the grids are built from them,
# and what combinations are refused.
#
# Included by solve.jl, after its `using` block and before anything that
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
