# =============================================================================
# params.jl
#
# The parameter object for this directory: its declaration, its constructor and
# its validation. Split out of the solver so that solve_history_dependent_tax.jl is the solver
# and nothing else -- the same separation BewleyCommon made for shared code, and
# the one Discrete_HA draws between +setup and +solver.
#
# model_settings.jl remains the single source of truth for VALUES. The struct
# carries no defaults of its own, so a value is set there or passed explicitly;
# this file says what the values are called, how the grids are built from them,
# and what combinations are refused.
#
# Included by solve_history_dependent_tax.jl, after its `using` block and before anything that
# needs the type. It is not a module: the definitions land in the same scope they
# did when they lived in the solver.
#
# Marek Kapicka, 2026
# =============================================================================

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
"""
    HDParams(; kwargs...)

Parameters for the history-dependent tax model. Model and solver keywords are
REQUIRED (no defaults): construct via `make_history_dependent_params`, which
fills them from `HD_SETTINGS` -- the single source of truth in
model_settings.jl. `alpha` is a free setting: pass a number, or `:paper` for the
paper mixture `(rho - mu1)/(mu2 - mu1)` (1 when the roots coincide). `theta0` is
DERIVED and cannot be set: it follows from the finite-horizon restriction
`sum_{s=0}^{J} beta^s theta_s = 1`, so it depends on `(alpha, mu1, mu2, beta, J)`.
The asset choice is always grid search and hours are always chosen on the labor
grid (the static labor FOC is invalid because hours move s').
"""
struct HDParams <: AbstractBewleyParams
    # ---------------------------------------------------------------------
    # PREFERENCES AND TAX
    # ---------------------------------------------------------------------
    beta::Float64
    eta::Float64
    phi::Float64
    tau::Float64
    theta0::Float64
    alpha::Float64
    mu1::Float64
    mu2::Float64
    pow::Float64                     # (1 - tau) * theta0

    # Asset-market access. Households are in one of two exogenous states:
    # savers (S), who choose a' freely subject to the borrowing limit, and
    # hand-to-mouth (H), who have no access to the asset market and whose
    # assets follow a' = a/qSav for a >= 0 and a' = a for a < 0. `pSS` and
    # `pHH` are `s` and `h` in psmodel.tex. piS/piH are derived.
    #
    # THIS DIRECTORY COMBINES BOTH COMPLICATIONS. Like `hdinf_htm`, hours move
    # the past-income stocks, so the hand-to-mouth hours choice is DYNAMIC and
    # `solve_block_htm!` must maximize rather than look up a static payoff.
    # Like `hi_htm`, the horizon is finite, so at the terminal age `a' >= 0` is
    # imposed and the rollover is clamped: a' = max(htm rule, 0). A terminal
    # hand-to-mouth debtor therefore settles up, consuming y + a rather than
    # y + (1-qBorr)a. psmodel.tex is infinite-horizon and says nothing about
    # that last point; it is a choice made here, matching `hi_htm`.
    pSS::Float64                     # Pr(S' = S | S)
    pHH::Float64                     # Pr(H' = H | H)
    piS::Float64
    piH::Float64

    # ---------------------------------------------------------------------
    # HORIZON, AGES AND SHOCKS
    # ---------------------------------------------------------------------
    J::Int

    # Model age is the 1-based array index: model age 1 is j = 0, at real age
    # age0_real. Statistics are averaged over model ages stats_age_lo to
    # stats_age_hi inclusive; `stats_age_hi = 0` resolves to J+1. The
    # equilibrium reports both the window and all ages, as `hi` does.
    age0_real::Int
    stats_age_lo::Int
    stats_age_hi::Int

    # Initial assets at model age 1. a0_scales_with_kappa multiplies it by
    # exp(kappa), as wages and the borrowing limit already scale with the
    # permanent type. Placed by the same Young lottery used for a'.
    a0::Float64
    a0_scales_with_kappa::Bool
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

    # ---------------------------------------------------------------------
    # ASSET GRID AND ASSET CHOICE
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # PRICES AND GOVERNMENT
    # ---------------------------------------------------------------------
    qBorr::Float64
    qSav::Float64
    qGov::Float64
    G::Float64

    # ---------------------------------------------------------------------
    # LABOR
    # ---------------------------------------------------------------------
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
    s_grid_method::Symbol            # :linear or :quantile

    # ---------------------------------------------------------------------
    # LAMBDA SOLVER
    # ---------------------------------------------------------------------
    lambdaMin::Float64
    lambdaMax::Float64
    nLambdaSearch::Int
    maxIterLambda::Int
    tolLambda::Float64
    tolGovBudget::Float64

    # ---------------------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------------------- and solver behavior
    verbose::Bool
    massTol::Float64
    collect_distributions::Bool
    exploit_hours_monotonicity::Bool
end

function HDParams(;
    # No defaults here: HD_SETTINGS is the single source of truth, applied
    # through make_history_dependent_params. A direct HDParams() call missing a
    # keyword raises UndefKeywordError rather than solving a different model.
    beta,
    eta,
    phi,
    tau,
    alpha,
    mu1,
    mu2,
    pSS,
    pHH,
    J,
    age0_real,
    stats_age_lo,
    stats_age_hi,
    a0,
    a0_scales_with_kappa,
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
    s_grid_method,
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
    0.0 <= mu1 < 1.0 || error("mu1 must be in [0, 1)")
    0.0 <= mu2 < 1.0 || error("mu2 must be in [0, 1)")
    0.0 <= pSS <= 1.0 || error("pSS must satisfy 0 <= pSS <= 1")
    0.0 <= pHH <= 1.0 || error("pHH must satisfy 0 <= pHH <= 1")
    piS, piH = access_stationary_distribution(pSS, pHH)
    nS1 >= 1 || error("nS1 must be at least 1")
    nS2 >= 1 || error("nS2 must be at least 1")
    bbar <= 0.0 || error("Use bbar <= 0. For a borrowing limit B > 0, pass bbar = -B.")
    hMin > 0.0 || error("hMin must be positive")
    hMax > hMin || error("hMax must exceed hMin")
    0.0 < s_hours_floor < hMax || error("s_hours_floor must lie in (0, hMax)")
    labor_grid_size >= 2 || error("labor_grid_size must be at least 2")
    labor_grid_spacing in (:log, :uniform) ||
        error("labor_grid_spacing must be :log or :uniform")
    s_grid_method in (:linear, :quantile) ||
        error("s_grid_method must be :linear or :quantile")
    asset_grid_method in (:nonuniform, :linear) ||
        error("asset_grid_method must be :nonuniform or :linear")
    0.0 <= asset_grid_zero_share < 1.0 ||
        error("asset_grid_zero_share must be in [0, 1)")
    asset_grid_zero_width >= 0.0 || error("asset_grid_zero_width must be nonnegative")

    # The mixture weight is a free setting: a number, used as given, or
    # :paper for (rho - mu1)/(mu2 - mu1), which is what build_theta uses in the
    # no-savings code and stays consistent when the roots move. Equal roots are
    # the one-root case, where alpha is irrelevant and set to 1.
    #
    # alpha lies in [0, 1] iff the roots bracket rho. Outside that range the
    # mixture is signed, which warns rather than errors; the denom and pow
    # checks below catch the degenerate cases.
    alpha = if alpha === :paper
        isapprox(mu1, mu2) ? 1.0 : (rho - mu1) / (mu2 - mu1)
    elseif alpha isa Real
        Float64(alpha)
    else
        error("alpha must be a number or :paper (got $(repr(alpha)))")
    end
    if !(0.0 <= alpha <= 1.0)
        @warn "alpha is outside [0, 1]: the two root blocks carry opposite signs" alpha rho mu1 mu2
    end

    # Promise keeping pins down theta0 through the FINITE-horizon
    # normalization,
    #
    #   sum_{s=0}^{J} beta^s theta_s = 1,   theta_s = theta0*M_s,
    #   M_s = alpha*mu1^s + (1-alpha)*mu2^s,
    #
    # written as an explicit sum so it matches build_theta in the no-savings
    # code term for term. The infinite-horizon version understates theta0; see
    # NOTES.md. mu = 0 contributes M_0 = 1 and M_s = 0 for s >= 1, so the
    # mu1 = mu2 = 0 limit still gives theta0 = 1 exactly.
    M = [alpha * mu1^s + (1.0 - alpha) * mu2^s for s in 0:J]
    denom = sum(beta^s * M[s+1] for s in 0:J)
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

    h_grid = BewleyCommon.build_labor_grid(hMin, hMax, labor_grid_size, h_grid;
                              spacing = labor_grid_spacing)
    h_grid_disutility = phi .* (h_grid .^ (1.0 + eta)) ./ (1.0 + eta)
    log_h_grid = log.(h_grid)
    h_income_power = h_grid .^ pow

    # Effective hours floor for the s-grid bounds: with hMin >= s_hours_floor
    # every realizable s' lies inside the grid (no clamping from low hours).
    s_floor = max(Float64(hMin), Float64(s_hours_floor))
    # Shared inputs to the closed-form stock moments; ignored by :linear.
    moment_args = (; alpha, mu1, mu2, theta0, beta, rho, tau, eta,
                   sigma_omega, sigma_epsilon, sigma_kappa)
    s1_grid = build_s_grid(mu1, nS1, J, kappa_grid, z_grid, eps_grid,
                           s_floor, hMax; method = s_grid_method, moment_args)
    s2_grid = build_s_grid(mu2, nS2, J, kappa_grid, z_grid, eps_grid,
                           s_floor, hMax; method = s_grid_method, moment_args)
    s_factor = Matrix{Float64}(undef, length(s1_grid), length(s2_grid))
    for i1 in eachindex(s1_grid), i2 in eachindex(s2_grid)
        s_factor[i1, i2] =
            exp(pow * (alpha * s1_grid[i1] + (1.0 - alpha) * s2_grid[i2]))
    end

    return HDParams(
        beta, eta, phi, tau, theta0, alpha, mu1, mu2, pow,
        Float64(pSS), Float64(pHH), piS, piH,
        J, age0_real, stats_age_lo, stats_age_hi,
        Float64(a0), a0_scales_with_kappa,
        z_grid, Pz, z0_probs, eps_grid, Peps, kappa_grid, Pkappa,
        z_discretization_method, tauchen_width, rho,
        bbar, aMax, nA, a_grid,
        asset_grid_method, asset_grid_curvature_borrow,
        asset_grid_curvature_save, asset_grid_borrow_share,
        asset_grid_zero_share, asset_grid_zero_width,
        qBorr, qSav, qGov, G,
        hMin, hMax, h_grid, h_grid_disutility, log_h_grid, h_income_power,
        labor_grid_spacing,
        s1_grid, s2_grid, s_factor, Float64(s_hours_floor), s_grid_method,
        lambdaMin, lambdaMax, nLambdaSearch, maxIterLambda,
        tolLambda, tolGovBudget,
        verbose, massTol, collect_distributions, exploit_hours_monotonicity,
    )
end
