# =============================================================================
# model_settings.jl
#
# User settings for the history-independent tax model. Edit this tuple to
# change the model, the grids, or the solver. It is the single source of
# truth: HIParams carries no defaults of its own, so a value can only be set
# here or by an explicit override to make_history_independent_params.
#
# Marek Kapicka, 2026
# =============================================================================

const SETTINGS = (;
    # ---------------------------------------------------------------------
    # DIMENSIONS AND AGES
    # ---------------------------------------------------------------------
    J = 39,

    # Initial assets at model age 1. Set a0_scales_with_kappa to read a0 as a
    # multiplier on exp(kappa), as wages and the borrowing limit already are.
    a0 = 0.0,
    a0_scales_with_kappa = false,

    # Statistics window, in model ages (the 1-based array index):
    # real age = age0_real + model age - 1, so 3-40 here is real 22-59.
    # The three move together. At J = 39 the window reaches the terminal age,
    # which enters the asset and income means but not the borrowing-limit
    # averages. See NOTES.md for the data moment behind the window.
    age0_real = 20,
    stats_age_lo = 3,
    stats_age_hi = 40,
    nZ = 5,
    nEps = 5,
    nKappa = 3,
    nA = 101,

    # ---------------------------------------------------------------------
    # PREFERENCES AND TAX
    # ---------------------------------------------------------------------
    beta = 0.960,
    eta = 2.0,
    phi = 1.0,
    tau = 0.181,

    # Asset-market access (psmodel.tex: s and h). pSS = Pr(stay saver),
    # pHH = Pr(stay hand-to-mouth). The stationary HtM share is
    #     piH = (1 - pSS) / (2 - pSS - pHH),
    # printed in the options header, and the initial cross-section is drawn
    # from it, so the HtM share is constant over the life cycle.
    #
    # CALIBRATED TO KAPLAN, VIOLANTE AND WEIDNER (2014), THEIR TABLE 4 -- the
    # printed SCF 2007-2009 two-year transition matrix across poor-HtM,
    # wealthy-HtM and non-HtM status, collapsed to two states (weighting the
    # two HtM rows by their ergodic mass) and annualized by taking the square
    # root of the second eigenvalue. Gives piH = 0.3166 against the 0.317 the
    # paper reports, and an expected HtM spell of 4.15 years against their
    # stated 3.5 (W-HtM) and 4.5 (P-HtM). Matches hiinf_htm and hdinf_htm.
    # See references/KVW2014_WealthyHandToMouth/.
    #
    #     pSS = 1.00,  pHH = 0.00   -> piH = 0   reproduces `hi` exactly
    #     pSS = 0.00,  pHH = 1.00   -> piH = 1   everyone hand-to-mouth
    pSS = 0.8882970895,
    pHH = 0.7589294614,

    # ---------------------------------------------------------------------
    # SHOCKS
    # ---------------------------------------------------------------------
    rho = 0.958,
    sigma_omega = sqrt(0.017),
    sigma_epsilon = sqrt(0.081),
    sigma_kappa = sqrt(0.065 + 0.036),
    z_discretization_method = :rouwenhorst,
    tauchen_width = 3.0,
    z_initial = 0.0,

    # ---------------------------------------------------------------------
    # ASSET GRID AND ASSET CHOICE
    # ---------------------------------------------------------------------
    asset_grid_method = :nonuniform,
    asset_grid_borrow_share = 0.35,
    asset_grid_curvature_borrow = 1.8,
    asset_grid_curvature_save = 2.5,
    asset_grid_zero_share = 0.30,
    asset_grid_zero_width = 0.08,
    bbar = -0.20,
    # Raised from 15.0 for the median-NET-WORTH target (62442/52745 = 1.1838),
    # which puts mean assets near 2.1 times labour income -- five times the
    # liquid-wealth target -- and left 0.32 percent of the mass on a ceiling of
    # 15. `hi` still uses 15.0, so pass aMax explicitly to both when running the
    # hi / hi_htm nesting check.
    aMax = 60.0,
    asset_choice_method = :grid_search,
    asset_choice_tol = 1e-8,
    asset_choice_max_iter = 50,

    # ---------------------------------------------------------------------
    # PRICES AND GOVERNMENT
    # ---------------------------------------------------------------------
    qBorr = 0.97,
    qSav = 0.99,
    qGov = 0.99,
    G = 0.0,

    # ---------------------------------------------------------------------
    # LABOR
    # ---------------------------------------------------------------------
    hMin = 1e-8,
    hMax = 5.0,
    labor_solver = :hybrid_newton,
    labor_grid_size = 151,

    # ---------------------------------------------------------------------
    # LAMBDA SOLVER
    # ---------------------------------------------------------------------
    lambdaMin = 0.20,
    lambdaMax = 2.50,
    nLambdaSearch = 15,
    maxIterLambda = 60,
    # The residual steps on the discrete a'-grid, so tighter tolerances ask
    # Brent to resolve jumps with no grid support.
    tolGovBudget = 1e-5,
    tolLambda = 1e-5,

    # ---------------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------------
    verbose = true,
    printEveryLambda = 1,
    massTol = 1e-14,
    store_solutions = false,
    collect_distributions = true,
)

make_history_independent_params(; kwargs...) = hi_params(; SETTINGS..., kwargs...)
