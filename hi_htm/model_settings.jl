# User settings. Edit this tuple to change the model, grids, or solver.
#
# This is the single source of truth: `HIParams` in
# `solve_history_independent_tax.jl` carries no defaults of its own for these
# parameters, so a value can only be changed here or by an explicit override
# passed to `make_history_independent_params`.
const SETTINGS = (;
    # dimensions
    J = 39,

    # Households are born at real age 22, which is model age 1 (model age is
    # the 1-based array index, so real age = age0_real + model age - 1).
    #
    # Cross-sectional statistics are averaged over MODEL ages stats_age_lo to
    # stats_age_hi inclusive. Kaplan-Violante (2014) Table 2 build the 0.588
    # target on a 2001 SCF cross-section of households aged 22-59, which with
    # age0_real = 22 is model ages 1-38. stats_age_hi = 0 means "through the
    # last age" and is resolved to J+1 by `hi_params`; that covers every age
    # and reproduces the behaviour from before the window existed, which is
    # why it is the default here. Set 1 and 38 to match the data moment.
    # Initial asset holdings at model age 1. a0 = 0.0 reproduces the original
    # "born with nothing" condition exactly. Set a0_scales_with_kappa = true to
    # read a0 as a multiplier on exp(kappa), matching how wages and the
    # borrowing limit already scale with the permanent type.
    a0 = 0.0,
    a0_scales_with_kappa = false,

    # CALIBRATION WINDOW. Model age is the 1-based array index, so
    # real age = age0_real + model age - 1. With age0_real = 20 the window
    # model ages 3-40 is REAL AGES 22-59, which is the cross-section Kaplan
    # and Violante (2014) Table 2 build the 0.588 mean-liquid-wealth target on
    # (2001 SCF, households aged 22-59, top 5% by net worth dropped). Model
    # ages 1-2 are real ages 20-21, excluded: the data moment starts at 22.
    #
    # These three move together. Changing age0_real without changing the
    # window silently shifts which real ages are averaged.
    #
    # NOTE for the FINITE solvers (hi, hi_htm, hd): stats_age_hi = 40 equals
    # J + 1 at J = 39, so the window now reaches the TERMINAL age, where
    # a' >= 0 replaces the borrowing limit. That age contributes to the asset
    # and income means but not to the borrowing-limit averages, which divide
    # by borrowing_limit_mass and so still cover j = 0,...,J-1 only.
    age0_real = 20,
    stats_age_lo = 3,
    stats_age_hi = 40,
    nZ = 5,
    nEps = 5,
    nKappa = 3,
    nA = 101,

    # preferences and tax
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

    # shocks
    rho = 0.958,
    sigma_omega = sqrt(0.017),
    sigma_epsilon = sqrt(0.081),
    sigma_kappa = sqrt(0.065 + 0.036),
    z_discretization_method = :rouwenhorst,
    tauchen_width = 3.0,
    z_initial = 0.0,

    # asset grid
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

    # financial and government
    qBorr = 0.97,
    qSav = 0.99,
    qGov = 0.99,
    G = 0.0,

    # labor
    hMin = 1e-8,
    hMax = 5.0,
    labor_solver = :hybrid_newton,
    labor_grid_size = 151,

    # lambda solver
    lambdaMin = 0.20,
    lambdaMax = 2.50,
    nLambdaSearch = 15,
    maxIterLambda = 60,
    # The residual is a step function of lambda on the discrete a'-grid;
    # tighter tolerances can ask Brent to resolve jumps with no grid support.
    tolGovBudget = 1e-5,
    tolLambda = 1e-5,

    # output
    verbose = true,
    printEveryLambda = 1,
    massTol = 1e-14,
    store_solutions = false,
    collect_distributions = true,
)

make_history_independent_params(; kwargs...) = hi_params(; SETTINGS..., kwargs...)
