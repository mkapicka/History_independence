# Settings for the history-dependent tax model. This file is loaded INSIDE the
# HistoryDependentTax module by solve_history_dependent_tax.jl and is fully
# standalone: it does not reference the history-independent SETTINGS. Edit the
# values here to change the model, grids, or solver; after editing, re-include
# solve_history_dependent_tax.jl (the module is replaced, with a harmless
# warning).
#
# theta0 is NOT set here: it is implied by the restriction
#   theta0 * ( alpha/(1-beta*mu1) + (1-alpha)/(1-beta*mu2) ) = 1.
const HD_SETTINGS = (;
    # preferences and tax level
    beta = 0.96,
    eta = 2.0,
    phi = 1.0,
    tau = 0.181,

    # tax-function history dependence (preliminary values; mu1 and mu2 are to
    # be set according to the optimal-tax proposition / calibrated later;
    # mu1 = mu2 = 0 with nS1 = nS2 = 1 reproduces the history-independent
    # model exactly)
    alpha = 0.5,
    mu1 = 0.30,
    mu2 = 0.05,

    # horizon and shocks
    J = 39,
    rho = 0.958,
    sigma_omega = sqrt(0.017),
    sigma_epsilon = sqrt(0.081),
    sigma_kappa = sqrt(0.065 + 0.036),
    nZ = 5,
    nEps = 5,
    nKappa = 3,
    z_discretization_method = :rouwenhorst,

    # asset grid
    bbar = -0.1747771615355756,
    aMax = 12.0,
    nA = 101,
    asset_grid_method = :nonuniform,
    asset_grid_curvature_borrow = 1.8,
    asset_grid_curvature_save = 2.5,
    asset_grid_borrow_share = 0.35,
    asset_grid_zero_share = 0.30,
    asset_grid_zero_width = 0.08,

    # financial and government
    qBorr = 0.9484058794159081,
    qSav = 1.000389811142351,
    qGov = 0.99,
    G = 0.0,

    # labor: the joint (a', h) grid search is roughly nS1*nS2 times more
    # expensive than the history-independent solver, so the hours grid is
    # coarser here. The grid is LOG-SPACED (uniform in ln h), which matches
    # the model: s' depends on ln h, and income/disutility are power
    # functions, so relative hours resolution is what matters. hMin = 0.05
    # (not 1e-8) keeps ln(hMin) finite-scaled and, together with
    # s_hours_floor, guarantees every realizable s' lies inside the s-grids.
    hMin = 0.05,
    hMax = 5.0,
    labor_grid_size = 41,
    labor_grid_spacing = :log,

    # rigorous acceleration of the hours scan (Topkis monotonicity of the
    # optimal hours index in current assets, per asset choice); set false
    # only to cross-check against the unaccelerated scan
    exploit_hours_monotonicity = true,

    # past-income stock grids; s_hours_floor is used only for the s-grid
    # bounds (NOT a constraint on hours): bounds use
    # ln h in [log(s_hours_floor), log(hMax)]
    nS1 = 7,
    nS2 = 7,
    s_hours_floor = 0.05,

    # lambda solver
    lambdaMin = 0.20,
    lambdaMax = 2.50,
    nLambdaSearch = 15,
    maxIterLambda = 60,
    tolLambda = 1e-6,
    tolGovBudget = 1e-6,

    # output
    verbose = true,
    massTol = 1e-14,
    collect_distributions = true,
)

make_history_dependent_params(; kwargs...) = HDParams(; HD_SETTINGS..., kwargs...)
