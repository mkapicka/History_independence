# User settings. Edit this tuple to change the model, grids, or solver.
#
# This is the single source of truth: `HIParams` in
# `solve_history_independent_tax.jl` carries no defaults of its own for these
# parameters, so a value can only be changed here or by an explicit override
# passed to `make_history_independent_params`.
const SETTINGS = (;
    # dimensions
    J = 39,
    nZ = 5,
    nEps = 5,
    nKappa = 3,
    nA = 101,

    # preferences and tax
    beta = 0.960,
    eta = 2.0,
    phi = 1.0,
    tau = 0.181,

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
    aMax = 15.0,
    asset_choice_method = :grid_search,
    asset_choice_tol = 1e-8,
    asset_choice_max_iter = 50,

    # financial and government
    qBorr = 0.90,
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
