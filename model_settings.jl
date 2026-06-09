# User settings. Edit this tuple to change the model, grids, or solver.
const SETTINGS = (;
    # dimensions
    J = 12,
    nZ = 5,
    nEps = 5,
    nKappa = 3,
    nA = 61,

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
    aMax = 12.0,

    # financial and government
    qBorr = 0.90,
    qSav = 0.99,
    qGov = 0.99,
    G = 0.0,

    # labor
    hMin = 1e-8,
    hMax = 5.0,
    labor_solver = :brent,

    # lambda solver
    lambdaMin = 0.20,
    lambdaMax = 2.50,
    nLambdaSearch = 15,
    # The residual is a step function of lambda on the discrete a'-grid;
    # tighter tolerances can ask Brent to resolve jumps with no grid support.
    tolGovBudget = 1e-5,
    tolLambda = 1e-5,

    # output
    verbose = true,
    printEveryLambda = 1,
    store_solutions = false,
)

make_history_independent_params(; kwargs...) = HIParams(; SETTINGS..., kwargs...)
