# User settings. Edit this tuple to change the model, grids, or solver.
#
# This is the single source of truth: `HIParams` in
# `solve_history_independent_tax.jl` carries no defaults of its own for these
# parameters, so a value can only be changed here or by an explicit override
# passed to `make_history_independent_params`.
const SETTINGS = (;
    # dimensions
    # INFINITE HORIZON: no J. The agent's problem is stationary, so V and the
    # policies carry no age index. What remain are solver controls; see the
    # header of solve_history_independent_tax.jl.
    maxAge = 600,
    tolV = 1e-8,
    maxIterV = 2000,
    howardSteps = 40,
    tolDist = 1e-10,
    # Warning threshold on drift RELATIVE to Y (see HIParams). Inf silences the
    # warning; 1e-8 is the setting that raises it, sized so the 1.7e-09 a
    # converged solve shows passes while a genuinely unsettled path, which runs
    # 1e-04 and worse, is caught.
    #
    # SILENCED DELIBERATELY, and it matters most during a CALIBRATION.
    # `warn_if_unsettled` runs from `attach_elapsed`, i.e. once per returned
    # equilibrium, and a beta/bbar calibration returns one per solve -- 18 to 75
    # of them. The warning was drowning the search table it was printed next to.
    #
    # It has no teeth at the horizons this is run at. The closed-form PV tail it
    # guards carries qGov^maxAge, which at maxAge = 1500 and qGov = 0.98357431
    # is 1.6e-11. Against an observed worst drift of 1.9e-04 of Y, the implied
    # budget error is at most 5.7e-12 even on the pessimistic assumption that
    # the path decays at 0.9993 per period -- seven orders below
    # tolGovBudget = 1e-05.
    #
    # THE COST. Inf switches the check off for EVERY run, not only the long
    # ones, and the same drift at maxAge = 100 would carry qGov^100 = 0.19 and
    # cost ~5e-05, ABOVE tolGovBudget rather than far below. Restore 1e-8 before
    # trusting a short-horizon run, or read `eq.diagnostics.finalDrift`, which
    # is recorded whatever this is set to.
    tolDriftRel = Inf,

    # Initial asset holdings at model age 0. a0 = 0.0 reproduces the original
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
