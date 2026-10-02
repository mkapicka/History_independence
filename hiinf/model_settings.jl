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
    # INFINITE HORIZON: no J. The agent's problem is stationary, so V and the
    # policies carry no age index. What remain are solver controls; see the
    # header of solve.jl.
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
    aMax = 15.0,
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
