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
    # warning; 1e-8 is the setting that raises it, sized so that the 1.7e-09 a
    # converged solve shows passes while a genuinely unsettled path, which runs
    # 1e-04 and worse, is caught.
    #
    # SILENCED DELIBERATELY. At the production settings -- maxAge = 1000 with
    # qGov = 0.98357431 -- the warning has no teeth: the closed-form PV tail it
    # guards is discounted by qGov^1000 = 6.4e-08, so even the worst observed
    # drift of 3.0e-04 of Y injects at most ~2e-09 into the government budget,
    # four orders below tolGovBudget = 1e-05. The statistics window (ages 1-38)
    # never touches the tail and the welfare tail carries beta^1000.
    #
    # THE COST OF SILENCING IT. Inf switches the check off for EVERY run, not
    # just the long ones, and the same 3e-04 drift at maxAge = 100 would cost
    # qGov^100 = 0.19 of it -- a budget error of ~6e-05, above tolGovBudget
    # rather than far below. So restore 1e-8 before trusting a short-horizon
    # run, or judge the drift directly from `eq.diagnostics.finalDrift`, which
    # is still recorded whatever this is set to.
    tolDriftRel = Inf,

    # Share of all-ages mass at the top asset node above which the
    # hand-to-mouth rollover clipping is reported. Inf silences it.
    #
    # SILENCED ON REQUEST, AND THIS ONE IS NOT COSMETIC. Unlike tolDriftRel
    # above, where the quantity at stake was 5.7e-12 against a 1e-05 tolerance,
    # the clipped share is large and it moves the calibration. Measured at
    # maxAge = 1500, qSav = 0.98357431, beta = 0.98289691, bbar = -0.1723:
    #
    #   config                          window share   all-ages share   A[end]
    #   pHH = 1/3,        aMax = 100      0.00e+00        0.489           97.3
    #   pHH = 0.7589,     aMax = 100      0.00e+00        0.485           97.8
    #   pHH = 0.7589,     aMax = 400      0.00e+00        0.209          313.2
    #   no HtM (piH = 0), aMax = 100      0.00e+00        0.000           15.3
    #
    # Three things to read off it. The CALIBRATION WINDOW is clean -- no window
    # mass ever sits at the bound. The pile-up is caused entirely by the
    # hand-to-mouth rollover, since it vanishes at piH = 0, where the settled
    # level is an interior 15.3. And raising aMax does not fix it: the settled
    # level just tracks the new bound, the signature of a process with no
    # interior stationary distribution.
    #
    # The damage is indirect but real. Mean assets over the window fall from
    # 1.1946 to 0.7076, a 41 percent change, when aMax moves from 100 to 400 --
    # a bound no household in the window ever reaches. The continuation value
    # sees the whole asset space, so the calibration target is aMax-dependent.
    # Fix the rollover rule rather than the grid; see the README.
    htmClipWarnShare = Inf,

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

    # Asset-market access (psmodel.tex: s and h). pSS = Pr(stay saver),
    # pHH = Pr(stay hand-to-mouth). The stationary HtM share is
    #     piH = (1 - pSS) / (2 - pSS - pHH),
    # printed in the options header, and the initial cross-section is drawn
    # from it, so the HtM share is constant over the life cycle.
    #
    # CALIBRATED TO KAPLAN, VIOLANTE AND WEIDNER (2014), THEIR TABLE 4 -- the
    # printed SCF 2007-2009 two-year transition matrix across poor-HtM,
    # wealthy-HtM and non-HtM status. Collapsing P and W into H, weighting the
    # two rows by their ergodic mass, gives a two-year chain pSS = 0.8160,
    # pHH = 0.6029 whose stationary HtM share is 0.3166 against the 0.317 the
    # paper reports. Annualizing preserves the stationary distribution and
    # takes the square root of the second eigenvalue, 0.4189 -> 0.6472, which
    # gives the two numbers below. See references/KVW2014_WealthyHandToMouth/.
    #
    # WHY NOT THE IID RESTRICTION. pHH = 1 - pSS was the earlier default and
    # matched the one-third aggregate share, but it forces the second
    # eigenvalue to zero: an expected HtM spell of 1.50 years against the 4.15
    # implied here, and against the 3.5 (W-HtM) and 4.5 (P-HtM) the paper
    # states directly. Persistence is a separate moment from the share, and
    # Table 4 identifies it.
    #
    # The implied spells are 4.15 years in H and 8.95 years in S. Note the
    # paper's age profile is NOT flat -- total HtM falls from about 50 percent
    # at age 22 to about 20 percent in retirement (their Figure 6) -- while
    # this chain is stationary by construction, so piH is a life-cycle average.
    #
    #     pSS = 1.00,  pHH = 0.00   -> piH = 0       switches HtM off entirely
    #     pSS = 0.00,  pHH = 1.00   -> piH = 1       everyone hand-to-mouth
    pSS = 0.8882970895,
    pHH = 0.7589294614,

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
