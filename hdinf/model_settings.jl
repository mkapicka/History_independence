# Settings for the history-dependent tax model. This file is loaded INSIDE the
# HistoryDependentTax module by solve_history_dependent_tax.jl and is fully
# standalone: it does not reference the history-independent SETTINGS. Edit the
# values here to change the model, grids, or solver; after editing, re-include
# solve_history_dependent_tax.jl (the module is replaced, with a harmless
# warning).
#
# HD_SETTINGS is the SINGLE source of truth: the HDParams constructor has no
# defaults for these keywords, so removing an entry here fails fast with an
# UndefKeywordError rather than falling back to a stale duplicate.
#
# alpha IS set here and is free. Two forms are accepted:
#   alpha = <number>   used exactly as given, whatever the roots are;
#   alpha = :paper     the paper mixture (rho - mu1)/(mu2 - mu1), 1 if mu1 = mu2.
# A number does NOT track the roots, so it goes stale if you edit mu1 or mu2;
# :paper stays consistent with them. Use a number when alpha is an object of
# study, :paper when the roots are meant to be the optimal ones.
#
# theta0 is NOT set here. It is derived from (alpha, mu1, mu2, beta) by
#   theta0 = 1 / ( alpha/(1-beta*mu1) + (1-alpha)/(1-beta*mu2) ),
# the INFINITE-horizon normalization, matching infinite_horizon.jl in the
# no-savings code. Adding theta0 back as a setting is a MethodError, not a
# silent override.
#
# NOTE ON THE ROOTS. alpha lands in [0, 1] iff mu1 <= rho <= mu2, i.e. the
# roots BRACKET the income persistence rho = 0.958. An alpha outside [0, 1]
# still solves but gives a signed mixture and a warning.
const HD_SETTINGS = (;
    # preferences and tax level
    beta = 0.96,
    eta = 2.0,
    phi = 1.0,
    tau = 0.181,

    # tax-function history dependence (preliminary values; mu1 and mu2 are to
    # be set according to the optimal-tax proposition / calibrated later;
    # mu1 = mu2 = 0 with nS1 = nS2 = 1 reproduces the history-independent
    # model exactly).
    #
    # These are the optimal roots of the infinite-horizon no-savings model
    # (../../infinite_horizon.jl, reported in nosavings_benchmark.jmd), so the
    # Bewley economy is solved at the analytical model's optimal tax. They
    # bracket rho = 0.958 as required, so with alpha = :paper below,
    #   alpha  = (rho - mu1)/(mu2 - mu1) = 0.922170   (benchmark: 0.9222)
    #   theta0 = 0.269711
    # theta0 now DOES match the no-savings benchmark's 0.2698, because both use
    # the infinite-horizon normalization. The finite-J hd solver gives 0.283385
    # at J = 39 instead. Compare this solver against infinite_horizon.jl, not
    # against main_finite.
    #
    # COST NOTE: mu2 = 0.9877 makes the s2-grid width scale as
    # mu2/(1-mu2) = 80.3 (against 1.54 for the s1 grid at mu1 = 0.6061), so
    # nS2 = 7 leaves the s2 dimension badly under-resolved. Raise nS2 well
    # above nS1 before reading any result off this baseline.
    mu1 = 0.6061,
    mu2 = 0.9877,

    # mixture weight: a number, or :paper for (rho - mu1)/(mu2 - mu1).
    # :paper keeps it consistent with the roots above; replace with e.g.
    # alpha = 0.5 to set it independently.
    alpha = :paper,

    # INFINITE HORIZON. No J: the agent's problem is stationary, so V and the
    # policies carry no age index. What remains are solver controls.
    #
    #   maxAge      cap on the FORWARD pass, which still runs age by age from
    #               the birth condition because aggregates vary over the life
    #               cycle and the government budget is a present value. The
    #               pass stops early once the cross-section settles (tolDist)
    #               and the discounted tail is then summed in closed form, so
    #               maxAge is a safety net rather than the usual stopping rule.
    #   tolV        sup-norm tolerance on the value function.
    #   maxIterV    cap on maximizing sweeps.
    #   howardSteps policy-evaluation sweeps between maximizations. 0 gives
    #               plain VFI, which contracts at beta = 0.96 and so needs
    #               ln(tol)/ln(beta) ~ 451 sweeps at tol = 1e-8 -- far more
    #               than the 100 age sweeps of the J = 99 finite model. With
    #               Howard the expensive maximizations number a few dozen.
    #               Both settings must reach the same fixed point; disagreement
    #               is a bug, and comparing them is the cheapest check there is.
    #   tolDist     drift in (Y_j, C_j) below which the cross-section counts as
    #               settled and the PV tail is closed analytically.
    maxAge = 600,

    # Households are born at real age age0_real, which is model age 1: real age
    # = age0_real + model age - 1. Cross-sectional statistics are averaged over
    # model ages stats_age_lo..stats_age_hi. Defaults match the
    # history-independent solver exactly, so both report a 22-59 cross-section
    # out of the box; set stats_age_lo = 1, stats_age_hi = maxAge to recover the
    # all-ages number this solver used to report as its only statistic.
    # age0_real = 21, stats_age_lo = 2, stats_age_hi = 39 is the same real
    # window with birth at 21.

    # Initial asset holdings at model age 1. a0 = 0.0 reproduces the original
    # "born with nothing" condition exactly; a0_scales_with_kappa reads a0 as a
    # multiplier on exp(kappa), matching how wages and the borrowing limit
    # already scale with the permanent type.
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
    tolV = 1e-8,
    maxIterV = 2000,
    howardSteps = 40,
    tolDist = 1e-10,

    # shocks
    rho = 0.958,
    sigma_omega = sqrt(0.017),
    sigma_epsilon = sqrt(0.081),
    sigma_kappa = sqrt(0.065 + 0.036),
    nZ = 5,
    nEps = 5,
    nKappa = 3,
    z_discretization_method = :rouwenhorst,

    # asset grid
    bbar = -0.17014842731387517,
    aMax = 15.0,
    nA = 101,
    asset_grid_method = :nonuniform,
    asset_grid_curvature_borrow = 1.8,
    asset_grid_curvature_save = 2.5,
    asset_grid_borrow_share = 0.35,
    asset_grid_zero_share = 0.30,
    asset_grid_zero_width = 0.08,

    # financial and government
    qBorr = 1.0123473140001613,
    qSav = 0.9723466284755885,
    qGov = 0.99,
    G = 0.0,

    # labor: hours are chosen on this grid (the joint (a', h) search is the
    # only option here, since hours move s' and the static labor FOC is
    # invalid). The grid is LOG-SPACED (uniform in ln h), which matches the
    # model: s' depends on ln h, and income/disutility are power functions,
    # so relative hours resolution is what matters. hMin = 0.05 (not 1e-8)
    # keeps ln(hMin) finite-scaled and, together with s_hours_floor,
    # guarantees every realizable s' lies inside the s-grids.
    #
    # labor_grid_size drives the discretization error in hours: adjacent
    # points differ by a factor of 1.122 at 41 points but only 1.047 at 101.
    # Measured at mu1 = mu2 = 0 against the history-independent solver, whose
    # continuous labor FOC gives mean assets / mean labor income = 0.58750135
    # at these prices (nA = 101, J = 39):
    #     nH =  41  ->  0.60017143  (+2.16%),  10s
    #     nH = 101  ->  0.59306286  (+0.95%),  20s
    #     nH = 161  ->  0.58814079  (+0.11%),  26s
    #     nH = 321  ->  0.58782189  (+0.05%),  43s
    # Convergence alternates in sign rather than decaying monotonically, as
    # grid-rounding error does. Raise to 161 if the residual percent matters.
    # The 41-point default predated the Topkis acceleration of the hours
    # scan, which now prunes most of the extra work.
    hMin = 0.05,
    hMax = 5.0,
    labor_grid_size = 101,
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

    # spacing of the s1/s2 grids inside their (reachable) bounds:
    #   :linear    equally spaced;
    #   :quantile  placed at quantiles of the age-pooled stock distribution,
    #              which is normal in closed form (s_stock_moments) because the
    #              no-savings model is jointly lognormal with closed-form hours.
    # The stock distribution is concentrated well inside the reachable range, so
    # equal spacing wastes points on tails the model rarely visits. :quantile is
    # a PLACEMENT heuristic only -- the endpoints still span the reachable range,
    # so it changes how efficiently the grid resolves the model, never what the
    # model is.
    s_grid_method = :linear,

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

function make_history_dependent_params(; kwargs...)
    p = HDParams(; HD_SETTINGS..., kwargs...)
    # The same two checks the history-independent solver makes in `validate`.
    1 <= p.stats_age_lo <= p.stats_age_hi ||
        error("need 1 <= stats_age_lo <= stats_age_hi, got $(p.stats_age_lo), $(p.stats_age_hi)")
    p.stats_age_hi <= p.maxAge ||
        error("stats_age_hi = $(p.stats_age_hi) exceeds the last model age maxAge = $(p.maxAge)")
    return p
end
