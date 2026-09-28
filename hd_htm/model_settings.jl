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
# theta0 is NOT set here. It is derived from (alpha, mu1, mu2, beta, J) by
#   theta0 = 1 / sum_{s=0}^{J} beta^s (alpha*mu1^s + (1-alpha)*mu2^s),
# the FINITE-horizon normalization, matching build_theta in the no-savings
# code, so theta0 depends on J as well as on alpha and the roots. Adding
# theta0 back as a setting is a MethodError, not a silent override.
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
    #   theta0 = 0.283385 at J = 39
    # theta0 does NOT match the benchmark's 0.2698: that figure is the
    # INFINITE-horizon normalization, while theta0 here is the finite-J one and
    # so depends on J (0.283385 at J = 39, 0.270247 at J = 99, approaching
    # 0.269711 as J grows). Compare against the finite-horizon no-savings
    # driver (main_finite/solve_finite_recursive at the same J), not against
    # the infinite-horizon numbers.
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

    # horizon and shocks
    J = 39,

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
    # Initial asset holdings at model age 1. a0 = 0.0 reproduces the original
    # "born with nothing" condition exactly; a0_scales_with_kappa reads a0 as a
    # multiplier on exp(kappa).
    a0 = 0.0,
    a0_scales_with_kappa = false,
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

    # Asset-market access (psmodel.tex: s and h). pSS = Pr(stay saver),
    # pHH = Pr(stay hand-to-mouth); piH = (1-pSS)/(2-pSS-pHH) is the stationary
    # hand-to-mouth share, printed in the summary, and the initial
    # cross-section is drawn from it.
    #
    # CALIBRATED TO KAPLAN, VIOLANTE AND WEIDNER (2014) TABLE 4, matching
    # hi_htm, hiinf_htm and hdinf_htm: the SCF 2007-2009 two-year transition
    # matrix collapsed to two states and annualized, giving piH = 0.3166 and an
    # expected hand-to-mouth spell of 4.15 years. See
    # references/KVW2014_WealthyHandToMouth/ and
    # paper/notes/DOT_AccessChainAnnualization.tex.
    #
    #     pSS = 1.00, pHH = 0.00 -> piH = 0   reproduces `hd` exactly
    #     pSS = 0.00, pHH = 1.00 -> piH = 1   everyone hand-to-mouth
    pSS = 0.8882970895,
    pHH = 0.7589294614,

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
    opts = merge(NamedTuple(HD_SETTINGS), NamedTuple(kwargs))
    # `stats_age_hi = 0` means "through the last age". Resolved here rather than
    # as a struct default because it depends on J, itself a setting: a
    # hard-coded number would silently mismatch whenever J moved.
    if opts.stats_age_hi <= 0
        opts = merge(opts, (; stats_age_hi = opts.J + 1))
    end
    p = HDParams(; opts...)
    1 <= p.stats_age_lo <= p.stats_age_hi ||
        error("need 1 <= stats_age_lo <= stats_age_hi, got $(p.stats_age_lo), $(p.stats_age_hi)")
    p.stats_age_hi <= p.J + 1 ||
        error("stats_age_hi = $(p.stats_age_hi) exceeds the last model age $(p.J + 1)")
    for kappa in p.kappa_grid
        a0k = p.a0_scales_with_kappa ? p.a0 * exp(kappa) : p.a0
        first(p.a_grid) - 1e-12 <= a0k <= last(p.a_grid) + 1e-12 ||
            error("initial assets a0 = $a0k (kappa = $kappa) fall outside a_grid")
    end
    return p
end
