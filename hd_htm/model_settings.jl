# =============================================================================
# model_settings.jl
#
# Settings for the history-dependent tax model. Loaded INSIDE the
# HistoryDependentTax module by solve.jl and standalone:
# it does not reference the history-independent SETTINGS. After editing,
# re-include solve.jl.
#
# HD_SETTINGS is the single source of truth: HDParams has no defaults for these
# keywords, so a missing entry raises UndefKeywordError rather than falling
# back to a stale duplicate. theta0 is NOT a setting; it is derived from
# (alpha, mu1, mu2, beta, J) by the finite-horizon normalization, and adding it
# back raises a MethodError.
#
# alpha accepts a number, used as given, or :paper for the mixture
# (rho - mu1)/(mu2 - mu1). A number does not track the roots and goes stale if
# mu1 or mu2 change. alpha lands in [0, 1] iff the roots bracket rho; outside
# that range the model still solves, with a signed mixture and a warning.
#
# Marek Kapicka, 2026
# =============================================================================

const HD_SETTINGS = (;
    # ---------------------------------------------------------------------
    # PREFERENCES AND TAX LEVEL
    # ---------------------------------------------------------------------
    beta = 0.96,
    eta = 2.0,
    phi = 1.0,
    tau = 0.181,

    # ---------------------------------------------------------------------
    # TAX-FUNCTION HISTORY DEPENDENCE
    # ---------------------------------------------------------------------
    # Optimal roots of the infinite-horizon no-savings model, so the Bewley
    # economy is solved at the analytical model's optimal tax. mu1 = mu2 = 0
    # with nS1 = nS2 = 1 reproduces the history-independent model exactly.
    # Compare theta0 against the finite-horizon no-savings driver at the same
    # J, not against the infinite-horizon figure. mu2 = 0.9877 leaves the s2
    # dimension under-resolved at nS2 = 7; see NOTES.md.
    mu1 = 0.6061,
    mu2 = 0.9877,

    # mixture weight: a number, or :paper for (rho - mu1)/(mu2 - mu1).
    # :paper keeps it consistent with the roots above; replace with e.g.
    # alpha = 0.5 to set it independently.
    alpha = :paper,

    # ---------------------------------------------------------------------
    # HORIZON, AGES AND SHOCKS
    # ---------------------------------------------------------------------
    J = 39,

    # Statistics window, in model ages (the 1-based array index):
    # real age = age0_real + model age - 1, so 3-40 here is real 22-59.
    # The three move together. At J = 39 the window reaches the terminal age,
    # which enters the asset and income means but not the borrowing-limit
    # averages. See NOTES.md for the data moment behind the window.
    age0_real = 20,
    stats_age_lo = 3,
    stats_age_hi = 40,

    # Initial assets at model age 1; a0_scales_with_kappa reads a0 as a
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

    # ---------------------------------------------------------------------
    # ASSET GRID AND ASSET CHOICE
    # ---------------------------------------------------------------------
    bbar = -0.17014842731387517,
    aMax = 15.0,
    nA = 101,
    asset_grid_method = :nonuniform,
    asset_grid_curvature_borrow = 1.8,
    asset_grid_curvature_save = 2.5,
    asset_grid_borrow_share = 0.35,
    asset_grid_zero_share = 0.30,
    asset_grid_zero_width = 0.08,

    # ---------------------------------------------------------------------
    # PRICES AND GOVERNMENT
    # ---------------------------------------------------------------------
    qBorr = 1.0123473140001613,
    qSav = 0.9723466284755885,
    qGov = 0.99,
    G = 0.0,

    # ---------------------------------------------------------------------
    # LABOR
    # ---------------------------------------------------------------------
    # Hours are chosen on a grid: the joint (a', h) search is the only option
    # here, since hours move s' and the static labor FOC is invalid. The grid
    # is log-spaced, which matches a model where s' depends on ln h and income
    # and disutility are power functions. hMin = 0.05 keeps ln(hMin)
    # finite-scaled and, with s_hours_floor, keeps every realizable s' inside
    # the s-grids. labor_grid_size drives the hours discretization error; see
    # NOTES.md for the measured convergence.
    hMin = 0.05,
    hMax = 5.0,
    labor_grid_size = 101,
    labor_grid_spacing = :log,

    # Topkis monotonicity of the optimal hours index in current assets; set
    # false only to cross-check against the unaccelerated scan.
    exploit_hours_monotonicity = true,

    # ---------------------------------------------------------------------
    # PAST-INCOME STOCK GRIDS
    # ---------------------------------------------------------------------
    # s_hours_floor sets the s-grid bounds only and is not a constraint on
    # hours: bounds use ln h in [log(s_hours_floor), log(hMax)].
    nS1 = 7,
    nS2 = 7,
    s_hours_floor = 0.05,

    # Spacing inside the reachable bounds: :linear equally spaced, or
    # :quantile at quantiles of the age-pooled stock distribution. A placement
    # heuristic only -- the endpoints span the reachable range either way.
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

    # ---------------------------------------------------------------------
    # LAMBDA SOLVER
    # ---------------------------------------------------------------------
    lambdaMin = 0.20,
    lambdaMax = 2.50,
    nLambdaSearch = 15,
    maxIterLambda = 60,
    tolLambda = 1e-6,
    tolGovBudget = 1e-6,

    # ---------------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------------
    verbose = true,
    massTol = 1e-14,
    collect_distributions = true,
)

function make_history_dependent_params(; kwargs...)
    opts = merge(NamedTuple(HD_SETTINGS), NamedTuple(kwargs))
    # `stats_age_hi = 0` means "through the last age", resolved here because
    # it depends on J, itself a setting.
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
