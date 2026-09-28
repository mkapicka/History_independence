# =============================================================================
# model_settings.jl
#
# Settings for the history-dependent tax model. Loaded INSIDE the
# HistoryDependentTax module by solve_history_dependent_tax.jl and standalone:
# it does not reference the history-independent SETTINGS. After editing,
# re-include solve_history_dependent_tax.jl.
#
# HD_SETTINGS is the single source of truth: HDParams has no defaults for these
# keywords, so a missing entry raises UndefKeywordError rather than falling
# back to a stale duplicate. theta0 is NOT a setting; it is derived from
# (alpha, mu1, mu2, beta, J) by the finite-horizon normalization, and adding it
# back raises a MethodError.
#
# theta0 is NOT set here. It is derived from (alpha, mu1, mu2, beta) by
#   theta0 = 1 / ( alpha/(1-beta*mu1) + (1-alpha)/(1-beta*mu2) ),
# the INFINITE-horizon normalization, matching infinite_horizon.jl in the
# no-savings code. Adding theta0 back as a setting is a MethodError, not a
# silent override.
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


    # Initial assets at model age 1; a0_scales_with_kappa reads a0 as a
    # multiplier on exp(kappa), matching how wages and the borrowing limit
    # already scale with the permanent type.
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
    tolV = 1e-8,
    maxIterV = 2000,
    howardSteps = 40,
    tolDist = 1e-10,

    # ---------------------------------------------------------------------
    # SHOCKS
    # ---------------------------------------------------------------------
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
    p = HDParams(; HD_SETTINGS..., kwargs...)
    # The same two checks the history-independent solver makes in `validate`.
    1 <= p.stats_age_lo <= p.stats_age_hi ||
        error("need 1 <= stats_age_lo <= stats_age_hi, got $(p.stats_age_lo), $(p.stats_age_hi)")
    p.stats_age_hi <= p.maxAge ||
        error("stats_age_hi = $(p.stats_age_hi) exceeds the last model age maxAge = $(p.maxAge)")
    return p
end
