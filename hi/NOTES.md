# Measurement notes — `hi`

Extracted from the source comments on 2026-09-28, before those comments were
rewritten in the terse Discrete_HA register. Each entry records a number that
settled a choice and is not recoverable from the code itself. Line numbers are
from the pre-rewrite files and will drift; the anchor line identifies the site.

## `hi/calibrate_history_independent_tax.jl:1`

Anchor: `using Dates`

```
# =============================================================================
# calibrate_history_independent_tax.jl
#
# Calibrate the three financial / borrowing-limit parameters
#
#     qSav   (gross savings price,  a' >= 0)
#     qBorr  (gross borrowing price, a' <  0)
#     bbar   (borrowing-limit scale, bbar <= 0)
#
# so that the stationary cross-section produced by
# `solve_history_independent_tax` matches three data moments:
#
#     (i)   mean assets / mean labor income             = 0.588
#           (or median / mean labor income = 0.0498 when asset_moment = :median)
#     (ii)  true borrowing limit / mean labor income     = 0.185
#     (iii) share of households with negative liquid      = 0.260
#           assets
#
# These map onto the statistics returned by `finalize_statistics` as
#
#     (i)   eq.statistics.meanAssetsToMeanLaborIncome   (default), or
#           eq.statistics.medianAssetsToMeanLaborIncome (asset_moment = :median)
#     (ii)  eq.statistics.meanBorrowingLimitToMeanLaborIncome, which averages
#           -bbar*exp(kappa+rho*z) over ages j = 0,...,J-1; the terminal age is
#           excluded because a' >= 0 is imposed there and no limit is defined
#     (iii) eq.statistics.shareNegativeLiquidAssets
#
# The `solve_*` file is used unmodified; this file only wraps it.
#
# -----------------------------------------------------------------------------
# Identification logic
# -----------------------------------------------------------------------------
# The mapping (qSav, qBorr, bbar) -> (moment_i, moment_ii, moment_iii) is
# coupled, but it has a strong near-triangular structure that we exploit:
#
#   * bbar  is the *only* parameter entering the "true" borrowing limit
#     numerator  -bbar * E[exp(kappa + rho*z)]  (see `true_borrowing_limit`
#     in simulate_kappa!, averaged over ages j = 0,...,J-1).  It therefore
#     pins down moment (ii) almost mechanically, with only a second-order
#     feedback through mean labor income.  -> bbar solves (ii).
#
#   * qSav  governs the return to saving and hence the right tail / median of
#     the asset distribution.  -> qSav solves (i).
#
#   * qBorr governs the cost of borrowing and hence how many households choose
#     a' < 0.  -> qBorr solves (iii).
#
# We solve this with an outer block-Gauss-Seidel / nested-bisection scheme:
# each instrument is moved by a 1-D bracketed root finder (Roots.Brent, already
# a dependency of the solver) holding the others fixed, and we sweep the three
# blocks until the joint residual is below tolerance.  Each residual evaluation
# rebuilds HIParams (because the asset grid's lower bound depends on bbar) and
# calls the unmodified solver with verbose output suppressed.
#
# References for this calibration strategy in heterogeneous-agent models:
#   * Kaplan, Moll & Violante (2018, AER) -- two-asset HANK; liquid-asset
#     targets (median liquid wealth, share of hand-to-mouth / negative liquid
#     positions) calibrated to SCF.
#   * Guvenen, Karahan, Ozkan & Song (2021, Ecta) -- moment-matching of
#     earnings-driven wealth statistics.
#   * Standard SMM/just-identified GMM logic: 3 instruments, 3 moments.
# =============================================================================
```

## `hi/calibrate_history_independent_tax.jl:110`

Anchor: `medianAssetsToMeanLaborIncome::Float64       = 0.0498  # (i), asset_moment = :`

```
    # Targets: data moments to match (only the asset_moment-selected ratio
    # among the first two is targeted; the other is reported but left free).
    # Sources: (i) and (ii) are Kaplan and Violante (2014), Table III, a 2001
    # SCF cross-section of households aged 22-59 with the top 5% by net worth
    # dropped -- net LIQUID wealth over mean earnings-plus-benefits, mean
    # 31,001/52,745 = 0.588 and median 2,629/52,745 = 0.0498. (The median read
    # 0.043 until 2026-09-23, from a 2,269 transposition of the 2,629 in that
    # table; every other figure in the row matches.) (iii) is the same paper,
    # p. 1221, where the borrowing rate is set so that 26% of agents hold
    # negative liquid balances.
```

## `hi/model_settings.jl:11`

Anchor: `a0 = 0.0,`

```
    # Households are born at real age 22, which is model age 1 (model age is
    # the 1-based array index, so real age = age0_real + model age - 1).
    #
    # Cross-sectional statistics are averaged over MODEL ages stats_age_lo to
    # stats_age_hi inclusive. Kaplan-Violante (2014) Table 2 build the 0.588
    # target on a 2001 SCF cross-section of households aged 22-59, which with
    # age0_real = 22 is model ages 1-38. stats_age_hi = 0 means "through the
    # last age" and is resolved to J+1 by `hi_params`; that covers every age
    # and reproduces the behaviour from before the window existed, which is
    # why it is the default here. Set 1 and 38 to match the data moment.
    # Initial asset holdings at model age 1. a0 = 0.0 reproduces the original
    # "born with nothing" condition exactly. Set a0_scales_with_kappa = true to
    # read a0 as a multiplier on exp(kappa), matching how wages and the
    # borrowing limit already scale with the permanent type.
```

## `hi/model_settings.jl:28`

Anchor: `age0_real = 20,`

```
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
```

## `hi/model_settings.jl:57`

Anchor: `mpc_shock = 0.0063278,`

```
    # Impact-MPC windfall and the low-asset threshold, in model asset units,
    # both on Discrete_HA's 2019 numeraire of $72,000 mean annual income
    # (`+setup/Params.m`): $500/$72,000 = 0.0069444 and $1,000/$72,000 =
    # 0.0138889 of mean annual labor income, times this calibration's mean
    # labor income of 0.9112. Re-derive both after a recalibration; the summary
    # prints each as a ratio to the realized mean so drift is visible.
```

## `hi/solve_history_independent_tax.jl:51`

Anchor: `mpc_shock::Float64`

```
    # Size of the unanticipated one-time windfall used for the impact MPC, and
    # the asset threshold below which a second, conditional MPC is reported.
    # Both are in model asset units and both follow Discrete_HA's 2019
    # numeraire (`+setup/Params.m`: `numeraire_in_dollars = 72000`,
    # `shocks_dollars = [-1, -500, -5000, 1, 500, 5000]`,
    # `dollar_thresholds = [1000, ...]`):
    #
    #     $500  / $72,000 = 0.0069444 of mean annual labor income
    #     $1,000 / $72,000 = 0.0138889
    #
    # times this calibration's mean labor income of 0.9112. They are ABSOLUTE
    # numbers, not fractions, because a fraction of mean income could only be
    # resolved after the forward pass that needs it; the summary prints the
    # realized ratios so drift away from $500 and $1,000 is visible rather than
    # assumed. RECALIBRATE and these two move -- re-derive them from the new
    # mean labor income rather than leaving them.
```

## `hi/solve_history_independent_tax.jl:79`

Anchor: `age0_real::Int`

```
    # MODEL AGE is the 1-based array index, so model age 1 is j = 0, the first
    # simulated period, at real age `age0_real`; real age = age0_real + j.
    #
    # Cross-sectional statistics are averaged over model ages stats_age_lo to
    # stats_age_hi INCLUSIVE rather than over the whole life. The calibration
    # target is Kaplan-Violante (2014) Table 2, built on a 2001 SCF
    # cross-section of households aged 22-59, so with age0_real = 22 that
    # window is model ages 1-38. `stats_age_hi = 0` in SETTINGS is resolved to
    # J+1 by `hi_params`, which covers every age and is the pre-window
    # behaviour; the equilibrium reports BOTH the window and the all-ages
    # version, so the effect of restricting it is visible rather than implied.
```

## `hi/solve_history_independent_tax.jl:94`

Anchor: `a0::Float64`

```
    # Initial asset holdings at model age 1 (j = 0). a0 = 0.0 is the original
    # condition -- everyone born with nothing -- and remains the default, so
    # results are unchanged unless it is set. a0_scales_with_kappa multiplies
    # a0 by exp(kappa), matching how wages and the borrowing limit
    # (-bbar*exp(kappa + rho*z)) already scale with the permanent type: a flat
    # a0 would otherwise leave the lowest-kappa household starting relatively
    # far richer. a0 is placed on the grid by the same Young lottery used for
    # a', not snapped to the nearest node, so it stays exact between points.
```

## `hi/solve_history_independent_tax.jl:230`

Anchor: `if p.a0 == 0.0 && !p.a0_scales_with_kappa`

```
    # The original requirement was that a_grid contain 0.0 exactly, because
    # the initial holding was hard-coded to zero. What actually matters is that
    # it lies inside the grid and is feasible for every type.
```

## `hi/solve_history_independent_tax.jl:298`

Anchor: `sum_mpc_positive::Float64 = 0.0`

```
    # The distribution of MPCs, not just its mean, following the measures
    # Discrete_HA's MPCFinder reports beside `avg`: the mean over responders
    # (`mpc_condl`), the responder shares (`mpc_pos`, `mpc_neg`, `mpc0`), and
    # the MPC of the low-liquid-wealth group (`mpc_htm_a_lt_1000`).
```

## `hi/solve_history_independent_tax.jl:602`

Anchor: `if hasproperty(s, :meanMPC)`

```
    # Kaplan-Violante (2022) eq. (2) averaged over the ages this block covers.
    # The shock is printed beside it, absolutely and against this block's own
    # mean labor income, because the MPC is only interpretable with the windfall
    # size attached -- the consumption function is concave, so a larger windfall
    # buys a smaller MPC.
```

## `hi/solve_history_independent_tax.jl:1120`

Anchor: `@inline function interpolate_consumption(con::AbstractVector{Float64},`

```
# =============================================================================
# THE IMPACT MPC
# =============================================================================
# Kaplan and Violante (2022, Annu. Rev. Econ.), their equation (2): for a
# household whose state is (b, y) when an unanticipated one-time windfall of
# size x arrives,
#
#     m_0(x; b, y) = [ c(b + x, y) - c(b, y) ] / x,
#
# and the average (their Online Appendix, equation D.7) integrates that function
# under the distribution,
#
#     mbar_0(x) = int m_0(x; b, y) dmu(b, y).
#
# Here the windfall lands on current assets, which is the model's cash-on-hand
# margin: m_0 = [c(a + x, z, eps, j) - c(a, z, eps, j)] / x, averaged with the
# forward-pass mass. Two coverages are reported, which is the one departure from
# KV asked for: ages j = 0,...,J (`statisticsAllAges`) and the calibration
# window stats_age_lo,...,stats_age_hi (`statistics`). KV have no age dimension
# to choose between -- their baseline is infinite-horizon and mu is stationary.
#
# TWO THINGS THE LEVEL IS NOT COMPARABLE TO. First, KV report a QUARTERLY MPC
# out of $500; this model is annual, so this is an annual MPC out of an annual
# windfall and will be larger for the same household. Second, hours are
# endogenous here and exogenous in KV's baseline, so c(a + x) embeds a labor
# supply response: the windfall relaxes the budget, hours fall, and the
# consumption response is net of that. The FORMULA is theirs exactly; the
# consumption function it is applied to is this model's.
#
# c(a + x) is read off the consumption policy by linear interpolation in a,
# which is what the Kaplan-lineage codes do (`coninterp_mpc` in Discrete_HA).
#
# ACCURACY, MEASURED. At the terminal age a' = 0 binds, so the problem is static
# and c(a) has a closed form: h solves A(1-tau) = c phi h^(eta+tau) with
# A = lambda*tax_base. Against the EXACT ARC [c(a+x) - c(a)]/x built from that
# root, the reported MPC converges in the asset grid at better than first order
# (:interpolate, J=39, nZ=nEps=5, nKappa=3): worst cell 2.04e-2, 9.02e-3,
# 3.78e-3, 1.46e-3 at nA = 51, 101, 201, 401, mean error 4.55e-3 to 3.03e-4.
# Benchmark it against the DERIVATIVE instead and a 2.2% floor appears that no
# refinement removes -- that gap is real convexity of c over the windfall, not
# error, since dc/da = (eta+tau)/(phi h^(1+eta) + eta+tau) RISES with a as hours
# fall. KV's object is the arc, which is why they report MPCs per shock size.
#
# GRID RESOLUTION MATTERS MORE THAN THE METHOD, but both converge. The average
# MPC under :grid_search exceeds :interpolate by 5.41% at nA = 101 and 1.91% at
# nA = 201, then agrees to -0.48% at nA = 401 (0.28542 against 0.28681): a'(a)
# is a step function under grid search and c inherits a sawtooth, which biases
# the MPC UP at coarse grids rather than permanently. At the production nA = 151
# expect :grid_search to overstate by roughly 3%; raise nA or switch to
# :interpolate when the MPC is the object of interest. The summary reports which
# method produced the number.
```

## `hi/solve_history_independent_tax.jl:1506`

Anchor: `median_assets = interpolated_weighted_quantile(p.a_grid, stats.asset_mass, 0.5`

```
    # Mid-cumulative interpolation rather than StatsBase's weighted-quantile
    # convention, which is biased low on a coarse nonuniform grid holding a
    # discretized continuous distribution. See `interpolated_weighted_quantile`
    # in common/grids.jl for the measured comparison against a known median:
    # at nA = 151 StatsBase errs by 5.8% of the median and refinement does not
    # close the gap. Measured on this solver's own distribution at J = 39,
    # nA = 101, the two conventions differ by 4.6%, against a calibration
    # target of 0.0498. The infinite-horizon solvers have used this since they
    # were written; this brings the finite pair into line.
```

## `hi/solve_history_independent_tax.jl:1992`

Anchor: `if isfinite(residuals[i]) && isfinite(residuals[i + 1]) &&`

```
        # sign(0.0) is 0.0, so an exact zero at either end also brackets.
```
