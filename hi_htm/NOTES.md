# Measurement notes — `hi_htm`

Extracted from the source comments on 2026-09-28, before those comments were
rewritten in the terse Discrete_HA register. Each entry records a number that
settled a choice and is not recoverable from the code itself. Line numbers are
from the pre-rewrite files and will drift; the anchor line identifies the site.

## `hi_htm/calibrate_twoprice.jl:1`

Anchor: `using Dates`

```
# =============================================================================
# calibrate_twoprice.jl
#
# Calibrate the three financial / borrowing-limit parameters
#
#     qSav   (gross savings price,  a' >= 0)
#     qBorr  (gross borrowing price, a' <  0)
#     bbar   (borrowing-limit scale, bbar <= 0)
#
# so that the stationary cross-section produced by
# `solve_hi` matches three data moments:
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

## `hi_htm/calibrate_twoprice.jl:110`

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

## `hi_htm/calibrate_beta.jl:1`

Anchor: `# Reuses `solve_scalar`, `moments_from` and the solver entry points from the`

```
# =============================================================================
# calibrate_beta.jl
#
# A TWO-INSTRUMENT, TWO-TARGET calibration in which the three prices are GIVEN
# and the discount factor does the work:
#
#       qSav, qBorr, qGov  fixed by the caller,
#       beta               calibrated to the asset moment,
#       bbar               calibrated to the borrowing limit.
#
# The share of households with negative liquid assets is NOT targeted -- with
# the prices fixed there is no qBorr left to move it.
#
# -----------------------------------------------------------------------------
# WHY CALIBRATE BETA RATHER THAN A PRICE
# -----------------------------------------------------------------------------
# The one-price file solves the government-budget wedge by tying qSav = qBorr =
# qGov = q and then calibrating q to the asset moment. That works, but it spends
# the price on a wealth moment: the interest rate the model runs at is whatever
# the asset target dictates, and cannot be set to the rate the exercise wants.
# Here the assignment is reversed. The prices are inputs -- set them to 0.99,
# or to the same value everywhere for an exact telescoping, or to a measured
# spread -- and the asset moment is matched by the discount factor instead.
#
# The two files answer different questions. Use the one-price file when the
# interest rate is free and the asset moment must hold; use this one when the
# interest rate is part of the experiment and preferences absorb the residual.
#
# THE GOVERNMENT BUDGET IS NOT AUTOMATICALLY CLEAN HERE. The telescoping
# argument in the one-price header is a statement about prices, not about beta:
#
#   PV(Y - C) = PV(T) - A_1 - sum_j qGov^(j-1) (qGov - q_j) A_(j+1),
#
# so PV(Y - C) = G is the government budget ONLY when q_j = qGov at every state.
# Calibrating beta does nothing to that wedge. If the exercise needs the budget
# to be exact, pass qSav = qBorr = qGov; the result print reports the residual
# either way and says which case it is in.
#
# -----------------------------------------------------------------------------
# IDENTIFICATION
# -----------------------------------------------------------------------------
# Two blocks, swept in this order:
#
#   * bbar -> the borrowing-limit moment. bbar is the only parameter in the
#     numerator -bbar * E[exp(kappa + rho z)], so this block is very nearly
#     exact: one solve pins it down up to the second-order feedback through mean
#     labor income. It is swept first for that reason.
#
#   * beta -> the asset moment. A HIGHER beta is more patience and therefore
#     MORE saving, so the asset moment is INCREASING in beta. Note the sign: it
#     is the opposite of the q block in the one-price file, where a higher q is
#     a lower return and less saving.
#
# THE UPPER BRACKET IS AN ORDINARY BRACKET HERE. This is the FINITE-horizon
# port, and that changes the one thing the infinite-horizon file is most
# careful about.
#
# In `hiinf`/`hiinf_htm`, asset demand diverges as beta approaches qSav from
# below: the precautionary motive stops being offset by impatience and no
# stationary distribution exists. `beta_max` had to sit strictly below qSav,
# and far enough below that the cross-section could settle inside the forward
# pass -- the relaxation-time rule qSav - beta_max >= 3*qSav/maxAge, which gave
# beta_max <= 0.9851 at qSav = 0.99.
#
# NONE OF THAT APPLIES WITH A FINITE HORIZON. There is no stationary
# distribution to converge to and nothing to relax toward: the household lives
# J+1 periods and its assets are bounded by lifetime resources whatever beta/q
# is. beta > qSav is perfectly well defined here -- it just means the agent
# wants rising consumption and saves more over the life cycle. The only
# restriction is the model's own, 0 <= beta < 1, which `validate` enforces.
#
# So beta_min and beta_max are a plain search bracket, to be widened if the
# root is not inside them. A beta returning ON beta_max means the bracket was
# too narrow, not that the economics broke down -- which is the opposite of how
# to read the same event in the infinite-horizon files.
#
# -----------------------------------------------------------------------------
# THE ACCESS CHAIN IS GIVEN, NOT CALIBRATED
# -----------------------------------------------------------------------------
# `pSS` and `pHH` are taken from SETTINGS (or from base_kwargs) and held fixed
# through the search, exactly as the prices are. Two reasons, one of
# identification and one of arithmetic.
#
# IDENTIFICATION. This is a two-instrument, two-target calibration: bbar against
# the borrowing limit, beta against the asset ratio. The access chain adds a
# third free parameter and no third moment, so it cannot be calibrated here
# without a target. Under the SETTINGS default the chain is iid (pHH = 1 - pSS),
# which collapses it to the single parameter piH = pHH, and that parameter is
# set DIRECTLY to the data share rather than searched over -- it maps one for
# one onto `shareHandToMouth`, so a search would only rediscover the number
# already typed in. Separating pSS from pHH is a different exercise and needs a
# panel moment on exit from hand-to-mouth status, which no statistic here
# carries.
#
# ARITHMETIC. Hand-to-mouth households hold less, so the asset ratio falls in
# piH at a given beta and the calibrated beta has to RISE to hit the same
# target. The beta ceiling is therefore more likely to bind here than in
# `hiinf`, and it is not a number that can simply be raised: it is pinned by the
# relaxation-time rule below, so buying room means raising maxAge. A beta that
# comes back sitting on beta_max at piH > 0 may mean the asset target is not
# attainable with that many hand-to-mouth households at these prices -- which is
# a statement about the model, not a failure of the search. Re-run at piH = 0 to
# see whether the target was attainable without them.
#
# -----------------------------------------------------------------------------
# WHAT CHANGES WHEN BETA MOVES, BEYOND THE ASSET DISTRIBUTION
# -----------------------------------------------------------------------------
# Flow utility is normalized in the solver as
#
#   V = (1 - beta) * (log c - phi h^(1+eta)/(1+eta)) + beta * E[V'],
#
# so V is a per-period equivalent and the welfare simulation weights ages by
# (1 - beta) * beta^(age-1). Both the normalization and the age weights are
# functions of beta. Welfare numbers from two calibrations with DIFFERENT
# calibrated betas are therefore not the same functional evaluated at two
# points, and differencing them is not a welfare comparison. Within one
# calibrated beta -- comparing tax systems, say -- nothing changes.
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   include("calibrate_beta.jl")
#
#   # prices at their SETTINGS values (qSav 0.99, qBorr 0.97, qGov 0.99):
#   r = calibrate_beta(nZ = 15, nEps = 11,
#                                              nKappa = 5, nA = 151)
#
#   # one price everywhere, so the government budget telescopes exactly:
#   r = calibrate_beta(nZ = 15, nEps = 11, nKappa = 5,
#                                              nA = 151, qSav = 0.99,
#                                              qBorr = 0.99, qGov = 0.99)
#
#   # median instead of mean, and a tighter inner solve:
#   r = calibrate_beta(
#           calib = BetaCalibration(asset_moment = :median,
#                                   inner_xtol = 1e-7))
#
#   r.beta, r.bbar, r.moments.shareNegativeLiquidAssets
#
# Returns `(; beta, bbar, qSav, qBorr, qGov, eq, moments, residuals, converged,
#            stalled, sweeps, nSolves, elapsedSeconds, calib, params)`.
# =============================================================================
```

## `hi_htm/calibrate_beta.jl:163`

Anchor: `medianAssetsToMeanLaborIncome::Float64       = 0.0498`

```
    # Targets. Kaplan and Violante (2014), Table III, on a 2001 SCF cross-section
    # of households aged 22-59 with the top 5% by net worth dropped: mean net
    # LIQUID wealth over mean earnings-plus-benefits is 31,001/52,745 = 0.588,
    # and the median counterpart is 2,629/52,745 = 0.0498, which is the figure
    # carried by paper/Bewley.tex and by the three-instrument file. (Both read
    # 2,269 / 0.043 until 2026-09-23; that median was a transposition of the
    # 2,629 printed in the table, whose every other figure matched.)
    #
    # Do NOT put 62,442/52,745 = 1.184 here. That is median NET WORTH (it is
    # labelled `tgt_networthmedian_to_ymean` in code/old/.../parameters.jl and
    # sits commented out in paper/Bewley.tex). This is a one-asset liquid-wealth
    # model whose borrowing-limit target, 0.185, is itself a liquid object;
    # asking it for a net-worth median drives the search into the divergence
    # boundary and it will stall against the bracket cap.
```

## `hi_htm/calibrate_beta.jl:182`

Anchor: `beta_init::Float64 = 0.960`

```
    # Starting points. bbar_init is close to its root by construction: the
    # borrowing-limit block is linear in bbar, and -0.2 * 0.185/0.20895 was the
    # one-solve back-out at the production grids (retaken when the target moved
    # from 0.180 to 0.185; at 0.180 it read -0.17228836). beta_init is the
    # SETTINGS value, which is the natural neutral start.
```

## `hi_htm/model_settings.jl:11`

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

## `hi_htm/model_settings.jl:28`

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

## `hi_htm/model_settings.jl:57`

Anchor: `pSS = 0.8882970895,`

```
    # Asset-market access (psmodel.tex: s and h). pSS = Pr(stay saver),
    # pHH = Pr(stay hand-to-mouth). The stationary HtM share is
    #     piH = (1 - pSS) / (2 - pSS - pHH),
    # printed in the options header, and the initial cross-section is drawn
    # from it, so the HtM share is constant over the life cycle.
    #
    # CALIBRATED TO KAPLAN, VIOLANTE AND WEIDNER (2014), THEIR TABLE 4 -- the
    # printed SCF 2007-2009 two-year transition matrix across poor-HtM,
    # wealthy-HtM and non-HtM status, collapsed to two states (weighting the
    # two HtM rows by their ergodic mass) and annualized by taking the square
    # root of the second eigenvalue. Gives piH = 0.3166 against the 0.317 the
    # paper reports, and an expected HtM spell of 4.15 years against their
    # stated 3.5 (W-HtM) and 4.5 (P-HtM). Matches hiinf_htm and hdinf_htm.
    # See references/KVW2014_WealthyHandToMouth/.
    #
    #     pSS = 1.00,  pHH = 0.00   -> piH = 0   reproduces `hi` exactly
    #     pSS = 0.00,  pHH = 1.00   -> piH = 1   everyone hand-to-mouth
```

## `hi_htm/model_settings.jl:94`

Anchor: `aMax = 60.0,`

```
    # Raised from 15.0 for the median-NET-WORTH target (62442/52745 = 1.1838),
    # which puts mean assets near 2.1 times labour income -- five times the
    # liquid-wealth target -- and left 0.32 percent of the mass on a ceiling of
    # 15. `hi` still uses 15.0, so pass aMax explicitly to both when running the
    # hi / hi_htm nesting check.
```

## `hi_htm/solve.jl:119`

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

## `hi_htm/solve.jl:134`

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

## `hi_htm/solve.jl:279`

Anchor: `if p.a0 == 0.0 && !p.a0_scales_with_kappa`

```
    # The original requirement was that a_grid contain 0.0 exactly, because
    # the initial holding was hard-coded to zero. What actually matters is that
    # it lies inside the grid and is feasible for every type.
```

## `hi_htm/solve.jl:912`

Anchor: `@inbounds for i in eachindex(EVS_raw)`

```
        # The access shock is independent of (z', eps'), so the mixing is done
        # AFTER the expectation and the saver's solver sees a drop-in
        # replacement for EV. A state with no feasible choice carries the
        # FINITE sentinel, so these products are 0.0 when the weight is zero;
        # with -Inf the pSS = 1 corner would be NaN.
```

## `hi_htm/solve.jl:1605`

Anchor: `median_assets = interpolated_weighted_quantile(p.a_grid, stats.asset_mass, 0.5`

```
    # Mid-cumulative interpolation rather than StatsBase's weighted-quantile
    # convention, which is biased low on a coarse nonuniform grid holding a
    # discretized continuous distribution. See `interpolated_weighted_quantile`
    # in common/src/grids.jl for the measured comparison against a known median:
    # at nA = 151 StatsBase errs by 5.8% of the median and refinement does not
    # close the gap. Measured on this solver's own distribution at J = 39,
    # nA = 101, the two conventions differ by 4.6%, against a calibration
    # target of 0.0498. The infinite-horizon solvers have used this since they
    # were written; this brings the finite pair into line.
```

## `hi_htm/solve.jl:2064`

Anchor: `if isfinite(residuals[i]) && isfinite(residuals[i + 1]) &&`

```
        # sign(0.0) is 0.0, so an exact zero at either end also brackets.
```
