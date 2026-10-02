# Measurement notes — `hiinf_htm`

Extracted from the source comments on 2026-09-28, before those comments were
rewritten in the terse Discrete_HA register. Each entry records a number that
settled a choice and is not recoverable from the code itself. Line numbers are
from the pre-rewrite files and will drift; the anchor line identifies the site.

## `hiinf_htm/calibrate_twoprice.jl:1`

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

## `hiinf_htm/calibrate_twoprice.jl:110`

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

## `hiinf_htm/calibrate_twoprice.jl:126`

Anchor: `qSav_init::Float64  = 0.980681209802701`

```
    # Initial guesses for the instruments. These default to the SETTINGS values,
    # which is where the starting point used to come from -- and the ONLY place
    # it could come from, since qSav/qBorr/bbar are rejected in base_kwargs as
    # calibrated instruments. Overriding them here changes the starting point
    # without touching model_settings.jl:
    #
    #   calibrate_twoprice(calib = CalibrationParams(bbar_init = -0.17707415))
    #
    # bbar_init is the one worth setting deliberately. Block (ii) is EXACTLY
    # linear in bbar -- the true limit is -bbar * E[exp(kappa + rho*z)] and bbar
    # enters nothing else in that moment -- so a single solve pins it down:
    # at bbar = -0.2 the windowed ratio is 0.20895201 against a target of 0.185,
    # giving -0.2 * 0.185/0.20895201 = -0.17707415. Measured at nZ=15, nEps=11,
    # nKappa=5, nA=151; it moves little with the grid, and as a starting point
    # it only needs to be close. The old default of -0.2 was 14% away.
    # qSav_init/qBorr_init are calibrated values carried over from a previous
    # run, not the SETTINGS defaults (0.99 / 0.97): starting the search at a
    # near-root saves sweeps, and qSav is the instrument that costs the most
    # solves. They were calibrated against the ALL-AGES statistic, so with the
    # ages 3-40 window the mean asset ratio is 4.3% lower and the qSav root sits
    # somewhat above this -- still a far better start than 0.99.
```

## `hiinf_htm/calibrate_beta.jl:1`

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
# THE UPPER BRACKET IS NOT A FREE CHOICE. Asset demand diverges as beta*(1/qSav)
# approaches one, that is as beta approaches qSav from below: the precautionary
# motive stops being offset by impatience and the stationary distribution walks
# off the top of the grid. This is the same boundary the one-price file runs
# into from the other side, where q_min = 0.962 sits just above beta = 0.960.
# `beta_max` IS that ceiling, set directly. The only thing enforced is that it
# lies strictly below the qSav actually in force, so a beta_max carried over
# from a run at a different price is an error rather than a silent divergence.
# If the calibrated beta comes back sitting on beta_max, the asset target is not
# attainable at the given prices; that is information about the prices, not a
# failure of the search.
#
# HOW FAR BELOW qSav TO SET IT. The cross-section relaxes toward its stationary
# distribution on a timescale
#
#   1/(1 - beta/qSav) = qSav / (qSav - beta),
#
# so the horizon the ceiling asks for is qSav/(qSav - beta_max) periods, against
# a forward pass of maxAge. A beta_max within 0.002 of qSav wants roughly 500
# periods and at maxAge = 600 the cross-section provably has not settled -- that
# is `warn_if_unsettled`'s drift warning made structural rather than marginal.
# Keeping three relaxation times inside the horizon needs
#
#   qSav - beta_max  >=  3 * qSav / maxAge,
#
# which is beta_max <= 0.9851 at the SETTINGS qSav = 0.99, and beta_max <= 0.9787
# at qSav = 1/1.0167. The search header prints the implied relaxation time and
# flags it when it is short, so this is judged per run rather than guessed.
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

## `hiinf_htm/calibrate_beta.jl:169`

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

## `hiinf_htm/calibrate_beta.jl:188`

Anchor: `beta_init::Float64 = 0.960`

```
    # Starting points. bbar_init is close to its root by construction: the
    # borrowing-limit block is linear in bbar, and -0.2 * 0.185/0.20895 was the
    # one-solve back-out at the production grids (retaken when the target moved
    # from 0.180 to 0.185; at 0.180 it read -0.17228836). beta_init is the
    # SETTINGS value, which is the natural neutral start.
```

## `hiinf_htm/calibrate_beta.jl:196`

Anchor: `beta_min::Float64 = 0.900`

```
    # Brackets. beta_max is the ceiling, used exactly as given. It must sit
    # strictly below the qSav in force; the header's relaxation-time rule says
    # how far below. The default is sized for the SETTINGS qSav of 0.99, where
    # it asks for 198 periods against maxAge = 600. A lower qSav needs a lower
    # beta_max and says so rather than being capped silently.
```

## `hiinf_htm/model_settings.jl:17`

Anchor: `tolDriftRel = Inf,`

```
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
```

## `hiinf_htm/model_settings.jl:37`

Anchor: `htmClipWarnShare = Inf,`

```
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
```

## `hiinf_htm/model_settings.jl:65`

Anchor: `a0 = 0.0,`

```
    # Initial asset holdings at model age 0. a0 = 0.0 reproduces the original
    # "born with nothing" condition exactly. Set a0_scales_with_kappa = true to
    # read a0 as a multiplier on exp(kappa), matching how wages and the
    # borrowing limit already scale with the permanent type.
```

## `hiinf_htm/model_settings.jl:72`

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

## `hiinf_htm/model_settings.jl:95`

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
```

## `hiinf_htm/solve.jl:141`

Anchor: `tolDriftRel::Float64             # warn when drift/|Y[end]| exceeds this`

```
    # The WARNING threshold, relative to Y. tolDist is an absolute bound on a
    # sum of two aggregate differences, which is uninterpretable without the
    # scale: a drift of 1.5e-09 against Y = 0.927 fails an absolute 1e-10 test
    # while being nine orders below the quantity it measures, and lambda and W
    # were shown invariant to 15 digits across a doubled maxAge in exactly that
    # case. A genuinely unsettled path (maxAge = 40) instead sits at 1e-04 to
    # 1e-03 of Y, so a relative threshold separates the two cleanly. Set to Inf
    # to silence the warning entirely; tolDist still drives convergedAge.
```

## `hiinf_htm/solve.jl:154`

Anchor: `a0::Float64                      # initial assets (level, or scale if below)`

```
    # Initial asset holdings at model age 0. a0 = 0.0 is the original condition
    # (everyone born with nothing) and remains the default, so results are
    # unchanged unless it is set. a0_scales_with_kappa multiplies a0 by
    # exp(kappa), matching how wages and the borrowing limit
    # (-bbar*exp(kappa + rho*z)) already scale with the permanent type: a flat
    # a0 would otherwise leave the lowest-kappa household starting relatively
    # far richer than the highest. a0 is placed on the grid by the same Young
    # lottery used for a', not snapped to the nearest node, so it stays exact
    # between grid points.
```

## `hiinf_htm/solve.jl:165`

Anchor: `age0_real::Int                   # real age at model age 1 (birth)`

```
    # MODEL AGE is the 1-based period index: model age 1 is the first simulated
    # period, at real age `age0_real`. Real age = age0_real + model age - 1.
    # The model is born AT age0_real, so nothing before it is simulated.
    #
    # Cross-sectional statistics are averaged over model ages stats_age_lo to
    # stats_age_hi inclusive, not over the whole forward pass. The calibration
    # target is Kaplan-Violante (2014) Table 2: mean net liquid wealth over mean
    # earnings-plus-benefits, 31,001/52,745 = 0.588, computed on a 2001 SCF
    # cross-section of households aged 22-59 with the top 5% by net worth
    # dropped. With age0_real = 22 that window is model ages 1-38.
    # Averaging over all maxAge ages instead made the statistic depend on
    # maxAge (0.32034 at 600 against 0.32141 at 1200) and mixed in ages the
    # data moment excludes.
```

## `hiinf_htm/solve.jl:330`

Anchor: `if p.a0 == 0.0 && !p.a0_scales_with_kappa`

```
    # The original requirement was that a_grid contain 0.0 exactly, because the
    # initial condition was hard-coded to zero. What actually matters is that
    # the initial holding lies inside the grid and is feasible for every type.
```

## `hiinf_htm/solve.jl:625`

Anchor: `@printf("settled assets             = %.8f\n", eq.A[end])`

```
    # There is no terminal age here, so A[end] is the SETTLED level the profile
    # is carried forward at, not a terminal condition. Before the profiles were
    # padded this printed 0.0 -- the untouched tail of `zeros(maxAge)`.
```

## `hiinf_htm/solve.jl:791`

Anchor: `converged || return nothing`

```
    # A solve whose lambda never converged is not an equilibrium, so its
    # settling behaviour is not informative -- and the lambda failure is already
    # reported by the solver. Stacking a second warning on top buries the one
    # that matters. Observed: a calibration probe at qSav = 0.911 (floor 0.900)
    # where no lambda balances the budget, bottoming out at lambda = 0.018
    # against an equilibrium ~1.01, warned twice for one underlying problem.
```

## `hiinf_htm/solve.jl:1075`

Anchor: `@inbounds for i in eachindex(EV_S)`

```
        # A state with no feasible choice carries the FINITE sentinel, so these
        # products are 0.0 when the weight is zero. With -Inf they would be
        # NaN, and the pSS = 1 / pHH = 0 corner -- the one that has to
        # reproduce hiinf exactly -- is precisely where the weight is zero.
```

## `hiinf_htm/solve.jl:1566`

Anchor: `u_by_age = zeros(nAge)`

```
    # Flow utility per age, so the discounted sum can be closed analytically
    # past the settled age. Accumulating the discounted total directly would
    # silently truncate: stopping at age Jc drops a tail worth beta^Jc of
    # lifetime utility, 2.2e-3 at Jc = 150.
```

## `hiinf_htm/solve.jl:1736`

Anchor: `if age > 1`

```
        # Record where the cross-section settles, but DO NOT stop here. An
        # earlier version broke out of the loop at this point, which was a
        # speed optimization that corrupted everything downstream of the
        # profiles:
        #
        #   * C/H/Y/A keep their `zeros(maxAge)` initialization past the break,
        #     so the age profiles fell off a cliff to exactly 0.0 -- array
        #     initialization presented as model output;
        #   * each kappa settles at its own age and the aggregates are summed
        #     ACROSS kappa, so ages between the earliest and latest settled age
        #     held partial sums over only the kappas still running. The
        #     government budget read exactly that band, since it sums to Jc;
        #   * `discounted_sum` walks the whole maxAge-long array, so
        #     consumptionPV and outputPV were summing that zero tail;
        #   * worst, the `stats` accumulator also stopped, so each kappa
        #     contributed settled_age ages of mass and meanAssets came out
        #     weighted by settled age rather than by Pkappa. total_mass would
        #     be an integer if every kappa ran the same number of ages; it came
        #     out at 417.166667, which is the tell.
        #
        # Running every kappa the full maxAge fixes all four at once and costs
        # only the ages past settlement in the forward pass, which is small
        # next to the VFI. `converged_age` is kept purely as a diagnostic, so
        # an undersized maxAge is visible rather than silent.
```

## `hiinf_htm/solve.jl:1901`

Anchor: `diagnostics = (; vIters = vIters_by_kappa, vGap = vGap_by_kappa,`

```
    # Every kappa now runs the full maxAge, so the PV tail opens at maxAge for
    # all of them and `settledAge` is no longer a per-kappa quantity.
    # `convergedAge` is diagnostic: 0 for a kappa means the cross-section had
    # NOT settled by maxAge, in which case the closed-form tail rests on an
    # assumption the path has not yet earned and maxAge should be raised.
    # Report the ACHIEVED drift rather than a pass/fail on tolDist. Failing an
    # absolute 1e-10 test says little on its own: what the closed-form tail
    # needs is that (Y - C) has stopped moving, and a drift of 3e-10 and one of
    # 3e-4 are worlds apart while both "fail". Printing the number lets the
    # magnitude be judged. convergedAge = 0 means that kappa never got under
    # tolDist at any age.
    # The settling check is NOT emitted here. This routine runs once per lambda
    # probe, and the lambda root-finder deliberately visits degenerate corners:
    # observed warnings carried lambda = 0.01 (= lambdaMin) at qSav = 0.911 with
    # Y[end] = 0.155 against a normal 0.927. Those probes say nothing about the
    # answer and drowned the one solve that matters. The drift is stored in
    # `diagnostics.finalDrift` instead and judged once, on the FINAL
    # equilibrium, in `attach_elapsed`.
```

## `hiinf_htm/solve.jl:2405`

Anchor: `if isfinite(residuals[i]) && isfinite(residuals[i + 1]) &&`

```
        # sign(0.0) is 0.0, so an exact zero at either end also brackets.
```
