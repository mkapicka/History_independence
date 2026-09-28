# Measurement notes — `hiinf`

Extracted from the source comments on 2026-09-28, before those comments were
rewritten in the terse Discrete_HA register. Each entry records a number that
settled a choice and is not recoverable from the code itself. Line numbers are
from the pre-rewrite files and will drift; the anchor line identifies the site.

## `hiinf/calibrate_history_independent_tax_Claude.jl:1`

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

## `hiinf/calibrate_history_independent_tax_Claude.jl:110`

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

## `hiinf/calibrate_history_independent_tax_Claude.jl:126`

Anchor: `qSav_init::Float64  = 0.980681209802701`

```
    # Initial guesses for the instruments. These default to the SETTINGS values,
    # which is where the starting point used to come from -- and the ONLY place
    # it could come from, since qSav/qBorr/bbar are rejected in base_kwargs as
    # calibrated instruments. Overriding them here changes the starting point
    # without touching model_settings.jl:
    #
    #   calibrate_history_independent_tax(calib = CalibrationParams(bbar_init = -0.17707415))
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

## `hiinf/calibrate_history_independent_tax_beta.jl:1`

Anchor: `# Reuses `solve_scalar`, `moments_from` and the solver entry points from the`

```
# =============================================================================
# calibrate_history_independent_tax_beta.jl
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
#   include("calibrate_history_independent_tax_beta.jl")
#
#   # prices at their SETTINGS values (qSav 0.99, qBorr 0.97, qGov 0.99):
#   r = calibrate_history_independent_tax_beta(nZ = 15, nEps = 11,
#                                              nKappa = 5, nA = 151)
#
#   # one price everywhere, so the government budget telescopes exactly:
#   r = calibrate_history_independent_tax_beta(nZ = 15, nEps = 11, nKappa = 5,
#                                              nA = 151, qSav = 0.99,
#                                              qBorr = 0.99, qGov = 0.99)
#
#   # median instead of mean, and a tighter inner solve:
#   r = calibrate_history_independent_tax_beta(
#           calib = BetaCalibration(asset_moment = :median,
#                                   inner_xtol = 1e-7))
#
#   r.beta, r.bbar, r.moments.shareNegativeLiquidAssets
#
# Returns `(; beta, bbar, qSav, qBorr, qGov, eq, moments, residuals, converged,
#            stalled, sweeps, nSolves, elapsedSeconds, calib, params)`.
# =============================================================================
```

## `hiinf/calibrate_history_independent_tax_beta.jl:141`

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

## `hiinf/calibrate_history_independent_tax_beta.jl:160`

Anchor: `beta_init::Float64 = 0.960`

```
    # Starting points. bbar_init is close to its root by construction: the
    # borrowing-limit block is linear in bbar, and -0.2 * 0.185/0.20895 was the
    # one-solve back-out at the production grids (retaken when the target moved
    # from 0.180 to 0.185; at 0.180 it read -0.17228836). beta_init is the
    # SETTINGS value, which is the natural neutral start.
```

## `hiinf/calibrate_history_independent_tax_beta.jl:168`

Anchor: `beta_min::Float64 = 0.900`

```
    # Brackets. beta_max is the ceiling, used exactly as given. It must sit
    # strictly below the qSav in force; the header's relaxation-time rule says
    # how far below. The default is sized for the SETTINGS qSav of 0.99, where
    # it asks for 198 periods against maxAge = 600. A lower qSav needs a lower
    # beta_max and says so rather than being capped silently.
```

## `hiinf/calibrate_history_independent_tax_oneprice.jl:1`

Anchor: `# Reuses `solve_scalar`, `moments_from` and the solver entry points from the`

```
# =============================================================================
# calibrate_history_independent_tax_oneprice.jl
#
# A TWO-INSTRUMENT, TWO-TARGET calibration in which households face a single
# intertemporal price:
#
#       qSav = qBorr = qGov = q,
#
# with q calibrated to the asset moment and bbar to the borrowing limit. The
# share of households with negative liquid assets is NOT targeted -- with one
# price there is no qBorr left to hit it.
#
# -----------------------------------------------------------------------------
# WHY ONE PRICE
# -----------------------------------------------------------------------------
# In the three-instrument calibration the household's saving and borrowing
# prices both differ from the rate the government discounts at, and that wedge
# does not net out. Substituting the household budget into the resource
# constraint and discounting at qGov,
#
#   PV(Y - C) = PV(T) - A_1 - sum_j qGov^(j-1) (qGov - q_j) A_(j+1),
#
# so the asset terms telescope to the initial position A_1 ONLY when q_j = qGov
# at every state. Otherwise the solver, which drives PV(Y - C) to G, is not
# imposing PV(T) = G: it is imposing PV(T) = G plus a transfer to households
# that no agent in the model finances. Measured at the three-instrument
# calibration (qSav = 0.9807, qBorr = 0.9895, qGov = 0.99), that residual claim
# was worth 0.45 in present value, about 0.5 percent of lifetime output -- the
# same order as the welfare gains from history dependence the project measures,
# and five orders above the 1e-8 budget tolerance.
#
# Setting q = qGov removes it identically. The telescoping is then exact, and
# with a_0 = 0 the condition the solver imposes IS the government budget:
#
#   PV(Y - C) = G   <=>   PV(T) = G.
#
# The price is calibrated rather than fixed at 0.99 so the asset moment is still
# matched: with only one instrument left for the asset distribution, q has to do
# the job qSav did before.
#
# -----------------------------------------------------------------------------
# WHAT IS GIVEN UP
# -----------------------------------------------------------------------------
# The borrowing share is no longer matched, and should be reported rather than
# assumed. One price means no spread between lending and borrowing, so the model
# has nothing left to generate the observed mass of households at negative
# liquid wealth beyond what the borrowing limit and the shock process imply. In
# the three-instrument calibration that moment was 0.260 by construction; here
# it comes out wherever it comes out, and a large miss is informative about the
# one-price assumption rather than a failure of the search.
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
#   * q    -> the asset moment. q is the price of one unit of next-period
#     assets, so the gross return is 1/q and a HIGHER q means a LOWER return and
#     LESS saving. Measured at reduced grids with the borrowing limit held
#     fixed: q = 0.99 gave mean assets / mean labor income = 0.296, q = 0.97
#     gave 1.184, q = 0.95 gave 4.273. The moment is therefore DECREASING in q,
#     which is the opposite of what a comment in the three-instrument file
#     asserts; the numbers above are what this bracket is set from.
#
# The q bracket stops well above beta = 0.96. As q falls towards beta the return
# 1/q approaches 1/beta, the precautionary motive stops being offset by
# impatience, and asset demand diverges; at q = 0.93 the lambda solve failed
# outright and mean assets came back non-monotone. Keeping q_min at 0.962 stays
# inside the region where the asset moment is monotone in q.
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   include("calibrate_history_independent_tax_oneprice.jl")
#   r = calibrate_history_independent_tax_oneprice(nZ = 15, nEps = 11,
#                                                  nKappa = 5, nA = 151)
#   r.q, r.bbar, r.moments.shareNegativeLiquidAssets
#
#   # median instead of mean, and a tighter inner solve:
#   r = calibrate_history_independent_tax_oneprice(
#           calib = OnePriceCalibration(asset_moment = :median,
#                                       inner_xtol = 1e-7))
#
# Returns `(; q, bbar, eq, moments, residuals, converged, sweeps, nSolves,
#            elapsedSeconds, calib, params)`.
# =============================================================================
```

## `hiinf/calibrate_history_independent_tax_oneprice.jl:111`

Anchor: `medianAssetsToMeanLaborIncome::Float64       = 0.0498`

```
    # Targets. Kaplan and Violante (2014), Table III: mean net liquid wealth over
    # mean earnings-plus-benefits, 31,001/52,745 = 0.588, on a 2001 SCF
    # cross-section of households aged 22-59 with the top 5% by net worth
    # dropped. The median counterpart from the same table is 2,629/52,745 =
    # 0.0498. The other calibration files carried 0.043 here, a transposition
    # of the 2,629; they were brought into line on 2026-09-23.
```

## `hiinf/calibrate_history_independent_tax_oneprice.jl:122`

Anchor: `q_init::Float64    = 0.985`

```
    # Starting points. bbar_init is close to its root by construction: the
    # borrowing-limit block is linear in bbar, and -0.2 * 0.185/0.20895 was the
    # one-solve back-out at the production grids (retaken when the target moved
    # from 0.180 to 0.185; at 0.180 it read -0.17228836).
```

## `hiinf/calibrate_history_independent_tax_oneprice.jl:129`

Anchor: `q_min::Float64    = 0.950`

```
    # Brackets. See the identification note in the header for why q_min sits
    # above beta rather than at the 0.900 the three-instrument file uses.
```

## `hiinf/model_settings.jl:17`

Anchor: `tolDriftRel = Inf,`

```
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
```

## `hiinf/model_settings.jl:41`

Anchor: `a0 = 0.0,`

```
    # Initial asset holdings at model age 0. a0 = 0.0 reproduces the original
    # "born with nothing" condition exactly. Set a0_scales_with_kappa = true to
    # read a0 as a multiplier on exp(kappa), matching how wages and the
    # borrowing limit already scale with the permanent type.
```

## `hiinf/model_settings.jl:48`

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

## `hiinf/solve_history_independent_tax.jl:92`

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

## `hiinf/solve_history_independent_tax.jl:101`

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

## `hiinf/solve_history_independent_tax.jl:112`

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

## `hiinf/solve_history_independent_tax.jl:267`

Anchor: `if p.a0 == 0.0 && !p.a0_scales_with_kappa`

```
    # The original requirement was that a_grid contain 0.0 exactly, because the
    # initial condition was hard-coded to zero. What actually matters is that
    # the initial holding lies inside the grid and is feasible for every type.
```

## `hiinf/solve_history_independent_tax.jl:556`

Anchor: `@printf("settled assets             = %.8f\n", eq.A[end])`

```
    # There is no terminal age here, so A[end] is the SETTLED level the profile
    # is carried forward at, not a terminal condition. Before the profiles were
    # padded this printed 0.0 -- the untouched tail of `zeros(maxAge)`.
```

## `hiinf/solve_history_independent_tax.jl:675`

Anchor: `converged || return nothing`

```
    # A solve whose lambda never converged is not an equilibrium, so its
    # settling behaviour is not informative -- and the lambda failure is already
    # reported by the solver. Stacking a second warning on top buries the one
    # that matters. Observed: a calibration probe at qSav = 0.911 (floor 0.900)
    # where no lambda balances the budget, bottoming out at lambda = 0.018
    # against an equilibrium ~1.01, warned twice for one underlying problem.
```

## `hiinf/solve_history_independent_tax.jl:1239`

Anchor: `u_by_age = zeros(nAge)`

```
    # Flow utility per age, so the discounted sum can be closed analytically
    # past the settled age. Accumulating the discounted total directly would
    # silently truncate: stopping at age Jc drops a tail worth beta^Jc of
    # lifetime utility, 2.2e-3 at Jc = 150.
```

## `hiinf/solve_history_independent_tax.jl:1370`

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

## `hiinf/solve_history_independent_tax.jl:1526`

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

## `hiinf/solve_history_independent_tax.jl:2027`

Anchor: `if isfinite(residuals[i]) && isfinite(residuals[i + 1]) &&`

```
        # sign(0.0) is 0.0, so an exact zero at either end also brackets.
```
