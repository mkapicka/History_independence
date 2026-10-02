# Measurement notes — `hdinf_htm`

Extracted from the source comments on 2026-09-28, before those comments were
rewritten in the terse Discrete_HA register. Each entry records a number that
settled a choice and is not recoverable from the code itself. Line numbers are
from the pre-rewrite files and will drift; the anchor line identifies the site.

## `hdinf_htm/join_results.jl:26`

Anchor: `const RESULTS_DIR = length(ARGS) >= 1 ? ARGS[1] : "results"`

```
# Which files to merge. Edit MATCH to pick a sweep; the per-task files written
# by sweep.sh are all `sweep_mu2_n=1_...`, and the name now carries nS1, nS2,
# pSS and pHH, so any of them can be selected here. RESULTS_DIR and MATCH can
# also be overridden from the command line:
#     julia --project=. join_results.jl results_ratios "nS2=151"
```

## `hdinf_htm/join_results.jl:119`

Anchor: `let bad = findall(x -> isfinite(x) && abs(x - piH) > 1e-6 * piH, HtM)`

```
# The tolerance is RELATIVE and deliberately loose. The realized share is not
# exact: `massTol` in the forward pass skips cells below 1e-14, and that
# truncation accumulates over the state space without cancelling between the
# two access states. Measured, the deviation scales with the number of cells --
# 1.4e-12 at 43,050 cells, 7.8e-10 at the 7,524,330 of a production grid
# (nA=151, nS2=151, nZ=15, nEps=11). An ABSOLUTE 1e-10, which this used, fires
# on that arithmetic noise. 1e-6 relative sits four orders above the noise and
# four below anything economically meaningful, so it still catches a genuine
# loss of access mass.
```

## `hdinf_htm/model_settings.jl:1`

Anchor: `const HD_SETTINGS = (;`

```
# Settings for the history-dependent tax model. This file is loaded INSIDE the
# HistoryDependentTax module by solve.jl and is fully
# standalone: it does not reference the history-independent SETTINGS. Edit the
# values here to change the model, grids, or solver; after editing, re-include
# solve.jl (the module is replaced, with a harmless
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
```

## `hdinf_htm/model_settings.jl:35`

Anchor: `mu1 = 0.6061,`

```
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
```

## `hdinf_htm/model_settings.jl:58`

Anchor: `alpha = :paper,`

```
    # mixture weight: a number, or :paper for (rho - mu1)/(mu2 - mu1).
    # :paper keeps it consistent with the roots above; replace with e.g.
    # alpha = 0.5 to set it independently.
```

## `hdinf_htm/model_settings.jl:63`

Anchor: `maxAge = 600,`

```
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
```

## `hdinf_htm/model_settings.jl:94`

Anchor: `a0 = 0.0,`

```
    # Initial asset holdings at model age 1. a0 = 0.0 reproduces the original
    # "born with nothing" condition exactly; a0_scales_with_kappa reads a0 as a
    # multiplier on exp(kappa), matching how wages and the borrowing limit
    # already scale with the permanent type.
```

## `hdinf_htm/model_settings.jl:100`

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

## `hdinf_htm/model_settings.jl:150`

Anchor: `hMin = 0.05,`

```
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
```

## `hdinf_htm/model_settings.jl:200`

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

## `hdinf_htm/recover_ratios.jl:3`

Anchor: `include(joinpath(@__DIR__, "main.jl"))`

```
# =============================================================================
# recover_ratios.jl -- add the ratio fields to sweep files that stored LEVELS.
#
# WHY THIS EXISTS. Sweeps written before `meanLaborIncome` was recorded carry
# `meanAssets` and `medianAssets` as levels only, and `join_results.jl` refuses
# them because the calibration targets RATIOS to mean labor income and the
# denominator moves across mu2. The obvious fix -- re-run the sweep -- costs
# what the sweep cost: 28.25 hours for one point at the production grids.
#
# WHAT THIS DOES INSTEAD. Each sweep file records `lambda`, and `settings`
# records every parameter the run overrode. So the equilibrium can be rebuilt
# with ONE solve at the stored lambda, skipping the lambda root-find that the
# original sweep paid for over and over. The saving is the number of lambda
# evaluations that point used.
#
# THE CROSS-CHECK IS THE POINT. `meanAssets` and `medianAssets` ARE stored, so
# the rebuilt equilibrium must reproduce them. If it does, the recovered
# denominator belongs to the same equilibrium and the ratios are exact. If it
# does not, some parameter that mattered is NOT in `settings` -- it fell back to
# the current HD_SETTINGS, which has moved -- and that point must be re-run. The
# check is per point and printed, never assumed.
#
# maxAge is the field to watch: it is NOT in the stored `settings` tuple, so it
# comes from whatever HD_SETTINGS says now.
#
# IN THIS DIRECTORY, so is the access chain for foreign files. `sweep_mu2.jl`
# here forces pSS/pHH into `settings`, so anything this directory wrote rebuilds
# exactly. A sweep written by `hdinf` records no chain, falls back to
# HD_SETTINGS -- which defaults to piH = 1/3 -- and is then a DIFFERENT model.
# The level cross-check catches that, but do not recover `hdinf` files here in
# the first place: use hdinf/recover_ratios.jl.
#
# USAGE
#   julia --project=. recover_ratios.jl [SRC_DIR] [LIMIT]
#
#   julia --project=. recover_ratios.jl results 1     # one point first, ALWAYS
#   julia --project=. recover_ratios.jl results       # then the rest
#
# Recovered files are written to SRC_DIR * "_ratios"; nothing is overwritten.
# `join_results.jl` then reads that directory.
# =============================================================================
```

## `hdinf_htm/recover_ratios.jl:54`

Anchor: `const LEVEL_TOL = 1e-8`

```
# Relative agreement demanded of the re-solved levels. The solve is
# deterministic, so a genuine match is at round-off; 1e-8 is loose enough to
# survive a thread-count change in the reduction order and tight enough that a
# different equilibrium cannot slip through.
```

## `hdinf_htm/main.jl:116`

Anchor: `if abspath(PROGRAM_FILE) == abspath(@__FILE__)`

```
# Script entry point. Every keyword the REPL call accepts works here too:
#   julia -t 8 main.jl mu1=0 mu2=0.8343 alpha=0 nS2=101 \
#         s_grid_method=:quantile labor_grid_size=151
# The result is saved to results/ under a name built from the dimensions.
```

## `hdinf_htm/solve.jl:1`

Anchor: `module HistoryDependentTaxInfinite`

```
# =============================================================================
# solve.jl  --  STANDALONE
#
# Finite-horizon Bewley economy with a HISTORY-DEPENDENT tax system
# (Section 1 of Bewley.tex). Budget constraint:
#
#   c + q(a') a' <= lambda * exp( pow * (z + eps + kappa
#                                        + alpha*s1 + (1-alpha)*s2) ) * h^pow + a,
#
# pow = (1 - tau) * theta0, and past-income stocks
#
#   s1' = mu1 * (z + eps + kappa + ln h + s1),
#   s2' = mu2 * (z + eps + kappa + ln h + s2),
#
# with theta0 implied by the FINITE-horizon promise-keeping restriction
#   sum_{s=0}^{J} beta^s theta_s = 1,  theta_s = theta0*(alpha*mu1^s + (1-alpha)*mu2^s),
# which is what build_theta imposes in the no-savings code. The mixture weight
# alpha is a free setting: a number, or :paper for the paper convention
#   alpha = (rho - mu1) / (mu2 - mu1),   alpha = 1 when mu1 = mu2.
# theta0 is derived and cannot be set. alpha lies in [0, 1] iff mu1 <= rho <=
# mu2; outside that the mixture is signed, which warns but is not rejected.
#
# mu1 = mu2 = 0 implies alpha = 1, theta0 = 1 and s1 = s2 = 0, reproducing the
# history-independent model of Section 1.1 exactly (use nS1 = nS2 = 1).
#
# -----------------------------------------------------------------------------
# This file is SELF-CONTAINED: it does not include or call the
# history-independent code. All shared infrastructure (shock discretization,
# asset/labor grids, statistics, the lambda solver) is replicated here inside
# the module `HistoryDependentTax`, so the two codebases can evolve
# independently and be loaded in the same session without name collisions.
# Only the HD-specific API is exported:
#
#   HDParams, HD_SETTINGS, make_history_dependent_params,
#   solve_hd, print_hd_equilibrium_summary,
#   check_history_independent_limit
#
# Usage:
#   include("solve.jl")   # also loads the HD settings file
#   using .HistoryDependentTax
#   p  = make_history_dependent_params()        # HD_SETTINGS + overrides
#   eq = solve_hd(p)
#
# Solution method: hours affect s' and are therefore intertemporal, so (a', h)
# are chosen JOINTLY on grids against a continuation value BILINEARLY
# interpolated in (s1', s2'); the distribution uses the matching bilinear
# Young (1990) lottery; infeasible states carry the finite sentinel
# VINFEASIBLE (not -Inf, which would create 0 * Inf = NaN in the
# interpolation); s' outside the grid is clamped and the clamped mass share is
# reported in eq.sClampedMassShare with a warning when material.
# =============================================================================
```

## `hdinf_htm/solve.jl:147`

Anchor: `age0_real::Int`

```
    # Cross-sectional statistics window, mirroring the history-independent
    # solver so the two report the same object. Real age = age0_real + model
    # age - 1. Kaplan and Violante (2014) Table 2 build their targets on a 2001
    # SCF cross-section of households aged 22-59, which at age0_real = 22 is
    # model ages 1-38. Set stats_age_lo = 1, stats_age_hi = maxAge to recover
    # the all-ages average this solver used to report as its only statistic.
```

## `hdinf_htm/solve.jl:157`

Anchor: `a0::Float64`

```
    # Initial asset holdings at model age 1. a0 = 0.0 is the original "born
    # with nothing" condition and remains the default, so results are unchanged
    # unless it is set. a0_scales_with_kappa multiplies a0 by exp(kappa),
    # matching how wages and the borrowing limit already scale with the
    # permanent type. a0 is placed by the same Young lottery used for a', not
    # snapped to the nearest node, so it stays exact between grid points.
```

## `hdinf_htm/solve.jl:375`

Anchor: `denom = alpha / (1.0 - beta * mu1) + (1.0 - alpha) / (1.0 - beta * mu2)`

```
    # Promise-keeping restriction pins down theta0. The normalization is the
    # INFINITE-horizon one,
    #
    #   sum_{s=0}^{inf} beta^s theta_s = 1,  theta_s = theta0*M_s,
    #   M_s = alpha*mu1^s + (1-alpha)*mu2^s,
    #
    # whose closed form is the geometric sum below. The hd (finite-J) solver
    # truncates this at s = J, which understates theta0 there by about 10% at
    # J = 39 with mu2 near one; with an infinitely lived agent the untruncated
    # sum is the correct one, and the two agree as J grows.
    #
    # mu = 0 contributes M_0 = 0^0 = 1 and M_s = 0 for s >= 1, so the mu1 =
    # mu2 = 0 limit still gives theta0 = 1 exactly.
```

## `hdinf_htm/solve.jl:630`

Anchor: `w = S_GRID_UNIFORM_BLEND`

```
    # DEFENSIVE MIXING with a uniform, exactly as in importance sampling. Pure
    # quantile spacing packs points so tightly around the mean that the outermost
    # cells become enormous -- at nS = 11 the bottom cell ran from -174 to -15 --
    # and the Young lottery then puts a share of any mass falling in that cell
    # onto the extreme node, from which s' leaves the grid and is clamped.
    # Measured: pure quantile spacing left 5.3e-3 clamped mass and was WORSE than
    # :linear at nS <= 15. Blending caps the widest cell at about
    # (s_hi - s_lo)/(S_GRID_UNIFORM_BLEND*nS) while keeping most of the
    # concentration. See the tuning table at S_GRID_UNIFORM_BLEND.
```

## `hdinf_htm/solve.jl:1431`

Anchor: `@inbounds for i in eachindex(EVzS)`

```
            # A state with no feasible choice carries the FINITE sentinel, so
            # these products are 0.0 when the weight is zero. With -Inf they
            # would be NaN, and the pSS = 1 / pHH = 0 corner -- the one that
            # has to reproduce hdinf exactly -- is where the weight is zero.
```

## `hdinf_htm/solve.jl:1592`

Anchor: `u_by_age = zeros(nAge)`

```
    # Flow utility per age, kept separately so the discounted sum can be closed
    # analytically past the settled age. Accumulating the discounted total
    # directly would silently truncate: breaking at age Jc drops a tail worth
    # beta^Jc of lifetime utility, which is 2.2e-3 at Jc = 150 -- far above the
    # 1e-17 agreement this check is supposed to demonstrate.
```

## `hdinf_htm/solve.jl:1875`

Anchor: `Jc = diag.settledAge`

```
    # The budget is STILL a present value over ages: the agent is infinitely
    # lived but the cohort's aggregates vary over its life, so this is not the
    # stationary condition a steady-state Bewley model would use.
    #
    # The path is iterated to maxAge for every kappa, after which Y_j - C_j is
    # constant and the remaining terms sum in closed form. Jc is maxAge now; it
    # used to be maximum(settled_by_kappa), which made the loop below read the
    # band where kappas had dropped out one at a time and the aggregates were
    # partial sums across kappa:
    #
    #   sum_{j>Jc} qGov^j (Y-C)_inf = (Y-C)_inf * qGov^(Jc+1) / (1 - qGov).
    #
    # Iterating instead would need ln(1e-6)/ln(0.99) ~ 1375 ages for the same
    # accuracy at qGov = 0.99.
```

## `hdinf_htm/sweep_mu2.jl:1`

Anchor: `using JLD2`

```
# =============================================================================
# sweep_mu2.jl
#
# Sweep the second history-dependence root mu2 over a grid, holding mu1 = 0,
# and collect for each grid point
#
#   eq.welfare.overallValueFunction   (ex-ante welfare, kappa-averaged)
#   eq.statistics.meanAssets
#   eq.statistics.medianAssets
#
# in vectors indexed by the grid point. The default grid is 10 points on
# [0, 0.98]; mu1 = 0 throughout, so alpha loads the mu1 block onto a stock
# that is identically zero and theta0 falls with mu2 through
#
#   theta0 = 1 / ( alpha/(1 - beta*mu1) + (1-alpha)/(1 - beta*mu2) ),
#
# from theta0 = 1 at mu2 = 0 down to theta0 = 0.1118 at mu2 = 0.98 (alpha = 0.5,
# beta = 0.96). The first grid point mu2 = 0 is the history-independent limit
# and reproduces the hi solver, so it doubles as a consistency check.
#
# -----------------------------------------------------------------------------
# Cost and accuracy notes
# -----------------------------------------------------------------------------
# * COST. With mu1 = 0 the s1-grid collapses to a single point (build_s_grid
#   returns [0.0] whenever mu == 0, regardless of nS1), so the state space is
#   nS1 = 1 rather than 7 and each solve is roughly a seventh of the cost of
#   the two-stock baseline. Still budget tens of minutes for the full sweep at
#   nA = 101, nS2 = 7, labor_grid_size = 101, and run with `julia -t auto`.
#
# * S-GRID RESOLUTION DEGRADES IN mu2. The s2-grid spans
#   [mu2*m_lo/(1-mu2), mu2*m_hi/(1-mu2)], so its width grows like mu2/(1-mu2):
#   a factor 0.43 at mu2 = 0.3 but 49 at mu2 = 0.98. At fixed nS2 = 7 the
#   spacing at the top of the grid is therefore about 114 times coarser than at
#   mu2 = 0.3. Treat the high-mu2 end as indicative, not converged, and re-run
#   the last few points with a larger nS2 before reading anything off them.
#   The reported sClampedMassShare is the diagnostic for the grid *bounds*, not
#   for this spacing problem — it can be negligible while the interpolation
#   error is not.
#
# * THE LAMBDA BRACKET MUST WIDEN WITH mu2. The budget-clearing lambda falls
#   steeply in mu2 — theta0 falls, so the income scale lambda*exp(pow*m)*h^pow
#   needs a much smaller level factor to balance the budget — and it leaves the
#   HD_SETTINGS bracket [0.20, 2.50] well before the top of the grid. At
#   mu2 = 0.98 the root is around 0.007, two orders of magnitude below
#   lambdaMin. This sweep therefore overrides lambdaMin to LAMBDA_MIN_SWEEP =
#   1e-3 by default; pass lambdaMin = ... to change it. This matters because
#   the solver does NOT throw when the bracket fails: it returns the best
#   evaluated equilibrium, which sits at the bracket endpoint with a large
#   budget residual and asset statistics that are pure artifact (at mu2 = 0.98
#   with the stock bracket, lambda pins at 0.200000 with residual -0.6 and a
#   median-assets figure an order of magnitude off).
#
# * CONVERGENCE IS CHECKED, NOT ASSUMED. Because of the above, every point is
#   flagged `converged[i] = abs(govBudgetResidual[i]) <= tolGovBudget`. Read
#   only the converged points; the summary marks the others and warns.
#
# * FAILURES ARE RECORDED, NOT FATAL. Each point runs inside a try/catch; a
#   point that throws records NaN in every output vector and its message in
#   `errors`, and the sweep continues.
#
# Usage:
#   include("sweep_mu2.jl")
#   sweep = sweep_mu2()                                  # 10 points on [0, 0.98]
#   sweep = sweep_mu2(mu2_min = 0.2, mu2_max = 0.9, n_mu2 = 20)   # set the grid
#   sweep = sweep_mu2(nS2 = 15)                          # set the stock grids
#   sweep = sweep_mu2(range(0, 0.9, length = 5))         # or pass a grid directly
#   save_mu2_sweep(sweep)
#
# As a script the arguments are positional, mu2_min mu2_max n_mu2 nS1 nS2:
#   julia -t auto sweep_mu2.jl                 # defaults
#   julia -t auto sweep_mu2.jl 0.2 0.9 20
#   julia -t auto sweep_mu2.jl 0.2 0.9 20 7 15
# =============================================================================
```

## `hdinf_htm/sweep_mu2.jl:483`

Anchor: `sweep = if any(contains('='), ARGS)`

```
    # Two argument styles, chosen by whether any token contains '='.
    #
    #   key=value  any keyword sweep_mu2 accepts, same as the REPL call:
    #                julia sweep_mu2.jl mu2_min=0.96 n_mu2=4 alpha=0.5 nS2=101
    #   positional mu2_min mu2_max n_mu2 nS1 nS2, a leading subset allowed:
    #                julia sweep_mu2.jl 0.98 0.98 1 7 21
    #
    # The positional form is kept because existing batch scripts use it.
```
