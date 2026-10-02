# Measurement notes — `hd_htm`

Extracted from the source comments on 2026-09-28, before those comments were
rewritten in the terse Discrete_HA register. Each entry records a number that
settled a choice and is not recoverable from the code itself. Line numbers are
from the pre-rewrite files and will drift; the anchor line identifies the site.

## `hd_htm/join_results.jl:28`

Anchor: `const RESULTS_DIR = length(ARGS) >= 1 ? ARGS[1] : "results"`

```
# Which files to merge. Edit MATCH to pick a sweep; the per-task files written
# by sweep.sh are all `sweep_mu2_n=1_...`, and the name now carries nS1, nS2,
# pSS and pHH, so any of them can be selected here. RESULTS_DIR and MATCH can
# also be overridden from the command line:
#     julia --project=. join_results.jl results "nS2=101"
```

## `hd_htm/join_results.jl:121`

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

## `hd_htm/model_settings.jl:1`

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
# theta0 is NOT set here. It is derived from (alpha, mu1, mu2, beta, J) by
#   theta0 = 1 / sum_{s=0}^{J} beta^s (alpha*mu1^s + (1-alpha)*mu2^s),
# the FINITE-horizon normalization, matching build_theta in the no-savings
# code, so theta0 depends on J as well as on alpha and the roots. Adding
# theta0 back as a setting is a MethodError, not a silent override.
#
# NOTE ON THE ROOTS. alpha lands in [0, 1] iff mu1 <= rho <= mu2, i.e. the
# roots BRACKET the income persistence rho = 0.958. An alpha outside [0, 1]
# still solves but gives a signed mixture and a warning.
```

## `hd_htm/model_settings.jl:35`

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
```

## `hd_htm/model_settings.jl:60`

Anchor: `alpha = :paper,`

```
    # mixture weight: a number, or :paper for (rho - mu1)/(mu2 - mu1).
    # :paper keeps it consistent with the roots above; replace with e.g.
    # alpha = 0.5 to set it independently.
```

## `hd_htm/model_settings.jl:68`

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

## `hd_htm/model_settings.jl:86`

Anchor: `a0 = 0.0,`

```
    # Initial asset holdings at model age 1. a0 = 0.0 reproduces the original
    # "born with nothing" condition exactly; a0_scales_with_kappa reads a0 as a
    # multiplier on exp(kappa).
```

## `hd_htm/model_settings.jl:117`

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

## `hd_htm/model_settings.jl:167`

Anchor: `pSS = 0.8882970895,`

```
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
```

## `hd_htm/plot_welfare.jl:1`

Anchor: `isdefined(@__MODULE__, :oneroot_welfare_curve) || include("../../sweep_noasset`

```
# =============================================================================
# plot_welfare.jl
#
# Overlays the Bewley (with-savings) mu2 sweep on the no-savings one-root
# welfare curve, so the two models can be read off one axis.
#
#   blue   no savings, one root      100*(W(mu) - W_hi)      <- from ../../sweep_noasset.jl
#   red -- no savings, unrestricted two-root optimum
#   red    Bewley hd sweep           100*(W .- W[1])         <- from join_results.jl
#
# Each curve's optimum is a large dot in that curve's colour, labelled in place
# with the root rather than in the legend.
#
# ---------------------------------------------------------------------------
# RUNNING IT: the two halves live in different environments
# ---------------------------------------------------------------------------
# The no-savings code needs Optim and PyPlot (code/julia/Project.toml); this
# folder needs JLD2 to read the sweep files (Bewley/hd/Project.toml). Neither
# environment has both, so STACK them rather than adding dependencies to
# either -- Julia searches LOAD_PATH in order:
#
#   cd .../code/julia/Bewley/hd
#   JULIA_LOAD_PATH="../..:.:@stdlib" julia -e 'include("plot_welfare.jl"); plot_welfare()'
#
# `../..` is code/julia (Optim, PyPlot), `.` is this folder (JLD2, Plots).
# Nothing is installed and no Project.toml changes.
#
# ---------------------------------------------------------------------------
# THE TWO CURVES DO NOT SHARE A BASELINE
# ---------------------------------------------------------------------------
# The no-savings curve is a gain over HISTORY INDEPENDENCE: it is anchored at
# zero because mu = 0 collapses the one-root kernel to theta = (1,0,0,...) and
# P = 1 exactly, so the benchmark is a genuine model, not a normalization.
#
# The Bewley curve is `100*(W .- W[1])`, a gain over the FIRST POINT OF ITS OWN
# GRID. That coincides with history independence only if the sweep starts at
# mu2 = 0. The meta3 sweeps start at mu2 = 0.4, in which case the green curve is
# anchored at an arbitrary interior point and its height is not comparable with
# the blue one -- only its SHAPE is. `plot_welfare` prints the first mu2 so this
# is visible rather than silent, and warns when it is not zero.
# =============================================================================
```

## `hd_htm/plot_welfare.jl:121`

Anchor: `ylo, yhi = ylim()`

```
    # Legend in the north-west, in the empty band between the dashed two-root
    # reference (2.98%) and the blue curve, which is still below 1.8% over the
    # left half of the axis. A small headroom keeps the box clear of the dashed
    # line; `figstyle` has no anchor argument, so the legend is placed here
    # rather than through it.
```

## `hd_htm/main.jl:114`

Anchor: `if abspath(PROGRAM_FILE) == abspath(@__FILE__)`

```
# Script entry point. Every keyword the REPL call accepts works here too:
#   julia -t 8 main.jl mu1=0 mu2=0.8343 alpha=0 nS2=101 \
#         s_grid_method=:quantile labor_grid_size=151
# The result is saved to results/ under a name built from the dimensions.
```

## `hd_htm/solve.jl:1`

Anchor: `module HistoryDependentTax`

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

## `hd_htm/solve.jl:152`

Anchor: `a0::Float64`

```
    # Initial asset holdings at model age 1. a0 = 0.0 is the original "born
    # with nothing" condition and remains the default. a0_scales_with_kappa
    # multiplies it by exp(kappa), matching how wages and the borrowing limit
    # already scale with the permanent type. a0 is placed by the same Young
    # lottery used for a', not snapped to the nearest node.
```

## `hd_htm/solve.jl:355`

Anchor: `M = [alpha * mu1^s + (1.0 - alpha) * mu2^s for s in 0:J]`

```
    # Promise-keeping restriction pins down theta0. The normalization is the
    # FINITE-horizon one,
    #
    #   sum_{s=0}^{J} beta^s theta_s = 1,   theta_s = theta0*M_s,
    #   M_s = alpha*mu1^s + (1-alpha)*mu2^s,
    #
    # written as the explicit sum so it matches build_theta in the no-savings
    # code (../../finite_horizon.jl) term for term. Closed form, for reference:
    #   sum = alpha*(1-(beta*mu1)^(J+1))/(1-beta*mu1)
    #       + (1-alpha)*(1-(beta*mu2)^(J+1))/(1-beta*mu2).
    #
    # The infinite-horizon version (dropping the (beta*mu)^(J+1) terms) is what
    # this file used previously; it overstates the sum and so understates
    # theta0, by 10% at J = 39 with mu2 near one. The finite sum is the right
    # one here because s1 and s2 start at 0 at age 0, so the kernel a household
    # actually faces reaches back at most j periods and is truncated anyway.
    #
    # mu = 0 contributes M_0 = 0^0 = 1 and M_s = 0 for s >= 1, so the mu1 =
    # mu2 = 0 limit still gives theta0 = 1 exactly.
```

## `hd_htm/solve.jl:620`

Anchor: `function F_stock(x::Float64)`

```
    # Age-pooled CDF, equal weight per age, over ages 1..J ONLY.
    #
    # Age 0 is deliberately excluded. It would enter as a point mass at s = 0
    # (s_0 = 0 for everyone), i.e. a JUMP in F of height (1-w)/(J+1) -- 0.0175 at
    # J = 39, w = 0.30. Quantile levels are spaced 1/n apart, so as soon as
    # n > (J+1)/(1-w) ~ 57 two or more levels land inside that jump and every one
    # of them inverts to exactly s = 0. The duplicated node then failed the
    # strict-monotonicity check and the whole quantile grid was discarded, which
    # is the "not strictly increasing" fallback this branch used to hit at
    # nS2 >= 101 -- precisely the grid sizes the method exists to serve.
    #
    # Excluding it costs nothing: the initial condition is represented on the
    # grid by the explicit snap-to-zero node below, so keeping the atom here
    # double-counted it. With ages 1..J the mixture is continuous and strictly
    # increasing on the support, so distinct levels give distinct nodes at any n.
```

## `hd_htm/solve.jl:645`

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

## `hd_htm/solve.jl:793`

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

## `hd_htm/solve.jl:1316`

Anchor: `@inbounds for i in eachindex(EVzS)`

```
                # The access shock is independent of (z', eps'), so the mixing
                # is done AFTER the expectation and the saver's block sees a
                # drop-in replacement for EVz. A state with no feasible choice
                # carries the FINITE sentinel, so these products are 0.0 when
                # the weight is zero; with -Inf the pSS = 1 corner -- the one
                # that has to reproduce `hd` exactly -- would be NaN.
```

## `hd_htm/sweep_mu2.jl:1`

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

## `hd_htm/sweep_mu2.jl:482`

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
