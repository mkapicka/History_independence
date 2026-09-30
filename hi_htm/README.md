# `hi_htm` — finite-horizon history-independent tax with hand-to-mouth agents

Finite-horizon (`J`) model of `psmodel.tex`, section "History independent tax",
with the exogenous asset-market-access shock added. `hi` is left untouched;
this directory is a copy with the extension patched in, following the same
sequence used to build `hiinf_htm` from `hiinf`.

## The model

Two exogenous access states, `Pr(S'=S|S) = pSS`, `Pr(H'=H|H) = pHH` (`s` and
`h` in psmodel.tex). SAVERS solve the `hi` problem with a mixed continuation:

    V^S_j(a,x) = max (1-beta)(log c - phi h^(1+eta)/(1+eta))
                     + beta E[ pSS V^S_{j+1}(a',x') + (1-pSS) V^H_{j+1}(a',x') | x ]
    s.t.  c + q(a') a' = lambda (e^{kappa+z+eps} h)^{1-tau} + a,
          a' >= bbar e^{kappa + rho z}.

HAND-TO-MOUTH households face the same budget with `a'` imposed:

    a' = a/qSav  (a >= 0),   a' = a  (a < 0),
    c  = lambda (e^{kappa+z+eps} h)^{1-tau} + 1_{a<0} (1-qBorr) a.

The access chain is independent of `(z, eps)` and of the asset choice, and
starts at its stationary distribution `piS = (1-pHH)/(2-pSS-pHH)`,
`piH = (1-pSS)/(2-pSS-pHH)`, so the hand-to-mouth share is constant over the
life cycle.

## The finite-horizon wrinkle

`psmodel.tex` is written for an infinitely lived agent, so it never says what a
hand-to-mouth household does at the LAST age, where `hi` imposes `a' >= 0` so
nobody dies in debt. The rule `a' = a` would violate that for a debtor.

**At the terminal age only, the rollover is clamped: `a' = max(htm rule, 0)`.**
A terminal hand-to-mouth debtor therefore settles up, consuming `y + a` rather
than `y + (1-qBorr)a`. This is a modelling choice made here, not one the tex
dictates, and it is the single substantive difference from `hiinf_htm`.
`htm_transition(p; terminal = true)` builds the clamped variant, and because
its `cash` differs, the static HtM payoff is precomputed twice — once for the
ordinary ages and once for the last.

## Running it

    julia --project=. run_history_independent_tax.jl
    julia --project=. -e 'include("run_history_independent_tax.jl");
                          run_history_independent_tax(pSS = 1.0, pHH = 0.0)'

`SETTINGS` default to `pSS = 0.8882970895, pHH = 0.7589294614`, calibrated to
Kaplan, Violante and Weidner (2014) Table 4 and matching `hiinf_htm` and
`hdinf_htm`: `piH = 0.3166` with an expected hand-to-mouth spell of 4.15 years.
`pSS = 1, pHH = 0` switches the extension off and reproduces `hi`. See
`references/KVW2014_WealthyHandToMouth/`.

## What changed against `hi`

The saver's `solve_policy_age_grid_search!` and `solve_policy_age_interpolated!`
are untouched; they receive `EVmixS` in place of `EV`.

* `access_stationary_distribution`, and `pSS`/`pHH`/`piS`/`piH` on `HIParams`.
* `VINFEASIBLE = -1e18` replaces the bare `-Inf` in the saver's maximizer.
  `hi` can use `-Inf` safely because its value function is only swept
  backwards; here the access states are MIXED and `0 * (-Inf) = NaN`, and the
  `pSS = 1` corner -- the one the nesting test rests on -- is exactly where a
  weight is zero.
* `HTMTransition` / `htm_transition`: the exogenous `a'`, its
  `cash = a - q(a')a'`, and the Young lottery placing `a'` on the asset grid.
  The value function and the forward pass read the SAME weights.
* `precompute_htm_payoffs` / `update_htm_values!`: the HtM choice is STATIC
  (`a'` exogenous, nothing else linking periods), so its flow payoff is
  computed once per `(lambda, kappa)` and reused at every age, and its half of
  the recursion never maximizes. `hdinf_htm` has no such shortcut, because
  there hours move the past-income stocks.
* `solve_policies_for_kappa` recurses on the PAIR `(V^S, V^H)`. The access
  shock is independent of `(z', eps')`, so the mixing happens AFTER the
  expectation: `EVmixS = pSS E[V^S] + (1-pSS) E[V^H]` and
  `EVmixH = pHH E[V^H] + (1-pHH) E[V^S]`.
* `simulate_kappa!` carries a fourth distribution axis (1 = S, 2 = H). The
  forward pass uses `htm.cash[ia]` for consumption rather than recomputing
  `a - q(a')a'`, which agrees analytically but carries rounding noise on the
  positive branch where it must be exactly zero.
* Statistics gain `shareHandToMouth`, printed against the `piH` it must equal.

## Verification

All at `J=39, nA=41, nZ=3, nEps=3, nKappa=2, labor_grid_size=25, hMin=0.05`.

1. **Nesting.** At `pSS=1, pHH=0` every reported quantity -- lambda, both PVs,
   both asset ratios, the two shares and both welfare numbers -- matches `hi`
   to all 14 printed digits, lambda included. (`hiinf_htm` differs from `hiinf`
   by 1 ulp on lambda; the finite recursion has no such reassociation.)
2. **Closed form at `pSS=0, pHH=1`.** Everyone is hand-to-mouth and, from
   `a0 = 0`, stays at `a = 0` forever, so the model collapses to a static
   problem with `h* = ((1-tau)/phi)^(1/(1+eta)) = 0.94316422722837`. Measured:
   `max_j |H_j - h*| = 4.4e-16`, mean assets exactly `0.0`, and
   `PV(Y) - PV(C) = 7.1e-15`.
3. **Access shares.** `shareHandToMouth` reproduces `piH` exactly at
   `piH = 0, 0.3166, 1/3, 1`.
4. **Value function against simulation.** The independent forward-pass welfare
   matches the birth value function to 3.3e-16 at `piH = 0`, 5.0e-16 at
   `piH = 1`, and 3.9e-14 / 6.2e-14 in the interior. Being finite-horizon,
   there is no `beta^maxAge` truncation to absorb, so these are far tighter
   than the infinite-horizon solvers' 1e-4.

| case | piH | meanA/Y | welfare |
|------|-----|---------|---------|
| no HtM      | 0.000000 | 0.195310 | -0.39515692 |
| KVW default | 0.316642 | 0.112685 | -0.40347862 |
| iid piH=1/3 | 0.333333 | 0.118979 | -0.40414941 |
| all HtM     | 1.000000 | 0.000000 | -0.41881234 |

Note the KVW row holds a SMALLER hand-to-mouth share than the iid row
(0.3166 against 0.3333) yet ends with LOWER assets (0.1127 against 0.1190).
Persistence, not just incidence, is what destroys accumulation: the KVW chain
has second eigenvalue 0.647 and an expected spell of 4.15 years against the
iid chain's 0 and 1.50 years, so each unit of hand-to-mouth mass freezes
assets for longer. That is the substantive argument for dropping the iid
restriction.

## Problems

1. **The terminal rollover rule is a choice, not a derivation.** See above.
   It affects only the last age, but a terminal debtor's consumption jumps
   from `y + (1-qBorr)a` to `y + a`.
2. **`a' = a/qSav` has no fixed point above zero**, as in `hiinf_htm`: balances
   grow at the gross return with nothing decumulating. With `J = 39` the
   horizon is far too short for this to bite -- no mass reached the upper bound
   in any run here -- but it is the same defect, and it is what made 44% of the
   all-ages mass pile up at `aMax` in `hiinf_htm` at `maxAge = 1500`. There is
   no `warn_if_htm_rollover_clipped` here; judge it from
   `statistics.shareAtAssetUpperBound`.
3. **Not ported:** the three calibration drivers in `hi`
   (`calibrate_history_independent_tax.jl` and friends). They go through
   `make_history_independent_params` and read `eq.statistics`, so `pSS`/`pHH`
   flow through `base_kwargs` untouched; only the transcript filename would
   need them, exactly as in `hiinf_htm`.
