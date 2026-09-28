# `hiinf_htm` — history-independent tax with hand-to-mouth agents

Infinite-horizon Bewley model of `psmodel.tex`, section "History independent
tax", with the exogenous asset-market-access shock added. `hiinf` is left
untouched; this directory is a copy with the extension patched in.

## The model

Households are in one of two exogenous access states, with

    Pr(S' = S | S) = pSS  (= s in psmodel.tex),
    Pr(H' = H | H) = pHH  (= h in psmodel.tex).

SAVERS choose `a'` as in `hiinf`, against a mixed continuation:

    V^S(a,x) = max (1-beta)(log c - phi h^(1+eta)/(1+eta))
                 + beta E[ pSS V^S(a',x') + (1-pSS) V^H(a',x') | x ]
    s.t.  c + q(a') a' = lambda (e^{kappa+z+eps} h)^{1-tau} + a,
          a' >= bbar e^{kappa+rho z}.

HAND-TO-MOUTH households choose only hours; `a'` is imposed:

    a' = a/qSav  (a >= 0),      a' = a  (a < 0),
    c  = lambda (e^{kappa+z+eps} h)^{1-tau} + 1_{a<0} (1-qBorr) a.

The access chain is independent of `(z, eps)` and of the asset choice and is
seeded at its stationary distribution

    piS = (1-pHH)/(2-pSS-pHH),   piH = (1-pSS)/(2-pSS-pHH),

so the hand-to-mouth share is constant at `piH` over the life cycle rather than
converging to it.

## Running it

    julia --project=. run_history_independent_tax.jl                      # defaults
    julia --project=. -e 'include("run_history_independent_tax.jl");
                          run_history_independent_tax(pSS=0.95, pHH=0.5)'

`SETTINGS` in `model_settings.jl` default to
**`pSS = 0.8882970895, pHH = 0.7589294614`**, calibrated to Kaplan, Violante and
Weidner (2014) Table 4 — the printed SCF 2007-2009 transition matrix across
poor-HtM, wealthy-HtM and non-HtM status, collapsed to two states and
annualized. That gives `piH = 0.3167` against the 0.317 they report, with an
expected hand-to-mouth spell of 4.15 years against their stated 3.5 (W-HtM) and
4.5 (P-HtM). The earlier iid default `pSS = 2/3, pHH = 1/3` matched the share
but forced the second eigenvalue to zero, i.e. a 1.50-year spell. Working notes
and the replication package are in `references/KVW2014_WealthyHandToMouth/`.

`tau = 0.181` in both this directory and `hiinf` (changed from 0.161).

Passing `pSS = 1.0, pHH = 0.0` switches the extension off and reproduces `hiinf`
exactly, which is the regression test. The options header prints `(piS, piH)`
and flags `piH = 0` explicitly, so a run with the extension off is not
silent.

Three parameterizations are nested, as in psmodel.tex:

| setting              | piH   | model                          |
|----------------------|-------|--------------------------------|
| `pSS=1,   pHH=0`     | 0     | `hiinf` (standard Bewley)      |
| `pSS=0,   pHH=1`     | 1     | no-savings model               |
| `pHH = 1-pSS`        | `pHH` | iid access (no persistence)    |

## Calibration

`calibrate_history_independent_tax_beta.jl` is ported from `hiinf`: two
instruments (`bbar`, `beta`) against two targets (true borrowing limit / mean
labor income = 0.185, and the mean or median asset ratio). Run it with

    julia --project=. calibrate_history_independent_tax_beta.jl
    # or, from a session:
    r = calibrate_history_independent_tax_beta(calib = BetaCalibration(asset_moment = :mean),
                                               nZ = 15, nEps = 11, nKappa = 5, nA = 151)

`calibrate_history_independent_tax_Claude.jl` comes along because the beta
driver includes it for `solve_scalar`, `moments_from` and `with_tee`; its own
three-instrument calibration is carried over untouched and unused.

**The access chain is GIVEN, not calibrated.** `pSS` and `pHH` are held fixed
through the search exactly as the prices are, for two reasons. The search has
two instruments and two targets, so a third parameter has no moment to hit; and
under the iid default `piH` maps one for one onto `shareHandToMouth`, so
searching over it would only rediscover the number already typed into
`SETTINGS`. The header prints the chain in force and labels it, the result
carries `pSS`, `pHH`, `piH` and the realized `htmShare`, and the transcript
filename now interpolates `pSS`/`pHH` so two access chains cannot overwrite each
other.

**The beta ceiling binds harder with hand-to-mouth agents.** They hold less, so
the asset ratio falls at a given beta and the calibrated beta has to rise.
Measured at `nA=31, nZ=3, nEps=3, nKappa=2, maxAge=60`, `tau = 0.181`, prices at
their SETTINGS values:

| target                        | beta at `piH = 0` | beta at `piH = 1/3` | change |
|-------------------------------|-------------------|---------------------|--------|
| median assets / mean LI, 0.043 | 0.96012712       | 0.96895113          | +0.0088 |
| mean assets / mean LI, 0.588   | 0.97803030       | 0.98202918          | +0.0040 |

`bbar` barely moves, from -0.17052 to -0.16967. The mean-asset case is the one
to watch: at `piH = 1/3` the calibrated beta sits 0.0030 under the 0.985
ceiling, so a third of households hand-to-mouth consumes 57% of the headroom
that was there at `piH = 0`. The ceiling is not a number that can simply be
raised — it is pinned by `qSav - beta_max >= 3*qSav/maxAge` — so buying room
means raising `maxAge`. At production grids the levels will differ; the
direction will not.

These are also a cross-check on the port: `hiinf` and `hiinf_htm` at `piH = 0`
return the same calibrated `beta` and `bbar` to all eight printed digits, for
both asset moments.

**The ceiling itself looks conservative by about a factor of eight**, which
matters because it is exactly the headroom the table above is eating. The rule
`qSav - beta_max >= 3*qSav/maxAge` comes from treating `q/(q-beta)` as the
relaxation time of the cross-section. Measured instead, by the ratio of
successive increments of the aggregate path at `nA=41, nZ=3, nEps=3, nKappa=2,
maxAge=900`:

| beta | beta/qSav | measured decay of `A` | T | measured decay of `Y-C` | T | `q/(q-beta)` | meanA/LI |
|------|-----------|----------------------|------|------------------------|------|------|------|
| 0.9700 | 0.97980 | 0.95808 | 23.4 | 0.96031 | 24.7 | 49.5  | 0.371 |
| 0.9780 | 0.98788 | 0.95766 | 23.1 | 0.95668 | 22.6 | 82.5  | 0.578 |
| 0.9820 | 0.99192 | 0.95739 | 23.0 | 0.95655 | 22.5 | 123.8 | 0.811 |
| 0.9849 | 0.99485 | 0.95899 | 23.9 | 0.95728 | 22.9 | 194.1 | 1.139 |

The decay factor is `rho = 0.958` at every beta, not `beta/qSav`: Rouwenhorst's
second eigenvalue is exactly `rho`, so the z-marginal relaxing from
`z_initial = 0` is the slowest surviving mode, and the `beta/qSav` mode that
would govern a free AR(1) in assets never dominates because the borrowing
constraint is a reflecting barrier households keep returning to. Even at the
ceiling the horizon needed is 24 periods against the 194 the rule asks for.

Two things still bind before the horizon does, so do not simply raise
`beta_max` to `qSav`. It must stay STRICTLY below `qSav` -- asset demand
diverges at `beta = qSav` and there is no stationary distribution to converge
to. And mean assets rise steeply in beta, from 0.371 at 0.9700 to 1.139 at
0.9849, so `aMax` and `shareAtAssetUpperBound` become the constraint well
before `maxAge` does. What the rule proxies for is already measured ex post by
`tolDriftRel` on the drift at `maxAge`, which is the check to lean on.

## What changed against `hiinf`

`plot_history_independent_tax.jl` and
`calibrate_history_independent_tax_Claude.jl` are byte-identical; the solver,
settings, runner and beta calibration change by 471 / 32 / 15 / 60 lines. The saver's problem is untouched — it is the
`hiinf` code with `EV_S` substituted for `EV`.

* **`HIParams`** gains `pSS`, `pHH` and the derived `piS`, `piH`;
  `access_stationary_distribution` computes them and rejects `pSS = pHH = 1`.
* **`HTMTransition`** (`htm_transition`) precomputes, once per `HIParams`, the
  hand-to-mouth `a'`, its `cash = a - q(a')a'`, and the Young lottery placing
  `a'` on the grid. The value function and the forward pass read the SAME
  weights, which is what keeps the welfare cross-check meaningful.
* **`precompute_htm_payoffs`** / **`update_htm_values!`**: the hand-to-mouth
  choice is static (`a'` exogenous, so hours solve the same within-period FOC
  with no continuation term), so its flow payoff is computed once per
  `(lambda, kappa)` and its Bellman operator never maximizes.
* **`solve_policies_for_kappa`** iterates on the pair `(V^S, V^H)`. Because the
  access shock is independent of `(z', eps')`, the mixing happens after the
  expectation: `compute_expected_value!` is called twice, unchanged, and
  `EV_S = pSS E[V^S] + (1-pSS) E[V^H]`, `EV_H = pHH E[V^H] + (1-pHH) E[V^S]`.
  The sup-norm gap is measured over both value functions.
* **`simulate_kappa!`** carries a fourth distribution axis (1 = S, 2 = H). The
  `(a',z',eps')` mass is built exactly as before and then split across access
  states.
* **Statistics** gain `shareHandToMouth`, printed next to the `piH` it must
  equal.
* **`warn_if_htm_rollover_clipped`** reports mass piling up at `aMax` (see
  problem 2 below).

## Verification

All checks at `nA=41, nZ=3, nEps=3, nKappa=2, maxAge=60` unless noted. The
nesting and closed-form checks below were run at `tau = 0.161`, before the
change to 0.181; they are properties of the code path, not of the tax
parameter, and the `piH = 0` calibration cross-check above re-establishes the
nesting at `tau = 0.181`.

1. **Nesting.** At `pSS=1, pHH=0` every reported quantity — lambda, PV output,
   PV consumption, settled assets, mean and median assets over mean labor
   income, the three share statistics, and both welfare numbers — agrees with
   `hiinf` to all 15 printed digits. The one exception is the government budget
   residual, `-6.00204426687e-09` against `-6.00204426743e-09`: a difference of
   5.6e-19 in absolute terms, from the reassociation in
   `pSS*E[V^S] + (1-pSS)*E[V^H]`. Lambda itself is bit-identical.
2. **Closed form at `pSS=0, pHH=1`.** Everyone is hand-to-mouth and, starting
   from `a0 = 0`, stays at `a = 0` forever, so the model collapses to a static
   problem. With log utility hours are independent of the wage AND of lambda,
   `h* = ((1-tau)/phi)^(1/(1+eta)) = 0.943164227228366`, and lambda follows in
   closed form from the PV budget. The solver returns
   `max_j |H_j - h*| = 6.7e-16`, lambda to a relative `2.2e-16`, and PV output
   to `2.0e-15`. Mean assets are exactly zero.
3. **Access shares.** `shareHandToMouth` matches `piH` to 1e-13 at
   `pSS=0.7,pHH=0.3` (piH = 0.3), `pSS=0.95,pHH=0.5` (piH = 0.0909091) and
   `pSS=0.9,pHH=0.6` (piH = 0.2), in both the calibration window and the
   all-ages window. The residual gap is the `massTol = 1e-14` cutoff.
4. **Value function against simulation.** The independent forward-pass welfare
   tracks the birth value function at the same magnitude as the baseline: the
   gap is 3.26e-04 at `maxAge = 60` for `piH = 0, 0.0909, 0.3, 1` alike, and
   falls to -1.6e-07 at `maxAge = 200`. It is the `beta^maxAge` truncation, not
   an inconsistency between the two access blocks.
5. **Monotonicity.** Mean assets over mean labor income fall in `piH`
   (0.2053 at 0, 0.1981 at 0.0909, 0.1479 at 0.30, 0 at 1) and so does welfare
   (-0.49986, -0.50295, -0.50992, -0.52981).
6. `asset_choice_method = :interpolate` runs and gives the same `shareHtM`;
   only `:grid_search` was checked against `hiinf` digit for digit.

## Problems and inconsistencies found

1. **psmodel.tex uses one price `q`; the code uses two.** The tex writes
   `a' = a/q` and `c = y + (1-q)a` with a single intertemporal price, but the
   solver has `qSav = 0.99` for `a' >= 0` and `qBorr = 0.97` for `a' < 0`. I
   mapped the rollover to `qSav` and the debt-service term to `qBorr`, which is
   the only assignment under which the hand-to-mouth budget is the saver's
   budget `c + q(a')a' = y + a` with `a'` imposed. The paper should say which
   price it means.
2. **`a' = a/qSav` has no fixed point above zero.** A household that stays
   hand-to-mouth with `a > 0` sees its balances grow at `1/qSav = 1.0101` per
   period forever while it consumes only labor income — nothing decumulates.
   In an infinite horizon the only thing that stops it is `aMax`. Whether it
   bites is a question about `pHH`: the expected spell length is `1/(1-pHH)`,
   so 2 periods at `pHH = 0.5` and 50 at `pHH = 0.98`. No mass reached the
   upper bound in any run here (`share at upper asset bound = 0.00000000`
   throughout, with `aMax = 15` against mean assets near 0.2), but
   `warn_if_htm_rollover_clipped` reports it if it ever does. The rule is
   resource-consistent — `a' = a/q` with `c = y` is exactly the budget — so
   this is a calibration limit, not an accounting error.
3. **Debt is frozen in the H state and can end up below the borrowing limit.**
   A hand-to-mouth debtor holds `a' = a` regardless of `z'`, while the limit
   `bbar e^{kappa + rho z}` tightens when `z` falls. On returning to `S` the
   household can therefore find `a < bbar e^{kappa + rho z}` and must pay the
   debt down in one period. The mechanism exists in `hiinf` already — `a'` is
   chosen under today's `z` and faces tomorrow's limit — but hand-to-mouth
   spells make it more likely, because debt cannot be reduced during a spell.

   **How the code handles it, in four steps.** (a) `default_asset_grid` spans
   down to the loosest limit `min_{kappa,z} bbar e^{kappa+rho z}`, so sub-limit
   holdings are representable grid points: 259 of 1515 `(kappa, z, a)`
   combinations lie below the limit of their own `z`. (b) The constraint is on
   `a'`, never on `a` — `first_feasible_asset_indices` builds `ia_first[iz]`
   from today's `z` and the maximizer scans `iap in ia_first:nA` whatever `ia`
   is, so arriving below the limit is legal and only climbing back is required.
   (c) Feasibility is bought with hours: in `optimal_labor_foc`, when
   `cash + income_coeff*hMin^(1-tau) <= 0` the hours lower bound is raised to
   `(-cash/income_coeff)^(1/(1-tau))`, the point where `c > 0`, so the
   constraint is enforced as a MINIMUM-HOURS requirement rather than a
   rejection. (d) If even `hMax` cannot do it, the maximizer returns the FINITE
   sentinel `VINFEASIBLE = -1e18`, which enters `EV` multiplied by its
   probability: the worst one-step `z` drop under Rouwenhorst with `nZ = 5` has
   probability `(1-p)^4 = 1.94e-07` at `p = (1+rho)/2 = 0.979`, and
   `beta * 1.94e-07 * (-1e18) ~ -7e10` against attainable values of order one,
   so no saver ever chooses such an `a'`. Hand-to-mouth households cannot be
   deterred, but savers price the risk through
   `EV_S = pSS E[V^S] + (1-pSS) E[V^H]`, and the sentinel survives the
   dilution: a 20-period spell still leaves `beta^20 * 1e-07 * 1e18 ~ 4e10`.
   If the forward pass ever did reach one, the `c > 0.0 ||` error reports the
   full state — note the `h > 0.0` guard above it does NOT catch this, since
   `best_h = hMin = 1e-8`.

   **Measured**, replaying the solved policies over the forward pass at
   `nA=101, nZ=5, nEps=5, nKappa=3, maxAge=200`, `bbar = -0.2`, `hMax = 5`:

   | | `piH = 0` (= `hiinf`) | `piH = 0.0909` |
   |---|---|---|
   | mass arriving below the limit | 0.9097 of 200 (0.455%) | 1.0966 of 200 (0.548%) |
   | of which hand-to-mouth | — | 0.1544 (14.1%) |
   | pushed exactly onto `a' = limit` | 51.2% | 47.2% |
   | max shortfall `limit(z) - a` | 0.5354 | 0.5354 |
   | min consumption, all positive mass | 0.024912 | 0.024873 |
   | mean hours, all / below-limit | 0.9333 / 0.9997 | 0.9343 / 0.9915 |

   So it is not hypothetical: it happens in `hiinf` on 0.455% of observations,
   and the maximum shortfall of 0.5354 is the `kappa_max` household at the grid
   bottom drawing `z_min` — the extreme corner, carrying positive mass.
   Below-limit households work 7.1% more hours than average (0.9997 against
   0.9333), with a maximum of 2.25 against `hMax = 5`, which is the margin step
   (c) rents. Hand-to-mouth agents raise the below-limit mass by 20.5% and are
   over-represented among it by a factor of 1.55 (14.1% of that mass against
   9.09% of the population), in the direction the mechanism predicts, but they
   do not tighten the binding case: minimum consumption falls only 0.16%, and
   no positive-mass state was infeasible in either run. That clean bill is
   specific to `bbar = -0.2` and `hMax = 5`; a looser limit or a lower hours
   ceiling shrinks the margin step (c) relies on.
4. **The tex does not say that the initial access state is independent of
   `(a0, z_{-1}, kappa)`.** "The initial distribution over market-access states
   is assumed to be stationary" fixes the marginal, not the joint. I assumed
   independence, which is what makes the birth value function the `piS/piH` mix
   of `V^S` and `V^H` at the same asset placement.
5. **Two typos in psmodel.tex.** "The setap encompasses" → "setup"; the
   footnote "all agents face the same disutility parameter phi" is redundant
   with the rest of the section, where `phi` never carries a type index. The
   three nesting claims (`s=0,h=1`; `s=1,h=0`; `s=1-h`) and `piH = (1-s)/(2-s-h)`
   all check out.
6. **Reporting choice, not an error:** the "grid/choice borrowing limit" and
   "true borrowing limit" averages include hand-to-mouth mass, evaluated at the
   limit the household WOULD face as a saver. `share at effective borrowing
   bound` counts savers only, since hand-to-mouth households do not choose.
   With `piH` large the first two statistics are therefore about a constraint
   most of the population is not currently facing.
