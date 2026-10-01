# `hdinf_htm` — history-dependent tax with hand-to-mouth agents

Infinite-horizon history-dependent model of `psmodel.tex`, with the exogenous
asset-market-access shock added. `hdinf` is left untouched; this directory is a
copy with the extension patched in, following `hiinf_htm`.

## The model

Two exogenous access states, `Pr(S'=S|S) = pSS` and `Pr(H'=H|H) = pHH`
(`s` and `h` in psmodel.tex). SAVERS solve the `hdinf` problem with a mixed
continuation:

    V^S(a,m,x) = max (1-beta)(log c - phi l^(1+eta)/(1+eta))
                     + beta E[ pSS V^S(a',m',x') + (1-pSS) V^H(a',m',x') | x ]
    s.t.  c + q(a') a' = lambda (e^{z eps kappa} m l)^{(1-tau) theta0} + a,
          a' >= bbar e^{kappa + rho z},   m' = (e^{z eps kappa} l m)^mu.

HAND-TO-MOUTH households face the same budget with `a'` imposed:

    a' = a/qSav  (a >= 0),   a' = a  (a < 0),
    c  = lambda (e^{z eps kappa} m l)^{(1-tau) theta0} + 1_{a<0} (1-qBorr) a.

The access chain is independent of `(z, eps)`, of the asset choice and of the
past-income stocks, and starts at its stationary distribution
`piS = (1-pHH)/(2-pSS-pHH)`, `piH = (1-pSS)/(2-pSS-pHH)`, so the hand-to-mouth
share is constant over the life cycle.

## The one real difference from `hiinf_htm`

**Hand-to-mouth hours are a dynamic choice here, not a static one.** In
`hiinf_htm` the HtM household has `a'` exogenous and nothing else linking
periods, so its hours solve a within-period first-order condition, its flow
payoff is precomputed once per `(lambda, kappa)`, and its half of the Bellman
operator never maximizes. Here hours still move the past-income stocks through
`s' = mu*(log wage + log h + s)`, so the household trades current leisure
against future tax liabilities exactly as a saver does. `solve_block_htm!`
therefore maximizes over the hours grid on every sweep.

It is still much cheaper than the saver's block, which searches `(a', h)`
jointly: the `a'` loop collapses to the single exogenous point, so the HtM
block costs roughly `8/nA` of the saver's. The continuation is TRILINEAR --
bilinear in `(s1', s2')` as for the saver, and linear in `a'` as well, because
`a' = a/qSav` falls between asset grid nodes.

## Running it

    julia --project=. run_history_dependent_tax.jl
    julia --project=. -e 'include("run_history_dependent_tax.jl");
                          run_history_dependent_tax(mu1=0.0, mu2=0.6, alpha=0.0)'

`SETTINGS` default to `pSS = 0.8882970895, pHH = 0.7589294614`, calibrated to
Kaplan, Violante and Weidner (2014) Table 4 and matching `hiinf_htm`: `piH =
0.3167` with an expected hand-to-mouth spell of 4.15 years. `pSS = 1, pHH = 0`
switches the extension off and reproduces `hdinf`. See
`references/KVW2014_WealthyHandToMouth/`.

Note on `alpha`: with `mu1 = 0` and `alpha = :paper`, alpha comes out at
`(rho-mu1)/(mu2-mu1) = 1.60` for `mu2 = 0.6`, outside `[0,1]`, and theta0 blows
up to 5.28. Pass `alpha = 0.0` for the single-root case psmodel.tex specifies,
which gives `theta0 = 1 - beta*mu2` as the promise-keeping restriction requires.
The `sweep_mu2` runs in `hdinf/results` already do this.

## What changed against `hdinf`

| file | changed lines |
|------|---------------|
| `solve_history_dependent_tax.jl` | ~330 |
| `model_settings.jl` | 18 |
| `run_history_dependent_tax.jl` | 12 |
| `plot_history_dependent_tax.jl` | 0 |

The saver's `solve_block!` and `evaluate_block!` are untouched; they receive
`EVmixS` in place of `EVz`.

* `access_stationary_distribution`, and `pSS`/`pHH`/`piS`/`piH` on `HDParams`.
* `HTMTransition` / `htm_transition`: the exogenous `a'`, its
  `cash = a - q(a')a'`, and the Young lottery placing `a'` on the asset grid.
  The value function and the forward pass read the SAME weights.
* `solve_block_htm!` / `evaluate_block_htm!`: the HtM Bellman block and its
  Howard counterpart. `sc.EVh` is reused as `(h, a)` rather than `(h, a')`.
* `solve_value_function_for_kappa` iterates on the pair `(V^S, V^H)`. Because
  the access shock is independent of `(z', eps')`, the mixing happens after the
  expectation: `EVmixS = pSS E[V^S] + (1-pSS) E[V^H]` and
  `EVmixH = pHH E[V^H] + (1-pHH) E[V^S]`. The sup-norm gap covers both.
* `simulate_kappa!` carries a sixth distribution axis (1 = S, 2 = H). A saver
  lands on one asset node, a hand-to-mouth household on the two its Young
  lottery straddles.
* Statistics gain `shareHandToMouth`, printed against the `piH` it must equal.

## Verification

All at `nA=31, nZ=3, nEps=3, nKappa=2, nS1=1, nS2=7, maxAge=60,
labor_grid_size=31, mu1=0, mu2=0.6, alpha=0`.

1. **Nesting.** At `pSS=1, pHH=0` every reported quantity matches `hdinf`:
   PV output, PV consumption, both asset ratios, the three shares, the clamped
   share and the simulation welfare agree to all 14 printed digits. `lambda`
   and the value-function welfare each differ by 1 ulp
   (9.68515670972075e-01 against ...76e-01), from the reassociation in
   `pSS*evs + (1-pSS)*evh`.
2. **Access shares.** `shareHandToMouth` reproduces `piH` to ~1e-13 at
   `piH = 1/3`, `0.0909` and `1`.
3. **All hand-to-mouth.** At `pSS=0, pHH=1` mean and median assets are exactly
   zero and PV(Y) = PV(C) to 15 digits.
4. **Monotonicity.** Mean assets fall in `piH` (0.5662 at 0, 0.5215 at 0.0909,
   0.3227 at 1/3, 0 at 1) and so does welfare (-0.5318, -0.5335, -0.5346,
   -0.5439). `theta0 = 0.424 = 1 - beta*mu2` throughout, as promise-keeping
   requires.
5. **Value function against simulation.** The gap is 3.3e-4 at `piH = 0` and
   3.3e-4 to 3.6e-4 at every `piH` tested -- the `beta^maxAge` truncation at
   `maxAge = 60`, unchanged by the extension.
6. **Cross-model, against `hiinf_htm`.** At `mu1 = mu2 = 0, nS1 = nS2 = 1`,
   with the alignment recipe (`hi`: `labor_solver = :grid`,
   `asset_choice_method = :grid_search`; `hd`: `labor_grid_spacing = :uniform`;
   both: same `hMin`, `labor_grid_size`, `bbar`, prices), the two independent
   implementations agree to about 9 significant digits -- `lambda` to 3.7e-10
   relative, mean assets to 5.0e-9, welfare to 1.1e-8 -- at `piH = 1/3` AND at
   `piH = 0`. See the next section: the residual is not in the extension.

## Problems found

1. **`common/shocks.jl` uses a low-accuracy normal CDF in the solution, and
   its own comment says it does not.** Its `normal_cdf` is the Zelen-Severo
   rational approximation (A&S 26.2.17), documented there as "maximum absolute
   error 7.5e-8 ... used only to place s-grid nodes at distribution quantiles
   ... and never in the solution itself". But the same function builds
   `z0_probs`, the initial distribution over `z`, which the forward pass is
   seeded with and which weights the birth value function. `hiinf` uses
   `QuantEcon.std_norm_cdf` instead, accurate to machine precision.

   Dumping every constructed object from both solvers at identical settings:
   `h_grid`, `a_grid`, `z_grid`, `Pz`, `eps_grid`, `Peps`, `kappa_grid`,
   `Pkappa`, `pow`, `beta`, `tau`, `phi`, `eta`, all three prices, `G` and
   `hMax` agree to 0.0e+00. **Only `z0_probs` differs, by 1.68e-08** --
   consistent with the documented 7.5e-08 bound. That single difference
   accounts for the whole cross-model gap in item 6, and it is invariant to
   `tolV` (unchanged at 1e-8, 1e-10 and 1e-12), which rules out the VFI
   stopping rule. The gap is the same size with and without hand-to-mouth
   agents, so the extension neither causes nor amplifies it.

   This is pre-existing shared code and was NOT modified. Swapping
   `build_markov_shock`'s initial-probability call to `QuantEcon.std_norm_cdf`
   would close it, at the cost of touching `hd`, `hdinf` and `hiinf` together.
   Resolved 2026-09-19: the shared `normal_cdf` is now `QuantEcon.std_norm_cdf`
   for every solver (`common/src/shocks.jl`), and the hd and hi families agree
   to machine precision at `mu1 = mu2 = 0`.
2. **`save_hd_result` in `hdinf` references `p.J`**, which `HDParams` does not
   define in an infinite horizon, so the call throws a `FieldError` before
   writing anything. Fixed in this directory's copy (`maxAge` replaces it); the
   `hdinf` original still has it.
3. **Memory doubles.** Four `(nA, nS1, nS2, nZ, nE)` value-function arrays
   instead of two, plus one more policy array of the same shape. At the
   production sweep dimensions (`nA=151, nS1=1, nS2=151, nZ=15, nEps=11`) that
   is about 30 MB per array, so roughly 90 MB more per kappa solve.
4. **The hand-to-mouth rollover `a' = a/qSav` has no fixed point above zero**,
   exactly as in `hiinf_htm`: balances grow at the gross return with nothing
   decumulating, and only `aMax` stops it. `diagnostics.htmRolloverClipped`
   records whether the rule ran off the top of the grid. Unlike `hiinf_htm`
   there is no separate warning here, because `hdinf` reports upper-bound
   binding through `print_upper_bound_warning` already.

## The sweep harness

`sweep_mu2.jl`, `sweep.sh`, `join_results.jl` and `recover_ratios.jl` are here
and work standalone. Typical use:

    julia --project=. sweep_mu2.jl mu2_min=0.3 mu2_max=0.9 n_mu2=10 \
          mu1=0 alpha=0 pSS=0.6666666666666666 pHH=0.3333333333333333
    julia --project=. join_results.jl results "nS2=151"
    qsub sweep.sh                       # or -J 1-2 to test the pipeline first

**The access chain is recorded, not assumed.** `pSS`/`pHH` are read off the
params each solve actually used, stored as sweep scalars alongside `piH`, and
forced into `settings` as well -- `settings` is what `recover_ratios.jl` splats
back, and a chain that came from `HD_SETTINGS` rather than from `kwargs` would
otherwise be absent there. Per-point `shareHandToMouth` is stored too, so the
"must equal piH" check is verifiable from the file rather than only in the log.
Both appear in the filename:

    sweep_mu2_n=1_mu2=0.300-0.300_mu1=0.000_nS1=1_nS2=5_pSS=0.6667_pHH=0.3333.jld2

**`join_results.jl` refuses to mix chains.** Guard 0 runs before the others,
because a merged plot of two different access chains is the one failure that
looks plausible instead of erroring. It also refuses `hdinf` files, which carry
no chain and cannot be told apart from a `piH = 0` run here. `RESULTS_DIR` and
the filename filter are now command-line arguments, so one chain can be picked
out of a mixed directory:

    julia --project=. join_results.jl results "pSS=0.6667"

The reporting block at the bottom of `join_results.jl` is LIVE here; it is
commented out in `hdinf/join_results.jl`, which was the workaround for sweeps
written before the ratio fields existed. Every sweep this directory writes
carries them.

`sweep.sh` has `pSS`/`pHH` at the front of `MODEL_ARGS`, passed explicitly so a
submitted job records the chain it ran, and its `cd ~/hd` comments now say
`~/hdinf_htm`. Everything else -- the PBS directives, the scratch staging, the
julia module pin -- is unchanged.

Verified end to end at `nA=21, nZ=3, nEps=3, nKappa=2, nS1=1, nS2=5,
maxAge=40`: three `mu2` points swept and saved, `join_results.jl` merged and
reported them with the access-chain header and no share warning,
`recover_ratios.jl` passed them through as already carrying ratios, and adding
a fourth point at `pSS=1, pHH=0` made guard 0 fire with the two chains named.
