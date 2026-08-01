# 1. Setup
Load the calibration file and activate the project environment. Includes flow one
direction (`solve ← model_settings ← run ← calibrate`), so this single `include`
also defines `run_history_independent_tax` and the solver itself. Plotting is a
separate layer: `include("plot_history_independent_tax.jl")` when needed.

```julia
const PROJDIR = "/Users/mkapicka/Library/CloudStorage/Dropbox/Projects/HSV_HistoryDep_LifeCycle/code/julia/Bewley/hi"
cd(PROJDIR)         # sets pwd(); Pkg.activate alone does NOT change the working directory
using Pkg
Pkg.activate(PROJDIR)
Pkg.instantiate()   # first run only; downloads the packages the Manifest lists
include(joinpath(PROJDIR, "calibrate_history_independent_tax_Claude.jl"))
@assert pwd() == PROJDIR
```

# 2. Calibration Procedure
Calibrate the model to match the three targets in `CalibrationParams` by adjusting the three instruments `qSav`, `qBorr`, and `bbar`. The calibration function searches over the three instruments, solving the model at each step to evaluate the moments. The calibration stops when all three moment residuals are below `calib.moment_tol`, or when it has used `calib.outer_max_sweeps` block-coordinate sweeps, whichever comes first; only the former sets `converged = true`. Labor supply is solved for at each step using the specified `labor_solver`. The calibration function returns a `NamedTuple` with the calibrated instruments, the achieved moments, and other information about the calibration.

```julia
result = calibrate_history_independent_tax()
```

The calibration function takes the following main keyword arguments:
- `J`: terminal age (default: `39`). **Cohorts are indexed from `0` through `J`, giving `J + 1 = 40` cohorts.**
- `nA`: number of asset-grid points (default: `101`).
- `nZ`: number of persistent-productivity grid points (default: `5`).
- `nEps`: number of transitory-productivity shocks (default: `5`)
- `nKappa`: number of permanent-productivity shocks (default: `3`)
- `qGov`: government discount price (default: `0.99`)
- `labor_solver`: `:hybrid_newton` (default), `:brent`, `:grid`.
- `aMax`: upper bound of the asset grid (default: `15.0`)

Any other `SETTINGS` entry may be passed the same way, **except** `qSav`, `qBorr`,
`bbar`, and `verbose`: those are set by the calibration itself and passing them
raises an error.

Targets and search controls live together in one `CalibrationParams` struct,
passed as the `calib` keyword — not as bare keywords (passing e.g.
`asset_moment = :median` directly raises an error because it is forwarded to
the model constructor):

```julia
result = calibrate_history_independent_tax(
    calib = CalibrationParams(asset_moment = :median, moment_tol = 1e-4))
```

| Field                                          | Meaning                                                         |
|------------------------------------------------|-----------------------------------------------------------------|
| **Targets**                                    |                                                                 |
| `meanAssetsToMeanLaborIncome` (`0.588`)        | Target (i) under the default `asset_moment = :mean`             |
| `medianAssetsToMeanLaborIncome` (`0.043`)      | Target (i) under `asset_moment = :median`                       |
| `trueBorrowingLimitToMeanLaborIncome` (`0.18`) | Target (ii); see the moment (ii) note below                     |
| `shareNegativeLiquidAssets` (`0.26`)           | Target (iii)                                                    |
| `asset_moment` (`:mean`)                       | Picks which asset ratio is targeted; the other is left free     |
| **Search**                                     |                                                                 |
| `moment_tol` (`5e-4`)                          | Stop when the largest absolute moment residual falls below it   |
| `outer_max_sweeps` (`12`)                      | Cap on block-coordinate sweeps                                  |
| `qSav_min`/`qSav_max` (`0.900`/`1.040`)        | Bracket searched for `qSav`                                     |
| `qBorr_min`/`qBorr_max` (`0.700`/`1.040`)      | Bracket searched for `qBorr`                                    |
| `bbar_min`/`bbar_max` (`-0.80`/`-0.01`)        | Bracket searched for `bbar`                                     |
| `inner_xtol` (`1e-5`), `inner_maxevals` (`40`) | Tolerance and evaluation cap of each 1-D Brent solve            |
| `verbose` (`true`)                             | Print the calibration's own log (inner solves are always silent)|

When the file is run as a script, the full transcript is also written to
`calibration_results/calib_<asset_moment>_J<J>_nA<nA>_nZ<nZ>_nEps<nEps>_nKappa<nKappa>_<date>.txt`,
so archived runs and the console log come from one mechanism.

## Result structure
`result` is a `NamedTuple`; every field is documented below. Throughout,
`nAge = J + 1 = 40`: age-indexed vectors run over indices `1, …, J+1`,
corresponding to economic ages `j = 0, …, J`.

### `result` — top level
| Field            | Type                 | Meaning                                                   |
|------------------|----------------------|-----------------------------------------------------------|
| `qSav`           | `Float64`            | Calibrated saving price                                   |
| `qBorr`          | `Float64`            | Calibrated borrowing price                                |
| `bbar`           | `Float64`            | Calibrated borrowing limit (`≤ 0`; `B > 0` is `bbar = -B`)|
| `eq`             | `NamedTuple`         | Equilibrium at the calibrated instruments (see below)     |
| `moments`        | `NamedTuple`         | Achieved moments (see below)                              |
| `residuals`      | `NamedTuple`         | Achieved minus target, moment by moment                   |
| `converged`      | `Bool`               | `true` if `max|residual| ≤ calib.moment_tol`              |
| `sweeps`         | `Int`                | Block-coordinate sweeps used                              |
| `nSolves`        | `Int`                | Full model solves performed                               |
| `elapsedSeconds` | `Float64`            | Wall-clock time of the calibration                        |
| `calib`          | `CalibrationParams`  | Calibration parameters the run used (targets + search)    |
| `params`         | `HIParams`           | Calibrated parameter struct (see below)                   |

### `result.eq` — equilibrium at the calibrated point
| Field                | Type                | Meaning                                                      |
|----------------------|---------------------|--------------------------------------------------------------|
| `lambda`             | `Float64`           | Tax-level parameter clearing the government budget           |
| `govBudgetResidual`  | `Float64`           | `govBudgetLHS - govBudgetRHS`                                |
| `govBudgetLHS`       | `Float64`           | `(1-qGov) Σ_j qGov^j (Y_j - C_j)`                            |
| `govBudgetRHS`       | `Float64`           | `(1-qGov^nAge) G`                                            |
| `C`, `H`, `Y`, `A`   | `Vec{Float64, nAge}`| Aggregate consumption, hours, labor income, assets by age    |
| `consumptionPV`      | `Float64`           | `qGov`-discounted sum of `C`                                 |
| `outputPV`           | `Float64`           | `qGov`-discounted sum of `Y`                                 |
| `statistics`         | `NamedTuple`        | Cross-sectional statistics (see below)                       |
| `welfare`            | `NamedTuple`        | Welfare by `kappa` and overall (see below)                   |
| `solutions`          | `NamedTuple`        | Policies when `store_solutions = true`, else `nothing`       |
| `parameters`         | `HIParams`          | Parameters used for this solve                               |
| `converged`          | `Bool`              | `true` if `|govBudgetResidual| ≤ tolGovBudget`               |
| `elapsedSeconds`     | `Float64`           | Time of this single solve                                    |

Solver failure modes (no `lambda` bracket, Brent failure, residual above
tolerance) are reported with `@warn` when they occur; `eq` records only
`converged`.

### `result.eq.statistics` — cross-sectional moments
| Field                                              | Meaning                                                |
|----------------------------------------------------|--------------------------------------------------------|
| `totalMass`                                        | Total population mass (normalization check)            |
| `meanAssets`, `medianAssets`                       | Mean and mass-weighted median of `a`                   |
| `meanLaborIncome`                                  | Mean labor income                                      |
| `meanBorrowingLimit`                               | Mean true limit, ages `j = 0,…,J-1` (see note)         |
| `meanEffectiveGridBorrowingLimit`                  | Same after grid rounding; same age range               |
| `meanAssetsToMeanLaborIncome`                      | Moment (i) under `asset_moment = :mean`                |
| `medianAssetsToMeanLaborIncome`                    | Moment (i) under `asset_moment = :median`              |
| `meanBorrowingLimitToMeanLaborIncome`              | Moment (ii); the "true borrowing limit" ratio          |
| `meanEffectiveGridBorrowingLimitToMeanLaborIncome` | Same, using the grid-rounded limit                     |
| `shareNegativeLiquidAssets`                        | Moment (iii); share with `a < 0`                       |
| `shareZeroAssets`                                  | Share with `a = 0`                                     |
| `shareAtEffectiveBorrowingConstraint`              | Share at the (grid) borrowing constraint               |
| `assetUpperBound`, `hoursUpperBound`               | `aMax` and `hMax`                                      |
| `shareAtAssetUpperBound`, `shareAtHoursUpperBound` | Mass sitting at each bound                             |
| `maxMaterialNextAssets`, `maxMaterialHours`        | Largest `a'` and `h` over cells with mass above `1e-8` |
| `assetUpperBoundSlack`, `hoursUpperBoundSlack`     | Bound minus the material maximum                       |
| `assetUpperBoundBinding`, `hoursUpperBoundBinding` | `Bool`, share above `1e-8`                             |
| `upperBoundsBinding`                               | `Bool`, either of the two                              |
| `unconditionalDistributions`                       | `NamedTuple`, see below                                |

**Note on moment (ii).** A borrowing limit exists only at ages `j = 0,…,J-1`. At
the terminal age `j = J` the model imposes `a' ≥ 0` instead, so no limit is
defined there. Both `meanBorrowingLimit` and `meanEffectiveGridBorrowingLimit`
average `-bbar · exp(kappa + rho z)` over the mass of ages `j = 0,…,J-1`,
excluding the terminal age rather than counting it as a zero. Target (ii) is
matched against that average, so `meanBorrowingLimitToMeanLaborIncome = 0.180`
is the limit households actually face. The denominator is unchanged: mean labor
income remains the whole-population mean over all `nAge = 40` ages.

### `result.eq.statistics.unconditionalDistributions`
Populated when `collect_distributions = true` (the default); with it off, the
vectors are empty and only the asset-grid mass is filled.
| Field                     | Meaning                                                   |
|---------------------------|-----------------------------------------------------------|
| `assetGrid`, `assetMass`  | Asset grid and the probability mass on each point         |
| `assetMassTotal`          | `sum(assetMass)`; equals `totalMass`                      |
| `hours`, `consumption`    | Observation-level `h` and `c`, one entry per visited cell |
| `weights`                 | Matching observation weights                              |
| `observationWeightTotal`  | `sum(weights)`; equals `totalMass`                        |

### `result.eq.welfare`
| Field                                      | Type                  | Meaning                                                 |
|--------------------------------------------|-----------------------|---------------------------------------------------------|
| `kappaGrid`, `kappaProbabilities`          | `Vec{Float64, nKappa}`| Permanent-shock grid and its probabilities              |
| `valueFunctionByKappa`                     | `Vec{Float64}`        | Expected utility from the value function at `j = 0`     |
| `simulationByKappa`                        | `Vec{Float64}`        | Same object accumulated through the simulation          |
| `differenceByKappa`                        | `Vec{Float64}`        | Simulation minus value function; a check                |
| `overallValueFunction`, `overallSimulation`| `Float64`             | `Pkappa`-weighted aggregates                            |
| `overallDifference`                        | `Float64`             | Their difference; should be round-off, of order `1e-13` |

### `result.moments` — the four achieved ratios
| Field                                 | Type      | Meaning                                                          |
|---------------------------------------|-----------|------------------------------------------------------------------|
| `medianAssetsToMeanLaborIncome`       | `Float64` | Median assets / mean labor income                                |
| `meanAssetsToMeanLaborIncome`         | `Float64` | Mean assets / mean labor income                                  |
| `trueBorrowingLimitToMeanLaborIncome` | `Float64` | Copied from `eq.statistics.meanBorrowingLimitToMeanLaborIncome`  |
| `shareNegativeLiquidAssets`           | `Float64` | Share of agents with `a < 0`                                     |

### `result.residuals` — the three targeted gaps, achieved minus target
| Field                                 | Type      | Meaning                                                          |
|---------------------------------------|-----------|------------------------------------------------------------------|
| `assetsToMeanLaborIncome`             | `Float64` | Whichever asset ratio `asset_moment` selects                     |
| `trueBorrowingLimitToMeanLaborIncome` | `Float64` | Gap on the borrowing-limit ratio                                 |
| `shareNegativeLiquidAssets`           | `Float64` | Gap on the share of agents with `a < 0`                          |

### `result.calib` — the `CalibrationParams` used
The struct documented in the Section 2 table: the four targets, `asset_moment`,
and the search controls, exactly as the run used them.

### `result.params` — the calibrated `HIParams`
Defaults in parentheses are the values in `SETTINGS` (`model_settings.jl`), which
is the only place model parameters are defined; `HIParams` itself carries no
defaults for them. Four fields (`maxIterLambda`, `massTol`, and the derived
`omega_mean`/`epsilon_mean`/`kappa_mean`) are not in `SETTINGS` and default
inside the struct.
| Field                        | Type                  | Meaning                                                  |
|------------------------------|-----------------------|----------------------------------------------------------|
| `beta`                       | `Float64`             | Discount factor (0.96)                                   |
| `eta`                        | `Float64`             | Inverse Frisch elasticity  (2.0)                         |
| `phi`                        | `Float64`             | Level of the disutility of hours (1.0)                   |
| `tau`                        | `Float64`             | HSV progressivity (0.181)                                |
| `J`                          | `Int`                 | Terminal age; `nAge = J+1` periods, `j = 0, …, J` (39)   |
| `rho`                        | `Float64`             | Persistence of `z`; also scales borrowing limit (0.958)  |
| `sigma_omega`                | `Float64`             | S.d. of the innovation to `z` (`sqrt(0.017)`)            |
| `sigma_epsilon`              | `Float64`             | S.d. of the transitory shock `eps` (`sqrt(0.081)`)       |
| `sigma_kappa`                | `Float64`             | S.d. of the permanent shock `kappa` (`sqrt(0.101)`)      |
| `omega_mean`                 | `Float64`             | Mean innovation to `z`; `-sigma_omega^2/2` (-0.0085)     |
| `epsilon_mean`               | `Float64`             | Mean of `eps`; `-sigma_epsilon^2/2` (-0.0405)            |
| `kappa_mean`                 | `Float64`             | Mean of `kappa`; `-sigma_kappa^2/2` (-0.0505)            |
| `nZ`, `nEps`, `nKappa`       | `Int`                 | Grid sizes for `z`, `eps`, `kappa` (5, 5, 3)             |
| `z_initial`                  | `Float64`             | Value of `z` conditioning the `j = 0` distribution (0.0) |
| `z_grid`                     | `Vec{Float64, nZ}`    | Persistent-productivity grid                             |
| `Pz`                         | `Mat{Float64, nZ×nZ}` | Transition matrix for `z`                                |
| `z0_probs`                   | `Vec{Float64, nZ}`    | Distribution of `z` at `j = 0`                           |
| `eps_grid`                   | `Vec{Float64, nEps}`  | Transitory-shock grid                                    |
| `Peps`                       | `Vec{Float64, nEps}`  | I.i.d. probabilities of `eps`                            |
| `kappa_grid`                 | `Vec{Float64, nKappa}`| Permanent-shock grid                                     |
| `Pkappa`                     | `Vec{Float64, nKappa}`| Probabilities of `kappa`                                 |
| `z_discretization_method`    | `Symbol`              | `:rouwenhorst` (default) or `:tauchen`                   |
| `tauchen_width`              | `Float64`             | Grid width in s.d., used only for `:tauchen` (3.0)       |
| `bbar`                       | `Float64`             | Borrowing-limit scale, `≤ 0` (`B > 0` is `bbar = -B`)    |
| `aMax`                       | `Float64`             | Upper bound of the asset grid (15.0)                     |
| `nA`                         | `Int`                 | Number of asset-grid points (101)                        |
| `a_grid`                     | `Vec{Float64, nA}`    | The grid itself; contains `0.0` exactly                  |
| `asset_grid_method`          | `Symbol`              | `:nonuniform` (default) or `:linear`                     |
| `asset_grid_curvature_borrow`| `Float64`             | Spacing curvature below zero (1.8)                       |
| `asset_grid_curvature_save`  | `Float64`             | Spacing curvature above zero (2.5)                       |
| `asset_grid_borrow_share`    | `Float64`             | Share of points on the borrowing side (0.35)             |
| `asset_grid_zero_share`      | `Float64`             | Share of points in the band around zero (0.30)           |
| `asset_grid_zero_width`      | `Float64`             | Width of that band (0.08)                                |
| `asset_choice_method`        | `Symbol`              | `:grid_search` (default) or `:interpolate`               |
| `asset_choice_tol`           | `Float64`             | Tolerance for the `a'` search (`1e-8`)                   |
| `asset_choice_max_iter`      | `Int`                 | Iteration cap for that search (50)                       | 
| `qBorr`                      | `Float64`             | Price of borrowing, applied when `a' < 0` (0.90)         |
| `qSav`                       | `Float64`             | Price of saving, applied when `a' ≥ 0` (0.99)            |
| `qGov`                       | `Float64`             | Government discount price in gbc (0.99)                  |
| `G`                          | `Float64`             | Government spending per period; gbc right-hand side (0.0)|
| `hMin`                       | `Float64`             | Lower bound on hours (`1e-8`)                            |
| `hMax`                       | `Float64`             | Upper bound on hours (5.0)                               |
| `labor_grid_size`            | `Int`                 | Points in `h_grid`; used by `labor_solver = :grid` (151) |
| `h_grid`                     | `Vec{Float64}`        | Hours grid; `labor_grid_size` points (151)               |
| `h_grid_income_power`        | `Vec{Float64}`        | Precomputed `h^(1-tau)`                                  |
| `h_grid_disutility`          | `Vec{Float64}`        | Precomputed `phi h^(1+eta)/(1+eta)`                      |
| `labor_solver`               | `Symbol`              | `:hybrid_newton` (default), `:brent`, or `:grid`         |
| `lambdaMin`                  | `Float64`             | Lower end of the `lambda` bracket (0.20)                 |
| `lambdaMax`                  | `Float64`             | Upper end of the `lambda` bracket (2.50)                 |
| `nLambdaSearch`              | `Int`                 | Grid points in the fallback search (15)                  |
| `maxIterLambda`              | `Int`                 | Iteration cap on the root find (60)                      |
| `tolLambda`                  | `Float64`             | Tolerance on `lambda` itself (`1e-5`)                    |
| `tolGovBudget`               | `Float64`             | Tolerance on gbc residual (`1e-5`)                       |
| `verbose`                    | `Bool`                | Print the solver header and progress (`true`)            |
| `printEveryLambda`           | `Int`                 | Print every k-th `lambda` evaluation; `0` silences (1)   |
| `massTol`                    | `Float64`             | Mass below which a cell is dropped (`1e-14`)             |
| `store_solutions`            | `Bool`                | Attach policy functions as `eq.solutions` (`false`)      |
| `collect_distributions`      | `Bool`                | Populate `unconditionalDistributions` (`true`)           |

# 3. Benchmark Calibration
Calibration with the default settings. The transcript below is reproducible:
after the Section 1 setup it is the output of `calibrate_history_independent_tax()`,
regenerated against the current code and matching line for line apart from the
two wall-clock timings. Running the file as a script produces the same log and
archives it in `calibration_results/` automatically.

Note the `WARNING: upper bound is binding` in the solver log. At the calibrated
point a mass share of `3.87e-7` sits exactly at `aMax = 15`, with zero slack. The
share is far too small to move the targeted moments, but `aMax` is touching and
should be raised before using this calibration for anything sensitive to the
right tail of the asset distribution.

```
=== Calibration targets ===
mean assets / mean labor income          = 0.58800000
true borrowing limit / mean labor income = 0.18000000
share negative liquid assets             = 0.26000000

=== Calibration search ===
start:   qSav=0.990000 qBorr=0.900000 bbar=-0.200000

eval   qSav        qBorr       bbar        mean/LI     trueBL/LI   neg share max
   1  0.99000000  0.90000000  -0.20000000  0.29748125  0.21121353  0.15616924 3e-01
   2  0.97551787  1.00447969  -0.17042398  0.47461467  0.18014963  0.25999991 1e-01
   3  0.97274828  1.01161161  -0.17028271  0.57398997  0.18012487  0.25990744 1e-02
   4  0.97240861  1.01223082  -0.17016462  0.58590848  0.18001712  0.25996064 2e-03
   5  0.97234663  1.01234731  -0.17014843  0.58750135  0.18000194  0.25993506 5e-04

=== History-independent tax finite-horizon solver ===
Options:
  age dimension J             = 39
  shock grid dimension nZ     = 5
  shock grid dimension nEps   = 5
  shock grid dimension nKappa = 3
  z_discretization_method     = :rouwenhorst  (alternatives: :rouwenhorst, :tauchen)
  tauchen_width               = 3.000  (used when z_discretization_method = :tauchen)
  asset grid dimension nA     = 101
  asset_grid_method           = :nonuniform  (alternatives: :nonuniform, :linear)
  asset_grid_borrow_share     = 0.350
  asset_grid_curvatures       = borrow 1.800, save 2.500
  asset_grid_zero_band        = share 0.300, width 0.080
  asset_grid_bounds           = [-0.552187, 15.000000], bbar = -0.170148
  asset_choice_method         = :grid_search  (alternatives: :grid_search, :interpolate)
  labor_solver                = :hybrid_newton  (alternatives: :brent, :hybrid_newton, :grid)
  labor_bounds                = [1.00e-08, 5.0000]
  terminal_borrowing          = :zero
  lambda_solver               = :brent  (Roots.jl; fallback: grid search over 15 values)
  lambda_bracket              = [0.200000, 2.500000], tol = 1.00e-05
  collect_distributions       = true

lambda = 0.20000000: residual = 2.42257155e-01
lambda = 2.50000000: residual = -4.54330338e-01
lambda eval 1: lambda=0.99988725, residual=6.60182176e-04
lambda eval 2: lambda=1.00206983, residual=1.59663949e-06
lambda eval 3: lambda=1.00207513, residual=-6.19476248e-08
lambda root: lambda=1.00207513, residual=-6.19476248e-08
WARNING: upper bound is binding.
  asset upper bound       = BINDING (share = 3.86727009e-07, bound = 15.00000000, material max a' = 15.00000000, slack = 0.00000000e+00)
total solve time          = 1.397 seconds

=== Calibration result ===
converged                = true (after 4 sweep(s), maxgap=4.99e-04)
qSav                     = 0.97234663
qBorr                    = 1.01234731
bbar                     = -0.17014843
mean A / mean Y          = 0.58750135  (target 0.588000, resid -4.99e-04)
true borr lim / mean Y   = 0.18000194  (target 0.180000, resid  1.94e-06)
share negative liquid A  = 0.25993506  (target 0.260000, resid -6.49e-05)
median A / mean Y        = 0.23948908  (not targeted)

=== Welfare ===
overall value function utility = -0.3595290378
overall simulation utility     = -0.3595290378
overall difference             = 8.42659276e-14
kappa      prob        value function  simulation     difference
-0.600954  0.16666667  -0.7226366759  -0.7226366759   4.28324043e-13
-0.050500  0.66666667  -0.3595210399  -0.3595210399   1.64868119e-14
 0.499954  0.16666667   0.0035466088   0.0035466088   1.13281780e-14
model solves             = 144
calibration time         = 255.949 seconds
```
