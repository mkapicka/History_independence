# 1. Setup
Load the calibration file and activate the project environment. The `include` command loads the calibration script, which defines the calibration function and its dependencies.

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
Calibrate the model to match the three targets in `CalibrationTargets` by adjusting the three instruments `qSav`, `qBorr`, and `bbar`. The calibration function  searches over the three instruments, solving the model at each step to evaluate the moments. The calibration stops when all three moment residuals are below `config.moment_tol`, or when it has used `config.outer_max_sweeps` block-coordinate sweeps, whichever comes first; only the former sets `converged = true`. Labor supply is solved for at each step using the specified `labor_solver`. The calibration function returns a `NamedTuple` with the calibrated instruments, the achieved moments, and other information about the calibration.

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
- `asset_moment`: `:mean` (default); `:median`.
- Calibration targets (switch median for mean according to `asset_moment`):
    - mean assets / mean labor income: `meanAssetsToMeanLaborIncome` (default = 0.588)
    - true borrowing limit / mean labor income: `trueBorrowingLimitToMeanLaborIncome` (default = 0.18)
    - share of negative liquid assets: `shareNegativeLiquidAssets` (default = 0.26)
- `labor_solver`: `:hybrid_newton` (default), `:brent`, `:grid`.
- `aMax`: upper bound of the asset grid (default: `15.0`)

Any other `SETTINGS` entry may be passed the same way, **except** `qSav`, `qBorr`,
`bbar`, and `verbose`: those are set by the calibration itself and passing them
raises an error.

The search itself is controlled by a separate `config::CalibrationConfig`
keyword, not by the model keywords above:

```julia
result = calibrate_history_independent_tax(
    config = CalibrationConfig(outer_max_sweeps = 6, moment_tol = 1e-4))
```

| Field                                   | Meaning                                                     |
|-----------------------------------------|-------------------------------------------------------------|
| `moment_tol` (`5e-4`)                   | Stop when the largest absolute moment residual falls below it|
| `outer_max_sweeps` (`12`)               | Cap on block-coordinate sweeps                              |
| `qSav_min`/`qSav_max` (`0.900`/`1.040`) | Bracket searched for `qSav`                                 |
| `qBorr_min`/`qBorr_max` (`0.700`/`1.040`)| Bracket searched for `qBorr`                                |
| `bbar_min`/`bbar_max` (`-0.80`/`-0.01`) | Bracket searched for `bbar`                                 |
| `inner_xtol` (`1e-5`), `inner_maxevals` (`40`) | Tolerance and evaluation cap of each 1-D Brent solve  |
| `lambda_warm_start` (`true`), `lambda_warm_width` (`0.10`) | Bracket `lambda` near the previous root   |
| `shrink_brackets` (`true`), `bracket_shrink` (`0.15`) | Narrow each instrument bracket from sweep 2 on |
| `final_resolve` (`true`)                | Re-solve once at the calibrated point with full solver output|
| `verbose` (`true`)                      | Print the calibration's own log (inner solves are always silent)|

## Result structure
`result` is a `NamedTuple`. Throughout, `nAge = J + 1 = 40`: age-indexed vectors `1, …, J+1`, correspond to `j = 0, …, J`
Some fields are not documented here.

### `result` — top level
| Field            | Type                 | Meaning                                                   |
|------------------|----------------------|-----------------------------------------------------------|
| `qSav`           | `Float64`            | Calibrated saving price                                   |
| `qBorr`          | `Float64`            | Calibrated borrowing price                                |
| `bbar`           | `Float64`            | Calibrated borrowing limit (`≤ 0`; `B > 0` is `bbar = -B`)|
| `eq`             | `NamedTuple`         | Equilibrium at the calibrated instruments (see below)     |
| `moments`        | `NamedTuple`         | Achieved moments (see below)                              |
| `residuals`      | `NamedTuple`         | Achieved minus target, moment by moment                   |
| `converged`      | `Bool`               | `true` if `max|residual| ≤ config.moment_tol`             |
| `sweeps`         | `Int`                | Block-coordinate sweeps used                              |
| `nSolves`        | `Int`                | Full model solves performed                               |
| `elapsedSeconds` | `Float64`            | Wall-clock time of the calibration                        |
| `targets`        | `CalibrationTargets` | Targets the calibration was run against                   |
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
| `parameters`         | `HIParams`          | Parameters used for this solve                               |
| `converged`          | `Bool`              | `true` if `|govBudgetResidual| ≤ tolGovBudget`               |
| `bracketWarning`     | `Bool`              | `true` if no sign change was found in the `lambda` bracket   |
| `rootResidualWarning`| `Bool`              | `true` if Brent returned but the resid missed `tolGovBudget` |
| `elapsedSeconds`     | `Float64`           | Time of this single solve                                    |

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
| `maxNextAssets`, `maxHours`                        | Largest `a'` and `h` chosen                            |
| `maxMaterialNextAssets`, `maxMaterialHours`        | Same, over cells whose mass exceeds `1e-8`             |
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

## `result.eq.welfare`
| Field                                      | Type                  | Meaning                                             |
|--------------------------------------------|-----------------------|-----------------------------------------------------|
| `kappaGrid`, `kappaProbabilities`          | `Vec{Float64, nKappa}`| Permanent-shock grid and its probabilities          |
| `valueFunctionByKappa`                     | `Vec{Float64}`        | Expected utility from the value function at `j = 0` |
| `simulationByKappa`                        | `Vec{Float64}`        | Same object accumulated through the simulation      |
| `differenceByKappa`                        | `Vec{Float64}`        | Simulation minus value function; a check            |
| `overallValueFunction`, `overallSimulation`| `Float64`             | `Pkappa`-weighted aggregates                        |
| `overallDifference`                        | `Float64`             | Their difference; should be round-off, of order `1e-13` |
| `maxAbsDifferenceByKappa`                  | `Float64`             | Worst per-`kappa` discrepancy                       |

## `result.moments` — the four achieved ratios
| Field                                 | Type      | Meaning                                                          |
|---------------------------------------|-----------|------------------------------------------------------------------|
| `medianAssetsToMeanLaborIncome`       | `Float64` | Median assets / mean labor income                                |
| `meanAssetsToMeanLaborIncome`         | `Float64` | Mean assets / mean labor income                                  |
| `trueBorrowingLimitToMeanLaborIncome` | `Float64` | Copied from `eq.statistics.meanBorrowingLimitToMeanLaborIncome`  |
| `shareNegativeLiquidAssets`           | `Float64` | Share of agents with `a < 0`                                     |

## `result.residuals` — the three targeted gaps, achieved minus target
| Field                                 | Type      | Meaning                                                          |
|---------------------------------------|-----------|------------------------------------------------------------------|
| `assetsToMeanLaborIncome`             | `Float64` | Whichever asset ratio `asset_moment` selects                     |
| `trueBorrowingLimitToMeanLaborIncome` | `Float64` | Gap on the borrowing-limit ratio                                 |
| `shareNegativeLiquidAssets`           | `Float64` | Gap on the share of agents with `a < 0`                          |

## `result.targets` — the `CalibrationTargets` struct
Only the asset ratio picked by `asset_moment` is targeted; the other is reported but left free.
| Field                                 | Type      | Meaning                                                          |
|---------------------------------------|-----------|------------------------------------------------------------------|
| `medianAssetsToMeanLaborIncome`       | `Float64` | Target for median assets / mean labor income                     |
| `meanAssetsToMeanLaborIncome`         | `Float64` | Target for mean assets / mean labor income                       |
| `trueBorrowingLimitToMeanLaborIncome` | `Float64` | Target for the borrowing-limit ratio                             |
| `shareNegativeLiquidAssets`           | `Float64` | Target for the share of agents with `a < 0`                      |
| `asset_moment`                        | `Symbol`  | `:median` or `:mean`; picks which asset ratio is targeted        |

## `result.params` — the calibrated `HIParams`
Defaults in parentheses are the values in `SETTINGS` (`model_settings.jl`), which
is the only place model parameters are defined; `HIParams` itself carries no
defaults for them. Four fields (`maxIterLambda`, `massTol`, and the derived
`omega_mean`/`epsilon_mean`/`kappa_mean`) are not in `SETTINGS` and default
inside the struct.
| Field                        | Type                  | Meaning                                                |
|------------------------------|-----------------------|--------------------------------------------------------|
| `beta`                       | `Float64`             | Discount factor (0.96)                                 |
| `eta`                        | `Float64`             | Inverse Frisch elasticity  (2.0)                       |
| `phi`                        | `Float64`             | Level of the disutility of hours (1.0)                 |
| `tau`                        | `Float64`             | HSV progressivity (0.181)                              | 
| `J`                          | `Int`                 | Terminal age; `nAge = J+1` periods, `j = 0, …, J` (39) |
| `rho`                        | `Float64`             | Persistence of `z`; also scales borrowing limit (0.958)|
| `sigma_omega`                | `Float64`             | S.d. of the innovation to `z` (`sqrt(0.017)`)          |
| `sigma_epsilon`              | `Float64`             | S.d. of the transitory shock `eps` (`sqrt(0.081)`)     |
| `sigma_kappa`                | `Float64`             | S.d. of the permanent shock `kappa` (`sqrt(0.101)`)    |
| `omega_mean`                 | `Float64`             | Mean innovation to `z`; `-sigma_omega^2/2` (-0.0085)   |
| `epsilon_mean`               | `Float64`             | Mean of `eps`; `-sigma_epsilon^2/2` (-0.0405)          |
| `kappa_mean`                 | `Float64`             | Mean of `kappa`; `-sigma_kappa^2/2` (-0.0505)          |
| `nZ`, `nEps`, `nKappa`       | `Int`                 | Grid sizes for `z`, `eps`, `kappa` (5, 5, 3)           |
| `z_initial`                  | `Float64`             | Value of `z` conditioning the `j = 0` distribution (0.0)|
| `z_grid`                     | `Vec{Float64, nZ}`    | Persistent-productivity grid                           |
| `Pz`                         | `Mat{Float64, nZ×nZ}` | Transition matrix for `z`                              |
| `z0_probs`                   | `Vec{Float64, nZ}`    | Distribution of `z` at `j = 0`                         |
| `eps_grid`                   | `Vec{Float64, nEps}`  | Transitory-shock grid                                  |
| `Peps`                       | `Vec{Float64, nEps}`  | I.i.d. probabilities of `eps`                          |
| `kappa_grid`                 | `Vec{Float64, nKappa}`| Permanent-shock grid                                   |
| `Pkappa`                     | `Vec{Float64, nKappa}`| Probabilities of `kappa`                               |
| `z_discretization_method`    | `Symbol`              | `:rouwenhorst` (default) or `:tauchen`                 |
| `tauchen_width`              | `Float64`             | Grid width in s.d., used only for `:tauchen` (3.0)     |
| `bbar`                       | `Float64`             | Borrowing-limit scale, `≤ 0` (`B > 0` is `bbar = -B`)  |
| `aMax`                       | `Float64`             | Upper bound of the asset grid (15.0)                   |
| `nA`                         | `Int`                 | Number of asset-grid points (101)                      |
| `a_grid`                     | `Vec{Float64, nA}`    | The grid itself; contains `0.0` exactly                |
| `asset_grid_method`          | `Symbol`              | `:nonuniform` (default) or `:linear`                   |
| `asset_grid_curvature_borrow`| `Float64`             | Spacing curvature below zero (1.8)                     |
| `asset_grid_curvature_save`  | `Float64`             | Spacing curvature above zero (2.5)                     |
| `asset_grid_borrow_share`    | `Float64`             | Share of points on the borrowing side (0.35)           |
| `asset_grid_zero_share`      | `Float64`             | Share of points in the band around zero (0.30)         |
| `asset_grid_zero_width`      | `Float64`             | Width of that band (0.08)                              |
| `asset_choice_method`        | `Symbol`              | `:grid_search` (default) or `:interpolate`             |
| `asset_choice_tol`           | `Float64`             | Tolerance for the `a'` search (`1e-8`)                 |
| `asset_choice_max_iter`      | `Int`                 | Iteration cap for that search (50)                     |
| `qBorr`                      | `Float64`             | Price of borrowing, applied when `a' < 0` (0.90)       |
| `qSav`                       | `Float64`             | Price of saving, applied when `a' ≥ 0` (0.99)          |
| `qGov`                       | `Float64`             | Government discount price in gbc (0.99)                |
| `G`                          | `Float64`             | Government spending per period; gbc right-hand side (0.0)|
| `hMin`                       | `Float64`             | Lower bound on hours (`1e-8`)                          |
| `hMax`                       | `Float64`             | Upper bound on hours (5.0)                             |
| `labor_grid_size`            | `Int`                 | Points in `h_grid`; used by `labor_solver = :grid` (151)|
| `h_grid`                     | `Vec{Float64}`        | Hours grid; `labor_grid_size` points (151)             |
| `h_grid_income_power`        | `Vec{Float64}`        | Precomputed `h^(1-tau)`                                |
| `h_grid_disutility`          | `Vec{Float64}`        | Precomputed `phi h^(1+eta)/(1+eta)`                    |
| `labor_solver`               | `Symbol`              | `:hybrid_newton` (default), `:brent`, or `:grid`       |
| `lambdaMin`                  | `Float64`             | Lower end of the `lambda` bracket (0.20)               |
| `lambdaMax`                  | `Float64`             | Upper end of the `lambda` bracket (2.50)               |
| `nLambdaSearch`              | `Int`                 | Grid points in the fallback search (15)                |
| `maxIterLambda`              | `Int`                 | Iteration cap on the root find (60)                    |
| `tolLambda`                  | `Float64`             | Tolerance on `lambda` itself (`1e-5`)                  |
| `tolGovBudget`               | `Float64`             | Tolerance on gbc residual (`1e-5`)                     |
| `verbose`                    | `Bool`                | Print the solver header and progress (`true`)          |
| `printEveryLambda`           | `Int`                 | Print every k-th `lambda` evaluation; `0` silences (1) |
| `massTol`                    | `Float64`             | Mass below which a cell is dropped (`1e-14`)           |
| `store_solutions`            | `Bool`                | Keep policy functions in the result (`false`)          |
| `collect_distributions`      | `Bool`                | Populate `unconditionalDistributions` (`true`)         |


## 3. Benchmark Calibration
Calibration with the default settings. The transcript below is reproducible: it
was regenerated against the current code and matches line for line apart from the
two wall-clock timings.

Note the `WARNING: upper bound is binding` in the solver log. At the calibrated
point a mass share of `3.9e-7` sits exactly at `aMax = 15`, with zero slack. The
share is far too small to move the targeted moments, but `aMax` is touching and
should be raised before using this calibration for anything sensitive to the
right tail of the asset distribution.


=== Calibration targets ===
mean assets / mean labor income          = 0.58800000
true borrowing limit / mean labor income = 0.18000000
share negative liquid assets             = 0.26000000

=== Calibration search ===
start:   qSav=0.990000 qBorr=0.900000 bbar=-0.200000

eval   qSav        qBorr       bbar        mean/LI     trueBL/LI   neg share max
   1  0.99000000  0.90000000  -0.20000000  0.29748125  0.21121353  0.15616924 3e-01
   2  0.97551787  1.00447969  -0.17042398  0.47461467  0.18014963  0.25999991 1e-01
   3  0.97274804  1.01160653  -0.17028271  0.57399267  0.18012488  0.25990403 1e-02
   4  0.97240987  1.01222785  -0.17016462  0.58586051  0.18001705  0.25995736 2e-03
   5  0.97234571  1.01235421  -0.17014838  0.58757147  0.18000199  0.25994180 4e-04

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

lambda = 0.20000000: residual = 2.42256648e-01
lambda = 2.50000000: residual = -4.54330801e-01
lambda eval 1: lambda=0.99988563, residual=6.60432797e-04
lambda eval 2: lambda=1.00206904, residual=1.19497478e-06
lambda eval 3: lambda=1.00207300, residual=2.73600637e-09
lambda eval 4: lambda=1.00207301, residual=-8.91309679e-12
lambda root: lambda=1.00207301, residual=-8.91309679e-12
WARNING: upper bound is binding.
  asset upper bound       = BINDING (share = 3.89055482e-07, bound = 15.00000000, material max a' = 15.00000000, slack = 0.00000000e+00)
total solve time          = 2.125 seconds

=== Calibration result ===
converged                = true (after 4 sweep(s), maxgap=4.29e-04)
qSav                     = 0.97234571
qBorr                    = 1.01235421
bbar                     = -0.17014838
mean A / mean Y          = 0.58757147  (target 0.588000, resid -4.29e-04)
true borr lim / mean Y   = 0.18000199  (target 0.180000, resid  1.99e-06)
share negative liquid A  = 0.25994180  (target 0.260000, resid -5.82e-05)
median A / mean Y        = 0.23949389  (not targeted)

=== Welfare ===
overall value function utility = -0.3595301562
overall simulation utility     = -0.3595301562
overall difference             = 8.29336599e-14
kappa      prob        value function  simulation     difference
-0.600954  0.16666667  -0.7226378290  -0.7226378290   4.29878355e-13
-0.050500  0.66666667  -0.3595221565  -0.3595221565   1.39888101e-14
 0.499954  0.16666667   0.0035455175   0.0035455175   1.15723403e-14
model solves             = 127
calibration time         = 257.847 seconds
