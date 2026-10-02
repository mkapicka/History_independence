# Bewley solvers

Incomplete-markets life-cycle and stationary models with an HSV progressive income
tax, solved under history-independent and history-dependent taxation. Julia.

## The eight directories

The solver directories are the 2×2×2 product of three binary axes. Nothing else
distinguishes them.

| axis | values |
|---|---|
| **tax** | history-**i**ndependent (`hi*`) / history-**d**ependent (`hd*`, adds past-income stocks s1, s2 and a joint (a′, h) choice) |
| **horizon** | finite, `J = 39` / **inf**inite, a forward pass to `maxAge = 600` |
| **access** | all households save / **htm**, an exogenous two-state asset-market-access chain |

|  | finite | infinite |
|---|---|---|
| history-independent | `hi` (670) | `hiinf` (741) |
| history-independent + HtM | `hi_htm` (811) | `hiinf_htm` (1,039) |
| history-dependent | `hd` (635) | `hdinf` (1,076) |
| history-dependent + HtM | `hd_htm` (900) | `hdinf_htm` (1,370) |

Line counts are `solve.jl` only. Every directory holds the same seven files,
plus one `calibrate_*.jl` per calibration exercise:

| file | role |
|---|---|
| `model_settings.jl` | the values, and the only place they are set |
| `params.jl` | the parameter object: declaration, constructor, validation |
| `solve.jl` | the model: `solve_hi(p)` / `solve_hd(p)` -> equilibrium |
| `statistics.jl` | the accumulator and the per-cell update |
| `report.jl` | this variant's own printers |
| `main.jl` | **entry point**: settings -> solve -> print -> save |
| `plot.jl` | figures from a result: `plot_hi(r)` / `plot_hd(r)` |
| `calibrate_*.jl` | the other way of using the solver, calling `solve_hi(p)` directly |

`main.jl` is where to start; `solve.jl` is a pure function of a parameter object
and is what the calibration drivers call directly, 19 times across the hi family.

## The other directories

| | |
|---|---|
| `common/` | **`BewleyCommon`**, the shared package: 13 files, 2,172 lines. Everything the eight directories genuinely share. Has its own test suite. |
| `verify/` | the golden-master harness and the committed baselines |
| `LaTeX/` | the companion write-up |
| `backup/` | a superseded solver, kept out of habit rather than need |

## Running one solve

Each directory is its own Julia environment, so the project must point at it.
**Pin the version**: every `Manifest.toml` here is resolved for **Julia 1.12.6**,
which is the newest the MetaCentrum cluster offers. Resolving under 1.13 rewrites
them to a format 1.12.6 cannot use.

```bash
julia +1.12.6 --project=hi -t 4
```
```julia
include("hi/main.jl")
r = main_hi()                      # settings defaults
r = main_hi(; nA = 151, nZ = 15)   # overrides
r.eq.statistics.meanAssetsToMeanLaborIncome
```

`model_settings.jl` is the single source of truth for a directory: the params
struct carries no defaults of its own, so a value is set there or passed
explicitly. A verbose run prints its inputs — every one with full precision, so a
calibrated value can be copied back verbatim — then the equilibrium, the
statistics over the calibration window and over all ages, the entry-age
cross-section, and the welfare decomposition.

### Calibrating

Each file is named for the instruments it moves, not for the model:

| file | instruments | targets |
|---|---|---|
| `calibrate_twoprice.jl` | `qSav`, `qBorr`, `bbar` | assets, borrowing limit, share with negative assets |
| `calibrate_oneprice.jl` | `q = qSav = qBorr = qGov`, `bbar` | assets, borrowing limit (`hiinf` only) |
| `calibrate_beta.jl` | `beta`, `bbar`, prices GIVEN | assets, borrowing limit |

```julia
include("hi/calibrate_beta.jl")      # also pulls in calibrate_twoprice.jl for its helpers
r = calibrate_beta()
r.beta, r.bbar
```

Each writes a timestamped transcript to `<dir>/calibration_results/`. The
calibrated values print with every digit, so they can be pasted into
`model_settings.jl` without loss -- they were truncated at `%.8f`/`%.10f` until
2026-10-01, so a value copied from an older transcript is already rounded.

## Testing

Two layers, and they do different jobs.

**Unit tests** pin the contracts of the shared package. Seconds.

```bash
julia +1.12.6 --project=common -e 'using Pkg; Pkg.test()'
```

**The golden master** solves each directory at small grids and compares every
scalar, and the printed output, against a committed baseline. It answers "did any
number change?", which is what a refactoring needs.

```bash
./verify/run.sh mytag                 # all eight
./verify/run.sh mytag hi hiinf        # or a subset
diff -r verify/baseline verify/mytag  # nothing printed means nothing changed
```

Two things to know before relying on it. **It is not uniformly cheap**: the six
directories `hi hi_htm hiinf hiinf_htm hd hd_htm` finish in about two minutes
together, but `hdinf` and `hdinf_htm` take roughly 45 minutes **each**. Verify the
fast six first and launch the slow pair in the background. And **do not edit a
solver, `common/src`, or a Manifest while a sweep is in flight** — Julia reads
them at process start, so a directory that has not begun yet picks up the edit and
verifies the wrong thing.

Two further checks the harness cannot replace:

```bash
julia +1.12.6 --project=hd verify/unresolved.jl   # a lifted function still naming a solver-local global
julia +1.12.6 verify/consumers.jl                # a statistics name a sweep or plot script reads
```

The default asset choice is `:grid_search`. The `:interpolate` path has its own
baseline, `verify/baseline-interpolate/`, covering `hi`, `hiinf` and `hiinf_htm`
only — `hi_htm` throws a `DomainError` under `:interpolate`, which is a
pre-existing bug and not a regression. That tag is also sensitive to
recompilation: see the caveat at the top of `verify/golden.jl` before reading a
diff on it as a fault.

## Cluster runs

`hd*/sweep.sh` are PBS Pro array jobs for MetaCentrum. They load
`julia/1.12.6` explicitly, because a bare `module add julia` gives 1.7.0 there.

## Where the shared code sits

`BewleyCommon` is organised in three tiers, which is also the order of its
includes:

- **generic** — `params.jl`, `grids.jl`, `shocks.jl`: the parameter supertype and
  its accessors, asset grids, shock discretization, lookups.
- **history-independent family** — `labor.jl`, `assets.jl`, `asset_choice.jl`,
  `values.jl`, `access.jl`: the static labor FOC, the Young lottery, the
  `:interpolate` asset choice, continuation values, the access chain.
- **all eight** — `statistics.jl`, `report.jl`: the published statistics and the
  printers.

Plus `hi_model.jl` and `hd_model.jl`, which hold what one family shares but the
other does not. **Nothing in those two is exported**: several names have a
different implementation in a sibling directory, and a directory that both
`using`s an exported name and defines its own is a Julia error. The directories
that want them import by name, which also documents at the top of each solver
exactly what it takes from the package.

What is deliberately **not** shared: the forward pass, the policy solve and the λ
solver. Measured, they differ between neighbouring variants across 14–23 separate
hunks, and the horizon axis changes the rank of the policy arrays in the hottest
loop. `hi/NOTES.md` and the step 7 commit carry the numbers.

## NOTES.md

One per directory. Each entry records a number that settled a choice and is not
recoverable from the code — a measured error, a comparison against a closed form,
a calibration target and its source. Read these before changing a tolerance or a
grid.
