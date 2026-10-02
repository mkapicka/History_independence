# =============================================================================
# main.jl
#
# Entry point: builds a parameter set from the settings, solves it,
# prints the summary, and saves or reloads results.
# For the history-independent tax model with hand-to-mouth agents.
#
# Marek Kapicka, 2026
# =============================================================================

using JLD2
using Printf

include("solve.jl")
include("model_settings.jl")

"""
    main_hi(; kwargs...)

Solve the history-independent tax model and print the
equilibrium summary. `kwargs` override `SETTINGS` (e.g.
`main_hi(nA = 41, J = 4, qSav = 0.99)`).

Returns a NamedTuple `(; eq, params)` where `eq` is the equilibrium
(with policies under `eq.solutions` when `store_solutions = true`)
and `params` is the `HIParams` used.
"""
function main_hi(; kwargs...)
    # collect_distributions defaults to TRUE here: an interactive run is
    # normally headed for `plot_hi`, whose hours and
    # consumption histograms read the per-observation vectors and come back
    # empty without them. It sits BEFORE the splat, so `kwargs` can still turn
    # it off -- which is what the calibration drivers and the sweeps do.
    p = make_history_independent_params(; collect_distributions = true, kwargs...)

    eq = solve_hi(p)

    print_equilibrium_summary(eq, p)

    return (; eq = eq, params = p)
end

"""
    save_hi_result(result; dir = joinpath(@__DIR__, "results"))

Save a `main_hi()` result to `dir` under a filename built
from the model dimensions (shell-safe: no spaces or commas), e.g.
`result_hi_J=39_nA=101_nZ=5_nEps=5_nKappa=3.jld2`. Overwrites an existing
file of the same name. Returns the saved path. Reload with
`load_hi_result(path)` (after including this file, so `HIParams` is defined).
"""
function save_hi_result(result; dir = joinpath(@__DIR__, "results"))
    mkpath(dir)
    p = result.params
    # pSS/pHH are in the name: two runs differing only in the access chain are
    # different models and must not overwrite each other.
    name = @sprintf("result_hi_htm_J=%d_nA=%d_nZ=%d_nEps=%d_nKappa=%d_pSS=%.4f_pHH=%.4f.jld2",
                    p.J, length(p.a_grid), length(p.z_grid),
                    length(p.eps_grid), length(p.kappa_grid), p.pSS, p.pHH)
    path = joinpath(dir, name)
    jldsave(path; eq = result.eq, params = p)
    return path
end

"""
    load_hi_result(path)

Load a result saved by `save_hi_result`. Returns `(; eq, params)`.
"""
function load_hi_result(path::AbstractString)
    data = JLD2.load(path)
    return (; eq = data["eq"], params = data["params"])
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main_hi()
end
