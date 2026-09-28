# =============================================================================
# run_history_independent_tax.jl
#
# Entry point for the history-independent tax model. Builds a parameter set
# from SETTINGS, solves it, prints the summary, and saves or reloads results.
#
# Marek Kapicka, 2026
# =============================================================================

using JLD2
using Printf

include("solve_history_independent_tax.jl")
include("model_settings.jl")

"""
    run_history_independent_tax(; kwargs...)

Solve the model and print the equilibrium summary. `kwargs` override SETTINGS.
Returns `(; eq, params)`; policies sit under `eq.solutions` when
`store_solutions = true`.
"""
function run_history_independent_tax(; kwargs...)
    # collect_distributions is on by default here and off in the calibration
    # drivers and the sweeps, where the per-observation vectors are too large.
    p = make_history_independent_params(; collect_distributions = true, kwargs...)

    eq = solve_history_independent_tax(p)

    print_equilibrium_summary(eq, p)

    return (; eq = eq, params = p)
end

"""
    save_hi_result(result; dir = joinpath(@__DIR__, "results"))

Save a result to `dir` under a name built from the model dimensions. Overwrites
an existing file of the same name and returns the saved path.
"""
function save_hi_result(result; dir = joinpath(@__DIR__, "results"))
    mkpath(dir)
    p = result.params
    name = @sprintf("result_hi_J=%d_nA=%d_nZ=%d_nEps=%d_nKappa=%d.jld2",
                    p.J, length(p.a_grid), length(p.z_grid),
                    length(p.eps_grid), length(p.kappa_grid))
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
    run_history_independent_tax()
end
