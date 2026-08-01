using JLD2
using Printf

include("solve_history_independent_tax.jl")
include("model_settings.jl")

"""
    run_history_independent_tax(; kwargs...)

Solve the history-independent tax model and print the
equilibrium summary. `kwargs` override `SETTINGS` (e.g.
`run_history_independent_tax(nA = 41, J = 4, qSav = 0.99)`).

Returns a NamedTuple `(; eq, params)` where `eq` is the equilibrium
(with policies under `eq.solutions` when `store_solutions = true`)
and `params` is the `HIParams` used.
"""
function run_history_independent_tax(; kwargs...)
    p = make_history_independent_params(; kwargs...)

    eq = solve_history_independent_tax(p)

    print_equilibrium_summary(eq, p)

    return (; eq = eq, params = p)
end

"""
    save_hi_result(result; dir = joinpath(@__DIR__, "results"))

Save a `run_history_independent_tax()` result to `dir` under a filename built
from the model dimensions (shell-safe: no spaces or commas), e.g.
`result_hi_J=39_nA=101_nZ=5_nEps=5_nKappa=3.jld2`. Overwrites an existing
file of the same name. Returns the saved path. Reload with
`load_hi_result(path)` (after including this file, so `HIParams` is defined).
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
