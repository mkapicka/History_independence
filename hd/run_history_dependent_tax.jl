using JLD2
using Printf

include("solve_history_dependent_tax.jl")   # defines module HistoryDependentTax
                                            # (loads model_settings.jl internally)

using .HistoryDependentTax

"""
    run_history_dependent_tax(; kwargs...)

Solve the history-dependent tax model and print the equilibrium summary.
`kwargs` override `HD_SETTINGS` (e.g.
`run_history_dependent_tax(nA = 41, J = 4, mu1 = 0.0)`).

Returns a NamedTuple `(; eq, params)` where `eq` is the equilibrium and
`params` is the `HDParams` used. The history-independent limit
`mu1 = mu2 = 0` (with `nS1 = nS2 = 1`) is handled by this solver directly;
`check_history_independent_limit()` runs it with the standard consistency
checks.
"""
function run_history_dependent_tax(; kwargs...)
    p = make_history_dependent_params(; kwargs...)

    eq = solve_history_dependent_tax(p)

    print_hd_equilibrium_summary(eq, p)

    return (; eq = eq, params = p)
end

"""
    save_hd_result(result; dir = joinpath(@__DIR__, "results"))

Save a `run_history_dependent_tax()` result to `dir` under a filename built
from the model dimensions (shell-safe: no spaces or commas), e.g.
`result_mu1=0.300_mu2=0.050_J=39_nA=101_nS1=7_nS2=7_nZ=5_nEps=5_nKappa=3.jld2`.
Overwrites an existing file of the same name. Returns the saved path.
Reload with `load_hd_result(path)` (after including this file, so the
`HistoryDependentTax` module needed to reconstruct `HDParams` is defined).
"""
function save_hd_result(result; dir = joinpath(@__DIR__, "results"))
    mkpath(dir)
    p = result.params
    name = @sprintf(
        "result_mu1=%.3f_mu2=%.3f_J=%d_nA=%d_nS1=%d_nS2=%d_nZ=%d_nEps=%d_nKappa=%d.jld2",
        p.mu1, p.mu2, p.J, length(p.a_grid),
        length(p.s1_grid), length(p.s2_grid),
        length(p.z_grid), length(p.eps_grid), length(p.kappa_grid))
    path = joinpath(dir, name)
    jldsave(path; eq = result.eq, params = p)
    return path
end

"""
    load_hd_result(path)

Load a result saved by `save_hd_result`. Returns `(; eq, params)`.
"""
function load_hd_result(path::AbstractString)
    data = JLD2.load(path)
    return (; eq = data["eq"], params = data["params"])
end

# Script entry point: `julia -t 3 run_history_dependent_tax.jl` still works;
# from the REPL, include this file and call `result = run_history_dependent_tax()`.
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_history_dependent_tax()
end
