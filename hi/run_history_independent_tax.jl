# result = run_history_independent_tax(labor_solver = :hybrid_newton, J=39, nA=101, nZ=5, nEps=5, nKappa=3, hMin = 0.5, hMax = 1.25, labor_grid_size = 101);
using Printf

include("solve_history_independent_tax.jl")
include("model_settings.jl")

"""
    run_history_independent_tax(; kwargs...)

Solve the history-independent tax model and print the
equilibrium summary. `kwargs` override `SETTINGS` (e.g.
`run_history_independent_tax(nA = 41, J = 4, qSav = 0.99)`).

Returns a NamedTuple `(; eq, sol, params)` where `eq` is the
equilibrium, `sol` contains policy objects when `store_solutions = true`,
and `params` is the `HIParams` used.
"""
function run_history_independent_tax(; kwargs...)
    p = make_history_independent_params(; kwargs...)

    eq, sol = solve_history_independent_tax(p)

    print_equilibrium_summary(eq, p)

    return (; eq = eq, sol = sol, params = p)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_history_independent_tax()
end
