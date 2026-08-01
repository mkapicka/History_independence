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

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_history_independent_tax()
end
