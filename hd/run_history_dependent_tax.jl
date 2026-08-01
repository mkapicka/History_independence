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

# Script entry point: `julia -t 3 run_history_dependent_tax.jl` still works;
# from the REPL, include this file and call `result = run_history_dependent_tax()`.
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_history_dependent_tax()
end
