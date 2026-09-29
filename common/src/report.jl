# =============================================================================
# report.jl
#
# Printing shared by every solver. Only the genuinely identical blocks live
# here: the per-variant summaries stay with their solvers, because they report
# different quantities (the access chain for the htm variants, maxAge for the
# infinite-horizon ones).
#
# print_aggregate_statistics is NOT here, although all eight define it. It has
# eight distinct versions, which is a symptom of the statistics not carrying
# their own labels rather than a printing problem; it dissolves when the
# statistics registry lands.
# =============================================================================

function print_welfare_summary(w)
    @printf("\n=== Welfare ===\n")
    @printf("overall value function utility = %.10f\n", w.overallValueFunction)
    @printf("overall simulation utility     = %.10f\n", w.overallSimulation)
    @printf("overall difference             = %.8e\n", w.overallDifference)
    @printf("kappa      prob        value function  simulation     difference\n")
    for ik in eachindex(w.kappaGrid)
        @printf("% .6f  %.8f  % .10f  % .10f  % .8e\n",
                w.kappaGrid[ik], w.kappaProbabilities[ik],
                w.valueFunctionByKappa[ik], w.simulationByKappa[ik],
                w.differenceByKappa[ik])
    end
    return nothing
end

function print_lambda_warnings(eq)
    has_warning = (hasproperty(eq, :converged) && eq.converged === false) ||
                  eq_flag(eq, :bracketWarning) ||
                  eq_flag(eq, :rootResidualWarning) ||
                  eq_flag(eq, :rootSolverWarning)
    has_warning || return nothing

    println("WARNING: lambda solver returned an approximate solution.")
    if eq_flag(eq, :bracketWarning)
        @printf("  no sign change was found; using best grid-search lambda %.8f with residual %.8e\n",
                eq.lambda, eq.govBudgetResidual)
    end
    if eq_flag(eq, :rootResidualWarning)
        @printf("  root residual %.8e exceeds tolerance %.8e\n",
                abs(eq.govBudgetResidual), eq.parameters.tolGovBudget)
    end
    if eq_flag(eq, :rootSolverWarning)
        @printf("  Brent solver failed; using best evaluated lambda %.8f with residual %.8e\n",
                eq.lambda, eq.govBudgetResidual)
    end
    flush(stdout)
end
