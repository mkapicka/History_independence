using Printf
using Statistics

include("solve_history_independent_tax.jl")
include("plot_history_independent_tax.jl")
include("model_settings.jl")

p = make_history_independent_params()

eq, _ = solve_history_independent_tax(p)
asset_grid_path = save_asset_grid_figure(p)
figure_paths = save_unconditional_distribution_figures(eq)

@printf("\n=== Final history-independent equilibrium ===\n")
@printf("lambda                     = %.8f\n", eq.lambda)
@printf("government budget residual = %.8e\n", eq.govBudgetResidual)
@printf("PV output                  = %.8f\n", eq.outputPV)
@printf("PV consumption             = %.8f\n", eq.consumptionPV)
@printf("mean output                = %.8f\n", mean(eq.Y))
@printf("mean consumption           = %.8f\n", mean(eq.C))
@printf("terminal assets            = %.8f\n", eq.A[end])
@printf("solve time                 = %.3f seconds\n", eq.elapsedSeconds)

s = eq.statistics
@printf("\n=== Aggregate statistics ===\n")
@printf("mean assets / mean labor income          = %.8f\n", s.meanAssetsToMeanLaborIncome)
@printf("median assets / mean labor income        = %.8f\n", s.medianAssetsToMeanLaborIncome)
@printf("true borrowing limit / mean labor income = %.8f\n", s.meanBorrowingLimitToMeanLaborIncome)
@printf("grid borrowing limit / mean labor income = %.8f\n", s.meanEffectiveGridBorrowingLimitToMeanLaborIncome)
@printf("share negative liquid assets             = %.8f\n", s.shareNegativeLiquidAssets)
@printf("share at effective grid borrowing bound  = %.8f\n", s.shareAtEffectiveBorrowingConstraint)
@printf("share with zero assets                   = %.8f\n", s.shareZeroAssets)
@printf("share at upper asset bound               = %.8f\n", s.shareAtAssetUpperBound)
@printf("share at hours upper bound               = %.8f\n", s.shareAtHoursUpperBound)

w = eq.welfare
@printf("\n=== Welfare ===\n")
@printf("overall value function utility = %.10f\n", w.overallValueFunction)
@printf("overall simulation utility     = %.10f\n", w.overallSimulation)
@printf("overall difference             = %.8e\n", w.overallDifference)
@printf("max abs difference by kappa    = %.8e\n", w.maxAbsDifferenceByKappa)
@printf("kappa        prob         value function    simulation        difference\n")
for ik in eachindex(w.kappaGrid)
    @printf("% .6f  %.8f  % .10f  % .10f  % .8e\n",
            w.kappaGrid[ik], w.kappaProbabilities[ik],
            w.valueFunctionByKappa[ik], w.simulationByKappa[ik],
            w.differenceByKappa[ik])
end

if s.upperBoundsBinding
    @printf("\n=== Upper-bound warning ===\n")
    if s.assetUpperBoundBinding
        @printf("asset upper bound binding: bound = %.8f, material max a' = %.8f, slack = %.8e\n",
                s.assetUpperBound, s.maxMaterialNextAssets, s.assetUpperBoundSlack)
    end
    if s.hoursUpperBoundBinding
        @printf("hours upper bound binding: bound = %.8f, material max h = %.8f, slack = %.8e\n",
                s.hoursUpperBound, s.maxMaterialHours, s.hoursUpperBoundSlack)
    end
end

@printf("\n=== Figures saved ===\n")
@printf("asset grid                = %s\n", display_path(asset_grid_path))
@printf("assets                    = %s\n", display_path(figure_paths.assets))
@printf("hours worked              = %s\n", display_path(figure_paths.hours))
@printf("consumption               = %s\n", display_path(figure_paths.consumption))
