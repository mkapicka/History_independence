using Printf
using Statistics

include("solve_history_independent_tax.jl")
include("plot_history_independent_tax.jl")

# User settings. Edit these blocks to change the calibration, grids, or solver.
const MODEL_DIMENSIONS = (;
    J = 12,
    nZ = 5,
    nEps = 5,
    nKappa = 3,
    nA = 61,
)

const PREFERENCE_AND_TAX_SETTINGS = (;
    beta = 0.960,
    eta = 2.0,
    phi = 1.0,
    tau = 0.181,
)

const SHOCK_PROCESS_SETTINGS = (;
    rho = 0.958,
    sigma_omega = sqrt(0.017),
    sigma_epsilon = sqrt(0.081),
    sigma_kappa = sqrt(0.065 + 0.036),
    z_discretization_method = :rouwenhorst,
    tauchen_width = 3.0,
    z_initial = 0.0,
)

const ASSET_GRID_SETTINGS = (;
    method = :nonuniform,
    borrow_share = 0.35,
    curvature_borrow = 1.8,
    curvature_save = 2.5,
    zero_share = 0.30,  # Set positive to reserve grid points near zero.
    zero_width = 0.08,  # Positive value creates a band around zero.
    bbar = -0.20,
    aMax = 12.0,
)

const FINANCIAL_AND_GOVERNMENT_SETTINGS = (;
    qBorr = 0.90,
    qSav = 0.99,
    qGov = 0.99,
    G = 0.0,
)

const LABOR_SETTINGS = (;
    hMin = 1e-8,
    hMax = 5.0,
    labor_method = :foc,
)

const LAMBDA_SOLVER_SETTINGS = (;
    lambdaMin = 0.20,
    lambdaMax = 2.50,
    nLambdaSearch = 15,
    tolGovBudget = 1e-5,
    tolLambda = 1e-5,
)

const OUTPUT_SETTINGS = (;
    verbose = true,
    printEveryLambda = 1,
    store_solutions = false,
)

p = HIParams(
    beta = PREFERENCE_AND_TAX_SETTINGS.beta,
    eta = PREFERENCE_AND_TAX_SETTINGS.eta,
    phi = PREFERENCE_AND_TAX_SETTINGS.phi,
    tau = PREFERENCE_AND_TAX_SETTINGS.tau,
    J = MODEL_DIMENSIONS.J,
    rho = SHOCK_PROCESS_SETTINGS.rho,
    sigma_omega = SHOCK_PROCESS_SETTINGS.sigma_omega,
    sigma_epsilon = SHOCK_PROCESS_SETTINGS.sigma_epsilon,
    sigma_kappa = SHOCK_PROCESS_SETTINGS.sigma_kappa,
    nZ = MODEL_DIMENSIONS.nZ,
    nEps = MODEL_DIMENSIONS.nEps,
    nKappa = MODEL_DIMENSIONS.nKappa,
    z_discretization_method = SHOCK_PROCESS_SETTINGS.z_discretization_method,
    tauchen_width = SHOCK_PROCESS_SETTINGS.tauchen_width,
    z_initial = SHOCK_PROCESS_SETTINGS.z_initial,
    bbar = ASSET_GRID_SETTINGS.bbar,
    aMax = ASSET_GRID_SETTINGS.aMax,
    nA = MODEL_DIMENSIONS.nA,
    asset_grid_method = ASSET_GRID_SETTINGS.method,
    asset_grid_borrow_share = ASSET_GRID_SETTINGS.borrow_share,
    asset_grid_curvature_borrow = ASSET_GRID_SETTINGS.curvature_borrow,
    asset_grid_curvature_save = ASSET_GRID_SETTINGS.curvature_save,
    asset_grid_zero_share = ASSET_GRID_SETTINGS.zero_share,
    asset_grid_zero_width = ASSET_GRID_SETTINGS.zero_width,
    qBorr = FINANCIAL_AND_GOVERNMENT_SETTINGS.qBorr,
    qSav = FINANCIAL_AND_GOVERNMENT_SETTINGS.qSav,
    qGov = FINANCIAL_AND_GOVERNMENT_SETTINGS.qGov,
    G = FINANCIAL_AND_GOVERNMENT_SETTINGS.G,
    hMin = LABOR_SETTINGS.hMin,
    hMax = LABOR_SETTINGS.hMax,
    labor_method = LABOR_SETTINGS.labor_method,
    lambdaMin = LAMBDA_SOLVER_SETTINGS.lambdaMin,
    lambdaMax = LAMBDA_SOLVER_SETTINGS.lambdaMax,
    nLambdaSearch = LAMBDA_SOLVER_SETTINGS.nLambdaSearch,
    tolGovBudget = LAMBDA_SOLVER_SETTINGS.tolGovBudget,
    tolLambda = LAMBDA_SOLVER_SETTINGS.tolLambda,
    verbose = OUTPUT_SETTINGS.verbose,
    printEveryLambda = OUTPUT_SETTINGS.printEveryLambda,
    store_solutions = OUTPUT_SETTINGS.store_solutions,
)

eq, sol = solve_history_independent_bewley(p)
figure_paths = save_unconditional_distribution_figures(eq)

@printf("\n=== Final history-independent equilibrium ===\n")
@printf("lambda                     = %.8f\n", eq.lambda)
@printf("government budget residual =%.8e\n", eq.govBudgetResidual)
@printf("PV output                  = %.8f\n", eq.outputPV)
@printf("PV consumption             = %.8f\n", eq.consumptionPV)
@printf("mean output                = %.8f\n", mean(eq.Y))
@printf("mean consumption.          = %.8f\n", mean(eq.C))
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
@printf("assets                    = %s\n", display_path(figure_paths.assets))
@printf("hours worked              = %s\n", display_path(figure_paths.hours))
@printf("consumption               = %s\n", display_path(figure_paths.consumption))
