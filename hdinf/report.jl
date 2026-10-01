# =============================================================================
# report.jl
#
# Printing that is specific to this variant. The shared printers -- the
# aggregate-statistics block, the welfare table, the upper-bound and lambda
# warnings -- are in BewleyCommon's report.jl.
#
# Split out of the solver so solve_history_dependent_tax.jl is the solver and nothing else.
# Included by it, after `using` and params.jl.
#
# Marek Kapicka, 2026
# =============================================================================

# -----------------------------------------------------------------------------
# Printing
# -----------------------------------------------------------------------------
function print_solver_options(p::HDParams)
    println("Options:")
    @printf("  horizon                       = infinite (VFI fixed point)\n")
    @printf("  maxAge (forward pass cap)     = %d\n", p.maxAge)
    @printf("  tolV, maxIterV, howardSteps   = %.1e, %d, %d\n",
            p.tolV, p.maxIterV, p.howardSteps)
    @printf("  tolDist (PV tail closure)     = %.1e\n", p.tolDist)
    @printf("  shock grid dimension nZ       = %d\n", length(p.z_grid))
    @printf("  shock grid dimension nEps     = %d\n", length(p.eps_grid))
    @printf("  shock grid dimension nKappa   = %d\n", length(p.kappa_grid))
    @printf("  z_discretization_method       = :%s  (alternatives: :rouwenhorst, :tauchen)\n",
            String(p.z_discretization_method))
    @printf("  asset grid dimension nA       = %d\n", length(p.a_grid))
    @printf("  asset grid bounds             = [%.6f, %.6f]\n",
            minimum(p.a_grid), maximum(p.a_grid))
    @printf("  asset grid method             = :%s  (alternatives: :nonuniform, :linear)\n",
            String(p.asset_grid_method))
    # Calibrated inputs print with every digit (shortest representation that
    # round-trips to the same Float64) so they can be copied back verbatim.
    @printf("  beta                          = %-20s  (discount factor)\n", p.beta)
    @printf("  eta                           = %-20s  (labor disutility curvature)\n", p.eta)
    @printf("  phi                           = %-20s  (labor disutility weight)\n", p.phi)
    @printf("  tau                           = %-20s  (HSV tax progressivity)\n", p.tau)
    @printf("  a0                            = %-20s  (initial assets at model age 1)\n", p.a0)
    @printf("  rho                           = %-20s  (AR(1) persistence of z)\n", p.rho)
    # sigma_omega, sigma_epsilon, sigma_kappa and z_initial are settings that
    # build the grids rather than fields of HDParams, so they cannot be reported
    # here without widening the struct. The grids they produced are summarized
    # by their dimensions above.
    @printf("  bbar                          = %-20s  (borrowing limit scale)\n", p.bbar)
    @printf("  labor grid size               = %d on [%.4f, %.4f]\n",
            length(p.h_grid), p.hMin, p.hMax)
    @printf("  labor grid spacing            = :%s  (alternatives: :log, :uniform)\n",
            String(p.labor_grid_spacing))
    @printf("  exploit_hours_monotonicity    = %s\n",
            string(p.exploit_hours_monotonicity))
    @printf("  labor history 1 dimension nS1 = %d\n", length(p.s1_grid))
    @printf("  labor history 1 bounds        = [%.4f, %.4f]\n",
            p.s1_grid[1], p.s1_grid[end])
    @printf("  labor history 1 grid method   = :%s  (alternatives: :linear, :quantile)\n",
            String(p.s_grid_method))
    @printf("  labor history 2 dimension nS2 = %d\n", length(p.s2_grid))
    @printf("  labor history 2 bounds        = [%.4f, %.4f]\n",
            p.s2_grid[1], p.s2_grid[end])
    @printf("  labor history 2 grid method   = :%s  (alternatives: :linear, :quantile)\n",
            String(p.s_grid_method))
    @printf("  qSav                          = %-20s  (price of saving,    a' >= 0)\n", p.qSav)
    @printf("  qBorr                         = %-20s  (price of borrowing, a' < 0)\n", p.qBorr)
    @printf("  qGov                          = %-20s  (government discount price)\n", p.qGov)
    @printf("  theta0 (implied)              = %.6f\n", p.theta0)
    @printf("  alpha                         = %-20s  (kernel weight on mu1)\n", p.alpha)
    @printf("  mu1                           = %-20s  (kernel decay 1)\n", p.mu1)
    @printf("  mu2                           = %-20s  (kernel decay 2)\n", p.mu2)
    @printf("  hours exponent (1-tau)*theta0 = %.6f\n", p.pow)
    @printf("  s_hours_floor                 = %.4f\n", p.s_hours_floor)
    @printf("  terminal_borrowing            = :zero\n")
    @printf("  lambda_solver                 = :brent  (Roots.jl; fallback: grid search over %d values)\n",
            p.nLambdaSearch)
    @printf("  lambda_bracket                = [%.6f, %.6f], tol = %.2e\n",
            p.lambdaMin, p.lambdaMax, p.tolGovBudget)
    @printf("  collect_distributions         = %s\n", string(p.collect_distributions))
    println()
end

function print_hd_equilibrium_summary(eq, p::HDParams;
                                      title = "Final history-dependent equilibrium")
    @printf("\n=== %s ===\n", title)
    @printf("lambda                     = %.8f\n", eq.lambda)
    @printf("government budget residual = %.8e\n", eq.govBudgetResidual)
    @printf("PV output                  = %.8f\n", eq.outputPV)
    @printf("PV consumption             = %.8f\n", eq.consumptionPV)
    @printf("mean output                = %.8f\n", mean(eq.Y))
    @printf("mean consumption           = %.8f\n", mean(eq.C))
    @printf("terminal assets            = %.8f\n", eq.A[end])
    # The inputs -- beta, bbar and the three prices -- are NOT repeated here.
    # They are printed once, with every digit, in the options panel at the top of
    # the run, so this panel carries only what the solve produced.
    @printf("theta0 (implied)           = %.8f\n", p.theta0)
    @printf("s' clamped mass share      = %.3e\n", eq.sClampedMassShare)
    if hasproperty(eq, :elapsedSeconds)
        @printf("solve time                 = %.3f seconds\n", eq.elapsedSeconds)
    end
    print_aggregate_statistics(eq.statistics, p;
        label = @sprintf("ages %d-%d, real %d-%d  [CALIBRATION WINDOW]",
                         p.stats_age_lo, p.stats_age_hi,
                         p.age0_real + p.stats_age_lo - 1,
                         p.age0_real + p.stats_age_hi - 1))
    # The same statistics over the whole forward pass. Nothing is calibrated on
    # these; they are printed so the effect of restricting the moments to the
    # working-age window is visible rather than implied.
    if hasproperty(eq, :statisticsAllAges)
        print_aggregate_statistics(eq.statisticsAllAges, p;
            label = @sprintf("ages 1-%d, real %d-%d  [ALL AGES]",
                             p.maxAge, p.age0_real,
                             p.age0_real + p.maxAge - 1))
    end
    print_welfare_summary(eq.welfare)
    print_upper_bound_warning(eq.statistics)
    return nothing
end
