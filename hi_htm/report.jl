# =============================================================================
# report.jl
#
# Printing that is specific to this variant. The shared printers -- the
# aggregate-statistics block, the welfare table, the upper-bound and lambda
# warnings -- are in BewleyCommon's report.jl.
#
# Split out of the solver so solve_history_independent_tax.jl is the solver and nothing else.
# Included by it, after `using` and params.jl.
#
# Marek Kapicka, 2026
# =============================================================================

function print_solver_options(p::HIParams)
    println("Options:")
    @printf("  age dimension J             = %d\n", p.J)
    @printf("  shock grid dimension nZ     = %d\n", length(p.z_grid))
    @printf("  shock grid dimension nEps   = %d\n", length(p.eps_grid))
    @printf("  shock grid dimension nKappa = %d\n", length(p.kappa_grid))
    @printf("  z_discretization_method     = :%s  (alternatives: :rouwenhorst, :tauchen)\n",
            String(p.z_discretization_method))
    @printf("  tauchen_width               = %.3f  (used when z_discretization_method = :tauchen)\n",
            p.tauchen_width)
    @printf("  asset grid dimension nA     = %d\n", length(p.a_grid))
    @printf("  asset_grid_method           = :%s  (alternatives: :nonuniform, :linear)\n",
            String(p.asset_grid_method))
    if p.asset_grid_method == :nonuniform
        @printf("  asset_grid_borrow_share     = %.3f\n", p.asset_grid_borrow_share)
        @printf("  asset_grid_curvatures       = borrow %.3f, save %.3f\n",
                p.asset_grid_curvature_borrow, p.asset_grid_curvature_save)
        @printf("  asset_grid_zero_band        = share %.3f, width %.3f\n",
                p.asset_grid_zero_share, p.asset_grid_zero_width)
    end
    @printf("  asset_grid_bounds           = [%.6f, %.6f]\n",
            minimum(p.a_grid), maximum(p.a_grid))
    # Calibrated inputs print with every digit (shortest representation that
    # round-trips to the same Float64) so they can be copied back verbatim.
    @printf("  beta                        = %-20s  (discount factor)\n", p.beta)
    @printf("  eta                         = %-20s  (labor disutility curvature)\n", p.eta)
    @printf("  phi                         = %-20s  (labor disutility weight)\n", p.phi)
    @printf("  tau                         = %-20s  (HSV tax progressivity)\n", p.tau)
    @printf("  a0                          = %-20s  (initial assets at model age 1)\n", p.a0)
    @printf("  rho                         = %-20s  (AR(1) persistence of z)\n", p.rho)
    @printf("  sigma_omega                 = %-20s  (s.d. of the persistent innovation)\n", p.sigma_omega)
    @printf("  sigma_epsilon               = %-20s  (s.d. of the transitory shock)\n", p.sigma_epsilon)
    @printf("  sigma_kappa                 = %-20s  (s.d. of the fixed effect)\n", p.sigma_kappa)
    @printf("  z_initial                   = %-20s  (z at model age 1)\n", p.z_initial)
    @printf("  bbar                        = %-20s  (borrowing limit scale)\n", p.bbar)
    @printf("  pSS (stay saver)            = %-20s  (s in psmodel.tex)\n", p.pSS)
    @printf("  pHH (stay hand-to-mouth)    = %-20s  (h in psmodel.tex)\n", p.pHH)
    @printf("  access shares (piS, piH)    = (%.8f, %.8f)%s\n",
            p.piS, p.piH,
            p.piH == 0.0 ? "   [NO HtM AGENTS: this is the hi model]" : "")
    @printf("  qSav                        = %-20s  (price of saving,    a' >= 0)\n", p.qSav)
    @printf("  qBorr                       = %-20s  (price of borrowing, a' < 0)\n", p.qBorr)
    @printf("  qGov                        = %-20s  (government discount price)\n", p.qGov)
    @printf("  G                           = %-20s  (government spending)\n", p.G)
    @printf("  asset_choice_method         = :%s  (alternatives: :grid_search, :interpolate)\n",
            String(p.asset_choice_method))
    if p.asset_choice_method == :interpolate
        @printf("  asset_choice_optimizer      = golden search with linear continuation interpolation, tol = %.2e, max_iter = %d\n",
                p.asset_choice_tol, p.asset_choice_max_iter)
    end
    @printf("  labor_solver                = :%s  (alternatives: :brent, :hybrid_newton, :grid)\n",
            String(p.labor_solver))
    @printf("  labor_bounds                = [%.2e, %.4f]\n", p.hMin, p.hMax)
    if p.labor_solver == :grid
        @printf("  labor_grid_size             = %d\n", length(p.h_grid))
    end
    @printf("  terminal_borrowing          = :zero\n")
    @printf("  lambda_solver               = :brent  (Roots.jl; fallback: grid search over %d values)\n",
            p.nLambdaSearch)
    @printf("  lambda_bracket              = [%.6f, %.6f], tol = %.2e\n",
            p.lambdaMin, p.lambdaMax, p.tolGovBudget)
    @printf("  collect_distributions       = %s\n", string(p.collect_distributions))
    println()
end
