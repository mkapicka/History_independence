# =============================================================================
# hi_model.jl
#
# Machinery shared by the history-independent solvers: welfare
# aggregation, the income bases, and the interpolated policy step. Each block
# names the directories it came from.
#
# Nothing here is exported. Each of these names is defined with a DIFFERENT
# implementation in the sibling directories that do not share this version, and
# a directory that both `using`s an exported name and defines its own gets a
# Julia error. So the directories that want these import them by name:
#
#     using BewleyCommon: finalize_welfare, precompute_income_bases
#
# which also documents at the top of each solver exactly what it takes from the
# package. This follows build_labor_grid, unexported for the same reason.
#
# Marek Kapicka, 2026
# =============================================================================

# ---- finalize_welfare, precompute_income_bases: all four hi directories.

function finalize_welfare(value_function_by_kappa::Vector{Float64},
                          simulation_by_kappa::Vector{Float64},
                          p::AbstractBewleyParams)
    difference_by_kappa = simulation_by_kappa .- value_function_by_kappa
    overall_value_function = dot(p.Pkappa, value_function_by_kappa)
    overall_simulation = dot(p.Pkappa, simulation_by_kappa)
    overall_difference = overall_simulation - overall_value_function

    return (;
        kappaGrid = p.kappa_grid,
        kappaProbabilities = p.Pkappa,
        valueFunctionByKappa = value_function_by_kappa,
        simulationByKappa = simulation_by_kappa,
        differenceByKappa = difference_by_kappa,
        overallValueFunction = overall_value_function,
        overallSimulation = overall_simulation,
        overallDifference = overall_difference,
    )
end

function precompute_income_bases(kappa::Float64, p::AbstractBewleyParams)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    tax_base = Matrix{Float64}(undef, nZ, nE)
    wage_base = Matrix{Float64}(undef, nZ, nE)
    @inbounds for iz in 1:nZ, ie in 1:nE
        log_wage = kappa + p.z_grid[iz] + p.eps_grid[ie]
        tax_base[iz, ie] = exp((1.0 - p.tau) * log_wage)
        wage_base[iz, ie] = exp(log_wage)
    end
    return tax_base, wage_base
end

# ---- print_equilibrium_summary, asset_choice_lower_bound,
# solve_policy_age_interpolated!: hi and hi_htm. The infinite-horizon pair has
# its own versions (no terminal age, no age-indexed borrowing bound).

"""
    print_equilibrium_summary(eq, p; title, show_statistics, show_welfare)

Print the equilibrium, aggregate statistics, and welfare decomposition. Binding
upper bounds are reported by the solver itself (see `print_upper_bound_warning`,
called from `attach_elapsed`), so they are not repeated here.
"""
function print_equilibrium_summary(eq, p::AbstractBewleyParams;
                                   title = "Final history-independent equilibrium",
                                   show_statistics::Bool = true,
                                   show_welfare::Bool = true)
    @printf("\n=== %s ===\n", title)
    @printf("lambda                     = %.8f\n", eq.lambda)
    @printf("government budget residual = %.8e\n", eq.govBudgetResidual)
    @printf("PV output                  = %.8f\n", eq.outputPV)
    @printf("PV consumption             = %.8f\n", eq.consumptionPV)
    @printf("mean output                = %.8f\n", mean(eq.Y))
    @printf("mean consumption           = %.8f\n", mean(eq.C))
    @printf("terminal assets            = %.8f\n", eq.A[end])
    # Printed with every digit (shortest representation that round-trips to
    # the same Float64), so calibrated values can be copied back verbatim.
    @printf("qSav                       = %s\n", p.qSav)
    @printf("qBorr                      = %s\n", p.qBorr)
    @printf("bbar                       = %s\n", p.bbar)
    if hasproperty(eq, :elapsedSeconds)
        @printf("solve time                 = %.3f seconds\n", eq.elapsedSeconds)
    end

    if show_statistics && hasproperty(eq, :statistics)
        print_aggregate_statistics(eq.statistics, p;
            label = @sprintf("model ages %d-%d, real %d-%d  [CALIBRATION WINDOW]",
                             p.stats_age_lo, p.stats_age_hi,
                             p.age0_real + p.stats_age_lo - 1,
                             p.age0_real + p.stats_age_hi - 1))
        # The same statistics over the whole life. Nothing is calibrated on
        # these; they are printed so the effect of restricting the moments to
        # the working-age window is visible rather than implied.
        if hasproperty(eq, :statisticsAllAges)
            print_aggregate_statistics(eq.statisticsAllAges, p;
                label = @sprintf("model ages 1-%d, real %d-%d  [ALL AGES]",
                                 p.J + 1, p.age0_real, p.age0_real + p.J))
        end
    end
    if show_welfare && hasproperty(eq, :welfare)
        print_welfare_summary(eq.welfare)
    end
    return nothing
end

function asset_choice_lower_bound(age::Int, kappa::Float64, iz::Int, p::AbstractBewleyParams)
    if age == p.J + 1
        return 0.0
    end
    return borrowing_limit(kappa, iz, p)
end

function solve_policy_age_interpolated!(Vcur, policyAIndex, policyA, policyH,
                                        EV, age::Int, lambda::Float64,
                                        kappa::Float64,
                                        tax_base::Matrix{Float64}, p::AbstractBewleyParams,
                                        util_weight::Float64, beta::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)

    @inbounds for ia in 1:nA
        a = p.a_grid[ia]
        for iz in 1:nZ
            lower = asset_choice_lower_bound(age, kappa, iz, p)
            for ie in 1:nE
                income_coeff = lambda * tax_base[iz, ie]
                best_val, best_ap, best_iap, best_h = interpolated_asset_choice(
                    a, lower, income_coeff, EV, iz, p, util_weight, beta,
                )
                Vcur[ia, iz, ie] = best_val
                policyAIndex[ia, iz, ie, age] = Int32(best_iap)
                policyA[ia, iz, ie, age] = best_ap
                policyH[ia, iz, ie, age] = best_h
            end
        end
    end
    return nothing
end
