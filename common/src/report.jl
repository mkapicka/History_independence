# =============================================================================
# report.jl
#
# Printing shared by every solver. Only the genuinely identical blocks live
# here: the per-variant summaries stay with their solvers, because they report
# different quantities (the access chain for the htm variants, maxAge for the
# infinite-horizon ones).
#
# print_aggregate_statistics is here, and it is the only version. The eight
# copies it replaces were the same printer: the only real differences were the
# MPC block (present in hi alone), the hand-to-mouth line (the four *_htm
# variants) and the borrowing-limit labels (which vary with
# asset_choice_method, a setting the hd family does not have). All three are
# guarded on what the statistics and the parameters actually carry, so each
# variant prints exactly what it printed before.
# =============================================================================

"""
    print_aggregate_statistics(s, p; label = "")

The summary block for one statistics group. `s` is a finalized statistics
NamedTuple and `label` names the age window it covers.

Three blocks are conditional, because the variants do not all compute them:

  - the borrowing-limit and bound labels depend on `asset_choice_method`,
    which only the history-independent solvers have. Without it the choice is
    always a grid search and the grid labels are the right ones.
  - the MPC block is printed when the group carries `meanMPC`.
  - the hand-to-mouth line is printed when the group carries
    `shareHandToMouth`, i.e. by the *_htm variants.
"""
function print_aggregate_statistics(s, p::AbstractBewleyParams;
                                    label::AbstractString = "")
    on_grid = !hasproperty(p, :asset_choice_method) ||
              p.asset_choice_method == :grid_search
    limit_label = on_grid ? "grid borrowing limit / mean labor income" :
                            "choice borrowing limit / mean labor income"
    bound_label = on_grid ? "share at effective grid borrowing bound" :
                            "share at borrowing bound"

    @printf("\n=== Aggregate statistics%s ===\n",
            isempty(label) ? "" : ": " * label)
    @printf("mean assets / mean labor income          = %.8f\n",
            s.meanAssetsToMeanLaborIncome)
    @printf("median assets / mean labor income        = %.8f\n",
            s.medianAssetsToMeanLaborIncome)
    @printf("true borrowing limit / mean labor income = %.8f\n",
            s.meanBorrowingLimitToMeanLaborIncome)
    @printf("%-40s = %.8f\n", limit_label,
            s.meanEffectiveGridBorrowingLimitToMeanLaborIncome)
    @printf("share negative liquid assets             = %.8f\n",
            s.shareNegativeLiquidAssets)
    @printf("%-40s = %.8f\n", bound_label, s.shareAtEffectiveBorrowingConstraint)
    # Realized mass in the H state, against the stationary piH the initial
    # cross-section was drawn from. The access chain starts stationary and is
    # independent of everything else, so the two agree at every age; a gap
    # means the forward pass lost access mass, which nothing else would reveal.
    if hasproperty(s, :shareHandToMouth)
        @printf("share hand-to-mouth (target %.6f)    = %.8f\n",
                p.piH, s.shareHandToMouth)
    end
    @printf("share with zero assets                   = %.8f\n", s.shareZeroAssets)
    @printf("share at upper asset bound               = %.8f\n", s.shareAtAssetUpperBound)
    @printf("share at hours upper bound               = %.8f\n", s.shareAtHoursUpperBound)
    # Kaplan-Violante (2022) eq. (2), averaged over the ages this block
    # covers. The windfall is printed beside it: the consumption function is
    # concave, so the MPC is only interpretable with the shock size attached.
    if hasproperty(s, :meanMPC)
        @printf("average impact MPC                       = %.8f\n", s.meanMPC)
        @printf("  windfall                               = %.8f  (%.6f of mean labor income)\n",
                s.mpcShock, s.mpcShockToMeanLaborIncome)
        @printf("  mean MPC | responders (mpc > 0)        = %.8f\n",
                s.meanMPCConditionalOnPositive)
        @printf("  share mpc > 0 / mpc < 0 / mpc = 0      = %.6f / %.6f / %.6f\n",
                s.shareMPCPositive, s.shareMPCNegative, s.shareMPCZero)
        if isfinite(s.medianMPC)
            @printf("  median MPC                             = %.8f\n", s.medianMPC)
        end
        @printf("  mean MPC | a < %.6f (%.6f of Y)  = %.8f  over share %.6f\n",
                s.mpcLowAssetThreshold, s.mpcLowAssetThresholdToMeanLaborIncome,
                s.meanMPCAtLowAssets, s.shareAtLowAssets)
        if s.shareMPCExtrapolated > 1e-8
            @printf("  share extrapolated above the grid      = %.8f   [raise aMax]\n",
                    s.shareMPCExtrapolated)
        end
    end
    # Only the windowed statistics carry the entry-age block; the all-ages
    # block is printed through this same function and has no such age.
    if hasproperty(s, :meanAssetsAtStatsAgeLoToMeanLaborIncome)
        lo_real = p.age0_real + p.stats_age_lo - 1
        @printf("mean assets at age %-2d / mean labor income = %.8f\n",
                lo_real, s.meanAssetsAtStatsAgeLoToMeanLaborIncome)
        @printf("median assets at age %-2d / mean labor inc. = %.8f\n",
                lo_real, s.medianAssetsAtStatsAgeLoToMeanLaborIncome)
    end
    return nothing
end

"""
    print_upper_bound_warning(s)

Warn when the asset or hours upper bound carries mass. A bound that binds means
the grid, not the model, is setting the choice, so the run is only as good as
the bound: the share at it, the bound itself, the largest material choice and
the slack are all printed, because the share alone does not say whether the
bound is nearly slack or badly placed.

"Material" excludes cells holding less than UPPER_BOUND_SHARE_TOL of mass, so a
negligible state at the grid edge does not masquerade as a binding bound.

This is the history-independent solvers' format, adopted for all eight. The hd
family previously printed a banner without the share.
"""
function print_upper_bound_warning(s)
    s.upperBoundsBinding || return nothing

    println("WARNING: upper bound is binding.")
    if s.assetUpperBoundBinding
        @printf("  asset upper bound       = BINDING (share = %.8e, bound = %.8f, material max a' = %.8f, slack = %.8e)\n",
                s.shareAtAssetUpperBound, s.assetUpperBound,
                s.maxMaterialNextAssets, s.assetUpperBoundSlack)
    end
    if s.hoursUpperBoundBinding
        @printf("  hours upper bound       = BINDING (share = %.8e, bound = %.8f, material max h = %.8f, slack = %.8e)\n",
                s.shareAtHoursUpperBound, s.hoursUpperBound,
                s.maxMaterialHours, s.hoursUpperBoundSlack)
    end
    flush(stdout)
end

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

"""
    eq_flag(eq, field)

True when the equilibrium carries `field` and it is set. The warning flags are
attached only on the paths that raise them, so every read has to be guarded.
"""
eq_flag(eq, field::Symbol) = hasproperty(eq, field) && getproperty(eq, field) === true

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
