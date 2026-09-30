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

"""
    warn_if_unsettled(eq, p)

Report the cross-section drift of the RETURNED equilibrium, once.

The closed-form PV tail assumes Y_j - C_j has stopped moving past `maxAge`.
Checking that inside the per-lambda solve produced one warning per probe, and
the root-finder visits corners (lambda = lambdaMin, qSav near its bracket) where
the economy is degenerate and legitimately unsettled -- true but useless. Only
the equilibrium actually returned has to be clean, so the test lives here, on
the single funnel every return path passes through.

The drift is reported RELATIVE to Y: an absolute bound on a sum of aggregate
differences is uninterpretable without its scale. A converged solve was measured
at 1.7e-09 of Y with lambda and W invariant to 15 digits across a doubled
maxAge; a genuinely unsettled path runs 1e-04 and worse. `tolDriftRel` sits
between them, and `Inf` silences this entirely.
"""
function warn_if_unsettled(eq, p::AbstractBewleyParams; converged::Bool = true)
    # A solve whose lambda never converged is not an equilibrium, so its
    # settling behaviour is not informative -- and the lambda failure is already
    # reported by the solver. Stacking a second warning on top buries the one
    # that matters. Observed: a calibration probe at qSav = 0.911 (floor 0.900)
    # where no lambda balances the budget, bottoming out at lambda = 0.018
    # against an equilibrium ~1.01, warned twice for one underlying problem.
    converged || return nothing
    hasproperty(eq, :diagnostics) || return nothing
    d = eq.diagnostics.finalDrift
    yscale = abs(eq.Y[end]) > 0 ? abs(eq.Y[end]) : 1.0
    any(x -> !(x / yscale <= p.tolDriftRel), d) || return nothing
    lines = join((@sprintf("kappa %d (% .4f): drift %.3e (%.1e of Y)  convergedAge %s",
                           ik, p.kappa_grid[ik], d[ik], d[ik] / yscale,
                           eq.diagnostics.convergedAgeByKappa[ik] == 0 ? "never" :
                           string(eq.diagnostics.convergedAgeByKappa[ik]))
                  for ik in eachindex(d)), "\n")
    @warn("RETURNED equilibrium: cross-section drift at maxAge exceeds tolDriftRel " *
          "for at least one kappa; the closed-form PV tail assumes the path has " *
          "settled.\n" *
          @sprintf("maxAge = %d, tolDriftRel = %.1e, Y[end] = %.6f\n",
                   p.maxAge, p.tolDriftRel, eq.Y[end]) *
          @sprintf("qSav = %.8f, qBorr = %.8f, bbar = %.8f, lambda = %.8f\n",
                   p.qSav, p.qBorr, p.bbar, eq.lambda) * lines *
          "\nThis is a CONVERGED equilibrium being returned, not a lambda probe. " *
          "During a calibration it is still one instrument triple among many; " *
          "check qSav/qBorr/bbar above against the calibrated values. " *
          "Judge by the relative column; raise maxAge if it is not many orders below Y.")
    return nothing
end
