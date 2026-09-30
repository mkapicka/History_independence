# =============================================================================
# statistics.jl
#
# Turning a statistics accumulator into the finalized NamedTuple the solvers
# publish as `eq.statistics`.
#
# The accumulators themselves stay with their solvers. They are mutable structs
# touched once per state cell over millions of cells, and each variant tracks
# exactly the fields it needs: the hi solver adds eleven MPC fields, the *_htm
# variants add `htm_mass`, the hd family adds an unrestricted (max_next_assets,
# max_hours) pair beside the material one. Forcing them into one struct would
# put fields in every variant that only one of them fills.
#
# What is shared is the arithmetic. `core_statistics` computes the 24 fields
# every variant publishes; a variant adds its own with `merge`. That is why the
# accumulator argument is typed on AbstractStatsAccumulator rather than on a
# concrete struct: the function reads only the fifteen core fields, which all
# eight accumulators carry under the same names.
#
# This runs once per equilibrium, not per cell, so clarity beats speed here.
#
# Marek Kapicka, 2026
# =============================================================================

"""
    AbstractStatsAccumulator

Supertype of the per-variant statistics accumulators. Subtyping it is what
makes an accumulator usable with `core_statistics`; the contract is the fifteen
core field names, which every accumulator carries.
"""
abstract type AbstractStatsAccumulator end

# A bound counts as binding on the mass at it, not on the slack: a state can sit
# arbitrarily close to the upper bound without the bound doing any work.
const UPPER_BOUND_SHARE_TOL = 1e-8

"""
    core_statistics(stats, p)

The statistics every variant publishes, from the accumulated sums. Variants add
their own fields by merging onto this.
"""
function core_statistics(stats::AbstractStatsAccumulator, p::AbstractBewleyParams)
    total_mass = stats.total_mass
    mean_assets = stats.sum_current_assets / total_mass
    mean_labor_income = stats.sum_labor_income / total_mass
    # Mid-cumulative interpolation rather than StatsBase's weighted-quantile
    # convention, which is biased low on a coarse nonuniform grid holding a
    # discretized continuous distribution. See `interpolated_weighted_quantile`
    # in grids.jl for the measured comparison against a known median: at
    # nA = 151 StatsBase errs by 5.8% of the median and refinement does not
    # close the gap.
    median_assets = interpolated_weighted_quantile(p.a_grid, stats.asset_mass, 0.5)
    # Both limits exist only at ages j = 0,...,J-1, so they are averaged over
    # the mass of those ages rather than over the whole population.
    mean_borrowing_limit = safe_ratio(stats.sum_borrowing_limit,
                                      stats.borrowing_limit_mass)
    mean_effective_borrowing_limit = safe_ratio(stats.sum_effective_borrowing_limit,
                                                stats.borrowing_limit_mass)
    share_at_effective_borrowing_constraint = stats.borrowing_constraint_mass / total_mass
    share_at_asset_upper_bound = stats.upper_bound_mass / total_mass
    share_at_hours_upper_bound = stats.hours_upper_bound_mass / total_mass
    max_material_next_assets =
        isfinite(stats.max_material_next_assets) ? stats.max_material_next_assets : NaN
    max_material_hours = isfinite(stats.max_material_hours) ? stats.max_material_hours : NaN
    asset_upper = asset_upper_bound(p)
    hours_upper = hours_upper_bound(p)
    asset_upper_bound_slack = asset_upper - max_material_next_assets
    hours_upper_bound_slack = hours_upper - max_material_hours
    asset_upper_bound_binding = share_at_asset_upper_bound > UPPER_BOUND_SHARE_TOL
    hours_upper_bound_binding = share_at_hours_upper_bound > UPPER_BOUND_SHARE_TOL
    distributions = (;
        assetGrid = p.a_grid,
        assetMass = copy(stats.asset_mass),
        assetMassTotal = sum(stats.asset_mass),
        hours = copy(stats.hours_values),
        consumption = copy(stats.consumption_values),
        weights = copy(stats.distribution_weights),
        observationWeightTotal = sum(stats.distribution_weights),
    )

    return (;
        totalMass = total_mass,
        meanAssets = mean_assets,
        medianAssets = median_assets,
        meanLaborIncome = mean_labor_income,
        meanBorrowingLimit = mean_borrowing_limit,
        meanEffectiveGridBorrowingLimit = mean_effective_borrowing_limit,
        meanAssetsToMeanLaborIncome = safe_ratio(mean_assets, mean_labor_income),
        medianAssetsToMeanLaborIncome = safe_ratio(median_assets, mean_labor_income),
        meanBorrowingLimitToMeanLaborIncome =
            safe_ratio(mean_borrowing_limit, mean_labor_income),
        meanEffectiveGridBorrowingLimitToMeanLaborIncome =
            safe_ratio(mean_effective_borrowing_limit, mean_labor_income),
        shareNegativeLiquidAssets = stats.negative_asset_mass / total_mass,
        shareAtEffectiveBorrowingConstraint = share_at_effective_borrowing_constraint,
        shareZeroAssets = stats.zero_asset_mass / total_mass,
        shareAtAssetUpperBound = share_at_asset_upper_bound,
        shareAtHoursUpperBound = share_at_hours_upper_bound,
        assetUpperBound = asset_upper,
        hoursUpperBound = hours_upper,
        maxMaterialNextAssets = max_material_next_assets,
        maxMaterialHours = max_material_hours,
        assetUpperBoundSlack = asset_upper_bound_slack,
        hoursUpperBoundSlack = hours_upper_bound_slack,
        assetUpperBoundBinding = asset_upper_bound_binding,
        hoursUpperBoundBinding = hours_upper_bound_binding,
        upperBoundsBinding = asset_upper_bound_binding || hours_upper_bound_binding,
        unconditionalDistributions = distributions,
    )
end

"""
    mpc_statistics(stats, p)

The impact-MPC block, Kaplan-Violante (2022) eq. (2) averaged over whatever
ages the accumulator covered. Merged on by the variants that compute MPCs; the
accumulator must carry the eleven mpc fields.

`medianMPC` needs the per-observation vector and so is NaN unless
collect_distributions was on. Every other measure here is a running sum and is
always present.
"""
function mpc_statistics(stats::AbstractStatsAccumulator, p::AbstractBewleyParams)
    total_mass = stats.total_mass
    mean_labor_income = stats.sum_labor_income / total_mass
    # The MPC median needs its own sort: `interpolated_weighted_quantile` walks
    # the grid in order, and MPCs arrive in state order, not value order.
    median_mpc = if isempty(stats.mpc_values)
        NaN
    else
        ord = sortperm(stats.mpc_values)
        interpolated_weighted_quantile(stats.mpc_values[ord],
                                       stats.distribution_weights[ord], 0.5)
    end
    return (;
        # The mean, and the share of mass whose perturbed state left the top of
        # the asset grid and was extrapolated. A non-negligible extrapolated
        # share means aMax is too low for the windfall.
        meanMPC = safe_ratio(stats.sum_mpc, total_mass),
        shareMPCExtrapolated = stats.mpc_extrapolated_mass / total_mass,
        mpcShock = p.mpc_shock,
        mpcShockToMeanLaborIncome = safe_ratio(p.mpc_shock, mean_labor_income),
        # The distribution, as Discrete_HA's MPCFinder.m reports it: the mean
        # over responders, the responder shares, and the low-liquid-wealth group.
        meanMPCConditionalOnPositive =
            safe_ratio(stats.sum_mpc_positive, stats.mpc_positive_mass),
        shareMPCPositive = stats.mpc_positive_mass / total_mass,
        shareMPCNegative = stats.mpc_negative_mass / total_mass,
        shareMPCZero = stats.mpc_zero_mass / total_mass,
        medianMPC = median_mpc,
        meanMPCAtLowAssets = safe_ratio(stats.sum_mpc_lowasset, stats.lowasset_mass),
        shareAtLowAssets = stats.lowasset_mass / total_mass,
        mpcLowAssetThreshold = p.mpc_lowasset_threshold,
        mpcLowAssetThresholdToMeanLaborIncome =
            safe_ratio(p.mpc_lowasset_threshold, mean_labor_income),
    )
end
