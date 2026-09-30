# =============================================================================
# BewleyCommon.jl
#
# Infrastructure shared by all eight solver directories: asset and labor grids,
# shock discretization, interpolation lookups, and the small numerical helpers
# that go with them. Nothing here knows about any particular model -- every
# function takes plain numbers and arrays, never a parameter struct.
#
# grids.jl is included first: shocks.jl uses nearest_index,
# normalize_probabilities and validate_transition from it.
#
# `build_labor_grid` is deliberately NOT exported. Two different functions
# carry that name: the one here is 4-argument and log-spaced by default, used
# by the hd family; the hi family defines its own 3-argument uniform version.
# Exporting this one would make the hi family's definition an error rather than
# a second method, and silently swapping them would change every hours grid.
# Call it as BewleyCommon.build_labor_grid.
#
# Marek Kapicka, 2026
# =============================================================================

module BewleyCommon

using FastGaussQuadrature
using Printf
using QuantEcon
using Roots
using StatsBase

include("params.jl")
include("grids.jl")
include("shocks.jl")
include("labor.jl")
include("assets.jl")
include("statistics.jl")
include("report.jl")

export
    # params.jl
    AbstractBewleyParams,
    safe_ratio, upper_bound_level_tol,
    asset_price, asset_prices, asset_upper_bound, hours_upper_bound,
    borrowing_limit, first_feasible_asset_indices,
    # grids.jl
    asset_grid_with_zero, linear_asset_grid, nonnegative_asset_grid,
    two_region_asset_grid, zero_band_asset_grid,
    nearest_index, normalize_probabilities, validate_transition,
    interpolated_weighted_quantile, discounted_sum,
    # shocks.jl
    build_markov_shock, build_iid_normal_shock,
    quantecon_ar1, ar1_conditional_probabilities,
    normal_gauss_hermite, normal_cdf,
    grid_lookup_weights, find_bracket, discounted_sum_with_tail,
    # labor.jl
    optimal_labor_foc, solve_labor_root, optimal_labor_grid,
    labor_root_hybrid_newton, labor_foc_residual, labor_foc_residual_derivative,
    uniform_labor_grid, normalize_labor_grid,
    # assets.jl
    default_asset_grid, asset_choice_bound_tol,
    asset_transition_weights, nearest_asset_index,
    # statistics.jl
    AbstractStatsAccumulator, UPPER_BOUND_SHARE_TOL,
    core_statistics, mpc_statistics,
    # report.jl
    print_aggregate_statistics, print_upper_bound_warning,
    print_welfare_summary,
    print_lambda_warnings, eq_flag

end # module
