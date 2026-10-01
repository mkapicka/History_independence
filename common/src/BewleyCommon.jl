# =============================================================================
# BewleyCommon.jl
#
# Code shared by the eight solver directories, in three tiers:
#
#   generic     params.jl, grids.jl, shocks.jl -- the parameter supertype and
#               its accessors, asset grids, shock discretization, lookups.
#   hi family   labor.jl, assets.jl, asset_choice.jl, values.jl, access.jl --
#               the static labor FOC, the Young lottery, the :interpolate asset
#               choice, continuation values, the access chain. values.jl also
#               carries evaluate_block! for the infinite-horizon hd pair.
#   all eight   statistics.jl, report.jl -- the published statistics and the
#               printers.
#
# Functions that take a parameter struct annotate it as AbstractBewleyParams,
# which HIParams and HDParams subtype; Julia still specializes each call site on
# the concrete type, so the annotation costs nothing in the inner loops.
#
# The export list below is the API the solvers call. Helpers used only inside
# this package are not exported, so a solver may define a local function of the
# same name without a clash. hi_model.jl and hd_model.jl export nothing: each
# name there is shared by some directories and defined differently in others,
# so the directories that want one import it by name, e.g.
#
#     using BewleyCommon: BlockScratch, build_s_grid, solve_block!
#
# which also documents at the top of each solver what it takes from here.
# build_labor_grid is unexported for the same reason: it is the hd family's
# 4-argument log-spaced grid, while the hi family uses uniform_labor_grid.
#
# Marek Kapicka, 2026
# =============================================================================

module BewleyCommon

using FastGaussQuadrature
using LinearAlgebra
using Printf
using QuantEcon
using Roots
using StatsBase

include("params.jl")
include("grids.jl")
include("shocks.jl")
include("labor.jl")
include("assets.jl")
include("asset_choice.jl")
include("values.jl")
include("access.jl")
include("statistics.jl")
include("report.jl")
# Family-specific, nothing exported -- see the header of each file.
include("hi_model.jl")
include("hd_model.jl")

export
    # params.jl
    AbstractBewleyParams, VINFEASIBLE,
    safe_ratio, upper_bound_level_tol,
    asset_price, asset_prices, asset_upper_bound, hours_upper_bound,
    borrowing_limit, first_feasible_asset_indices,
    # grids.jl
    asset_grid_with_zero, nearest_index, normalize_probabilities,
    validate_transition, discounted_sum,
    # shocks.jl
    build_markov_shock, build_iid_normal_shock,
    quantecon_ar1, ar1_conditional_probabilities,
    normal_gauss_hermite, normal_cdf,
    grid_lookup_weights, find_bracket, discounted_sum_with_tail, ar1_transition,
    # labor.jl
    optimal_labor_foc, uniform_labor_grid, normalize_labor_grid,
    precompute_flow_payoffs,
    # assets.jl
    default_asset_grid, asset_choice_bound_tol,
    asset_transition_weights, first_nonnegative_asset_index,
    # asset_choice.jl  (the :interpolate path; its entry point only)
    interpolated_asset_choice,
    # values.jl
    compute_expected_value!, evaluate_policy_grid_search!, evaluate_block!,
    # access.jl
    access_stationary_distribution,
    # statistics.jl
    AbstractStatsAccumulator, UPPER_BOUND_SHARE_TOL,
    core_statistics, mpc_statistics,
    # report.jl
    print_aggregate_statistics, print_upper_bound_warning,
    print_welfare_summary, warn_if_unsettled, print_lambda_warnings

end # module
