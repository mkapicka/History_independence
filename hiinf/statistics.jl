# =============================================================================
# statistics.jl
#
# The statistics accumulator for this variant: what it tracks, how two of
# them reduce, and the per-cell update. `accumulate_stats!` is the hot loop --
# it runs once per state cell over millions of cells, so it stays type-stable
# and allocation-free and never sees a label. Turning the accumulated sums into
# the published NamedTuple is BewleyCommon's core_statistics; only the fields
# this variant adds are merged on here.
#
# Split out of the solver so solve_history_independent_tax.jl is the solver and nothing else.
# Included by it, after `using` and params.jl.
#
# Marek Kapicka, 2026
# =============================================================================

Base.@kwdef mutable struct HIStatsAccumulator <: AbstractStatsAccumulator
    asset_mass::Vector{Float64}
    distribution_weights::Vector{Float64} = Float64[]
    hours_values::Vector{Float64} = Float64[]
    consumption_values::Vector{Float64} = Float64[]
    total_mass::Float64 = 0.0
    sum_current_assets::Float64 = 0.0
    sum_labor_income::Float64 = 0.0
    sum_borrowing_limit::Float64 = 0.0
    sum_effective_borrowing_limit::Float64 = 0.0
    borrowing_limit_mass::Float64 = 0.0
    negative_asset_mass::Float64 = 0.0
    zero_asset_mass::Float64 = 0.0
    borrowing_constraint_mass::Float64 = 0.0
    upper_bound_mass::Float64 = 0.0
    hours_upper_bound_mass::Float64 = 0.0
    max_material_next_assets::Float64 = -Inf
    max_material_hours::Float64 = -Inf
end

function HIStatsAccumulator(nA::Int)
    nA > 0 || error("nA must be positive")
    return HIStatsAccumulator(asset_mass = zeros(nA))
end

# How each field combines when per-kappa accumulators are reduced.
const STATS_APPEND_FIELDS =
    (:distribution_weights, :hours_values, :consumption_values)

const STATS_SUM_FIELDS =
    (:total_mass, :sum_current_assets, :sum_labor_income, :sum_borrowing_limit,
     :sum_effective_borrowing_limit, :borrowing_limit_mass, :negative_asset_mass,
     :zero_asset_mass, :borrowing_constraint_mass, :upper_bound_mass,
     :hours_upper_bound_mass)

const STATS_MAX_FIELDS = (:max_material_next_assets, :max_material_hours)

function merge_stats!(dest::HIStatsAccumulator, src::HIStatsAccumulator)
    length(dest.asset_mass) == length(src.asset_mass) ||
        error("Cannot merge statistics with different asset-grid sizes")

    dest.asset_mass .+= src.asset_mass
    for f in STATS_APPEND_FIELDS
        append!(getfield(dest, f), getfield(src, f))
    end
    for f in STATS_SUM_FIELDS
        setfield!(dest, f, getfield(dest, f) + getfield(src, f))
    end
    for f in STATS_MAX_FIELDS
        setfield!(dest, f, max(getfield(dest, f), getfield(src, f)))
    end
    return dest
end

"""
    accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                      true_borrowing_limit, effective_borrowing_limit,
                      at_borrowing_constraint, at_asset_upper, h_upper,
                      collect::Bool, p)

Add one (age, state) observation to a statistics accumulator.

Factored out because the forward pass now feeds TWO accumulators -- one over the
calibration age window and one over every age -- and a copied block would let
the two definitions drift. `collect` is passed rather than read from `p` so the
all-ages accumulator can skip the distribution vectors: those are a
cross-sectional object, the cross-section is the window, and pushing three
Float64 per (age, state) across all `maxAge` ages is the allocation that
OOM-killed the 128 GiB cluster jobs.

Marked `@inline`: this is the innermost loop of the forward pass, called once
per accumulator per positive-mass state per age.
"""
@inline function accumulate_stats!(stats::HIStatsAccumulator, weighted_mass::Float64,
                                   ia::Int, a::Float64, ap::Float64, h::Float64,
                                   c::Float64, y::Float64,
                                   true_borrowing_limit::Float64,
                                   effective_borrowing_limit::Float64,
                                   at_borrowing_constraint::Bool,
                                   at_asset_upper::Bool, h_upper::Float64,
                                   collect::Bool, p::HIParams)
    @inbounds begin
        stats.asset_mass[ia] += weighted_mass
        if collect
            push!(stats.distribution_weights, weighted_mass)
            push!(stats.hours_values, h)
            push!(stats.consumption_values, c)
        end
        stats.total_mass += weighted_mass
        stats.sum_current_assets += weighted_mass * a
        stats.sum_labor_income += weighted_mass * y
        stats.sum_borrowing_limit += weighted_mass * true_borrowing_limit
        stats.sum_effective_borrowing_limit += weighted_mass * effective_borrowing_limit
        # Every age has a borrowing limit here, so all mass counts towards the
        # limit averages; the finite solver excludes its terminal age, where
        # a' >= 0 replaces the limit.
        stats.borrowing_limit_mass += weighted_mass
        if weighted_mass > UPPER_BOUND_SHARE_TOL
            stats.max_material_next_assets = max(stats.max_material_next_assets, ap)
            stats.max_material_hours = max(stats.max_material_hours, h)
        end
        if a < -1e-10
            stats.negative_asset_mass += weighted_mass
        end
        if abs(a) <= 1e-10
            stats.zero_asset_mass += weighted_mass
        end
        if at_borrowing_constraint
            stats.borrowing_constraint_mass += weighted_mass
        end
        if at_asset_upper
            stats.upper_bound_mass += weighted_mass
        end
        if h >= h_upper - upper_bound_level_tol(h_upper)
            stats.hours_upper_bound_mass += weighted_mass
        end
    end
    return nothing
end
