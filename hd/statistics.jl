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
# Split out of the solver so solve.jl is the solver and nothing else.
# Included by it, after `using` and params.jl.
#
# Marek Kapicka, 2026
# =============================================================================

# -----------------------------------------------------------------------------
# Statistics accumulator (same fields and semantics as the history-independent
# solver, so downstream statistics are directly comparable)
# -----------------------------------------------------------------------------
mutable struct StatsAccumulator <: AbstractStatsAccumulator
    asset_mass::Vector{Float64}
    distribution_weights::Vector{Float64}
    hours_values::Vector{Float64}
    consumption_values::Vector{Float64}
    total_mass::Float64
    sum_current_assets::Float64
    sum_labor_income::Float64
    sum_borrowing_limit::Float64
    sum_effective_borrowing_limit::Float64
    borrowing_limit_mass::Float64
    negative_asset_mass::Float64
    zero_asset_mass::Float64
    borrowing_constraint_mass::Float64
    upper_bound_mass::Float64
    hours_upper_bound_mass::Float64
    max_next_assets::Float64
    max_hours::Float64
    max_material_next_assets::Float64
    max_material_hours::Float64
end

function StatsAccumulator(nA::Int)
    nA > 0 || error("nA must be positive")
    return StatsAccumulator(
        zeros(nA), Float64[], Float64[], Float64[],
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        -Inf, -Inf, -Inf, -Inf,
    )
end

function merge_stats!(dest::StatsAccumulator, src::StatsAccumulator)
    length(dest.asset_mass) == length(src.asset_mass) ||
        error("Cannot merge statistics with different asset-grid sizes")
    dest.asset_mass .+= src.asset_mass
    append!(dest.distribution_weights, src.distribution_weights)
    append!(dest.hours_values, src.hours_values)
    append!(dest.consumption_values, src.consumption_values)
    dest.total_mass += src.total_mass
    dest.sum_current_assets += src.sum_current_assets
    dest.sum_labor_income += src.sum_labor_income
    dest.sum_borrowing_limit += src.sum_borrowing_limit
    dest.sum_effective_borrowing_limit += src.sum_effective_borrowing_limit
    dest.borrowing_limit_mass += src.borrowing_limit_mass
    dest.negative_asset_mass += src.negative_asset_mass
    dest.zero_asset_mass += src.zero_asset_mass
    dest.borrowing_constraint_mass += src.borrowing_constraint_mass
    dest.upper_bound_mass += src.upper_bound_mass
    dest.hours_upper_bound_mass += src.hours_upper_bound_mass
    dest.max_next_assets = max(dest.max_next_assets, src.max_next_assets)
    dest.max_hours = max(dest.max_hours, src.max_hours)
    dest.max_material_next_assets =
        max(dest.max_material_next_assets, src.max_material_next_assets)
    dest.max_material_hours = max(dest.max_material_hours, src.max_material_hours)
    return dest
end

function finalize_statistics(stats::StatsAccumulator, p::HDParams)
    # The unrestricted maxima, beside the material ones core_statistics
    # publishes. These are taken over every cell; the material pair is taken
    # only over cells carrying more than UPPER_BOUND_SHARE_TOL of mass, so a
    # state sitting at the grid edge with negligible mass shows up here and not
    # there. Only the hd family publishes both.
    max_next_assets = isfinite(stats.max_next_assets) ? stats.max_next_assets : NaN
    max_hours = isfinite(stats.max_hours) ? stats.max_hours : NaN
    return merge(core_statistics(stats, p),
                 (; maxNextAssets = max_next_assets, maxHours = max_hours))
end

# -----------------------------------------------------------------------------
# Distribution iteration for one kappa (bilinear Young lottery in s')
# -----------------------------------------------------------------------------
"""
    accumulate_stats!(stats, weighted_mass, ia, a, ap, h, c, y,
                      true_borrowing_limit, effective_borrowing_limit,
                      at_borrowing_constraint, at_asset_upper, h_upper,
                      binding_age, collect)

Add one (age, state) observation to a statistics accumulator.

Factored out of the forward pass so the several age windows below are fed from
ONE body and cannot drift apart -- the same arrangement `hdinf` and `hi` use.
`collect` is passed rather than read from `p` so an accumulator can skip the
distribution vectors, which are a cross-sectional object.

`binding_age` is the finite-horizon wrinkle: at the terminal age `a' >= 0`
replaces the borrowing limit, so that age contributes no mass to the limit
averages.
"""
@inline function accumulate_stats!(stats::StatsAccumulator, weighted_mass::Float64,
                                   ia::Int, a::Float64, ap::Float64, h::Float64,
                                   c::Float64, y::Float64,
                                   true_borrowing_limit::Float64,
                                   effective_borrowing_limit::Float64,
                                   at_borrowing_constraint::Bool,
                                   at_asset_upper::Bool, h_upper::Float64,
                                   binding_age::Bool, collect::Bool)
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
        if binding_age
            stats.borrowing_limit_mass += weighted_mass
        end
        stats.max_next_assets = max(stats.max_next_assets, ap)
        stats.max_hours = max(stats.max_hours, h)
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
