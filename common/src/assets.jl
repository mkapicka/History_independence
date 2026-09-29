# =============================================================================
# assets.jl
#
# Asset-grid construction and the lookups that place a chosen a' back on the
# grid: the Young (1990) two-point lottery, its rounded single-index form, and
# the tolerance for deciding a choice sits on a bound.
# =============================================================================

"""
    default_asset_grid(bbar, aMax, nA, rho, kappa_grid, z_grid; grid options...)

Asset grid spanning the loosest borrowing limit `min_{kappa,z} bbar*exp(kappa +
rho*z)` up to `aMax`.
"""
function default_asset_grid(bbar::Float64, aMax::Float64, nA::Int, rho::Float64,
                            kappa_grid::Vector{Float64}, z_grid::Vector{Float64};
                            kwargs...)
    bbar <= 0.0 ||
        error("Use bbar <= 0. For a borrowing limit B > 0, pass bbar = -B.")
    amin = minimum(bbar * exp(kappa + rho * z) for kappa in kappa_grid for z in z_grid)
    return asset_grid_with_zero(amin, aMax, nA; kwargs...)
end

asset_choice_bound_tol(bound::Real, p::AbstractBewleyParams) =
    max(1e-10, p.asset_choice_tol * max(1.0, abs(Float64(bound))))

function asset_transition_weights(ap::Float64, p::AbstractBewleyParams)
    grid = p.a_grid
    nA = length(grid)

    if ap <= grid[1]
        return 1, 1, 0.0
    elseif ap >= grid[nA]
        return nA, nA, 0.0
    end

    hi = searchsortedfirst(grid, ap)
    if hi <= nA && abs(ap - grid[hi]) <= asset_choice_bound_tol(grid[hi], p)
        return hi, hi, 0.0
    end

    lo = hi - 1
    weight_hi = (ap - grid[lo]) / (grid[hi] - grid[lo])
    return lo, hi, weight_hi
end

function nearest_asset_index(ap::Float64, p::AbstractBewleyParams)
    lo, hi, weight_hi = asset_transition_weights(ap, p)
    return weight_hi <= 0.5 ? lo : hi
end
