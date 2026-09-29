# =============================================================================
# params.jl
#
# The supertype every solver's parameter struct subtypes, and the handful of
# accessors that work for all of them.
#
# The functions below read fields (qBorr, qSav, a_grid, bbar, rho, z_grid,
# hMax) that HIParams and HDParams both carry, so annotating on the supertype
# is enough to serve all eight solvers from one definition. Julia specializes
# each call site on the concrete type, so the abstract annotation costs nothing
# in the inner loops.
# =============================================================================

"""
    AbstractBewleyParams

Supertype of every solver's parameter struct. Subtyping it is what lets the
shared accessors, the labor FOC and the asset helpers be written once.
"""
abstract type AbstractBewleyParams end

"""
    safe_ratio(num, den)

`num / den`, or `NaN` when the denominator is numerically zero.
"""
safe_ratio(num::Real, den::Real) = abs(den) > eps(Float64) ? Float64(num) / Float64(den) : NaN

"""
    upper_bound_level_tol(bound)

Relative tolerance for deciding whether a choice sits at a bound.
"""
upper_bound_level_tol(bound::Real) = 1e-8 * max(1.0, abs(Float64(bound)))

"""
    asset_price(ap, p)

Gross price of next-period assets: `qBorr` when borrowing, `qSav` when saving.
"""
asset_price(ap::Real, p::AbstractBewleyParams) = ap < 0.0 ? p.qBorr : p.qSav

"""
    asset_prices(p)

`asset_price` evaluated on the whole asset grid.
"""
asset_prices(p::AbstractBewleyParams) = [asset_price(ap, p) for ap in p.a_grid]

asset_upper_bound(p::AbstractBewleyParams) = maximum(p.a_grid)
hours_upper_bound(p::AbstractBewleyParams) = p.hMax

"""
    borrowing_limit(kappa, iz, p)

The limit `bbar*exp(kappa + rho*z)`, which scales with the permanent type and
the persistent state. The terminal-age override, where `a' >= 0` replaces it,
belongs to the finite-horizon solvers and stays with them.
"""
borrowing_limit(kappa::Real, iz::Int, p::AbstractBewleyParams) =
    p.bbar * exp(kappa + p.rho * p.z_grid[iz])

"""
    first_feasible_asset_indices(kappa, p)

Per-`z` index of the first grid point at or above the borrowing limit.
"""
function first_feasible_asset_indices(kappa::Float64, p::AbstractBewleyParams)
    nZ = length(p.z_grid)
    idx = Vector{Int}(undef, nZ)
    for iz in 1:nZ
        lower = borrowing_limit(kappa, iz, p)
        idx[iz] = searchsortedfirst(p.a_grid, lower - 1e-12)
        if idx[iz] > length(p.a_grid)
            error("No feasible next-period asset for kappa=$kappa, z=$(p.z_grid[iz])")
        end
    end
    return idx
end
