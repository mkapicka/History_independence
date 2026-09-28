# =============================================================================
# common/grids.jl
#
# Grid construction and small numeric helpers, shared by ALL FOUR solvers in
# this directory: hi, hiinf, hd and hdinf. Everything here was byte-identical in
# all four before extraction, so pooling it changes nothing and is verified to
# reproduce every solver bit-for-bit.
#
# WHY THIS FILE EXISTS. The four solvers were written self-contained, each with
# its own copy of these helpers, so the codebases could evolve independently.
# That worked, but copies do not stay in sync on their own: four bugs fixed in
# hiinf (the settled-age break truncating the age profiles, the cross-kappa
# staircase read by the government budget, the settled-age weighting of the
# statistics, and PV sums running over the zero tail) were still sitting in
# hdinf afterwards, because there was nothing to fix once.
#
# WHAT DOES NOT BELONG HERE. Only code with no model semantics and no
# HDParams/HIParams argument. The solvers themselves stay separate; that
# separation is deliberate and was never the cause of the drift. Helpers that
# had already diverged between the hi and hd families stay local to each -- see
# common/shocks_hd.jl for the hd-family set. `build_labor_grid` in particular
# takes a different number of arguments in the two families, so pooling it
# would be a behaviour change rather than a refactor.
#
# INCLUDED, NOT IMPORTED. Each solver `include`s this inside its own module, so
# the methods land in that module's scope exactly as when they were written out
# inline. No package, no LOAD_PATH entry, no Project.toml change.
# =============================================================================

function validate_transition(P::Matrix{Float64}, n::Int)
    size(P) == (n, n) || error("transition matrix must be $n x $n")
    for i in 1:n
        s = sum(P[i, :])
        abs(s - 1.0) <= 1e-8 || error("transition row $i sums to $s, not 1")
        all(P[i, :] .>= -1e-14) || error("transition row $i has negative probabilities")
    end
end

function normalize_probabilities(p::Vector{Float64}, name::String)
    all(p .>= -1e-14) || error("$name has negative probabilities")
    s = sum(p)
    s > 0.0 || error("$name sums to zero")
    p ./= s
    return p
end

# -----------------------------------------------------------------------------
# Asset and labor grids (replicated infrastructure)
# -----------------------------------------------------------------------------
function asset_grid_with_zero(amin::Float64, amax::Float64, nA::Int;
                              method::Symbol = :nonuniform,
                              curvature_borrow::Float64 = 1.8,
                              curvature_save::Float64 = 2.5,
                              borrow_share::Float64 = 0.35,
                              zero_share::Float64 = 0.0,
                              zero_width::Float64 = 0.0)
    amin <= 0.0 <= amax || error("asset grid bounds must contain 0")
    nA >= 3 || error("nA must be at least 3")
    method in (:nonuniform, :linear) || error("unknown asset grid method")

    if method == :linear
        return linear_asset_grid(amin, amax, nA)
    end

    curvature_borrow > 0.0 || error("curvature_borrow must be positive")
    curvature_save > 0.0 || error("curvature_save must be positive")
    0.0 < borrow_share < 1.0 || error("borrow_share must be in (0, 1)")
    0.0 <= zero_share < 1.0 || error("zero_share must be in [0, 1)")
    zero_width >= 0.0 || error("zero_width must be nonnegative")

    if abs(amin) <= 1e-14
        return nonnegative_asset_grid(amax, nA; curvature_save = curvature_save)
    end

    if zero_share <= 0.0 || zero_width <= 0.0
        return two_region_asset_grid(
            amin, amax, nA;
            borrow_share = borrow_share,
            curvature_borrow = curvature_borrow,
            curvature_save = curvature_save,
        )
    end

    return zero_band_asset_grid(
        amin, amax, nA;
        borrow_share = borrow_share,
        curvature_borrow = curvature_borrow,
        curvature_save = curvature_save,
        zero_share = zero_share,
        zero_width = zero_width,
    )
end

function linear_asset_grid(amin::Float64, amax::Float64, nA::Int)
    grid = collect(range(amin, amax, length = nA))
    grid[nearest_index(grid, 0.0)] = 0.0
    sort!(grid)
    return grid
end

function nonnegative_asset_grid(amax::Float64, nA::Int; curvature_save::Float64)
    x = collect(range(0.0, 1.0, length = nA))
    grid = amax .* (x .^ curvature_save)
    grid[1] = 0.0
    grid[end] = amax
    return grid
end

function two_region_asset_grid(amin::Float64, amax::Float64, nA::Int;
                               borrow_share::Float64,
                               curvature_borrow::Float64,
                               curvature_save::Float64)
    n_borrow = clamp(round(Int, borrow_share * (nA - 1)), 1, nA - 2)
    n_save = nA - n_borrow

    xb = collect(range(0.0, 1.0, length = n_borrow + 1))
    borrow_grid = amin .+ (0.0 - amin) .* (1.0 .- (1.0 .- xb) .^ curvature_borrow)

    xs = collect(range(0.0, 1.0, length = n_save))
    save_grid = amax .* (xs .^ curvature_save)

    grid = vcat(borrow_grid[1:end-1], save_grid)
    grid[nearest_index(grid, 0.0)] = 0.0
    grid[1] = amin
    grid[end] = amax
    sort!(grid)
    return grid
end

function zero_band_asset_grid(amin::Float64, amax::Float64, nA::Int;
                              borrow_share::Float64,
                              curvature_borrow::Float64,
                              curvature_save::Float64,
                              zero_share::Float64,
                              zero_width::Float64)
    zero_low = max(amin, -zero_width)
    zero_high = min(amax, zero_width)
    zero_low < 0.0 < zero_high || error("zero_width must create a band around zero")

    n_zero = clamp(round(Int, zero_share * nA), 3, nA - 2)
    n_outer = nA - n_zero
    n_borrow = clamp(round(Int, borrow_share * n_outer), 1, n_outer - 1)
    n_save = n_outer - n_borrow

    xb = collect(range(0.0, 1.0, length = n_borrow + 1))
    borrow_grid = amin .+ (zero_low - amin) .* (1.0 .- (1.0 .- xb) .^ curvature_borrow)

    xz = collect(range(0.0, 1.0, length = n_zero))
    zero_grid = zero_low .+ (zero_high - zero_low) .* xz

    xs = collect(range(0.0, 1.0, length = n_save + 1))
    save_grid = zero_high .+ (amax - zero_high) .* (xs .^ curvature_save)

    grid = vcat(borrow_grid[1:end-1], zero_grid, save_grid[2:end])
    grid[nearest_index(grid, 0.0)] = 0.0
    grid[1] = amin
    grid[end] = amax
    sort!(grid)
    return grid
end

function nearest_index(x::AbstractVector{<:Real}, value::Real)
    return argmin(abs.(x .- value))
end

"""
    interpolated_weighted_quantile(grid, mass, prob = 0.5)

The `prob` quantile of the discrete distribution putting weight `mass[k]` on
`grid[k]`, interpolated linearly in the CDF instead of snapped to a grid point.
`grid` must be sorted ascending and `mass` nonnegative.

WHY NOT `StatsBase.median(grid, weights(mass))`. Not because it snaps to the
grid -- it does not, it interpolates, and it is continuous in `mass`. The reason
is accuracy: its convention is built for frequency weights on observed data,
and on a coarse nonuniform grid carrying a DISCRETIZED CONTINUOUS distribution
it is biased low. Measured against a known truth -- an exponential with mean 2,
median 2 ln 2, discretized onto this asset grid by cell probability -- at
nA = 151 it errs by 7.98e-02, or 5.8% of the median, while the mid-cumulative
interpolation below errs by 5.69e-05. Refinement does not close the gap: from
nA = 51 to nA = 601 the StatsBase error falls only from 2.38e-01 to 2.06e-02,
while this one falls from 1.91e-03 to 1.03e-06. On a grid whose cells near the
median are 0.16 wide, that bias is the same order as the moments being matched.

BEHAVIOUR AT AN ATOM. The result is continuous in `mass` everywhere, atoms
included, which is precisely the property the calibration needs -- but the price
is that it is then NOT the textbook median of the discrete distribution. Where a
single grid point carries enough mass to straddle `prob`, the discrete median is
that point, while this function returns a position interpolated across the
neighbouring cell; the two coincide only when the straddling atom is centred on
`prob`, and differ by up to one cell otherwise. In this model the atom that
matters is the mass at the borrowing constraint, so a median sitting there is to
be read as the quantile of the SMOOTHED distribution rather than as the
constraint itself. Away from an atom -- where the model's median currently sits,
around a/y = 1.2 -- the two definitions differ only by the interpolation, which
is the whole point.

CONVENTION. Mid-cumulative positions: point `k` sits at `(C_k - mass_k/2)/W`,
with `C` the cumulative sum and `W` the total, and `grid` is read off linearly
against those positions, clamped to the end points. This is the usual
interpolated weighted percentile; it is continuous in `mass`, which is the
property the calibration needs. Zero-mass grid points are dropped first, so runs
of empty cells cannot produce a zero denominator.
"""
function interpolated_weighted_quantile(grid::AbstractVector{<:Real},
                                        mass::AbstractVector{<:Real},
                                        prob::Real = 0.5)
    length(grid) == length(mass) ||
        error("grid and mass must have the same length")
    0.0 <= prob <= 1.0 || error("prob must be in [0, 1], got $(prob)")

    keep = findall(>(0.0), mass)
    isempty(keep) && return NaN
    a = @view grid[keep]
    w = @view mass[keep]
    W = sum(w)
    W > 0.0 || return NaN
    length(a) == 1 && return Float64(a[1])

    cum = 0.0
    Fprev = 0.0
    aprev = Float64(a[1])
    for k in eachindex(a)
        cum += w[k]
        F = (cum - w[k] / 2) / W
        ak = Float64(a[k])
        if prob <= F
            den = F - Fprev
            return den > 0.0 ? aprev + (ak - aprev) * (prob - Fprev) / den : ak
        end
        Fprev = F
        aprev = ak
    end
    return Float64(a[end])
end

function discounted_sum(x::AbstractVector{<:Real}, q::Real)
    total = 0.0
    for (j, val) in enumerate(x)
        total += q^(j - 1) * val
    end
    return total
end
