# =============================================================================
# common/shocks.jl
#
# Shock discretization and interpolation helpers shared by ALL EIGHT solvers.
# Until 2026-09-29 the hi family carried its own copies of quantecon_ar1,
# ar1_conditional_probabilities, normal_gauss_hermite and normal_cdf; they were
# byte-identical to these (quantecon_ar1 up to one temp variable), so the copies
# were deleted and this file is now included by the hi family too. See
# common/grids.jl for the rest, and note that build_labor_grid here is NOT the
# hi family's: theirs is a 3-argument uniform grid, this one is 4-argument and
# log-spaced by default.
#
# `normal_cdf` here is the deduplicated version. Both hd and hdinf previously
# defined it TWICE -- once taking ::Float64 (Abramowitz and Stegun 26.2.17) and
# once taking ::Real (QuantEcon) -- so which implementation ran depended
# silently on the argument type, and the two disagreed by 5.6e-9.
#
# The extraction originally kept the A&S method, widened to ::Real. That was the
# wrong one of the two to keep: it is accurate to 7.5e-8, and `z0_probs` -- which
# `build_markov_shock` below produces, and which seeds the forward pass and
# weights the birth value function -- is built with it. As of 2026-09-19 the
# QuantEcon branch is what survives, matching `hi`/`hiinf` byte for byte. See the
# docstring at `normal_cdf` for the measurement. This DOES move every hd-family
# number in about the ninth digit relative to runs made before that date.
# =============================================================================

# -----------------------------------------------------------------------------
# Shock discretization (replicated infrastructure)
# -----------------------------------------------------------------------------
function build_markov_shock(name::String, n::Int, rho::Float64,
                            innovation_mean::Float64, innovation_sd::Float64,
                            initial_state::Float64, tauchen_width::Float64,
                            method::Symbol)
    grid, P = quantecon_ar1(n, rho, innovation_mean, innovation_sd;
                            method = method, width = tauchen_width)
    initial_probs = ar1_conditional_probabilities(initial_state, grid, rho,
                                                  innovation_mean, innovation_sd)
    initial_probs = normalize_probabilities(initial_probs, "$(name)0_probs")
    validate_transition(P, length(grid))
    length(initial_probs) == length(grid) ||
        error("$(name)0_probs length must match $(name)_grid")
    return grid, P, initial_probs
end

function build_iid_normal_shock(name::String, n::Int, mean::Float64, sd::Float64)
    grid, probs = normal_gauss_hermite(n, mean, sd)
    probs = normalize_probabilities(probs, "P$(name)")
    length(probs) == length(grid) || error("P$(name) length must match $(name)_grid")
    return grid, probs
end

function quantecon_ar1(n::Int, rho::Float64, innovation_mean::Float64,
                       innovation_sd::Float64; method::Symbol = :rouwenhorst,
                       width::Float64 = 3.0)
    n >= 1 || error("n must be positive")
    innovation_sd >= 0.0 || error("innovation_sd must be nonnegative")
    method in (:rouwenhorst, :tauchen) ||
        error("method must be :rouwenhorst or :tauchen")
    abs(rho) < 1.0 || error("rho must satisfy |rho| < 1")

    unconditional_mean = innovation_mean / (1.0 - rho)
    if n == 1 || innovation_sd == 0.0
        return [unconditional_mean], ones(1, 1)
    end

    mc = method == :rouwenhorst ?
         QuantEcon.rouwenhorst(n, rho, innovation_sd, innovation_mean) :
         QuantEcon.tauchen(n, rho, innovation_sd, innovation_mean, width)
    grid = collect(Float64.(mc.state_values))
    P = Matrix{Float64}(mc.p)
    return grid, P
end

function ar1_conditional_probabilities(current_z::Real, grid::AbstractVector{<:Real},
                                       rho::Float64, innovation_mean::Float64,
                                       innovation_sd::Float64)
    grid = collect(Float64.(grid))
    n = length(grid)
    n >= 1 || error("grid must be nonempty")
    if n == 1 || innovation_sd == 0.0
        p = zeros(n)
        p[nearest_index(grid, innovation_mean + rho * current_z)] = 1.0
        return p
    end

    mean_next = innovation_mean + rho * current_z
    cutoffs = [(grid[i] + grid[i + 1]) / 2.0 for i in 1:(n - 1)]
    probs = Vector{Float64}(undef, n)
    probs[1] = normal_cdf((cutoffs[1] - mean_next) / innovation_sd)
    for j in 2:(n - 1)
        upper = (cutoffs[j] - mean_next) / innovation_sd
        lower = (cutoffs[j - 1] - mean_next) / innovation_sd
        probs[j] = normal_cdf(upper) - normal_cdf(lower)
    end
    probs[n] = 1.0 - normal_cdf((cutoffs[end] - mean_next) / innovation_sd)
    return normalize_probabilities(probs, "AR(1) conditional probabilities")
end

function normal_gauss_hermite(n::Int, mean::Float64, sd::Float64)
    n >= 1 || error("n must be positive")
    sd >= 0.0 || error("sd must be nonnegative")
    if n == 1 || sd == 0.0
        return [mean], [1.0]
    end
    nodes, weights = gausshermite(n; normalize = true)
    probs = normalize_probabilities(collect(weights), "Gauss-Hermite weights")
    grid = mean .+ sd .* nodes
    return collect(grid), probs
end

"""
    normal_cdf(x)

Standard normal CDF, via `QuantEcon.std_norm_cdf`, clamped beyond +-8 sigma
where the result is 0 or 1 to double precision anyway. Byte-for-byte the same
function `hi`/`hiinf` define locally, which is what lets the hd and hi solvers
be compared at `mu1 = mu2 = 0`.

WHY NOT THE RATIONAL APPROXIMATION IT REPLACED. This used to be the Zelen and
Severo formula (Abramowitz and Stegun eq. 26.2.17), maximum absolute error
7.5e-8, justified in a comment as "used only to place s-grid nodes at
distribution quantiles ... and never in the solution itself". That was wrong on
the second half: `build_markov_shock` above calls it to build `z0_probs`, the
initial distribution over `z`, which seeds the forward pass and weights the
birth value function. Measured 2026-09-19, it put `z0_probs` 1.68e-08 away from
the `hi` values -- consistent with the 7.5e-08 bound -- and that single
difference accounted for the whole hd-vs-hi gap: lambda 1.1e-09, PV output
4.3e-08, welfare 2.3e-09, with every other constructed object agreeing to
0.0e+00. The s-grid placement this was written for is unaffected either way,
since it needs far less accuracy than either version delivers.

QuantEcon is a declared dependency of every solver that includes this file, so
nothing new is pulled in; the original note about avoiding SpecialFunctions no
longer applies.
"""
function normal_cdf(x::Real)
    z = Float64(x)
    if z < -8.0
        return 0.0
    elseif z > 8.0
        return 1.0
    end
    return QuantEcon.std_norm_cdf(z)
end

function build_labor_grid(hMin::Float64, hMax::Float64, labor_grid_size::Int,
                          h_grid; spacing::Symbol = :log)
    if isempty(h_grid)
        if spacing == :log
            # Geometric spacing: uniform in ln h. This matches the model's
            # structure -- s' depends on ln h, and income (h^pow) and
            # disutility (h^(1+eta)) are power functions, so RELATIVE hours
            # resolution is what matters. A uniform grid wastes points at
            # high h and is catastrophically coarse in ln h near hMin.
            return exp.(collect(range(log(hMin), log(hMax),
                                      length = labor_grid_size)))
        end
        return collect(range(hMin, hMax, length = labor_grid_size))
    end
    grid = sort(unique(collect(Float64.(h_grid))))
    all(h -> hMin - 1e-12 <= h <= hMax + 1e-12, grid) ||
        error("h_grid entries must lie inside [hMin, hMax]")
    grid = sort(unique(vcat(hMin, grid, hMax)))
    return grid
end

"""
    grid_lookup_weights(grid, x)

Clamped linear-interpolation lookup: `(lo, hi, w)` with
`x ~ (1-w)*grid[lo] + w*grid[hi]`; out-of-bounds clamps to the nearest
endpoint with `lo == hi`; a single-point grid returns `(1, 1, 0.0)`.
"""
function grid_lookup_weights(grid::Vector{Float64}, x::Float64)
    n = length(grid)
    if n == 1 || x <= grid[1]
        return 1, 1, 0.0
    elseif x >= grid[n]
        return n, n, 0.0
    end
    hi = searchsortedfirst(grid, x)
    lo = hi - 1
    w = (x - grid[lo]) / (grid[hi] - grid[lo])
    return lo, hi, w
end

function find_bracket(grid, residuals)
    for i in 1:(length(grid) - 1)
        # sign(0.0) is 0.0, so an exact zero at either end also brackets, and
        # the interval returned is never degenerate. Returning (i, i) on an
        # exact zero -- which this did until 2026-09-29 -- hands Roots.Brent a
        # zero-width interval, which throws ArgumentError("Need extrema to
        # return two distinct values"). Every caller wraps the solve in a
        # try/catch, so the throw was swallowed and the solver reported
        # convergence failure in the one case where it had found the exact
        # root. The hi family always used this convention; the hd family did
        # not, and this is what brings them together.
        if isfinite(residuals[i]) && isfinite(residuals[i + 1]) &&
           sign(residuals[i]) != sign(residuals[i + 1])
            return (i, i + 1)
        end
    end
    return nothing
end

"""
    discounted_sum_with_tail(x, q)

Present value of `x` over an INFINITE horizon, with the path held constant at
`x[end]` past the last element:

    sum_{j=1}^{n} q^(j-1) x_j  +  x_n * q^n / (1 - q).

`discounted_sum` alone stops dead at `maxAge`, dropping `q^maxAge` of the total
-- 0.24% at qGov = 0.99 and maxAge = 600. The government budget never had this
problem because it already closed its own tail; this applies the same closure
to the reported present values and makes them independent of maxAge.
"""
function discounted_sum_with_tail(x::AbstractVector{<:Real}, q::Real)
    isempty(x) && return 0.0
    0 <= q < 1 || error("discounted_sum_with_tail needs 0 <= q < 1, got q = $q")
    n = length(x)
    return discounted_sum(x, q) + x[n] * q^n / (1 - q)
end
