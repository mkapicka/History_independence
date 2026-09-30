# =============================================================================
# asset_choice.jl
#
# The `asset_choice_method = :interpolate` path: a continuous choice of a', found
# by maximizing a piecewise interpolant of the continuation value over each
# segment of the asset grid, instead of searching the grid points themselves.
#
# The five functions form a closed group -- they call each other and nothing
# else outside this package -- and were byte-identical in all four
# history-independent solvers before being lifted here. They are listed in
# dependency order, leaves first.
#
# This path is off by default (:grid_search is the shipped setting), but it is a
# live alternative rather than dead code: the solvers' own notes record that
# :grid_search biases the average MPC up by about 3% at the production nA = 151.
# Its golden master is the verify/baseline-interpolate capture.
#
# Marek Kapicka, 2026
# =============================================================================

function interpolate_asset_value(ap::Float64, a_grid::Vector{Float64}, values, iz::Int)
    nA = length(a_grid)
    if ap <= a_grid[1]
        return values[1, iz]
    elseif ap >= a_grid[nA]
        return values[nA, iz]
    end

    hi = searchsortedfirst(a_grid, ap)
    if hi <= nA && a_grid[hi] == ap
        return values[hi, iz]
    end
    lo = hi - 1
    weight_hi = (ap - a_grid[lo]) / (a_grid[hi] - a_grid[lo])
    return (1.0 - weight_hi) * values[lo, iz] + weight_hi * values[hi, iz]
end

function interpolated_choice_value(ap::Float64, a::Float64,
                                   income_coeff::Float64, EV, iz::Int,
                                   p::AbstractBewleyParams, util_weight::Float64,
                                   beta::Float64)
    cash = a - asset_price(ap, p) * ap
    u, h = optimal_labor_foc(cash, income_coeff, p)
    if !isfinite(u)
        return -Inf, p.hMin
    end
    continuation = interpolate_asset_value(ap, p.a_grid, EV, iz)
    return util_weight * u + beta * continuation, h
end

function maximize_asset_segment(lo::Float64, hi::Float64, a::Float64,
                                income_coeff::Float64, EV, iz::Int,
                                p::AbstractBewleyParams, util_weight::Float64,
                                beta::Float64)
    invphi = (sqrt(5.0) - 1.0) / 2.0
    c = hi - invphi * (hi - lo)
    d = lo + invphi * (hi - lo)
    vc, hc = interpolated_choice_value(c, a, income_coeff, EV, iz, p, util_weight, beta)
    vd, hd = interpolated_choice_value(d, a, income_coeff, EV, iz, p, util_weight, beta)

    for _ in 1:p.asset_choice_max_iter
        if hi - lo <= p.asset_choice_tol * max(1.0, abs(0.5 * (lo + hi)))
            break
        end

        if vc < vd
            lo = c
            c = d
            vc = vd
            hc = hd
            d = lo + invphi * (hi - lo)
            vd, hd = interpolated_choice_value(
                d, a, income_coeff, EV, iz, p, util_weight, beta,
            )
        else
            hi = d
            d = c
            vd = vc
            hd = hc
            c = hi - invphi * (hi - lo)
            vc, hc = interpolated_choice_value(
                c, a, income_coeff, EV, iz, p, util_weight, beta,
            )
        end
    end

    if vc >= vd
        return c, vc, hc
    end
    return d, vd, hd
end

function update_with_asset_segment_max(best_val::Float64, best_ap::Float64,
                                       best_h::Float64, lo::Float64, hi::Float64,
                                       a::Float64, income_coeff::Float64, EV,
                                       iz::Int, p::AbstractBewleyParams,
                                       util_weight::Float64, beta::Float64)
    if hi - lo <= p.asset_choice_tol * max(1.0, abs(hi))
        return best_val, best_ap, best_h
    end

    ap, val, h = maximize_asset_segment(
        lo, hi, a, income_coeff, EV, iz, p, util_weight, beta,
    )
    if val > best_val
        return val, ap, h
    end
    return best_val, best_ap, best_h
end

function interpolated_asset_choice(a::Float64, lower::Float64,
                                   income_coeff::Float64, EV, iz::Int,
                                   p::AbstractBewleyParams, util_weight::Float64,
                                   beta::Float64)
    upper = asset_upper_bound(p)
    lower = min(max(lower, p.a_grid[1]), upper)

    best_val, best_h = interpolated_choice_value(
        lower, a, income_coeff, EV, iz, p, util_weight, beta,
    )
    best_ap = lower

    val_upper, h_upper = interpolated_choice_value(
        upper, a, income_coeff, EV, iz, p, util_weight, beta,
    )
    if val_upper > best_val
        best_val = val_upper
        best_ap = upper
        best_h = h_upper
    end

    if lower < 0.0 < upper
        val_zero, h_zero = interpolated_choice_value(
            0.0, a, income_coeff, EV, iz, p, util_weight, beta,
        )
        if val_zero > best_val
            best_val = val_zero
            best_ap = 0.0
            best_h = h_zero
        end
    end

    if lower < 0.0
        segment_hi = min(0.0, upper)
        best_val, best_ap, best_h = update_with_asset_segment_max(
            best_val, best_ap, best_h, lower, segment_hi,
            a, income_coeff, EV, iz, p, util_weight, beta,
        )
    end

    if upper > 0.0
        segment_lo = max(0.0, lower)
        best_val, best_ap, best_h = update_with_asset_segment_max(
            best_val, best_ap, best_h, segment_lo, upper,
            a, income_coeff, EV, iz, p, util_weight, beta,
        )
    end

    return best_val, best_ap, nearest_asset_index(best_ap, p), best_h
end
