# =============================================================================
# labor.jl
#
# The static labor first-order condition and its root finders, plus the uniform
# hours grid. Used by the hi family; the hd family chooses (a', h) jointly,
# because hours there move the past-income stocks and the static FOC is invalid.
#
# `uniform_labor_grid` is the hi family's 3-argument equally spaced grid. It is
# NOT `build_labor_grid` in shocks.jl, which is 4-argument and log-spaced by
# default. They were distinct functions sharing one name until this move.
# =============================================================================

function optimal_labor_foc(cash::Float64, income_coeff::Float64, p::AbstractBewleyParams)
    tau = p.tau
    h_low = p.hMin
    h_high = p.hMax

    if income_coeff <= 0.0
        if cash <= 0.0
            return -Inf, NaN
        end
        h = h_low
        c = cash
        return log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta), h
    end

    if cash + income_coeff * h_high^(1.0 - tau) <= 0.0
        return -Inf, NaN
    end

    if cash + income_coeff * h_low^(1.0 - tau) <= 0.0
        h_low = ((-cash / income_coeff) * (1.0 + 1e-12))^(1.0 / (1.0 - tau))
        h_low = min(max(h_low, p.hMin), h_high)
        if cash + income_coeff * h_low^(1.0 - tau) <= 0.0
            h_low = nextfloat(h_low)
        end
    end

    if p.labor_solver == :grid
        return optimal_labor_grid(h_low, h_high, cash, income_coeff, p)
    end

    d_low = labor_foc_residual(h_low, cash, income_coeff, p)
    d_high = labor_foc_residual(h_high, cash, income_coeff, p)

    if d_low <= 0.0
        h = h_low
    elseif d_high >= 0.0
        h = h_high
    else
        h = solve_labor_root(h_low, h_high, cash, income_coeff, p)
    end

    c = cash + income_coeff * h^(1.0 - tau)
    if c <= 0.0
        return -Inf, NaN
    end
    u = log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta)
    return u, h
end

function solve_labor_root(h_low::Float64, h_high::Float64, cash::Float64,
                          income_coeff::Float64, p::AbstractBewleyParams)
    if p.labor_solver == :brent
        f(h) = labor_foc_residual(h, cash, income_coeff, p)
        return Roots.find_zero(f, (h_low, h_high), Roots.Brent())
    elseif p.labor_solver == :hybrid_newton
        return labor_root_hybrid_newton(h_low, h_high, cash, income_coeff, p)
    end
    error("Unknown labor_solver = $(p.labor_solver)")
end

function optimal_labor_grid(h_low::Float64, h_high::Float64, cash::Float64,
                            income_coeff::Float64, p::AbstractBewleyParams)
    first_h = searchsortedfirst(p.h_grid, h_low - 1e-12)
    best_u = -Inf
    best_h = NaN

    @inbounds for ih in first_h:length(p.h_grid)
        h = p.h_grid[ih]
        if h > h_high + 1e-12
            break
        end

        c = cash + income_coeff * p.h_grid_income_power[ih]
        if c <= 0.0
            continue
        end

        u = log(c) - p.h_grid_disutility[ih]
        if u > best_u
            best_u = u
            best_h = h
        end
    end

    if !isfinite(best_u)
        return -Inf, NaN
    end
    return best_u, best_h
end

function labor_root_hybrid_newton(h_low::Float64, h_high::Float64, cash::Float64,
                                  income_coeff::Float64, p::AbstractBewleyParams)
    lo = h_low
    hi = h_high
    h = 0.5 * (lo + hi)

    for _ in 1:50
        f = labor_foc_residual(h, cash, income_coeff, p)
        if abs(f) <= 1e-12
            return h
        end

        if f > 0.0
            lo = h
        else
            hi = h
        end

        fp = labor_foc_residual_derivative(h, cash, income_coeff, p)
        h_newton = h - f / fp
        if isfinite(h_newton) && lo < h_newton < hi
            h = h_newton
        else
            h = 0.5 * (lo + hi)
        end

        if hi - lo <= 1e-12 * max(1.0, abs(h))
            return 0.5 * (lo + hi)
        end
    end

    return 0.5 * (lo + hi)
end

function labor_foc_residual(h::Float64, cash::Float64, income_coeff::Float64, p::AbstractBewleyParams)
    c = cash + income_coeff * h^(1.0 - p.tau)
    if c <= 0.0
        return Inf
    end
    return income_coeff * (1.0 - p.tau) - p.phi * h^(p.eta + p.tau) * c
end

function labor_foc_residual_derivative(h::Float64, cash::Float64,
                                       income_coeff::Float64, p::AbstractBewleyParams)
    c = cash + income_coeff * h^(1.0 - p.tau)
    if c <= 0.0
        return -Inf
    end
    return -p.phi * (
        (p.eta + p.tau) * h^(p.eta + p.tau - 1.0) * c +
        income_coeff * (1.0 - p.tau) * h^(p.eta)
    )
end

function uniform_labor_grid(hMin::Float64, hMax::Float64, labor_grid_size::Int)
    hMin > 0.0 || error("hMin must be positive")
    hMax > hMin || error("hMax must exceed hMin")
    labor_grid_size >= 2 || error("labor_grid_size must be at least 2")
    return collect(range(hMin, hMax, length = labor_grid_size))
end

function normalize_labor_grid(h_grid, hMin::Float64, hMax::Float64)
    grid = sort(unique(collect(Float64.(h_grid))))
    all(h -> hMin - 1e-12 <= h <= hMax + 1e-12, grid) ||
        error("h_grid entries must lie inside [hMin, hMax]")
    return sort(unique(vcat(hMin, grid, hMax)))
end
