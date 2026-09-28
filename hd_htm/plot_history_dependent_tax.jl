# =============================================================================
# plot_history_dependent_tax.jl
#
# Plotting utilities for results returned by run_history_dependent_tax():
# age profiles, asset and hours distributions, and figure saving.
#
# Marek Kapicka, 2026
# =============================================================================

using Plots
using StatsBase

equilibrium_from(x) = hasproperty(x, :eq) ? x.eq : x

function params_from(x)
    if hasproperty(x, :params)
        return x.params
    elseif hasproperty(x, :parameters)
        return x.parameters
    end
    return x
end

function age_grid_for_series(result, series)
    p = params_from(result)
    if hasproperty(p, :J) && length(series) == p.J + 1
        return collect(0:p.J)
    end
    return collect(0:(length(series) - 1))
end

function weighted_histogram(values, weights; nbins::Int = 45, lo = nothing, hi = nothing)
    length(values) == length(weights) || error("values and weights must have the same length")
    total_weight = sum(weights)
    total_weight > 0.0 || error("histogram weights must sum to a positive value")

    lo_val = lo === nothing ? minimum(values) : Float64(lo)
    hi_val = hi === nothing ? maximum(values) : Float64(hi)
    if hi_val <= lo_val
        return [lo_val], [1.0], 1.0
    end

    edges = collect(range(lo_val, nextfloat(hi_val), length = nbins + 1))
    histogram = StatsBase.fit(
        StatsBase.Histogram, values, StatsBase.weights(weights), edges;
        closed = :left,
    )
    sum(histogram.weights) > 0.0 || error("histogram has no mass inside the plotted range")
    histogram = StatsBase.normalize(histogram, mode = :probability)
    masses = collect(histogram.weights)
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in 1:nbins]
    return centers, masses, edges[2] - edges[1]
end

function plot_weighted_distribution(values, weights; xlabel, title,
                                    nbins::Int = 45, bound = nothing)
    x, p, w = weighted_histogram(values, weights; nbins = nbins)
    plt = bar(x, p;
              bar_width = w,
              legend = isnothing(bound) ? false : :topright,
              label = isnothing(bound) ? "" : "probability mass",
              xlabel = xlabel,
              ylabel = "Probability mass",
              title = title,
              size = (900, 550))
    if !isnothing(bound)
        vline!(plt, [bound];
               linestyle = :dash, linewidth = 2, color = :red,
               label = "upper bound")
    end
    return plt
end

function plot_asset_grid(p)
    i = collect(eachindex(p.a_grid))
    return plot(i, p.a_grid;
                marker = :circle,
                linewidth = 2,
                legend = false,
                xlabel = "Grid index",
                ylabel = "Assets",
                title = "Asset grid",
                size = (900, 550))
end

function save_asset_grid_figure(p; output_dir = joinpath(@__DIR__, "figures"))
    mkpath(output_dir)
    path = joinpath(output_dir, "asset_grid.png")
    savefig(plot_asset_grid(p), path)
    return path
end

"""
    plot_sweep_welfare(sweep; field = :overallValueFunction, ylabel = "Welfare")

Plot a sweep field against mu2. Takes either a `sweep_mu2` result or the output
of `merge_mu2_sweeps`, so it works on a single-process sweep and on the merged
files of an array job alike.

Points that did not converge are drawn as hollow red markers rather than
dropped: a gap in the line is easy to misread as a feature of the welfare
profile, whereas a marked point is not. The optimum over the converged points is
annotated, since that is normally what the sweep is for.
"""
function plot_sweep_welfare(sweep; field::Symbol = :overallValueFunction,
                            ylabel = "Welfare", title = nothing)
    y = getproperty(sweep, field)
    ok = sweep.converged
    any(ok) || error("no converged points to plot")

    ttl = title === nothing ?
        "$(string(field)) against mu2  (mu1 = $(round(sweep.mu1, digits = 4)))" : title
    plt = plot(sweep.mu2[ok], y[ok];
               marker = :circle, markersize = 4, linewidth = 2,
               label = "converged", legend = :best,
               xlabel = "mu2", ylabel = ylabel, title = ttl,
               size = (900, 550))
    if any(.!ok)
        scatter!(plt, sweep.mu2[.!ok], y[.!ok];
                 marker = :circle, markersize = 6, markercolor = :white,
                 markerstrokecolor = :red, markerstrokewidth = 2,
                 label = "NOT converged")
    end
    i = argmax(y[ok])                       # welfare: higher is better
    xbest, ybest = sweep.mu2[ok][i], y[ok][i]
    scatter!(plt, [xbest], [ybest]; marker = :star5, markersize = 9,
             markercolor = :orange, label = @sprintf("max at mu2 = %.4f", xbest))
    return plt
end

function save_sweep_welfare_figure(sweep; output_dir = joinpath(@__DIR__, "figures"),
                                   filename = "sweep_welfare.png", kwargs...)
    mkpath(output_dir)
    path = joinpath(output_dir, filename)
    savefig(plot_sweep_welfare(sweep; kwargs...), path)
    return path
end

"""
    plot_s_grid(p; stock = 2, compare = nothing)

Plot one past-income stock grid: node value against grid index, in the same form
as `plot_asset_grid`. `stock = 1` or `2` selects s1 or s2. Pass a second
`HDParams` as `compare` to overlay another grid on the same axes -- the intended
use is `:quantile` against `:linear` at identical dimensions, where the point is
how far the quantile nodes concentrate around the initial condition s = 0.

The right panel shows the cell width `diff(grid)` against the cell midpoint on a
log scale, which is what actually governs interpolation error: a flat line is
equal spacing, and a deep trough marks where the grid is buying resolution.
"""
function plot_s_grid(p; stock::Int = 2, compare = nothing,
                     label = "grid", compare_label = "comparison")
    stock in (1, 2) || error("stock must be 1 or 2")
    g = stock == 1 ? p.s1_grid : p.s2_grid
    length(g) > 1 || error("s$(stock) grid is a single point (mu$(stock) = 0)")
    name = "s$(stock)"

    left = plot(eachindex(g), g;
                marker = :circle, markersize = 2, linewidth = 2,
                label = label, legend = :topleft,
                xlabel = "Grid index", ylabel = "$name",
                title = "$name grid")
    right = plot(midpoints_of(g), diff(g);
                 marker = :circle, markersize = 2, linewidth = 2,
                 yscale = :log10, label = label, legend = :top,
                 xlabel = "$name", ylabel = "Cell width",
                 title = "$name cell width")
    if compare !== nothing
        gc = stock == 1 ? compare.s1_grid : compare.s2_grid
        plot!(left, eachindex(gc), gc;
              marker = :circle, markersize = 2, linewidth = 2,
              linestyle = :dash, label = compare_label)
        plot!(right, midpoints_of(gc), diff(gc);
              marker = :circle, markersize = 2, linewidth = 2,
              linestyle = :dash, label = compare_label)
    end
    hline!(left, [0.0]; linestyle = :dot, color = :grey, label = "")
    return plot(left, right; layout = (1, 2), size = (1200, 500))
end

midpoints_of(g) = [(g[i] + g[i+1]) / 2 for i in 1:(length(g) - 1)]

function save_s_grid_figure(p; stock::Int = 2, compare = nothing,
                            output_dir = joinpath(@__DIR__, "figures"), kwargs...)
    mkpath(output_dir)
    path = joinpath(output_dir, "s$(stock)_grid.png")
    savefig(plot_s_grid(p; stock = stock, compare = compare, kwargs...), path)
    return path
end

function plot_average_assets_by_age(result)
    eq = equilibrium_from(result)
    hasproperty(eq, :A) || error("equilibrium object must contain aggregate assets `A`")
    ages = age_grid_for_series(result, eq.A)
    length(ages) == length(eq.A) || error("age grid length does not match eq.A")
    return plot(ages, eq.A;
                marker = :circle,
                linewidth = 2,
                legend = false,
                xlabel = "Age",
                ylabel = "Average assets",
                title = "Average assets by age",
                size = (900, 550))
end

function save_average_assets_by_age_figure(result; output_dir = joinpath(@__DIR__, "figures"))
    mkpath(output_dir)
    path = joinpath(output_dir, "average_assets_by_age.png")
    savefig(plot_average_assets_by_age(result), path)
    return path
end

function plot_average_hours_by_age(result)
    eq = equilibrium_from(result)
    hasproperty(eq, :H) || error("equilibrium object must contain aggregate hours `H`")
    ages = age_grid_for_series(result, eq.H)
    length(ages) == length(eq.H) || error("age grid length does not match eq.H")
    return plot(ages, eq.H;
                marker = :circle,
                linewidth = 2,
                legend = false,
                xlabel = "Age",
                ylabel = "Average hours worked",
                title = "Average hours worked by age",
                size = (900, 550))
end

function save_average_hours_by_age_figure(result; output_dir = joinpath(@__DIR__, "figures"))
    mkpath(output_dir)
    path = joinpath(output_dir, "average_hours_by_age.png")
    savefig(plot_average_hours_by_age(result), path)
    return path
end

function plot_average_consumption_by_age(result)
    eq = equilibrium_from(result)
    hasproperty(eq, :C) ||
        error("equilibrium object must contain aggregate consumption `C`")
    ages = age_grid_for_series(result, eq.C)
    length(ages) == length(eq.C) || error("age grid length does not match eq.C")
    return plot(ages, eq.C;
                marker = :circle,
                linewidth = 2,
                legend = false,
                xlabel = "Age",
                ylabel = "Average consumption",
                title = "Average consumption by age",
                size = (900, 550))
end

function save_average_consumption_by_age_figure(result;
                                                output_dir = joinpath(@__DIR__, "figures"))
    mkpath(output_dir)
    path = joinpath(output_dir, "average_consumption_by_age.png")
    savefig(plot_average_consumption_by_age(result), path)
    return path
end

function check_unconditional_distribution_masses(d; tol::Float64 = 1e-10)
    asset_total = hasproperty(d, :assetMassTotal) ? d.assetMassTotal : sum(d.assetMass)
    observation_total =
        hasproperty(d, :observationWeightTotal) ? d.observationWeightTotal : sum(d.weights)
    scale = max(1.0, abs(asset_total), abs(observation_total))
    if abs(asset_total - observation_total) > tol * scale
        error("Unconditional distribution mass mismatch: asset grid mass = $asset_total, observation weight mass = $observation_total")
    end
    return nothing
end

function save_unconditional_distribution_figures(eq; output_dir = joinpath(@__DIR__, "figures"))
    mkpath(output_dir)
    d = eq.statistics.unconditionalDistributions
    if isempty(d.weights)
        error("Cannot plot unconditional hours and consumption distributions because observation weights are empty. Recompute the final equilibrium with collect_distributions = true; if using cached calibration output, keep final_resolve = true.")
    end
    check_unconditional_distribution_masses(d)

    # Assets are already accumulated as probability mass on the asset grid.
    asset_plot = plot_weighted_distribution(
        d.assetGrid, d.assetMass;
        xlabel = "Assets",
        title = "Unconditional distribution of assets",
    )
    # Hours and consumption use observation-level values with matching weights.
    # The hours histogram marks the upper bound hMax so binding is visible.
    hours_bound = hasproperty(eq, :parameters) ? eq.parameters.hMax : nothing
    hours_plot = plot_weighted_distribution(
        d.hours, d.weights;
        xlabel = "Hours worked",
        title = "Unconditional distribution of hours worked",
        bound = hours_bound,
    )
    consumption_plot = plot_weighted_distribution(
        d.consumption, d.weights;
        xlabel = "Consumption",
        title = "Unconditional distribution of consumption",
    )

    paths = (
        assets = joinpath(output_dir, "unconditional_assets.png"),
        hours = joinpath(output_dir, "unconditional_hours_worked.png"),
        consumption = joinpath(output_dir, "unconditional_consumption.png"),
    )
    savefig(asset_plot, paths.assets)
    savefig(hours_plot, paths.hours)
    savefig(consumption_plot, paths.consumption)
    return paths
end

function save_history_dependent_figures(result; output_dir = joinpath(@__DIR__, "figures"))
    eq = equilibrium_from(result)
    params = params_from(result)
    asset_grid_path = save_asset_grid_figure(params; output_dir = output_dir)
    average_assets_by_age_path =
        save_average_assets_by_age_figure(result; output_dir = output_dir)
    average_hours_by_age_path =
        save_average_hours_by_age_figure(result; output_dir = output_dir)
    average_consumption_by_age_path =
        save_average_consumption_by_age_figure(result; output_dir = output_dir)
    distribution_paths = save_unconditional_distribution_figures(eq; output_dir = output_dir)
    return (; assetGrid = asset_grid_path,
            averageAssetsByAge = average_assets_by_age_path,
            averageHoursByAge = average_hours_by_age_path,
            averageConsumptionByAge = average_consumption_by_age_path,
            assets = distribution_paths.assets,
            hours = distribution_paths.hours,
            consumption = distribution_paths.consumption)
end

function display_path(path::AbstractString)
    # Prefer a short project-relative path (e.g. "figures/assets.png").
    project_dir = @__DIR__
    if startswith(path, project_dir)
        rel = relpath(path, project_dir)
        startswith(rel, "..") || return rel
    end

    home = homedir()
    if startswith(path, home)
        return replace(path, home => "~"; count = 1)
    end
    return path
end

"""
    plot_history_dependent_tax(result; output_dir = joinpath(@__DIR__, "figures"),
                               print_paths = true)

Save the standard history-dependent-tax figures from either a raw equilibrium
or a result object returned by `run_history_dependent_tax()`.

Returns a NamedTuple
`(; assetGrid, averageAssetsByAge, averageHoursByAge, averageConsumptionByAge,
assets, hours, consumption)` with the saved figure paths.
"""
function plot_history_dependent_tax(result; output_dir = joinpath(@__DIR__, "figures"),
                                    print_paths::Bool = true)
    figures = save_history_dependent_figures(result; output_dir = output_dir)

    if print_paths
        println("\n=== Figures saved ===")
        println("asset grid                = $(display_path(figures.assetGrid))")
        println("average assets by age     = $(display_path(figures.averageAssetsByAge))")
        println("average hours by age      = $(display_path(figures.averageHoursByAge))")
        println("average consumption by age= $(display_path(figures.averageConsumptionByAge))")
        println("assets                    = $(display_path(figures.assets))")
        println("hours worked              = $(display_path(figures.hours))")
        println("consumption               = $(display_path(figures.consumption))")
    end

    return figures
end
