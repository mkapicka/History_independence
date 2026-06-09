using Plots
using StatsBase

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

function plot_weighted_distribution(values, weights; xlabel, title, nbins::Int = 45)
    x, p, w = weighted_histogram(values, weights; nbins = nbins)
    return bar(x, p;
               bar_width = w,
               legend = false,
               xlabel = xlabel,
               ylabel = "Probability mass",
               title = title,
               size = (900, 550))
end

plot_grid_mass_distribution(grid, mass; kwargs...) =
    plot_weighted_distribution(grid, mass; kwargs...)

plot_weighted_observation_distribution(values, weights; kwargs...) =
    plot_weighted_distribution(values, weights; kwargs...)

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
    check_unconditional_distribution_masses(d)

    asset_plot = plot_grid_mass_distribution(
        d.assetGrid, d.assetMass;
        xlabel = "Assets",
        title = "Unconditional distribution of assets",
    )
    hours_plot = plot_weighted_observation_distribution(
        d.hours, d.weights;
        xlabel = "Hours worked",
        title = "Unconditional distribution of hours worked",
    )
    consumption_plot = plot_weighted_observation_distribution(
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

function display_path(path::AbstractString)
    cloud_documents = joinpath(homedir(), "Library", "CloudStorage", "Dropbox",
                               "Mac", "Documents")
    if startswith(path, cloud_documents)
        return replace(path, cloud_documents => joinpath("~", "Documents"); count = 1)
    end

    home = homedir()
    if startswith(path, home)
        return replace(path, home => "~"; count = 1)
    end
    return path
end
