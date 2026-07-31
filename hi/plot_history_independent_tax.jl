using Plots
using Printf
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

function plot_asset_grid(p)
    p = params_from(p)
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

function age_grid_for_series(result, series)
    p = params_from(result)
    if hasproperty(p, :J) && length(series) == p.J + 1
        return collect(0:p.J)
    end
    return collect(0:(length(series) - 1))
end

# One row per age-profile panel. Add a row to add a panel: it is picked up by
# the plotting, saving, and path-reporting code below.
const AGE_FIGURES = (
    (key = :averageAssetsByAge, field = :A,
     ylabel = "Average assets", title = "Average assets by age",
     file = "average_assets_by_age.png"),
    (key = :averageHoursByAge, field = :H,
     ylabel = "Average hours worked", title = "Average hours worked by age",
     file = "average_hours_by_age.png"),
)

function plot_by_age(result, spec)
    eq = equilibrium_from(result)
    hasproperty(eq, spec.field) ||
        error("equilibrium object must contain the aggregate `$(spec.field)`")
    series = getproperty(eq, spec.field)
    ages = age_grid_for_series(result, series)
    length(ages) == length(series) ||
        error("age grid length does not match eq.$(spec.field)")
    return plot(ages, series;
                marker = :circle,
                linewidth = 2,
                legend = false,
                xlabel = "Age",
                ylabel = spec.ylabel,
                title = spec.title,
                size = (900, 550))
end

function age_figure_spec(field::Symbol)
    idx = findfirst(s -> s.field === field, AGE_FIGURES)
    idx === nothing && error("no age figure defined for eq.$field")
    return AGE_FIGURES[idx]
end

plot_average_assets_by_age(result) = plot_by_age(result, age_figure_spec(:A))
plot_average_hours_by_age(result) = plot_by_age(result, age_figure_spec(:H))

function save_age_figure(result, field::Symbol;
                         output_dir = joinpath(@__DIR__, "figures"))
    spec = age_figure_spec(field)
    mkpath(output_dir)
    path = joinpath(output_dir, spec.file)
    savefig(plot_by_age(result, spec), path)
    return path
end

save_age_figures(result; output_dir = joinpath(@__DIR__, "figures")) =
    (; (spec.key => save_age_figure(result, spec.field; output_dir = output_dir)
        for spec in AGE_FIGURES)...)

save_average_assets_by_age_figure(result; output_dir = joinpath(@__DIR__, "figures")) =
    save_age_figure(result, :A; output_dir = output_dir)
save_average_hours_by_age_figure(result; output_dir = joinpath(@__DIR__, "figures")) =
    save_age_figure(result, :H; output_dir = output_dir)

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
    eq = equilibrium_from(eq)
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
    hours_plot = plot_weighted_distribution(
        d.hours, d.weights;
        xlabel = "Hours worked",
        title = "Unconditional distribution of hours worked",
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

function save_history_independent_figures(result; output_dir = joinpath(@__DIR__, "figures"))
    return merge(
        (; assetGrid = save_asset_grid_figure(result; output_dir = output_dir)),
        save_age_figures(result; output_dir = output_dir),
        save_unconditional_distribution_figures(result; output_dir = output_dir),
    )
end

# Label for each entry of `save_history_independent_figures`, in print order.
const FIGURE_LABELS = (;
    assetGrid          = "asset grid",
    averageAssetsByAge = "average assets by age",
    averageHoursByAge  = "average hours by age",
    assets             = "assets",
    hours              = "hours worked",
    consumption        = "consumption",
)

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

"""
    plot_history_independent_tax(result; output_dir = joinpath(@__DIR__, "figures"),
                                 print_paths = true)

Save the standard history-independent-tax figures from either a raw equilibrium
`eq` or a result object returned by `run_history_independent_tax()` or
`calibrate_history_independent_tax()`.

Returns a NamedTuple `(; assetGrid, assets, hours, consumption)` with the saved
figure paths.
"""
function plot_history_independent_tax(result; output_dir = joinpath(@__DIR__, "figures"),
                                      print_paths::Bool = true)
    figures = save_history_independent_figures(result; output_dir = output_dir)

    if print_paths
        println("\n=== Figures saved ===")
        for (key, label) in pairs(FIGURE_LABELS)
            @printf("%-25s = %s\n", label, display_path(getproperty(figures, key)))
        end
    end

    return figures
end
