using Printf

include("solve_history_independent_tax.jl")
include("plot_history_independent_tax.jl")
include("model_settings.jl")

function main()
    p = make_history_independent_params()

    eq, _ = solve_history_independent_tax(p)
    asset_grid_path = save_asset_grid_figure(p)
    figure_paths = save_unconditional_distribution_figures(eq)

    print_equilibrium_summary(eq, p)

    @printf("\n=== Figures saved ===\n")
    @printf("asset grid                = %s\n", display_path(asset_grid_path))
    @printf("assets                    = %s\n", display_path(figure_paths.assets))
    @printf("hours worked              = %s\n", display_path(figure_paths.hours))
    @printf("consumption               = %s\n", display_path(figure_paths.consumption))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
