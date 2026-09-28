# =============================================================================
# run_history_dependent_tax.jl
#
# Entry point for the history-dependent tax model. Builds a parameter set from
# HD_SETTINGS, solves it, prints the summary, and saves or reloads results.
# Also parses key=value command-line overrides for script use.
#
# Marek Kapicka, 2026
# =============================================================================

using JLD2
using Printf

include("solve_history_dependent_tax.jl")   # defines module HistoryDependentTax
                                            # (loads model_settings.jl internally)

using .HistoryDependentTax

"""
    run_history_dependent_tax(; kwargs...)

Solve the model and print the equilibrium summary. `kwargs` override
HD_SETTINGS. Returns `(; eq, params)`. The history-independent limit
`mu1 = mu2 = 0` with `nS1 = nS2 = 1` is handled by this solver directly.
"""
function run_history_dependent_tax(; kwargs...)
    # collect_distributions is on by default here and off in the calibration
    # drivers and the sweeps, where the per-observation vectors are too large.
    p = make_history_dependent_params(; collect_distributions = true, kwargs...)

    eq = solve_history_dependent_tax(p)

    print_hd_equilibrium_summary(eq, p)

    return (; eq = eq, params = p)
end

"""
    save_hd_result(result; dir = joinpath(@__DIR__, "results"))

Save a result to `dir` under a name built from the model dimensions. Overwrites
an existing file of the same name and returns the saved path. Reload with
`load_hd_result` after including this file, so `HDParams` is defined.
"""
function save_hd_result(result; dir = joinpath(@__DIR__, "results"))
    mkpath(dir)
    p = result.params
    name = @sprintf(
        "result_mu1=%.3f_mu2=%.3f_J=%d_nA=%d_nS1=%d_nS2=%d_nZ=%d_nEps=%d_nKappa=%d.jld2",
        p.mu1, p.mu2, p.J, length(p.a_grid),
        length(p.s1_grid), length(p.s2_grid),
        length(p.z_grid), length(p.eps_grid), length(p.kappa_grid))
    path = joinpath(dir, name)
    jldsave(path; eq = result.eq, params = p)
    return path
end

"""
    load_hd_result(path)

Load a result saved by `save_hd_result`. Returns `(; eq, params)`.
"""
function load_hd_result(path::AbstractString)
    data = JLD2.load(path)
    return (; eq = data["eq"], params = data["params"])
end

"""
    parse_cli_value(s)

Convert one command-line token to the type the solver expects: Symbol for a
leading colon, then Bool, Int, Float64, else String. Int is tried before Float64
so that `nA=101` is an Int and `mu2=0.8343` is not.
"""
function parse_cli_value(s::AbstractString)
    startswith(s, ":") && return Symbol(s[2:end])
    s == "true" && return true
    s == "false" && return false
    v = tryparse(Int, s);     v === nothing || return v
    v = tryparse(Float64, s); v === nothing || return v
    return String(s)
end

"""
    parse_cli_kwargs(args)

Turn `["mu1=0", "mu2=0.8343"]` into a NamedTuple for splatting into
`run_history_dependent_tax` or `sweep_mu2`. Unknown keys fail at `HDParams`,
which names the offending key.
"""
function parse_cli_kwargs(args)
    pairs = Pair{Symbol,Any}[]
    for a in args
        i = findfirst('=', a)
        i === nothing &&
            error("expected key=value, got \"$a\" (e.g. mu2=0.8343 s_grid_method=:quantile)")
        push!(pairs, Symbol(strip(a[1:i-1])) => parse_cli_value(strip(a[i+1:end])))
    end
    return (; pairs...)
end

# -----------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# -----------------------------------------------------------------------------
# Every keyword the REPL call accepts works here too:
#   julia -t 8 run_history_dependent_tax.jl mu1=0 mu2=0.8343 nS2=101
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    result = run_history_dependent_tax(; parse_cli_kwargs(ARGS)...)
    @printf("saved to %s\n", save_hd_result(result))
end
