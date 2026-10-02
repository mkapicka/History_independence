# =============================================================================
# main.jl
#
# Entry point: builds a parameter set from the settings, solves it,
# prints the summary, and saves or reloads results.
# For the infinite-horizon history-dependent tax model.
#
# Marek Kapicka, 2026
# =============================================================================

using JLD2
using Printf

include("solve.jl")   # defines module HistoryDependentTax
                                            # (loads model_settings.jl internally)

using .HistoryDependentTaxInfinite

"""
    main_hd(; kwargs...)

Solve the history-dependent tax model and print the equilibrium summary.
`kwargs` override `HD_SETTINGS` (e.g.
`main_hd(nA = 41, J = 4, mu1 = 0.0)`).

Returns a NamedTuple `(; eq, params)` where `eq` is the equilibrium and
`params` is the `HDParams` used. The history-independent limit
`mu1 = mu2 = 0` (with `nS1 = nS2 = 1`) is handled by this solver directly;
`check_history_independent_limit()` runs it with the standard consistency
checks.
"""
function main_hd(; kwargs...)
    # collect_distributions is on by default here and off in the calibration
    # drivers and the sweeps, where the per-observation vectors are too large.
    p = make_history_dependent_params(; collect_distributions = true, kwargs...)

    eq = solve_hd(p)

    print_hd_equilibrium_summary(eq, p)

    return (; eq = eq, params = p)
end

"""
    save_hd_result(result; dir = joinpath(@__DIR__, "results"))

Save a `main_hd()` result to `dir` under a filename built
from the model dimensions (shell-safe: no spaces or commas), e.g.
`result_mu1=0.300_mu2=0.050_J=39_nA=101_nS1=7_nS2=7_nZ=5_nEps=5_nKappa=3.jld2`.
Overwrites an existing file of the same name. Returns the saved path.
Reload with `load_hd_result(path)` (after including this file, so the
`HistoryDependentTax` module needed to reconstruct `HDParams` is defined).
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

Convert one command-line token to the type the solver expects. A leading colon
gives a `Symbol` (`:quantile`), `true`/`false` a `Bool`, then `Int` is tried
before `Float64`, and anything left over stays a `String`. Int before Float
matters: `nA=101` must be an `Int`, while `mu2=0.8343` must not be.
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

Turn `["mu1=0", "mu2=0.8343", "s_grid_method=:quantile"]` into a NamedTuple
suitable for splatting into `main_hd` or `sweep_mu2`, so a
command line can carry any override the REPL call can. Unknown keys are not
checked here -- they fail at `HDParams` with an unsupported-keyword error, which
names the offending key.
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
#   julia -t 8 main.jl mu1=0 mu2=0.8343 nS2=101
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    result = main_hd(; parse_cli_kwargs(ARGS)...)
    @printf("saved to %s\n", save_hd_result(result))
end
