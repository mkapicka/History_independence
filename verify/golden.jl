# =============================================================================
# golden.jl
#
# Golden-master harness. Solves one directory's model at small grids and writes
# every scalar of the equilibrium to a flat text file, one `key = value` line
# per scalar, sorted. Comparing two such files proves whether a refactoring
# changed any number the solver produces.
#
# Run it through verify/run.sh rather than directly: each directory has its own
# Julia environment, and the grids differ by model family.
#
#   julia --project=<dir> verify/golden.jl <dir> <out-file>
#
# Values are written with `repr`, which round-trips Float64 exactly, so the
# comparison is bit-for-bit and a plain `diff` is the check.
#
# Marek Kapicka, 2026
# =============================================================================

const DIR = ARGS[1]
const OUT = ARGS[2]

# Trailing `key=value` arguments override the grids below, so a non-default
# configuration can be captured under its own tag. The default asset choice is
# :grid_search, which leaves the :interpolate path uncovered; passing
# asset_choice_method=:interpolate gives it a golden master of its own.
#
# One caveat on the :interpolate tag. That path picks a' by optimizing over a
# continuous segment, so a last-bit change in the objective can move the argmax,
# and in the infinite-horizon solvers the forward pass amplifies it while
# settling on a marginally different fixed point. Measured: making common/ a
# package in cee3167 -- a pure file move plus `include` becoming `using`, with
# no arithmetic touched at all -- moved 78 of hiinf's 130 values by about 1e-8
# relative, because crossing a module boundary changes inlining and therefore
# floating-point contraction. The same run repeated on one build is exactly
# reproducible; only recompilation moves it.
#
# So a diff on this tag after code crosses a module boundary is expected and is
# not by itself a bug. Check the magnitude before concluding anything: 1e-8
# relative is this effect, while a real error shows up in :grid_search too,
# which selects an integer index and is immune. :grid_search has stayed
# bit-identical through every step of the consolidation.
parse_override(s) = startswith(s, ":") ? Symbol(s[2:end]) :
                    s == "true"  ? true :
                    s == "false" ? false :
                    something(tryparse(Int, s), tryparse(Float64, s), String(s))
const OVERRIDES = NamedTuple(
    Symbol(strip(a[1:findfirst('=', a)-1])) => parse_override(strip(a[findfirst('=', a)+1:end]))
    for a in ARGS[3:end] if occursin('=', a))

# Small grids: seconds per solve rather than the ~31s median of a production
# run, while still exercising every code path. The hd family carries the two
# stock dimensions and a coarser hours grid; the inf family needs maxAge.
const GRIDS = Dict(
    "hi"        => (; J = 39, nA = 41, nZ = 3, nEps = 3, nKappa = 3),
    "hi_htm"    => (; J = 39, nA = 41, nZ = 3, nEps = 3, nKappa = 3),
    "hiinf"     => (; nA = 41, nZ = 3, nEps = 3, nKappa = 3, maxAge = 200),
    "hiinf_htm" => (; nA = 41, nZ = 3, nEps = 3, nKappa = 3, maxAge = 200),
    "hd"        => (; J = 39, nA = 41, nZ = 3, nEps = 3, nKappa = 3,
                      nS1 = 3, nS2 = 3, labor_grid_size = 41),
    "hd_htm"    => (; J = 39, nA = 41, nZ = 3, nEps = 3, nKappa = 3,
                      nS1 = 3, nS2 = 3, labor_grid_size = 41),
    "hdinf"     => (; nA = 41, nZ = 3, nEps = 3, nKappa = 3,
                      nS1 = 3, nS2 = 3, labor_grid_size = 41, maxAge = 200),
    "hdinf_htm" => (; nA = 41, nZ = 3, nEps = 3, nKappa = 3,
                      nS1 = 3, nS2 = 3, labor_grid_size = 41, maxAge = 200),
)

haskey(GRIDS, DIR) || error("unknown directory $(DIR); add its grids to GRIDS")

# Flatten anything the equilibrium carries into `key = value` lines. Arrays are
# reduced to length and a checksum rather than written out: the point is to
# detect change, and a full asset distribution would swamp the file.
function emit!(lines, prefix, x)
    if x isa Number || x isa Bool
        push!(lines, "$(prefix) = $(repr(x))")
    elseif x isa Symbol || x isa AbstractString
        push!(lines, "$(prefix) = $(repr(String(x)))")
    elseif x isa AbstractArray && eltype(x) <: Number
        push!(lines, "$(prefix).length = $(length(x))")
        push!(lines, "$(prefix).sum = $(repr(sum(Float64.(x))))")
        isempty(x) || push!(lines, "$(prefix).first = $(repr(Float64(first(x))))")
        isempty(x) || push!(lines, "$(prefix).last = $(repr(Float64(last(x))))")
    elseif x isa NamedTuple
        for k in keys(x)
            emit!(lines, "$(prefix).$(k)", getproperty(x, k))
        end
    end
    return lines
end

# Every directory's entry point is main.jl, so there is nothing to select here.
# The entry FUNCTION still carries a family tag, because the hd solver is a
# module that exports its name and is meant to be loadable beside an hi one.
include(joinpath(@__DIR__, "..", DIR, "main.jl"))
const SOLVE = startswith(DIR, "hd") ? main_hd : main_hi

# One verbose solve with stdout captured: its return value gives the values
# and the capture gives the printed summary. The printed form is checked
# separately because the value file cannot see formatting; `verbose` only
# controls printing, so one solve serves both. Volatile lines (timings) are
# filtered out.
const PRINTED = tempname()
r = open(PRINTED, "w") do io
    redirect_stdout(io) do
        SOLVE(; GRIDS[DIR]..., verbose = true, collect_distributions = true, OVERRIDES...)
    end
end

let
    volatile = r"solve time|elapsed|seconds|Precompiling|precompiled"
    lines = filter(l -> !occursin(volatile, l), readlines(PRINTED))
    rm(PRINTED; force = true)
    mkpath(dirname(OUT))
    open(replace(OUT, ".txt" => ".printed"), "w") do io
        for l in lines
            println(io, rstrip(l))
        end
    end
end

lines = String[]
emit!(lines, "lambda", r.eq.lambda)
emit!(lines, "govBudgetResidual", r.eq.govBudgetResidual)
for field in (:statistics, :statisticsAllAges, :welfare)
    hasproperty(r.eq, field) && emit!(lines, String(field), getproperty(r.eq, field))
end
for field in (:C, :H, :Y, :A)
    hasproperty(r.eq, field) && emit!(lines, String(field), getproperty(r.eq, field))
end

sort!(lines)
mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, "# golden master: ", DIR, isempty(OVERRIDES) ? "" : "  " * string(OVERRIDES))
    for l in lines
        println(io, l)
    end
end
println("  ", DIR, ": ", length(lines), " values -> ", OUT)
