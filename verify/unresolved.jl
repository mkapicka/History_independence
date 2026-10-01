# =============================================================================
# unresolved.jl
#
# Checks that every name the package calls resolves inside the package.
#
# Code lifted from a solver into BewleyCommon keeps whatever global names it
# referenced, but those names now resolve in the package's scope, not the
# solver's. A helper or constant left behind in the solver then fails only
# when the moved code runs: `eq_flag` after step 5 broke every lambda warning,
# and `S_GRID_UNIFORM_BLEND` after step 9 broke every `:quantile` s-grid. The
# golden harness cannot see either, because both sit on paths it does not
# exercise. This script can: it parses every file in common/src, collects each
# name in call position plus every ALL_CAPS identifier, and reports those that
# are not defined in BewleyCommon or Base. Names defined inside a function body
# (closures such as `F(x) = ...`) are excluded.
#
#   julia --project=<any solver dir> verify/unresolved.jl
#
# Exits nonzero if anything is unresolved, so it can gate a commit.
#
# Marek Kapicka, 2026
# =============================================================================

using BewleyCommon

const SRC = normpath(joinpath(@__DIR__, "..", "common", "src"))

used = Set{Symbol}()
defined = Set{Symbol}()

# A definition at any nesting: `function f(...)`, `f(...) = ...`, or a
# keyword-argument/where-clause form of either.
function record_definition!(ex)
    if ex isa Expr && ex.head == :function && !isempty(ex.args)
        sig = ex.args[1]
        sig isa Expr && sig.head == :where && (sig = sig.args[1])
        sig isa Expr && sig.head == :call && sig.args[1] isa Symbol && push!(defined, sig.args[1])
    elseif ex isa Expr && ex.head == :(=) && ex.args[1] isa Expr
        sig = ex.args[1]
        sig.head == :where && (sig = sig.args[1])
        sig isa Expr && sig.head == :call && sig.args[1] isa Symbol && push!(defined, sig.args[1])
    end
end

function walk(ex)
    if ex isa Expr
        record_definition!(ex)
        if ex.head in (:call, :macrocall) && ex.args[1] isa Symbol
            push!(used, ex.args[1])
        end
        foreach(walk, ex.args)
    elseif ex isa Symbol
        s = String(ex)
        occursin(r"^[A-Z][A-Z0-9_]+$", s) && length(s) > 2 && push!(used, ex)
    end
end

for f in sort(filter(endswith(".jl"), readdir(SRC)))
    f == "BewleyCommon.jl" && continue
    src = read(joinpath(SRC, f), String)
    pos = 1
    while pos <= lastindex(src)
        ex, pos = Meta.parse(src, pos; raise = false)
        ex === nothing || walk(ex)
    end
end

# Broadcasting operators parse as symbols like `.+`; they are syntax, not names.
syntax(s) = startswith(String(s), ".")
bad = sort([s for s in used
            if !(s in defined) && !syntax(s) &&
               !isdefined(BewleyCommon, s) && !isdefined(Base, s)])

if isempty(bad)
    println("  PASS: every name used in common/src resolves inside BewleyCommon.")
else
    println("  FAIL: ", length(bad), " name(s) used in common/src do not resolve inside BewleyCommon:")
    foreach(s -> println("    ", s), bad)
    exit(1)
end
