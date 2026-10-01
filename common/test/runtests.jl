# =============================================================================
# runtests.jl
#
# Unit tests for BewleyCommon. Run with
#
#     julia +1.12.6 --project=common -e 'using Pkg; Pkg.test()'
#
# or directly, which is faster while writing them:
#
#     julia +1.12.6 --project=common common/test/runtests.jl
#
# WHAT THESE ARE FOR, and why verify/ does not replace them. The golden master
# in verify/ compares a whole solve against a stored baseline: it detects THAT
# something changed, over eight directories and about two hours for a full
# sweep. It cannot say WHAT broke, and it is blind to any path a harness run
# does not execute. Three real bugs in this package were found the hard way:
#
#   - find_bracket returned (i, i) on an exact-zero residual, handing Roots a
#     zero-width interval. Every caller swallows the throw in a try/catch, so
#     the hd solvers reported convergence FAILURE in the one case where they had
#     found the exact root.
#   - eq_flag was left behind in the solvers when print_lambda_warnings moved
#     here, and the harness ran verbose = false, so the printer never executed.
#   - S_GRID_UNIFORM_BLEND, same shape, found only on a later review.
#
# Each is a one-line assertion here and runs in milliseconds. Tests that pin a
# contract belong in this file; tests that need a full equilibrium belong in
# verify/.
#
# Marek Kapicka, 2026
# =============================================================================

using BewleyCommon
using Test

# Step 10 trimmed BewleyCommon's exports to entry points, so a few functions
# tested here are internal. Reaching in is deliberate: these carry contracts the
# solvers depend on indirectly, and a test that only exercised the public surface
# would not have caught the find_bracket bug.
using BewleyCommon: interpolated_weighted_quantile, eq_flag, labor_foc_residual

# A minimal parameter object. The shared functions are typed on
# AbstractBewleyParams and read only the fields they need, so a test only has to
# supply those -- which is also a check that the abstraction is honest: if a
# shared function reaches for a field not listed here, the test fails loudly
# rather than silently depending on a concrete solver type.
Base.@kwdef struct TestParams <: AbstractBewleyParams
    a_grid::Vector{Float64} = [-1.0, -0.5, 0.0, 1.0, 2.5, 5.0]
    z_grid::Vector{Float64} = [-0.4, 0.0, 0.4]
    qSav::Float64 = 0.99
    qBorr::Float64 = 0.97
    qGov::Float64 = 0.99
    bbar::Float64 = 0.2
    rho::Float64 = 0.958
    tau::Float64 = 0.181
    eta::Float64 = 2.0
    phi::Float64 = 1.0
    hMin::Float64 = 1e-8
    hMax::Float64 = 5.0
    aMax::Float64 = 5.0
    asset_choice_tol::Float64 = 1e-10
    # Reached by the labor path. :brent and :hybrid_newton need only the bounds;
    # :grid additionally reads the three h_grid vectors, which is why those are
    # built lazily in the solvers.
    labor_solver::Symbol = :hybrid_newton
    eps_grid::Vector{Float64} = [-0.3, 0.0, 0.3]
end

@testset "BewleyCommon" begin

# -----------------------------------------------------------------------------
@testset "safe_ratio" begin
    @test safe_ratio(1.0, 4.0) == 0.25
    # A zero denominator is the normal case for a statistic over an empty group,
    # so it must be NaN rather than Inf or a throw: the caller prints it.
    @test isnan(safe_ratio(1.0, 0.0))
    @test isnan(safe_ratio(0.0, 0.0))
    @test isnan(safe_ratio(1.0, eps(Float64) / 2))
    @test safe_ratio(-2.0, 4.0) == -0.5
end

# -----------------------------------------------------------------------------
@testset "find_bracket" begin
    # The ordinary case: a sign change between adjacent nodes.
    @test find_bracket([0.0, 1.0, 2.0], [1.0, -1.0, -2.0]) == (1, 2)
    @test find_bracket([0.0, 1.0, 2.0], [2.0, 1.0, -1.0]) == (2, 3)

    # THE REGRESSION. An exact zero at a node must still return a two-node
    # interval. (i, i) is a zero-width bracket and Roots.Brent throws
    # ArgumentError("Need extrema to return two distinct values") on it.
    for res in ([0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0])
        b = find_bracket([0.0, 1.0, 2.0], res)
        if b !== nothing
            @test b[1] != b[2]
            @test b[2] == b[1] + 1
        end
    end
    @test find_bracket([0.0, 1.0, 2.0], [1.0, 0.0, 2.0]) == (1, 2)

    # No sign change anywhere, and non-finite residuals, are reported as
    # `nothing` rather than guessed at.
    @test find_bracket([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]) === nothing
    @test find_bracket([0.0, 1.0, 2.0], [NaN, NaN, NaN]) === nothing
    # A non-finite residual cannot bracket, and the pair beyond it is only a
    # bracket if it changes sign on its own.
    @test find_bracket([0.0, 1.0, 2.0], [Inf, -1.0, -2.0]) === nothing
    @test find_bracket([0.0, 1.0, 2.0], [Inf, 1.0, -2.0]) == (2, 3)
end

# -----------------------------------------------------------------------------
@testset "interpolated_weighted_quantile" begin
    grid = collect(0.0:0.01:1.0)                     # 101 points on [0,1]
    mass = fill(1 / 101, 101)
    # Mid-cumulative interpolation puts F_k = (k - 1/2)/n, so equal mass on a
    # symmetric grid returns the centre EXACTLY. This is the property the
    # StatsBase convention lacks and the reason this function exists.
    @test interpolated_weighted_quantile(grid, mass, 0.5) ≈ 0.5 atol=1e-14

    # Degenerate inputs.
    @test interpolated_weighted_quantile([3.0], [1.0], 0.5) == 3.0
    @test isnan(interpolated_weighted_quantile(grid, zeros(101), 0.5))
    @test interpolated_weighted_quantile(grid, mass, 0.0) ≈ grid[1] atol=1e-12
    @test interpolated_weighted_quantile(grid, mass, 1.0) ≈ grid[end] atol=1e-12

    # Monotone in prob, and inside the support.
    qs = [interpolated_weighted_quantile(grid, mass, p) for p in 0.0:0.1:1.0]
    @test issorted(qs)
    @test all(grid[1] .<= qs .<= grid[end])

    # Zero-mass points are skipped, not treated as support.
    @test interpolated_weighted_quantile([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 0.5) == 1.0

    # A known asymmetric case: all mass at one point.
    @test interpolated_weighted_quantile([0.0, 1.0, 2.0], [0.0, 0.0, 1.0], 0.5) == 2.0

    # Contract violations are errors, not silent wrong answers.
    @test_throws Exception interpolated_weighted_quantile([0.0, 1.0], [1.0], 0.5)
    @test_throws Exception interpolated_weighted_quantile(grid, mass, 1.5)
end

# -----------------------------------------------------------------------------
@testset "grid_lookup_weights" begin
    grid = [0.0, 1.0, 3.0]
    # The contract is that (1-w) on lo plus w on hi reproduces x.
    for x in (0.25, 1.0, 2.0, 2.999)
        lo, hi, w = grid_lookup_weights(grid, x)
        @test grid[lo] * (1 - w) + grid[hi] * w ≈ x atol=1e-12
        @test 0.0 <= w <= 1.0
    end
    # Off the ends clamps to a single node with no weight, so mass is never
    # placed outside the grid.
    @test grid_lookup_weights(grid, -5.0) == (1, 1, 0.0)
    @test grid_lookup_weights(grid, 99.0) == (3, 3, 0.0)
    @test grid_lookup_weights([2.0], 7.0) == (1, 1, 0.0)
end

# -----------------------------------------------------------------------------
@testset "asset_transition_weights" begin
    p = TestParams()
    g = p.a_grid
    # Same reproduce-x contract, and the weights are a distribution.
    for ap in (-0.75, -0.25, 0.5, 2.0, 4.0)
        lo, hi, w = asset_transition_weights(ap, p)
        @test g[lo] * (1 - w) + g[hi] * w ≈ ap atol=1e-12
        @test 0.0 <= w <= 1.0
    end
    # Landing on a node goes to that node alone rather than splitting with a
    # neighbour at weight 0 or 1.
    for (k, node) in enumerate(g)
        lo, hi, w = asset_transition_weights(node, p)
        @test lo == hi == k
        @test w == 0.0
    end
    # Off the ends clamps, so the forward pass cannot lose mass off the grid.
    @test asset_transition_weights(-99.0, p) == (1, 1, 0.0)
    @test asset_transition_weights(99.0, p) == (length(g), length(g), 0.0)
end

# -----------------------------------------------------------------------------
@testset "first_nonnegative_asset_index" begin
    @test first_nonnegative_asset_index(TestParams()) == 3          # a_grid[3] == 0.0
    @test first_nonnegative_asset_index(TestParams(a_grid = [0.5, 1.0])) == 1
    # An all-negative grid cannot support a zero-asset state, which is a
    # configuration error rather than something to return an index for.
    @test_throws Exception first_nonnegative_asset_index(TestParams(a_grid = [-2.0, -1.0]))
end

# -----------------------------------------------------------------------------
@testset "discounted_sum and the infinite tail" begin
    # Against the closed form of a geometric series.
    q = 0.96
    x = ones(10)
    @test discounted_sum(x, q) ≈ (1 - q^10) / (1 - q) atol=1e-12
    @test discounted_sum(Float64[], q) == 0.0
    @test discounted_sum([2.0, 3.0], 0.5) ≈ 2.0 + 1.5 atol=1e-14

    # The tail continues the LAST value forever, so a constant sequence gives
    # exactly the undiscounted infinite sum.
    @test discounted_sum_with_tail(ones(10), q) ≈ 1 / (1 - q) atol=1e-12
    @test discounted_sum_with_tail(ones(1), q) ≈ 1 / (1 - q) atol=1e-12
    @test discounted_sum_with_tail(Float64[], q) == 0.0
    # q = 1 has no finite tail and must be refused rather than return Inf.
    @test_throws Exception discounted_sum_with_tail(ones(3), 1.0)
end

# -----------------------------------------------------------------------------
@testset "normalize_probabilities" begin
    # NOTE: this mutates its argument in place and also returns it.
    v = [1.0, 1.0, 2.0]
    out = normalize_probabilities(v, "test")
    @test sum(out) ≈ 1.0 atol=1e-15
    @test out === v                       # in-place, by design
    @test out ≈ [0.25, 0.25, 0.5]
    # A tiny negative from rounding is tolerated; a real one is an error.
    @test sum(normalize_probabilities([1.0, -1e-15], "test")) ≈ 1.0 atol=1e-12
    @test_throws Exception normalize_probabilities([1.0, -0.5], "test")
    @test_throws Exception normalize_probabilities([0.0, 0.0], "test")
end

# -----------------------------------------------------------------------------
@testset "normal_cdf" begin
    @test normal_cdf(0.0) ≈ 0.5 atol=1e-12
    @test normal_cdf(-1.0) + normal_cdf(1.0) ≈ 1.0 atol=1e-12   # symmetry
    @test normal_cdf(-8.0) >= 0.0
    @test normal_cdf(8.0) <= 1.0
    @test normal_cdf(1.96) ≈ 0.975 atol=1e-3
    # Monotone.
    xs = -3.0:0.25:3.0
    @test issorted([normal_cdf(x) for x in xs])
end

# -----------------------------------------------------------------------------
@testset "prices and bounds" begin
    p = TestParams()
    # The borrowing price applies strictly below zero; zero itself saves.
    @test asset_price(-0.1, p) == p.qBorr
    @test asset_price(0.0, p) == p.qSav
    @test asset_price(1.0, p) == p.qSav
    @test upper_bound_level_tol(1.0) == 1e-8
    @test upper_bound_level_tol(100.0) ≈ 1e-6      # scales with the bound
    @test upper_bound_level_tol(0.01) == 1e-8      # but never below absolute
    # The borrowing limit scales with the fixed effect and the persistent state.
    @test borrowing_limit(0.0, 2, p) ≈ p.bbar * exp(p.rho * p.z_grid[2])
    @test borrowing_limit(1.0, 2, p) > borrowing_limit(0.0, 2, p)
end

# -----------------------------------------------------------------------------
@testset "labor FOC" begin
    p = TestParams()
    # The solved hours must actually zero the residual the solver is built on,
    # and must respect the bounds. This is the pairing that matters: a root
    # finder that returns a number outside [hMin, hMax], or one the residual
    # does not vanish at, is wrong however plausible the number looks.
    for cash in (0.1, 0.5, 2.0), coeff in (0.5, 1.0, 2.0)
        v, h = optimal_labor_foc(cash, coeff, p)
        if isfinite(v) && !isnan(h)
            @test p.hMin <= h <= p.hMax
            @test abs(labor_foc_residual(h, cash, coeff, p)) < 1e-6
        end
    end
    # No income and no cash is infeasible, and says so rather than returning a
    # number that would win a maximization.
    v, _ = optimal_labor_foc(-1.0, 0.0, p)
    @test v == -Inf
end

# -----------------------------------------------------------------------------
@testset "labor grids" begin
    g = uniform_labor_grid(0.1, 2.0, 11)
    @test length(g) == 11
    @test g[1] ≈ 0.1 && g[end] ≈ 2.0
    @test issorted(g)
    @test all(diff(g) .≈ (2.0 - 0.1) / 10)        # uniform, as the name says
end

# -----------------------------------------------------------------------------
@testset "VINFEASIBLE and UPPER_BOUND_SHARE_TOL" begin
    # The sentinel must lose every maximization against any admissible value,
    # and must not be -Inf: an -Inf propagates through the expectation and
    # poisons states that do have feasible choices.
    @test VINFEASIBLE < -1e17
    @test isfinite(VINFEASIBLE)
    @test max(VINFEASIBLE, -1e6) == -1e6
    @test UPPER_BOUND_SHARE_TOL > 0.0
    @test UPPER_BOUND_SHARE_TOL < 1e-6
end

# -----------------------------------------------------------------------------
@testset "core_statistics" begin
    # A hand-built accumulator with numbers chosen so every published field has
    # a value that can be checked by hand rather than against the code.
    Base.@kwdef mutable struct Acc <: AbstractStatsAccumulator
        asset_mass::Vector{Float64} = [1.0, 0.0, 2.0, 1.0, 0.0, 0.0]
        distribution_weights::Vector{Float64} = Float64[]
        hours_values::Vector{Float64} = Float64[]
        consumption_values::Vector{Float64} = Float64[]
        total_mass::Float64 = 4.0
        sum_current_assets::Float64 = 2.0
        sum_labor_income::Float64 = 8.0
        sum_borrowing_limit::Float64 = 1.0
        sum_effective_borrowing_limit::Float64 = 0.5
        borrowing_limit_mass::Float64 = 2.0
        negative_asset_mass::Float64 = 1.0
        zero_asset_mass::Float64 = 2.0
        borrowing_constraint_mass::Float64 = 1.0
        upper_bound_mass::Float64 = 0.0
        hours_upper_bound_mass::Float64 = 0.0
        max_material_next_assets::Float64 = 3.0
        max_material_hours::Float64 = 1.0
    end
    s = core_statistics(Acc(), TestParams())

    @test s.totalMass == 4.0
    @test s.meanAssets == 0.5                       # 2.0 / 4.0
    @test s.meanLaborIncome == 2.0                  # 8.0 / 4.0
    @test s.meanAssetsToMeanLaborIncome == 0.25     # 0.5 / 2.0
    # Both limits are averaged over the ages that HAVE a limit, not over
    # everyone, so the denominator is borrowing_limit_mass and not total_mass.
    @test s.meanBorrowingLimit == 0.5               # 1.0 / 2.0
    @test s.meanEffectiveGridBorrowingLimit == 0.25 # 0.5 / 2.0
    @test s.shareNegativeLiquidAssets == 0.25
    @test s.shareZeroAssets == 0.5
    @test s.shareAtEffectiveBorrowingConstraint == 0.25
    @test s.shareAtAssetUpperBound == 0.0
    # No mass at either bound, so neither binds.
    @test s.assetUpperBoundBinding == false
    @test s.hoursUpperBoundBinding == false
    @test s.upperBoundsBinding == false
    # The median of mass [1,0,2,1,0,0] on [-1,-0.5,0,1,2.5,5] is 0.
    @test s.medianAssets ≈ 0.0 atol=1e-12
    @test haskey(s, :unconditionalDistributions)
    # The MPC block is NOT part of the core: only variants that accumulate it
    # merge it on, and publishing a zero here would be a wrong number.
    @test !haskey(s, :meanMPC)

    # Mass at the upper bound flips the flag.
    s2 = core_statistics(Acc(upper_bound_mass = 1.0), TestParams())
    @test s2.shareAtAssetUpperBound == 0.25
    @test s2.assetUpperBoundBinding == true
    @test s2.upperBoundsBinding == true
end

# -----------------------------------------------------------------------------
@testset "eq_flag" begin
    # Warning flags are attached only on the paths that raise them, so every
    # read has to tolerate the field being absent. This is the function whose
    # absence broke step 5 on a path the harness never ran.
    @test eq_flag((; bracketWarning = true), :bracketWarning) == true
    @test eq_flag((; bracketWarning = false), :bracketWarning) == false
    @test eq_flag((; lambda = 1.0), :bracketWarning) == false
    @test eq_flag((; bracketWarning = 1), :bracketWarning) == false   # not `true`
end

end # BewleyCommon
