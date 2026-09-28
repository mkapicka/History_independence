# =============================================================================
# plot_welfare.jl
#
# Overlays the Bewley (with-savings) mu2 sweep on the no-savings one-root
# welfare curve, so the two models can be read off one axis.
#
#   blue   no savings, one root      100*(W(mu) - W_hi)      <- from ../../sweep_noasset.jl
#   red -- no savings, unrestricted two-root optimum
#   red    Bewley hd sweep           100*(W .- W[1])         <- from join_results.jl
#
# Each curve's optimum is a large dot in that curve's colour, labelled in place
# with the root rather than in the legend.
#
# ---------------------------------------------------------------------------
# RUNNING IT: the two halves live in different environments
# ---------------------------------------------------------------------------
# The no-savings code needs Optim and PyPlot (code/julia/Project.toml); this
# folder needs JLD2 to read the sweep files (Bewley/hd/Project.toml). Neither
# environment has both, so STACK them rather than adding dependencies to
# either -- Julia searches LOAD_PATH in order:
#
#   cd .../code/julia/Bewley/hd
#   JULIA_LOAD_PATH="../..:.:@stdlib" julia -e 'include("plot_welfare.jl"); plot_welfare()'
#
# `../..` is code/julia (Optim, PyPlot), `.` is this folder (JLD2, Plots).
# Nothing is installed and no Project.toml changes.
#
# ---------------------------------------------------------------------------
# THE TWO CURVES DO NOT SHARE A BASELINE
# ---------------------------------------------------------------------------
# The no-savings curve is a gain over HISTORY INDEPENDENCE: it is anchored at
# zero because mu = 0 collapses the one-root kernel to theta = (1,0,0,...) and
# P = 1 exactly, so the benchmark is a genuine model, not a normalization.
#
# The Bewley curve is `100*(W .- W[1])`, a gain over the FIRST POINT OF ITS OWN
# GRID. That coincides with history independence only if the sweep starts at
# mu2 = 0. The meta3 sweeps start at mu2 = 0.4, in which case the green curve is
# anchored at an arbitrary interior point and its height is not comparable with
# the blue one -- only its SHAPE is. `plot_welfare` prints the first mu2 so this
# is visible rather than silent, and warns when it is not zero.
# =============================================================================

isdefined(@__MODULE__, :oneroot_welfare_curve) || include("../../sweep_noasset.jl")

using Printf

# join_results.jl assigns mu2, W and A at top level, so it must be included
# HERE rather than inside plot_welfare: an `include` in function scope does not
# leave those names visible to the caller. It globs `results/` relative to the
# working directory, so run this from the hd folder.
include("join_results.jl")

"""
    plot_welfare(; npoints = 300, tau = nothing, lambda = :opt,
                 mulo = 0.0, muhi = 0.999, save = true)

Draw both welfare curves against the root. Returns
`(; mu_ns, gain_ns, mu2_hd, gain_hd, opt, W_two)`.

`tau` defaults to the wedge the Bewley sweep used (`HD_SETTINGS.tau` is 0.181),
so the no-savings curve is drawn at the same progressivity. Pass a number to
override.

Reads the Bewley sweep through `join_results.jl` in this folder, which globs
`results/sweep_mu2_n=1_*nS2=151*.jld2`. Point that file elsewhere if your
results live in a subdirectory.
"""
function plot_welfare(; npoints::Int = 300, tau = nothing, lambda::Symbol = :opt,
                      mulo = 0.0, muhi = 0.999, save::Bool = true)
    # ---- Bewley sweep: mu2, W, A come from join_results.jl, included above --
    mu2_hd = copy(mu2)
    W_hd   = copy(W)
    isempty(mu2_hd) && error("join_results.jl returned no points; check results/")
    ord    = sortperm(mu2_hd)
    mu2_hd, W_hd = mu2_hd[ord], W_hd[ord]
    gain_hd = 100 .* (W_hd .- W_hd[1])

    # ---- no-savings one-root curve ----------------------------------------
    p  = Params()
    τ  = tau === nothing ? 0.181 : tau         # HD_SETTINGS.tau
    c  = oneroot_welfare_curve(p, range(mulo, muhi, length = npoints);
                               tau = τ, lambda)
    opt = optimal_mu(p; lambda, tau = τ, mulo, muhi)
    w   = oneroot_welfare(p, opt.mu; tau = τ, lambda)
    s   = solveHistDep(p, p.Kmax)
    W_two = Wfun(p, w.tau, s.P)

    # Optimum of the Bewley curve: the best point ON THE GRID, since that curve
    # is 46 solved points rather than a closed form. Its resolution is the mu2
    # spacing, unlike the no-savings optimum which is refined by Brent.
    ihd = argmax(gain_hd)
    mu_hd_opt, gain_hd_opt = mu2_hd[ihd], gain_hd[ihd]

    # ---- figure ------------------------------------------------------------
    figure(figsize = (8, 5.5))
    # Drawn FIRST so it heads the legend: matplotlib orders entries by the
    # order artists are added, and the unrestricted optimum is the ceiling the
    # other two curves are read against. Grey rather than red, since the Bewley
    # line is red and two red objects meaning different things would confuse.
    axhline(100 * (W_two - w.W_hi); color = MyDarkGrey, linestyle = "--",
            linewidth = 1.3, zorder = 1,
            label = "no savings, unrestricted (two roots)")
    plot(c.mu, 100 .* c.gain; color = MyBlue, linewidth = 1.8, zorder = 2,
         label = "no savings, one root")
    plot(mu2_hd, gain_hd; color = MyRed, linewidth = 1.8, zorder = 2,
         label = "Bewley (savings)")
    axhline(0.0; color = MyDarkGrey, linewidth = 0.8, zorder = 0)

    # Both optima: a large dot in the colour of its own line, labelled in place
    # with the root. "_nolegend_" keeps them out of the legend, which otherwise
    # carries five entries for three curves.
    for (mu_o, g_o, col) in ((opt.mu, 100 * w.gain, MyBlue),
                             (mu_hd_opt, gain_hd_opt, MyRed))
        plot([mu_o], [g_o]; color = col, marker = "o", markersize = 11,
             linestyle = "none", zorder = 5, label = "_nolegend_")
        annotate(@sprintf("\$\\mu\$ = %.4f", mu_o), xy = (mu_o, g_o),
                 xytext = (mu_o - 0.015, g_o + 0.11), color = col,
                 fontsize = "medium", ha = "right", zorder = 6)
    end

    # Legend in the north-west, in the empty band between the dashed two-root
    # reference (2.98%) and the blue curve, which is still below 1.8% over the
    # left half of the axis. A small headroom keeps the box clear of the dashed
    # line; `figstyle` has no anchor argument, so the legend is placed here
    # rather than through it.
    ylo, yhi = ylim()
    ylim(ylo, yhi + 0.08 * (yhi - ylo))
    figstyle(xlab = L"root $\mu$", ylab = "welfare gain (%)", showlegend = false)
    legend(framealpha = 1, loc = "upper left", bbox_to_anchor = (0.02, 0.88))
    title(@sprintf("History dependence with and without savings (τ = %.4f)", τ),
          weight = "bold")
    tight_layout()

    if save
        mkpath(joinpath(@__DIR__, "figures"))
        savefig(joinpath(@__DIR__, "figures", "welfare_vs_mu.pdf"))
    end

    @printf("no-savings one root : optimum mu = %.4f, gain %.4f%%\n", opt.mu, 100 * w.gain)
    @printf("Bewley sweep        : %d points, mu2 in [%.4f, %.4f]\n",
            length(mu2_hd), mu2_hd[1], mu2_hd[end])
    @printf("                      gain range %.4f%% to %.4f%%\n",
            minimum(gain_hd), maximum(gain_hd))
    @printf("Bewley optimum      : mu2 = %.4f, gain %.4f%% (best GRID point)\n",
            mu_hd_opt, gain_hd_opt)
    if abs(mu2_hd[1]) > 1e-12
        @printf("NOTE: the Bewley curve is normalized at mu2 = %.4f, NOT at history\n",
                mu2_hd[1])
        @printf("      independence, so its LEVEL is not comparable with the blue curve.\n")
    end

    return (; mu_ns = c.mu, gain_ns = 100 .* c.gain,
            mu2_hd, gain_hd, opt, W_two,
            mu_hd_opt, gain_hd_opt)
end
