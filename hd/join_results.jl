# =============================================================================
# join_results.jl
#
# Merges the per-point sweep files written by sweep_mu2.jl and concatenates the
# series of interest. The hd_htm copy is the full version, with guards.
#
# Marek Kapicka, 2026
# =============================================================================

using JLD2

  # Merge the per-point sweep files.
  files = sort(filter(f -> occursin("nS2=151", f) && startswith(f, "sweep_mu2_n=1_"),
                      readdir("results")))
  s   = [JLD2.load(joinpath("results", f))["sweep"] for f in files]

  # Concatenate the series of interest.
  mu2 = vcat((x.mu2 for x in s)...)
  W   = vcat((x.overallValueFunction for x in s)...)
  A   = vcat((x.meanAssets for x in s)...)

  # Every point must have converged.
  all(vcat((x.converged for x in s)...))       # must be true
