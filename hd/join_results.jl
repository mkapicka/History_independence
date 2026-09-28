using JLD2

  # merge files
  files = sort(filter(f -> occursin("nS2=151", f) && startswith(f, "sweep_mu2_n=1_"),
                      readdir("results")))
  s   = [JLD2.load(joinpath("results", f))["sweep"] for f in files]

  # concatenate variables of interest
  mu2 = vcat((x.mu2 for x in s)...)
  W   = vcat((x.overallValueFunction for x in s)...)
  A   = vcat((x.meanAssets for x in s)...)

  # check if all converged 
  all(vcat((x.converged for x in s)...))       # must be true
