# =============================================================================
# access.jl
#
# The exogenous two-state asset-market-access chain carried by the *_htm
# variants: an agent either has access to the asset market (S) or is
# hand-to-mouth (H), independently of everything else in the model.
#
# Marek Kapicka, 2026
# =============================================================================

"""
    access_stationary_distribution(pSS, pHH)

Stationary shares `(piS, piH)` of the two-state asset-market-access chain

    Pr(S'=S | S) = pSS,   Pr(H'=H | H) = pHH,

which is `(1-pHH, 1-pSS) / (2 - pSS - pHH)`. The initial cross-section is drawn
from this distribution (psmodel.tex), so the hand-to-mouth share is constant
over the life cycle rather than drifting towards it.

Three parameterizations are nested. `pSS = 1, pHH = 0` makes everyone a saver,
which reproduces the corresponding no-access-chain solver exactly (`hi_htm`
reduces to `hi`, `hiinf_htm` to `hiinf`, `hd_htm` to `hd`, `hdinf_htm` to
`hdinf`); `pSS = 0, pHH = 1` makes everyone hand-to-mouth; `pHH = 1 - pSS`
makes access iid with `piH = pHH`.
"""
function access_stationary_distribution(pSS::Real, pHH::Real)
    denom = 2.0 - Float64(pSS) - Float64(pHH)
    denom > 0.0 || error("pSS = $pSS and pHH = $pHH make the access chain " *
                         "reducible (2 - pSS - pHH = $denom); the stationary " *
                         "distribution is not unique")
    return ((1.0 - Float64(pHH)) / denom, (1.0 - Float64(pSS)) / denom)
end
