# =============================================================================
# values.jl
#
# Evaluating the continuation value: the expectation over next period's shocks,
# and the per-candidate value used by the grid-search asset choice.
#
# `compute_expected_value!` is shared by all four history-independent solvers;
# `evaluate_policy_grid_search!` by the two infinite-horizon ones;
# `evaluate_block!` by the two infinite-horizon history-dependent ones. Each was
# byte-identical across the directories that have it.
#
# Marek Kapicka, 2026
# =============================================================================

function compute_expected_value!(EV, Vnext, p::AbstractBewleyParams)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    fill!(EV, 0.0)

    @inbounds for ia in 1:nA
        for iz in 1:nZ
            total = 0.0
            for izp in 1:nZ
                pe_z = p.Pz[iz, izp]
                if pe_z == 0.0
                    continue
                end
                eps_total = 0.0
                for iep in 1:nE
                    eps_total += p.Peps[iep] * Vnext[ia, izp, iep]
                end
                total += pe_z * eps_total
            end
            EV[ia, iz] = total
        end
    end
    return EV
end

"""
    evaluate_policy_grid_search!(Vcur, policyAIndex, flow_u, EV, p, util_weight, beta)

Policy evaluation: apply the stored asset choice without searching over it.
This is the cheap half of Howard's method -- one lookup per state instead of a
scan over `ia_first:nA` -- so it costs roughly 1/nA of a maximizing sweep.

Only the grid-search branch has this, because it is the branch whose flow
payoff `flow_u` is precomputed; reconstructing the payoff would cost as much as
re-optimizing and defeat the purpose.
"""
function evaluate_policy_grid_search!(Vcur, policyAIndex, flow_u, EV,
                                      p::AbstractBewleyParams, util_weight::Float64,
                                      beta::Float64)
    nA = length(p.a_grid)
    nZ = length(p.z_grid)
    nE = length(p.eps_grid)
    @inbounds for ia in 1:nA, iz in 1:nZ, ie in 1:nE
        iap = Int(policyAIndex[ia, iz, ie])
        u = flow_u[iap, ia, iz, ie]
        # A state with no feasible choice carries the `ia_first` fallback
        # policy, whose flow payoff is -Inf. Writing that into V would undo the
        # finite sentinel: the next maximizing sweep would see EV = -Inf, fail
        # every comparison, and fall back again -- so the state must keep the
        # same finite value the maximizer gave it.
        Vcur[ia, iz, ie] = isfinite(u) ?
            util_weight * u + beta * EV[iap, iz] : VINFEASIBLE
    end
    return nothing
end

"""
    evaluate_block!(Vcur, policyAIndex, policyH, sc, EVz, cash, ie, is1, is2, iz,
                    m_base, coeff, p)

One policy-evaluation pass over a block: apply the STORED policy and compute the
resulting value, with no maximization. This is the cheap half of Howard's
method -- it replaces a search over the nA*nH choice set with a single lookup
per state, so it costs on the order of 1/(nA*nH) of `solve_block!`.

Mirrors `solve_block!` exactly in how the continuation is formed (bilinear in
(s1', s2') at the chosen a'), so the two are consistent by construction; a
mismatch here would show up as Howard converging to the wrong fixed point.
"""
function evaluate_block!(Vcur, policyAIndex, policyH, EVz, cash,
                         ie::Int, is1::Int, is2::Int, iz::Int,
                         m_base::Float64, coeff::Float64, p::AbstractBewleyParams)
    nA = size(cash, 1)
    beta = p.beta
    util_weight = 1.0 - beta

    @inbounds for ia in 1:nA
        iap = Int(policyAIndex[ia, is1, is2, iz, ie])
        h = policyH[ia, is1, is2, iz, ie]
        c = coeff * h^p.pow + cash[iap, ia]
        # States below the borrowing limit are infeasible and carry no mass;
        # `solve_block!` marks them with the finite sentinel rather than -Inf
        # (which would give 0 * Inf = NaN in the bilinear interpolation), and
        # policy evaluation must use the same convention or Howard and plain
        # VFI would converge to different objects on those states.
        if c <= 0.0
            Vcur[ia, is1, is2, iz, ie] = VINFEASIBLE
            continue
        end
        u = log(c) - p.phi * h^(1.0 + p.eta) / (1.0 + p.eta)

        s1n = p.mu1 * (m_base + log(h) + p.s1_grid[is1])
        s2n = p.mu2 * (m_base + log(h) + p.s2_grid[is2])
        l1, h1, w1 = grid_lookup_weights(p.s1_grid, s1n)
        l2, h2, w2 = grid_lookup_weights(p.s2_grid, s2n)
        ev = (1.0 - w1) * ((1.0 - w2) * EVz[iap, l1, l2] + w2 * EVz[iap, l1, h2]) +
             w1 * ((1.0 - w2) * EVz[iap, h1, l2] + w2 * EVz[iap, h1, h2])

        Vcur[ia, is1, is2, iz, ie] = util_weight * u + beta * ev
    end
    return nothing
end
