#=
Reframe the hedge in recovery terms: of the excess cost the belief-drift
misread inflicts on P1, what fraction does the hedge recover?

  damage(d)   = P1cost[cNR(15625), d] - P1cost[cNR(15625), 0]   (misread's cost
                with hedging off, structure held fixed; paired per seed)
  hedge(c,d)  = P1cost[cNR(15625), d] - P1cost[cNR(c), d]
  recovery    = hedge / damage  (in %)

Run from repo root:  julia --project=. exp/_rvr_hedge_recovery.jl
=#

using Statistics
using Printf
using Serialization

const CS = [5, 25, 125]
const C_INERT = 15625
const DS = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]
const GRID_CACHE = "./exp/senate/outputs/analysis/rvr_p1drift/p1drift_cache.dat"
grid = deserialize(GRID_CACHE)

function paired(a, b)
    ks = intersect(keys(a), keys(b))
    vals = [a[k].p1 - b[k].p1 for k in ks]
    (mean=mean(vals), sem=std(vals) / sqrt(length(vals)), n=length(vals))
end

base0 = grid[(C_INERT, 0.0, :cNR)]
println("Misread damage and hedge recovery (P1 cost, inert-robust arm, paired per seed)")
@printf("%6s | %12s |", "d", "damage")
for c in CS
    @printf(" %20s |", "hedge c=$c (recov)")
end
println()
for d in DS
    dmg = paired(grid[(C_INERT, d, :cNR)], base0)
    @printf("%+6.2f | %+9.4f%s |", d, dmg.mean, abs(dmg.mean) > 1.96 * dmg.sem ? "*" : " ")
    for c in CS
        h = paired(grid[(C_INERT, d, :cNR)], grid[(c, d, :cNR)])
        rec = dmg.mean > 0 ? 100 * h.mean / dmg.mean : NaN
        @printf(" %+9.4f%s (%5.1f%%) |", h.mean, abs(h.mean) > 1.96 * h.sem ? "*" : " ",
            rec)
    end
    println()
end
println("\ndamage = cost(cNR(15625), d) - cost(cNR(15625), 0); recov = hedge/damage.")
println("Negative-d rows: damage may be ~0 or negative (misread not costly there).")
