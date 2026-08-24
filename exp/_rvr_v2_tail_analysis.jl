#=
Tail-risk view of the v2 RvR grid: does robustness compress the BAD-seed tail
of executed cost, beyond what the mean shows?

For P1 at own-level a vs own-level b (same opponent, same seeds), reports both
cost distributions (mean/std/p90/max) and the per-seed paired difference
Δ = cost(b) - cost(a) (positive = level a, the more robust one, is cheaper):
mean, p10/p50/p90 of Δ, fraction of seeds where a is cheaper, and the
correlation of Δ with cost(b) (positive corr = benefit concentrated on seeds
that are bad for the less-robust player, i.e. insurance-like).

Run from repo root (after exp/_rvr_grid_v2_analysis.jl):
    julia --project=. exp/_rvr_v2_tail_analysis.jl
=#

using Statistics
using Printf
using Serialization

const SUMMARY = "./exp/senate/outputs/analysis/rvr_grid_v2/grid_summary.dat"
summary_all = deserialize(SUMMARY)

seed_totals(cells, l1, l2) = Dict(zip(cells[(l1, l2)].seeds, cells[(l1, l2)].p1_totals))

function dist_line(label, v)
    @printf("    %-14s mean=%8.3f  std=%6.3f  p90=%8.3f  max=%8.3f\n",
        label, mean(v), std(v), quantile(v, 0.9), maximum(v))
end

function compare(cells; robust_lvl, nominal_lvl, opp, label)
    a = seed_totals(cells, robust_lvl, opp)    # more robust self
    b = seed_totals(cells, nominal_lvl, opp)   # less robust self
    ks = sort(collect(intersect(keys(a), keys(b))))
    av = [a[k] for k in ks]; bv = [b[k] for k in ks]
    d = bv .- av                                # >0: robust cheaper on that seed
    println("\n  ", label)
    dist_line("self=$(robust_lvl):", av)
    dist_line("self=$(nominal_lvl):", bv)
    @printf("    Δ (%s - %s): mean=%+.3f  p10=%+.3f  p50=%+.3f  p90=%+.3f  robust-cheaper=%d/%d  cor(Δ, cost@%s)=%+.2f\n",
        string(nominal_lvl), string(robust_lvl),
        mean(d), quantile(d, 0.1), quantile(d, 0.5), quantile(d, 0.9),
        count(>(0), d), length(d), string(nominal_lvl), cor(d, bv))
end

for arm in sort(collect(keys(summary_all)))
    cells = summary_all[arm]
    println("\n", "="^90)
    println("ARM: $arm — P1 executed total cost, tail view (25 seeds)")
    println("="^90)
    compare(cells; robust_lvl=5,   nominal_lvl=:NR, opp=:NR, label="max-robust vs nominal self, NOMINAL opponent   [(5,NR) vs (NR,NR)]")
    compare(cells; robust_lvl=5,   nominal_lvl=:NR, opp=5,   label="max-robust vs nominal self, ROBUST opponent    [(5,5) vs (NR,5)]")
    compare(cells; robust_lvl=5,   nominal_lvl=625, opp=5,   label="max-robust vs weak-robust self, ROBUST opponent [(5,5) vs (625,5)]")
end
println("\nDone.")
