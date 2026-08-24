#=
Paired seed-level effects for the v2 RvR grid, from the grid_summary.dat cache
written by exp/_rvr_grid_v2_analysis.jl (no .dat reloading needed).

Claims tested (P1's total cost, matched by seed):
  A. Opponent-robustness externality: P1 at (5, x) - (5, 5) as opponent x
     becomes less robust (25 ... 15625, NR).
  B. Own-robustness value: P1 at (x, 5) - (5, 5) as P1 itself becomes less
     robust, holding a maximally robust opponent fixed.
  C. Prisoner's-dilemma gap: (NR, NR) - (5, 5) for both players.

Run from repo root (after the analysis script):
    julia --project=. exp/_rvr_v2_paired_effects.jl [arm...]
=#

using Statistics
using Printf
using Serialization

const SUMMARY = "./exp/senate/outputs/analysis/rvr_grid_v2/grid_summary.dat"
const LEVELS = Any[5, 25, 125, 625, 3125, 15625, :NR]

summary_all = deserialize(SUMMARY)
arm_sel = isempty(ARGS) ? sort(collect(keys(summary_all))) : ARGS

"Per-seed totals of player `pidx` in cell, as Dict(seed => total)."
function seed_totals(cells, l1, l2, pidx)
    s = cells[(l1, l2)]
    v = pidx == 1 ? s.p1_totals : s.p2_totals
    return Dict(zip(s.seeds, v))
end

"Paired diff b - a matched by seed; returns (mean, std, n, t)."
function paired(a::Dict, b::Dict)
    ks = sort(collect(intersect(keys(a), keys(b))))
    d = [b[k] - a[k] for k in ks]
    d = filter(isfinite, d)
    n = length(d)
    m, s = mean(d), std(d)
    return (mean=m, std=s, n=n, t=m / (s / sqrt(n)))
end

fmt(p) = @sprintf("%+8.3f +- %6.3f  (n=%2d, t=%+7.1f)", p.mean, p.std, p.n, p.t)

for arm in arm_sel
    cells = summary_all[arm]
    println("\n", "="^78)
    println("ARM: $arm  — paired per-seed differences (positive = costlier)")
    println("="^78)

    base = seed_totals(cells, 5, 5, 1)

    println("\nA. P1 cost change when the OPPONENT becomes less robust:  (5, x) - (5, 5)")
    for x in LEVELS[2:end]
        println("   x = $(rpad(x, 6)): ", fmt(paired(base, seed_totals(cells, 5, x, 1))))
    end

    println("\nB. P1 cost change when P1 ITSELF becomes less robust:     (x, 5) - (5, 5)")
    for x in LEVELS[2:end]
        println("   x = $(rpad(x, 6)): ", fmt(paired(base, seed_totals(cells, x, 5, 1))))
    end

    println("\nC. Mutual-nominal vs mutual-robust:  (NR, NR) - (5, 5)")
    println("   P1: ", fmt(paired(base, seed_totals(cells, :NR, :NR, 1))))
    println("   P2: ", fmt(paired(seed_totals(cells, 5, 5, 2), seed_totals(cells, :NR, :NR, 2))))

    println("\nD. Own-robustness value against a NOMINAL opponent:  (x, NR) - (5, NR), P1")
    base_nr = seed_totals(cells, 5, :NR, 1)
    for x in LEVELS[2:end]
        println("   x = $(rpad(x, 6)): ", fmt(paired(base_nr, seed_totals(cells, x, :NR, 1))))
    end
end
println("\nDone.")
