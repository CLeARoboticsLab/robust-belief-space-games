#=
Calibrated-noise pilot analysis (option 2, ~5 seeds): real sensor gain
g = gt_drift_sensor_scale in {1, 2, 4}, execution EKF calibrated (post-fix),
planner self-beliefs calibrated, players blind to the opponent's gain.
Cells per gain: (5,5), (NR,5) [P2 robust], (NR,NR).

Reports per gain:
  A. P1 own effect  = P1@(NR,5) - P1@(5,5)    (>0: robustness cheaper for P1)
  B. P2 own effect  = P2@(NR,NR) - P2@(NR,5)  (>0: robustness cheaper for P2)
  C. Externality    = P1@(NR,5) - P1@(NR,NR)  (>0: opponent robustness hurts P1)
  D. Mutual         = per-player (NR,NR) - (5,5)
Plus, at g=1.0: paired per-seed comparison against the v2 grid cells
(miscalibrated execution EKF) on the shared seeds — the pure EKF-fix effect.

Run from repo root:
    julia --project=. exp/_rvr_calib_pilot_analysis.jl
=#

using Statistics
using Printf
using Serialization

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files
const STA = SenateTrajectoryAnalysis

const GAINS = [1.0, 2.0, 4.0]
const RUN_ROOT = "./exp/senate/outputs/runs"
const V2_SUMMARY = "./exp/senate/outputs/analysis/rvr_grid_v2/grid_summary.dat"
const BAILOUT_REJ = 26

const CELLS = Dict(
    :RR  => Regex("p1nm_5_p2nm_5_p1oc"),
    :NR5 => Regex("p1t_non_robust_p2nm_5_p1oc"),
    :NN  => Regex("p1t_non_robust_p2t_non_robust"),
)

function load_cell(dir, pat)
    load_and_analyze_senate_solution_files(directory=dir, file_pattern=pat)
    entries = copy(SENATE_TRAJECTORY_TRACKER.entries)
    out = Dict{Int, NamedTuple}()
    for e in entries
        dec1 = STA._decompose_entry_costs(e, 1)
        dec2 = STA._decompose_entry_costs(e, 2)
        rej = Dict(1 => Float64[], 2 => Float64[])
        if e.cost_history isa Dict
            for pidx in (1, 2)
                ch = get(e.cost_history, pidx, nothing)
                (isnothing(ch) || isempty(ch)) && continue
                for c in ch
                    (c isa NamedTuple) || continue
                    (hasproperty(c, :solver_iterations) && hasproperty(c, :solver_improvement_iterations)) || continue
                    (isnothing(c.solver_iterations) || isnothing(c.solver_improvement_iterations)) && continue
                    push!(rej[pidx], Float64(c.solver_iterations - c.solver_improvement_iterations))
                end
            end
        end
        bail(v) = isempty(v) ? NaN : count(>=(BAILOUT_REJ), v) / length(v)
        out[e.random_seed] = (
            p1 = isnothing(dec1) ? NaN : dec1.total,
            p2 = isnothing(dec2) ? NaN : dec2.total,
            p1_bail = bail(rej[1]), p2_bail = bail(rej[2]),
        )
    end
    return out
end

function paired(label, a::Dict, b::Dict, fa, fb)   # Δ = fb(b) - fa(a) on shared seeds
    ks = sort(collect(intersect(keys(a), keys(b))))
    d = [fb(b[k]) - fa(a[k]) for k in ks]
    @printf("    %-46s mean=%+8.3f  min=%+8.3f  max=%+8.3f  n=%d\n",
        label, mean(d), minimum(d), maximum(d), length(d))
    return mean(d)
end

cellstats(c::Dict, f) = (v = [f(x) for x in values(c)]; (mean(v), std(v)))

v2 = isfile(V2_SUMMARY) ? deserialize(V2_SUMMARY)["noobs"] : nothing

for g in GAINS
    dir = joinpath(RUN_ROOT, "rvr_calib_pilot_noobs_g$(g)")
    cells = Dict(k => load_cell(dir, pat) for (k, pat) in CELLS)

    println("\n", "="^80)
    println("GAIN g = $g  (real & assumed sensor std gain = $(0.1 + g)x base)")
    println("="^80)
    for (k, label) in ((:RR, "(5,5)  mutual robust"), (:NR5, "(NR,5) P2 robust"), (:NN, "(NR,NR) mutual nominal"))
        c = cells[k]
        m1, s1 = cellstats(c, x -> x.p1); m2, s2 = cellstats(c, x -> x.p2)
        b1 = mean([x.p1_bail for x in values(c)]); b2 = mean([x.p2_bail for x in values(c)])
        @printf("  %-24s n=%d  P1=%8.3f±%.3f  P2=%8.3f±%.3f  bail=(%.2f, %.2f)\n",
            label, length(c), m1, s1, m2, s2, b1, b2)
    end
    println()
    paired("A: P1 own effect   (NR,5)-(5,5), P1", cells[:RR], cells[:NR5], x -> x.p1, x -> x.p1)
    paired("B: P2 own effect   (NR,NR)-(NR,5), P2", cells[:NR5], cells[:NN], x -> x.p2, x -> x.p2)
    paired("C: externality on P1 of P2 robust  (NR,5)-(NR,NR)", cells[:NN], cells[:NR5], x -> x.p1, x -> x.p1)
    paired("D: mutual  (NR,NR)-(5,5), P1", cells[:RR], cells[:NN], x -> x.p1, x -> x.p1)
    paired("D: mutual  (NR,NR)-(5,5), P2", cells[:RR], cells[:NN], x -> x.p2, x -> x.p2)

    if g == 1.0 && !isnothing(v2)
        println("\n  EKF-fix effect at g=1.0: pilot (calibrated exec EKF) vs v2 grid (miscalibrated), shared seeds")
        v2cell(l1, l2) = Dict(zip(v2[(l1, l2)].seeds,
            [(p1 = a, p2 = b) for (a, b) in zip(v2[(l1, l2)].p1_totals, v2[(l1, l2)].p2_totals)]))
        for (k, l1, l2) in ((:RR, 5, 5), (:NR5, :NR, 5), (:NN, :NR, :NR))
            paired("  Δ(pilot - v2) @($(l1),$(l2)), P1", v2cell(l1, l2), cells[k], x -> x.p1, x -> x.p1)
            paired("  Δ(pilot - v2) @($(l1),$(l2)), P2", v2cell(l1, l2), cells[k], x -> x.p2, x -> x.p2)
        end
    end
end
println("\nDone.")
