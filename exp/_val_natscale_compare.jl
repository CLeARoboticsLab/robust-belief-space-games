#=
Compare a nature-cost-normalized validation run against the original grid run.

Usage (from repo root):
    julia --project=. exp/_val_natscale_compare.jl 625 _val_natscale_625

Loads seed 1001 of cell (c, c) from rvr_nature_grid_sym_noobs (old solver
numerics) and from the validation dir (normalized nature cost), prints
executed cost decompositions per player plus per-solve solver iteration /
rejected-step stats.
=#

using Statistics
using Printf

c1 = parse(Int, ARGS[1])
val_name = ARGS[2]
c2 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : c1

include("./SenateTrajectoryAnalysis.jl")
STA = SenateTrajectoryAnalysis

const GRID_DIR = "./exp/senate/outputs/runs/rvr_nature_grid_sym_noobs"
const VAL_DIR = "./exp/senate/outputs/runs/" * val_name

function summarize(dir, pattern, label)
    STA.load_and_analyze_senate_solution_files(directory=dir, file_pattern=pattern)
    entries = STA.SENATE_TRAJECTORY_TRACKER.entries
    isempty(entries) && (println("$label: NO ENTRIES for $pattern"); return nothing)
    e = entries[1]
    println("\n===== $label (seed $(e.random_seed)) =====")
    for pidx in 1:2
        d = STA._decompose_entry_costs(e, pidx)
        isnothing(d) && (println("  P$pidx: decompose failed"); continue)
        @printf("  P%d total=%.4f  pref=%.4f ctrl=%.4f cov=%.4f obst=%.4f\n",
            pidx, d.total, d.preference, d.control, d.covariance, d.obstacle)
    end
    for pidx in sort(collect(keys(e.cost_history)))
        hist = e.cost_history[pidx]
        iters = [h.solver_iterations for h in hist]
        imps = [h.solver_improvement_iterations for h in hist]
        rejected = iters .- imps
        @printf("  P%d solves=%d  iters min/med/max = %d/%.0f/%d  rejected min/med/max = %d/%.0f/%d  bailouts(rej>=26)=%d\n",
            pidx, length(hist), minimum(iters), median(iters), maximum(iters),
            minimum(rejected), median(rejected), maximum(rejected), count(>=(26), rejected))
    end
    return e
end

summarize(GRID_DIR, Regex("seed_1001_p1nm_$(c1)_p2nm_$(c2)_p1oc"), "ORIGINAL ($c1,$c2)")
summarize(VAL_DIR, Regex("seed_1001"), "NORMALIZED ($c1,$c2)")
println("\nDone.")
