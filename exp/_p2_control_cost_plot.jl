#=
Generate the P2 control cost vs time plot for the nature_control_sweep —
one line per nature_multiplier value, NR baseline dashed gray.

Loads existing .dat files from outputs/merged/nature_control_sweep and only
runs the new plot function (skips the rest of the heavy sweep analysis).

Run from repo root:
    julia --project=. exp/_p2_control_cost_plot.jl
=#

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files,
                                  create_sweep_p2_control_cost_plot,
                                  SenateTrajectoryAnalysisEntry

const SWEEP_DIR = "./exp/senate/outputs/merged/nature_control_sweep"
const OUT_DIR   = "./exp/senate/outputs/analysis/nature_control_sweep/merged"
const MULTIPLIERS = [1, 2, 5, 10, 25, 50, 125, 250, 625, 1250, 3125, 6250]

mkpath(OUT_DIR)

sweep_collected = Dict{Any, NamedTuple}()
for m in MULTIPLIERS
    println("\n===== Loading nature_multiplier=$m =====")
    load_and_analyze_senate_solution_files(
        directory=SWEEP_DIR,
        file_pattern=Regex("p2_nature_multiplier_$(m)_p2_type"))
    entries = copy(SENATE_TRAJECTORY_TRACKER.entries)
    sweep_collected[m] = (
        robust_entries     = [e for e in entries if e.robust],
        non_robust_entries = [e for e in entries if !e.robust],
    )
    println("  robust: $(length(sweep_collected[m].robust_entries)), nr: $(length(sweep_collected[m].non_robust_entries))")
end

create_sweep_p2_control_cost_plot(sweep_collected, "Nature Multiplier";
    directory=OUT_DIR, sweep_name="nature_multiplier")

println("\nDone.")
