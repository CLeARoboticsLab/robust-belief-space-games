#=
Did high-c solves exit via KKT convergence or the regularization bailout?

A reg bailout (reg > 1000, exit while UNCONVERGED) needs ~26+ consecutive
rejected steps (reg *= 1.3 per rejection from ~1.0, *= 0.98 per acceptance).
Rejected steps per solve = solver_iterations - solver_improvement_iterations.
This script prints the distribution of rejected steps per RH solve, per
multiplier, for the R-vs-R sweep (P1 and P2).

Run from repo root:
    julia --project=. exp/_rvr_convergence_check.jl
=#

using Statistics
using Printf

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files

const RVR_DIR = "./exp/senate/outputs/merged/rvr_nature_control_sweep"
const MULTIPLIERS = [10, 25, 50, 125, 250, 625]

@printf "%6s | %3s | %7s | %10s %10s %10s | %14s %14s\n" "c" "P" "solves" "rej_mean" "rej_p95" "rej_max" "frac_rej>=15" "frac_rej>=26"
println("-"^100)

for m in MULTIPLIERS
    load_and_analyze_senate_solution_files(directory=RVR_DIR,
        file_pattern=Regex("p2_nature_multiplier_$(m)_p2_type"))
    entries = copy(SENATE_TRAJECTORY_TRACKER.entries)

    for pidx in (1, 2)
        rejected = Float64[]
        for e in entries
            (e.cost_history isa Dict) || continue
            ch = get(e.cost_history, pidx, nothing)
            (isnothing(ch) || isempty(ch)) && continue
            for c in ch
                (c isa NamedTuple) || continue
                (hasproperty(c, :solver_iterations) && hasproperty(c, :solver_improvement_iterations)) || continue
                (isnothing(c.solver_iterations) || isnothing(c.solver_improvement_iterations)) && continue
                push!(rejected, Float64(c.solver_iterations - c.solver_improvement_iterations))
            end
        end
        isempty(rejected) && continue
        sort!(rejected)
        n = length(rejected)
        p95 = rejected[max(1, floor(Int, 0.95 * n))]
        @printf "%6d | %3d | %7d | %10.2f %10.2f %10.2f | %14.4f %14.4f\n" m pidx n mean(rejected) p95 maximum(rejected) count(>=(15), rejected)/n count(>=(26), rejected)/n
    end
end

println("\nDone.")
