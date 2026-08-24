#=
Pull raw P2 final-cumulative-cost values per multiplier per series.

The R-vs-R data is in the "p2_believes_..." filename style (no p1_nature_multiplier
in the filename — that field is in entry.fixed/params, not the iteration name).
The "p1_nature_multiplier_..._p1_type_robust_..." files are 11-byte completion stubs.
So the diagnostic just filters by size > 100 bytes on top of the regex.

Run from repo root:
    julia --project=. exp/_rvr_overlay_diagnostic.jl
=#

using Statistics
using Printf

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files

const RVR_DIR = "./exp/senate/outputs/merged/rvr_nature_control_sweep"
const RNR_DIR = "./exp/senate/outputs/merged/nature_control_sweep"
const MULTIPLIERS = [1, 2, 5, 10, 25, 50, 125, 250, 625, 1250, 3125, 6250]
const PLAYER_IDX = 2

function final_costs(entries)
    out = Tuple{Float64, Int}[]
    for e in entries
        if !isempty(e.incurred_cost_history) && haskey(e.incurred_cost_history, PLAYER_IDX)
            traj = Float64[]
            for step in e.incurred_cost_history[PLAYER_IDX]
                if step isa Tuple && length(step) >= 2
                    push!(traj, Float64(step[2]))
                elseif step isa Number
                    push!(traj, Float64(step))
                end
            end
            isempty(traj) && continue
            push!(out, (sum(traj), e.random_seed))
        end
    end
    return out
end

function summarize(label, pairs)
    if isempty(pairs)
        @printf "  %-30s  n=  0  (no data)\n" label
        return
    end
    vals = sort([p[1] for p in pairs])
    n = length(vals)
    m = mean(vals); md = median(vals); s = n > 1 ? std(vals) : 0.0
    p05 = vals[max(1, ceil(Int, 0.05*n))]
    p95 = vals[max(1, floor(Int, 0.95*n))]
    @printf "  %-30s  n=%3d  mean=%7.2f  med=%7.2f  std=%5.2f  p05=%7.2f  p95=%7.2f  max=%7.2f\n" label n m md s p05 p95 maximum(vals)
end

# Build a one-row-per-multiplier summary table at the end too.
table_rows = Tuple{Int, Float64, Float64, Float64, Float64, Int, Int, Int}[]

println("="^110)
println("P2 final cumulative cost per multiplier per series")
println("="^110)

for m in MULTIPLIERS
    println("\n----- nature_multiplier = $m -----")

    # All matching files in the rvr dir — then we'll filter out stubs by size.
    load_and_analyze_senate_solution_files(directory=RVR_DIR,
        file_pattern=Regex("p2_nature_multiplier_$(m)_p2_type"))
    rvr_all = final_costs(copy(SENATE_TRAJECTORY_TRACKER.entries))

    load_and_analyze_senate_solution_files(directory=RNR_DIR,
        file_pattern=Regex("p2_nature_multiplier_$(m)_p2_type_robust"))
    rnr_r = final_costs(copy(SENATE_TRAJECTORY_TRACKER.entries))

    load_and_analyze_senate_solution_files(directory=RNR_DIR,
        file_pattern=Regex("p2_nature_multiplier_$(m)_p2_type_non_robust"))
    rnr_nr = final_costs(copy(SENATE_TRAJECTORY_TRACKER.entries))

    summarize("R-vs-R", rvr_all)
    summarize("R-vs-NR (P2=Robust)", rnr_r)
    summarize("R-vs-NR (P2=Non-Robust)", rnr_nr)

    mean_rvr = isempty(rvr_all) ? NaN : mean([p[1] for p in rvr_all])
    mean_rnrR = isempty(rnr_r) ? NaN : mean([p[1] for p in rnr_r])
    mean_rnrN = isempty(rnr_nr) ? NaN : mean([p[1] for p in rnr_nr])
    push!(table_rows, (m, mean_rvr, mean_rnrR, mean_rnrN,
        isempty(rvr_all) ? NaN : maximum([p[1] for p in rvr_all]),
        length(rvr_all), length(rnr_r), length(rnr_nr)))
end

println("\n" * "="^110)
println("SUMMARY TABLE (mean P2 final cum cost; max in parens):")
println("="^110)
@printf "  %5s | %18s | %18s | %18s\n" "m" "R-vs-R" "R-vs-NR(P2=Robust)" "R-vs-NR(P2=NR)"
println("  " * "-"^75)
for (m, rvr, rnrR, rnrN, rvr_max, n1, n2, n3) in table_rows
    fmt(x) = isnan(x) ? "       n/a       " : @sprintf("%6.2f (n=%3d)", x, 0)
    rvr_s = isnan(rvr) ? "      n/a       " : @sprintf("%6.2f (n=%3d)", rvr, n1)
    rnrR_s = isnan(rnrR) ? "      n/a       " : @sprintf("%6.2f (n=%3d)", rnrR, n2)
    rnrN_s = isnan(rnrN) ? "      n/a       " : @sprintf("%6.2f (n=%3d)", rnrN, n3)
    @printf "  %5d | %18s | %18s | %18s\n" m rvr_s rnrR_s rnrN_s
end

println("\nDone.")
