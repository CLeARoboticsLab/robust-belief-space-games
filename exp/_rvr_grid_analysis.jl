#=
Analysis for the RvR nature-cost grid sweep (4x4 cells x 25 seeds, symmetric
drift, obstacle on/off arms).

Per arm (noobs / obs) and per cell (c1, c2) in {5, 25, 125, 625}^2:
  - executed deterministic total cost per player (via _decompose_entry_costs)
  - solver convergence: rejected steps per RH solve, regularization-bailout
    fraction (rejected >= 26 implies reg > 1000 exit while unconverged)

Outputs (./exp/senate/outputs/analysis/rvr_grid/):
  - grid_report.txt          text tables per arm
  - cost_heatmaps_<arm>.png  mean/median total cost per player, 4x4
  - bailout_heatmap_<arm>.png  bailout fraction per player, 4x4
  - grid_summary.dat         serialized per-cell summaries

Run from repo root:
    julia --project=. exp/_rvr_grid_analysis.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files
const STA = SenateTrajectoryAnalysis

const COSTS = [5, 25, 125, 625]
const ARMS = Dict(
    "noobs" => "./exp/senate/outputs/runs/rvr_nature_grid_sym_noobs",
    "obs"   => "./exp/senate/outputs/runs/rvr_nature_grid_sym_obs",
)
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_grid"
const BAILOUT_REJ = 26

mkpath(OUT_DIR)

"Per-cell summary from currently loaded tracker entries."
function summarize_cell(entries)
    totals = Dict(1 => Float64[], 2 => Float64[])
    rej_frac_ge = Dict(1 => Float64[], 2 => Float64[])  # per-solve bailout indicator pooled
    for e in entries
        for pidx in (1, 2)
            dec = STA._decompose_entry_costs(e, pidx)
            isnothing(dec) || push!(totals[pidx], dec.total)

            (e.cost_history isa Dict) || continue
            ch = get(e.cost_history, pidx, nothing)
            (isnothing(ch) || isempty(ch)) && continue
            for c in ch
                (c isa NamedTuple) || continue
                (hasproperty(c, :solver_iterations) && hasproperty(c, :solver_improvement_iterations)) || continue
                (isnothing(c.solver_iterations) || isnothing(c.solver_improvement_iterations)) && continue
                push!(rej_frac_ge[pidx], Float64(c.solver_iterations - c.solver_improvement_iterations))
            end
        end
    end
    stat(v) = isempty(v) ? (mean=NaN, median=NaN, std=NaN, n=0) :
        (mean=mean(v), median=median(v), std=std(v), n=length(v))
    bail(v) = isempty(v) ? NaN : count(>=(BAILOUT_REJ), v) / length(v)
    return (
        n_runs = length(entries),
        p1 = stat(totals[1]), p2 = stat(totals[2]),
        p1_bailout = bail(rej_frac_ge[1]), p2_bailout = bail(rej_frac_ge[2]),
        p1_rej_mean = isempty(rej_frac_ge[1]) ? NaN : mean(rej_frac_ge[1]),
        p2_rej_mean = isempty(rej_frac_ge[2]) ? NaN : mean(rej_frac_ge[2]),
    )
end

"4x4 heatmap panel with direct cell labels. x = P1 cost index, y = P2 cost index."
function heat_panel!(fig, pos, mat, title; colormap=:Blues, fmt=v -> @sprintf("%.1f", v))
    ax = Axis(fig[pos...],
        title=title,
        xlabel="P1 nature cost c1", ylabel="P2 nature cost c2",
        xticks=(1:4, string.(COSTS)), yticks=(1:4, string.(COSTS)))
    finite = filter(isfinite, vec(mat))
    lo, hi = isempty(finite) ? (0.0, 1.0) : extrema(finite)
    hi = hi == lo ? lo + 1 : hi
    heatmap!(ax, 1:4, 1:4, mat; colormap=colormap, colorrange=(lo, hi))
    for i in 1:4, j in 1:4
        v = mat[i, j]
        isfinite(v) || continue
        frac = (v - lo) / (hi - lo)
        text!(ax, i, j; text=fmt(v), align=(:center, :center),
            color=frac > 0.6 ? :white : :black, fontsize=12)
    end
    return ax
end

report = open(joinpath(OUT_DIR, "grid_report.txt"), "w")
summary_all = Dict{String, Dict{Tuple{Int, Int}, Any}}()

for (arm, dir) in sort(collect(ARMS))
    println("\n############ ARM: $arm ($dir) ############")
    cells = Dict{Tuple{Int, Int}, Any}()

    for c1 in COSTS, c2 in COSTS
        load_and_analyze_senate_solution_files(directory=dir,
            file_pattern=Regex("p1nm_$(c1)_p2nm_$(c2)_p1oc"))
        cells[(c1, c2)] = summarize_cell(copy(SENATE_TRAJECTORY_TRACKER.entries))
    end
    summary_all[arm] = cells

    # ---- text report ----
    println(report, "="^110)
    println(report, "ARM: $arm")
    println(report, "="^110)
    @printf(report, "%5s %5s | %4s | %10s %10s %9s | %10s %10s %9s | %8s %8s\n",
        "c1", "c2", "n", "P1 mean", "P1 med", "P1 std", "P2 mean", "P2 med", "P2 std",
        "P1 bail", "P2 bail")
    println(report, "-"^110)
    for c1 in COSTS, c2 in COSTS
        s = cells[(c1, c2)]
        @printf(report, "%5d %5d | %4d | %10.2f %10.2f %9.2f | %10.2f %10.2f %9.2f | %8.3f %8.3f\n",
            c1, c2, s.n_runs, s.p1.mean, s.p1.median, s.p1.std,
            s.p2.mean, s.p2.median, s.p2.std, s.p1_bailout, s.p2_bailout)
    end

    # symmetry check: P2 cost at (c1,c2) should mirror P1 cost at (c2,c1)
    println(report, "\nSymmetry check (mean totals): P2(c1,c2) vs P1(c2,c1)")
    for c1 in COSTS, c2 in COSTS
        s_a = cells[(c1, c2)]; s_b = cells[(c2, c1)]
        @printf(report, "  (%3d,%3d): P2=%10.2f  P1_mirror=%10.2f  diff=%8.2f\n",
            c1, c2, s_a.p2.mean, s_b.p1.mean, s_a.p2.mean - s_b.p1.mean)
    end
    println(report)

    # ---- cost heatmaps ----
    mat(f) = [f(cells[(c1, c2)]) for c1 in COSTS, c2 in COSTS]
    fig = Figure(size=(1150, 950))
    heat_panel!(fig, (1, 1), mat(s -> s.p1.mean),   "P1 mean total cost")
    heat_panel!(fig, (1, 2), mat(s -> s.p2.mean),   "P2 mean total cost")
    heat_panel!(fig, (2, 1), mat(s -> s.p1.median), "P1 median total cost")
    heat_panel!(fig, (2, 2), mat(s -> s.p2.median), "P2 median total cost")
    Label(fig[0, :], "RvR grid ($arm): executed deterministic total cost", fontsize=18)
    save(joinpath(OUT_DIR, "cost_heatmaps_$(arm).png"), fig)

    # ---- bailout heatmaps ----
    figb = Figure(size=(1150, 500))
    heat_panel!(figb, (1, 1), mat(s -> s.p1_bailout), "P1 bailout fraction";
        colormap=:Oranges, fmt=v -> @sprintf("%.3f", v))
    heat_panel!(figb, (1, 2), mat(s -> s.p2_bailout), "P2 bailout fraction";
        colormap=:Oranges, fmt=v -> @sprintf("%.3f", v))
    Label(figb[0, :], "RvR grid ($arm): solver regularization-bailout fraction (rej >= $BAILOUT_REJ)", fontsize=18)
    save(joinpath(OUT_DIR, "bailout_heatmap_$(arm).png"), figb)

    println("Saved heatmaps for arm $arm")
end

close(report)
serialize(joinpath(OUT_DIR, "grid_summary.dat"), summary_all)
println("\nWrote $(joinpath(OUT_DIR, "grid_report.txt")) and grid_summary.dat")
println("Done.")
