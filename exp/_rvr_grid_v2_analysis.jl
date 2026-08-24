#=
Analysis for the v2 RvR nature-cost grid (normalized nature cost, commit
e84f1793): 6x6 robust grid c in {5,25,125,625,3125,15625} + nominal
(non_robust) row/col/corner = 7x7 cells x 25 seeds, obstacle on/off arms.

The nominal player is treated as the c = Inf limit and indexed "NR" at the
end of each axis.

Per arm and per cell:
  - executed deterministic total cost per player (via _decompose_entry_costs),
    stored per-seed to allow paired seed-level differences downstream
  - solver convergence: rejected steps per RH solve, bailout fraction
    (rejected >= 26 implies reg > 1000 exit while unconverged)

Outputs (./exp/senate/outputs/analysis/rvr_grid_v2/):
  - grid_report.txt            text tables per arm
  - cost_heatmaps_<arm>.png    mean/median total cost per player, 7x7
  - bailout_heatmap_<arm>.png  bailout fraction per player, 7x7
  - grid_summary.dat           serialized per-cell summaries incl. per-seed totals

Run from repo root:
    julia --project=. exp/_rvr_grid_v2_analysis.jl [arm...]
Optional args restrict to a subset of arms (noobs / obs); default both.
grid_summary.dat is merged across invocations, so arms can run separately.
=#

using Statistics
using Printf
using Serialization
using CairoMakie

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files
const STA = SenateTrajectoryAnalysis

const COSTS = [5, 25, 125, 625, 3125, 15625]
const LEVELS = vcat(Any[c for c in COSTS], Any[:NR])   # :NR = nominal player
const NL = length(LEVELS)
const LEVEL_LABELS = vcat(string.(COSTS), ["NR"])
const ARMS = Dict(
    "noobs" => "./exp/senate/outputs/runs/rvr_nature_grid_v2_noobs",
    "obs"   => "./exp/senate/outputs/runs/rvr_nature_grid_v2_obs",
)
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_grid_v2"
const BAILOUT_REJ = 26

mkpath(OUT_DIR)

"Filename regex for cell (l1, l2), each level an Int multiplier or :NR."
function cell_pattern(l1, l2)
    l1 === :NR && l2 === :NR && return Regex("p1t_non_robust_p2t_non_robust")
    l1 === :NR && return Regex("p1t_non_robust_p2nm_$(l2)_p1oc")
    l2 === :NR && return Regex("p1nm_$(l1)_p2t_non_robust")
    return Regex("p1nm_$(l1)_p2nm_$(l2)_p1oc")
end

"Per-cell summary from currently loaded tracker entries."
function summarize_cell(entries)
    totals = Dict(1 => Float64[], 2 => Float64[])
    seeds = Int[]
    rejected = Dict(1 => Float64[], 2 => Float64[])
    for e in entries
        push!(seeds, e.random_seed)
        for pidx in (1, 2)
            dec = STA._decompose_entry_costs(e, pidx)
            push!(totals[pidx], isnothing(dec) ? NaN : dec.total)

            (e.cost_history isa Dict) || continue
            ch = get(e.cost_history, pidx, nothing)
            (isnothing(ch) || isempty(ch)) && continue
            for c in ch
                (c isa NamedTuple) || continue
                (hasproperty(c, :solver_iterations) && hasproperty(c, :solver_improvement_iterations)) || continue
                (isnothing(c.solver_iterations) || isnothing(c.solver_improvement_iterations)) && continue
                push!(rejected[pidx], Float64(c.solver_iterations - c.solver_improvement_iterations))
            end
        end
    end
    stat(v) = (fv = filter(isfinite, v); isempty(fv) ? (mean=NaN, median=NaN, std=NaN, n=0) :
        (mean=mean(fv), median=median(fv), std=std(fv), n=length(fv)))
    bail(v) = isempty(v) ? NaN : count(>=(BAILOUT_REJ), v) / length(v)
    return (
        n_runs = length(entries),
        seeds = seeds,
        p1_totals = totals[1], p2_totals = totals[2],
        p1 = stat(totals[1]), p2 = stat(totals[2]),
        p1_bailout = bail(rejected[1]), p2_bailout = bail(rejected[2]),
        p1_rej_mean = isempty(rejected[1]) ? NaN : mean(rejected[1]),
        p2_rej_mean = isempty(rejected[2]) ? NaN : mean(rejected[2]),
    )
end

"NLxNL heatmap panel with direct cell labels. x = P1 level index, y = P2 level index."
function heat_panel!(fig, pos, mat, title; colormap=:Blues, fmt=v -> @sprintf("%.1f", v))
    ax = Axis(fig[pos...],
        title=title,
        xlabel="P1 nature cost c1", ylabel="P2 nature cost c2",
        xticks=(1:NL, LEVEL_LABELS), yticks=(1:NL, LEVEL_LABELS),
        xticklabelrotation=pi / 4)
    finite = filter(isfinite, vec(mat))
    lo, hi = isempty(finite) ? (0.0, 1.0) : extrema(finite)
    hi = hi == lo ? lo + 1 : hi
    heatmap!(ax, 1:NL, 1:NL, mat; colormap=colormap, colorrange=(lo, hi))
    for i in 1:NL, j in 1:NL
        v = mat[i, j]
        isfinite(v) || continue
        frac = (v - lo) / (hi - lo)
        text!(ax, i, j; text=fmt(v), align=(:center, :center),
            color=frac > 0.6 ? :white : :black, fontsize=10)
    end
    return ax
end

arm_sel = isempty(ARGS) ? sort(collect(keys(ARMS))) : ARGS
report = open(joinpath(OUT_DIR, "grid_report_$(join(arm_sel, '_')).txt"), "w")
summary_path = joinpath(OUT_DIR, "grid_summary.dat")
summary_all = isfile(summary_path) ? deserialize(summary_path) :
    Dict{String, Dict{Tuple{Any, Any}, Any}}()

for arm in arm_sel
    dir = ARMS[arm]
    println("\n############ ARM: $arm ($dir) ############")
    cells = Dict{Tuple{Any, Any}, Any}()

    for l1 in LEVELS, l2 in LEVELS
        load_and_analyze_senate_solution_files(directory=dir,
            file_pattern=cell_pattern(l1, l2))
        s = summarize_cell(copy(SENATE_TRAJECTORY_TRACKER.entries))
        cells[(l1, l2)] = s
        s.n_runs == 25 || @warn "cell ($l1, $l2) has $(s.n_runs) runs (expected 25)"
    end
    summary_all[arm] = cells

    # ---- text report ----
    println(report, "="^110)
    println(report, "ARM: $arm  (v2, normalized nature cost)")
    println(report, "="^110)
    @printf(report, "%6s %6s | %4s | %10s %10s %9s | %10s %10s %9s | %8s %8s\n",
        "c1", "c2", "n", "P1 mean", "P1 med", "P1 std", "P2 mean", "P2 med", "P2 std",
        "P1 bail", "P2 bail")
    println(report, "-"^110)
    for l1 in LEVELS, l2 in LEVELS
        s = cells[(l1, l2)]
        @printf(report, "%6s %6s | %4d | %10.2f %10.2f %9.2f | %10.2f %10.2f %9.2f | %8.3f %8.3f\n",
            string(l1), string(l2), s.n_runs, s.p1.mean, s.p1.median, s.p1.std,
            s.p2.mean, s.p2.median, s.p2.std, s.p1_bailout, s.p2_bailout)
    end

    # symmetry check: P2 cost at (l1,l2) should mirror P1 cost at (l2,l1)
    println(report, "\nSymmetry check (mean totals): P2(c1,c2) vs P1(c2,c1)")
    for l1 in LEVELS, l2 in LEVELS
        s_a = cells[(l1, l2)]; s_b = cells[(l2, l1)]
        @printf(report, "  (%6s,%6s): P2=%10.2f  P1_mirror=%10.2f  diff=%8.2f\n",
            string(l1), string(l2), s_a.p2.mean, s_b.p1.mean, s_a.p2.mean - s_b.p1.mean)
    end
    println(report)

    # ---- cost heatmaps ----
    mat(f) = [f(cells[(l1, l2)]) for l1 in LEVELS, l2 in LEVELS]
    fig = Figure(size=(1350, 1100))
    heat_panel!(fig, (1, 1), mat(s -> s.p1.mean),   "P1 mean total cost")
    heat_panel!(fig, (1, 2), mat(s -> s.p2.mean),   "P2 mean total cost")
    heat_panel!(fig, (2, 1), mat(s -> s.p1.median), "P1 median total cost")
    heat_panel!(fig, (2, 2), mat(s -> s.p2.median), "P2 median total cost")
    Label(fig[0, :], "RvR grid v2 ($arm): executed deterministic total cost (NR = nominal)", fontsize=18)
    save(joinpath(OUT_DIR, "cost_heatmaps_$(arm).png"), fig)

    # ---- bailout heatmaps ----
    figb = Figure(size=(1350, 560))
    heat_panel!(figb, (1, 1), mat(s -> s.p1_bailout), "P1 bailout fraction";
        colormap=:Oranges, fmt=v -> @sprintf("%.3f", v))
    heat_panel!(figb, (1, 2), mat(s -> s.p2_bailout), "P2 bailout fraction";
        colormap=:Oranges, fmt=v -> @sprintf("%.3f", v))
    Label(figb[0, :], "RvR grid v2 ($arm): solver bailout fraction (rej >= $BAILOUT_REJ)", fontsize=18)
    save(joinpath(OUT_DIR, "bailout_heatmap_$(arm).png"), figb)

    println("Saved heatmaps for arm $arm")
end

close(report)
serialize(summary_path, summary_all)
println("\nWrote grid report and grid_summary.dat to $OUT_DIR")
println("Done.")
