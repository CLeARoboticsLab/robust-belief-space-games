#=
No-mismatch control vs v2 grid comparison (both arms, 49 cells x 25 seeds).

nomm = same gt drift (1.0) but every belief correct (players model own AND
opponent noise gain correctly). v2 = opponent-blind mismatch (believes
opponent drift 0). Certainty equivalence predicts the noobs arm is
policy-identical between the two; the obs arm may genuinely differ.

Per arm:
  - 7x7 paired per-seed mean diff (nomm - v2) per player, text table + heatmap
  - headline effects A/B/C/D at c=5 and c=625 computed in BOTH datasets

Outputs (./exp/senate/outputs/analysis/rvr_nomm/):
  - nomm_report.txt, diff_heatmap_<arm>.png, nomm_summary.dat

Run from repo root:
    julia --project=. exp/_rvr_nomm_control_analysis.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files
const STA = SenateTrajectoryAnalysis

const LEVELS = ["5", "25", "125", "625", "3125", "15625", "NR"]
const ARMS = [
    ("noobs", "./exp/senate/outputs/runs/rvr_nature_grid_nomm_noobs",
              "./exp/senate/outputs/runs/rvr_nature_grid_v2_noobs"),
    ("obs",   "./exp/senate/outputs/runs/rvr_nature_grid_nomm_obs",
              "./exp/senate/outputs/runs/rvr_nature_grid_v2_obs"),
]
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_nomm"
mkpath(OUT_DIR)

function cell_pattern(a, b)
    a == "NR" && b == "NR" && return Regex("p1t_non_robust_p2t_non_robust")
    a == "NR" && return Regex("p1t_non_robust_p2nm_$(b)_")
    b == "NR" && return Regex("p1nm_$(a)_p2t_non_robust")
    return Regex("p1nm_$(a)_p2nm_$(b)_")
end

"cell (a,b) -> Dict(seed -> (p1, p2))"
function load_grid(dir)
    cells = Dict{Tuple{String, String}, Dict{Int, NamedTuple}}()
    for a in LEVELS, b in LEVELS
        load_and_analyze_senate_solution_files(directory=dir,
            file_pattern=cell_pattern(a, b))
        d = Dict{Int, NamedTuple}()
        for e in SENATE_TRAJECTORY_TRACKER.entries
            d1 = STA._decompose_entry_costs(e, 1)
            d2 = STA._decompose_entry_costs(e, 2)
            (isnothing(d1) || isnothing(d2)) && continue
            d[e.random_seed] = (p1=d1.total, p2=d2.total)
        end
        cells[(a, b)] = d
    end
    return cells
end

"Paired per-seed diffs f(x[k]) - f(y[k]) over shared seeds."
function pdiff(x, y, f)
    ks = sort(collect(intersect(keys(x), keys(y))))
    isempty(ks) && return (mean=NaN, lo=NaN, hi=NaN, n=0)
    d = [f(x[k]) - f(y[k]) for k in ks]
    (mean=mean(d), lo=minimum(d), hi=maximum(d), n=length(d))
end

"Headline paired effects within one dataset at robust cost level c."
function effects(cells, c)
    (
        A  = pdiff(cells[("NR", c)], cells[(c, c)], v -> v.p1),      # P1 own effect of robustness
        B  = pdiff(cells[("NR", "NR")], cells[("NR", c)], v -> v.p2),# P2 own effect
        C  = pdiff(cells[("NR", c)], cells[("NR", "NR")], v -> v.p1),# externality on P1
        D1 = pdiff(cells[("NR", "NR")], cells[(c, c)], v -> v.p1),
        D2 = pdiff(cells[("NR", "NR")], cells[(c, c)], v -> v.p2),
    )
end

report = open(joinpath(OUT_DIR, "nomm_report.txt"), "w")
summary_all = Dict{String, Any}()

for (arm, dir_nomm, dir_v2) in ARMS
    println("\n############ ARM: $arm ############")
    nomm = load_grid(dir_nomm)
    v2   = load_grid(dir_v2)

    println(report, "="^100)
    println(report, "ARM: $arm — paired per-seed mean diff (nomm - v2), per cell")
    println(report, "="^100)
    diffs = Dict{Tuple{String, String}, Any}()
    @printf(report, "%7s %7s | %4s | %10s [%9s, %9s] | %10s [%9s, %9s]\n",
        "c1", "c2", "n", "dP1 mean", "min", "max", "dP2 mean", "min", "max")
    println(report, "-"^100)
    for a in LEVELS, b in LEVELS
        d1 = pdiff(nomm[(a, b)], v2[(a, b)], v -> v.p1)
        d2 = pdiff(nomm[(a, b)], v2[(a, b)], v -> v.p2)
        diffs[(a, b)] = (p1=d1, p2=d2)
        @printf(report, "%7s %7s | %4d | %+10.3f [%+9.3f, %+9.3f] | %+10.3f [%+9.3f, %+9.3f]\n",
            a, b, d1.n, d1.mean, d1.lo, d1.hi, d2.mean, d2.lo, d2.hi)
    end

    absmeans = [abs(diffs[(a, b)].p1.mean) for a in LEVELS, b in LEVELS if diffs[(a, b)].p1.n > 0]
    append!(absmeans, [abs(diffs[(a, b)].p2.mean) for a in LEVELS, b in LEVELS if diffs[(a, b)].p2.n > 0])
    @printf(report, "\n  |mean diff| across all cells/players: median=%.3f  max=%.3f\n\n",
        median(absmeans), maximum(absmeans))
    println(@sprintf("  [%s] |mean diff|: median=%.3f max=%.3f", arm, median(absmeans), maximum(absmeans)))

    println(report, "Headline effects (paired), nomm vs v2:")
    for c in ["5", "625"]
        en, ev = effects(nomm, c), effects(v2, c)
        for (lbl, key) in (("A: P1 own (NR,c)-(c,c)", :A), ("B: P2 own (NR,NR)-(NR,c)", :B),
                           ("C: ext on P1 (NR,c)-(NR,NR)", :C),
                           ("D1: (NR,NR)-(c,c) P1", :D1), ("D2: (NR,NR)-(c,c) P2", :D2))
            n, v = getfield(en, key), getfield(ev, key)
            @printf(report, "  c=%-5s %-30s nomm=%+8.3f   v2=%+8.3f   gap=%+7.3f\n",
                c, lbl, n.mean, v.mean, n.mean - v.mean)
        end
        println(report)
    end

    # diff heatmaps: diverging, symmetric range (per Makie convention: Reverse(:RdBu))
    m1 = [get(diffs, (a, b), nothing) === nothing ? NaN : diffs[(a, b)].p1.mean for a in LEVELS, b in LEVELS]
    m2 = [get(diffs, (a, b), nothing) === nothing ? NaN : diffs[(a, b)].p2.mean for a in LEVELS, b in LEVELS]
    lim = maximum(abs, filter(isfinite, vcat(vec(m1), vec(m2))))
    lim = lim == 0 ? 1.0 : lim
    fig = Figure(size=(1250, 560))
    for (pos, mat, ttl) in (((1, 1), m1, "Δ P1 total (nomm − v2)"), ((1, 2), m2, "Δ P2 total (nomm − v2)"))
        ax = Axis(fig[pos...], title=ttl, xlabel="P1 level c1", ylabel="P2 level c2",
            xticks=(1:7, LEVELS), yticks=(1:7, LEVELS))
        heatmap!(ax, 1:7, 1:7, mat; colormap=Reverse(:RdBu), colorrange=(-lim, lim))
        for i in 1:7, j in 1:7
            isfinite(mat[i, j]) || continue
            text!(ax, i, j; text=@sprintf("%+.2f", mat[i, j]), align=(:center, :center), fontsize=10)
        end
    end
    Label(fig[0, :], "No-mismatch control minus v2 ($arm): paired per-seed mean cost diff", fontsize=17)
    save(joinpath(OUT_DIR, "diff_heatmap_$(arm).png"), fig)

    summary_all[arm] = (diffs=diffs,
        nomm_effects=Dict(c => effects(nomm, c) for c in ["5", "625"]),
        v2_effects=Dict(c => effects(v2, c) for c in ["5", "625"]))
end

close(report)
serialize(joinpath(OUT_DIR, "nomm_summary.dat"), summary_all)
println("\nWrote nomm_report.txt, diff heatmaps, nomm_summary.dat under $OUT_DIR")
println("Done.")
