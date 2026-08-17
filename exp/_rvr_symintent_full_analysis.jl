#=
Full symmetric intent-mismatch sweep analysis.

Design: t in {0, 0.25, 0.5, 0.75, 1.0} x budget c in {5, 625} x arm {noobs, obs},
4 cells per condition ((c,c), (NR,c), (c,NR), (NR,NR)), 25 seeds.
Both players hold mirrored wrong beliefs about the opponent's target
(P1 believes P2 wants [1+2t, 3-2t]; P2 believes P1 wants [3-2t, 1+2t]).

Per (arm, c, t), role-pooled paired effects (pair key = (role, seed)):
  ownR: own effect of robustness vs ROBUST opponent
        P1: (NR,c)-(c,c)  pooled with  P2: (c,NR)-(c,c)
  ownN: own effect of robustness vs NOMINAL opponent
        P2: (NR,NR)-(NR,c)  pooled with  P1: (NR,NR)-(c,NR)
  ext:  externality of opponent robustness on a nominal player
        P1: (NR,c)-(NR,NR)  pooled with  P2: (c,NR)-(NR,NR)
  mut:  (NR,NR)-(c,c), both players pooled
  sym:  symmetry check — ownR computed from P1 only vs from P2 only

Outputs (./exp/senate/outputs/analysis/rvr_symintent/):
  - symintent_report.txt
  - dose_response_<arm>.png   effect curves vs t, one panel per budget c
  - symintent_summary.dat

Run from repo root (optional ARGS: arm names to restrict, e.g. noobs):
    julia --project=. exp/_rvr_symintent_full_analysis.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files
const STA = SenateTrajectoryAnalysis

const TS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]  # t<0: believed goal pushed AWAY (adversarial misread)
const BUDGETS = [5, 625]
const ARMS = [("noobs", "rvr_symintent_full_noobs"), ("obs", "rvr_symintent_full_obs")]
const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_symintent"
mkpath(OUT_DIR)

arm_filter = isempty(ARGS) ? ARMS : [a for a in ARMS if a[1] in ARGS]

# Order-agnostic cell patterns (combo-key order varies between sweeps).
cell_patterns(c) = Dict(
    :RR => Regex("(?=.*p1nm_$(c)_)(?=.*p2nm_$(c)_)"),
    :NRc => Regex("(?=.*p1t_non_robust)(?=.*p2nm_$(c)_)"),
    :cNR => Regex("(?=.*p1nm_$(c)_)(?=.*p2t_non_robust)"),
    :NN => Regex("(?=.*p1t_non_robust)(?=.*p2t_non_robust)"),
)

function load_cell(dir, pat)
    load_and_analyze_senate_solution_files(directory=dir, file_pattern=pat)
    d = Dict{Int, NamedTuple}()
    for e in SENATE_TRAJECTORY_TRACKER.entries
        d1 = STA._decompose_entry_costs(e, 1)
        d2 = STA._decompose_entry_costs(e, 2)
        (isnothing(d1) || isnothing(d2)) && continue
        d[e.random_seed] = (p1=d1.total, p2=d2.total)
    end
    return d
end

"Pooled paired diffs: list of (from_cell, to_cell, from_field, to_field, role)."
function pooled_diff(cells, specs)
    d = Float64[]
    for (from, to, ffrom, fto) in specs
        ks = intersect(keys(cells[from]), keys(cells[to]))
        append!(d, [fto(cells[to][k]) - ffrom(cells[from][k]) for k in ks])
    end
    isempty(d) && return (mean=NaN, sem=NaN, lo=NaN, hi=NaN, n=0)
    (mean=mean(d), sem=std(d) / sqrt(length(d)), lo=minimum(d), hi=maximum(d), n=length(d))
end

p1f = v -> v.p1
p2f = v -> v.p2

function condition_effects(cells)
    (
        ownR  = pooled_diff(cells, [(:RR, :NRc, p1f, p1f), (:RR, :cNR, p2f, p2f)]),
        ownR1 = pooled_diff(cells, [(:RR, :NRc, p1f, p1f)]),   # P1-only, symmetry check
        ownR2 = pooled_diff(cells, [(:RR, :cNR, p2f, p2f)]),   # P2-only, symmetry check
        ownN  = pooled_diff(cells, [(:NRc, :NN, p2f, p2f), (:cNR, :NN, p1f, p1f)]),
        ext   = pooled_diff(cells, [(:NN, :NRc, p1f, p1f), (:NN, :cNR, p2f, p2f)]),
        mut   = pooled_diff(cells, [(:RR, :NN, p1f, p1f), (:RR, :NN, p2f, p2f)]),
        base  = (mean([v.p1 for v in values(cells[:RR])] ∪ [v.p2 for v in values(cells[:RR])]),),
    )
end

report = open(joinpath(OUT_DIR, "symintent_report.txt"), "w")
summary_all = Dict{Tuple{String, Int, Float64}, Any}()

for (arm, prefix) in arm_filter
    for c in BUDGETS
        println(report, "="^100)
        println(report, "ARM: $arm   budget c=$c")
        println(report, "="^100)
        @printf(report, "%6s | %5s | %18s %18s %18s %18s | %14s\n",
            "t", "n/cell", "ownR (vs robust)", "ownN (vs nominal)", "ext (on nominal)",
            "mut (NN - RR)", "sym ownR P1|P2")
        for t in TS
            dir = joinpath(RUN_ROOT, "$(prefix)_c$(c)_t$(t)")
            isdir(dir) || (println(report, "  t=$t: MISSING $dir"); continue)
            pats = cell_patterns(c)
            cells = Dict(k => load_cell(dir, p) for (k, p) in pats)
            any(isempty, values(cells)) && (println(report, "  t=$t: INCOMPLETE ($(join([string(k, "=", length(v)) for (k, v) in cells], ", ")))"); continue)
            eff = condition_effects(cells)
            summary_all[(arm, c, t)] = eff
            @printf(report, "%6.2f | %5d | %+8.3f ± %5.3f  %+8.3f ± %5.3f  %+8.3f ± %5.3f  %+8.3f ± %5.3f | %+6.3f|%+6.3f\n",
                t, length(cells[:RR]),
                eff.ownR.mean, eff.ownR.sem, eff.ownN.mean, eff.ownN.sem,
                eff.ext.mean, eff.ext.sem, eff.mut.mean, eff.mut.sem,
                eff.ownR1.mean, eff.ownR2.mean)
            println("  [$arm c=$c t=$t] ownR=$(round(eff.ownR.mean, digits=3)) ownN=$(round(eff.ownN.mean, digits=3)) ext=$(round(eff.ext.mean, digits=3)) mut=$(round(eff.mut.mean, digits=3))")
        end
        println(report)
    end

    # dose-response figure: one panel per budget
    fig = Figure(size=(1250, 520))
    for (i, c) in enumerate(BUDGETS)
        ax = Axis(fig[1, i], title="$arm, budget c=$c",
            xlabel="intent-mismatch magnitude t", ylabel="paired cost effect")
        series = [(:ownR, "own effect vs robust opp", RGBf(0.00, 0.45, 0.70)),
                  (:ownN, "own effect vs nominal opp", RGBf(0.90, 0.62, 0.00)),
                  (:ext, "externality on nominal", RGBf(0.00, 0.62, 0.45)),
                  (:mut, "mutual (NN − RR)", RGBf(0.80, 0.47, 0.65))]
        for (key, lbl, col) in series
            pts = [(t, summary_all[(arm, c, t)]) for t in TS if haskey(summary_all, (arm, c, t))]
            isempty(pts) && continue
            xs = [p[1] for p in pts]
            ys = [getfield(p[2], key).mean for p in pts]
            es = [getfield(p[2], key).sem for p in pts]
            band!(ax, xs, ys .- 1.96 .* es, ys .+ 1.96 .* es, color=(col, 0.18))
            scatterlines!(ax, xs, ys, color=col, label=lbl)
        end
        hlines!(ax, [0.0], color=:gray, linestyle=:dash)
        i == length(BUDGETS) && axislegend(ax, position=:lt, framevisible=false, labelsize=11)
    end
    Label(fig[0, :], "Symmetric intent mismatch: dose-response of robustness effects ($arm)", fontsize=17)
    save(joinpath(OUT_DIR, "dose_response_$(arm).png"), fig)
    println("Saved dose_response_$(arm).png")
end

close(report)
serialize(joinpath(OUT_DIR, "symintent_summary.dat"), summary_all)
println("\nWrote symintent_report.txt and summary under $OUT_DIR")
println("Done.")
