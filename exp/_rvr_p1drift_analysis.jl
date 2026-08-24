#=
One-sided belief-drift experiment (P1 ego, only P1's model of P2's belief
evolution is wrong): robustness effects.

Main figure ("#1-style"): effect vs robustness budget c at fixed drift
d = -0.1 (P1 misreads P2 as aggressive) and d = +0.1 (as cooperative).
Supporting figure: effect vs d at c = 5 (dose-response).

Cells per (c, d) in rvr_p1drift_noobs_c<c>_d<d>: RR, NRc (P1 nominal / P2
robust), cNR (P1 robust / P2 nominal), NN. d = 0 control is the clean game
rvr_symintent_full_noobs_c<c>_t0.0. Effects are paired per seed, in % of that
condition's pooled RR baseline mean, sign so that POSITIVE = being robust
helps that player:
  ego  (P1, mismatched): vs nominal opp = NN.p1 - cNR.p1
                         vs robust  opp = NRc.p1 - RR.p1
  opp  (P2, clean):      vs nominal opp = NN.p2 - NRc.p2
                         vs robust  opp = cNR.p2 - RR.p2
Caches per-seed cell totals in p1drift_cache.dat (delete to reload).

Run from repo root:
    julia --project=. exp/_rvr_p1drift_analysis.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const CS = [5, 25, 125, 625, 3125, 15625]
const DS = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
const DS_BY_C = Dict(c => DS for c in CS)
const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_p1drift"
const CACHE = joinpath(OUT_DIR, "p1drift_cache.dat")
mkpath(OUT_DIR)

run_dir(c, d) = d == 0.0 ?
    joinpath(RUN_ROOT, "rvr_symintent_full_noobs_c$(c)_t0.0") :
    joinpath(RUN_ROOT, "rvr_p1drift_noobs_c$(c)_d$(d)")

cell_patterns(c) = Dict(
    :RR => Regex("(?=.*p1nm_$(c)_)(?=.*p2nm_$(c)_)"),
    :NRc => Regex("(?=.*p1t_non_robust)(?=.*p2nm_$(c)_)"),
    :cNR => Regex("(?=.*p1nm_$(c)_)(?=.*p2t_non_robust)"),
    :NN => Regex("(?=.*p1t_non_robust)(?=.*p2t_non_robust)"),
)

function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)
    out = Dict{Tuple{Int, Float64, Symbol}, Dict{Int, NamedTuple}}()
    for c in CS, d in DS_BY_C[c]
        dir = run_dir(c, d)
        isdir(dir) || (println("MISSING $dir"); continue)
        for (cell, pat) in cell_patterns(c)
            Base.invokelatest(STA.load_and_analyze_senate_solution_files;
                directory=dir, file_pattern=pat)
            dd = Dict{Int, NamedTuple}()
            for e in STA.SENATE_TRAJECTORY_TRACKER.entries
                d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
                d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
                (isnothing(d1) || isnothing(d2)) && continue
                dd[e.random_seed] = (p1=d1.total, p2=d2.total)
            end
            out[(c, d, cell)] = dd
        end
        println("loaded c=$c d=$d")
    end
    serialize(CACHE, out)
    return out
end

data = isfile(CACHE) ? deserialize(CACHE) : build_cache()
println("Cell totals loaded ($(isfile(CACHE) ? "cache" : "fresh")).")

p1f = v -> v.p1
p2f = v -> v.p2

"Paired diff (to - from) in % of the condition's pooled RR baseline mean."
function eff(c, d, from, to, f)
    (haskey(data, (c, d, from)) && haskey(data, (c, d, to)) && haskey(data, (c, d, :RR))) ||
        return (mean=NaN, sem=NaN, n=0)
    rr = data[(c, d, :RR)]
    isempty(rr) && return (mean=NaN, sem=NaN, n=0)
    base = mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
    ks = intersect(keys(data[(c, d, from)]), keys(data[(c, d, to)]))
    vals = [(f(data[(c, d, to)][k]) - f(data[(c, d, from)][k])) / base * 100 for k in ks]
    isempty(vals) && return (mean=NaN, sem=NaN, n=0)
    (mean=mean(vals), sem=std(vals) / sqrt(length(vals)), n=length(vals))
end

# positive = being robust helps that player
# robust opponents always use the SAME budget c as the panel/line (matched-c cells)
const CURVES = [
    (:ego_vs_N, "ego (mismatched P1) vs nominal opp", (c, d) -> eff(c, d, :cNR, :NN, p1f)),
    (:ego_vs_R, "ego (mismatched P1) vs robust opp (same c)", (c, d) -> eff(c, d, :RR, :NRc, p1f)),
    (:opp_vs_N, "opponent (clean P2) vs nominal opp", (c, d) -> eff(c, d, :NRc, :NN, p2f)),
    (:opp_vs_R, "opponent (clean P2) vs robust opp (same c)", (c, d) -> eff(c, d, :RR, :cNR, p2f)),
]

# ---- main figure: effect vs c at d = -0.1 / +0.1 ----------------------------
fig = Figure(size=(2300, 560))
for (panel, d) in enumerate([-0.2, -0.1, 0.1, 0.2])
    ax = Axis(fig[1, panel],
        title=d < 0 ? "d = $(d)  (P1 misreads P2 as aggressive)" :
                      "d = +$(d)  (P1 misreads P2 as cooperative)",
        xlabel="robustness budget c (lower = more robust)",
        ylabel="value of being robust, % of RR baseline",
        xticks=(1:length(CS), string.(CS)))
    cols = Makie.wong_colors()
    for (k, (key, lab, fe)) in enumerate(CURVES)
        pts = [(j, fe(c, d)) for (j, c) in enumerate(CS)]
        pts = [(j, r) for (j, r) in pts if isfinite(r.mean)]
        isempty(pts) && continue
        xs = [p[1] for p in pts]
        ys = [p[2].mean for p in pts]
        es = [1.96 * p[2].sem for p in pts]
        band!(ax, xs, ys .- es, ys .+ es, color=(cols[k], 0.15))
        scatterlines!(ax, xs, ys, color=cols[k], label=lab)
    end
    hlines!(ax, [0.0], color=:gray, linestyle=:dash)
    panel == 1 && axislegend(ax, position=:lb, framevisible=false)
end
Label(fig[0, :],
    "One-sided belief drift (only P1's model of P2's belief evolves wrongly), noobs, 50 seeds",
    fontsize=15)
save(joinpath(OUT_DIR, "p1drift_vs_c.png"), fig)
println("Saved p1drift_vs_c.png")

# ---- supporting figure: effect vs d at c = 5 --------------------------------
fig2 = Figure(size=(760, 560))
ax = Axis(fig2[1, 1],
    title="Dose-response at c = 5",
    xlabel="belief-drift rate d  (- = misread as aggressive, + = as cooperative)",
    ylabel="value of being robust, % of RR baseline")
cols = Makie.wong_colors()
for (k, (key, lab, fe)) in enumerate(CURVES)
    pts = [(d, fe(5, d)) for d in DS_BY_C[5]]
    pts = [(d, r) for (d, r) in pts if isfinite(r.mean)]
    isempty(pts) && continue
    xs = [p[1] for p in pts]
    ys = [p[2].mean for p in pts]
    es = [1.96 * p[2].sem for p in pts]
    band!(ax, xs, ys .- es, ys .+ es, color=(cols[k], 0.15))
    scatterlines!(ax, xs, ys, color=cols[k], label=lab)
end
hlines!(ax, [0.0], color=:gray, linestyle=:dash)
vlines!(ax, [0.0], color=:gray, linestyle=:dot)
axislegend(ax, position=:lt, framevisible=false)
save(joinpath(OUT_DIR, "p1drift_vs_d.png"), fig2)
println("Saved p1drift_vs_d.png")

# ---- budget-frontier lines (symintent_budget_frontier style): ego's
# robustness value vs d, one line per c --------------------------------------
fig3 = Figure(size=(1500, 620))
for (panel, (curvekey, paneltitle)) in enumerate([
    (:ego_vs_N, "vs NOMINAL opponent"),
    (:ego_vs_R, "vs ROBUST opponent (opp budget = same c as line)"),
])
    fe = CURVES[findfirst(cv -> cv[1] == curvekey, CURVES)][3]
    ax = Axis(fig3[1, panel],
        title="Ego's (mismatched P1) robustness payoff, $paneltitle",
        xlabel="belief-drift rate d  (- = misread as aggressive, + = as cooperative)",
        ylabel="value of being robust, % of RR baseline")
    fcols = cgrad(:viridis, length(CS), categorical=true)
    for (j, c) in enumerate(CS)
        pts = [(d, fe(c, d)) for d in DS_BY_C[c]]
        pts = [(d, r) for (d, r) in pts if isfinite(r.mean)]
        isempty(pts) && continue
        xs = [p[1] for p in pts]
        ys = [p[2].mean for p in pts]
        es = [1.96 * p[2].sem for p in pts]
        band!(ax, xs, ys .- es, ys .+ es, color=(fcols[j], 0.15))
        scatterlines!(ax, xs, ys, color=fcols[j], label="ego c=$c")
    end
    hlines!(ax, [0.0], color=:gray, linestyle=:dash)
    vlines!(ax, [0.0], color=:gray, linestyle=:dot)
    axislegend(ax, position=:lt, framevisible=false)
end
save(joinpath(OUT_DIR, "p1drift_budget_frontier.png"), fig3)
println("Saved p1drift_budget_frontier.png")

# ---- text table -------------------------------------------------------------
open(joinpath(OUT_DIR, "p1drift_report.txt"), "w") do io
    for out in (stdout, io)
        println(out, "One-sided belief drift, % of condition RR baseline ('*' = 95% CI excludes 0)")
        for c in CS
            println(out, "\n-- c = $c --")
            @printf(out, "%8s |", "d")
            for (key, _, _) in CURVES
                @printf(out, " %12s", key)
            end
            println(out)
            for d in DS_BY_C[c]
                @printf(out, "%8.2f |", d)
                for (_, _, fe) in CURVES
                    r = fe(c, d)
                    if isfinite(r.mean)
                        @printf(out, "   %+7.2f%%%s", r.mean, abs(r.mean) > 1.96 * r.sem ? "*" : " ")
                    else
                        @printf(out, " %12s", "--")
                    end
                end
                println(out)
            end
        end
    end
end
println("Wrote p1drift_report.txt")
