#=
One-sided STATIC intent mismatch (P1 ego, only P1's model of P2's preference
target is wrong: P1 believes P2 wants [1+2t, 3-2t]): robustness effects.
Static counterpart of the belief-drift experiment — same cells, same effect
definitions, so the two dose-response figures are directly comparable
(channel-independence check: preference-channel error vs belief-dynamics error).

c = 5 for every robust player in this sweep. t = 0 control is the clean game
rvr_symintent_full_noobs_c5_t0.0. Effects are paired per seed, in % of that
condition's pooled RR baseline mean, sign so that POSITIVE = being robust
helps that player.

Run from repo root:
    julia --project=. exp/_rvr_asymintent_analysis.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const C = 5
const TS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_asymintent"
const CACHE = joinpath(OUT_DIR, "asymintent_cache.dat")
mkpath(OUT_DIR)

run_dir(t) = t == 0.0 ?
    joinpath(RUN_ROOT, "rvr_symintent_full_noobs_c$(C)_t0.0") :
    joinpath(RUN_ROOT, "rvr_asymintent_full_noobs_c$(C)_t$(t)")

const CELLS = Dict(
    :RR => Regex("(?=.*p1nm_$(C)_)(?=.*p2nm_$(C)_)"),
    :NRc => Regex("(?=.*p1t_non_robust)(?=.*p2nm_$(C)_)"),
    :cNR => Regex("(?=.*p1nm_$(C)_)(?=.*p2t_non_robust)"),
    :NN => Regex("(?=.*p1t_non_robust)(?=.*p2t_non_robust)"),
)

function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)
    out = Dict{Tuple{Float64, Symbol}, Dict{Int, NamedTuple}}()
    for t in TS
        dir = run_dir(t)
        isdir(dir) || (println("MISSING $dir"); continue)
        for (cell, pat) in CELLS
            Base.invokelatest(STA.load_and_analyze_senate_solution_files;
                directory=dir, file_pattern=pat)
            dd = Dict{Int, NamedTuple}()
            for e in STA.SENATE_TRAJECTORY_TRACKER.entries
                d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
                d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
                (isnothing(d1) || isnothing(d2)) && continue
                dd[e.random_seed] = (p1=d1.total, p2=d2.total)
            end
            out[(t, cell)] = dd
        end
        println("loaded t=$t")
    end
    serialize(CACHE, out)
    return out
end

data = isfile(CACHE) ? deserialize(CACHE) : build_cache()
println("Cell totals loaded ($(isfile(CACHE) ? "cache" : "fresh")).")

p1f = v -> v.p1
p2f = v -> v.p2

"Paired diff (to - from) in % of the condition's pooled RR baseline mean."
function eff(t, from, to, f)
    (haskey(data, (t, from)) && haskey(data, (t, to)) && haskey(data, (t, :RR))) ||
        return (mean=NaN, sem=NaN, n=0)
    rr = data[(t, :RR)]
    isempty(rr) && return (mean=NaN, sem=NaN, n=0)
    base = mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
    ks = intersect(keys(data[(t, from)]), keys(data[(t, to)]))
    vals = [(f(data[(t, to)][k]) - f(data[(t, from)][k])) / base * 100 for k in ks]
    isempty(vals) && return (mean=NaN, sem=NaN, n=0)
    (mean=mean(vals), sem=std(vals) / sqrt(length(vals)), n=length(vals))
end

# positive = being robust helps that player; all robust players use c = 5
const CURVES = [
    (:ego_vs_N, "ego (mismatched P1) vs nominal opp", t -> eff(t, :cNR, :NN, p1f)),
    (:ego_vs_R, "ego (mismatched P1) vs robust opp (c=$(C))", t -> eff(t, :RR, :NRc, p1f)),
    (:opp_vs_N, "opponent (clean P2) vs nominal opp", t -> eff(t, :NRc, :NN, p2f)),
    (:opp_vs_R, "opponent (clean P2) vs robust opp (c=$(C))", t -> eff(t, :RR, :cNR, p2f)),
]

# ---- dose-response figure (matches p1drift_vs_d layout) ---------------------
fig = Figure(size=(760, 560))
ax = Axis(fig[1, 1],
    title="One-sided STATIC intent mismatch, robust budget c = $(C)",
    xlabel="intent error t  (- = misread as MORE opposed, + = as cooperative)",
    ylabel="value of being robust, % of RR baseline")
cols = Makie.wong_colors()
for (k, (key, lab, fe)) in enumerate(CURVES)
    pts = [(t, fe(t)) for t in TS]
    pts = [(t, r) for (t, r) in pts if isfinite(r.mean)]
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
save(joinpath(OUT_DIR, "asymintent_vs_t.png"), fig)
println("Saved asymintent_vs_t.png")

# ---- text table -------------------------------------------------------------
open(joinpath(OUT_DIR, "asymintent_report.txt"), "w") do io
    for out in (stdout, io)
        println(out, "One-sided static intent mismatch, c=$(C), % of condition RR baseline ('*' = 95% CI excludes 0)")
        @printf(out, "%8s |", "t")
        for (key, _, _) in CURVES
            @printf(out, " %12s", key)
        end
        println(out)
        for t in TS
            @printf(out, "%8.2f |", t)
            for (_, _, fe) in CURVES
                r = fe(t)
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
println("Wrote asymintent_report.txt")
