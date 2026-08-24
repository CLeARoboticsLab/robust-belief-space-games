#=
Decompose the ego's robustness value (vs a NOMINAL opponent) into:
  structure component: NN - cNR(c=15625)   (same for every c; solver-structure
                       / equilibrium-selection effect of merely having the
                       nature player, hedge ~inert at c=15625)
  hedge component:     cNR(c=15625) - cNR(c)   (same solver structure in both
                       arms, only the adversary budget differs -> attributable
                       to hedging alone)
Exact additive on paired seeds: total = structure + hedge.

Also checks the inert baseline is stable: cNR(3125) vs cNR(15625) paired diff.

Uses the p1drift cache (all cells, c in {5..15625}, d in {0, +-0.05, +-0.1, +-0.2}).
Units: % of the matched (c, d) pooled RR baseline, consistent with prior figures.

Run from repo root:
    julia --project=. exp/_rvr_hedge_component.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const CS = [5, 25, 125, 625, 3125]
const C_INERT = 15625
const DS = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_plateau"
const GRID_CACHE = "./exp/senate/outputs/analysis/rvr_p1drift/p1drift_cache.dat"
mkpath(OUT_DIR)

grid = deserialize(GRID_CACHE)   # (c, d, cell) => Dict(seed => (p1, p2))

function baseline(c, d)
    rr = grid[(c, d, :RR)]
    mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
end

function paired_pct(a, b, base)
    ks = intersect(keys(a), keys(b))
    vals = [(a[k].p1 - b[k].p1) / base * 100 for k in ks]
    (mean=mean(vals), sem=std(vals) / sqrt(length(vals)), n=length(vals))
end

hedge(c, d)  = paired_pct(grid[(C_INERT, d, :cNR)], grid[(c, d, :cNR)], baseline(c, d))
strct(d)     = paired_pct(grid[(C_INERT, d, :NN)], grid[(C_INERT, d, :cNR)], baseline(C_INERT, d))
total(c, d)  = paired_pct(grid[(c, d, :NN)], grid[(c, d, :cNR)], baseline(c, d))
stab(d)      = paired_pct(grid[(3125, d, :cNR)], grid[(C_INERT, d, :cNR)], baseline(C_INERT, d))

# ---- figure ----------------------------------------------------------------
fig = Figure(size=(1500, 540))
cols = Makie.wong_colors()

ax1 = Axis(fig[1, 1], title="HEDGE component: cNR(inert c=$(C_INERT)) − cNR(c)\n(same solver structure both arms)",
    xlabel="P1 belief drift d", ylabel="% of matched RR baseline")
for (k, c) in enumerate(CS)
    rs = [hedge(c, d) for d in DS]
    ys = [r.mean for r in rs]; es = [1.96 * r.sem for r in rs]
    band!(ax1, DS, ys .- es, ys .+ es, color=(cols[k], 0.15))
    scatterlines!(ax1, DS, ys, color=cols[k], label="ego c=$c")
end
hlines!(ax1, [0.0], color=:gray, linestyle=:dash)
vlines!(ax1, [0.0], color=:gray, linestyle=:dot)
axislegend(ax1, position=:lt, framevisible=false, labelsize=10)

ax2 = Axis(fig[1, 2], title="STRUCTURE component: NN − cNR(inert)\n(solver artifact, same for every c)",
    xlabel="P1 belief drift d", ylabel="% of RR baseline")
rs = [strct(d) for d in DS]
ys = [r.mean for r in rs]; es = [1.96 * r.sem for r in rs]
band!(ax2, DS, ys .- es, ys .+ es, color=(cols[6], 0.2))
scatterlines!(ax2, DS, ys, color=cols[6], label="structure")
rs2 = [stab(d) for d in DS]
scatterlines!(ax2, DS, [r.mean for r in rs2], color=:gray, linestyle=:dash,
    label="stability: cNR(3125) − cNR(15625)")
hlines!(ax2, [0.0], color=:gray, linestyle=:dash)
axislegend(ax2, position=:lt, framevisible=false, labelsize=10)

ax3 = Axis(fig[1, 3], title="TOTAL (= structure + hedge): NN − cNR(c)\n(the original vs-nominal curves)",
    xlabel="P1 belief drift d", ylabel="% of matched RR baseline")
for (k, c) in enumerate(CS)
    rs = [total(c, d) for d in DS]
    ys = [r.mean for r in rs]; es = [1.96 * r.sem for r in rs]
    band!(ax3, DS, ys .- es, ys .+ es, color=(cols[k], 0.15))
    scatterlines!(ax3, DS, ys, color=cols[k], label="ego c=$c")
end
hlines!(ax3, [0.0], color=:gray, linestyle=:dash)
vlines!(ax3, [0.0], color=:gray, linestyle=:dot)
axislegend(ax3, position=:lt, framevisible=false, labelsize=10)

linkyaxes!(ax1, ax2, ax3)
Label(fig[0, :],
    "Ego robustness value vs NOMINAL opponent, decomposed: hedge (clean, same-structure) + structure (artifact) — noobs, 50 seeds, 95% bands",
    fontsize=14)
save(joinpath(OUT_DIR, "hedge_vs_structure.png"), fig)
println("Saved hedge_vs_structure.png")

# ---- report ----------------------------------------------------------------
open(joinpath(OUT_DIR, "hedge_component_report.txt"), "w") do io
    for out in (stdout, io)
        println(out, "Decomposition: NN - cNR(c) = [NN - cNR(15625)] + [cNR(15625) - cNR(c)], % of matched RR base ('*' = 95% CI excludes 0)")
        for d in DS
            s = strct(d); sb = stab(d)
            @printf(out, "\nd=%+.2f   structure: %+.2f%%%s   stability(3125 vs 15625): %+.3f%%%s\n",
                d, s.mean, abs(s.mean) > 1.96 * s.sem ? "*" : " ",
                sb.mean, abs(sb.mean) > 1.96 * sb.sem ? "*" : " ")
            @printf(out, "%8s %12s %12s\n", "ego c", "hedge", "total")
            for c in CS
                h = hedge(c, d); t = total(c, d)
                @printf(out, "%8d   %+7.2f%%%s   %+7.2f%%%s\n", c,
                    h.mean, abs(h.mean) > 1.96 * h.sem ? "*" : " ",
                    t.mean, abs(t.mean) > 1.96 * t.sem ? "*" : " ")
            end
        end
    end
end
println("Wrote hedge_component_report.txt")
