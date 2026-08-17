#=
Percent-delta view of the full symmetric intent-mismatch sweep.

Each effect from symintent_summary.dat divided by that condition's own
mutual-robust (c,c) baseline mean cost, x100. Text tables + dose-response
curves in % units.

Run from repo root (after _rvr_symintent_full_analysis.jl):
    julia --project=. exp/_rvr_symintent_pct.jl
=#

using Statistics, Printf, Serialization, CairoMakie

const TS = [0.0, 0.25, 0.5, 0.75, 1.0]
const BUDGETS = [5, 625]
const ARMS = ["noobs", "obs"]
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_symintent"

summary_all = deserialize(joinpath(OUT_DIR, "symintent_summary.dat"))

for arm in ARMS
    for c in BUDGETS
        println("="^96)
        println("ARM: $arm   budget c=$c   (effects as % of mutual-robust baseline cost)")
        println("="^96)
        @printf("%6s | %10s | %16s %16s %16s %16s\n",
            "t", "base cost", "ownR %", "ownN %", "ext %", "mut %")
        for t in TS
            haskey(summary_all, (arm, c, t)) || continue
            eff = summary_all[(arm, c, t)]
            base = eff.base[1]
            pct(x) = 100 * x.mean / base
            pcte(x) = 100 * 1.96 * x.sem / base
            @printf("%6.2f | %10.2f | %+7.2f%% ± %4.2f  %+7.2f%% ± %4.2f  %+7.2f%% ± %4.2f  %+7.2f%% ± %4.2f\n",
                t, base, pct(eff.ownR), pcte(eff.ownR), pct(eff.ownN), pcte(eff.ownN),
                pct(eff.ext), pcte(eff.ext), pct(eff.mut), pcte(eff.mut))
        end
        println()
    end

    fig = Figure(size=(1250, 520))
    for (i, c) in enumerate(BUDGETS)
        ax = Axis(fig[1, i], title="$arm, budget c=$c",
            xlabel="intent-mismatch magnitude t", ylabel="effect, % of baseline cost")
        series = [(:ownR, "own effect vs robust opp", RGBf(0.00, 0.45, 0.70)),
                  (:ownN, "own effect vs nominal opp", RGBf(0.90, 0.62, 0.00)),
                  (:ext, "externality on nominal", RGBf(0.00, 0.62, 0.45)),
                  (:mut, "mutual (NN − RR)", RGBf(0.80, 0.47, 0.65))]
        for (key, lbl, col) in series
            pts = [(t, summary_all[(arm, c, t)]) for t in TS if haskey(summary_all, (arm, c, t))]
            isempty(pts) && continue
            xs = [p[1] for p in pts]
            ys = [100 * getfield(p[2], key).mean / p[2].base[1] for p in pts]
            es = [100 * getfield(p[2], key).sem / p[2].base[1] for p in pts]
            band!(ax, xs, ys .- 1.96 .* es, ys .+ 1.96 .* es, color=(col, 0.18))
            scatterlines!(ax, xs, ys, color=col, label=lbl)
        end
        hlines!(ax, [0.0], color=:gray, linestyle=:dash)
        i == length(BUDGETS) && axislegend(ax, position=:lt, framevisible=false, labelsize=11)
    end
    Label(fig[0, :], "Symmetric intent mismatch: robustness effects as % of baseline cost ($arm)", fontsize=17)
    save(joinpath(OUT_DIR, "dose_response_pct_$(arm).png"), fig)
    println("Saved dose_response_pct_$(arm).png")
end
println("Done.")
