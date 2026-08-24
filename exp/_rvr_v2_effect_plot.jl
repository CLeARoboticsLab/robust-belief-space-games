#=
Headline effect figure for the v2 RvR grid: paired per-seed % cost differences
relative to the MUTUAL-NOMINAL baseline (NR, NR), from grid_summary.dat.

Two panels (noobs / obs). x-axis: the varied player's nature-cost level, from
least robust (15625) to most robust (5). Two curves per panel, P1's executed
total cost as seed-paired % change vs (NR, NR):
  - opponent effect: (NR, x) vs (NR, NR)   [opponent becomes more robust]
  - own effect:      (x, NR) vs (NR, NR)   [P1 itself becomes more robust]
Error bars: +- 1 std of the paired per-seed % differences.

Run from repo root (after exp/_rvr_grid_v2_analysis.jl):
    julia --project=. exp/_rvr_v2_effect_plot.jl
=#

using Statistics
using Serialization
using CairoMakie

const SUMMARY = "./exp/senate/outputs/analysis/rvr_grid_v2/grid_summary.dat"
const OUT = "./exp/senate/outputs/analysis/rvr_grid_v2/effect_curves.png"
const XLEVELS = Any[15625, 3125, 625, 125, 25, 5]   # least -> most robust
const XLABELS = string.(XLEVELS)

summary_all = deserialize(SUMMARY)

function seed_totals(cells, l1, l2)
    s = cells[(l1, l2)]
    return Dict(zip(s.seeds, s.p1_totals))
end

function paired_pct_curve(cells, cellof)
    base = seed_totals(cells, :NR, :NR)
    ms, ss = Float64[], Float64[]
    for x in XLEVELS
        other = seed_totals(cells, cellof(x)...)
        ks = sort(collect(intersect(keys(base), keys(other))))
        d = filter(isfinite, [(other[k] / base[k] - 1) * 100 for k in ks])
        push!(ms, mean(d)); push!(ss, std(d))
    end
    return ms, ss
end

fig = Figure(size=(1150, 480))
for (col, arm) in enumerate(["noobs", "obs"])
    cells = summary_all[arm]
    opp_m, opp_s = paired_pct_curve(cells, x -> (:NR, x))
    own_m, own_s = paired_pct_curve(cells, x -> (x, :NR))

    ax = Axis(fig[1, col],
        title=arm == "noobs" ? "No obstacle" : "Obstacle (w = 8)",
        xlabel="nature-cost level of the varied player (right = more robust)",
        ylabel=col == 1 ? "P1 cost Δ% vs (NR, NR) baseline" : "",
        xticks=(1:length(XLEVELS), XLABELS))
    hlines!(ax, [0.0], color=(:black, 0.4), linestyle=:dash)
    xs = 1:length(XLEVELS)
    for (m, s, lbl, clr) in ((opp_m, opp_s, "opponent more robust  (NR, x)", :crimson),
                             (own_m, own_s, "self more robust  (x, NR)", :steelblue))
        errorbars!(ax, xs, m, s, color=clr, whiskerwidth=8)
        scatterlines!(ax, xs, m, color=clr, label=lbl)
    end
    axislegend(ax, position=arm == "noobs" ? :lt : :lb, framevisible=false)
end
Label(fig[0, :],
    "RvR v2: paired per-seed % cost effect of adding robustness vs mutual-nominal (25 seeds, +-1 SD of paired Δ%)",
    fontsize=16)
save(OUT, fig)
println("Saved $OUT")
