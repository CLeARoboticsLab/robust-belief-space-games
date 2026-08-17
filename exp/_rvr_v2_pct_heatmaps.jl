#=
Percent-delta heatmaps for the v2 mismatch grid itself (7x7, 25 seeds).

Each cell: seed-paired mean of (cost / cost at (5,5) - 1) * 100 per player,
i.e. how much costlier each cell is than the mutual-robust baseline.
Same style as pct_heatmaps_<arm>.png; '*' = paired 95% CI excludes 0.

Reuses the per-seed totals cached by _rvr_nomm_pct_heatmaps.jl.

Run from repo root:
    julia --project=. exp/_rvr_v2_pct_heatmaps.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const LEVELS = ["5", "25", "125", "625", "3125", "15625", "NR"]
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_symintent"
const CACHE = "./exp/senate/outputs/analysis/rvr_nomm/nomm_totals_cache.dat"
const BASE = ("5", "5")
mkpath(OUT_DIR)

data = deserialize(CACHE)

"Seed-paired (cell/baseline - 1)*100 for one player."
function pct_stats(cells, getter)
    means = fill(NaN, 7, 7); sigs = fill(false, 7, 7)
    base = cells[BASE]
    for (i, a) in enumerate(LEVELS), (j, b) in enumerate(LEVELS)
        d = Float64[]
        for (seed, v) in cells[(a, b)]
            haskey(base, seed) || continue
            push!(d, (getter(v) / getter(base[seed]) - 1) * 100)
        end
        isempty(d) && continue
        m = mean(d)
        half = length(d) > 1 ? 1.96 * std(d) / sqrt(length(d)) : Inf
        means[i, j] = m; sigs[i, j] = abs(m) > half
    end
    return means, sigs
end

function pct_panel!(fig, pos, means, sigs, title, vmax)
    ax = Axis(fig[pos[1], pos[2]],
        title=title,
        xlabel="P1 level c1 (higher = less robust)",
        ylabel="P2 level c2 (higher = less robust)",
        xticks=(1:7, LEVELS), yticks=(1:7, LEVELS))
    hm = heatmap!(ax, 1:7, 1:7, means; colormap=Reverse(:RdBu), colorrange=(-vmax, vmax))
    for i in 1:7, j in 1:7
        v = means[i, j]
        isfinite(v) || continue
        frac = clamp((v + vmax) / (2vmax), 0, 1)
        text!(ax, i, j;
            text=@sprintf("%+.1f%%%s", v, sigs[i, j] ? "*" : ""),
            align=(:center, :center),
            color=abs(frac - 0.5) > 0.35 ? :white : :black, fontsize=11)
    end
    return hm
end

for arm in ["noobs", "obs"]
    cells = data["$(arm)_v2"]
    p1_m, p1_s = pct_stats(cells, v -> v.p1)
    p2_m, p2_s = pct_stats(cells, v -> v.p2)
    vmax = max(maximum(abs, filter(isfinite, vec(p1_m))),
               maximum(abs, filter(isfinite, vec(p2_m))))
    vmax = vmax == 0 ? 1.0 : vmax

    fig = Figure(size=(1500, 620))
    pct_panel!(fig, (1, 1), p1_m, p1_s, "P1 cost, Δ% vs (5,5)", vmax)
    hm = pct_panel!(fig, (1, 2), p2_m, p2_s, "P2 cost, Δ% vs (5,5)", vmax)
    Colorbar(fig[1, 3], hm; label="Δ% vs mutual-robust (5,5)  (red = costlier)")
    Label(fig[0, :],
        "v2 mismatch grid ($arm): seed-paired % cost vs (5,5) baseline  ('*' = 95% CI excludes 0)",
        fontsize=17)
    save(joinpath(OUT_DIR, "pct_heatmap_v2_$(arm).png"), fig)
    println("Saved pct_heatmap_v2_$(arm).png")
end

println("Done.")
