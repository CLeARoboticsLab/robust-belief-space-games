#=
Single combined Δ% heatmap per arm: own executed cost vs the mutual-robust
(5,5) baseline, axes = own vs opponent nature cost, roles pooled by symmetry
(P1 at (a,b) with P2 at (b,a); per-(role,seed) paired ratios, n = 50).

Reads grid_totals_cache.dat (built by exp/_rvr_grid_pct_heatmaps.jl).

Run from repo root:
    julia --project=. exp/_rvr_grid_combined_pct.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const COSTS = [5, 25, 125, 625]
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_grid"
const ALL_ARMS = deserialize(joinpath(OUT_DIR, "grid_totals_cache.dat"))

"own-cost samples for (own=a, opp=b), keyed (role, seed)"
function own_samples(cells, a, b)
    out = Dict{Tuple{Int, Int}, Float64}()
    for (seed, v) in cells[(a, b)]; out[(1, seed)] = v.p1; end
    for (seed, v) in cells[(b, a)]; out[(2, seed)] = v.p2; end
    return out
end

function pct_grid(cells)
    base = own_samples(cells, COSTS[1], COSTS[1])
    means = fill(NaN, 4, 4); sigs = fill(false, 4, 4)
    for (i, a) in enumerate(COSTS), (j, b) in enumerate(COSTS)
        s = own_samples(cells, a, b)
        d = [(s[k] / base[k] - 1) * 100 for k in intersect(keys(s), keys(base))]
        isempty(d) && continue
        m = mean(d)
        half = length(d) > 1 ? 1.96 * std(d) / sqrt(length(d)) : Inf
        means[i, j] = m; sigs[i, j] = abs(m) > half
    end
    return means, sigs
end

arms = ["noobs", "obs"]
grids = Dict(arm => pct_grid(ALL_ARMS[arm]) for arm in arms)
vmax = maximum(maximum(abs, filter(isfinite, vec(g[1]))) for g in values(grids))

function render(arms, grids, vmax)
fig = Figure(size=(1350, 560))
hm = nothing
for (col, arm) in enumerate(arms)
    means, sigs = grids[arm]
    ax = Axis(fig[1, col],
        title=(arm == "noobs" ? "No obstacle" : "With obstacle"),
        xlabel="own nature cost c (higher = less robust)",
        ylabel="opponent nature cost c (higher = less robust)",
        xticks=(1:4, string.(COSTS)), yticks=(1:4, string.(COSTS)))
    hm = heatmap!(ax, 1:4, 1:4, means;
        colormap=Reverse(:RdBu), colorrange=(-vmax, vmax))
    for i in 1:4, j in 1:4
        v = means[i, j]
        isfinite(v) || continue
        frac = clamp((v + vmax) / (2vmax), 0, 1)
        text!(ax, i, j;
            text=@sprintf("%+.1f%%%s", v, sigs[i, j] ? "*" : ""),
            align=(:center, :center),
            color=abs(frac - 0.5) > 0.35 ? :white : :black, fontsize=14)
    end
end
Colorbar(fig[1, 3], hm; label="own cost Δ% vs mutual-robust (red = costlier)")
Label(fig[0, :],
    "RvR grid: own executed cost, seed-paired Δ% vs mutual-robust (5,5) — roles pooled by symmetry ('*' = 95% CI excludes 0, n = 50)",
    fontsize=16)
save(joinpath(OUT_DIR, "combined_pct_heatmap.png"), fig)
println("Saved combined_pct_heatmap.png")
end

render(arms, grids, vmax)
