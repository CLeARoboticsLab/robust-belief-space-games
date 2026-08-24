#=
Difference heatmaps for the RvR nature-cost grid — the quantities with signal:

  1. Matchup advantage: mean over seeds of (P1 total - P2 total) within the SAME
     run. Same-game pairing cancels seed noise. Positive = the c1 player pays
     more than the c2 player. Antisymmetric by construction; diverging palette.
  2. Social cost vs mutual-robust: per-seed paired mean of (P1+P2)/2 at (c1,c2)
     minus (P1+P2)/2 at (5,5). Positive = welfare worse than both-max-robust.

Cell labels show mean with a '*' when the paired 95% CI excludes 0.

Run from repo root:
    julia --project=. exp/_rvr_grid_diff_heatmaps.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const COSTS = [5, 25, 125, 625]
const ARMS = [
    ("noobs", "./exp/senate/outputs/runs/rvr_nature_grid_sym_noobs"),
    ("obs",   "./exp/senate/outputs/runs/rvr_nature_grid_sym_obs"),
]
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_grid"
mkpath(OUT_DIR)

# per-seed totals cache built by exp/_rvr_grid_pct_heatmaps.jl
const ALL_ARMS = deserialize(joinpath(OUT_DIR, "grid_totals_cache.dat"))

"mean, CI-significance flag over a vector of per-seed values"
function mstat(v)
    n = length(v)
    n == 0 && return (mean=NaN, sig=false)
    m = mean(v)
    half = n > 1 ? 1.96 * std(v) / sqrt(n) : Inf
    return (mean=m, sig=abs(m) > half)
end

function labeled_heat!(fig, pos, means, sigs, title;
        colormap, colorrange, cbar_label)
    ax = Axis(fig[pos[1], pos[2]],
        title=title,
        xlabel="P1 nature cost c1 (higher = less robust)",
        ylabel="P2 nature cost c2 (higher = less robust)",
        xticks=(1:4, string.(COSTS)), yticks=(1:4, string.(COSTS)))
    hm = heatmap!(ax, 1:4, 1:4, means; colormap=colormap, colorrange=colorrange)
    lo, hi = colorrange
    for i in 1:4, j in 1:4
        v = means[i, j]
        isfinite(v) || continue
        frac = clamp((v - lo) / (hi - lo), 0, 1)
        txt = @sprintf("%+.2f%s", v, sigs[i, j] ? "*" : "")
        text!(ax, i, j; text=txt, align=(:center, :center),
            color=abs(frac - 0.5) > 0.3 ? :white : :black, fontsize=13)
    end
    Colorbar(fig[pos[1], pos[2] + 1], hm; label=cbar_label)
    return ax
end

for (arm, dir) in ARMS
    println("\n############ ARM: $arm ############")
    cells = ALL_ARMS[arm]

    # 1. matchup advantage: within-run P1 - P2
    adv_m = fill(NaN, 4, 4); adv_s = fill(false, 4, 4)
    for (i, c1) in enumerate(COSTS), (j, c2) in enumerate(COSTS)
        s = mstat([v.p1 - v.p2 for v in values(cells[(c1, c2)])])
        adv_m[i, j] = s.mean; adv_s[i, j] = s.sig
    end

    # 2. social cost vs (5,5), paired by seed
    base = Dict(seed => (v.p1 + v.p2) / 2 for (seed, v) in cells[(COSTS[1], COSTS[1])])
    soc_m = fill(NaN, 4, 4); soc_s = fill(false, 4, 4)
    for (i, c1) in enumerate(COSTS), (j, c2) in enumerate(COSTS)
        diffs = Float64[]
        for (seed, v) in cells[(c1, c2)]
            haskey(base, seed) || continue
            push!(diffs, (v.p1 + v.p2) / 2 - base[seed])
        end
        s = mstat(diffs)
        soc_m[i, j] = s.mean; soc_s[i, j] = s.sig
    end

    amax = maximum(abs, filter(isfinite, vec(adv_m)))
    smax = maximum(abs, filter(isfinite, vec(soc_m)))

    fig = Figure(size=(1450, 560))
    labeled_heat!(fig, (1, 1), adv_m, adv_s,
        "Matchup advantage: P1 cost − P2 cost (same game)";
        colormap=Reverse(:RdBu), colorrange=(-amax, amax),
        cbar_label="Δ cost  (+ = P1 pays more)")
    labeled_heat!(fig, (1, 3), soc_m, soc_s,
        "Social cost vs mutual-robust (5,5), seed-paired";
        colormap=:Reds, colorrange=(0.0, smax),
        cbar_label="Δ mean cost per player vs (5,5)")
    Label(fig[0, :],
        "RvR grid ($arm): paired difference maps ('*' = 95% CI excludes 0, n = 25 seeds)",
        fontsize=17)
    save(joinpath(OUT_DIR, "diff_heatmaps_$(arm).png"), fig)
    println("Saved diff_heatmaps_$(arm).png")
end

println("Done.")
