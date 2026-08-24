#=
Percent-delta cost heatmaps for the RvR nature-cost grid.

Each cell: seed-paired mean of (cost(c1,c2)/cost(5,5) - 1) * 100 per player.
Diverging palette centered at 0 (red = costlier than mutual-robust, blue =
cheaper). '*' = paired 95% CI excludes 0.

Caches per-seed totals in grid_totals_cache.dat so re-renders are instant.

Run from repo root:
    julia --project=. exp/_rvr_grid_pct_heatmaps.jl
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
const CACHE = joinpath(OUT_DIR, "grid_totals_cache.dat")
mkpath(OUT_DIR)

function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)
    all_arms = Dict{String, Dict{Tuple{Int, Int}, Dict{Int, NamedTuple}}}()
    for (arm, dir) in ARMS
        cells = Dict{Tuple{Int, Int}, Dict{Int, NamedTuple}}()
        for c1 in COSTS, c2 in COSTS
            Base.invokelatest(STA.load_and_analyze_senate_solution_files;
                directory=dir, file_pattern=Regex("p1nm_$(c1)_p2nm_$(c2)_p1oc"))
            d = Dict{Int, NamedTuple}()
            for e in STA.SENATE_TRAJECTORY_TRACKER.entries
                d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
                d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
                (isnothing(d1) || isnothing(d2)) && continue
                d[e.random_seed] = (p1=d1.total, p2=d2.total)
            end
            cells[(c1, c2)] = d
        end
        all_arms[arm] = cells
    end
    serialize(CACHE, all_arms)
    return all_arms
end

all_arms = isfile(CACHE) ? deserialize(CACHE) : build_cache()
println("Totals loaded ($(isfile(CACHE) ? "cache" : "fresh")).")

function pct_stats(cells, getter)
    base = Dict(seed => getter(v) for (seed, v) in cells[(COSTS[1], COSTS[1])])
    means = fill(NaN, 4, 4); sigs = fill(false, 4, 4)
    for (i, c1) in enumerate(COSTS), (j, c2) in enumerate(COSTS)
        d = Float64[]
        for (seed, v) in cells[(c1, c2)]
            haskey(base, seed) || continue
            push!(d, (getter(v) / base[seed] - 1) * 100)
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
        xlabel="P1 nature cost c1 (higher = less robust)",
        ylabel="P2 nature cost c2 (higher = less robust)",
        xticks=(1:4, string.(COSTS)), yticks=(1:4, string.(COSTS)))
    hm = heatmap!(ax, 1:4, 1:4, means; colormap=Reverse(:RdBu), colorrange=(-vmax, vmax))
    for i in 1:4, j in 1:4
        v = means[i, j]
        isfinite(v) || continue
        frac = clamp((v + vmax) / (2vmax), 0, 1)
        text!(ax, i, j;
            text=@sprintf("%+.1f%%%s", v, sigs[i, j] ? "*" : ""),
            align=(:center, :center),
            color=abs(frac - 0.5) > 0.35 ? :white : :black, fontsize=14)
    end
    return hm
end

for (arm, _) in ARMS
    cells = all_arms[arm]
    p1_m, p1_s = pct_stats(cells, v -> v.p1)
    p2_m, p2_s = pct_stats(cells, v -> v.p2)
    vmax = max(maximum(abs, filter(isfinite, vec(p1_m))),
               maximum(abs, filter(isfinite, vec(p2_m))))

    fig = Figure(size=(1350, 560))
    pct_panel!(fig, (1, 1), p1_m, p1_s, "P1 cost, Δ% vs (5,5)", vmax)
    hm = pct_panel!(fig, (1, 2), p2_m, p2_s, "P2 cost, Δ% vs (5,5)", vmax)
    Colorbar(fig[1, 3], hm; label="Δ% vs mutual-robust (red = costlier)")
    Label(fig[0, :],
        "RvR grid ($arm): executed cost, seed-paired % change vs mutual-robust (5,5)  ('*' = 95% CI excludes 0)",
        fontsize=17)
    save(joinpath(OUT_DIR, "pct_heatmaps_$(arm).png"), fig)
    println("Saved pct_heatmaps_$(arm).png")
end

println("Done.")
