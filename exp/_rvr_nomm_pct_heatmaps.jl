#=
Percent-delta heatmaps for the no-mismatch control vs v2 grid (7x7, 25 seeds).

Each cell: seed-paired mean of (nomm/v2 - 1) * 100 per player. Diverging
palette centered at 0 (red = nomm costlier, blue = nomm cheaper), '*' = paired
95% CI excludes 0. Same style as pct_heatmaps_<arm>.png.

Caches per-seed totals in nomm_totals_cache.dat so re-renders are instant.

Run from repo root:
    julia --project=. exp/_rvr_nomm_pct_heatmaps.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const LEVELS = ["5", "25", "125", "625", "3125", "15625", "NR"]
const ARMS = [
    ("noobs", "./exp/senate/outputs/runs/rvr_nature_grid_nomm_noobs",
              "./exp/senate/outputs/runs/rvr_nature_grid_v2_noobs"),
    ("obs",   "./exp/senate/outputs/runs/rvr_nature_grid_nomm_obs",
              "./exp/senate/outputs/runs/rvr_nature_grid_v2_obs"),
]
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_nomm"
const CACHE = joinpath(OUT_DIR, "nomm_totals_cache.dat")
mkpath(OUT_DIR)

function cell_pattern(a, b)
    a == "NR" && b == "NR" && return Regex("p1t_non_robust_p2t_non_robust")
    a == "NR" && return Regex("p1t_non_robust_p2nm_$(b)_")
    b == "NR" && return Regex("p1nm_$(a)_p2t_non_robust")
    return Regex("p1nm_$(a)_p2nm_$(b)_")
end

function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)
    out = Dict{String, Dict{Tuple{String, String}, Dict{Int, NamedTuple}}}()
    for (arm, dir_nomm, dir_v2) in ARMS, (tag, dir) in (("nomm", dir_nomm), ("v2", dir_v2))
        cells = Dict{Tuple{String, String}, Dict{Int, NamedTuple}}()
        for a in LEVELS, b in LEVELS
            Base.invokelatest(STA.load_and_analyze_senate_solution_files;
                directory=dir, file_pattern=cell_pattern(a, b))
            d = Dict{Int, NamedTuple}()
            for e in STA.SENATE_TRAJECTORY_TRACKER.entries
                d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
                d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
                (isnothing(d1) || isnothing(d2)) && continue
                d[e.random_seed] = (p1=d1.total, p2=d2.total)
            end
            cells[(a, b)] = d
        end
        out["$(arm)_$(tag)"] = cells
    end
    serialize(CACHE, out)
    return out
end

data = isfile(CACHE) ? deserialize(CACHE) : build_cache()
println("Totals loaded ($(isfile(CACHE) ? "cache" : "fresh")).")

"Seed-paired (nomm/v2 - 1)*100 per cell for one player."
function pct_stats(nomm, v2, getter)
    means = fill(NaN, 7, 7); sigs = fill(false, 7, 7)
    for (i, a) in enumerate(LEVELS), (j, b) in enumerate(LEVELS)
        d = Float64[]
        for (seed, v) in nomm[(a, b)]
            haskey(v2[(a, b)], seed) || continue
            push!(d, (getter(v) / getter(v2[(a, b)][seed]) - 1) * 100)
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

for (arm, _, _) in ARMS
    nomm, v2 = data["$(arm)_nomm"], data["$(arm)_v2"]
    p1_m, p1_s = pct_stats(nomm, v2, v -> v.p1)
    p2_m, p2_s = pct_stats(nomm, v2, v -> v.p2)
    vmax = max(maximum(abs, filter(isfinite, vec(p1_m))),
               maximum(abs, filter(isfinite, vec(p2_m))))
    vmax = vmax == 0 ? 1.0 : vmax

    fig = Figure(size=(1500, 620))
    pct_panel!(fig, (1, 1), p1_m, p1_s, "P1 cost, Δ% (nomm vs v2)", vmax)
    hm = pct_panel!(fig, (1, 2), p2_m, p2_s, "P2 cost, Δ% (nomm vs v2)", vmax)
    Colorbar(fig[1, 3], hm; label="Δ% vs v2 (red = no-mismatch costlier)")
    Label(fig[0, :],
        "No-mismatch control vs v2 ($arm): seed-paired % cost change  ('*' = 95% CI excludes 0)",
        fontsize=17)
    save(joinpath(OUT_DIR, "pct_diff_heatmap_$(arm).png"), fig)
    println("Saved pct_diff_heatmap_$(arm).png")
end

println("Done.")
