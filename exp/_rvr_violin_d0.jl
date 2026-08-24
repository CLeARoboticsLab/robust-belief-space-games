#=
Paper-style overlay violin for the RvR design at d=0 (no belief drift), cloned
from _plot_overlay_violin in SenateTrajectoryAnalysis.jl but drawn from the
p1drift cost cache (final cumulative deterministic P1 cost per seed).

Per budget c: R-vs-R violin (solid stroke) and R-vs-NR violin (no stroke,
P1 robust vs nominal P2 = the cNR cell), pooled NN baseline on the right.

Run from repo root:  julia --project=. exp/_rvr_violin_d0.jl
Pass "rnr" to draw ONLY the R-vs-NR violins (centered), saved with _rnr suffix.
=#

using Statistics
using Serialization
using CairoMakie

const CS = [5, 25, 125, 625, 3125, 15625]
const D = 0.0
const GRID_CACHE = "./exp/senate/outputs/analysis/rvr_p1drift/p1drift_cache.dat"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_violin"
mkpath(OUT_DIR)
grid = deserialize(GRID_CACHE)

function sweep_color(idx, n)
    t = n <= 1 ? 0.0 : (idx - 1) / (n - 1)
    stops = [
        (0.0,  RGBf(0.2, 0.4, 1.0)),
        (0.25, RGBf(0.2, 0.8, 0.4)),
        (0.5,  RGBf(0.9, 0.9, 0.2)),
        (0.75, RGBf(1.0, 0.6, 0.2)),
        (1.0,  RGBf(0.9, 0.2, 0.2)),
    ]
    for i in 1:length(stops)-1
        t0, c0 = stops[i]; t1, c1 = stops[i+1]
        if t <= t1
            s = (t - t0) / (t1 - t0)
            return RGBf(c0.r + s*(c1.r-c0.r), c0.g + s*(c1.g-c0.g), c0.b + s*(c1.b-c0.b))
        end
    end
    return stops[end][2]
end
nr_color = RGBf(0.6, 0.6, 0.6)

function iqr_filter(costs::Vector{Float64})
    length(costs) < 4 && return costs
    q1 = quantile(costs, 0.25); q3 = quantile(costs, 0.75); iqr = q3 - q1
    return filter(c -> q1 - 1.5*iqr <= c <= q3 + 1.5*iqr, costs)
end
p1_costs(cell) = Float64[v.p1 for v in values(cell)]

const ONLY_RNR = !isempty(ARGS) && lowercase(ARGS[1]) == "rnr"
n_sv = length(CS)
rvr_offset = -0.22
rnr_offset = ONLY_RNR ? 0.0 : +0.22
violin_width = ONLY_RNR ? 0.62 : 0.40

fig = Figure(size=(max(900, 130 * (n_sv + 1)), 650),
    backgroundcolor=:transparent, fontsize=22)
update_theme!(fonts = (; regular = "Palatino Linotype",
                          bold = "Palatino Linotype",
                          italic = "Palatino Linotype"))
ax = Axis(fig[1, 1],
    backgroundcolor=:transparent,
    xlabel = "Nature's Control Effort Cost (c)",
    ylabel = "Total Cost (Ego, P1)",
    title  = "d = 0 (no model mismatch) — 50 seeds",
    xlabelsize = 32, ylabelsize = 32, titlesize = 24,
    xticklabelsize = 26, yticklabelsize = 26,
    xticks = (collect(1:n_sv+1), vcat(string.(CS), ["NR"])),
    xticklabelrotation = π/12,
    topspinevisible = false, rightspinevisible = false,
    xgridvisible = false, ygridvisible = false,
)
xlims!(ax, 0.4, n_sv + 1.6)

# NN is the same game at every c; merge per seed so the pooled baseline
# has one value per seed, not one per (c, seed).
nn_by_seed = Dict{Int, Float64}()
for c in CS
    haskey(grid, (c, D, :NN)) || continue
    for (s, v) in grid[(c, D, :NN)]
        nn_by_seed[s] = v.p1
    end
end

for (idx, c) in enumerate(CS)
    col = sweep_color(idx, n_sv)
    rvr_costs = ONLY_RNR ? Float64[] : iqr_filter(p1_costs(grid[(c, D, :RR)]))
    rnr_costs = iqr_filter(p1_costs(grid[(c, D, :cNR)]))

    if !isempty(rvr_costs)
        x = idx + rvr_offset
        violin!(ax, fill(x, length(rvr_costs)), rvr_costs;
            color=(col, 0.65), width=violin_width, strokewidth=2, strokecolor=:black)
        scatter!(ax, fill(x, length(rvr_costs)) .+ randn(length(rvr_costs)).*0.025, rvr_costs;
            color=(col, 0.5), markersize=6)
        mr = mean(rvr_costs)
        lines!(ax, [x - 0.10, x + 0.10], [mr, mr]; color=:black, linewidth=2)
    end
    if !isempty(rnr_costs)
        x = idx + rnr_offset
        violin!(ax, fill(x, length(rnr_costs)), rnr_costs;
            color=(col, 0.30), width=violin_width)
        scatter!(ax, fill(x, length(rnr_costs)) .+ randn(length(rnr_costs)).*0.025, rnr_costs;
            color=(col, 0.4), markersize=6)
        mr = mean(rnr_costs)
        lines!(ax, [x - 0.10, x + 0.10], [mr, mr]; color=:black, linewidth=2, linestyle=:dash)
    end
end

pooled_nr = iqr_filter(collect(values(nn_by_seed)))
nr_x = n_sv + 1
nr_mean = isempty(pooled_nr) ? nothing : mean(pooled_nr)
if !isempty(pooled_nr)
    violin!(ax, fill(nr_x, length(pooled_nr)), pooled_nr;
        color=(nr_color, 0.6), width=0.9)
    scatter!(ax, fill(nr_x, length(pooled_nr)) .+ randn(length(pooled_nr)).*0.06, pooled_nr;
        color=(nr_color, 0.5), markersize=7)
    lines!(ax, [nr_x - 0.15, nr_x + 0.15], [nr_mean, nr_mean]; color=:black, linewidth=2)
end
if !isnothing(nr_mean)
    hlines!(ax, [nr_mean]; color=RGBAf(0,0,0,0.4), linewidth=1.5, linestyle=:dash)
end

legend_box = ONLY_RNR ? [
    PolyElement(color=(:gray, 0.30), strokecolor=(:black, 0.0)),
    PolyElement(color=(nr_color, 0.6), strokecolor=(:black, 0.0)),
] : [
    PolyElement(color=(:gray, 0.65), strokecolor=:black, strokewidth=2),
    PolyElement(color=(:gray, 0.30), strokecolor=(:black, 0.0)),
    PolyElement(color=(nr_color, 0.6), strokecolor=(:black, 0.0)),
]
legend_labels = ONLY_RNR ?
    ["R-vs-NR, P1 robust", "NN baseline (P1 cost)"] :
    ["R-vs-R (solid stroke)", "R-vs-NR, P1 robust (no stroke)", "NN baseline (P1 cost)"]
Legend(fig[1, 2], legend_box, legend_labels;
    framevisible=false, labelsize=20)

filename = joinpath(OUT_DIR, ONLY_RNR ? "rvr_violin_d0_rnr" : "rvr_violin_d0")
save(filename * ".png", fig, px_per_unit=3)
save(filename * ".pdf", fig)
println("Saved $(filename).png")
