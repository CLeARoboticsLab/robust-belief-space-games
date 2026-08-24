#=
Paired-difference violin at d=0: per budget c, the PER-SEED paired cost
difference NN.p1 - cNR.p1 (value of P1 being robust vs a nominal opponent),
in % of that condition's pooled RR baseline. Same overlay-violin style as
_rvr_violin_d0.jl. Seeds are common random numbers across cells, so this is
the distribution the significance stars actually come from.

Run from repo root:  julia --project=. exp/_rvr_violin_d0_paired.jl
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

function iqr_filter(costs::Vector{Float64})
    length(costs) < 4 && return costs
    q1 = quantile(costs, 0.25); q3 = quantile(costs, 0.75); iqr = q3 - q1
    return filter(c -> q1 - 1.5*iqr <= c <= q3 + 1.5*iqr, costs)
end

"Per-seed paired diff NN - cNR in % of the condition's pooled RR baseline."
function paired_diffs(c)
    nn = grid[(c, D, :NN)]
    rnr = grid[(c, D, :cNR)]
    rr = grid[(c, D, :RR)]
    base = mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
    ks = intersect(keys(nn), keys(rnr))
    return [(nn[k].p1 - rnr[k].p1) / base * 100 for k in ks]
end

n_sv = length(CS)
fig = Figure(size=(max(900, 130 * n_sv), 650),
    backgroundcolor=:transparent, fontsize=22)
update_theme!(fonts = (; regular = "Palatino Linotype",
                          bold = "Palatino Linotype",
                          italic = "Palatino Linotype"))
ax = Axis(fig[1, 1],
    backgroundcolor=:transparent,
    xlabel = "Nature's Control Effort Cost (c)",
    ylabel = "Paired Cost Advantage of Robust P1 (%)",
    title  = "d = 0, per-seed NN − (R-vs-NR) difference — 50 seeds",
    xlabelsize = 32, ylabelsize = 32, titlesize = 24,
    xticklabelsize = 26, yticklabelsize = 26,
    xticks = (collect(1:n_sv), string.(CS)),
    xticklabelrotation = π/12,
    topspinevisible = false, rightspinevisible = false,
    xgridvisible = false, ygridvisible = false,
)
xlims!(ax, 0.4, n_sv + 0.6)

for (idx, c) in enumerate(CS)
    col = sweep_color(idx, n_sv)
    diffs = iqr_filter(paired_diffs(c))
    isempty(diffs) && continue
    violin!(ax, fill(idx, length(diffs)), diffs;
        color=(col, 0.55), width=0.62, strokewidth=2, strokecolor=:black)
    scatter!(ax, fill(idx, length(diffs)) .+ randn(length(diffs)).*0.035, diffs;
        color=(col, 0.5), markersize=6)
    m = mean(diffs)
    lines!(ax, [idx - 0.14, idx + 0.14], [m, m]; color=:black, linewidth=2)
end
hlines!(ax, [0.0]; color=RGBAf(0,0,0,0.4), linewidth=1.5, linestyle=:dash)

filename = joinpath(OUT_DIR, "rvr_violin_d0_paired")
save(filename * ".png", fig, px_per_unit=3)
save(filename * ".pdf", fig)
println("Saved $(filename).png")
