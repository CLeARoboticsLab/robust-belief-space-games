#=
Unified symmetric-intent surface: robustness effects over (signed intent error t) x
(robustness budget c), noobs arm, 50 seeds.

The "clean argument" figure family from one homogeneous experiment
(rvr_symintent_full_noobs_c<c>_t<t> dirs, 4 cells each):
  - surface heatmaps (t x c) of ownR / ownN / ext / mut, in % of that
    condition's mutual-robust baseline cost, '*' = paired 95% CI excludes 0
  - budget-frontier line plot: ownR% vs t, one line per c
Caches per-seed cell totals in symintent_surface_cache.dat (delete to reload).

Run from repo root:
    julia --project=. exp/_rvr_symintent_surface.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const TS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
const CS = [5, 25, 125, 625, 3125, 15625]
const PREFIX = "rvr_symintent_full_noobs"
const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_symintent"
const CACHE = joinpath(OUT_DIR, "symintent_surface_cache.dat")
mkpath(OUT_DIR)

cell_patterns(c) = Dict(
    :RR => Regex("(?=.*p1nm_$(c)_)(?=.*p2nm_$(c)_)"),
    :NRc => Regex("(?=.*p1t_non_robust)(?=.*p2nm_$(c)_)"),
    :cNR => Regex("(?=.*p1nm_$(c)_)(?=.*p2t_non_robust)"),
    :NN => Regex("(?=.*p1t_non_robust)(?=.*p2t_non_robust)"),
)

function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)
    out = Dict{Tuple{Int, Float64, Symbol}, Dict{Int, NamedTuple}}()
    for c in CS, t in TS
        dir = joinpath(RUN_ROOT, "$(PREFIX)_c$(c)_t$(t)")
        isdir(dir) || (println("MISSING $dir"); continue)
        for (cell, pat) in cell_patterns(c)
            Base.invokelatest(STA.load_and_analyze_senate_solution_files;
                directory=dir, file_pattern=pat)
            d = Dict{Int, NamedTuple}()
            for e in STA.SENATE_TRAJECTORY_TRACKER.entries
                d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
                d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
                (isnothing(d1) || isnothing(d2)) && continue
                d[e.random_seed] = (p1=d1.total, p2=d2.total)
            end
            out[(c, t, cell)] = d
        end
        println("loaded c=$c t=$t")
    end
    serialize(CACHE, out)
    return out
end

data = isfile(CACHE) ? deserialize(CACHE) : build_cache()
println("Cell totals loaded ($(isfile(CACHE) ? "cache" : "fresh")).")

p1f = v -> v.p1
p2f = v -> v.p2

"Role-pooled paired diffs in % of the condition's RR baseline mean."
function pooled_pct(c, t, specs)
    all(haskey(data, (c, t, s[1])) && haskey(data, (c, t, s[2])) for s in specs) ||
        return (mean=NaN, sem=NaN, n=0)
    rr = data[(c, t, :RR)]
    isempty(rr) && return (mean=NaN, sem=NaN, n=0)
    base = mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
    d = Float64[]
    for (from, to, ffrom, fto) in specs
        ks = intersect(keys(data[(c, t, from)]), keys(data[(c, t, to)]))
        append!(d, [(fto(data[(c, t, to)][k]) - ffrom(data[(c, t, from)][k])) / base * 100 for k in ks])
    end
    isempty(d) && return (mean=NaN, sem=NaN, n=0)
    (mean=mean(d), sem=std(d) / sqrt(length(d)), n=length(d))
end

const EFFECTS = [
    (:ownR, "own robustness effect vs robust opp",
        c -> [(:RR, :NRc, p1f, p1f), (:RR, :cNR, p2f, p2f)]),
    (:ownN, "own robustness effect vs nominal opp",
        c -> [(:NRc, :NN, p2f, p2f), (:cNR, :NN, p1f, p1f)]),
    (:ext, "externality of opp robustness on nominal",
        c -> [(:NN, :NRc, p1f, p1f), (:NN, :cNR, p2f, p2f)]),
    (:mut, "mutual effect (NN − RR)",
        c -> [(:RR, :NN, p1f, p1f), (:RR, :NN, p2f, p2f)]),
]

# ---- surface heatmaps -------------------------------------------------------
fig = Figure(size=(1650, 1150))
for (idx, (key, title, specf)) in enumerate(EFFECTS)
    means = fill(NaN, length(TS), length(CS))
    sigs = fill(false, length(TS), length(CS))
    ns = fill(0, length(TS), length(CS))
    for (i, t) in enumerate(TS), (j, c) in enumerate(CS)
        r = pooled_pct(c, t, specf(c))
        means[i, j] = r.mean
        ns[i, j] = r.n
        isfinite(r.mean) && (sigs[i, j] = abs(r.mean) > 1.96 * r.sem)
    end
    vmax = maximum(abs, filter(isfinite, vec(means)); init=1e-9)
    vmax = vmax == 0 ? 1.0 : vmax
    row, col = fldmod1(idx, 2)
    ax = Axis(fig[row, 2col - 1],
        title=title,
        xlabel="signed intent error t  (− = misread as hostile, + = misread as aligned)",
        ylabel="robustness budget c (lower = more robust)",
        xticks=(1:length(TS), string.(TS)), yticks=(1:length(CS), string.(CS)))
    hm = heatmap!(ax, 1:length(TS), 1:length(CS), means;
        colormap=Reverse(:RdBu), colorrange=(-vmax, vmax))
    for i in 1:length(TS), j in 1:length(CS)
        v = means[i, j]
        isfinite(v) || continue
        frac = clamp((v + vmax) / (2vmax), 0, 1)
        text!(ax, i, j;
            text=@sprintf("%+.2f%s", v, sigs[i, j] ? "*" : ""),
            align=(:center, :center),
            color=abs(frac - 0.5) > 0.35 ? :white : :black, fontsize=10)
    end
    Colorbar(fig[row, 2col], hm; label="% of baseline cost")
end
Label(fig[0, :],
    "Symmetric intent error x robustness budget (noobs, 50 seeds): paired effects, % of mutual-robust baseline ('*' = 95% CI excludes 0)",
    fontsize=16)
save(joinpath(OUT_DIR, "symintent_surface.png"), fig)
println("Saved symintent_surface.png")

# ---- budget-frontier lines: ownR% vs t, one line per c ----------------------
fig2 = Figure(size=(950, 620))
ax = Axis(fig2[1, 1],
    title="Own robustness payoff vs signed model error, by budget",
    xlabel="signed intent error t", ylabel="own effect vs robust opp, % of baseline")
cols = cgrad(:viridis, length(CS), categorical=true)
for (j, c) in enumerate(CS)
    pts = [(t, pooled_pct(c, t, EFFECTS[1][3](c))) for t in TS]
    pts = [(t, r) for (t, r) in pts if isfinite(r.mean)]
    isempty(pts) && continue
    xs = [p[1] for p in pts]
    ys = [p[2].mean for p in pts]
    es = [1.96 * p[2].sem for p in pts]
    band!(ax, xs, ys .- es, ys .+ es, color=(cols[j], 0.15))
    scatterlines!(ax, xs, ys, color=cols[j], label="c=$c")
end
hlines!(ax, [0.0], color=:gray, linestyle=:dash)
vlines!(ax, [0.0], color=:gray, linestyle=:dot)
axislegend(ax, position=:lt, framevisible=false)
save(joinpath(OUT_DIR, "symintent_budget_frontier.png"), fig2)
println("Saved symintent_budget_frontier.png")

# ---- text table -------------------------------------------------------------
open(joinpath(OUT_DIR, "symintent_surface_report.txt"), "w") do io
    for (key, title, specf) in EFFECTS
        println(io, "== $key: $title (% of baseline) ==")
        @printf(io, "%8s |", "t\\c")
        for c in CS
            @printf(io, " %9d", c)
        end
        println(io)
        for t in TS
            @printf(io, "%8.2f |", t)
            for c in CS
                r = pooled_pct(c, t, specf(c))
                isfinite(r.mean) ? @printf(io, " %+8.2f%%", r.mean) : @printf(io, " %9s", "--")
            end
            println(io)
        end
        println(io)
    end
end
println("Wrote symintent_surface_report.txt")
println("Done.")
