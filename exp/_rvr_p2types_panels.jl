#=
Original-design view of sweeps #1-#3: one panel per ACTUAL P2 type
(nominal, robust c2=5/25/125/625), x = P1's belief drift d, one line per
ego budget c1. Value = paired per-seed cost diff of P1 being robust vs
non-robust against that opponent, in % of the ego's matched RR baseline.

Data coverage:
  - vs nominal: full grid, c1 in {5,25,125,625,3125,15625}, d in +-{0.05,0.1,0.2} and 0
  - vs robust c2: matched line (c1 == c2) has the full d range; cross-budget
    lines (c1 in {5,125}, c1 != c2) only exist at d = +-0.1 (p2types sweep)

Requires p1drift_cache.dat and p2types_cache.dat (run those analyses first).

Run from repo root:
    julia --project=. exp/_rvr_p2types_panels.jl
=#

using Statistics
using Serialization
using CairoMakie

const CS = [5, 25, 125, 625, 3125, 15625]
const DS = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
const MIXED_C1S = [5, 25, 125, 625, 15625]
const MIXED_DS = DS   # fill sweeps completed the cross-budget cells over the full d range
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_p2types"
const GRID_CACHE = "./exp/senate/outputs/analysis/rvr_p1drift/p1drift_cache.dat"
const MIXED_CACHE = joinpath(OUT_DIR, "p2types_cache.dat")

isfile(GRID_CACHE) || error("missing $GRID_CACHE - run exp/_rvr_p1drift_analysis.jl first")
isfile(MIXED_CACHE) || error("missing $MIXED_CACHE - run exp/_rvr_p2types_analysis.jl first")
grid = deserialize(GRID_CACHE)     # (c, d, cell) => Dict(seed => (p1, p2))
mixed = deserialize(MIXED_CACHE)   # (c1, c2, d)  => Dict(seed => (p1, p2))

function baseline(c1, d)
    rr = grid[(c1, d, :RR)]
    mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
end

function paired_pct(a, b, base)
    ks = intersect(keys(a), keys(b))
    isempty(ks) && return nothing
    vals = [(a[k].p1 - b[k].p1) / base * 100 for k in ks]
    (mean=mean(vals), sem=std(vals) / sqrt(length(vals)), n=length(vals))
end

"Value of P1's robustness (budget c1) at drift d vs opponent type opp (:N or c2)."
function ego_value(c1, opp, d)
    base = baseline(c1, d)
    if opp == :N
        haskey(grid, (c1, d, :NN)) || return nothing
        return paired_pct(grid[(c1, d, :NN)], grid[(c1, d, :cNR)], base)
    end
    c2 = opp
    haskey(grid, (c2, d, :NRc)) || return nothing
    rc = c1 == c2 ? get(grid, (c1, d, :RR), nothing) : get(mixed, (c1, c2, d), nothing)
    isnothing(rc) && return nothing
    return paired_pct(grid[(c2, d, :NRc)], rc, base)
end

const OPPS = [:N, 5, 25, 125, 625]
opp_title(o) = o == :N ? "vs NOMINAL P2" : "vs ROBUST P2 (c2=$o)"
c1s_for(o) = o == :N ? CS : sort(unique(vcat([o], [c for c in MIXED_C1S if c != o])))
ds_for(o, c1) = (o == :N || c1 == o) ? DS : MIXED_DS

const COLS = Dict(c => Makie.wong_colors()[i] for (i, c) in enumerate(CS))

fig = Figure(size=(1650, 950))
axes = Axis[]
for (i, o) in enumerate(OPPS)
    r, cl = divrem(i - 1, 3)
    ax = Axis(fig[r + 1, cl + 1], title=opp_title(o),
        xlabel="P1 belief drift d  (d<0: misread as aggressive, d>0: as cooperative)",
        ylabel="value of P1 robust, % of matched RR baseline")
    push!(axes, ax)
    for c1 in c1s_for(o)
        pts = [(d, ego_value(c1, o, d)) for d in ds_for(o, c1)]
        pts = [(d, r) for (d, r) in pts if !isnothing(r)]
        isempty(pts) && continue
        xs = [p[1] for p in pts]
        ys = [p[2].mean for p in pts]
        es = [1.96 * p[2].sem for p in pts]
        band!(ax, xs, ys .- es, ys .+ es, color=(COLS[c1], 0.15))
        scatterlines!(ax, xs, ys, color=COLS[c1],
            linestyle=(o != :N && c1 != o) ? :dash : :solid,
            label="ego c1=$c1" * ((o != :N && c1 != o) ? " (cross-budget)" : ""))
    end
    hlines!(ax, [0.0], color=:gray, linestyle=:dash)
    vlines!(ax, [0.0], color=:gray, linestyle=:dot)
    axislegend(ax, position=:lt, framevisible=false, labelsize=10)
end
linkyaxes!(axes...)
Label(fig[0, :],
    "Ego (mismatched P1) robustness value by ACTUAL opponent type — panels: P2 type, lines: ego budget c1 — noobs, 50 seeds, 95% bands",
    fontsize=15)
save(joinpath(OUT_DIR, "p2types_panels_by_opponent.png"), fig)
println("Saved p2types_panels_by_opponent.png")
