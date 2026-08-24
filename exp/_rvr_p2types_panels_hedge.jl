#=
HEDGE-component version of the panels-by-opponent figure (_rvr_p2types_panels.jl
is the TOTAL version and stays as-is). The ego's non-robust arm is replaced by
an inert-budget ROBUST ego, so both arms share the solver structure and the
equilibrium-selection artifact cancels:

  vs nominal P2:    hedge = P1cost[cNR(inert)] - P1cost[cNR(c1)],       inert = 15625
  vs robust-c2 P2:  hedge = P1cost[R(inert)R(c2)] - P1cost[R(c1)R(c2)]
     (true R(15625)xR(c2) baseline cells from the p2typesmorec sweep)

Units: % of the ego's matched (c1, d) pooled RR baseline, as everywhere else.
Requires p1drift_cache.dat and p2types_cache.dat (full-d fill).

Run from repo root:
    julia --project=. exp/_rvr_p2types_panels_hedge.jl
=#

using Statistics
using Serialization
using CairoMakie

const CS = [5, 25, 125, 625, 3125]        # 15625 omitted vs nominal (== 0 by construction)
const C_INERT = 15625
const DS = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
const MIXED_C1S = [5, 25, 125, 625]
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_p2types"
const GRID_CACHE = "./exp/senate/outputs/analysis/rvr_p1drift/p1drift_cache.dat"
const MIXED_CACHE = joinpath(OUT_DIR, "p2types_cache.dat")

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

"RR-style cell with P1 budget ca vs P2 budget c2 (matched from grid, else mixed)."
rr_cell(ca, c2, d) = ca == c2 ? get(grid, (ca, d, :RR), nothing) :
                                get(mixed, (ca, c2, d), nothing)

"Hedge value of ego budget c1 vs opponent type opp at drift d."
function ego_hedge(c1, opp, d)
    base = baseline(c1, d)
    if opp == :N
        return paired_pct(grid[(C_INERT, d, :cNR)], grid[(c1, d, :cNR)], base)
    end
    c2 = opp
    arm_inert = rr_cell(C_INERT, c2, d)
    arm_c1 = rr_cell(c1, c2, d)
    (isnothing(arm_inert) || isnothing(arm_c1)) && return nothing
    return paired_pct(arm_inert, arm_c1, base)
end

const OPPS = [:N, 5, 25, 125, 625]
opp_title(o) = o == :N ? "vs NOMINAL P2  (baseline: ego R($(C_INERT)))" :
                         "vs ROBUST P2 (c2=$o)  (baseline: ego R($(C_INERT)))"
c1s_for(o) = o == :N ? CS : sort(unique(vcat([o], [c for c in MIXED_C1S if c != o])))

const ALL_CS = [5, 25, 125, 625, 3125, 15625]
const COLS = Dict(c => Makie.wong_colors()[i] for (i, c) in enumerate(ALL_CS))

fig = Figure(size=(1650, 950))
axes = Axis[]
for (i, o) in enumerate(OPPS)
    r, cl = divrem(i - 1, 3)
    ax = Axis(fig[r + 1, cl + 1], title=opp_title(o),
        xlabel="P1 belief drift d  (d<0: misread as aggressive, d>0: as cooperative)",
        ylabel="HEDGE value of P1's budget, % of matched RR baseline")
    push!(axes, ax)
    for c1 in c1s_for(o)
        pts = [(d, ego_hedge(c1, o, d)) for d in DS]
        pts = [(d, r) for (d, r) in pts if !isnothing(r)]
        isempty(pts) && continue
        xs = [p[1] for p in pts]
        ys = [p[2].mean for p in pts]
        es = [1.96 * p[2].sem for p in pts]
        band!(ax, xs, ys .- es, ys .+ es, color=(COLS[c1], 0.15))
        scatterlines!(ax, xs, ys, color=COLS[c1],
            linestyle=(o != :N && c1 != o) ? :dash : :solid,
            label="ego c1=$c1")
    end
    hlines!(ax, [0.0], color=:gray, linestyle=:dash)
    vlines!(ax, [0.0], color=:gray, linestyle=:dot)
    axislegend(ax, position=:lt, framevisible=false, labelsize=10)
end
linkyaxes!(axes...)
Label(fig[0, :],
    "HEDGE component by ACTUAL opponent type (structure artifact cancelled: both arms use the robust solver) — noobs, 50 seeds, 95% bands",
    fontsize=15)
save(joinpath(OUT_DIR, "p2types_panels_hedge.png"), fig)
println("Saved p2types_panels_hedge.png")
