#=
Sweep #3 of the one-sided belief-drift design: value of the mismatched ego's
(P1, drift d) robustness as a function of WHO P2 ACTUALLY IS — nominal, or
robust with budget c2 (possibly != P1's own budget c1).

Value vs nominal P2:      P1cost[NN] - P1cost[cNR]          (both in rvr_p1drift_noobs_c{c1}_d{d})
Value vs robust-c2 P2:    P1cost[N,Rc2] - P1cost[Rc1,Rc2]
  - [N,Rc2]   = NRc cell of rvr_p1drift_noobs_c{c2}_d{d}
  - [Rc1,Rc2] = RR cell of rvr_p1drift_noobs_c{c1}_d{d} when c1 == c2, else the
                single-cell dir rvr_p1drift_p2types_noobs_d{d}_p1c{c1}_p2c{c2}
All diffs paired per seed, in % of the ego's own matched-grid RR baseline
(pooled RR mean of rvr_p1drift_noobs_c{c1}_d{d}) so each line has one unit.

Requires p1drift_cache.dat (run exp/_rvr_p1drift_analysis.jl first).

Run from repo root:
    julia --project=. exp/_rvr_p2types_analysis.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const P1_CS = [5, 25, 125, 625, 15625]
const P2_CS = [5, 25, 125, 625]
const DS = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]   # mixed cells now cover the full drift range
const FIG_DS = [-0.1, 0.1]                             # the 2-panel overview figure keeps these
const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_p2types"
const CACHE = joinpath(OUT_DIR, "p2types_cache.dat")
const GRID_CACHE = "./exp/senate/outputs/analysis/rvr_p1drift/p1drift_cache.dat"
mkpath(OUT_DIR)

isfile(GRID_CACHE) || error("missing $GRID_CACHE - run exp/_rvr_p1drift_analysis.jl first")
grid = deserialize(GRID_CACHE)   # keyed (c, d, cell) => Dict(seed => (p1=..., p2=...))

# ---- load mixed-budget RR cells (c1 != c2) ----------------------------------
function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)
    out = Dict{Tuple{Int, Int, Float64}, Dict{Int, NamedTuple}}()
    for d in DS, c1 in P1_CS, c2 in P2_CS
        c1 == c2 && continue
        dir = joinpath(RUN_ROOT, "rvr_p1drift_p2types_noobs_d$(d)_p1c$(c1)_p2c$(c2)")
        isdir(dir) || (println("MISSING $dir"); continue)
        Base.invokelatest(STA.load_and_analyze_senate_solution_files;
            directory=dir, file_pattern=Regex("(?=.*p1nm_$(c1)_)(?=.*p2nm_$(c2)_)"))
        dd = Dict{Int, NamedTuple}()
        for e in STA.SENATE_TRAJECTORY_TRACKER.entries
            d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
            d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
            (isnothing(d1) || isnothing(d2)) && continue
            dd[e.random_seed] = (p1=d1.total, p2=d2.total)
        end
        out[(c1, c2, d)] = dd
        println("loaded mixed cell c1=$c1 c2=$c2 d=$d (n=$(length(dd)))")
    end
    serialize(CACHE, out)
    return out
end

mixed = isfile(CACHE) ? deserialize(CACHE) : build_cache()

# ---- effect helpers ---------------------------------------------------------
"Pooled RR baseline of the ego's matched condition (c1, d)."
function baseline(c1, d)
    rr = grid[(c1, d, :RR)]
    mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
end

"Paired mean/sem of (a[k].p1 - b[k].p1) in % of base."
function paired_pct(a, b, base)
    ks = intersect(keys(a), keys(b))
    isempty(ks) && return (mean=NaN, sem=NaN, n=0)
    vals = [(a[k].p1 - b[k].p1) / base * 100 for k in ks]
    (mean=mean(vals), sem=std(vals) / sqrt(length(vals)), n=length(vals))
end

"Value of P1's robustness (budget c1) vs opponent type opp at drift d."
function ego_value(c1, opp, d)
    base = baseline(c1, d)
    if opp == :N
        return paired_pct(grid[(c1, d, :NN)], grid[(c1, d, :cNR)], base)
    else
        c2 = opp
        rc = c1 == c2 ? grid[(c1, d, :RR)] :
             get(mixed, (c1, c2, d), nothing)
        isnothing(rc) && return (mean=NaN, sem=NaN, n=0)
        return paired_pct(grid[(c2, d, :NRc)], rc, base)
    end
end

const OPPS = [:N, 5, 25, 125, 625]
opp_label(o) = o == :N ? "nominal" : "R(c=$o)"

# ---- figure: panels = d, x = opponent type, lines = ego's c1 ----------------
fig = Figure(size=(1400, 600))
for (panel, d) in enumerate(FIG_DS)
    ax = Axis(fig[1, panel],
        title=d < 0 ? "d = $(d)  (P1 misreads P2 as aggressive)" :
                      "d = +$(d)  (P1 misreads P2 as cooperative)",
        xlabel="who P2 actually is",
        ylabel="value of P1 being robust, % of matched RR baseline",
        xticks=(1:length(OPPS), opp_label.(OPPS)))
    cols = Makie.wong_colors()
    for (k, c1) in enumerate(P1_CS)
        pts = [(j, ego_value(c1, o, d)) for (j, o) in enumerate(OPPS)]
        pts = [(j, r) for (j, r) in pts if isfinite(r.mean)]
        isempty(pts) && continue
        xs = [p[1] for p in pts]
        ys = [p[2].mean for p in pts]
        es = [1.96 * p[2].sem for p in pts]
        band!(ax, xs, ys .- es, ys .+ es, color=(cols[k], 0.15))
        scatterlines!(ax, xs, ys, color=cols[k], label="ego c1=$c1")
    end
    hlines!(ax, [0.0], color=:gray, linestyle=:dash)
    panel == 1 && axislegend(ax, position=:lt, framevisible=false)
end
Label(fig[0, :],
    "Sweep #3: mismatched ego (P1, belief drift d) vs different ACTUAL P2 types, noobs, 50 seeds",
    fontsize=15)
save(joinpath(OUT_DIR, "p2types_vs_opponent.png"), fig)
println("Saved p2types_vs_opponent.png")

# ---- text table -------------------------------------------------------------
open(joinpath(OUT_DIR, "p2types_report.txt"), "w") do io
    for out in (stdout, io)
        println(out, "Sweep #3: ego robustness value vs actual P2 type, % of matched RR baseline ('*' = 95% CI excludes 0)")
        for d in DS
            println(out, "\n-- d = $d --")
            @printf(out, "%10s |", "P2 type")
            for c1 in P1_CS
                @printf(out, " %14s", "ego c1=$c1")
            end
            println(out)
            for o in OPPS
                @printf(out, "%10s |", opp_label(o))
                for c1 in P1_CS
                    r = ego_value(c1, o, d)
                    if isfinite(r.mean)
                        @printf(out, "    %+7.2f%%%s  ", r.mean, abs(r.mean) > 1.96 * r.sem ? "*" : " ")
                    else
                        @printf(out, " %14s", "--")
                    end
                end
                println(out)
            end
        end
    end
end
println("Wrote p2types_report.txt")
