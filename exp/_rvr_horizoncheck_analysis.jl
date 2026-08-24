#=
Horizon robustness check: do the headline effects survive doubling the planning
horizon (h=10 -> h=20)?

Compares, at c=5, t in {0.0, 1.0}:
  h=10: rvr_symintent_full_noobs_c5_t<t>      (default horizon)
  h=20: rvr_horizoncheck_noobs_h20_t<t>
via the same role-pooled paired effects as the surface analysis
(ownR / ownN / ext / mut, in % of that condition's RR baseline mean).

Run from repo root:
    julia --project=. exp/_rvr_horizoncheck_analysis.jl
=#

using Statistics
using Printf

include("./SenateTrajectoryAnalysis.jl")
STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)

const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_horizoncheck"
mkpath(OUT_DIR)

const C = 5
const CELLS = Dict(
    :RR => Regex("(?=.*p1nm_$(C)_)(?=.*p2nm_$(C)_)"),
    :NRc => Regex("(?=.*p1t_non_robust)(?=.*p2nm_$(C)_)"),
    :cNR => Regex("(?=.*p1nm_$(C)_)(?=.*p2t_non_robust)"),
    :NN => Regex("(?=.*p1t_non_robust)(?=.*p2t_non_robust)"),
)
const CONDITIONS = [
    (t=0.0, h=10, dir="rvr_symintent_full_noobs_c5_t0.0"),
    (t=0.0, h=20, dir="rvr_horizoncheck_noobs_h20_t0.0"),
    (t=1.0, h=10, dir="rvr_symintent_full_noobs_c5_t1.0"),
    (t=1.0, h=20, dir="rvr_horizoncheck_noobs_h20_t1.0"),
]

data = Dict{Tuple{Float64, Int, Symbol}, Dict{Int, NamedTuple}}()
for cond in CONDITIONS
    dir = joinpath(RUN_ROOT, cond.dir)
    isdir(dir) || (println("MISSING $dir"); continue)
    for (cell, pat) in CELLS
        Base.invokelatest(STA.load_and_analyze_senate_solution_files;
            directory=dir, file_pattern=pat)
        d = Dict{Int, NamedTuple}()
        for e in STA.SENATE_TRAJECTORY_TRACKER.entries
            d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
            d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
            (isnothing(d1) || isnothing(d2)) && continue
            d[e.random_seed] = (p1=d1.total, p2=d2.total)
        end
        data[(cond.t, cond.h, cell)] = d
    end
    println("loaded t=$(cond.t) h=$(cond.h): ", join(["$c=$(length(data[(cond.t, cond.h, c)]))" for c in keys(CELLS)], " "))
end

p1f = v -> v.p1
p2f = v -> v.p2

function pooled_pct(t, h, specs)
    all(haskey(data, (t, h, s[1])) && haskey(data, (t, h, s[2])) for s in specs) ||
        return (mean=NaN, sem=NaN, n=0)
    rr = data[(t, h, :RR)]
    isempty(rr) && return (mean=NaN, sem=NaN, n=0)
    base = mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
    d = Float64[]
    for (from, to, ffrom, fto) in specs
        ks = intersect(keys(data[(t, h, from)]), keys(data[(t, h, to)]))
        append!(d, [(fto(data[(t, h, to)][k]) - ffrom(data[(t, h, from)][k])) / base * 100 for k in ks])
    end
    isempty(d) && return (mean=NaN, sem=NaN, n=0)
    (mean=mean(d), sem=std(d) / sqrt(length(d)), n=length(d))
end

const EFFECTS = [
    (:ownR, "own robustness effect vs robust opp", [(:RR, :NRc, p1f, p1f), (:RR, :cNR, p2f, p2f)]),
    (:ownN, "own robustness effect vs nominal opp", [(:NRc, :NN, p2f, p2f), (:cNR, :NN, p1f, p1f)]),
    (:ext, "externality of opp robustness on nominal", [(:NN, :NRc, p1f, p1f), (:NN, :cNR, p2f, p2f)]),
    (:mut, "mutual effect (NN - RR)", [(:RR, :NN, p1f, p1f), (:RR, :NN, p2f, p2f)]),
]

open(joinpath(OUT_DIR, "horizoncheck_report.txt"), "w") do io
    for out in (stdout, io)
        println(out, "Horizon check, c=5 noobs, % of that condition's RR baseline ('*' = 95% CI excludes 0)")
        @printf(out, "%-6s %-6s |", "t", "h")
        for (key, _, _) in EFFECTS
            @printf(out, " %14s", key)
        end
        println(out)
        for cond in CONDITIONS
            @printf(out, "%-6.1f %-6d |", cond.t, cond.h)
            for (_, _, specs) in EFFECTS
                r = pooled_pct(cond.t, cond.h, specs)
                if isfinite(r.mean)
                    sig = abs(r.mean) > 1.96 * r.sem ? "*" : " "
                    @printf(out, "  %+7.2f%%%s n=%d", r.mean, sig, r.n)
                else
                    @printf(out, " %14s", "--")
                end
            end
            println(out)
        end
        # raw RR baselines so % values are comparable across horizons
        println(out)
        for cond in CONDITIONS
            rr = get(data, (cond.t, cond.h, :RR), Dict{Int, NamedTuple}())
            isempty(rr) && continue
            base = mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
            @printf(out, "RR baseline mean cost  t=%.1f h=%d: %.4f\n", cond.t, cond.h, base)
        end
    end
end
println("Wrote horizoncheck_report.txt")
