#=
Empirical solver runtime: nominal vs robust (by nature budget c), extracted
from the archived per-step plan_cost of the clean-point runs
(rvr_symintent_full_noobs_c{c}_t0.0). Each player's solve() wall time and
iteration count is recorded per replanning step on the server; NN cells give
nominal timings, RR cells give robust-at-budget-c timings (both players).

Caveat: times were measured with 50 concurrent workers on the server, so
absolute values include contention noise — but it is uniform across cells,
so the nominal-vs-robust comparison is fair. Medians/IQR are used so the
one-time JIT compilation outlier per worker doesn't skew anything.

Run from repo root:
    julia --project=. exp/_rvr_runtime_analysis.jl
=#

using Statistics
using Printf
using Serialization
using CairoMakie

const CS = [5, 25, 125, 625, 3125, 15625]
const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_runtime"
const CACHE = joinpath(OUT_DIR, "runtime_cache.dat")
mkpath(OUT_DIR)

function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)
    out = Dict{Any, Vector{NamedTuple}}()   # :nominal or c => [(t=solve_time, it=iters), ...]
    out[:nominal] = NamedTuple[]
    for c in CS
        out[c] = NamedTuple[]
        dir = joinpath(RUN_ROOT, "rvr_symintent_full_noobs_c$(c)_t0.0")
        isdir(dir) || (println("MISSING $dir"); continue)
        for (cell, pat, sink1, sink2) in (
                (:NN, Regex("(?=.*p1t_non_robust)(?=.*p2t_non_robust)"), :nominal, :nominal),
                (:RR, Regex("(?=.*p1nm_$(c)_)(?=.*p2nm_$(c)_)"), c, c))
            Base.invokelatest(STA.load_and_analyze_senate_solution_files;
                directory=dir, file_pattern=pat)
            nsteps = 0
            for e in STA.SENATE_TRAJECTORY_TRACKER.entries
                for (pidx, sink) in ((1, sink1), (2, sink2))
                    haskey(e.cost_history, pidx) || continue
                    for pc in e.cost_history[pidx]
                        hasproperty(pc, :solve_time) || continue
                        push!(out[sink], (t=pc.solve_time, it=pc.solver_iterations))
                        nsteps += 1
                    end
                end
            end
            println("c=$c $cell: $nsteps player-solves collected")
        end
    end
    serialize(CACHE, out)
    return out
end

times = isfile(CACHE) ? deserialize(CACHE) : build_cache()

qstats(v) = (med=median(v), q25=quantile(v, 0.25), q75=quantile(v, 0.75),
             mean=mean(v), p95=quantile(v, 0.95), n=length(v))

nom_t = qstats([x.t for x in times[:nominal]])
nom_i = qstats([Float64(x.it) for x in times[:nominal]])
rob_t = Dict(c => qstats([x.t for x in times[c]]) for c in CS if !isempty(times[c]))
rob_i = Dict(c => qstats([Float64(x.it) for x in times[c]]) for c in CS if !isempty(times[c]))

# ---- figure ----------------------------------------------------------------
fig = Figure(size=(1250, 520))
cols = Makie.wong_colors()
xs = collect(1:length(CS))

ax1 = Axis(fig[1, 1], title="Per-solve wall time",
    xlabel="robust nature budget c", ylabel="seconds per solve() call",
    xticks=(xs, string.(CS)))
med = [rob_t[c].med for c in CS]
lo = [rob_t[c].q25 for c in CS]
hi = [rob_t[c].q75 for c in CS]
band!(ax1, xs, lo, hi, color=(cols[1], 0.2))
scatterlines!(ax1, xs, med, color=cols[1], label="robust (median, IQR band)")
hlines!(ax1, [nom_t.med], color=cols[6], linewidth=2, label="nominal (median)")
hspan!(ax1, nom_t.q25, nom_t.q75, color=(cols[6], 0.15))
axislegend(ax1, position=:rt, framevisible=false)

ax2 = Axis(fig[1, 2], title="Solver iterations",
    xlabel="robust nature budget c", ylabel="iterations per solve",
    xticks=(xs, string.(CS)))
medi = [rob_i[c].med for c in CS]
loi = [rob_i[c].q25 for c in CS]
hii = [rob_i[c].q75 for c in CS]
band!(ax2, xs, loi, hii, color=(cols[1], 0.2))
scatterlines!(ax2, xs, medi, color=cols[1], label="robust (median, IQR band)")
hlines!(ax2, [nom_i.med], color=cols[6], linewidth=2, label="nominal (median)")
hspan!(ax2, nom_i.q25, nom_i.q75, color=(cols[6], 0.15))
axislegend(ax2, position=:rt, framevisible=false)

Label(fig[0, :],
    "Empirical solver cost, nominal vs robust — archived per-step solve() times, clean-point runs (t=0), 50 seeds x 9 steps x 2 players",
    fontsize=14)
save(joinpath(OUT_DIR, "runtime_vs_c.png"), fig)
println("Saved runtime_vs_c.png")

# ---- report ----------------------------------------------------------------
open(joinpath(OUT_DIR, "runtime_report.txt"), "w") do io
    for out in (stdout, io)
        println(out, "Empirical per-solve runtime, server archives (50 concurrent workers; medians robust to JIT outliers)")
        @printf(out, "%-16s %10s %10s %10s %10s %10s %8s %10s\n",
            "config", "median s", "q25", "q75", "mean", "p95", "n", "overhead")
        @printf(out, "%-16s %10.3f %10.3f %10.3f %10.3f %10.3f %8d %10s\n",
            "nominal", nom_t.med, nom_t.q25, nom_t.q75, nom_t.mean, nom_t.p95, nom_t.n, "--")
        for c in CS
            r = rob_t[c]
            @printf(out, "%-16s %10.3f %10.3f %10.3f %10.3f %10.3f %8d %+9.1f%%\n",
                "robust c=$c", r.med, r.q25, r.q75, r.mean, r.p95, r.n,
                (r.med - nom_t.med) / nom_t.med * 100)
        end
        println(out, "\nIterations per solve:")
        @printf(out, "%-16s %10.1f (median)   IQR [%.1f, %.1f]\n", "nominal", nom_i.med, nom_i.q25, nom_i.q75)
        for c in CS
            r = rob_i[c]
            @printf(out, "%-16s %10.1f (median)   IQR [%.1f, %.1f]\n", "robust c=$c", r.med, r.q25, r.q75)
        end
    end
end
println("Wrote runtime_report.txt")
