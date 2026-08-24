#=
High-c plateau diagnostic (clean point t=0, from local archives).

Question: why does robustness keep its full advantage as c -> 15625, where the
internal nature adversary should be priced out of doing anything?

Test: compare seed-matched cNR (P1 robust at c, P2 nominal) vs NN (both nominal)
runs at c=5 (active hedge) and c=15625 (supposedly inert):
  1. P1's internal nature control norm per replanning step (from archived
     nature diagnostics) - is nature actually inert at high c?
  2. ||P1 executed control (cNR) - P1 executed control (NN)|| per step, same seed
     - does the robust policy still act differently?
  3. Ground-truth trajectory divergence (mean over senators) per step.
If (1) collapses at high c but (2)/(3) do not, the plateau is structural
(the nature player's presence changes which local equilibrium the solver
lands on), not hedging against a live adversary.

Run from repo root:
    julia --project=. exp/_rvr_plateau_diag.jl
=#

using Statistics
using Printf
using LinearAlgebra
using Serialization
using CairoMakie

const CS = [5, 15625]
const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_plateau"
const CACHE = joinpath(OUT_DIR, "plateau_cache.dat")
mkpath(OUT_DIR)

function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)

    function load_cell(dir, pat)
        Base.invokelatest(STA.load_and_analyze_senate_solution_files;
            directory=dir, file_pattern=pat)
        # keep only plain data per seed to keep the cache small
        out = Dict{Int, NamedTuple}()
        for e in STA.SENATE_TRAJECTORY_TRACKER.entries
            # field access only (.blocks): methods loaded by the include are too new
            # for this function's world age, so avoid BlockArrays API calls
            gt = [[Vector{Float64}(x) for x in b.blocks] for b in e.gt_state_history]
            u1 = [reduce(vcat, [Vector{Float64}(x) for x in s.controls[1].blocks])[1:6]
                  for s in e.solution_history[1]]
            nat = nothing
            if !isnothing(e.nature_diagnostics_history) && haskey(e.nature_diagnostics_history, 1) &&
               !isnothing(e.nature_diagnostics_history[1])
                nat = [[Float64(step.nature_control_norm) for step in replan]
                       for replan in e.nature_diagnostics_history[1]]
            end
            out[e.random_seed] = (gt=gt, u1=u1, nat=nat)
        end
        return out
    end

    cache = Dict{Any, Dict{Int, NamedTuple}}()
    for c in CS
        dir = joinpath(RUN_ROOT, "rvr_symintent_full_noobs_c$(c)_t0.0")
        cache[(c, :cNR)] = load_cell(dir, Regex("(?=.*p1nm_$(c)_)(?=.*p2t_non_robust)"))
        println("c=$c cNR: $(length(cache[(c, :cNR)])) seeds")
    end
    dir5 = joinpath(RUN_ROOT, "rvr_symintent_full_noobs_c5_t0.0")
    cache[:NN] = load_cell(dir5, Regex("(?=.*p1t_non_robust)(?=.*p2t_non_robust)"))
    println("NN: $(length(cache[:NN])) seeds")
    serialize(CACHE, cache)
    return cache
end

cache = isfile(CACHE) ? deserialize(CACHE) : build_cache()

nn = cache[:NN]
nsteps = minimum(length(v.u1) for v in values(nn))

"mean +- sem over seeds of f(cnr_entry, nn_entry) at each step 1:nsteps"
function step_series(c, f)
    cnr = cache[(c, :cNR)]
    ks = sort(collect(intersect(keys(cnr), keys(nn))))
    m = Float64[]; s = Float64[]
    for t in 1:nsteps
        vals = [f(cnr[k], nn[k], t) for k in ks]
        vals = filter(isfinite, vals)
        push!(m, mean(vals)); push!(s, std(vals) / sqrt(length(vals)))
    end
    (m=m, s=s, n=length(ks))
end

nature_norm(e_c, e_n, t) = isnothing(e_c.nat) ? NaN :
    (t <= length(e_c.nat) && !isempty(e_c.nat[t]) ? e_c.nat[t][1] : NaN)
ctrl_diff(e_c, e_n, t) = norm(e_c.u1[t] - e_n.u1[t])
traj_div(e_c, e_n, t) = mean(norm(e_c.gt[t][s] - e_n.gt[t][s]) for s in 1:length(e_c.gt[t]))

series = Dict((c, which) => step_series(c, f)
    for c in CS
    for (which, f) in ((:nat, nature_norm), (:ctrl, ctrl_diff), (:traj, traj_div)))

# ---- figure ----------------------------------------------------------------
fig = Figure(size=(1500, 500))
cols = Makie.wong_colors()
titles = [
    (:nat,  "P1's internal nature control norm", "||u_nature|| (first planned step)"),
    (:ctrl, "Executed-control divergence vs NN", "||u1_robust - u1_nominal||, same seed"),
    (:traj, "Ground-truth trajectory divergence vs NN", "mean over senators ||x_robust - x_nominal||"),
]
for (i, (which, ttl, ylab)) in enumerate(titles)
    ax = Axis(fig[1, i], title=ttl, xlabel="replanning step", ylabel=ylab)
    for (k, c) in enumerate(CS)
        sr = series[(c, which)]
        xs = 1:length(sr.m)
        band!(ax, xs, sr.m .- 1.96 .* sr.s, sr.m .+ 1.96 .* sr.s, color=(cols[k], 0.2))
        scatterlines!(ax, xs, sr.m, color=cols[k], label="c=$c")
    end
    axislegend(ax, position=:rt, framevisible=false)
end
Label(fig[0, :],
    "High-c plateau diagnostic: seed-matched cNR vs NN at the clean point (t=0), 50 seeds",
    fontsize=15)
save(joinpath(OUT_DIR, "plateau_diag.png"), fig)
println("Saved plateau_diag.png")

# ---- report ----------------------------------------------------------------
open(joinpath(OUT_DIR, "plateau_report.txt"), "w") do io
    for out in (stdout, io)
        println(out, "High-c plateau diagnostic (clean point, cNR vs NN, seed-matched)")
        for c in CS
            nat = series[(c, :nat)]; ct = series[(c, :ctrl)]; tj = series[(c, :traj)]
            @printf(out, "\nc=%d (n=%d pairs)\n", c, ct.n)
            @printf(out, "  nature control norm  mean over steps: %.3g   step1: %.3g   last: %.3g\n",
                mean(nat.m), nat.m[1], nat.m[end])
            @printf(out, "  exec control diff    mean over steps: %.3g   step1: %.3g   last: %.3g\n",
                mean(ct.m), ct.m[1], ct.m[end])
            @printf(out, "  trajectory diff      mean over steps: %.3g   final: %.3g\n",
                mean(tj.m), tj.m[end])
        end
        r = mean(series[(5, :nat)].m) / mean(series[(15625, :nat)].m)
        @printf(out, "\nnature norm ratio c=5 / c=15625: %.1fx\n", r)
        rc = mean(series[(15625, :ctrl)].m) / mean(series[(5, :ctrl)].m)
        @printf(out, "exec-control divergence at c=15625 as fraction of c=5: %.2f\n", rc)
    end
end
println("Wrote plateau_report.txt")
