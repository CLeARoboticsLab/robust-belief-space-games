#=
Analyze the force-zero-nature pilot: fz vs cnr vs nn (see _rvr_forcezero_pilot.jl).
Per-seed P1 cost diffs and step-1 executed-control diffs for each pairing.
Verdict: fz==cnr!=nn -> structure confirmed; fz==nn -> residual hedge mattered.

Run from repo root (after all three phases finish):
    julia --project=. exp/_rvr_forcezero_analysis.jl
=#

using Statistics
using Printf
using LinearAlgebra

include("./SenateTrajectoryAnalysis.jl")
const STA = SenateTrajectoryAnalysis

const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_plateau"

function load_cell(name)
    Base.invokelatest(STA.load_and_analyze_senate_solution_files;
        directory=joinpath(RUN_ROOT, name), file_pattern=r"")
    out = Dict{Int, NamedTuple}()
    for e in STA.SENATE_TRAJECTORY_TRACKER.entries
        d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
        isnothing(d1) && continue
        u1 = [reduce(vcat, [Vector{Float64}(x) for x in s.controls[1].blocks])[1:6]
              for s in e.solution_history[1]]
        gt = [[Vector{Float64}(x) for x in b.blocks] for b in e.gt_state_history]
        nat = nothing
        if !isnothing(e.nature_diagnostics_history) && haskey(e.nature_diagnostics_history, 1) &&
           !isnothing(e.nature_diagnostics_history[1])
            nat = [isempty(r) ? NaN : Float64(r[1].nature_control_norm)
                   for r in e.nature_diagnostics_history[1]]
        end
        out[e.random_seed] = (cost=d1.total, u1=u1, gt=gt, nat=nat)
    end
    println("$name: $(length(out)) seeds")
    return out
end

cells = Dict(p => load_cell("rvr_fz_pilot_$p") for p in ("fz", "cnr", "nn"))

function compare(a, b, label)
    ks = sort(collect(intersect(keys(cells[a]), keys(cells[b]))))
    isempty(ks) && (println("$label: NO COMMON SEEDS"); return)
    cd = [cells[a][k].cost - cells[b][k].cost for k in ks]
    ud = [norm(cells[a][k].u1[1] - cells[b][k].u1[1]) for k in ks]
    td = [mean(norm(cells[a][k].gt[end][s] - cells[b][k].gt[end][s]) for s in 1:3) for k in ks]
    @printf("%-12s (n=%2d)  P1 cost diff: mean %+9.5f  max|.| %9.5f   step1 ctrl diff: mean %8.5f   final traj diff: mean %8.5f\n",
        label, length(ks), mean(cd), maximum(abs.(cd)), mean(ud), mean(td))
end

println("\n=== force-zero-nature pilot comparison ===")
compare("fz", "cnr", "fz vs cnr")
compare("fz", "nn", "fz vs nn")
compare("cnr", "nn", "cnr vs nn")

for p in ("fz", "cnr")
    nats = [mean(filter(isfinite, v.nat)) for v in values(cells[p]) if !isnothing(v.nat)]
    isempty(nats) || @printf("%-4s mean nature control norm: %.3g\n", p, mean(nats))
end
