#=
Verification: decompose ownR into Δpreference / Δcontrol for the intent sweep
extremes (c=5, t=-1, 0, +1). Prediction from the bias mechanism: the preference
benefit is roughly flat; the control premium scales with fight intensity, so it
should be large at t=-1 (escalated fight) and small at t=+1 (de-escalated).

Run: julia --project=. exp/_rvr_intent_decomp_check.jl
=#
using Statistics, Printf

include("./SenateTrajectoryAnalysis.jl")
STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)

const C = 5
const RUN_ROOT = "./exp/senate/outputs/runs"
cell_patterns(c) = Dict(
    :RR => Regex("(?=.*p1nm_$(c)_)(?=.*p2nm_$(c)_)"),
    :NRc => Regex("(?=.*p1t_non_robust)(?=.*p2nm_$(c)_)"),
    :cNR => Regex("(?=.*p1nm_$(c)_)(?=.*p2t_non_robust)"),
)

mstats(v) = (mean=mean(v), sem=std(v) / sqrt(length(v)))

@printf("%6s | %18s %18s %18s | %16s\n", "t", "Δtotal", "Δpref", "Δctrl", "RR ctrl level")
for t in [-1.0, 0.0, 1.0]
    dir = joinpath(RUN_ROOT, "rvr_symintent_full_noobs_c5_t$(t)")
    isdir(dir) || (println("MISSING $dir"); continue)
    cells = Dict{Symbol, Dict{Int, NamedTuple}}()
    for (cell, pat) in cell_patterns(C)
        Base.invokelatest(STA.load_and_analyze_senate_solution_files;
            directory=dir, file_pattern=pat)
        d = Dict{Int, NamedTuple}()
        for e in STA.SENATE_TRAJECTORY_TRACKER.entries
            d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
            d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
            (isnothing(d1) || isnothing(d2)) && continue
            d[e.random_seed] = (p1=d1, p2=d2)
        end
        cells[cell] = d
    end
    dp = Float64[]; dc = Float64[]; dt_ = Float64[]
    for (cellB, f) in ((:NRc, :p1), (:cNR, :p2))
        for k in intersect(keys(cells[:RR]), keys(cells[cellB]))
            a, b = getfield(cells[:RR][k], f), getfield(cells[cellB][k], f)
            push!(dp, b.preference - a.preference)
            push!(dc, b.control - a.control)
            push!(dt_, b.total - a.total)
        end
    end
    rrctrl = mean(vcat([v.p1.control for v in values(cells[:RR])],
                       [v.p2.control for v in values(cells[:RR])]))
    T, P, Cc = mstats(dt_), mstats(dp), mstats(dc)
    @printf("%6.1f | %+8.3f ± %5.3f  %+8.3f ± %5.3f  %+8.3f ± %5.3f | %10.2f\n",
        t, T.mean, 1.96 * T.sem, P.mean, 1.96 * P.sem, Cc.mean, 1.96 * Cc.sem, rrctrl)
end
println("Done.")
