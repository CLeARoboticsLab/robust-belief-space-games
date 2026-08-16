#=
Structured-error pilot analysis (options 4, 5, 7; 5 seeds each), sharing the
3-cell layout (5,5) / (NR,5) [P2 robust] / (NR,NR):

  bias   b in {0.1, 0.3}: P2's real sensor offset by +b on every dim, unmodeled
  intent mid/swap:        P1 believes P2's target is [2,2] / [3,1] (true [1,3])
  asymnoise g=4:          only P2's sensor noisy; P2 calibrated, P1 blind

Per condition:
  A. P1 own effect  = P1@(NR,5) - P1@(5,5)    (>0: robustness cheaper for P1)
  B. P2 own effect  = P2@(NR,NR) - P2@(NR,5)  (>0: robustness cheaper for P2)
  C. Externality    = P1@(NR,5) - P1@(NR,NR)  (>0: opponent robustness hurts P1)
  D. Mutual         = per-player (NR,NR) - (5,5)
Reference (clean, calib pilot g=1): A=+0.097  B=+0.193  C=+1.201  D=(-1.105, -1.092)

Run from repo root:
    julia --project=. exp/_rvr_structured_pilot_analysis.jl
=#

using Statistics
using Printf

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files
const STA = SenateTrajectoryAnalysis

const RUN_ROOT = "./exp/senate/outputs/runs"
const BAILOUT_REJ = 26

const CONDITIONS = [
    ("bias b=0.1",      "rvr_bias_pilot_noobs_b0.1"),
    ("bias b=0.3",      "rvr_bias_pilot_noobs_b0.3"),
    ("intent mid [2,2]", "rvr_intent_pilot_noobs_mid"),
    ("intent swap [3,1]", "rvr_intent_pilot_noobs_swap"),
    ("asym noise g=4",  "rvr_asymnoise_pilot_noobs_g4.0"),
]

const CELLS = Dict(
    :RR  => Regex("p1nm_5_p2nm_5_"),
    :NR5 => Regex("p1t_non_robust_p2nm_5_"),
    :NN  => Regex("p1t_non_robust_p2t_non_robust"),
)

function load_cell(dir, pat)
    load_and_analyze_senate_solution_files(directory=dir, file_pattern=pat)
    entries = copy(SENATE_TRAJECTORY_TRACKER.entries)
    out = Dict{Int, NamedTuple}()
    for e in entries
        dec1 = STA._decompose_entry_costs(e, 1)
        dec2 = STA._decompose_entry_costs(e, 2)
        rej = Dict(1 => Float64[], 2 => Float64[])
        if e.cost_history isa Dict
            for pidx in (1, 2)
                ch = get(e.cost_history, pidx, nothing)
                (isnothing(ch) || isempty(ch)) && continue
                for c in ch
                    (c isa NamedTuple) || continue
                    (hasproperty(c, :solver_iterations) && hasproperty(c, :solver_improvement_iterations)) || continue
                    (isnothing(c.solver_iterations) || isnothing(c.solver_improvement_iterations)) && continue
                    push!(rej[pidx], Float64(c.solver_iterations - c.solver_improvement_iterations))
                end
            end
        end
        bail(v) = isempty(v) ? NaN : count(>=(BAILOUT_REJ), v) / length(v)
        out[e.random_seed] = (
            p1 = isnothing(dec1) ? NaN : dec1.total,
            p2 = isnothing(dec2) ? NaN : dec2.total,
            p1_bail = bail(rej[1]), p2_bail = bail(rej[2]),
        )
    end
    return out
end

function paired(label, a::Dict, b::Dict, fa, fb)   # Δ = fb(b) - fa(a) on shared seeds
    ks = sort(collect(intersect(keys(a), keys(b))))
    d = [fb(b[k]) - fa(a[k]) for k in ks]
    @printf("    %-48s mean=%+8.3f  min=%+8.3f  max=%+8.3f  n=%d\n",
        label, mean(d), minimum(d), maximum(d), length(d))
end

cellstats(c::Dict, f) = (v = [f(x) for x in values(c)]; (mean(v), std(v)))

for (label, dirname) in CONDITIONS
    dir = joinpath(RUN_ROOT, dirname)
    isdir(dir) || (println("SKIP $label ($dir missing)"); continue)
    cells = Dict(k => load_cell(dir, pat) for (k, pat) in CELLS)

    println("\n", "="^84)
    println("CONDITION: $label   ($dirname)")
    println("="^84)
    for (k, clabel) in ((:RR, "(5,5)  mutual robust"), (:NR5, "(NR,5) P2 robust"), (:NN, "(NR,NR) mutual nominal"))
        c = cells[k]
        m1, s1 = cellstats(c, x -> x.p1); m2, s2 = cellstats(c, x -> x.p2)
        b1 = mean([x.p1_bail for x in values(c)]); b2 = mean([x.p2_bail for x in values(c)])
        @printf("  %-24s n=%d  P1=%8.3f±%.3f  P2=%8.3f±%.3f  bail=(%.2f, %.2f)\n",
            clabel, length(c), m1, s1, m2, s2, b1, b2)
    end
    println()
    paired("A: P1 own effect   (NR,5)-(5,5), P1", cells[:RR], cells[:NR5], x -> x.p1, x -> x.p1)
    paired("B: P2 own effect   (NR,NR)-(NR,5), P2", cells[:NR5], cells[:NN], x -> x.p2, x -> x.p2)
    paired("C: externality on P1 of P2 robust  (NR,5)-(NR,NR)", cells[:NN], cells[:NR5], x -> x.p1, x -> x.p1)
    paired("D: mutual  (NR,NR)-(5,5), P1", cells[:RR], cells[:NN], x -> x.p1, x -> x.p1)
    paired("D: mutual  (NR,NR)-(5,5), P2", cells[:RR], cells[:NN], x -> x.p2, x -> x.p2)
end
println("\nReference (clean, calib pilot g=1): A=+0.097  B=+0.193  C=+1.201  D=(-1.105, -1.092)")
println("Done.")
