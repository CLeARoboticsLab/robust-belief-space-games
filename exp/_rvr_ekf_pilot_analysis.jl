#=
EKF-fix disentangling pilot analysis (obs arm, c=5 cells, 5 seeds).

The pilot reran v2's exact config (opponent-blind beliefs, gt drift 1.0,
obstacle weight 8.0) on POST-EKF-fix code. Per cell, paired per-seed:
  pilot - v2    = effect of the EKF calibration fix alone
  pilot - nomm  = effect of the remaining belief mismatch (blind vs correct)
The two should roughly sum to the observed nomm - v2 gap of ~ -17..-23.

Run from repo root:
    julia --project=. exp/_rvr_ekf_pilot_analysis.jl
=#

using Statistics
using Printf

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files
const STA = SenateTrajectoryAnalysis

const DIRS = Dict(
    :pilot => "./exp/senate/outputs/runs/rvr_ekfpilot_obs",
    :v2    => "./exp/senate/outputs/runs/rvr_nature_grid_v2_obs",
    :nomm  => "./exp/senate/outputs/runs/rvr_nature_grid_nomm_obs",
)
const CELLS = [
    ("(5,5)",   Regex("p1nm_5_p2nm_5_")),
    ("(NR,5)",  Regex("p1t_non_robust_p2nm_5_")),
    ("(5,NR)",  Regex("p1nm_5_p2t_non_robust")),
    ("(NR,NR)", Regex("p1t_non_robust_p2t_non_robust")),
]

function load_cell(dir, pat)
    load_and_analyze_senate_solution_files(directory=dir, file_pattern=pat)
    d = Dict{Int, NamedTuple}()
    for e in SENATE_TRAJECTORY_TRACKER.entries
        d1 = STA._decompose_entry_costs(e, 1)
        d2 = STA._decompose_entry_costs(e, 2)
        (isnothing(d1) || isnothing(d2)) && continue
        d[e.random_seed] = (p1=d1.total, p2=d2.total)
    end
    return d
end

function pdiff(x, y, f)
    ks = sort(collect(intersect(keys(x), keys(y))))
    isempty(ks) && return (mean=NaN, lo=NaN, hi=NaN, n=0)
    d = [f(x[k]) - f(y[k]) for k in ks]
    (mean=mean(d), lo=minimum(d), hi=maximum(d), n=length(d))
end

for (clabel, pat) in CELLS
    data = Dict(k => load_cell(dir, pat) for (k, dir) in DIRS)
    m1, m2 = mean([v.p1 for v in values(data[:pilot])]), mean([v.p2 for v in values(data[:pilot])])
    @printf("\n%-8s pilot: n=%d  P1=%8.3f  P2=%8.3f\n", clabel, length(data[:pilot]), m1, m2)
    for (lbl, ref) in (("EKF fix alone      (pilot - v2)  ", :v2),
                       ("belief mismatch    (pilot - nomm)", :nomm))
        d1 = pdiff(data[:pilot], data[ref], v -> v.p1)
        d2 = pdiff(data[:pilot], data[ref], v -> v.p2)
        @printf("  %s  dP1=%+8.3f [%+8.3f, %+8.3f]  dP2=%+8.3f [%+8.3f, %+8.3f]  n=%d\n",
            lbl, d1.mean, d1.lo, d1.hi, d2.mean, d2.lo, d2.hi, d1.n)
    end
end
println("\nDone.")
