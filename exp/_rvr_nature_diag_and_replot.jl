#=
Two jobs in one load pass over the RvR / RvNR sweeps:

1. Nature + solver diagnostics table across ALL multipliers (incl. c=1):
   - mean planned ‖u_nature‖ (first RH step) for P2 (and P1 in RvR)
   - ‖u_nature‖ · c   (tests the ~1/c equilibrium scaling — flat means 1/c holds)
   - mean solver iterations and improvement iterations (convergence health)
2. Re-render all overlay plots EXCLUDING c=1 (suspected solver convergence issues):
   summary, violin, and cost-component perstep/cumulative figures.

Run from repo root:
    julia --project=. exp/_rvr_nature_diag_and_replot.jl
=#

using Statistics
using Printf

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis
using .SenateTrajectoryAnalysis: _load_rvr_rnr_series

const ALL_M = [1, 2, 5, 10, 25, 50, 125, 250, 625]
const PLOT_M = [2, 5, 10, 25, 50, 125, 250, 625]
const RVR_DIR = "./exp/senate/outputs/merged/rvr_nature_control_sweep"
const RNR_DIR = "./exp/senate/outputs/merged/nature_control_sweep"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_vs_rnr_overlay"

rvr, rnr_r, rnr_nr = _load_rvr_rnr_series(ALL_M;
    rvr_directory=RVR_DIR, rnr_directory=RNR_DIR)

# ---------------------------------------------------------------------------
# 1. Diagnostics table
# ---------------------------------------------------------------------------

function entry_nature_norm(e, pidx)
    ndh = e.nature_diagnostics_history
    (isnothing(ndh) || !(ndh isa Dict)) && return nothing
    dh = get(ndh, pidx, nothing)
    (isnothing(dh) || isempty(dh)) && return nothing
    vals = Float64[]
    for rh in dh
        (isnothing(rh) || isempty(rh)) && continue
        d = rh[1]  # first planning step = the executed one
        hasproperty(d, :nature_control_norm) || continue
        push!(vals, d.nature_control_norm)
    end
    return isempty(vals) ? nothing : mean(vals)
end

function entry_solver_iters(e, pidx)
    (e.cost_history isa Dict) || return (nothing, nothing)
    ch = get(e.cost_history, pidx, nothing)
    (isnothing(ch) || isempty(ch)) && return (nothing, nothing)
    it = Float64[]; imp = Float64[]
    for c in ch
        (c isa NamedTuple) || continue
        if hasproperty(c, :solver_iterations) && !isnothing(c.solver_iterations)
            push!(it, Float64(c.solver_iterations))
        end
        if hasproperty(c, :solver_improvement_iterations) && !isnothing(c.solver_improvement_iterations)
            push!(imp, Float64(c.solver_improvement_iterations))
        end
    end
    return (isempty(it) ? nothing : mean(it), isempty(imp) ? nothing : mean(imp))
end

function summarize_series(entries, pidx)
    norms = Float64[]; iters = Float64[]; imps = Float64[]
    for e in entries
        nn = entry_nature_norm(e, pidx)
        !isnothing(nn) && push!(norms, nn)
        (it, imp) = entry_solver_iters(e, pidx)
        !isnothing(it) && push!(iters, it)
        !isnothing(imp) && push!(imps, imp)
    end
    return (
        n = length(entries),
        n_diag = length(norms),
        mean_norm = isempty(norms) ? NaN : mean(norms),
        mean_iters = isempty(iters) ? NaN : mean(iters),
        mean_imps = isempty(imps) ? NaN : mean(imps),
    )
end

fmt(x) = isnan(x) ? "     n/a" : @sprintf("%8.4f", x)

println("\n" * "="^115)
println("NATURE + SOLVER DIAGNOSTICS  (planned ‖u_nature‖ at first RH step; solver iterations per RH solve)")
println("="^115)
@printf "%6s | %-18s | %4s %6s | %12s %12s | %10s %10s\n" "c" "series" "n" "n_diag" "‖u_nat‖" "‖u_nat‖·c" "iters" "improve"
println("-"^115)
for m in ALL_M
    specs = [
        ("RvR      P2", get(rvr, m, []), 2),
        ("RvR      P1", get(rvr, m, []), 1),
        ("RvNR-Rob P2", get(rnr_r, m, []), 2),
    ]
    for (label, entries, pidx) in specs
        s = summarize_series(entries, pidx)
        @printf "%6d | %-18s | %4d %6d | %12s %12s | %10s %10s\n" m label s.n s.n_diag fmt(s.mean_norm) fmt(s.mean_norm * m) fmt(s.mean_iters) fmt(s.mean_imps)
    end
    println("-"^115)
end

# ---------------------------------------------------------------------------
# 2. Re-render plots without c=1
# ---------------------------------------------------------------------------

println("\n>>> Re-rendering overlay plots without c=1 (multipliers=$(PLOT_M))")
analyze_rvr_vs_rnr_overlay(multipliers=PLOT_M, output_directory=OUT_DIR,
    preloaded=(rvr, rnr_r, rnr_nr))
analyze_rvr_vs_rnr_cost_components(multipliers=PLOT_M, output_directory=OUT_DIR,
    preloaded=(rvr, rnr_r, rnr_nr))

println("\nDone.")
