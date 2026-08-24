#=
Force-zero-nature pilot (structure vs hedging discriminator), run locally.

Three cells, same seeds, clean point (d=0), RvR noobs config:
  fz  - P1 robust c=15625, nature control PINNED to 0 (RBSG_FORCE_ZERO_NATURE=1)
  cnr - P1 robust c=15625, unforced (local rerun of the sweep cell)
  nn  - both nominal
If fz == cnr != nn: the equilibrium shift comes from the nature block's mere
presence in the solver (structure). If fz == nn: the tiny residual nature
action was load-bearing after all.

Usage (one phase per process; set the env var ONLY for fz):
    julia --project=. exp/_rvr_forcezero_pilot.jl fz|cnr|nn
=#

include(joinpath(@__DIR__, "senate", "versions", "test_drift_sweep.jl"))

phase = ARGS[1]
@assert phase in ("fz", "cnr", "nn")
if phase == "fz"
    @assert get(ENV, "RBSG_FORCE_ZERO_NATURE", "0") == "1" "fz phase needs RBSG_FORCE_ZERO_NATURE=1"
else
    @assert get(ENV, "RBSG_FORCE_ZERO_NATURE", "0") != "1" "$phase phase must run WITHOUT the flag"
end

common = _rvr_pilot_common(; experiment_name="rvr_fz_pilot_$(phase)",
    obstacle_weight=0.0, control_cost_weight=2.0,
    num_seeds=15, offset=1000, cores=4, override=false)

if phase == "nn"
    run_parallel_sweep(; common..., p1_type=[non_robust], p2_type=[non_robust])
else
    run_parallel_sweep(; common..., p1_type=robust, p2_type=[non_robust],
        p1_nature_multiplier=[15625])
end
println("PHASE $phase DONE")
