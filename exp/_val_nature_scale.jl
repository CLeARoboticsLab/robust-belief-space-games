#=
Validation run for the nature-cost normalization (cost.jl nature_cost_scale).

Usage (from repo root):
    julia --project=. exp/_val_nature_scale.jl <multiplier> <experiment_name>

Runs ONE seed (1001) of the RvR grid cell (c, c), no obstacle, with the
normalized nature cost, into exp/senate/outputs/runs/<experiment_name>.
Compare against the corresponding cell of rvr_nature_grid_sym_noobs.
=#

c1 = parse(Int, ARGS[1])
name = ARGS[2]
c2 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : c1

include(joinpath(@__DIR__, "senate", "versions", "test_drift_sweep.jl"))

run_rvr_nature_grid_sweep(cores=1, num_seeds=1, offset=1000,
    p1_multipliers=[c1], p2_multipliers=[c2],
    experiment_name=name)
