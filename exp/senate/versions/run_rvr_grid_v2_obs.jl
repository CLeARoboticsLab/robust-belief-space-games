#=
WITH-OBSTACLE twin of run_rvr_grid_v2.jl: full v2 grid (6x6 robust +
nominal row/col/corner, 25 seeds, offset 1000, 1225 runs) with obstacle
weight 8.0 at (1.5, 1.5), normalized nature cost.

Run from anywhere:
    julia --project=. exp/senate/versions/run_rvr_grid_v2_obs.jl
=#

# save_file_prefix is relative to cwd, so run from repo root
cd(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "test_drift_sweep.jl"))

run_rvr_nature_grid_expansion(cores=10, num_seeds=25,
    obstacle_weight=8.0,
    experiment_name="rvr_nature_grid_v2_obs")
