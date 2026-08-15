#=
Launch the WITH-OBSTACLE twin of the R-vs-R grid expansion.

Identical to run_rvr_grid_expansion.jl (6x6 robust grid with 3125/15625,
nominal row/col/corner, 25 seeds, offset 1000) except obstacle weight 8.0
at (1.5, 1.5). Writes into rvr_nature_grid_sym_obs; existing files skip.

Run from anywhere:
    julia --project=. exp/senate/versions/run_rvr_grid_expansion_obs.jl
=#

# save_file_prefix is relative to cwd, so run from repo root
cd(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "test_drift_sweep.jl"))

# cores=6 so it can run alongside the no-obstacle expansion (8 workers) on 16 logical cores
run_rvr_nature_grid_expansion(cores=6, num_seeds=25,
    obstacle_weight=8.0,
    experiment_name="rvr_nature_grid_sym_obs")
