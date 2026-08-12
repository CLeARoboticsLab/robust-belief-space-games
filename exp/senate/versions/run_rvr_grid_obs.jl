#=
Launch the WITH-OBSTACLE twin of the R-vs-R nature-cost grid sweep.

Identical to run_rvr_grid.jl (c1 x c2 in {5,25,125,625}, symmetric drift,
25 seeds, offset 1000) except obstacle weight 8.0 at (1.5, 1.5) — same
obstacle as the existing sweep functions.

Run from anywhere:
    julia --project=. exp/senate/versions/run_rvr_grid_obs.jl
=#

# save_file_prefix is relative to cwd, so run from repo root
cd(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "test_drift_sweep.jl"))

# cores=6 so it can run alongside the no-obstacle sweep (8 workers) on 16 logical cores
run_rvr_nature_grid_sweep(cores=6, num_seeds=25,
    obstacle_weight=8.0,
    experiment_name="rvr_nature_grid_sym_obs")
