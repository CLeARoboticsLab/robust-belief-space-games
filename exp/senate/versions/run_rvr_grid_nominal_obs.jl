#=
WITH-OBSTACLE twin of run_rvr_grid_nominal.jl: only the nominal cells
against the existing multipliers {5, 25, 125, 625} (+ nominal-vs-nominal),
obstacle weight 8.0 at (1.5, 1.5). 225 runs into rvr_nature_grid_sym_obs.

Run from anywhere:
    julia --project=. exp/senate/versions/run_rvr_grid_nominal_obs.jl
=#

# save_file_prefix is relative to cwd, so run from repo root
cd(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "test_drift_sweep.jl"))

# cores=6 so it can run alongside the no-obstacle sweep (8 workers) on 16 logical cores
run_rvr_nature_grid_expansion(cores=6, num_seeds=25,
    multipliers=[5, 25, 125, 625],
    parts=[:nominal],
    obstacle_weight=8.0,
    experiment_name="rvr_nature_grid_sym_obs")
