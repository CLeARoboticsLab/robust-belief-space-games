#=
Launch the R-vs-R nature-cost grid sweep (c1 x c2, symmetric drift, no obstacle).

16 cells: c1, c2 in {5, 25, 125, 625} (nature effective weight = 2*c -> 10/50/250/1250),
25 seeds per cell (offset 1000) => 400 runs.

Run from anywhere:
    julia --project=. exp/senate/versions/run_rvr_grid.jl
=#

# save_file_prefix is relative to cwd, so run from repo root
cd(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "test_drift_sweep.jl"))

run_rvr_nature_grid_sweep(cores=8, num_seeds=25)
