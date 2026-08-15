#=
Launch the FULL v2 RvR nature-cost grid (no obstacle) under the normalized
nature cost (cost.jl nature_cost_scale — equilibrium-invariant rescaling that
fixes the high-c solver pathology; see commit message).

Design: 6x6 robust grid, c in {5, 25, 125, 625, 3125, 15625} + nominal
(non_robust) row/col/corner = 49 cells x 25 seeds (offset 1000) = 1225 runs.
Fresh experiment dir — the v1 dirs (rvr_nature_grid_sym_*) keep the
old-solver data for provenance. Rerun after a crash to resume (skip-existing).

Run from anywhere:
    julia --project=. exp/senate/versions/run_rvr_grid_v2.jl
=#

# save_file_prefix is relative to cwd, so run from repo root
cd(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "test_drift_sweep.jl"))

run_rvr_nature_grid_expansion(cores=12, num_seeds=25,
    experiment_name="rvr_nature_grid_v2_noobs")
