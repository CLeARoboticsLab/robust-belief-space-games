#=
Launch ONLY the nominal (non_robust, c -> Inf) cells of the RvR grid
expansion, against the EXISTING multipliers — no obstacle arm.

Cells: nominal-vs-robust both orientations x {5, 25, 125, 625} + nominal-vs-
nominal = 9 cells x 25 seeds (offset 1000) => 225 runs. The high-c robust
grid (3125/15625) is deliberately excluded; run run_rvr_grid_expansion.jl
later to fill it (existing files skip).

Run from anywhere:
    julia --project=. exp/senate/versions/run_rvr_grid_nominal.jl
=#

# save_file_prefix is relative to cwd, so run from repo root
cd(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "test_drift_sweep.jl"))

run_rvr_nature_grid_expansion(cores=8, num_seeds=25,
    multipliers=[5, 25, 125, 625],
    parts=[:nominal])
