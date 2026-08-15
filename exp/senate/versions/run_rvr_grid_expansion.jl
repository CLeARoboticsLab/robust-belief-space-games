#=
Launch the EXPANSION of the R-vs-R nature-cost grid sweep (no obstacle).

Adds to rvr_nature_grid_sym_noobs:
  - higher multipliers 3125, 15625 (full 6x6 robust grid; original 4x4 skips
    on existing files, so this is resumable — just rerun after a crash)
  - nominal (non_robust, c -> Inf) players: nominal-vs-robust both
    orientations across all 6 multipliers, plus nominal-vs-nominal

33 new cells x 25 seeds (offset 1000) => 825 new runs (1225 files total).

Run from anywhere:
    julia --project=. exp/senate/versions/run_rvr_grid_expansion.jl
=#

# save_file_prefix is relative to cwd, so run from repo root
cd(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(@__DIR__, "test_drift_sweep.jl"))

run_rvr_nature_grid_expansion(cores=8, num_seeds=25)
