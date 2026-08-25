#!/bin/bash
cd ~/robust-belief-space-games
exec ~/.juliaup/bin/julia --project=. -e 'include("exp/senate/versions/test_drift_sweep.jl"); run_rvr_nature_grid_sweep(cores=24, num_seeds=25, obstacle_weight=8.0, experiment_name="rvr_nature_grid_sym_obs")'
