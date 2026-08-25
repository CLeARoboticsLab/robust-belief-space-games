include("/home/lihuan/robust-belief-space-games/exp/senate/versions/test_drift_sweep.jl")
run_rvr_p1drift_full(cores=50, num_seeds=50, offset=1000,
    nature_multipliers=[1.0, 2.5, 12.5, 62.5],
    ds=[-0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
run_rvr_p1drift_full(cores=50, num_seeds=50, offset=1000,
    nature_multipliers=[5, 25, 125, 15625],
    ds=[-0.4, -0.3, 0.3, 0.4])
