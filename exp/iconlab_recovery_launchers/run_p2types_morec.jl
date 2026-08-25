include("/home/lihuan/robust-belief-space-games/exp/senate/versions/test_drift_sweep.jl")
run_rvr_p1drift_p2types(cores=50, num_seeds=50, offset=1000,
    ds=[-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
    p1_cs=[25, 625, 15625], p2_cs=[5, 25, 125, 625])
