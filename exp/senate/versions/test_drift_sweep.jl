include("param_sweep.jl")
using Senate  # for non_robust, robust enums

function run_drift_test_sweep(; cores=10, override=false, num_seeds=100)
    run_parallel_sweep(
        p1_type=non_robust,
        p2_type=[non_robust, robust],
        p2_believes_p1_drift_sensor_scale=0.0,
        p2_nature_multiplier=2.0,
        p1_obstacle_centers=[[1.5, 1.5]],
        p2_obstacle_centers=[[1.5, 1.5]],
        p1_obstacle_weights=[8.0],
        p2_obstacle_weights=[8.0],
        horizon=7,
        experiment_name_prefix="drift_test",
        num_seeds=num_seeds,
        cores=cores,
        override=override
    )
end
