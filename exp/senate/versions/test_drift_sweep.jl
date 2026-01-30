include("param_sweep.jl")
using Senate  # for non_robust, robust enums

function run_drift_test_sweep(; cores=4, override=false)
    run_parallel_sweep(
        p1_type=[non_robust],
        p2_type=[non_robust, robust],
        p2_believes_p1_drift_sensor_scale=[0.0, 1.0],
        p2_nature_multiplier=[0.5, 8.0],
        horizon=7,
        experiment_name_prefix="drift_test",
        cores=cores,
        override=override
    )
end
