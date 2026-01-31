include("param_sweep.jl")
using Senate  # for non_robust, robust enums
using BlockArrays

function run_drift_test_sweep(; cores=10, override=false, num_seeds=100)
    run_parallel_sweep(
        p1_type=non_robust,
        p2_type=[non_robust, robust],
        # Cost model templates for obstacles
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        # Obstacle settings
        p1_obstacle_centers=[[1.5, 1.5]],
        p2_obstacle_centers=[[1.5, 1.5]],
        p1_obstacle_weights=8.0,
        p2_obstacle_weights=8.0,
        p1_ellipsoidal_cost_weight=0.5,
        p2_ellipsoidal_cost_weight=0.5,
        p1_control_cost_weight=2.0,
        p2_control_cost_weight=2.0,
        # Drift settings
        gt_drift_sensor_scale=1.0,
        p1_believes_self_drift_sensor_scale=1.0,
        p2_believes_p1_drift_sensor_scale=0.0,
        # Robustness
        p2_nature_multiplier=2.0,
        # Dynamics
        dynamics_model_template=:default,
        dt=0.75,
        horizon=10,
        ground_truth_initial_states=mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]]),
        experiment_name_prefix="drift_test",
        num_seeds=num_seeds,
        cores=cores,
        override=override
    )
end
