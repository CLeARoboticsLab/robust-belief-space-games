include("assymetric_experiment.jl")
using Infiltrator
using BlockArrays
using LinearAlgebra

function run_all_drift_experiments(override=false)
    try
        senator_ground_truths = [
            mortar([[0.75, 0.75], [1.75, 1.0], [1.0, 1.75]]),
            # mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
            # mortar([[0.75, 0.75], [2.0, 0.5], [1.25, 2.5]]),
        ]

        nature_multiplier_values = [0.5, 4.0, 0.1]

        p2_belief_drift = [0.0, 1.0, 10, 100]

        dynamics_types = [:default, :under_actuated]
        
        run_asymmetric_experiment(
            p1_type=[non_robust],
            p2_type=[non_robust, robust],
            dynamics_model_template=dynamics_types,
            p2_believes_p1_drift_sensor_scale=p2_belief_drift,
            p2_nature_multiplier=nature_multiplier_values,
            ground_truth_initial_states=senator_ground_truths,
            horizon=7,
            experiment_name_prefix="asym",
            override=override
        )

        println("\n\n" * "="^60)
        println("COMPLETED ALL EXPERIMENTS")
        println("="^60)

    catch e
        println("An error occurred: ", e)
        showerror(stdout, e, catch_backtrace())
        rethrow(e)
    end
end

function run_all_covariance_cost_drift_experiments(override=false)
    try
        senator_ground_truths = [
            mortar([[0.75, 0.75], [1.75, 1.0], [1.0, 1.75]]),
            # mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
            # mortar([[0.75, 0.75], [2.0, 0.5], [1.25, 2.5]]),
        ]

        nature_multiplier_values = [0.01, 0.5]

        # p2_belief_drift = [0.0, 1.0, 10, 100]
        p2_belief_drift = [0.0, 1]

        dynamics_types = [:under_actuated]
        
        run_asymmetric_experiment(
            p1_type=[non_robust],
            p2_type=[non_robust, robust],
            p1_non_terminal_cost_model_template=covariance_non_terminal_cost_function_generator,
            p2_non_terminal_cost_model_template=covariance_non_terminal_cost_function_generator,
            p1_terminal_cost_model_template=covariance_terminal_cost_function_generator,
            p2_terminal_cost_model_template=covariance_terminal_cost_function_generator,
            dynamics_model_template=dynamics_types,
            p2_believes_p1_drift_sensor_scale=p2_belief_drift,
            p2_nature_multiplier=nature_multiplier_values,
            ground_truth_initial_states=senator_ground_truths,
            horizon=7,
            experiment_name_prefix="cov_asym",
            override=override
        )

        println("\n\n" * "="^60)
        println("COMPLETED ALL EXPERIMENTS")
        println("="^60)

    catch e
        println("An error occurred: ", e)
        showerror(stdout, e, catch_backtrace())
        rethrow(e)
    end
end

function run_all_obstacle_cost_drift_experiments(override=false;)
    senator_ground_truths = [
        mortar([[0.75, 0.75], [1.75, 1.0], [1.0, 1.75]]),
        # mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
        # mortar([[0.75, 0.75], [2.0, 0.5], [1.25, 2.5]]),
    ]

    nature_multiplier_values = [0.1, 1.0]

    # p2_belief_drift = [0.0, 10]
    p2_belief_drift = [7.0]

    dynamics_types = [:default]

    dt_values = [0.75]

    obstacle_centers = [[1.7, 1.7]]
    obstacle_weights = [0.3, 1.0, 2.0]
    
    run_asymmetric_experiment(
        p1_type=[non_robust],
        p2_type=[non_robust, robust],
        # p2_type=[robust],
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p1_obstacle_centers = obstacle_centers,
        p1_obstacle_weights = obstacle_weights,
        p1_ellipsoidal_cost_weight = 0.5,
        p1_obstacle_cost_function = obstacle_cost,
        # p1_sigmoid_scale = [0.75],
        dynamics_model_template=dynamics_types,
        p2_believes_p1_drift_sensor_scale=p2_belief_drift,
        p2_nature_multiplier=nature_multiplier_values,
        ground_truth_initial_states=senator_ground_truths,
        horizon=10,
        experiment_name_prefix="obst_v2_asym",
        override=override,
        dt=dt_values,
    )

    println("\n\n" * "="^60)
    println("COMPLETED ALL EXPERIMENTS")
    println("="^60)
end