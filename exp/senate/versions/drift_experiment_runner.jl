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

function run_drift_exp_v2(override=false;)
    senator_ground_truths = [
        mortar([[0.75, 0.75], [1.75, 1.0], [1.0, 1.75]]),
        # mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
        # mortar([[0.75, 0.75], [2.0, 0.5], [1.25, 2.5]]),
    ]

    nature_multiplier_values = [0.1]

    p2_belief_drift = [7.0, 10.0, 20, 50.0]

    dynamics_types = [:default]

    dt_values = [0.75]

    obstacle_centers = [[1.7, 1.7]]
    obstacle_weights = [1.0]
    
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


function run_obstacle_blocking(override=false;)
    senator_ground_truths = [
        mortar([[0.75, 0.75], [1.75, 1.0], [1.0, 1.75]]),
        mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
        mortar([[0.75, 0.75], [2.0, 0.5], [1.25, 2.5]]),
    ]

    nature_multiplier_values = [0.05, 0.5, 5.0]

    p2_belief_drift = [0.0, 50.0, 2500.0]

    dynamics_types = [:default]

    dt_values = [0.75]

    # Obstacle positions more in the trajectory corridor
    obstacle_centers = [
        mortar([[[1.25, 1.0]]]),   # directly in main flow
        mortar([[[1.4, 1.2]]]),    # slightly higher, still blocking
        mortar([[[1.5, 0.85]]]),   # lower path blockage
    ]
    # obstacle_centers = [
    #     mortar([[[1.25, 1.0], [1.34, 1.07], [1.19, 0.92], [1.29, 0.96]]]),
    #     mortar([[[1.40, 1.20], [1.50, 1.26], [1.33, 1.12]]]),
    #     mortar([[[1.50, 0.85], [1.60, 0.92], [1.44, 0.80], [1.55, 0.78]]]),
    # ]

    obstacle_weights = [0.0, 1.0, 4.0]
    
    ellipsoidal_weights = [0.5, 1.0, 2.0]
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
        experiment_name_prefix="obst_block",
        override=override,
        dt=dt_values,
    )

    println("\n\n" * "="^60)
    println("COMPLETED ALL EXPERIMENTS")
   
    println("="^60)
end

# V1: Fast weight sweep at a single, deeper-in-corridor obstacle location.
# Purpose: quickly see how obstacle weight changes behavior.
# Model mismatch: GT has drift, P2 underestimates it (thinks P1 has no drift)
function run_obstacle_blocking_v1_weight_sweep(override=false;)
    senator_ground_truths = [
        mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
    ]

    nature_multiplier_values = [0.5]
    dynamics_types = [:default]
    dt_values = [0.75]

    # Ground truth drift values (reality is drifty)
    gt_drift_values = [5.0, 10.0]
    # P1 correctly knows its own drift (same as GT)
    p1_self_drift_values = [5.0, 10.0]
    # P2 belief: 0 for mismatch, same as GT for control
    p2_belief_drift = [0.0, 10.0]

    obstacle_centers = [
        mortar([[[1.78, 1.00]]]),
    ]

    obstacle_weights = [4.0, 8.0]

    run_asymmetric_experiment(
        p1_type=[non_robust],
        p2_type=[non_robust, robust],
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p1_obstacle_centers=obstacle_centers,
        p1_obstacle_weights=obstacle_weights,
        p1_ellipsoidal_cost_weight=0.5,
        p1_obstacle_cost_function=obstacle_cost,
        dynamics_model_template=dynamics_types,
        gt_drift_sensor_scale=gt_drift_values,
        p1_believes_self_drift_sensor_scale=p1_self_drift_values,
        p2_believes_p1_drift_sensor_scale=p2_belief_drift,
        p2_nature_multiplier=nature_multiplier_values,
        ground_truth_initial_states=senator_ground_truths,
        horizon=10,
        experiment_name_prefix="obst_block_v1_wtsweep",
        override=override,
        dt=dt_values,
    )

    println("\n\n" * "="^60)
    println("COMPLETED V1 (weight sweep)")
    println("="^60)
end


# V2: Fast position sweep (3 positions) at a fixed high weight.
# Purpose: find where "further into trajectory" blocks best.
# Model mismatch: GT has drift, P2 underestimates it (thinks P1 has no drift)
function run_obstacle_blocking_v2_position_sweep(override=false;)
    senator_ground_truths = [
        mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
    ]

    nature_multiplier_values = [0.5]
    dynamics_types = [:default]
    dt_values = [0.75]

    # Ground truth drift values (reality is drifty)
    gt_drift_values = [5.0, 10.0]
    # P1 correctly knows its own drift (same as GT)
    p1_self_drift_values = [5.0, 10.0]
    # P2 belief: 0 for mismatch, same as GT for control
    p2_belief_drift = [0.0, 5.0]

    # three deeper-in-corridor placements
    obstacle_centers = [
        mortar([[[1.70, 0.95]]]),
        mortar([[[1.82, 1.00]]]),
        mortar([[[1.92, 1.05]]]),
    ]

    obstacle_weights = [6.0]

    run_asymmetric_experiment(
        p1_type=[non_robust],
        p2_type=[non_robust, robust],
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p1_obstacle_centers=obstacle_centers,
        p1_obstacle_weights=obstacle_weights,
        p1_ellipsoidal_cost_weight=0.5,
        p1_obstacle_cost_function=obstacle_cost,
        dynamics_model_template=dynamics_types,
        gt_drift_sensor_scale=gt_drift_values,
        p1_believes_self_drift_sensor_scale=p1_self_drift_values,
        p2_believes_p1_drift_sensor_scale=p2_belief_drift,
        p2_nature_multiplier=nature_multiplier_values,
        ground_truth_initial_states=senator_ground_truths,
        horizon=10,
        experiment_name_prefix="obst_block_v2_possweep",
        override=override,
        dt=dt_values,
    )

    println("\n\n" * "="^60)
    println("COMPLETED V2 (position sweep)")
    println("="^60)
end


# V3: Single "wall/cluster" obstacle configuration deeper in the corridor.
# Purpose: multi-center diagonal wall blocking the main corridor.
# Model mismatch: GT has drift, P2 underestimates it (thinks P1 has no drift)
function run_obstacle_blocking_v3_cluster_wall(override=false;)
    senator_ground_truths = [
        mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
    ]

    nature_multiplier_values = [0.5]
    dynamics_types = [:default]
    dt_values = [0.75]

    # Ground truth drift values (reality is drifty)
    gt_drift_values = [5.0, 10.0]
    # P1 correctly knows its own drift (same as GT)
    p1_self_drift_values = [5.0, 10.0]
    # P2 belief: 0 for mismatch, same as GT for control
    p2_belief_drift = [0.0, 5.0]

    # Multiple centers forming a diagonal "wall" in the main corridor
    obstacle_centers = [
        mortar([[[1.72, 0.92],
                 [1.80, 0.98],
                 [1.88, 1.04],
                 [1.96, 1.10]]]),
    ]

    obstacle_weights = [6.0]

    run_asymmetric_experiment(
        p1_type=[non_robust],
        p2_type=[non_robust, robust],
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p1_obstacle_centers=obstacle_centers,
        p1_obstacle_weights=obstacle_weights,
        p1_ellipsoidal_cost_weight=0.5,
        p1_obstacle_cost_function=obstacle_cost,
        dynamics_model_template=dynamics_types,
        gt_drift_sensor_scale=gt_drift_values,
        p1_believes_self_drift_sensor_scale=p1_self_drift_values,
        p2_believes_p1_drift_sensor_scale=p2_belief_drift,
        p2_nature_multiplier=nature_multiplier_values,
        ground_truth_initial_states=senator_ground_truths,
        horizon=10,
        experiment_name_prefix="obst_block_v3_clusterwall",
        override=override,
        dt=dt_values,
    )

    println("\n\n" * "="^60)
    println("COMPLETED V3 (cluster wall)")
    println("="^60)
end


function run_obstacle_blocking_deeper_symmetry_check(override=false;)
    # Keep it fast: 1 GT, 1 dt, 1 dynamics, 1 belief drift, 1 nature multiplier
    senator_ground_truths = [
        mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
    ]

    dt_values = [0.75]
    dynamics_types = [:default]

    nature_multiplier_values = [0.5]
    belief_drift_values = [50.0]

    # Move obstacles further into the trajectory corridor (shift right vs ~1.25–1.5)
    obstacle_centers = [
        mortar([[[1.75, 1.00]]]),
        mortar([[[1.85, 1.08]]]),
        mortar([[[1.92, 0.92]]]),
    ]

    # Requested larger weights
    obstacle_weights = [128.0, 16.0, 0.0]

    # --------------------------
    # A) ORIGINAL ORIENTATION
    # Obstacles + cost templates on P1; belief/nature knobs on P2 (about P1)
    # --------------------------
    run_asymmetric_experiment(
        p1_type=[non_robust],
        p2_type=[non_robust, robust],

        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p1_obstacle_centers=obstacle_centers,
        p1_obstacle_weights=obstacle_weights,
        p1_ellipsoidal_cost_weight=0.5,
        p1_obstacle_cost_function=obstacle_cost,

        dynamics_model_template=dynamics_types,
        p2_believes_p1_drift_sensor_scale=belief_drift_values,
        p2_nature_multiplier=nature_multiplier_values,

        ground_truth_initial_states=senator_ground_truths,
        horizon=10,
        dt=dt_values,
        experiment_name_prefix="obst_block_deeper_A",
        override=override,
    )

    # --------------------------
    # B) FLIPPED ORIENTATION
    # Swap who is P1 vs P2 for robustness/belief/nature.
    # Obstacles still live on the P1 slot, so this makes "original P2" now face them.
    # --------------------------
    run_asymmetric_experiment(
        p1_type=[non_robust, robust],
        p2_type=[non_robust],

        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p1_obstacle_centers=obstacle_centers,
        p1_obstacle_weights=obstacle_weights,
        p1_ellipsoidal_cost_weight=0.5,
        p1_obstacle_cost_function=obstacle_cost,

        dynamics_model_template=dynamics_types,
        p1_believes_p2_drift_sensor_scale=belief_drift_values,
        p1_nature_multiplier=nature_multiplier_values,

        ground_truth_initial_states=senator_ground_truths,
        horizon=10,
        dt=dt_values,
        experiment_name_prefix="obst_block_deeper_B_flip",
        override=override,
    )

    println("\n\n" * "="^60)
    println("COMPLETED symmetry check (A then B_flip)")
    println("="^60)
end
