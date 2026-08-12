include(joinpath(@__DIR__, "param_sweep.jl"))
using Senate  # for non_robust, robust enums
using BlockArrays
using LinearAlgebra

function run_drift_test_sweep(; cores=10, override=false, num_seeds=100)
    experiment_name = "drift_test"
    #Check for folder existence
    output_dir = joinpath(@__DIR__, "..", "outputs", experiment_name)
    if !isdir(output_dir)
        #Confirm directory creation
        print("Creating output directory at $output_dir... press y to continue\n")
        if readline() != "y"
            error("User did not confirm directory creation")
        end
        mkpath(output_dir)
    end

    run_parallel_sweep(
        p1_type=non_robust,
        p2_type=[non_robust, robust],
        # Cost model templates for obstacles
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        # Obstacle settings
        p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p1_obstacle_weights=[8.0],
        p2_obstacle_weights=[8.0],
        p1_ellipsoidal_cost_weight=[0.5],
        p2_ellipsoidal_cost_weight=[0.5],
        p1_control_cost_weight=[2.0],
        p2_control_cost_weight=[2.0],
        # Drift settings
        gt_drift_sensor_scale=[1.0],
        p1_believes_self_drift_sensor_scale=[1.0],
        p2_believes_p1_drift_sensor_scale=[0.0],
        # Robustness
        p2_nature_multiplier=[2.0],
        # Noise
        process_noise_covariance=0.01 * I(6),
        sensor_noise_covariance=0.01 * I(6),
        # Dynamics
        dynamics_model_template=:default,
        dt=0.75,
        horizon=10,
        ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
        experiment_name_prefix="drift_test",
        num_seeds=num_seeds,
        cores=cores,
        override=override
    )
end

function run_planning_horizon_sweep(; cores=10, override=false, num_seeds=100)
    experiment_name = "planning_horizon_sweep"
    output_dir = joinpath(@__DIR__, "..", "outputs", experiment_name)
    if !isdir(output_dir)
        print("Creating output directory at $output_dir... press y to continue\n")
        if readline() != "y"
            error("User did not confirm directory creation")
        end
        mkpath(output_dir)
    end

    run_parallel_sweep(
        p1_type=non_robust,
        p2_type=[non_robust, robust],
        planning_horizon=[2, 5],
        # Cost model templates for obstacles
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        # Obstacle settings
        p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p1_obstacle_weights=[8.0],
        p2_obstacle_weights=[8.0],
        p1_ellipsoidal_cost_weight=[0.5],
        p2_ellipsoidal_cost_weight=[0.5],
        p1_control_cost_weight=[2.0],
        p2_control_cost_weight=[2.0],
        # Drift settings
        gt_drift_sensor_scale=[1.0],
        p1_believes_self_drift_sensor_scale=[1.0],
        p2_believes_p1_drift_sensor_scale=[0.0],
        # Robustness
        p2_nature_multiplier=[2.0],
        # Noise
        process_noise_covariance=0.01 * I(6),
        sensor_noise_covariance=0.01 * I(6),
        # Dynamics
        dynamics_model_template=:default,
        dt=0.75,
        horizon=15,
        ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
        experiment_name_prefix="planning_horizon_sweep",
        num_seeds=num_seeds,
        cores=cores,
        override=override
    )
end

# Sweep (1,2,3): Higher control costs + higher horizon + more planning horizon variation
# - Smaller controls (higher cost weight) → cleaner movement differences
# - Higher simulation horizon → longer trends
# - Wide planning horizon range → shortsighted vs compounding inaccuracy
function run_control_planning_sweep(; cores=10, override=false, num_seeds=100)
    experiment_name = "control_planning_sweep"
    output_dir = joinpath(@__DIR__, "..", "outputs", experiment_name)
    if !isdir(output_dir)
        print("Creating output directory at $output_dir... press y to continue\n")
        if readline() != "y"
            error("User did not confirm directory creation")
        end
        mkpath(output_dir)
    end

    run_parallel_sweep(
        p1_type=non_robust,
        p2_type=[non_robust, robust],
        planning_horizon=[2, 5, 8, 12],
        # Cost model templates for obstacles
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        # Obstacle settings
        p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p1_obstacle_weights=[8.0],
        p2_obstacle_weights=[8.0],
        p1_ellipsoidal_cost_weight=[0.5],
        p2_ellipsoidal_cost_weight=[0.5],
        # Higher control costs → smaller controls
        p1_control_cost_weight=[5.0],
        p2_control_cost_weight=[5.0],
        # Drift settings
        gt_drift_sensor_scale=[1.0],
        p1_believes_self_drift_sensor_scale=[1.0],
        p2_believes_p1_drift_sensor_scale=[0.0],
        # Robustness
        p2_nature_multiplier=[2.0],
        # Noise (default)
        process_noise_covariance=0.001 * I(6),
        sensor_noise_covariance=0.001 * I(6),
        # Dynamics
        dynamics_model_template=:default,
        dt=0.75,
        horizon=20,
        ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
        experiment_name_prefix="control_planning_sweep",
        num_seeds=num_seeds,
        cores=cores,
        override=override
    )
end

# Sweep (4,5): Drift magnitude × partial mismatch gradient
# - Greater drift magnitude → larger information asymmetry
# - p2 belief ranges from 0% to 100% of true drift (scaled per gt level)
function run_drift_mismatch_sweep(; cores=10, override=false, num_seeds=100)
    experiment_name = "drift_mismatch_sweep"
    output_dir = joinpath(@__DIR__, "..", "outputs", experiment_name)
    if !isdir(output_dir)
        print("Creating output directory at $output_dir... press y to continue\n")
        if readline() != "y"
            error("User did not confirm directory creation")
        end
        mkpath(output_dir)
    end

    # Loop over drift magnitudes; p1 belief tracks gt, p2 belief swept as fraction of gt
    for drift_scale in [1.0, 2.0, 4.0]
        p2_beliefs = drift_scale .* [0.0, 0.25, 0.5, 0.75, 1.0]
        run_parallel_sweep(
            p1_type=non_robust,
            p2_type=[non_robust, robust],
            # Cost model templates for obstacles
            p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
            p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
            p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
            p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
            # Obstacle settings
            p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
            p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
            p1_obstacle_weights=[8.0],
            p2_obstacle_weights=[8.0],
            p1_ellipsoidal_cost_weight=[0.5],
            p2_ellipsoidal_cost_weight=[0.5],
            p1_control_cost_weight=[2.0],
            p2_control_cost_weight=[2.0],
            # Drift — gt and p1 coupled, p2 belief swept
            gt_drift_sensor_scale=[drift_scale],
            p1_believes_self_drift_sensor_scale=[drift_scale],
            p2_believes_p1_drift_sensor_scale=p2_beliefs,
            # Robustness
            p2_nature_multiplier=[2.0],
            # Noise (default)
            process_noise_covariance=0.001 * I(6),
            sensor_noise_covariance=0.001 * I(6),
            # Dynamics
            dynamics_model_template=:default,
            dt=0.75,
            horizon=10,
            ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
            experiment_name_prefix="drift_mismatch_sweep/gt_$(drift_scale)",
            num_seeds=num_seeds,
            cores=cores,
            override=override
        )
    end
end

# Sweep (7): Does robustness help the informed player (p1)?
function run_robustness_comparison_sweep(; cores=10, override=false, num_seeds=100)
    experiment_name = "robustness_comparison_sweep"
    output_dir = joinpath(@__DIR__, "..", "outputs", experiment_name)
    if !isdir(output_dir)
        print("Creating output directory at $output_dir... press y to continue\n")
        if readline() != "y"
            error("User did not confirm directory creation")
        end
        mkpath(output_dir)
    end

    run_parallel_sweep(
        p1_type=[non_robust, robust],
        p2_type=[non_robust, robust],
        # Cost model templates for obstacles
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        # Obstacle settings
        p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p1_obstacle_weights=[8.0],
        p2_obstacle_weights=[8.0],
        p1_ellipsoidal_cost_weight=[0.5],
        p2_ellipsoidal_cost_weight=[0.5],
        p1_control_cost_weight=[2.0],
        p2_control_cost_weight=[2.0],
        # Drift settings
        gt_drift_sensor_scale=[1.0],
        p1_believes_self_drift_sensor_scale=[1.0],
        p2_believes_p1_drift_sensor_scale=[0.0],
        # Robustness
        p1_nature_multiplier=[2.0],
        p2_nature_multiplier=[2.0],
        # Noise (default)
        process_noise_covariance=0.001 * I(6),
        sensor_noise_covariance=0.001 * I(6),
        # Dynamics
        dynamics_model_template=:default,
        dt=0.75,
        horizon=10,
        ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
        experiment_name_prefix="robustness_comparison_sweep",
        num_seeds=num_seeds,
        cores=cores,
        override=override
    )
end

function run_nature_control_sweep(; cores=8, override=false, num_seeds=100, offset = 1000)
    experiment_name = "nature_control_sweep"
    output_dir = joinpath(@__DIR__, "..", "outputs", experiment_name)
    if !isdir(output_dir)
        print("Creating output directory at $output_dir... press y to continue\n")
        if readline() != "y"
            error("User did not confirm directory creation")
        end
        mkpath(output_dir)
    end

    run_parallel_sweep(
        p1_type=non_robust,
        p2_type=[non_robust, robust],
        p2_nature_multiplier=[1, 2, 5, 10, 25, 50, 125, 250, 625, 1250, 3125, 6250],
        # Cost model templates for obstacles
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        # Obstacle settings
        p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p1_obstacle_weights=[8.0],
        p2_obstacle_weights=[8.0],
        p1_ellipsoidal_cost_weight=[0.5],
        p2_ellipsoidal_cost_weight=[0.5],
        p1_control_cost_weight=[2.0],
        p2_control_cost_weight=[2.0],
        # Drift settings
        gt_drift_sensor_scale=[1.0],
        p1_believes_self_drift_sensor_scale=[1.0],
        p2_believes_p1_drift_sensor_scale=[0.0],
        # Noise
        process_noise_covariance=0.001 * I(6),
        sensor_noise_covariance=0.001 * I(6),
        # Dynamics
        dynamics_model_template=:default,
        dt=0.75,
        horizon=10,
        ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
        experiment_name_prefix="nature_control_sweep",
        num_seeds=num_seeds,
        offset=offset,
        cores=cores,
        override=override
    )
end

function run_nature_control_sweep_no_drift(; cores=8, override=false, num_seeds=100, offset = 1000)
    experiment_name = "nature_control_sweep_no_drift"
    output_dir = joinpath(@__DIR__, "..", "outputs", experiment_name)
    if !isdir(output_dir)
        print("Creating output directory at $output_dir... press y to continue\n")
        if readline() != "y"
            error("User did not confirm directory creation")
        end
        mkpath(output_dir)
    end

    run_parallel_sweep(
        p1_type=non_robust,
        p2_type=[non_robust, robust],
        p2_nature_multiplier=[1, 2, 5, 10, 25, 50, 125, 250, 625, 1250, 3125, 6250],
        # Cost model templates for obstacles
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        # Obstacle settings
        p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p1_obstacle_weights=[8.0],
        p2_obstacle_weights=[8.0],
        p1_ellipsoidal_cost_weight=[0.5],
        p2_ellipsoidal_cost_weight=[0.5],
        p1_control_cost_weight=[2.0],
        p2_control_cost_weight=[2.0],
        # Drift settings
        gt_drift_sensor_scale=[0.0],
        p1_believes_self_drift_sensor_scale=[0.0],
        p2_believes_p1_drift_sensor_scale=[0.0],
        # Noise
        process_noise_covariance=0.001 * I(6),
        sensor_noise_covariance=0.001 * I(6),
        # Dynamics
        dynamics_model_template=:default,
        dt=0.75,
        horizon=10,
        ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
        experiment_name_prefix="nature_control_sweep_no_drift",
        num_seeds=num_seeds,
        offset=offset,
        cores=cores,
        override=override
    )
end

function run_rvr_nature_control_sweep(; cores=8, override=false, num_seeds=100, offset=1000,
                                       multipliers=[1, 2, 5, 10, 25, 50, 125, 250, 625, 1250, 3125, 6250])
    experiment_name = "rvr_nature_control_sweep"
    output_dir = joinpath(@__DIR__, "..", "outputs", experiment_name)
    if !isdir(output_dir)
        print("Creating output directory at $output_dir... press y to continue\n")
        if readline() != "y"
            error("User did not confirm directory creation")
        end
        mkpath(output_dir)
    end

    for mult in multipliers
        run_parallel_sweep(
            p1_type=robust,
            p2_type=[robust],
            p1_nature_multiplier=mult,
            p2_nature_multiplier=[mult],
            p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
            p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
            p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
            p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
            p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
            p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
            p1_obstacle_weights=[8.0],
            p2_obstacle_weights=[8.0],
            p1_ellipsoidal_cost_weight=[0.5],
            p2_ellipsoidal_cost_weight=[0.5],
            p1_control_cost_weight=[2.0],
            p2_control_cost_weight=[2.0],
            gt_drift_sensor_scale=[1.0],
            p1_believes_self_drift_sensor_scale=[1.0],
            p2_believes_p1_drift_sensor_scale=[0.0],
            process_noise_covariance=0.001 * I(6),
            sensor_noise_covariance=0.001 * I(6),
            dynamics_model_template=:default,
            dt=0.75,
            horizon=10,
            ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
            experiment_name_prefix=experiment_name,
            num_seeds=num_seeds,
            offset=offset,
            cores=cores,
            override=override,
        )
    end

    println("\n\n" * "="^60)
    println("COMPLETED R-vs-R NATURE CONTROL SWEEP ($(length(multipliers)) multipliers)")
    println("="^60)
end

function run_rvr_nature_control_sweep_no_drift(; cores=8, override=false, num_seeds=100, offset=1000,
                                                multipliers=[1, 2, 5, 10, 25, 50, 125, 250, 625, 1250, 3125, 6250])
    experiment_name = "rvr_nature_control_sweep_no_drift"
    output_dir = joinpath(@__DIR__, "..", "outputs", experiment_name)
    if !isdir(output_dir)
        print("Creating output directory at $output_dir... press y to continue\n")
        if readline() != "y"
            error("User did not confirm directory creation")
        end
        mkpath(output_dir)
    end

    for mult in multipliers
        run_parallel_sweep(
            p1_type=robust,
            p2_type=[robust],
            p1_nature_multiplier=mult,
            p2_nature_multiplier=[mult],
            p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
            p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
            p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
            p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
            p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
            p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
            p1_obstacle_weights=[8.0],
            p2_obstacle_weights=[8.0],
            p1_ellipsoidal_cost_weight=[0.5],
            p2_ellipsoidal_cost_weight=[0.5],
            p1_control_cost_weight=[2.0],
            p2_control_cost_weight=[2.0],
            gt_drift_sensor_scale=[0.0],
            p1_believes_self_drift_sensor_scale=[0.0],
            p2_believes_p1_drift_sensor_scale=[0.0],
            process_noise_covariance=0.001 * I(6),
            sensor_noise_covariance=0.001 * I(6),
            dynamics_model_template=:default,
            dt=0.75,
            horizon=10,
            ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
            experiment_name_prefix=experiment_name,
            num_seeds=num_seeds,
            offset=offset,
            cores=cores,
            override=override,
        )
    end

    println("\n\n" * "="^60)
    println("COMPLETED R-vs-R NATURE CONTROL SWEEP NO-DRIFT ($(length(multipliers)) multipliers)")
    println("="^60)
end

# R-vs-R grid over each player's *nature* control-cost multiplier (c1 x c2),
# with symmetric drift beliefs (each player models its own sensor drift
# correctly, is blind to the opponent's) and no obstacle (weight 0; obstacle
# cost templates kept so analysis decomposition stays format-compatible).
function run_rvr_nature_grid_sweep(; cores=8, override=false, num_seeds=25, offset=1000,
                                    p1_multipliers=[5, 25, 125, 625],
                                    p2_multipliers=[5, 25, 125, 625],
                                    control_cost_weight=2.0,
                                    obstacle_weight=0.0,
                                    experiment_name="rvr_nature_grid_sym_noobs")
    output_dir = joinpath(@__DIR__, "..", "outputs", "runs", experiment_name)
    isdir(output_dir) || mkpath(output_dir)

    run_parallel_sweep(
        abbrev_names=true,  # keep filenames under the 255-char Windows component limit
        p1_type=robust,
        p2_type=robust,
        p1_nature_multiplier=collect(p1_multipliers),
        p2_nature_multiplier=collect(p2_multipliers),
        p1_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p1_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p2_non_terminal_cost_model_template=obstacle_non_terminal_cost_function_generator,
        p2_terminal_cost_model_template=obstacle_terminal_cost_function_generator,
        p1_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p2_obstacle_centers=[mortar([[[1.5, 1.5]]])],
        p1_obstacle_weights=[obstacle_weight],
        p2_obstacle_weights=[obstacle_weight],
        p1_ellipsoidal_cost_weight=[0.5],
        p2_ellipsoidal_cost_weight=[0.5],
        p1_control_cost_weight=[control_cost_weight],
        p2_control_cost_weight=[control_cost_weight],
        # Symmetric drift: GT drift on both sensors (as before); each player
        # believes its own drift, neither believes the opponent's.
        gt_drift_sensor_scale=[1.0],
        p1_believes_self_drift_sensor_scale=[1.0],
        p2_believes_self_drift_sensor_scale=[1.0],
        p1_believes_p2_drift_sensor_scale=[0.0],
        p2_believes_p1_drift_sensor_scale=0.0,  # scalar: keeps it out of the filename prefix
        process_noise_covariance=0.001 * I(6),
        sensor_noise_covariance=0.001 * I(6),
        dynamics_model_template=:default,
        dt=0.75,
        horizon=10,
        ground_truth_initial_states=[mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])],
        experiment_name_prefix=experiment_name,
        num_seeds=num_seeds,
        offset=offset,
        cores=cores,
        override=override,
    )

    println("\n\n" * "="^60)
    println("COMPLETED R-vs-R NATURE GRID SWEEP ($(length(p1_multipliers))x$(length(p2_multipliers)) cells x $(num_seeds) seeds)")
    println("="^60)
end
