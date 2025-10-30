include("mass_experiment.jl")

function run_all_drift_experiments()
    drift_scale = 0.1
    no_drift_scale = 0.0

    # Sensor Drift Experiments
    run_mass_experiment(;
        p1_type=non_robust, p2_type=non_robust,
        p1_drift_sensor_scale=no_drift_scale,
        p2_drift_sensor_scale=no_drift_scale,
        gt_drift_sensor_scale=no_drift_scale,
        experiment_name_prefix="sensor_non_robust_no_drift_non_robust_no_drift_gt_no_drift"
    )
    run_mass_experiment(;
        p1_type=non_robust, p2_type=non_robust,
        p1_drift_sensor_scale=drift_scale,
        p2_drift_sensor_scale=drift_scale,
        gt_drift_sensor_scale=no_drift_scale,
        experiment_name_prefix="sensor_non_robust_drift_non_robust_drift_gt_no_drift"
    )
    run_mass_experiment(;
        p1_type=non_robust, p2_type=non_robust,
        p1_drift_sensor_scale=drift_scale,
        p2_drift_sensor_scale=no_drift_scale,
        gt_drift_sensor_scale=no_drift_scale,
        experiment_name_prefix="sensor_non_robust_drift_non_robust_no_drift_gt_no_drift"
    )
    run_mass_experiment(;
        p1_type=non_robust, p2_type=non_robust,
        p1_drift_sensor_scale=no_drift_scale,
        p2_drift_sensor_scale=ndrift_scale,
        gt_drift_sensor_scale=no_drift_scale,
        experiment_name_prefix="sensor_non_robust_no_drift_non_robust_drift_gt_no_drift"
    )
    run_mass_experiment(;
        p1_type=non_robust, p2_type=robust,
        p1_drift_sensor_scale=no_drift_scale,
        p2_drift_sensor_scale=no_drift_scale,
        gt_drift_sensor_scale=no_drift_scale,
        experiment_name_prefix="sensor_robust_no_drift_robust_no_drift_gt_no_drift"
    )
    run_mass_experiment(;
        p1_type=non_robust, p2_type=robust,
        p1_drift_sensor_scale=drift_scale,
        p2_drift_sensor_scale=no_drift_scale,
        gt_drift_sensor_scale=no_drift_scale,
        experiment_name_prefix="sensor_robust_no_drift_robust_drift_gt_no_drift"
    )
    run_mass_experiment(;
        p1_type=non_robust, p2_type=robust,
        p1_drift_sensor_scale=drift_scale,
        p2_drift_sensor_scale=no_drift_scale,
        gt_drift_sensor_scale=drift_scale,
        experiment_name_prefix="sensor_robust_drift_robust_no_drift_gt_drift"
    )

    # Dynamics Drift Experiments
    for dynamics_model in [:default, :under_actuated, :attraction]
        run_mass_experiment(;
            p1_type=non_robust, p2_type=non_robust,
            p1_drift_dynamics_scale=no_drift_scale,
            p2_drift_dynamics_scale=no_drift_scale,
            gt_drift_dynamics_scale=no_drift_scale,
            dynamics_model_template=dynamics_model,
            experiment_name_prefix="dynamics_non_robust_no_drift_non_robust_no_drift_gt_no_drift"
        )
        run_mass_experiment(;
            p1_type=non_robust, p2_type=non_robust,
            p1_drift_dynamics_scale=drift_scale,
            p2_drift_dynamics_scale=drift_scale,
            gt_drift_dynamics_scale=no_drift_scale,
            dynamics_model_template=dynamics_model,
            experiment_name_prefix="dynamics_non_robust_drift_non_robust_drift_gt_no_drift"
        )
        run_mass_experiment(;
            p1_type=non_robust, p2_type=non_robust,
            p1_drift_dynamics_scale=drift_scale,
            p2_drift_dynamics_scale=no_drift_scale,
            gt_drift_dynamics_scale=no_drift_scale,
            dynamics_model_template=dynamics_model,
            experiment_name_prefix="dynamics_non_robust_drift_non_robust_no_drift_gt_no_drift"
        )
        run_mass_experiment(;
            p1_type=non_robust, p2_type=non_robust,
            p1_drift_dynamics_scale=no_drift_scale,
            p2_drift_dynamics_scale=drift_scale,
            gt_drift_dynamics_scale=no_drift_scale,
            dynamics_model_template=dynamics_model,
            experiment_name_prefix="dynamics_non_robust_no_drift_non_robust_drift_gt_no_drift"
        )
        run_mass_experiment(;
            p1_type=non_robust, p2_type=robust,
            p1_drift_dynamics_scale=no_drift_scale,
            p2_drift_dynamics_scale=no_drift_scale,
            gt_drift_dynamics_scale=no_drift_scale,
            dynamics_model_template=dynamics_model,
            experiment_name_prefix="dynamics_robust_no_drift_robust_no_drift_gt_no_drift"
        )
        run_mass_experiment(;
            p1_type=non_robust, p2_type=robust,
            p1_drift_dynamics_scale=drift_scale,
            p2_drift_dynamics_scale=no_drift_scale,
            gt_drift_dynamics_scale=no_drift_scale,
            dynamics_model_template=dynamics_model,
            experiment_name_prefix="dynamics_robust_no_drift_robust_drift_gt_no_drift"
        )
        run_mass_experiment(;
            p1_type=non_robust, p2_type=robust,
            p1_drift_dynamics_scale=drift_scale,
            p2_drift_dynamics_scale=no_drift_scale,
            gt_drift_dynamics_scale=drift_scale,
            dynamics_model_template=dynamics_model,
            experiment_name_prefix="dynamics_robust_drift_robust_no_drift_gt_drift"
        )
        run_mass_experiment(;
            p1_type=non_robust, p2_type=robust,
            p1_drift_dynamics_scale=no_drift_scale,
            p2_drift_dynamics_scale=no_drift_scale,
            gt_drift_dynamics_scale=drift_scale,
            dynamics_model_template=dynamics_model,
            p1_drift_sensor_scale=no_drift_scale,
            p2_drift_sensor_scale=no_drift_scale,
            gt_drift_sensor_scale=drift_scale,
            experiment_name_prefix="dynamics_non_robust_no_drift_robust_no_drift_gt_drift_AND_gt_sensor_drift"
        )
        run_mass_experiment(;
            p1_type=non_robust, p2_type=non_robust,
            p1_drift_dynamics_scale=no_drift_scale,
            p2_drift_dynamics_scale=no_drift_scale,
            gt_drift_dynamics_scale=drift_scale,
            dynamics_model_template=dynamics_model,
            p1_drift_sensor_scale=no_drift_scale,
            p2_drift_sensor_scale=no_drift_scale,
            gt_drift_sensor_scale=drift_scale,
            experiment_name_prefix="dynamics_non_robust_no_drift_non_robust_no_drift_gt_drift_AND_gt_sensor_drift"
        )
    end
end
