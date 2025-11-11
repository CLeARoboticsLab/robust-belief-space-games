include("assymetric_experiment.jl")
using Infiltrator
using BlockArrays
using LinearAlgebra

function run_all_drift_experiments(override=false)
    try
        drift_scale = 10
        no_drift_scale = 0.0

        senator_ground_truths = [
            mortar([[0.75, 0.75], [1.75, 1.0], [1.0, 1.75]]),
            # mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
            # mortar([[0.75, 0.75], [2.0, 0.5], [1.25, 2.5]]),
        ]

        nature_multiplier_values = [0.5]
        
        run_asymmetric_experiment(
            p1_type=[non_robust],
            p2_type=[non_robust],
            dynamics_model_template=:under_actuated,
            p2_believes_p1_drift_sensor_scale=no_drift_scale,
            p2_nature_multiplier=nature_multiplier_values,
            ground_truth_initial_states=senator_ground_truths,
            horizon=7,
            experiment_name_prefix="v3_test",
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
