include("assymetric_experiment.jl")
using Infiltrator
using BlockArrays
using LinearAlgebra

function run_all_drift_experiments(override=false)
    # --- File Output Setup ---
    log_filename = "exp/senate/outputs/logs/drift_exp_output.txt"
    println("Starting experiment run. All output will be logged to $(log_filename)")

    # open(log_filename, "w") do log_file
    #     redirect_stdout(log_file) do
            try
                # --- Experiment Configuration ---
                drift_scale = 0.1
                no_drift_scale = 0.0

                player_type_combos = [
                    (non_robust, non_robust),
                    (non_robust, robust)
                ]

                senator_ground_truths = [
                    mortar([[0.75, 0.75], [1.75, 1.0], [1.0, 1.75]]),
                    mortar([[0.75, 0.75], [2.0, 0.5], [1.0, 1.75]]),
                    mortar([[0.75, 0.75], [2.0, 0.5], [1.25, 2.5]]),
                ]

                # attraction_matrices = [
                #     # I(3),
                #     [
                #     1 0.25 0.25;
                #     0.7 1 0;
                #     0.7 0 1
                #     ]
                # ]

                drift_values = [no_drift_scale, drift_scale]
                dynamics_models = [:default, :under_actuated]
                experiments = []

                # --- Experiment Generation ---
                for (dynamics_idx, dynamics_model) in enumerate(dynamics_models)
                    for (p1_type, p2_type) in player_type_combos
                        for (gt_idx, gt_initial_states) in enumerate(senator_ground_truths)
                            for p2_believes_p1_drift in drift_values
                                
                                p2_believes_p1_sensor_model_template = if p2_believes_p1_drift == no_drift_scale
                                    base_sensor_model
                                else
                                    covariance_drift_sensor_model
                                end

                                p1_type_str = p1_type == non_robust ? "nr" : "r"
                                p2_type_str = p2_type == non_robust ? "nr" : "r"
                                
                                p2_belief_model_str = if p2_believes_p1_drift == no_drift_scale
                                    ""
                                else
                                    "wrong_p2"
                                end

                                exp_name = "asym_gt_$(gt_idx)_$(p1_type_str)_$(p2_type_str)_:$(dynamics_model)_$(p2_belief_model_str)"

                                params = Dict(
                                    :p1_type => p1_type,
                                    :p2_type => p2_type,
                                    :p2_believes_p1_drift_sensor_scale => p2_believes_p1_drift,
                                    :p2_believes_p1_sensor_model => p2_believes_p1_sensor_model_template,
                                    :ground_truth_initial_states => gt_initial_states,
                                    :dynamics_model_template => dynamics_model,
                                    :horizon => 7,
                                    :experiment_name_prefix => exp_name,
                                    :override => override
                                )
                                push!(experiments, params)
                            end
                        end
                    end
                end
                # --- Run Experiments ---
                for (idx, exp_params) in enumerate(experiments)
                    println("\n" * "="^60)
                    println("RUNNING EXPERIMENT $idx/$(length(experiments))")
                    println("Name: ", exp_params[:experiment_name_prefix])
                    println("Params: ", exp_params)
                    println("="^60 * "\n")
                    
                    run_asymmetric_experiment(; exp_params...)
                end

                println("\n\n" * "="^60)
                println("COMPLETED ALL $(length(experiments)) EXPERIMENTS")
                println("="^60)

            catch e
                println("An error occurred: ", e)
                showerror(stdout, e, catch_backtrace())
                rethrow(e)
            finally
                # This block is empty, but the redirect_stdout will restore the original stdout
            end
    #     end
    # end
end
