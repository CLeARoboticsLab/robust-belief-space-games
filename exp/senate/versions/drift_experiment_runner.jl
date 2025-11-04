include("assymetric_experiment.jl")
using Infiltrator
using BlockArrays
using LinearAlgebra

function run_all_drift_experiments(override=false)
    # --- File Output Setup ---
    log_filename = "exp/senate/outputs/logs/drift_exp_output.txt"
    println("Starting experiment run. All output will be logged to $(log_filename)")

    open(log_filename, "w") do log_file
        redirect_stdout(log_file) do
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

                attraction_matrices = [
                    I(3),
                    [
                    1 0.5 0.5;
                    1 1 0;
                    1 0 1
                    ]
                ]

                drift_values = [no_drift_scale, drift_scale]
                dynamics_models = [:default, :attraction]

                experiments = []

                # --- Experiment Generation ---
                for (att_idx, attraction_matrix) in enumerate(attraction_matrices)
                    for (p1_type, p2_type) in player_type_combos
                        for (gt_idx, gt_initial_states) in enumerate(senator_ground_truths)
                            for gt_sensor_drift in drift_values
                                p1_believes_p2_drift = gt_sensor_drift
                                for p2_believes_p1_drift in drift_values
                                    for dynamics_model in dynamics_models
                                        p1_actual_sensor_drift = gt_sensor_drift
                                        p2_actual_sensor_drift = gt_sensor_drift
                                        
                                        p1_type_str = p1_type == non_robust ? "nr" : "r"
                                        p2_type_str = p2_type == non_robust ? "nr" : "r"
                                        
                                        p1_belief_str = p1_believes_p2_drift == no_drift_scale ? "b_nod" : "b_d"
                                        p2_belief_str = p2_believes_p1_drift == no_drift_scale ? "b_nod" : "b_d"

                                        gt_sensor_str = gt_sensor_drift == no_drift_scale ? "gt_nod" : "gt_d"

                                        exp_name = "asym_att_$(att_idx)_gt_$(gt_idx)_$(p1_type_str)_$(p2_type_str)_p1_$(p1_belief_str)_p2_$(p2_belief_str)_$(gt_sensor_str)_$(dynamics_model)"

                                        params = Dict(
                                            :p1_type => p1_type,
                                            :p2_type => p2_type,
                                            :p1_drift_sensor_scale => p1_actual_sensor_drift,
                                            :p2_drift_sensor_scale => p2_actual_sensor_drift,
                                            :p1_believes_p2_drift_sensor_scale => p1_believes_p2_drift,
                                            :p2_believes_p1_drift_sensor_scale => p2_believes_p1_drift,
                                            :gt_drift_sensor_scale => gt_sensor_drift,
                                            :ground_truth_initial_states => gt_initial_states,
                                            :attraction_matrix => attraction_matrix,
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
        end
    end
end
