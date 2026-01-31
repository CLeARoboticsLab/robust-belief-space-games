include("param_sweep.jl")
using Senate  # for non_robust, robust enums

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
        p2_believes_p1_drift_sensor_scale=0.0,
        p2_nature_multiplier=2.0,
        p1_obstacle_centers=[[1.5, 1.5]],
        p2_obstacle_centers=[[1.5, 1.5]],
        p1_obstacle_weights=8.0,
        p2_obstacle_weights=8.0,
        horizon=7,
        experiment_name_prefix=experiment_name,
        num_seeds=num_seeds,
        cores=cores,
        override=override
    )
end
