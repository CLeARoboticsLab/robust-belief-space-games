using Senate
using BlockArrays
using Serialization
using Infiltrator
function run_experiment(params::SenateParams;override::Bool=false, experiment_name::String="experiment", save_file_prefix::String="exp/senate")
    solution_filename = "$(save_file_prefix)/outputs/runs/$(experiment_name).dat"
    if experiment_name != "" && isfile(solution_filename)
        println("Solution already exists at $solution_filename. Override is $override.")
        if !override
            return
        end
    end

    results = run_receding_horizon_trials(params; override=override)
    println("Saving solution to $solution_filename")
    open(solution_filename, "w") do f
        serialize(f,results)
    end
    return results
end

function base_experiment(;override::Bool=false)
    params = DefaultSenateParams(;
        player_configs = Dict(
            1 => DefaultPlayerConfig(player_idx=1, type=non_robust,
                ellipsoid_centers = [
                    [1, -1],
                ],
                ellipsoid_radii = [
                    [3, 1]
                ],
                ),
            2 => DefaultPlayerConfig(player_idx=2, type=non_robust,
                ellipsoid_centers = [
                    [-1, 1],
                ],
                ellipsoid_radii = [
                    [1, 3]
                ],
                ),
        ),
    )
    run_experiment(params;override=override, experiment_name="base_experiment")
end

function robust_base_experiment(;override::Bool=false)
    params = DefaultSenateParams(;
    player_configs = Dict(
        2 => DefaultPlayerConfig(player_idx=2, type=non_robust,
            ellipsoid_centers = [
                [1, -1],
            ],
            ellipsoid_radii = [
                [3, 1]
            ],
            ),
        1 => DefaultPlayerConfig(player_idx=1, type=robust,
            ellipsoid_centers = [
                [-1, 1],
            ],
            ellipsoid_radii = [
                [1, 3]
            ],
            ),
        ),
    )
    run_experiment(params;override=override, experiment_name="robust_base_experiment")
end