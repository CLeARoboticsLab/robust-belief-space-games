using Senate
using BlockArrays

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
                control_cost_weight = 2.0),
            2 => DefaultPlayerConfig(player_idx=2, type=non_robust,
                ellipsoid_centers = [
                    [-1, 1],
                ],
                ellipsoid_radii = [
                    [1, 3]
                ],
                control_cost_weight = 2.0),
        ),
        ground_truth_initial_states = mortar([
            [0.5, 0.5],
            [0.0, 0.0],
            [-0.5, -0.5]
        ]),
        horizon=3,
    )

    params.player_configs[1].other_player_configs[2].type = non_robust

    results = run_receding_horizon_trials(params; override=override)
    return results
end