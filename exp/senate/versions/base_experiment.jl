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

    results = run_receding_horizon_trials(params; override=override)
    return results
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

    results = run_receding_horizon_trials(params; override=override)
    return results
end