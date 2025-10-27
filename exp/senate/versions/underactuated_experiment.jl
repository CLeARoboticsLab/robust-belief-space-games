using Senate
using BlockArrays

function non_robust_underactuated_experiment(;override::Bool=false)
    params = DefaultSenateParams(;
        player_configs = Dict(
            1 => DefaultPlayerConfig(player_idx=1, type=non_robust,
                ellipsoid_centers = [
                    [1, -1],
                ],
                ellipsoid_radii = [
                    [3, 1]
                ],
                self_dynamics_model_template=under_actuated_dynamics,
                ),
            2 => DefaultPlayerConfig(player_idx=2, type=non_robust,
                ellipsoid_centers = [
                    [-1, 1],
                ],
                ellipsoid_radii = [
                    [1, 3]
                ],
                self_dynamics_model_template=under_actuated_dynamics,
                ),
        ),
    )

    results = run_receding_horizon_trials(params; override=override)
    return results
end

function robust_underactuated_experiment(;override::Bool=false)
    params = DefaultSenateParams(;
    player_configs = Dict(
        2 => DefaultPlayerConfig(player_idx=2, type=non_robust,
            ellipsoid_centers = [
                [1, -1],
            ],
            ellipsoid_radii = [
                [3, 1]
            ],
            self_dynamics_model_template=under_actuated_dynamics,
            ),
        1 => DefaultPlayerConfig(player_idx=1, type=robust,
            ellipsoid_centers = [
                [-1, 1],
            ],
            ellipsoid_radii = [
                [1, 3]
            ],
            self_dynamics_model_template=under_actuated_dynamics,
            ),
        ),
    )

    results = run_receding_horizon_trials(params; override=override)
    return results
end

function non_robust_underactuated_attraction_experiment(;override::Bool=false)
    params = DefaultSenateParams(;
        player_configs = Dict(
            1 => DefaultPlayerConfig(player_idx=1, type=non_robust,
                ellipsoid_centers = [
                    [1, -1],
                ],
                ellipsoid_radii = [
                    [3, 1]
                ],
                self_dynamics_model_template=attraction_dynamics_model,
                ),
            2 => DefaultPlayerConfig(player_idx=2, type=non_robust,
                ellipsoid_centers = [
                    [-1, 1],
                ],
                ellipsoid_radii = [
                    [1, 3]
                ],
                self_dynamics_model_template=attraction_dynamics_model,
                ),
        ),
    )

    results = run_receding_horizon_trials(params; override=override)
    return results
end

function robust_underactuated_attraction_experiment(;override::Bool=false)
    params = DefaultSenateParams(;
    player_configs = Dict(
        2 => DefaultPlayerConfig(player_idx=2, type=non_robust,
            ellipsoid_centers = [
                [1, -1],
            ],
            ellipsoid_radii = [
                [3, 1]
            ],
            self_dynamics_model_template=attraction_dynamics_model,
            ),
        1 => DefaultPlayerConfig(player_idx=1, type=robust,
            ellipsoid_centers = [
                [-1, 1],
            ],
            ellipsoid_radii = [
                [1, 3]
            ],
            self_dynamics_model_template=attraction_dynamics_model,
            ),
        ),
    )

    results = run_receding_horizon_trials(params; override=override)
    return results
end