function _populate_configs!(config::PlayerConfig; force::Bool=false)
    if isnothing(config.self_dynamics_model) || force
        config.self_dynamics_model = (x::BlockVector, u::BlockVector, m::BlockVector) -> base_dynamics(x, u, m; config=config)
    end
    if isnothing(config.self_sensor_model) || force
        config.self_sensor_model = (x::BlockVector, ns::BlockVector) -> base_sensor_model(x, ns; config=config)
    end
    if isnothing(config.self_non_terminal_cost_model) || force
        config.self_non_terminal_cost_model = (beliefs::Beliefs, u::BlockVector) -> base_non_terminal_cost_function_generator(config)(beliefs, u)
    end
    if isnothing(config.self_terminal_cost_model) || force
        config.self_terminal_cost_model = (beliefs::Beliefs) -> base_terminal_cost_function_generator(config)(beliefs)
    end
end

function _sync_params_to_configs_dims!(params::SenateParams)
    _sync_config_to_params_game_constants!(params, collect(values(params.player_configs)))
end

function _sync_config_to_params_game_constants!(params::SenateParams, configs::Vector{PlayerConfig})
    for player_config in configs
        player_config.num_senators = params.num_senators
        player_config.num_activists = params.num_activists
        player_config.state_dims_per_activist = params.state_dims_per_activist
        player_config.control_dims_per_activist = params.control_dims_per_activist
        player_config.belief_dims_per_activist = params.belief_dims_per_activist
        player_config.sensor_dims_per_activist = params.sensor_dims_per_activist
        player_config.control_dims_per_senator = params.control_dims_per_senator # self-believed...
    end
    return params
end

function _sync_params_to_configs_other_configs!(params::SenateParams)
    for (_, player_config) in params.player_configs
        beliefs_about_others = deepcopy(params.player_configs)
        for (_, belief_config) in beliefs_about_others
            _populate_configs!(belief_config; force=true) 
            belief_config.other_player_configs = beliefs_about_others
        end
        player_config.other_player_configs = beliefs_about_others
    end
    return params
end

function DefaultPlayerConfig(;kwargs...)
    config = PlayerConfig(;kwargs...)
    _populate_configs!(config)
    return config
end

function DefaultSenateParams(;kwargs...)
    params = SenateParams(;kwargs...)

    if isnothing(params.ground_truth_dynamics_model)
        _populate_configs!(params.ground_truth_dynamics_config)
        _sync_config_to_params_game_constants!(params, [params.ground_truth_dynamics_config])
        params.ground_truth_dynamics_model = params.ground_truth_dynamics_config.self_dynamics_model
    end
    if !(length(params.ground_truth_sensor_configs) == params.num_activists)
        error("length(ground_truth_sensor_configs) not equal to num_activists")end
    if isempty(params.ground_truth_dynamics_models)
        _sync_config_to_params_game_constants!(params, collect(values(params.ground_truth_dynamics_configs)))
        params.ground_truth_dynamics_models = Dict(idx => params.ground_truth_dynamics_configs[idx].self_dynamics_model for idx in keys(params.ground_truth_dynamics_configs))end
    if isempty(params.ground_truth_sensor_models)
        _sync_config_to_params_game_constants!(params, collect(values(params.ground_truth_sensor_configs)))
        params.ground_truth_sensor_models = Dict(idx => params.ground_truth_sensor_configs[idx].self_sensor_model for idx in keys(params.ground_truth_sensor_configs))end
    if !(length(params.ground_truth_initial_states.blocks) == params.num_senators)
        error("length(ground_truth_initial_states) not equal to num_senators")end
    if !(params.num_activists == length(params.player_configs))
        @warn("num_activists not equal to length of player_configs, setting num_activists to length(player_configs)")
        params.num_activists = length(params.player_configs)end
    if !(params.state_dims_per_activist == length.(params.ground_truth_initial_states.blocks))
        @warn("state_dims_per_activist not equal to length.(ground_truth_initial_states), setting state_dims_per_activist to length.(ground_truth_initial_states)")
        params.state_dims_per_activist = length.(params.ground_truth_initial_states)end
    if !(length(params.control_dims_per_activist) == params.num_senators)
        error("length(control_dims_per_activist) not equal to num_senators")end
    if !(sum(sum.(params.belief_dims_per_activist)) == sum(params.state_dims_per_activist)+sum(params.state_dims_per_activist.^2))
        error("sum(sum.(belief_dims_per_activist)) not equal to sum(state_dims_per_activist)+sum(state_dims_per_activist)^2")end
    if !(params.belief_dims_per_activist[1][1]^2 == params.belief_dims_per_activist[1][2])
        error("belief_dims_per_activist[1][1]^2 not equal to belief_dims_per_activist[1][2]")end
    if !isnothing(params.initial_beliefs)
        if !(length(params.initial_beliefs.beliefs) == params.num_senators*params.num_activists)
            error("length(initial_beliefs.beliefs) not equal to num_senators*num_activists")end
        if !(params.initial_beliefs.beliefs[1].belief_dim == params.belief_dims_per_activist[1][1])
            error("initial_beliefs.beliefs[1].belief_dim not equal to belief_dims_per_activist[1][1]")end
    else
        params.initial_beliefs = Beliefs(map(1:params.num_senators*params.num_activists) do belief_idx
            senator = 1 + (belief_idx-1) % params.num_senators
            mean = params.ground_truth_initial_states[Block(senator)]
            covariance = params.inital_belief_covariance_func(mean)
            Belief(mean, covariance)
        end)end
    if isnothing(params.process_noise_distribution)
        params.process_noise_distribution = MvNormal(params.process_noise_mean, params.process_noise_covariance)end
    if isnothing(params.sensor_noise_distribution)
        params.sensor_noise_distribution = MvNormal(params.sensor_noise_mean, params.sensor_noise_covariance)end
    calculated_control_dims_per_senator = [[config.control_dims_per_activist[senator] for config in sort(collect(values(params.player_configs)))] for senator in 1:params.num_senators]
    if !(calculated_control_dims_per_senator == params.control_dims_per_senator)
        @warn("control_dims_per_senator not equal to calculated_control_dims_per_senator, setting control_dims_per_senator to calculated_control_dims_per_senator")
        params.control_dims_per_senator = calculated_control_dims_per_senator
    end
    return _sync_params_to_configs_other_configs!(_sync_params_to_configs_dims!(params))
end