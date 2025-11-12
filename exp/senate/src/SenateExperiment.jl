#Assertions for global variables
function init_checks()
    if (length(non_robust_activist.pos) != opinion_dim || length(robust_activist.pos) != opinion_dim ||
        length(non_robust_activist.scale) != opinion_dim || length(robust_activist.scale) != opinion_dim)
        throw(ErrorException("Position or scale dimension mismatch with opinion"))
    end
    if sum(non_robust_activist.scale) != 1 || sum(robust_activist.scale) != 1
        throw(ErrorException("Scale does not add up to 1")) 
        #Consider helping normalize instead of throwing an exception
        # if sum(non_robust_activist.scale) != 1 || sum(robust_activist.scale) == 0
        #     throw(ErrorException("Scale is zero vector"))
        # end
        # non_robust_activist.scale /= norm(non_robust_activist.scale)
        # robust_activist.scale /= norm(robust_activist.scale)
    end
end

function run_receding_horizon_trials(
    params::SenateParams; override::Bool=false)::Dict{String, Any}
    results = Dict{String, Any}()
    for trial in 1:params.trials
        results["trial_$trial"] = run_receding_horizon_trial(params; override=override)
    end
    return results
end

function run_receding_horizon_trial(params::SenateParams; override::Bool=false)
    # --- Receding Horizon Loop ---
    if isfile("exp/senate/outputs/$(params.name).dat") && !override
        println("Loading solution from exp/senate/outputs/$(params.name).dat")
        solutions = open(deserialize, "exp/senate/outputs/$(params.name).dat", "r")
        return solutions
    end

    gt_state_history = [copy(params.ground_truth_initial_states)]
    observation_history = []
    
    # Use player_idx as keys for consistency
    player_indices = sort(collect(keys(params.player_configs)))
    solution_history = Dict(idx => [] for idx in player_indices)
    
    current_beliefs = copy(params.initial_beliefs)
    current_gt_state = copy(params.ground_truth_initial_states)
    
    warm_starts = Dict{Int, Any}(idx => nothing for idx in player_indices)
    representative_games = Dict{Int, Any}(idx => nothing for idx in player_indices)
    plan_cost_history = Dict(idx => Any[] for idx in player_indices)

    Random.seed!(params.random_seed)

    dimensions = dims(params)

    self_environments = [
        BeliefEnvironment(
            params.player_configs[player_idx].self_dynamics_model,
            params.ground_truth_initial_states,
            params.player_configs[player_idx].self_sensor_model
        ) for player_idx in player_indices
    ]
    EKF_game = BeliefGame(
        self_environments,
        [],
        current_beliefs,
        params.planning_horizon,
        dimensions,
        params.ground_truth_initial_states,
        []
    )

    for t in 1:(params.horizon)-1
        println("Receding Horizon Step $t / $((params.horizon)-1)")

        for player_idx in player_indices
            player_config = params.player_configs[player_idx]
            if player_config.type == nature
                continue
            end
            is_robust_player = player_config.type == robust
            robust_players = [] # TODO: wip to potentially do more than one robust player. Should just be length one (or zero) for now.
            for idx in player_indices
                if player_config.other_player_configs[idx].type == robust
                    push!(robust_players, idx)
                end
            end
            # --- Construct Environments & Costs from this player's perspective ---
            environments = [
                BeliefEnvironment(
                    player_config.other_player_configs[idx].self_dynamics_model,
                    params.ground_truth_initial_states,
                    player_config.other_player_configs[idx].self_sensor_model
                ) for idx in player_indices
            ] # Nature doesn't have an environment (environment does a lot of EKF stuff, don't need nature to have their own belief/belief propogation)
            
            # What this player believes about everyone's costs
            costs = [
                BeliefCost(
                    player_config.other_player_configs[idx].self_non_terminal_cost_model,
                    player_config.other_player_configs[idx].self_terminal_cost_model
                )
                for idx in sort(collect(keys(player_config.other_player_configs)))
            ] # nature should have their own costs though

            game_horizon = min(params.planning_horizon, params.horizon - t + 1)
            game = BeliefGame(
                environments,
                costs,
                current_beliefs,
                game_horizon,
                dimensions, 
                current_gt_state,
                robust_players
            )

            if t == 1
                representative_games[player_idx] = game
            end

            nominal_beliefs, nominal_controls, _, plan_cost = solve(game; debug=false, warm_start=warm_starts[player_idx])

            push!(solution_history[player_idx], (; beliefs=nominal_beliefs, controls=nominal_controls))
            
            # --- Warm Start Logic ---
            if length(nominal_beliefs) > 1
                shifted_beliefs = nominal_beliefs[2:end]
                shifted_controls = nominal_controls[2:end]

                zero_control = BlockVector(zeros(length(nominal_controls[end])), length.(nominal_controls[end].blocks))
                
                last_belief = shifted_beliefs[end]
                g, _ = ekf_update(last_belief, zero_control, game)
                extended_belief = unvec(g, dims(last_belief))

                warm_start_beliefs = vcat(shifted_beliefs, [extended_belief])
                warm_start_controls = vcat(shifted_controls, [zero_control])
                warm_starts[player_idx] = (; beliefs=warm_start_beliefs, controls=warm_start_controls)
            else
                # If the horizon was 1, just reuse the solution
                warm_starts[player_idx] = (; beliefs=nominal_beliefs, controls=nominal_controls)
            end
            push!(plan_cost_history[player_idx], plan_cost)
        end

        # --- Execute Actions and Update Ground Truth ---
        
        
        control_indices = [1:6, 7:12] # TODO 1:6 for first player, 7:12 for second but dynamically from control_dims_per_activist or something
        merged_controls = BlockVector(vcat(
            [solution_history[idx][end].controls[1][control_indices[idx]] for idx in player_indices]...),
            vcat(collect(params.control_dims_per_activist for player_idx in player_indices)...))
        process_noise_vec = rand(params.process_noise_distribution)
        process_noise = BlockVector(process_noise_vec, length.(current_gt_state.blocks))

        
        # TODO got lazy...
        current_gt_state = params.ground_truth_dynamics_model(current_gt_state, merged_controls, process_noise)
        push!(gt_state_history, current_gt_state)

        observations = [
            params.ground_truth_sensor_models[idx](current_gt_state, BlockVector(rand(params.sensor_noise_distribution), length.(current_gt_state.blocks))) 
            for idx in player_indices
        ]
        observations = mortar(observations)
        push!(observation_history, observations)
        
        current_beliefs = ekf_update_with_observations(current_beliefs, merged_controls, EKF_game, observations)
    end

    # --- Package Results ---
    solutions_dict = Dict(
        idx => (
            gt_state_history=gt_state_history,
            observation_history=observation_history,
            solution_history=solution_history[idx],
            cost_history=plan_cost_history[idx]
        ) for idx in player_indices
    )
    return solutions_dict, representative_games, params
end