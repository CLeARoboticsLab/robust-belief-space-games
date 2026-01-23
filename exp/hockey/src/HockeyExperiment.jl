using Infiltrator
using TrajectoryGamesBase
using RobustBeliefGame
using LinearAlgebra
using BlockArrays
using JLD2
using FileIO
using Random



function run_receding_horizon_trials(
    params::HockeyParams; override::Bool=false)::Dict{String, Any}
    results = Dict{String, Any}()
    for trial in 1:params.trials
        results["trial_$trial"] = run_receding_horizon_trial(params; override=override, trial_number=trial)
    end
    return results
end

    function run_receding_horizon_trial(params::HockeyParams; override::Bool=false, trial_number::Int=1)
    base_name = params.name
    output_dir = params.output_dir
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    output_path = joinpath(output_dir, "$(base_name)_trial_$(trial_number).jld2")
    
    if isfile(output_path) && !override
        println("Loading solution from $output_path")
        @load output_path solutions
        return solutions
    end

    random_seed = params.random_seed + trial_number - 1 # Ensure different seed per trial
    Random.seed!(random_seed)

    ground_truth_initial_states = params.ground_truth_initial_states
    
    # If initial_beliefs is nothing, construct default
    current_beliefs = if isnothing(params.initial_beliefs)
         # Default Hockey belief construction
         initial_belief_covariance = [
             [0.1 0 0 0; 0 0.1 0 0; 0 0 0.25 0; 0 0 0 0.25], # Attacker
             [0.1 0 0 0; 0 0.1 0 0; 0 0 0.25 0; 0 0 0 0.25], # Defender
         ]
         Beliefs([
            Belief(ground_truth_initial_states[Block(1)], initial_belief_covariance[1]), # Attacker's belief of Attacker
            Belief(ground_truth_initial_states[Block(2)], initial_belief_covariance[2]), # Attacker's belief of Defender
            Belief(ground_truth_initial_states[Block(1)], initial_belief_covariance[1]), # Defender's belief of Attacker
            Belief(ground_truth_initial_states[Block(2)], initial_belief_covariance[2]), # Defender's belief of Defender
        ])
    else
        copy(params.initial_beliefs)
    end

    current_gt_state = copy(ground_truth_initial_states)
    
    dimensions = Hockey.dims(params)
    
    # Cost setup
    # Player 1: Attacker
    attacker_config = params.player_configs[1]
    # Player 2: Defender
    defender_config = params.player_configs[2]
    
    # Costs
    attacker_cost_fn = BeliefCost(
        (bs, us) -> attacker_non_terminal_cost(bs.beliefs[1], bs.beliefs[2], us, params; explicit_covariance=false),
        (bs) -> attacker_terminal_cost(bs.beliefs[1], bs.beliefs[2], params)
    )
    
    defender_cost_fn = BeliefCost(
        (bs, us) -> defender_non_terminal_cost(bs.beliefs[3], bs.beliefs[4], us, params; explicit_covariance=false),
        (bs) -> defender_terminal_cost(bs.beliefs[3], bs.beliefs[4], params)
    )
    
    nature_cost_fn = BeliefCost(
        (bs, us) -> nature_non_terminal_cost(bs.beliefs[3], bs.beliefs[4], us, params; explicit_covariance=false),
        (bs) -> nature_terminal_cost(bs.beliefs[3], bs.beliefs[4], params)
    )
    
    costs = if defender_config.type == robust
        [attacker_cost_fn, defender_cost_fn, nature_cost_fn]
    else
        [attacker_cost_fn, defender_cost_fn]
    end
    
    h_func = params.sensor_model
    
    # Environments for the BeliefGame (what players think)
    environments = [
        BeliefEnvironment(basic_dynamics, ground_truth_initial_states, h_func) 
        for _ in 1:dimensions.num_players
    ]
    
    # History containers
    gt_state_history = [copy(current_gt_state)]
    observation_history = []
    solution_history = Dict(idx => [] for idx in 1:dimensions.num_players)
    plan_cost_history = Dict(idx => Any[] for idx in 1:dimensions.num_players)
    warm_starts = Dict{Int, Any}(idx => nothing for idx in 1:dimensions.num_players)

    # --- Receding Horizon Loop ---
    for t in 1:(params.horizon)-1
        println("Receding Horizon Step $t / $((params.horizon)-1)")
        
        game_horizon = min(params.planning_horizon, params.horizon - t + 1)
        
        sols = Vector{Any}(undef, dimensions.num_players)
        
        for player_idx in 1:dimensions.num_players
            is_robust = params.player_configs[player_idx].type == robust

            current_player_costs = if is_robust
                [attacker_cost_fn, defender_cost_fn, nature_cost_fn]
            else
                [attacker_cost_fn, defender_cost_fn]
            end
            
            robust_ids = is_robust ? [2] : Int[]
            
            game = BeliefGame(
                environments,
                current_player_costs,
                current_beliefs,
                game_horizon,
                dimensions,
                current_gt_state,
                robust_ids
            )
            
            nominal_beliefs, nominal_controls, kkt_error, costs = solve(game; debug=false, warm_start=warm_starts[player_idx])
            
            sols[player_idx] = (nominal_beliefs, nominal_controls)
            
            push!(solution_history[player_idx], (; beliefs=nominal_beliefs, controls=nominal_controls, kkt_error=kkt_error, costs=costs))
            
            # Warm Start Update
            if length(nominal_beliefs) > 1
                shifted_beliefs = nominal_beliefs[2:end]
                shifted_controls = nominal_controls[2:end]
                
                zero_control = BlockVector(zeros(length(nominal_controls[end])), length.(nominal_controls[end].blocks))
                
                last_belief = shifted_beliefs[end]
                g, _ = ekf_update(last_belief, zero_control, game)
                
                extended_belief = unvec(g, dimensions.belief)
                
                warm_start_beliefs = vcat(shifted_beliefs, [extended_belief])
                warm_start_controls = vcat(shifted_controls, [zero_control])
                warm_starts[player_idx] = (; beliefs=warm_start_beliefs, controls=warm_start_controls)
            else
                warm_starts[player_idx] = (; beliefs=nominal_beliefs, controls=nominal_controls)
            end
        end
        
        u1 = sols[1][2][1][Block(1)]
        u2 = sols[2][2][1][Block(2)]
        
        merged_controls = BlockVector(vcat(u1, u2), [length(u1), length(u2)])
        
        process_noise_vec = rand(params.process_noise_distribution)
        process_noise = BlockVector(process_noise_vec, length.(current_gt_state.blocks))
        
        current_gt_state = basic_dynamics(current_gt_state, merged_controls, process_noise)
        push!(gt_state_history, current_gt_state)
        
        sensor_noise_vec = rand(params.sensor_noise_distribution)
        sensor_noise = BlockVector(sensor_noise_vec, length.(current_gt_state.blocks)) # Dims match state for this sensor model
        
        observations = mortar([
            h_func(current_gt_state, sensor_noise) for _ in 1:dimensions.num_players
        ])
        push!(observation_history, observations)
        
        dummy_game = BeliefGame(
            environments, 
            [attacker_cost_fn],
            current_beliefs, 
            params.planning_horizon, 
            dimensions, 
            ground_truth_initial_states, 
            Int[]
        )
        current_beliefs = ekf_update_with_observations(current_beliefs, merged_controls, dummy_game, observations)
                
        current_beliefs.beliefs[2] = copy(current_beliefs.beliefs[4])
        current_beliefs.beliefs[3] = copy(current_beliefs.beliefs[1])
        
    end
    
    # --- Save Results ---
    @save output_path gt_state_history observation_history solution_history params
    println("Saved results to $(abspath(output_path))")
    
    return (; gt_state_history, observation_history, solution_history)
end
