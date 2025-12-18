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
    # --- Check for existing results ---
    # Construct a unique filename/identifier derived from params
    base_name = params_to_name(params)
    output_dir = "outputs" 
    # Ensure output directory exists (relative to CWD)
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

    # --- Setup ---
    # Unpack params
    ground_truth_initial_states = params.ground_truth_initial_states
    # Assuming initial beliefs are set or we construct them from covariance?
    # Hockey.jl constructs them.
    # We should probably put this logic in `HockeyParams` or `Hockey` module init, 
    # but here is fine for now as we transition.
    
    # If initial_beliefs is nothing, construct default
    current_beliefs = if isnothing(params.initial_beliefs)
         # Default Hockey belief construction
         initial_belief_covariance = [
             [0.1 0 0 0; 0 0.1 0 0; 0 0 0.25 0; 0 0 0 0.25], # Attacker
             [0.1 0 0 0; 0 0.1 0 0; 0 0 0.25 0; 0 0 0 0.25], # Defender
         ]
         # 4 beliefs: A->A, A->D, D->A, D->D
         Beliefs([
            Belief(ground_truth_initial_states[Block(1)], initial_belief_covariance[1]),
            Belief(ground_truth_initial_states[Block(2)], initial_belief_covariance[2]),
            Belief(ground_truth_initial_states[Block(1)], initial_belief_covariance[1]),
            Belief(ground_truth_initial_states[Block(2)], initial_belief_covariance[2]),
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
    
    # Pass params to defender cost
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
    
    h_func = h_low_noise # Defaulting to low noise for the refactor baseline
    
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
        
        # 1. Solve Game
        # We need to construct the Game object for the solver
        # Note: Hockey uses `MCPGame` for the LQ approximation in `Hockey.jl` lines 612-615?
        # That seems to be for initialization or comparison?
        # The main loop in `run_receding_horizon_scenario` uses `BeliefGame`.
        
        robust_flags = [c.type == robust for c in [attacker_config, defender_config]]
        robust_indices = findall(robust_flags)
        
        game_horizon = min(params.planning_horizon, params.horizon - t + 1)
        
        # We solve from perspective of each player? 
        # Senate does:
        # for player_idx in player_indices ... construct game from perspective ...
        # Hockey.jl `run_receding_horizon_scenario` does:
        # for ii in 1:dims.n ... game = BeliefGame(...) ... solve(...)
        
        # We will follow Hockey.jl's loop over players to get their plans
        
        sols = Vector{Any}(undef, dimensions.num_players)
        
        for player_idx in 1:dimensions.num_players
            # Construct game from player's perspective if needed, or shared game?
            # Hockey.jl effectively uses shared environment/costs but sets `robust_indices` differently?
            # No, `Hockey.jl` uses `robust[ii] ? [2] : Int[]` depending on if that player thinks the opponent (or themselves?) is robust?
            # Wait, `robust` in `Hockey.jl` is a boolean vector `[false, true]` e.g.
            # If `robust[ii]` is true, then we pass `[2]` (Nature index) to `BeliefGame`.
            # This implies if WE are optimizing robustly, we include Nature?
            
            # Senate: `robust_players = [] ... if player_config.other_player_configs[idx].type == robust ... push!`
            # It seems Senate adds robust players based on configuration.
            
            is_robust = params.player_configs[player_idx].type == robust
            robust_idx = is_robust ? [3] : Int[] # Nature is player 3 in 1-based indexing if p1, p2 present?
            # RobustBeliefGame expects indices of nature players?
            # In Hockey.jl `dims.n` is 2. `robust[ii]` -> `[2]`.
            # Wait, `robust_hockey_game` in `belief_main` uses `[2]`.
            # But `robust_hockey_game` has players [Attacker, Defender, Nature]?
            # `costs = [attacker_cost, defender_cost, nature_cost]`
            # So Nature is index 3.
            
            # The `BeliefGame` struct takes `rigid_players` or `robust_players`?
            # `BeliefGame(..., robust_players)`
            
            # Let's assume index 3 is Nature if robust.
            # If `is_robust` is true, we include nature cost in the game formulation?
            # `costs` defined above includes nature if defender is robust.
            # But what if Attacker is non-robust? They shouldn't see Nature?
            # Hockey.jl `costs` is `[[att, def], [att, def, nature]]` array of arrays?
            # Line 546: `costs = [[attacker_cost, defender_cost], [attacker_cost, defender_cost, nature_cost]]`
            # Line 621: `costs[ii]`
            
            current_player_costs = if is_robust
                [attacker_cost_fn, defender_cost_fn, nature_cost_fn]
            else
                [attacker_cost_fn, defender_cost_fn] # And Nature effectively nonexistent or ignored?
            end
            
            # Robust indices for BeliefGame:
            # If robust, we probably have a nature player index.
            # If `current_player_costs` has 3 elements, nature is likely index 3.
            # `BeliefGame` last arg is `stochastic_players` or `robust_players`?
            # Hockey.jl line 626: `robust[ii] ? [2] : Int[]`.
            # This is mysterious. `[2]` usually means player 2.
            # Is Defender (player 2) treated as the robust player or is Nature index 2?
            # In `belief_main` (line 485), `[2]` is passed.
            # And `robust_hockey_game` has 3 costs.
            # If `robust_hockey_game` has 3 players, why is `[2]` passed?
            # Maybe it indicates which player is being "robustified" against?
            # OR maybe legacy index?
            
            # Let's check `RobustBeliefGame` definition if possible? No time.
            # I will trust `Hockey.jl`.
            # `Hockey.jl` line 27: Attacker=0, Defender=1, Nature=2.
            # But Julia is 1-based. So Attacker=1, Defender=2, Nature=3.
            # So `[2]` means Defender?
            # Maybe it means "The player at index 2 is robust"?
            
            # In `SenateExperiment.jl`, `robust_players` list seems to be `robust_players` indices.
            
            # I will stick to what Hockey does:
            # If this player is robust (defender), pass `[2]`?
            # If attacker is not robust, pass `Int[]`.
            
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
            
            nominal_beliefs, nominal_controls, _ = solve(game; debug=false, warm_start=warm_starts[player_idx])
            
            sols[player_idx] = (nominal_beliefs, nominal_controls)
            
            push!(solution_history[player_idx], (; beliefs=nominal_beliefs, controls=nominal_controls))
            
            # Warm Start Update
            if length(nominal_beliefs) > 1
                shifted_beliefs = nominal_beliefs[2:end]
                shifted_controls = nominal_controls[2:end]
                
                # Zero control padding
                # Need to match control dim: 2 for normal, +4 for nature?
                # Using 0 vector for simplicity or match dims.
                # If robust, `control_dims` might include nature?
                # Senate logic handles this with `BlockVector` and `nominal_controls[end]`.
                
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
        
        # struct BeliefGame
        #     environments::Vector{BeliefEnvironment}
        #     costs::Vector{BeliefCost}
        #     initial_beliefs::Beliefs
        #     horizon::Int
        #     dims::NamedTuple
        #     gt_initial_state::BlockVector
        #     robust_players::Vector{Int}
        # end
        current_beliefs = ekf_update_with_observations(current_beliefs, merged_controls, dummy_game, observations)
                
        current_beliefs.beliefs[2] = copy(current_beliefs.beliefs[4])
        current_beliefs.beliefs[3] = copy(current_beliefs.beliefs[1])
        
    end
    
    # --- Save Results ---
    # --- Save Results ---
    @save output_path gt_state_history observation_history solution_history
    println("Saved results to $(abspath(output_path))")
    
    return (; gt_state_history, observation_history, solution_history)
end
