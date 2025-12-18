

@enum PlayerType begin
    non_robust = 1
    robust = 2
    nature = 3
    ground_truth_config = 4
end

Base.@kwdef mutable struct PlayerConfig
    player_idx::Int = -1
    type::PlayerType = non_robust
    
    # Costs
    control_cost_weight::Float64 = 0.05
    terminal_cost_weight::Float64 = 2.0
    
    boundary_cost_weight::Float64 = 10.0
    steal_dist_weight::Float64 = 0.1
    shot_uncertainty_weight::Float64 = 20.0
    
    # Nature properites (relevant if player is robust)
    nature_control_cost_weight::Float64 = 300.0
    nature_bounds_cost_weight::Float64 = 10.0
    
    # Dynamics (if needed specific per player)
    
    # Derived
    self_dynamics_model::Union{Function, Nothing} = nothing
    self_sensor_model::Union{Function, Nothing} = nothing
    self_non_terminal_cost_model::Union{Function, Nothing} = nothing
    self_terminal_cost_model::Union{Function, Nothing} = nothing
    
    other_player_configs::Dict{Int, PlayerConfig} = Dict()
    
    # Dims (for compatibility)
    state_dims_per_activist::Vector{Int} = [4, 4] # Using 4 for Hockey (x, y, vx, vy)
    control_dims_per_activist::Vector{Int} = [2, 2] # (ax, ay)
    belief_dims_per_activist=[(4, 16), (4, 16)]
end

Base.@kwdef mutable struct HockeyParams
    name::String = "hockey_default"
    
    # Game Params
    player_configs::Dict{Int, PlayerConfig} = Dict(
        1 => PlayerConfig(player_idx=1, type=non_robust, control_cost_weight=2.0), # Attacker
        2 => PlayerConfig(player_idx=2, type=robust, control_cost_weight=0.5)      # Defender
    )
    
    # Hockey Specific
    goal_position::Vector{Vector{Float64}} = [[0.25, -1.5], [-0.25, -1.5]]
    
    # Receding Horizon
    planning_horizon::Int = 5
    horizon::Int = 10
    dt::Float64 = 0.3
    
    # Noise
    process_noise_distribution::Union{Distribution, Nothing} = MvNormal(zeros(8), 0.001 * I(8)) # 2 players * 4 states
    sensor_noise_distribution::Union{Distribution, Nothing} = MvNormal(zeros(8), 0.001 * I(8))
    
    # Dimensions
    state_dim::Int = 4
    control_dim::Int = 2
    
    # Initial Conditions
    ground_truth_initial_states::BlockVector = mortar([
        [0.0, 5.0, 0.5, 0.0],  # Attacker
        [0.0, 1.5, 0.0, 0.0],  # Defender
    ])
    initial_beliefs::Union{Any, Nothing} = nothing # Will be mapped to Beliefs
    
    # Helper for Senate compat (if needed)
    control_dims_per_activist::Vector{Int} = [2, 2]
    
    trials::Int = 1
    random_seed::Int = 1
end

function dims(params::HockeyParams)
    # Mapping to TrajectoryGamesBase/Solver dims
    # Assuming standard mapping:
    # num_players = 2
    # states per player = 4
    # controls per player = 2
    
    return (;
        num_players=length(params.player_configs),
        
        # Breakdown
        state_dims_per_activist=[4, 4], player_state_dims=[4,4],# Fixed for Hockey 
        control_dims_per_activist=[2, 2],
        belief_dims_per_activist=[(4, 16), (4, 16)],
        nature_controls_dim=8,

        total_controls_dim = [2,2],
        num_beliefs_per_player = [2,2],

        total_states_dim=vcat([params.player_configs[i].state_dims_per_activist for i in sort(collect(keys(params.player_configs)))]...),
        total_beliefs_dim=vcat([params.player_configs[i].belief_dims_per_activist for i in sort(collect(keys(params.player_configs)))]...),       
        
        states=[4, 4],
        controls=[2, 2],
        belief=[4, 4, 4, 4], # 4 beliefs (P1->P1, P1->P2, P2->P1, P2->P2) each of dim 4 (mean) + cov?
        # The solver `BeliefGame` seems to need `dims.belief` to be the dimension of the belief state vector?
        # In `Hockey.jl` line 545: `belief=[state_dim for _ in 1:4]` which is `[4, 4, 4, 4]`.
        # This likely refers to the mean dimension. The covariance is handled separately or implicitly?
        # Actually `Belief` struct usually has mean and covariance.
        
        sensor=[4, 4] # Sensor output dim
    )
end

function params_to_name(params::HockeyParams)
    # Construct a descriptive name
    # e.g. "hockey_robust_def_cc0.1" or "hockey_baseline"
    
    parts = String[]
    push!(parts, "hockey")
    
    # Check player types
    p1_type = params.player_configs[1].type
    p2_type = params.player_configs[2].type
    
    if p2_type == robust
        push!(parts, "robust_def")
        clean_weight = replace(string(params.player_configs[2].control_cost_weight), "." => "p")
        push!(parts, "cc_$(clean_weight)")
    else
        push!(parts, "baseline")
    end
    
    return join(parts, "_")
end
