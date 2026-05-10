@enum PlayerType begin
    non_robust = 1
    robust = 2
    nature = 3
    ground_truth_config = 4
end

Base.@kwdef mutable struct PlayerConfig
    player_idx::Int = -1
    type::PlayerType = non_robust
    # For nature configs, the player_idx of the robust player nature is adversary for.
    # Unused (-1) for non-nature configs.
    # Position 3 in the struct is load-bearing: a custom Serialization.deserialize
    # method (see bottom of this file) discriminates old (pre-field-add) data from
    # new data by reading position 3 and checking whether it's an Int.
    nature_target_player_idx::Int = -1

    # Player game params
    ellipsoid_centers::Vector{Vector{Float64}} = [[0.0, 0.0]]
    ellipsoid_radii::Vector{Vector{Float64}} = [[1.0, 1.0]]
    ellipsoidal_cost_weight::Float64 = 0.1
    control_cost_weight::Float64 = 1.0
    terminal_cost_weight::Float64 = 1.0
    nature_multiplier::Float64 = 5.0
    covariance_weight::Float64 = 0.0
    obstacle_sigmoid_scales::Vector{Float64} = [1.0]
    obstacle_sigmoid_offsets::Vector{Float64} = [0.0]
    obstacle_centers::Vector{Vector{Float64}} = [[0.0, 0.0]]
    obstacle_weights::Vector{Float64} = [1.0]
    obstacle_covariance_scale::Float64 = 1.0  # α in: cost = base_cost / (1 + α * tr(Σ))
    obstacle_cost_function::Function = obstacle_cost_v4

    # attraction params
    attraction_numerator::Float64 = 1.0
    attraction_strength::Float64 = 1.0
    attraction_steepness::Float64 = 3.0
    attraction_offset::Float64 = 0.25
    attraction_matrix::Matrix{Float64} = I(3)

    # drift params
    drift_dynamics_scale::Float64 = 0.0
    drift_sensor_scale::Float64 = 0.0

    # sensor params
    sensor_noise_scale::Float64 = 0.1
    
    # dynamics params
    dt::Float64 = 1.0
    
    # These can be populated by the synchronize function
    self_dynamics_model::Union{Function, Nothing} = nothing
    self_dynamics_model_template::Function = base_dynamics
    self_sensor_model::Union{Function, Nothing} = nothing
    self_sensor_model_template::Function = base_sensor_model
    self_non_terminal_cost_model::Union{Function, Nothing} = nothing
    self_non_terminal_cost_model_template::Function = base_non_terminal_cost_function_generator
    self_terminal_cost_model::Union{Function, Nothing} = nothing
    self_terminal_cost_model_template::Function = base_terminal_cost_function_generator
    other_player_configs::Dict{Int, PlayerConfig} = Dict()
    # definitely intended to be set by synchronize function, not by the constructor
    num_senators::Int = -1
    num_activists::Int = -1
    state_dims_per_activist::Vector{Int} = [2, 2, 2]
    control_dims_per_activist::Vector{Int} = [2, 2, 2]
    belief_dims_per_activist::Vector{Tuple{Int, Int}} = [(2, 4), (2, 4), (2, 4)]
    sensor_dims_per_activist::Vector{Int} = [2, 2, 2]
    control_dims_per_senator::Vector{Vector{Int}} = [[2, 2],[2, 2],[2, 2]]
end

Base.isless(a::PlayerConfig, b::PlayerConfig) = isless(a.player_idx, b.player_idx)

# Backward-compatible deserialization for PlayerConfig.
#
# History: the field `nature_target_player_idx::Int` was added at struct position 3
# in commit 733dabb2 ("Allow either player to be robust"). Existing `.dat` archives
# (notably the R-vs-NR `nature_control_sweep` from March) were serialized before the
# field existed; loading them with the new struct triggers EOF / type-mismatch errors
# because Julia's default deserializer reads exactly `nfields(T)` items per struct.
#
# This override reads position 3 and dispatches: an Int means new-format data
# (position 3 IS nature_target_player_idx); anything else means old-format data
# (position 3 is the OLD position 3, ellipsoid_centers::Vector{Vector{Float64}}).
# In the old branch, every subsequent stream value is shifted by one position
# relative to the new struct, so we walk both sides in lockstep.
import Serialization
function Serialization.deserialize(s::Serialization.AbstractSerializer, ::Type{PlayerConfig})
    # Allocate first and register in the serializer's backref table BEFORE reading
    # fields. Without this, any later BACKREF tag pointing to this object fails
    # ("Inconsistent Serializer state ... Attempt to access internal table with key X").
    pc = ccall(:jl_new_struct_uninit, Any, (Any,), PlayerConfig)::PlayerConfig
    Serialization.deserialize_cycle(s, pc)

    fnames = fieldnames(PlayerConfig)
    # Positions 1 and 2 are unchanged across schema versions.
    setfield!(pc, fnames[1], Serialization.deserialize(s))
    setfield!(pc, fnames[2], Serialization.deserialize(s))
    v3 = Serialization.deserialize(s)
    if v3 isa Int
        # New-format stream: v3 IS nature_target_player_idx.
        setfield!(pc, :nature_target_player_idx, v3)
        for i in 4:length(fnames)
            setfield!(pc, fnames[i], Serialization.deserialize(s))
        end
    else
        # Old-format stream: v3 was the OLD position-3 field (ellipsoid_centers).
        # Default the missing field, then write each stream value into the field one
        # past its old position.
        setfield!(pc, :nature_target_player_idx, -1)
        setfield!(pc, fnames[4], v3)
        for i in 5:length(fnames)
            setfield!(pc, fnames[i], Serialization.deserialize(s))
        end
    end
    return pc
end

Base.@kwdef mutable struct SenateParams
    # --- Saving Parameters ---
    name::String = "default"
    
    # --- Ground Truth ---
    ground_truth_dynamics_configs::Dict{Int, PlayerConfig} = Dict(
        1 => DefaultPlayerConfig(player_idx=1, type=ground_truth_config),
        2 => DefaultPlayerConfig(player_idx=2, type=ground_truth_config),
    )
    ground_truth_dynamics_config::PlayerConfig =DefaultPlayerConfig(player_idx=1, type=ground_truth_config)
    ground_truth_dynamics_models::Dict{Int, Function} = Dict()
    ground_truth_dynamics_model::Union{Function, Nothing} = nothing
    ground_truth_sensor_configs::Dict{Int, PlayerConfig} = Dict(
        1 => DefaultPlayerConfig(player_idx=1, type=ground_truth_config),
        2 => DefaultPlayerConfig(player_idx=2, type=ground_truth_config),
    )
    ground_truth_sensor_models::Dict{Int, Function} = Dict()
    ground_truth_initial_states::BlockVector = mortar([
        [0.5, 0.5],
        [0.0, 0.0],
        [0.5, 0.0]
    ])

    # --- Game Parameters ---
    player_configs::Dict{Int, PlayerConfig} = Dict(
        1 => PlayerConfig(player_idx=1, type=non_robust),
        2 => PlayerConfig(player_idx=2, type=robust)
    )
    opinion_dim::Int = 2
    
    # ... Receding Horizon params ...
    planning_horizon::Int = 5
    horizon::Int = 10
    dt::Float64 = 1.0
    process_noise_mean::Vector{Float64} = zeros(6)
    process_noise_covariance::Matrix{Float64} = 0.001 * I(6)
    process_noise_distribution::Union{Distribution, Nothing} = nothing
    sensor_noise_mean::Vector{Float64} = zeros(6)
    sensor_noise_covariance::Matrix{Float64} = 0.001 * I(6)
    sensor_noise_distribution::Union{Distribution, Nothing} = nothing

    # --- Simulation Parameters ---
    trials::Int = 1
    random_seed::Int = 1

    # --- Derived Parameters ---
    num_senators::Int = 3
    num_activists::Int = 2
    state_dims_per_activist::Vector{Int} = [2, 2, 2]
    control_dims_per_activist::Vector{Int} = [2, 2, 2]
    belief_dims_per_activist::Vector{Tuple{Int, Int}} = [(2, 4) for _ in 1:3]
    sensor_dims_per_activist::Vector{Int} = [2, 2, 2]
    initial_beliefs::Union{Beliefs, Nothing} = nothing
    inital_belief_covariance_func::Function = (mean::Vector) -> 0.2 * Symmetric(I(length(mean)))
    control_dims_per_senator::Vector{Vector{Int}} = [[2, 2],[2, 2],[2, 2]]
end

function dims(params::SenateParams)
    return (;
        num_senators=params.num_senators,
        num_activists=params.num_activists,
        state_dims_per_activist=params.state_dims_per_activist,
        control_dims_per_activist=params.control_dims_per_activist,
        belief_dims_per_activist=params.belief_dims_per_activist,
        sensor_dims_per_activist=params.sensor_dims_per_activist,
            
        total_states_dim=vcat([params.player_configs[i].state_dims_per_activist for i in sort(collect(keys(params.player_configs)))]...),
        total_controls_dim=vcat([sum(params.player_configs[i].control_dims_per_activist) for i in sort(collect(keys(params.player_configs)))]...),
        nature_controls_dim=sum([sum(params.player_configs[i].control_dims_per_activist) for i in sort(collect(keys(params.player_configs))) if params.player_configs[i].type == robust]),
        total_beliefs_dim=vcat([params.player_configs[i].belief_dims_per_activist for i in sort(collect(keys(params.player_configs)))]...),
        num_beliefs_per_player=[length(params.player_configs[i].belief_dims_per_activist) for i in sort(collect(keys(params.player_configs)))],
        player_state_dims=params.state_dims_per_activist,
        num_players=length(params.player_configs)
    )
end
