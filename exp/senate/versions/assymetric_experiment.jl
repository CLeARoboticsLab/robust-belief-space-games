include("base_experiment.jl")
using Senate
using BlockArrays
using Infiltrator
using Base.Threads

function generate_range(param_spec)
    if param_spec isa Tuple && length(param_spec) == 3
        start, stop, step_func = param_spec
        @assert start < stop "Start value must be less than stop value, if single value is intended pass a Float64 or Int"
        
        values = [start]
        current = start
        while current < stop
            next_val = step_func(current)
            @assert next_val > current "Step function must increase value"
            if next_val > stop
                break
            end
            push!(values, next_val)
            current = next_val
        end
        return values
    elseif param_spec isa Float64 || param_spec isa Int
        # Single value
        return [param_spec]
    elseif param_spec isa AbstractArray || param_spec isa AbstractVector
        # Array/Vector of discrete values
        return collect(param_spec)
    else
        error("Invalid parameter specification: $param_spec. Must be Float64/Int, (start, stop, step_func) tuple, or Array/Vector")
    end
end

function build_asymmetric_player_configs(combo, fixed_params)
    player_configs = Dict()
    
    p1_type = haskey(combo, :p1_type) ? combo[:p1_type] : (haskey(fixed_params, :p1_type) ? fixed_params[:p1_type] : non_robust)
    # Player 1 config
    p1_config = DefaultPlayerConfig(player_idx=1, type=p1_type)
    for (key, value) in combo
        if startswith(String(key), "p1_") && !occursin("believes", String(key)) && !occursin("cost_model_template", String(key))
            field_name = Symbol(replace(String(key), "p1_" => ""))
            # Special handling for sigmoid_scale -> obstacle_sigmoid_scales mapping
            if field_name == :sigmoid_scale
                @assert value isa Vector "p1_sigmoid_scale must be a Vector"
                p1_config.obstacle_sigmoid_scales = Vector{Float64}(value)
            elseif hasproperty(p1_config, field_name)
                # Ensure obstacle_weights is always a Vector{Float64} to prevent serialization issues
                if field_name == :obstacle_weights && value isa Vector
                    setproperty!(p1_config, field_name, Vector{Float64}(value))
                else
                    setproperty!(p1_config, field_name, value)
                end
            else
                error("Player 1 config has no property named $field_name")
            end
        end
    end
    # Apply fixed player 1 params
    for (key, value) in fixed_params
        if startswith(String(key), "p1_") && !occursin("cost_model_template", String(key))
            field_name = Symbol(replace(String(key), "p1_" => ""))
            # Special handling for sigmoid_scale -> obstacle_sigmoid_scales mapping
            if field_name == :sigmoid_scale
                @assert value isa Vector "p1_sigmoid_scale must be a Vector"
                p1_config.obstacle_sigmoid_scales = Vector{Float64}(value)
            elseif hasproperty(p1_config, field_name)
                # Ensure obstacle_weights is always a Vector{Float64} to prevent serialization issues
                if field_name == :obstacle_weights && value isa Vector
                    setproperty!(p1_config, field_name, Vector{Float64}(value))
                else
                    setproperty!(p1_config, field_name, value)
                end
            end
        end
    end
    player_configs[1] = p1_config
    
    p2_type = haskey(combo, :p2_type) ? combo[:p2_type] : (haskey(fixed_params, :p2_type) ? fixed_params[:p2_type] : robust)
    # Player 2 config
    p2_config = DefaultPlayerConfig(player_idx=2, type=p2_type)
    for (key, value) in combo
        if startswith(String(key), "p2_") && !occursin("believes", String(key)) && !occursin("cost_model_template", String(key))
            field_name = Symbol(replace(String(key), "p2_" => ""))
            # Special handling for sigmoid_scale -> obstacle_sigmoid_scales mapping
            if field_name == :sigmoid_scale
                @assert value isa Vector "p2_sigmoid_scale must be a Vector"
                p2_config.obstacle_sigmoid_scales = Vector{Float64}(value)
            elseif hasproperty(p2_config, field_name)
                # Ensure obstacle_weights is always a Vector{Float64} to prevent serialization issues
                if field_name == :obstacle_weights && value isa Vector
                    setproperty!(p2_config, field_name, Vector{Float64}(value))
                else
                    setproperty!(p2_config, field_name, value)
                end
            else
                error("Player 2 config has no property named $field_name")
            end
        end
    end

    # Apply fixed player 2 params
    for (key, value) in fixed_params
        if startswith(String(key), "p2_") && !occursin("cost_model_template", String(key))
            field_name = Symbol(replace(String(key), "p2_" => ""))
            # Special handling for sigmoid_scale -> obstacle_sigmoid_scales mapping
            if field_name == :sigmoid_scale
                @assert value isa Vector "p2_sigmoid_scale must be a Vector"
                p2_config.obstacle_sigmoid_scales = Vector{Float64}(value)
            elseif hasproperty(p2_config, field_name)
                # Ensure obstacle_weights is always a Vector{Float64} to prevent serialization issues
                if field_name == :obstacle_weights && value isa Vector
                    setproperty!(p2_config, field_name, Vector{Float64}(value))
                else
                    setproperty!(p2_config, field_name, value)
                end
            end
        end
    end
    player_configs[2] = p2_config

    if haskey(fixed_params, :attraction_matrix)
        p1_config.attraction_matrix = fixed_params[:attraction_matrix]
        p2_config.attraction_matrix = fixed_params[:attraction_matrix]
    end

    dynamics_model = haskey(combo, :dynamics_model_template) ? combo[:dynamics_model_template] : get(fixed_params, :dynamics_model_template, nothing)
    if !isnothing(dynamics_model)
        if dynamics_model == :under_actuated
            p2_config.self_dynamics_model_template = under_actuated_dynamics
            p1_config.self_dynamics_model_template = under_actuated_dynamics
        elseif dynamics_model == :attraction
            p2_config.self_dynamics_model_template = attraction_dynamics_model
            p1_config.self_dynamics_model_template = attraction_dynamics_model
        end
    end

    # Set cost model templates (check combo first for variations, then fixed_params)
    if haskey(combo, :p1_non_terminal_cost_model_template)
        p1_config.self_non_terminal_cost_model_template = combo[:p1_non_terminal_cost_model_template]
    elseif haskey(fixed_params, :p1_non_terminal_cost_model_template)
        p1_config.self_non_terminal_cost_model_template = fixed_params[:p1_non_terminal_cost_model_template]
    end
    if haskey(combo, :p1_terminal_cost_model_template)
        p1_config.self_terminal_cost_model_template = combo[:p1_terminal_cost_model_template]
    elseif haskey(fixed_params, :p1_terminal_cost_model_template)
        p1_config.self_terminal_cost_model_template = fixed_params[:p1_terminal_cost_model_template]
    end
    if haskey(combo, :p1_obstacle_cost_function)
        p1_config.obstacle_cost_function = combo[:p1_obstacle_cost_function]
    elseif haskey(fixed_params, :p1_obstacle_cost_function)
        p1_config.obstacle_cost_function = fixed_params[:p1_obstacle_cost_function]
    end
    if haskey(combo, :p2_non_terminal_cost_model_template)
        p2_config.self_non_terminal_cost_model_template = combo[:p2_non_terminal_cost_model_template]
    elseif haskey(fixed_params, :p2_non_terminal_cost_model_template)
        p2_config.self_non_terminal_cost_model_template = fixed_params[:p2_non_terminal_cost_model_template]
    end
    if haskey(combo, :p2_terminal_cost_model_template)
        p2_config.self_terminal_cost_model_template = combo[:p2_terminal_cost_model_template]
    elseif haskey(fixed_params, :p2_terminal_cost_model_template)
        p2_config.self_terminal_cost_model_template = fixed_params[:p2_terminal_cost_model_template]
    end
    if haskey(combo, :p2_obstacle_cost_function)
        p2_config.obstacle_cost_function = combo[:p2_obstacle_cost_function]
    elseif haskey(fixed_params, :p2_obstacle_cost_function)
        p2_config.obstacle_cost_function = fixed_params[:p2_obstacle_cost_function]
    end

    # Set the base sensor model and re-populate
    p1_config.self_sensor_model_template = covariance_drift_sensor_model
    p2_config.self_sensor_model_template = covariance_drift_sensor_model
    Senate._populate_configs!(p1_config, force=true)
    Senate._populate_configs!(p2_config, force=true)

    # --- Asymmetric Beliefs ---
    # Player 1's beliefs
    p1_belief_about_p2 = deepcopy(p2_config)
    if haskey(combo, :p1_believes_p2_drift_sensor_scale)
        p1_belief_about_p2.drift_sensor_scale = combo[:p1_believes_p2_drift_sensor_scale]
        # Automatically set sensor model based on drift value
        if combo[:p1_believes_p2_drift_sensor_scale] == 0.0
            p1_belief_about_p2.self_sensor_model_template = base_sensor_model
        else
            p1_belief_about_p2.self_sensor_model_template = covariance_drift_sensor_model
        end
    end
    p1_belief_about_p2.type = non_robust

    p1_belief_about_self = deepcopy(p1_config)
    if haskey(combo, :p1_believes_self_drift_sensor_scale)
        p1_belief_about_self.drift_sensor_scale = combo[:p1_believes_self_drift_sensor_scale]
        # Automatically set sensor model based on drift value
        if combo[:p1_believes_self_drift_sensor_scale] == 0.0
            p1_belief_about_self.self_sensor_model_template = base_sensor_model
        else
            p1_belief_about_self.self_sensor_model_template = covariance_drift_sensor_model
        end
    end

    p1_beliefs = Dict(
        1 => p1_belief_about_self,
        2 => p1_belief_about_p2
    )
    if p1_config.type == robust
        nature_idx = max(keys(p1_beliefs)...) + 1
        p1_beliefs[nature_idx] = DefaultNaturePlayerConfig(base_player_config=p1_config, player_idx=nature_idx)
    end
    p1_config.other_player_configs = p1_beliefs

    # Player 2's beliefs
    p2_belief_about_p1 = deepcopy(p1_config)
    if haskey(combo, :p2_believes_p1_drift_sensor_scale)
        p2_belief_about_p1.drift_sensor_scale = combo[:p2_believes_p1_drift_sensor_scale]
        # Automatically set sensor model based on drift value if not explicitly provided
        if !haskey(fixed_params, :p2_believes_p1_sensor_model)
            if combo[:p2_believes_p1_drift_sensor_scale] == 0.0
                p2_belief_about_p1.self_sensor_model_template = base_sensor_model
            else
                p2_belief_about_p1.self_sensor_model_template = covariance_drift_sensor_model
            end
        end
    end
    if haskey(fixed_params, :p2_believes_p1_sensor_model)
        p2_belief_about_p1.self_sensor_model_template = fixed_params[:p2_believes_p1_sensor_model]
    end
    p2_belief_about_p1.type = non_robust

    p2_belief_about_self = deepcopy(p2_config)
    if haskey(combo, :p2_believes_self_drift_sensor_scale)
        p2_belief_about_self.drift_sensor_scale = combo[:p2_believes_self_drift_sensor_scale]
        # Automatically set sensor model based on drift value
        if combo[:p2_believes_self_drift_sensor_scale] == 0.0
            p2_belief_about_self.self_sensor_model_template = base_sensor_model
        else
            p2_belief_about_self.self_sensor_model_template = covariance_drift_sensor_model
        end
    end

    p2_beliefs = Dict(
        1 => p2_belief_about_p1,
        2 => p2_belief_about_self
    )
    if p2_config.type == robust
        nature_idx = max(keys(p2_beliefs)...) + 1
        p2_beliefs[nature_idx] = DefaultNaturePlayerConfig(base_player_config=p2_config, player_idx=nature_idx)
    end
    p2_config.other_player_configs = p2_beliefs

    # --- Clear circular references from belief configs ---
    # The deepcopy operations above copied other_player_configs, creating circular references.
    # We need to clear them from all belief configs to prevent serialization issues.
    for belief_config in values(p1_beliefs)
        belief_config.other_player_configs = Dict{Int, Senate.PlayerConfig}()
    end
    for belief_config in values(p2_beliefs)
        belief_config.other_player_configs = Dict{Int, Senate.PlayerConfig}()
    end
    # Determine the number of senators for this run
    num_senators_val = get(combo, :num_senators, get(fixed_params, :num_senators, 3))

    # Determine other dimensional parameters based on num_senators
    state_dims = fill(2, num_senators_val)
    control_dims = fill(2, num_senators_val)
    belief_dims = [(2, 4) for _ in 1:num_senators_val]
    sensor_dims = fill(2, num_senators_val)
    num_activists_val = 2 # This is fixed at 2 players

    # Create a list of all belief configs that need syncing
    all_belief_configs = []
    append!(all_belief_configs, values(p1_beliefs))
    append!(all_belief_configs, values(p2_beliefs))

    # Apply the derived parameters to every configuration object
    for config in all_belief_configs
        config.num_senators = num_senators_val
        config.num_activists = num_activists_val
        config.state_dims_per_activist = state_dims
        config.control_dims_per_activist = control_dims
        config.belief_dims_per_activist = belief_dims
        config.sensor_dims_per_activist = sensor_dims
        # We also need to populate the model functions themselves, which is done by _populate_configs!
        # This function is not exported, so we call it via the module.
        Senate._populate_configs!(config, force=true)
    end

    return player_configs
end

function build_senate_params(combo, fixed_params, player_configs)
    senate_kwargs = Dict{Symbol, Any}()
    senate_kwargs[:player_configs] = player_configs
    
    # Add experiment parameters
    for (key, value) in combo
        if !startswith(String(key), "p1_") && !startswith(String(key), "p2_") && !startswith(String(key), "p1_believes") && !startswith(String(key), "p2_believes")
            if key in [:dynamics_model_template, :gt_drift_sensor_scale, :gt_drift_dynamics_scale]
                continue
            end
            senate_kwargs[key] = value
        end
    end
    
    # Add fixed parameters (excluding player-specific ones)
    for (key, value) in fixed_params
        if !startswith(String(key), "p1_") && !startswith(String(key), "p2_")
            if key in [:gt_drift_dynamics_scale, :gt_drift_sensor_scale, :dynamics_model_template, :attraction_matrix]
                continue
            end
            senate_kwargs[key] = value
        end
    end
    
    # Handle num_senators changes - update dependent parameters
    if haskey(combo, :num_senators)
        n_sens = combo[:num_senators]
        if !haskey(fixed_params, :state_dims_per_activist)
            senate_kwargs[:state_dims_per_activist] = fill(2, n_sens)
        end
        if !haskey(fixed_params, :control_dims_per_activist)
            senate_kwargs[:control_dims_per_activist] = fill(2, n_sens)
        end
        if !haskey(fixed_params, :ground_truth_initial_states)
            senate_kwargs[:ground_truth_initial_states] = mortar([fill(0.0, 2) for _ in 1:n_sens])
        end
        senate_kwargs[:belief_dims_per_activist] = [(2, 4) for _ in 1:n_sens]
        senate_kwargs[:sensor_dims_per_activist] = fill(2, n_sens)
    end
    
    # Validate horizon ordering if both present
    if haskey(senate_kwargs, :planning_horizon) && haskey(senate_kwargs, :horizon)
        @assert senate_kwargs[:planning_horizon] <= senate_kwargs[:horizon] "Planning horizon must be <= horizon"
    end
    
    if haskey(fixed_params, :gt_drift_dynamics_scale)
        gt_dynamics_configs = Dict(
            1 => DefaultPlayerConfig(player_idx=1, type=ground_truth_config),
            2 => DefaultPlayerConfig(player_idx=2, type=ground_truth_config),
        )
        for (_, config) in gt_dynamics_configs
            config.drift_dynamics_scale = fixed_params[:gt_drift_dynamics_scale]
        end
        senate_kwargs[:ground_truth_dynamics_configs] = gt_dynamics_configs
    end
    # Check combo first (for variations), then fixed_params
    gt_sensor_drift_val = haskey(combo, :gt_drift_sensor_scale) ? combo[:gt_drift_sensor_scale] : get(fixed_params, :gt_drift_sensor_scale, nothing)
    if !isnothing(gt_sensor_drift_val)
        gt_sensor_configs = Dict(
            1 => DefaultPlayerConfig(player_idx=1, type=ground_truth_config),
            2 => DefaultPlayerConfig(player_idx=2, type=ground_truth_config),
        )
        for (_, config) in gt_sensor_configs
            config.drift_sensor_scale = gt_sensor_drift_val
            # Must set sensor model template to one that uses drift_sensor_scale
            config.self_sensor_model_template = covariance_drift_sensor_model
            Senate._populate_configs!(config, force=true)
        end
        senate_kwargs[:ground_truth_sensor_configs] = gt_sensor_configs
    end


    return DefaultSenateParams(; senate_kwargs...)
end


#All experiment parameters are optional and can be single values or (start, stop, step_function) tuples
#Ellipsoid centers and radii are single values only (Vector{Vector{Real}})
#Example: julia> using Revise; includet("exp/senate/versions/assymetric_experiment.jl"); run_asymmetric_experiment(;p1_control_cost_weight = (1.0, 2.0, STEP_ADD(0.5)), experiment_name_prefix="control_cost_test")
function run_asymmetric_experiment(;
    # Player 1 parameters
    p1_ellipsoid_centers = [[3, 1]],  # Single value only: Vector{Vector{Real}}
    p1_ellipsoid_radii = [[1.5, 1]],    # Single value only: Vector{Vector{Real}}
    p1_obstacle_centers = [[1.7, 1.7]],  # Single value only: Vector{Vector{Real}}
    p1_obstacle_weights = [1.0],  # Can vary: Vector{Float64} (single value) or (start, stop, step_func) tuple or Vector{Float64} (multiple values)
    p1_ellipsoidal_cost_weight = nothing,
    p1_control_cost_weight = nothing,
    p1_terminal_cost_weight = nothing,
    p1_nature_multiplier = nothing,
    p1_attraction_strength = nothing,
    p1_drift_dynamics_scale = nothing,
    p1_drift_sensor_scale = nothing,
    p1_type = nothing,
    p1_non_terminal_cost_model_template = nothing,
    p1_terminal_cost_model_template = nothing,
    p1_obstacle_cost_function = nothing,
    p1_sigmoid_scale = nothing,
    p1_obstacle_covariance_scale = nothing,

    # Player 2 parameters
    p2_ellipsoid_centers = [[1, 3]],  # Single value only: Vector{Vector{Real}}
    p2_ellipsoid_radii = [[1, 1.5]],    # Single value only: Vector{Vector{Real}}
    p2_ellipsoidal_cost_weight = nothing,
    p2_control_cost_weight = nothing,
    p2_terminal_cost_weight = nothing,
    p2_nature_multiplier = nothing,
    p2_attraction_strength = nothing,
    dynamics_model_template = nothing,
    p2_drift_dynamics_scale = nothing,
    p2_drift_sensor_scale = nothing,
    p2_type = nothing,
    p2_non_terminal_cost_model_template = nothing,
    p2_terminal_cost_model_template = nothing,
    p2_obstacle_cost_function = nothing,
    p2_obstacle_centers = [[1.7, 1.7]],  # Single value only: Vector{Vector{Real}}
    p2_obstacle_weights = [1.0],  # Can vary: Vector{Float64} (single value) or (start, stop, step_func) tuple or Vector{Float64} (multiple values)
    p2_sigmoid_scale = nothing,
    p2_obstacle_covariance_scale = nothing,
    attraction_matrix = nothing,

    # Asymmetric belief parameters
    p1_believes_self_drift_sensor_scale = nothing,
    p1_believes_p2_drift_sensor_scale = nothing,
    p2_believes_self_drift_sensor_scale = nothing,
    p2_believes_p1_drift_sensor_scale = nothing,
    p2_believes_p1_sensor_model = nothing,

    gt_drift_dynamics_scale = nothing,
    gt_drift_sensor_scale = nothing,
    ground_truth_initial_states = nothing,
    # Experiment parameters
    planning_horizon = nothing,
    horizon = nothing,
    dt = nothing,
    trials = nothing,  # Single int only
    random_seed = nothing,  # Single int only
    num_senators = nothing,
    
    # Control parameters
    override = false,
    experiment_name_prefix = "asymmetric_exp",
    save_file_prefix="exp/senate",
    save_intermediate_results = true,
    num_threads = nothing  # Number of threads to use (nothing = use all available, 1 = sequential)
)
    
    # Collect all parameter variations
    param_variations = Dict()
    fixed_params = Dict()
    
    #region Param parsing
    # Process Player 1 ellipsoid parameters (single values only)
    if !isnothing(p1_ellipsoid_centers)
        @assert p1_ellipsoid_centers isa Vector{<:Vector{<:Real}} "p1_ellipsoid_centers must be Vector{Vector{Real}}"
        fixed_params[:p1_ellipsoid_centers] = p1_ellipsoid_centers
    end
    
    if !isnothing(p1_ellipsoid_radii)
        @assert p1_ellipsoid_radii isa Vector{<:Vector{<:Real}} "p1_ellipsoid_radii must be Vector{Vector{Real}}"
        @assert all(v -> all(x -> x >= 0, v), p1_ellipsoid_radii) "Ellipsoid radii must be non-negative"
        fixed_params[:p1_ellipsoid_radii] = p1_ellipsoid_radii
    end
    
    if !isnothing(p1_obstacle_centers)
        if p1_obstacle_centers isa Vector{<:Vector{<:Real}} || p1_obstacle_centers isa Array{<:Vector{<:Real}}
            fixed_params[:p1_obstacle_centers] = p1_obstacle_centers
        else
            param_variations[:p1_obstacle_centers] = collect(p1_obstacle_centers)
        end
    end
    
    if !isnothing(p1_obstacle_weights)
        if p1_obstacle_weights isa Tuple && length(p1_obstacle_weights) == 3
            # Range specification: (start, stop, step_func)
            weight_range = generate_range(p1_obstacle_weights)
            @assert all(x -> x >= 0, weight_range) "Obstacle weights must be non-negative"
            param_variations[:p1_obstacle_weights] = [Vector{Float64}([w]) for w in weight_range]
        elseif p1_obstacle_weights isa Real
            # Single float: wrap in vector
            @assert p1_obstacle_weights >= 0 "Obstacle weights must be non-negative"
            fixed_params[:p1_obstacle_weights] = Vector{Float64}([p1_obstacle_weights])
        elseif p1_obstacle_weights isa Vector{<:Real}
            @assert all(x -> x >= 0, p1_obstacle_weights) "Obstacle weights must be non-negative"
            if length(p1_obstacle_weights) == 1
                fixed_params[:p1_obstacle_weights] = Vector{Float64}(p1_obstacle_weights)
            else
                param_variations[:p1_obstacle_weights] = [Vector{Float64}([w]) for w in p1_obstacle_weights]
            end
        else
            error("p1_obstacle_weights must be Real, Vector{Real}, or (start, stop, step_func) tuple")
        end
    end

    if !isnothing(p2_obstacle_centers)
        if p2_obstacle_centers isa Vector{<:Vector{<:Real}} || p2_obstacle_centers isa Array{<:Vector{<:Real}}
            fixed_params[:p2_obstacle_centers] = p2_obstacle_centers
        else
            param_variations[:p2_obstacle_centers] = collect(p2_obstacle_centers)
        end
    end

    if !isnothing(p2_obstacle_weights)
        if p2_obstacle_weights isa Tuple && length(p2_obstacle_weights) == 3
            weight_range = generate_range(p2_obstacle_weights)
            @assert all(x -> x >= 0, weight_range) "Obstacle weights must be non-negative"
            param_variations[:p2_obstacle_weights] = [Vector{Float64}([w]) for w in weight_range]
        elseif p2_obstacle_weights isa Real
            # Single float: wrap in vector
            @assert p2_obstacle_weights >= 0 "Obstacle weights must be non-negative"
            fixed_params[:p2_obstacle_weights] = Vector{Float64}([p2_obstacle_weights])
        elseif p2_obstacle_weights isa Vector{<:Real}
            @assert all(x -> x >= 0, p2_obstacle_weights) "Obstacle weights must be non-negative"
            if length(p2_obstacle_weights) == 1
                fixed_params[:p2_obstacle_weights] = Vector{Float64}(p2_obstacle_weights)
            else
                param_variations[:p2_obstacle_weights] = [Vector{Float64}([w]) for w in p2_obstacle_weights]
            end
        else
            error("p2_obstacle_weights must be Real, Vector{Real}, or (start, stop, step_func) tuple")
        end
    end

    # Process Player 1 range parameters
    if !isnothing(p1_ellipsoidal_cost_weight)
        param_variations[:p1_ellipsoidal_cost_weight] = generate_range(p1_ellipsoidal_cost_weight)
        @assert all(x -> x >= 0, param_variations[:p1_ellipsoidal_cost_weight]) "Cost weights must be non-negative"
    end
    if !isnothing(p1_control_cost_weight)
        param_variations[:p1_control_cost_weight] = generate_range(p1_control_cost_weight)
        @assert all(x -> x >= 0, param_variations[:p1_control_cost_weight]) "Cost weights must be non-negative"
    end
    if !isnothing(p1_terminal_cost_weight)
        param_variations[:p1_terminal_cost_weight] = generate_range(p1_terminal_cost_weight)
        @assert all(x -> x >= 0, param_variations[:p1_terminal_cost_weight]) "Cost weights must be non-negative"
    end
    if !isnothing(p1_nature_multiplier)
        param_variations[:p1_nature_multiplier] = generate_range(p1_nature_multiplier)
        @assert all(x -> x >= 0, param_variations[:p1_nature_multiplier]) "Nature multiplier must be non-negative"
    end
    if !isnothing(p1_attraction_strength)
        param_variations[:p1_attraction_strength] = generate_range(p1_attraction_strength)
    end
    if !isnothing(p1_drift_dynamics_scale)
        param_variations[:p1_drift_dynamics_scale] = generate_range(p1_drift_dynamics_scale)
    end
    if !isnothing(p1_drift_sensor_scale)
        param_variations[:p1_drift_sensor_scale] = generate_range(p1_drift_sensor_scale)
    end
    if !isnothing(p1_type)
        if p1_type isa AbstractArray || p1_type isa AbstractVector
            param_variations[:p1_type] = collect(p1_type)
        else
            fixed_params[:p1_type] = p1_type
        end
    end
    if !isnothing(p1_non_terminal_cost_model_template)
        if p1_non_terminal_cost_model_template isa AbstractArray || p1_non_terminal_cost_model_template isa AbstractVector
            param_variations[:p1_non_terminal_cost_model_template] = collect(p1_non_terminal_cost_model_template)
        else
            fixed_params[:p1_non_terminal_cost_model_template] = p1_non_terminal_cost_model_template
        end
    end
    if !isnothing(p1_terminal_cost_model_template)
        if p1_terminal_cost_model_template isa AbstractArray || p1_terminal_cost_model_template isa AbstractVector
            param_variations[:p1_terminal_cost_model_template] = collect(p1_terminal_cost_model_template)
        else
            fixed_params[:p1_terminal_cost_model_template] = p1_terminal_cost_model_template
        end
    end
    if !isnothing(p1_obstacle_cost_function)
        if p1_obstacle_cost_function isa AbstractArray || p1_obstacle_cost_function isa AbstractVector
            param_variations[:p1_obstacle_cost_function] = collect(p1_obstacle_cost_function)
        else
            fixed_params[:p1_obstacle_cost_function] = p1_obstacle_cost_function
        end
    end
    if !isnothing(p1_sigmoid_scale)
        if p1_sigmoid_scale isa Tuple && length(p1_sigmoid_scale) == 3
            # Range specification: (start, stop, step_func)
            # Generate range and wrap each value in a vector since obstacle_sigmoid_scales expects Vector{Float64}
            scale_range = generate_range(p1_sigmoid_scale)
            @assert all(x -> x >= 0, scale_range) "Sigmoid scales must be non-negative"
            # Ensure each variation is a proper Vector{Float64} to prevent serialization issues
            param_variations[:p1_sigmoid_scale] = [Vector{Float64}([s]) for s in scale_range]
        elseif p1_sigmoid_scale isa Vector{<:Real}
            @assert all(x -> x >= 0, p1_sigmoid_scale) "Sigmoid scales must be non-negative"
            if length(p1_sigmoid_scale) == 1
                # Single value: fixed parameter - ensure it's Vector{Float64}
                fixed_params[:p1_sigmoid_scale] = Vector{Float64}(p1_sigmoid_scale)
            else
                # Multiple values: variations (each value becomes [value])
                # Ensure each variation is a proper Vector{Float64} to prevent serialization issues
                param_variations[:p1_sigmoid_scale] = [Vector{Float64}([s]) for s in p1_sigmoid_scale]
            end
        else
            error("p1_sigmoid_scale must be Vector{Real} or (start, stop, step_func) tuple")
        end
    end
    if !isnothing(p1_obstacle_covariance_scale)
        param_variations[:p1_obstacle_covariance_scale] = generate_range(p1_obstacle_covariance_scale)
        @assert all(x -> x >= 0, param_variations[:p1_obstacle_covariance_scale]) "Obstacle covariance scale must be non-negative"
    end

    # Process Player 2 ellipsoid parameters (single values only)
    if !isnothing(p2_ellipsoid_centers)
        @assert p2_ellipsoid_centers isa Vector{<:Vector{<:Real}} "p2_ellipsoid_centers must be Vector{Vector{Real}}"
        fixed_params[:p2_ellipsoid_centers] = p2_ellipsoid_centers
    end
    
    if !isnothing(p2_ellipsoid_radii)
        @assert p2_ellipsoid_radii isa Vector{<:Vector{<:Real}} "p2_ellipsoid_radii must be Vector{Vector{Real}}"
        @assert all(v -> all(x -> x >= 0, v), p2_ellipsoid_radii) "Ellipsoid radii must be non-negative"
        fixed_params[:p2_ellipsoid_radii] = p2_ellipsoid_radii
    end
    
    # Process Player 2 range parameters
    if !isnothing(p2_ellipsoidal_cost_weight)
        param_variations[:p2_ellipsoidal_cost_weight] = generate_range(p2_ellipsoidal_cost_weight)
        @assert all(x -> x >= 0, param_variations[:p2_ellipsoidal_cost_weight]) "Cost weights must be non-negative"
    end
    if !isnothing(p2_control_cost_weight)
        param_variations[:p2_control_cost_weight] = generate_range(p2_control_cost_weight)
        @assert all(x -> x >= 0, param_variations[:p2_control_cost_weight]) "Cost weights must be non-negative"
    end
    if !isnothing(p2_terminal_cost_weight)
        param_variations[:p2_terminal_cost_weight] = generate_range(p2_terminal_cost_weight)
        @assert all(x -> x >= 0, param_variations[:p2_terminal_cost_weight]) "Cost weights must be non-negative"
    end
    if !isnothing(p2_nature_multiplier)
        param_variations[:p2_nature_multiplier] = generate_range(p2_nature_multiplier)
        @assert all(x -> x >= 0, param_variations[:p2_nature_multiplier]) "Nature multiplier must be non-negative"
    end
    if !isnothing(p2_attraction_strength)
        param_variations[:p2_attraction_strength] = generate_range(p2_attraction_strength)
    end
    if !isnothing(p2_drift_dynamics_scale)
        param_variations[:p2_drift_dynamics_scale] = generate_range(p2_drift_dynamics_scale)
    end
    if !isnothing(p2_drift_sensor_scale)
        param_variations[:p2_drift_sensor_scale] = generate_range(p2_drift_sensor_scale)
    end
    if !isnothing(p2_type)
        if p2_type isa AbstractArray || p2_type isa AbstractVector
            param_variations[:p2_type] = collect(p2_type)
        else
            fixed_params[:p2_type] = p2_type
        end
    end
    if !isnothing(p2_non_terminal_cost_model_template)
        if p2_non_terminal_cost_model_template isa AbstractArray || p2_non_terminal_cost_model_template isa AbstractVector
            param_variations[:p2_non_terminal_cost_model_template] = collect(p2_non_terminal_cost_model_template)
        else
            fixed_params[:p2_non_terminal_cost_model_template] = p2_non_terminal_cost_model_template
        end
    end
    if !isnothing(p2_terminal_cost_model_template)
        if p2_terminal_cost_model_template isa AbstractArray || p2_terminal_cost_model_template isa AbstractVector
            param_variations[:p2_terminal_cost_model_template] = collect(p2_terminal_cost_model_template)
        else
            fixed_params[:p2_terminal_cost_model_template] = p2_terminal_cost_model_template
        end
    end
    if !isnothing(p2_obstacle_cost_function)
        if p2_obstacle_cost_function isa AbstractArray || p2_obstacle_cost_function isa AbstractVector
            param_variations[:p2_obstacle_cost_function] = collect(p2_obstacle_cost_function)
        else
            fixed_params[:p2_obstacle_cost_function] = p2_obstacle_cost_function
        end
    end
    if !isnothing(p2_sigmoid_scale)
        if p2_sigmoid_scale isa Tuple && length(p2_sigmoid_scale) == 3
            # Range specification: (start, stop, step_func)
            # Generate range and wrap each value in a vector since obstacle_sigmoid_scales expects Vector{Float64}
            scale_range = generate_range(p2_sigmoid_scale)
            @assert all(x -> x >= 0, scale_range) "Sigmoid scales must be non-negative"
            # Ensure each variation is a proper Vector{Float64} to prevent serialization issues
            param_variations[:p2_sigmoid_scale] = [Vector{Float64}([s]) for s in scale_range]
        elseif p2_sigmoid_scale isa Vector{<:Real}
            @assert all(x -> x >= 0, p2_sigmoid_scale) "Sigmoid scales must be non-negative"
            if length(p2_sigmoid_scale) == 1
                # Single value: fixed parameter - ensure it's Vector{Float64}
                fixed_params[:p2_sigmoid_scale] = Vector{Float64}(p2_sigmoid_scale)
            else
                # Multiple values: variations (each value becomes [value])
                # Ensure each variation is a proper Vector{Float64} to prevent serialization issues
                param_variations[:p2_sigmoid_scale] = [Vector{Float64}([s]) for s in p2_sigmoid_scale]
            end
        else
            error("p2_sigmoid_scale must be Vector{Real} or (start, stop, step_func) tuple")
        end
    end
    if !isnothing(p2_obstacle_covariance_scale)
        param_variations[:p2_obstacle_covariance_scale] = generate_range(p2_obstacle_covariance_scale)
        @assert all(x -> x >= 0, param_variations[:p2_obstacle_covariance_scale]) "Obstacle covariance scale must be non-negative"
    end
    if !isnothing(attraction_matrix)
        fixed_params[:attraction_matrix] = attraction_matrix
    end
    
    if !isnothing(p1_believes_self_drift_sensor_scale)
        param_variations[:p1_believes_self_drift_sensor_scale] = generate_range(p1_believes_self_drift_sensor_scale)
    end
    if !isnothing(p1_believes_p2_drift_sensor_scale)
        param_variations[:p1_believes_p2_drift_sensor_scale] = generate_range(p1_believes_p2_drift_sensor_scale)
    end
    if !isnothing(p2_believes_self_drift_sensor_scale)
        param_variations[:p2_believes_self_drift_sensor_scale] = generate_range(p2_believes_self_drift_sensor_scale)
    end
    if !isnothing(p2_believes_p1_drift_sensor_scale)
        param_variations[:p2_believes_p1_drift_sensor_scale] = generate_range(p2_believes_p1_drift_sensor_scale)
    end
    if !isnothing(p2_believes_p1_sensor_model)
        fixed_params[:p2_believes_p1_sensor_model] = p2_believes_p1_sensor_model
    end

    # Process experiment parameters
    if !isnothing(planning_horizon)
        param_variations[:planning_horizon] = generate_range(planning_horizon)
        @assert all(x -> x >= 1, param_variations[:planning_horizon]) "Planning horizon must be at least 1"
    end
    if !isnothing(horizon)
        param_variations[:horizon] = generate_range(horizon)
        @assert all(x -> x >= 1, param_variations[:horizon]) "Horizon must be at least 1"
    end
    if !isnothing(num_senators)
        param_variations[:num_senators] = generate_range(num_senators)
        @assert all(x -> x >= 1, param_variations[:num_senators]) "Number of senators must be at least 1"
    end
    
    # Single value parameters (no range)
    if !isnothing(trials) && trials isa Int
        @assert trials >= 1 "Trials must be at least 1"
        fixed_params[:trials] = trials
    end
    if !isnothing(random_seed) && random_seed isa Int
        @assert random_seed >= 1 "Random seed must be at least 1"
        fixed_params[:random_seed] = random_seed
    end

    if !isnothing(dynamics_model_template)
        if dynamics_model_template isa AbstractArray || dynamics_model_template isa AbstractVector
            values = collect(dynamics_model_template)
            @assert all(v -> v in [:default, :under_actuated, :attraction], values) "dynamics_model_template must be :default, :under_actuated, or :attraction"
            param_variations[:dynamics_model_template] = values
        else
            @assert dynamics_model_template in [:default, :under_actuated, :attraction] "dynamics_model_template must be :default, :under_actuated, or :attraction"
            fixed_params[:dynamics_model_template] = dynamics_model_template
        end
    end
    if !isnothing(gt_drift_dynamics_scale)
        fixed_params[:gt_drift_dynamics_scale] = gt_drift_dynamics_scale
    end
    if !isnothing(gt_drift_sensor_scale)
        if gt_drift_sensor_scale isa AbstractArray || gt_drift_sensor_scale isa AbstractVector
            values = collect(gt_drift_sensor_scale)
            if length(values) == 1
                fixed_params[:gt_drift_sensor_scale] = values[1]
            else
                param_variations[:gt_drift_sensor_scale] = values
            end
        else
            fixed_params[:gt_drift_sensor_scale] = gt_drift_sensor_scale
        end
    end
    if !isnothing(ground_truth_initial_states)
        if ground_truth_initial_states isa AbstractArray || ground_truth_initial_states isa AbstractVector
            param_variations[:ground_truth_initial_states] = collect(ground_truth_initial_states)
        else
            fixed_params[:ground_truth_initial_states] = ground_truth_initial_states
        end
    end
    if !isnothing(dt)
        if dt isa Tuple && length(dt) == 3
            # Range specification: (start, stop, step_func)
            param_variations[:dt] = generate_range(dt)
            @assert all(x -> x > 0, param_variations[:dt]) "dt must be positive"
        elseif dt isa AbstractArray || dt isa AbstractVector
            # Array/Vector of values
            dt_values = collect(dt)
            @assert all(x -> x > 0, dt_values) "dt must be positive"
            if length(dt_values) == 1
                # Single value: fixed parameter
                fixed_params[:dt] = dt_values[1]
            else
                # Multiple values: variations
                param_variations[:dt] = dt_values
            end
        elseif dt isa Real
            # Single value: fixed parameter
            @assert dt > 0 "dt must be positive"
            fixed_params[:dt] = dt
        else
            error("dt must be Real, Vector{Real}, or (start, stop, step_func) tuple")
        end
    end
    #endregion
    
    # Generate all combinations
    param_keys = collect(keys(param_variations))
    param_values = [param_variations[k] for k in param_keys]
    
    if isempty(param_values)
        combinations = [Dict()]
    else
        combinations = vec([
            Dict(zip(param_keys, combo))
            for combo in Iterators.product(param_values...)
        ])
    end
    
    #region Debug Run Info
    println("\n" * "="^60)
    println("ASYMMETRIC EXPERIMENT RUN INFO")
    println("="^60)
    println("Total number of experiments to run: $(length(combinations))")
    if !isempty(param_keys)
        println("\nParameter variations:")
        for key in param_keys
            vals = param_variations[key]
            println("  $key: $(length(vals)) value(s) → $vals")
        end
    end
    if !isempty(fixed_params)
        println("\nFixed parameters:")
        for (key, value) in fixed_params
            println("  $key: $value")
        end
    end
    println("="^60 * "\n")
    #endregion
    
    # Clear old files with the same prefix if override is true
    if override
        runs_dir = "exp/senate/outputs/runs"
        if isdir(runs_dir)
            files_to_delete = []
            for file in readdir(runs_dir)
                if startswith(file, experiment_name_prefix) && endswith(file, ".dat")
                    filepath = joinpath(runs_dir, file)
                    push!(files_to_delete, filepath)
                end
            end
            if !isempty(files_to_delete)
                println("Clearing $(length(files_to_delete)) old file(s) with prefix '$experiment_name_prefix'...")
                for filepath in files_to_delete
                    rm(filepath)
                    println("  Deleted: $(basename(filepath))")
                end
            end
        end
    end
    
    # Helper function to generate experiment name
    function abbrev_from_key(k::AbstractString)
        join(first.(split(k, "_")))
    end
    
    function abbrev_key(k::AbstractString)
        for prefix in ("p1", "p2")
            if startswith(k, prefix)
                rest = k[length(prefix)+2:end]
                return prefix * abbrev_from_key(rest)
            end
        end
        return abbrev_from_key(k)
    end
    
    function generate_exp_name(combo, prefix)
        exp_name = prefix
        for (key, value) in combo
            kstr = String(key)   
            abbr = abbrev_key(kstr)
            exp_name *= "_$(abbr)_$(value)"
        end
        return exp_name
    end
    
    all_results = []
    
    # Multi-threading or sequential mode
    available_threads = Threads.nthreads()
    if num_threads === nothing
        use_threads = available_threads
    else
        use_threads = min(num_threads, available_threads)
    end
    
    if use_threads > 1
        println("Using $(use_threads) thread(s) for parallel execution (available: $available_threads)")
    else
        println("Running experiments sequentially (1 thread)")
    end
    
    results_lock = ReentrantLock()
    print_lock = ReentrantLock()
    
    # Run experiments (parallel if use_threads > 1)
    if use_threads > 1
        # Multi-threaded execution
        @threads for idx in 1:length(combinations)
            combo = combinations[idx]
            
            # Thread-safe printing
            lock(print_lock) do
                println("\n=== Running experiment $idx/$(length(combinations)) (Thread $(threadid())) ===")
                println("Parameters: ", combo)
            end
            
            try
                # Build configs and params
                player_configs = build_asymmetric_player_configs(combo, fixed_params)
                params = build_senate_params(combo, fixed_params, player_configs)
                
                # Generate experiment name
                exp_name = generate_exp_name(combo, experiment_name_prefix)
                
                # Run experiment
                results = run_experiment(
                    params;
                    override=override,
                    experiment_name=exp_name,
                    save_file_prefix=save_file_prefix
                )
                
                # Thread-safe result collection
                result_entry = (
                    params=combo,
                    fixed=fixed_params,
                    results=results,
                    name=exp_name
                )
                
                lock(results_lock) do
                    push!(all_results, result_entry)
                end
                
                lock(print_lock) do
                    println("✓ Completed experiment $idx/$(length(combinations)): $exp_name")
                end
                
                # Note: Intermediate saves disabled during parallel execution for thread safety
                # Results will be saved once at the end
                
            catch e
                lock(print_lock) do
                    println("✗ Error in experiment $idx/$(length(combinations)): $e")
                    showerror(stdout, e, catch_backtrace())
                end
            end
        end
    else
        # Sequential execution
        for (idx, combo) in enumerate(combinations)
            println("\n=== Running experiment $idx/$(length(combinations)) ===")
            println("Parameters: ", combo)
            
            try
                # Build configs and params
                player_configs = build_asymmetric_player_configs(combo, fixed_params)
                params = build_senate_params(combo, fixed_params, player_configs)
                
                # Generate experiment name
                exp_name = generate_exp_name(combo, experiment_name_prefix)
                
                # Run experiment
                results = run_experiment(
                    params;
                    override=override,
                    experiment_name=exp_name,
                    save_file_prefix=save_file_prefix
                )
                
                # Store results with metadata
                push!(all_results, (
                    params=combo,
                    fixed=fixed_params,
                    results=results,
                    name=exp_name
                ))
                
                println("✓ Completed experiment $idx/$(length(combinations)): $exp_name")
                
                # Save intermediate cumulative results if needed
                if save_intermediate_results
                    mass_filename = "$(save_file_prefix)/outputs/merged/$(experiment_name_prefix)_mass_results.dat"
                    println("Saving all results to $mass_filename")
                    
                    open(mass_filename, "w") do f
                        serialize(f, all_results)
                    end
                end
                
            catch e
                println("✗ Error in experiment $idx/$(length(combinations)): $e")
                showerror(stdout, e, catch_backtrace())
            end
        end
    end
    
    println("\n=== Completed all $(length(combinations)) experiments ===")

    # Save all results to a single file
    mass_filename = "$(save_file_prefix)/outputs/merged/$(experiment_name_prefix)_mass_results.dat"
    println("Saving all results to $mass_filename")

    open(mass_filename, "w") do f
        serialize(f, all_results)
    end
end
