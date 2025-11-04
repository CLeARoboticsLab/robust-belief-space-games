include("base_experiment.jl")
using Senate
using BlockArrays
using Infiltrator
#Sample Step Functions
const STEP_ADD = n -> (x -> x + n)
const STEP_MULTIPLY = n -> (x -> n * x)
const STEP_MULTIPLY_CEIL = n -> (x -> ceil(Int, n * x))
const STEP_POWER = n -> (x -> x^n)

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
    else
        error("Invalid parameter specification: $param_spec")
    end
end

function build_asymmetric_player_configs(combo, fixed_params)
    player_configs = Dict()
    
    p1_type = haskey(fixed_params, :p1_type) ? fixed_params[:p1_type] : non_robust
    # Player 1 config
    p1_config = DefaultPlayerConfig(player_idx=1, type=p1_type)
    for (key, value) in combo
        if startswith(String(key), "p1_") && !occursin("believes", String(key))
            field_name = Symbol(replace(String(key), "p1_" => ""))
            if hasproperty(p1_config, field_name)
                setproperty!(p1_config, field_name, value)
            else
                error("Player 1 config has no property named $field_name")
            end
        end
    end
    # Apply fixed player 1 params
    for (key, value) in fixed_params
        if startswith(String(key), "p1_")
            field_name = Symbol(replace(String(key), "p1_" => ""))
            if hasproperty(p1_config, field_name)
                setproperty!(p1_config, field_name, value)
            end
        end
    end
    player_configs[1] = p1_config
    
    p2_type = haskey(fixed_params, :p2_type) ? fixed_params[:p2_type] : robust
    # Player 2 config
    p2_config = DefaultPlayerConfig(player_idx=2, type=p2_type)
    for (key, value) in combo
        if startswith(String(key), "p2_") && !occursin("believes", String(key))
            field_name = Symbol(replace(String(key), "p2_" => ""))
            if hasproperty(p2_config, field_name)
                setproperty!(p2_config, field_name, value)
            else
                error("Player 2 config has no property named $field_name")
            end
        end
    end
    
    # Apply fixed player 2 params
    for (key, value) in fixed_params
        if startswith(String(key), "p2_")
            field_name = Symbol(replace(String(key), "p2_" => ""))
            if hasproperty(p2_config, field_name)
                setproperty!(p2_config, field_name, value)
            end
        end
    end
    player_configs[2] = p2_config

    if haskey(fixed_params, :attraction_matrix)
        p1_config.attraction_matrix = fixed_params[:attraction_matrix]
        p2_config.attraction_matrix = fixed_params[:attraction_matrix]
    end

    if haskey(fixed_params, :dynamics_model_template)
        if fixed_params[:dynamics_model_template] == :under_actuated
            p2_config.self_dynamics_model_template = under_actuated_dynamics
        elseif fixed_params[:dynamics_model_template] == :attraction
            p2_config.self_dynamics_model_template = attraction_dynamics_model
        end
        if fixed_params[:dynamics_model_template] == :under_actuated
            p1_config.self_dynamics_model_template = under_actuated_dynamics
        elseif fixed_params[:dynamics_model_template] == :attraction
            p1_config.self_dynamics_model_template = attraction_dynamics_model
        end
    end

    # --- Asymmetric Beliefs ---
    # Player 1's beliefs
    p1_belief_about_p2 = deepcopy(p2_config)
    if haskey(combo, :p1_believes_p2_drift_sensor_scale)
        p1_belief_about_p2.drift_sensor_scale = combo[:p1_believes_p2_drift_sensor_scale]
    end
    p1_belief_about_p2.type = non_robust

    p1_belief_about_self = deepcopy(p1_config)
    if haskey(combo, :p1_believes_self_drift_sensor_scale)
        p1_belief_about_self.drift_sensor_scale = combo[:p1_believes_self_drift_sensor_scale]
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
    end
    p2_belief_about_p1.type = non_robust

    p2_beliefs = Dict(
        1 => p2_belief_about_p1,
        2 => deepcopy(p2_config)
    )
    if p2_config.type == robust
        nature_idx = max(keys(p2_beliefs)...) + 1
        p2_beliefs[nature_idx] = DefaultNaturePlayerConfig(base_player_config=p2_config, player_idx=nature_idx)
    end
    p2_config.other_player_configs = p2_beliefs

    # --- Synchronize derived parameters ---
    # This is critical because we manually constructed other_player_configs,
    # bypassing the normal synchronization that happens in DefaultSenateParams.

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
            senate_kwargs[key] = value
        end
    end
    
    # Add fixed parameters (excluding player-specific ones)
    for (key, value) in fixed_params
        if !startswith(String(key), "p1_") && !startswith(String(key), "p2_")
            # These are meta-parameters for the experiment script, not for SenateParams
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
    if haskey(fixed_params, :gt_drift_sensor_scale)
        gt_sensor_configs = Dict(
            1 => DefaultPlayerConfig(player_idx=1, type=ground_truth_config),
            2 => DefaultPlayerConfig(player_idx=2, type=ground_truth_config),
        )
        for (_, config) in gt_sensor_configs
            config.drift_sensor_scale = fixed_params[:gt_drift_sensor_scale]
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
    p1_ellipsoidal_cost_weight = nothing,
    p1_control_cost_weight = nothing,
    p1_terminal_cost_weight = nothing,
    p1_nature_multiplier = nothing,
    p1_attraction_strength = nothing,
    p1_drift_dynamics_scale = nothing,
    p1_drift_sensor_scale = nothing,
    p1_type = nothing,
    
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
    attraction_matrix = nothing,

    # Asymmetric belief parameters
    p1_believes_p2_drift_sensor_scale = nothing,
    p2_believes_p1_drift_sensor_scale = nothing,

    gt_drift_dynamics_scale = nothing,
    gt_drift_sensor_scale = nothing,
    ground_truth_initial_states = nothing,
    # Experiment parameters
    planning_horizon = nothing,
    horizon = nothing,
    trials = nothing,  # Single int only
    random_seed = nothing,  # Single int only
    num_senators = nothing,
    
    # Control parameters
    override = false,
    experiment_name_prefix = "asymmetric_exp",
    save_intermediate_results = true
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
        fixed_params[:p1_type] = p1_type
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
        fixed_params[:p2_type] = p2_type
    end
    if !isnothing(attraction_matrix)
        fixed_params[:attraction_matrix] = attraction_matrix
    end
    
    # Process asymmetric belief parameters
    if !isnothing(p1_believes_p2_drift_sensor_scale)
        param_variations[:p1_believes_p2_drift_sensor_scale] = generate_range(p1_believes_p2_drift_sensor_scale)
    end
    if !isnothing(p2_believes_p1_drift_sensor_scale)
        param_variations[:p2_believes_p1_drift_sensor_scale] = generate_range(p2_believes_p1_drift_sensor_scale)
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
        @assert dynamics_model_template in [:default, :under_actuated, :attraction] "dynamics_model_template must be :default, :under_actuated, or :attraction"
        fixed_params[:dynamics_model_template] = dynamics_model_template
    end
    if !isnothing(gt_drift_dynamics_scale)
        fixed_params[:gt_drift_dynamics_scale] = gt_drift_dynamics_scale
    end
    if !isnothing(gt_drift_sensor_scale)
        fixed_params[:gt_drift_sensor_scale] = gt_drift_sensor_scale
    end
    if !isnothing(ground_truth_initial_states)
        fixed_params[:ground_truth_initial_states] = ground_truth_initial_states
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
            println("  $key: $(length(param_variations[key])) value(s)")
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
    
    # Run experiments for each combination
    all_results = []
    
    for (idx, combo) in enumerate(combinations)
        println("\n=== Running experiment $idx/$(length(combinations)) ===")
        println("Parameters: ", combo)
        
        # Build configs and params
        player_configs = build_asymmetric_player_configs(combo, fixed_params)
        params = build_senate_params(combo, fixed_params, player_configs)
        
        # Generate experiment name
        abbrev_from_key(k::AbstractString) = join(first.(split(k, "_")))

        function abbrev_key(k::AbstractString)
            for prefix in ("p1", "p2")
                if startswith(k, prefix)
                    rest = k[length(prefix)+2:end]
                    return prefix * abbrev_from_key(rest)
                end
            end
            return abbrev_from_key(k)
        end

        exp_name = experiment_name_prefix
        for (key, value) in combo
            kstr = String(key)   
            abbr = abbrev_key(kstr)
            exp_name *= "_$(abbr)_$(value)"
        end
        
        # Run experiment
        results = run_experiment(
            params;
            override=override,
            experiment_name=exp_name
        )
        
        # Store results with metadata
        push!(all_results, (
            params=combo,
            fixed=fixed_params,
            results=results,
            name=exp_name
        ))

        # Save intermediate cumulative results if needed
        if save_intermediate_results
            mass_filename = "exp/senate/outputs/merged/$(experiment_name_prefix)_mass_results.dat"
            println("Saving all results to $mass_filename")

            open(mass_filename, "w") do f
                serialize(f, all_results)
            end
        end
    end
    
    println("\n=== Completed all $(length(combinations)) experiments ===")

    # Save all results to a single file
    mass_filename = "exp/senate/outputs/merged/$(experiment_name_prefix)_mass_results.dat"
    println("Saving all results to $mass_filename")

    open(mass_filename, "w") do f
        serialize(f, all_results)
    end
end
