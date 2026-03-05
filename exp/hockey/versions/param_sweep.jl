using Distributed
using BlockArrays
using LinearAlgebra
using Distributions
using Hockey
include(joinpath(@__DIR__, "..", "src", "ExperimentRunner.jl"))  
using .ExperimentRunner

"""
    apply_config!(params::HockeyParams, config::Dict)

Recursively applies a configuration dictionary to a HockeyParams object.
Supports nested dictionaries for `player_configs`.
"""
function apply_config!(params::HockeyParams, config::Dict)
    for (key, value) in config
        if key == :player_configs
            for (pid, pconfig) in value
                if !haskey(params.player_configs, pid)
                    println("Warning: Player config for player $pid not found in params.")
                    continue
                end
                apply_player_config!(params.player_configs[pid], pconfig)
            end
        elseif key == :sensor_model && value isa String
            params.sensor_model = h_noise_dict[value]
        elseif hasproperty(params, key)
            setproperty!(params, key, value)
        else
            println("Warning: Key $key not found in HockeyParams.")
        end
    end
end

function apply_player_config!(player_config::PlayerConfig, config::Dict)
    for (key, value) in config
        if hasproperty(player_config, key)
            setproperty!(player_config, key, value)
        else
             println("Warning: Key $key not found in PlayerConfig.")
        end
    end
end

"""
    generate_name_from_config(config::Dict)

Generates a descriptive name string from a flat or nested configuration dictionary.
"""
function generate_name_from_config(config::Dict)
    # Helper to format values cleanly (avoid floating point artifacts)
    function format_val(v)
        if v isa AbstractFloat
            # Round to avoid precision artifacts, then remove trailing zeros
            rounded = round(v, sigdigits=10)
            if rounded == floor(rounded)
                return string(Int(rounded))
            else
                return string(rounded)
            end
        else
            return string(v)
        end
    end
    
    parts = String[]
    for (key, value) in sort(collect(config), by=x->string(x[1]))
        if key in (:name, :trials, :horizon, :planning_horizon)
            continue # these are excluded from name generation
        elseif key == :player_configs
            for (pid, pconfig) in sort(collect(value), by=x->x[1])
                 for (pkey, pval) in sort(collect(pconfig), by=x->string(x[1]))
                    if pkey == :type 
                        continue
                    end
                    k_str = replace(string(pkey), "control_cost_weight" => "cc", "nature_control_cost_weight" => "ncc", "nature_bounds_cost_weight" => "nbc", "steal_dist_weight" => "sd", "terminal_cost_weight" => "tc")
                    push!(parts, "p$(pid)_$(k_str)$(format_val(pval))")
                 end
            end
        else
            k_str = replace(string(key), "sensor_model" => "sns")
            push!(parts, "$(k_str)$(format_val(value))")
        end
    end
    return join(parts, "_")
end

"""
    run_param_sweep(config_list::Vector{Dict{Symbol, Any}}; 
                   cores=4, 
                   output_subdir="param_sweep",
                   base_params_generator=nothing)

Runs a generic parameter sweep.
- `config_list`: List of dictionaries containing overrides.
- `output_subdir`: Subdirectory in `outputs/` to save files.
- `base_params_generator`: Optional function to generate baseline params. If nothing, uses default `HockeyParams()`.
"""
function run_param_sweep(config_list::Vector{Dict{Symbol, Any}}; 
                        cores=4, 
                        output_subdir="param_sweep",
                        base_params_generator=nothing)
    
    println("Preparing parameter sweep with $(length(config_list)) configurations...")
    
    params_list = HockeyParams[]
    
    # Use absolute path relative to this script location (exp/hockey/versions/param_sweep.jl)
    # Target: exp/hockey/outputs
    output_dir = joinpath(@__DIR__, "..", "outputs", output_subdir)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    for (i, config) in enumerate(config_list)
        # Create base params
        params = isnothing(base_params_generator) ? HockeyParams() : base_params_generator()
        
        # Apply overrides
        apply_config!(params, config)
        
        # Set output directory
        sub_sub_dir = generate_name_from_config(config)
        params.output_dir = joinpath(output_dir, sub_sub_dir)
        if !isdir(params.output_dir)
            mkpath(params.output_dir)
        end
        
        # Determine name
        if haskey(config, :name)
            params.name = config[:name]
        else
            # Generate descriptive name
            desc_name = generate_name_from_config(config)
            if isempty(desc_name)
                desc_name = "config_$i"
            end
            params.name = "hockey_sweep_$(desc_name)"
        end
        
        push!(params_list, params)
    end
    
    # Run batch
    results = run_experiment_batch(params_list; cores=cores)
    
    println("Sweep completed.")
    return results
end


