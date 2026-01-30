using Distributed
using BlockArrays
using LinearAlgebra
using Distributions
using Senate

# Include the runner module if not already defined
if !@isdefined(ExperimentRunner)
    @eval include("../src/ExperimentRunner.jl")
end
using .ExperimentRunner

"""
    apply_config!(params::SenateParams, config::Dict)

Recursively applies a configuration dictionary to a SenateParams object.
Supports nested dictionaries for `player_configs`.
"""
function apply_config!(params::SenateParams, config::Dict)
    for (key, value) in config
        if key == :player_configs
            for (pid, pconfig) in value
                if !haskey(params.player_configs, pid)
                    println("Warning: Player config for player $pid not found in params.")
                    continue
                end
                apply_player_config!(params.player_configs[pid], pconfig)
            end
        elseif hasproperty(params, key)
            setproperty!(params, key, value)
        else
            println("Warning: Key $key not found in SenateParams.")
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
        if key in (:name, :horizon, :planning_horizon)
            continue # these are excluded from name generation
        elseif key == :player_configs
            for (pid, pconfig) in sort(collect(value), by=x->x[1])
                 for (pkey, pval) in sort(collect(pconfig), by=x->string(x[1]))
                    if pkey == :type
                        continue
                    end
                    k_str = replace(string(pkey), "control_cost_weight" => "cc", "nature_multiplier" => "nm", "ellipsoidal_cost_weight" => "ec", "terminal_cost_weight" => "tc")
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
                   override=false,
                   base_params_generator=nothing)

Runs a generic parameter sweep.
- `config_list`: List of dictionaries containing overrides.
- `output_subdir`: Subdirectory in `outputs/` to save files.
- `override`: Whether to override existing results.
- `base_params_generator`: Optional function to generate baseline params. If nothing, uses default `DefaultSenateParams()`.
"""
function run_param_sweep(config_list::Vector{Dict{Symbol, Any}};
                        cores=4,
                        output_subdir="param_sweep",
                        override=false,
                        base_params_generator=nothing)

    println("Preparing parameter sweep with $(length(config_list)) configurations...")

    # Build list of (params, experiment_name) tuples
    tasks = Tuple{SenateParams, String}[]

    # Use absolute path relative to this script location (exp/senate/versions/param_sweep.jl)
    # Target: exp/senate/outputs
    output_dir = joinpath(@__DIR__, "..", "outputs", output_subdir)
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    for (i, config) in enumerate(config_list)
        # Create base params
        params = isnothing(base_params_generator) ? DefaultSenateParams() : base_params_generator()

        # Apply overrides
        apply_config!(params, config)

        # Determine experiment name
        if haskey(config, :name)
            exp_name = config[:name]
        else
            # Generate descriptive name
            desc_name = generate_name_from_config(config)
            if isempty(desc_name)
                desc_name = "config_$i"
            end
            exp_name = "senate_sweep_$(desc_name)"
        end

        push!(tasks, (params, exp_name))
    end

    # Run batch
    results = run_experiment_batch(tasks; cores=cores, override=override, save_file_prefix=joinpath(@__DIR__, ".."))

    println("Sweep completed.")
    return results
end


