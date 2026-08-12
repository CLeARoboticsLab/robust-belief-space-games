using Distributed

if !@isdefined(ExperimentRunner)
    include(joinpath(@__DIR__, "..", "src", "ExperimentRunner.jl"))
end
using .ExperimentRunner

"""
    run_parallel_sweep(; cores=4, override=false, save_file_prefix="exp/senate", num_seeds=1000, kwargs...)

Parallel version of run_asymmetric_experiment.

Only expands: random_seed (Monte Carlo), p1_type, p2_type.
All other parameters pass through as-is to run_asymmetric_experiment.

Example:
    run_parallel_sweep(
        p1_type=non_robust,
        p2_type=[non_robust, robust],
        horizon=7,
        num_seeds=100,
        cores=4
    )
"""
# Short keys for the per-task filename prefix. Windows caps a single filename
# component at 255 chars; the asymmetric-experiment suffix alone is ~185, so
# long-form param names in the prefix can push grid sweeps over the limit.
const SWEEP_NAME_ABBREV = Dict(
    :p1_nature_multiplier => "p1nm",
    :p2_nature_multiplier => "p2nm",
    :p1_type => "p1t",
    :p2_type => "p2t",
    :p2_believes_p1_drift_sensor_scale => "p2bp1dss",
    :planning_horizon => "ph",
)

function run_parallel_sweep(;
    cores=4,
    override=false,
    save_file_prefix="exp/senate",
    num_seeds=100,
    offset=0,
    abbrev_names=false,  # use SWEEP_NAME_ABBREV in filenames (new sweeps only — changes names, breaking resume of old ones)
    kwargs...
)
    # Only expand these specific parameters
    expandable_keys = Set([:p1_type, :p2_type, :planning_horizon, :p1_nature_multiplier, :p2_nature_multiplier, :p2_believes_p1_drift_sensor_scale])

    array_params = Dict{Symbol,Vector}()
    fixed_params = Dict{Symbol,Any}()

    for (k, v) in kwargs
        if k in expandable_keys && v isa AbstractVector
            array_params[k] = collect(v)
        else
            fixed_params[k] = v
        end
    end

    # Add random_seed for Monte Carlo
    array_params[:random_seed] = collect((1+offset):(num_seeds+offset))

    # Generate all combinations
    if isempty(array_params)
        combinations = [Dict{Symbol,Any}()]
    else
        param_keys = collect(keys(array_params))
        param_values = [array_params[k] for k in param_keys]
        combinations = vec([
            Dict{Symbol,Any}(zip(param_keys, combo))
            for combo in Iterators.product(param_values...)
        ])
    end

    # Build tasks: each is a kwargs dict with single values
    # Append seed AND other varied params to experiment_name_prefix so files don't overwrite
    base_prefix = get(fixed_params, :experiment_name_prefix, "sweep")
    tasks = Dict{Symbol,Any}[]
    for combo in combinations
        task_kwargs = merge(fixed_params, combo)
        seed = get(combo, :random_seed, 1)
        # Include all non-seed combo params in the prefix to avoid overwrites
        extra_parts = String[]
        for (k, v) in combo
            if k != :random_seed
                name_k = abbrev_names ? get(SWEEP_NAME_ABBREV, k, string(k)) : string(k)
                push!(extra_parts, "$(name_k)_$(v)")
            end
        end
        suffix = isempty(extra_parts) ? "" : "_" * join(sort(extra_parts), "_")
        task_kwargs[:experiment_name_prefix] = "$(base_prefix)/seed_$(seed)$(suffix)"
        push!(tasks, task_kwargs)
    end

    println("\n" * "="^60)
    println("PARALLEL SWEEP: $(length(tasks)) experiments on $cores workers")
    if !isempty(array_params)
        println("Varying: $(keys(array_params))")
    end
    println("="^60 * "\n")

    run_experiment_batch(tasks; cores, override, save_file_prefix)
end
