using Distributed

if !@isdefined(ExperimentRunner)
    include("../src/ExperimentRunner.jl")
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
function run_parallel_sweep(;
    cores=4,
    override=false,
    save_file_prefix="exp/senate",
    num_seeds=100,
    kwargs...
)
    # Only expand these specific parameters
    expandable_keys = Set([:p1_type, :p2_type])

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
    array_params[:random_seed] = collect(1:num_seeds)

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
                push!(extra_parts, "$(k)_$(v)")
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
