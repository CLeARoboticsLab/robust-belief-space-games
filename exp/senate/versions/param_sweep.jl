using Distributed

if !@isdefined(ExperimentRunner)
    include("../src/ExperimentRunner.jl")
end
using .ExperimentRunner

"""
    run_parallel_sweep(; cores=4, override=false, save_file_prefix="exp/senate", num_seeds=1000, kwargs...)

Parallel version of run_asymmetric_experiment.

Pass parameter arrays just like run_asymmetric_experiment - this function
expands them into a grid and distributes single experiments to workers.

Each parameter combination is run with random seeds 1:num_seeds for Monte Carlo.

Example:
    run_parallel_sweep(
        p1_type=[non_robust],
        p2_type=[non_robust, robust],
        p2_nature_multiplier=[0.5, 8.0],
        horizon=7,
        num_seeds=100,  # 100 seeds per config
        cores=4
    )
"""
function run_parallel_sweep(;
    cores=4,
    override=false,
    save_file_prefix="exp/senate",
    num_seeds=1000,
    kwargs...
)
    # Separate array params (to expand) from single params (fixed)
    array_params = Dict{Symbol,Vector}()
    fixed_params = Dict{Symbol,Any}()

    for (k, v) in kwargs
        if v isa BlockVector
            # BlockVector (e.g. ground_truth_initial_states) should not be expanded
            fixed_params[k] = v
        elseif v isa AbstractVector && !(v isa Vector{<:Vector})
            array_params[k] = collect(v)
        else
            fixed_params[k] = v
        end
    end

    # Add random_seed as an array param for Monte Carlo
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
    # Append seed to experiment_name_prefix so files don't overwrite
    base_prefix = get(fixed_params, :experiment_name_prefix, "sweep")
    tasks = Dict{Symbol,Any}[]
    for combo in combinations
        task_kwargs = merge(fixed_params, combo)
        seed = get(combo, :random_seed, 1)
        task_kwargs[:experiment_name_prefix] = "$(base_prefix)/seed_$(seed)"
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
