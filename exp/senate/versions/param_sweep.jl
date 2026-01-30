using Distributed

if !@isdefined(ExperimentRunner)
    include("../src/ExperimentRunner.jl")
end
using .ExperimentRunner

"""
    run_parallel_sweep(; cores=4, override=false, save_file_prefix="exp/senate", kwargs...)

Parallel version of run_asymmetric_experiment.

Pass parameter arrays just like run_asymmetric_experiment - this function
expands them into a grid and distributes single experiments to workers.

Example:
    run_parallel_sweep(
        p1_type=[non_robust],
        p2_type=[non_robust, robust],
        p2_nature_multiplier=[0.5, 8.0],
        horizon=7,
        cores=4
    )
"""
function run_parallel_sweep(;
    cores=4,
    override=false,
    save_file_prefix="exp/senate",
    kwargs...
)
    # Separate array params (to expand) from single params (fixed)
    array_params = Dict{Symbol,Vector}()
    fixed_params = Dict{Symbol,Any}()

    for (k, v) in kwargs
        if v isa AbstractVector && !(v isa Vector{<:Vector})  # arrays but not nested vectors like ellipsoid_centers
            array_params[k] = collect(v)
        else
            fixed_params[k] = v
        end
    end

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
    tasks = Dict{Symbol,Any}[]
    for combo in combinations
        task_kwargs = merge(fixed_params, combo)
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
