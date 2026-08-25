module ExperimentRunner

using Distributed
using Hockey
using BlockArrays
using LinearAlgebra
using Distributions

export run_experiment_batch, setup_workers, cleanup_workers, recycle_workers

"""
    recycle_workers()

Kill all existing workers and force garbage collection on master.
"""
function recycle_workers()
    if nworkers() > 0 && nprocs() > 1
        println("Recycling workers: removing $(nworkers()) existing workers...")
        rmprocs(workers())
        GC.gc(true)
        println("All workers removed. Fresh workers will be created on next batch.")
    end
end

function setup_workers(num_procs)
    # Add processes if needed
    current_procs = nprocs()
    if current_procs < num_procs
        addprocs(num_procs - current_procs; exeflags="--project=$(Base.active_project())")
    end

    # Ensure code is loaded on all workers
    @everywhere eval(quote
        using Hockey
        using BlockArrays
        using LinearAlgebra
        using Distributions
        if !@isdefined(ExperimentRunner)
            include("./src/ExperimentRunner.jl")
        end
        using .ExperimentRunner

        function run_single_trial_wrapper(args)
            params, trial_idx = args
            println("\n=== Running Experiment: $(params.name) (Trial $trial_idx) on process $(myid()) ===")
            
            result = run_receding_horizon_trial(params; override=true, trial_number=trial_idx)
            
            status = (
                success = !isnothing(result),
                trial = trial_idx,
                name = params.name,
                pid = myid()
            )
            
            result = nothing
            params = nothing
            
            GC.gc(false)
            
            println("Finished trial $trial_idx on process $(myid()).")
            return status
        end
    end)
end

"""
    run_experiment_batch(params_list::Vector{HockeyParams}; cores=4, auto_recycle=true, trial_offset=0)

Runs a batch of experiments in parallel, distributing individual trials across workers.

If `auto_recycle=true` (default), workers are recycled when the number of tasks exceeds
the number of workers.

`trial_offset` is added to each trial index, allowing extra trials to start from 
existing_count + 1 instead of 1 (to avoid overwriting existing trial files).
"""
function run_experiment_batch(params_list::Vector{HockeyParams}; cores=4, auto_recycle=true,
                               trial_offset::Int=0,
                               trial_offsets::Union{Nothing, Vector{Int}}=nothing)
    if !isnothing(trial_offsets) && length(trial_offsets) != length(params_list)
        error("trial_offsets length ($(length(trial_offsets))) must match params_list length ($(length(params_list)))")
    end
    tasks = []
    for (i, params) in enumerate(params_list)
        off = isnothing(trial_offsets) ? trial_offset : trial_offsets[i]
        for t in 1:params.trials
            push!(tasks, (params, t + off))
        end
    end
    
    num_tasks = length(tasks)

    effective_cores = min(cores, num_tasks)
    
    if auto_recycle && num_tasks > effective_cores
        println("Tasks ($num_tasks) > workers ($effective_cores): recycling for clean slate...")
        recycle_workers()
    end
    
    println("Setting up $effective_cores processes for $num_tasks trials...")
    setup_workers(effective_cores + 1)
    
    println("Starting parallel execution of $num_tasks trials with $(nprocs()) processes...")
    results = pmap(Main.run_single_trial_wrapper, tasks)
    
    cleanup_workers()
    println("All experiments completed.")
    return results
end

"""
    cleanup_workers()

Force garbage collection on all workers to free memory from old problem references.
"""
function cleanup_workers()
    if nprocs() > 1
        println("Forcing garbage collection on $(nworkers()) workers...")
        @everywhere begin
            GC.gc(true)
        end
        GC.gc(true)
        println("Worker cleanup completed.")
    end
end

end # module
