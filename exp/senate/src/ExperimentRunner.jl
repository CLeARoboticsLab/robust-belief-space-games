module ExperimentRunner

using Distributed
using Senate
using BlockArrays
using LinearAlgebra
using Distributions
using Serialization

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

function setup_workers(num_procs; save_file_prefix="exp/senate")
    # Add processes if needed
    current_procs = nprocs()
    if current_procs < num_procs
        addprocs(num_procs - current_procs; exeflags="--project=$(Base.active_project())")
    end

    # Resolve assymetric_experiment.jl to an absolute path at the master so the
    # include below works regardless of each worker's cwd or source-path state.
    # (Previously this was a `./exp/senate/versions/...` relative path, which
    # broke when workers' include base wasn't the repo root — e.g. when the
    # entry point lived under exp/senate/versions/.)
    asym_path = abspath(joinpath(@__DIR__, "..", "versions", "assymetric_experiment.jl"))

    # Build the setup expression on the master so the absolute path is baked in
    # via `$asym_path` interpolation, then evaluate it on every proc.
    setup_expr = quote
        using Senate
        using BlockArrays
        using LinearAlgebra
        using Distributions
        using Serialization

        # Load assymetric_experiment.jl (includes base_experiment.jl)
        if !@isdefined(run_asymmetric_experiment)
            include($asym_path)
        end

        function run_single_experiment_wrapper(args)
            kwargs, save_file_prefix, override = args
            println("\n=== Running Experiment on process $(myid()) ===")

            # Just call run_asymmetric_experiment with single values
            run_asymmetric_experiment(;
                kwargs...,
                override=override,
                save_file_prefix=save_file_prefix
            )

            GC.gc(false)
            println("Finished on process $(myid()).")
            return (success=true, pid=myid())
        end
    end

    Distributed.remotecall_eval(Main, procs(), setup_expr)
end

"""
    run_experiment_batch(tasks; cores=4, auto_recycle=true, override=false, save_file_prefix="exp/senate")

Runs a batch of senate experiments in parallel using pmap.

Each task is a Dict of kwargs to pass to run_asymmetric_experiment.
Workers just call run_asymmetric_experiment with single values.
"""
function run_experiment_batch(
    tasks::Vector{<:Dict};
    cores=4,
    auto_recycle=true,
    override=false,
    save_file_prefix="exp/senate"
)
    # Wrap tasks with save_file_prefix and override
    wrapped_tasks = [(kwargs, save_file_prefix, override) for kwargs in tasks]

    num_tasks = length(wrapped_tasks)
    effective_cores = min(cores, num_tasks)

    if auto_recycle && num_tasks > effective_cores
        println("Tasks ($num_tasks) > workers ($effective_cores): recycling for clean slate...")
        recycle_workers()
    end

    println("Setting up $effective_cores processes for $num_tasks experiments...")
    setup_workers(effective_cores + 1; save_file_prefix=save_file_prefix)

    println("Starting parallel execution of $num_tasks experiments with $(nprocs()) processes...")
    results = pmap(Main.run_single_experiment_wrapper, wrapped_tasks)

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
