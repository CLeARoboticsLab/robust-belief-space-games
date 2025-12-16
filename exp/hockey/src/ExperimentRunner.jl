module ExperimentRunner

using Distributed
using Hockey
using BlockArrays
using LinearAlgebra
using Distributions

export run_experiment_batch, setup_workers

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
        include("./src/ExperimentRunner.jl")
        using .ExperimentRunner

        function run_single_trial_wrapper(params::HockeyParams)
            try
                println("\n=== Running Experiment: $(params.name) on process $(myid()) ===")
                
                # Run Trials
                println("Running trials for $(params.name)...")
                results = run_receding_horizon_trials(params; override=true)
                
                println("Finished trials for $(params.name).")
                return results
            catch e
                println("ERROR on process $(myid()): $e")
                Base.showerror(stdout, e, catch_backtrace())
                rethrow(e)
            end
        end
    end)
end

"""
    run_experiment_batch(params_list::Vector{HockeyParams}; cores=4)

Runs a batch of experiments in parallel.
"""
function run_experiment_batch(params_list::Vector{HockeyParams}; cores=4)
    num_experiments = length(params_list)
    effective_cores = min(cores, num_experiments)
    println("Setting up $effective_cores processes...")
    setup_workers(effective_cores + 1)
    
    println("Starting parallel execution with $(nprocs()) processes...")
    
    # Use pmap to distribute tasks.
    # The function `run_single_trial_wrapper` is defined in Main on all workers by setup_workers.
    # We pass the global function itself directly to avoid closure serialization issues.
    results = pmap(Main.run_single_trial_wrapper, params_list)
    
    println("All experiments completed.")
    return results
end

end # module
