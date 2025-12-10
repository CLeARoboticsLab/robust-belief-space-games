using Distributed
using BlockArrays
using LinearAlgebra
using Distributions
using Hockey

function setup_workers(num_procs)
    # Add processes if needed
    current_procs = nprocs()
    if current_procs < num_procs
        addprocs(num_procs - current_procs; exeflags="--project=$(Base.active_project())")
    end

    # Ensure code is loaded on all workers (idempotent-ish)
    @everywhere eval(quote
        using Hockey
        using BlockArrays
        using LinearAlgebra
        using Distributions

        function run_single_trial(weight)
            println("\n=== Running with Defender Control Cost Weight: $weight on process $(myid()) ===")
            
            # Configure Params
            params = HockeyParams(name="hockey_cc_$(weight)")
            
            # Defender (Player 2) - Robust
            params.player_configs[2].control_cost_weight = weight
            params.player_configs[2].type = robust
            
            # Attacker (Player 1) - Non-Robust
            params.player_configs[1].type = non_robust
            
            # Run Trials
            println("Running trials for weight $weight...")
            results = run_receding_horizon_trials(params; override=true)
            
            println("Finished trials for weight $weight.")
            return results
        end
    end)
end

"""
    run_control_cost_study(; weights=[0.01, 0.1, 0.5, 1.0], cores=4)

Runs the receding horizon experiment sweep over defender control cost weights in parallel.
`cores` specifies the total number of processes (master + workers) to use.
"""
function run_control_cost_study(; weights=[0.01, 0.1, 0.5, 1.0], cores=4)
    println("Setting up $cores processes...")
    cores = min(cores, length(weights))
    setup_workers(cores)
    println("Starting parallel execution with $(nprocs()) processes...")
    
    # Use pmap to distribute tasks
    results = pmap(run_single_trial, weights)
    
    println("All trials completed.")
    return results
end