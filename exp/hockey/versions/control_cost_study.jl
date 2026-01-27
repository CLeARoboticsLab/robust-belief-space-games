using Distributed
using BlockArrays
using LinearAlgebra
using Distributions
using Hockey

# Include the runner module
include("../src/ExperimentRunner.jl")
using .ExperimentRunner

"""
    generate_params(weight)

Creates a HockeyParams object with the specified defender control cost weight.
"""
function generate_params(weight)
    # Configure Params
    params = HockeyParams(name="hockey_cc_$(weight)_test")
    
    # Defender (Player 2) - Robust
    params.player_configs[2].control_cost_weight = weight
    params.player_configs[2].type = robust
    
    # Attacker (Player 1) - Non-Robust
    params.player_configs[1].type = non_robust
    params.player_configs[1].control_cost_weight = 5.0
    
    return params
end

"""
    run_control_cost_study(; weights=[0.01, 0.1, 0.5, 1.0], cores=4)

Runs the receding horizon experiment sweep over defender control cost weights using the ExperimentRunner.
"""
function run_control_cost_study(; weights=[0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cores=4)
    println("Preparing study for weights: $weights")
    
    # Generate parameters for all trials
    params_list = [generate_params(w) for w in weights]
    
    # Run batch
    results = run_experiment_batch(params_list; cores=cores)
    
    println("Study completed.")
    return results
end