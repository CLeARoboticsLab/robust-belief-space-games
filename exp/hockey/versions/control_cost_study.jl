using Hockey
using BlockArrays
using LinearAlgebra
using Distributions

"""
    run_control_cost_study(; weights=[0.1, 1.0, 10.0])

Runs the receding horizon experiment sweep over defender control cost weights.
"""
function run_control_cost_study(; weights=[0.01, 0.1, 0.5, 1.0])
    
    for weight in weights
        println("\n=== Running with Defender Control Cost Weight: $weight ===")
        
        # Configure Params
        params = HockeyParams(name="hockey_cc_$(weight)")
        
        # Defender (Player 2) - Robust
        params.player_configs[2].control_cost_weight = weight
        params.player_configs[2].type = robust
        
        # Attacker (Player 1) - Non-Robust
        params.player_configs[1].type = non_robust
        
        # Run Trials
        println("Running trials...")
        run_receding_horizon_trials(params; override=true)
        
        println("Finished trials for weight $weight.")
    end
end