module Hockey

using GLMakie
using BlockArrays
using LinearAlgebra
using Distributions
using Random

include("HockeyParams.jl")
export PlayerType, PlayerConfig, HockeyParams, dims, non_robust, robust, nature, ground_truth_config


include("dynamics.jl")
include("sensor.jl")
include("cost.jl")
include("HockeyExperiment.jl")

# Visuals - we will include them here or let the user include them separately? 
# Senate includes `SenateVisuals.jl` in `Senate.jl`? 
# Checking Senate.jl list_dir: `Senate.jl` was small.
# Creating a separate `HockeyVisuals.jl` file in src is good practice.
include("HockeyVisuals.jl")

export 
    # Dynamics
    basic_dynamics, M_static, M_state_based,
    
    # Cost
    attacker_cost, defender_cost, nature_cost,
    attacker_non_terminal_cost, defender_non_terminal_cost, nature_non_terminal_cost,
    attacker_terminal_cost, defender_terminal_cost, nature_terminal_cost,
    
    # Sensor
    h_state_based, h_noise, h_low_noise, h_dict,
    
    # Experiment
    run_receding_horizon_trial, run_receding_horizon_trials

end
