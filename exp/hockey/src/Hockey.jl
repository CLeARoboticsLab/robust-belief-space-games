module Hockey

# using GLMakie
using JLD2
using FileIO
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
# include("HockeyVisuals.jl")

export 
    # Dynamics
    basic_dynamics, M_static, M_state_based,
    
    # Cost
    attacker_cost, defender_cost, nature_cost,
    attacker_non_terminal_cost, defender_non_terminal_cost, nature_non_terminal_cost,
    attacker_terminal_cost, defender_terminal_cost, nature_terminal_cost,
    player_cost_components,
    
    # Sensor
    h_state_based, h_noise, h_low_noise, h_noise_dict,
    
    # Experiment
    run_receding_horizon_trial, run_receding_horizon_trials

    # Visuals
    # load_hockey_results, visualize_receding_horizon_solutions_multi_figure

end
