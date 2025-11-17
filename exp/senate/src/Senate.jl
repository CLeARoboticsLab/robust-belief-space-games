module Senate

using LinearAlgebra
using BlockArrays
using RobustBeliefGame
using Distributions
using Random
using Statistics
using Serialization
using Infiltrator
import RobustBeliefGame.dims

include("./SenateParams.jl")
export SenateParams, PlayerConfig, PlayerType, non_robust, robust, nature, ground_truth_config, dims
include("./dynamics.jl")
export base_dynamics, under_actuated_dynamics, attraction_dynamics_model, drift_dynamics_model, drift_dynamics_model_generator
include("./cost.jl")
export non_terminal_cost_components, terminal_cost_components, ellipsoidal_cost, control_cost, base_non_terminal_cost_function_generator, base_terminal_cost_function_generator
export covariance_non_terminal_cost_function_generator, covariance_terminal_cost_function_generator, obstacle_non_terminal_cost_function_generator, obstacle_terminal_cost_function_generator
include("./sensor.jl")
export base_sensor_model, drift_sensor_model, covariance_drift_sensor_model
include("./SenateParamUtils.jl")
export _sync_params_to_configs_dims!, _sync_params_to_configs_other_configs!, DefaultPlayerConfig, DefaultSenateParams, DefaultNaturePlayerConfig
include("./SenateExperiment.jl")
export run_receding_horizon_trials, run_receding_horizon_trial, init_checks

end # module Senate
