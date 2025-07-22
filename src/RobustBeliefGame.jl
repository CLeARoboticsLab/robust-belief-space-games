module RobustBeliefGame

using TrajectoryGamesBase
using TrajectoryGamesExamples
using MixedComplementarityProblems
using Symbolics
using LinearAlgebra
using DifferentiableTrajectoryOptimization: get_constraints_from_box_bounds
using BlockArrays
using BlockDiagonals: BlockDiagonal
using CairoMakie
using Printf
using ForwardDiff
using FiniteDifferences
using Statistics

using Infiltrator

include("MCPGame.jl")
export MCPGame

include("ProblemFormulation.jl")
export MCPGame

include("Solve.jl")
export solve

include("BeliefSpaceUtils.jl")
export Belief, Beliefs, means, covs, dims, BeliefGame, BeliefEnvironment, BeliefCost, vec, unvec

include("EKF.jl")
export ekf_update, ekf_update_gradient, ekf_update_with_observations

include("RobustBeliefSpaceSolver.jl")
export solve

include("Plotting.jl")
export plot_feed_forward_norms


DEBUG = true
DEBUG_FILE = "./exp/hockey/outputs/belief_diagnostics.txt"

ϵ = 1e-7
clip_norm = 100

ForwardDiff.set_preferences!(ForwardDiff, "nansafe_mode" => true)

end # module