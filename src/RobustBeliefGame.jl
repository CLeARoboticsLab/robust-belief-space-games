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

using Infiltrator

include("MCPGame.jl")
export MCPGame

include("ProblemFormulation.jl")
export MCPGame

include("Solve.jl")
export solve

include("BeliefSpaceUtils.jl")
export Belief, Beliefs, means, covs, dims, BeliefGame, BeliefEnvironment, BeliefCost

include("EKF.jl")
export ekf_update, ekf_update_gradient

include("RobustBeliefSpaceSolver.jl")
export solve


DEBUG = true
DEBUG_FILE = "./exp/hockey/outputs/belief_diagnostics.txt"

ϵ = 1e-6
gradient_clip = 100

end # module