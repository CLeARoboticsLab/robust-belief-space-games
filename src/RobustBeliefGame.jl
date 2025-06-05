module RobustBeliefGame

using TrajectoryGamesBase
using TrajectoryGamesExamples
using MixedComplementarityProblems
using Symbolics
using LinearAlgebra
using DifferentiableTrajectoryOptimization: get_constraints_from_box_bounds
using BlockArrays
using CairoMakie
using Printf

using Infiltrator

include("MCPGame.jl")
export MCPGame

include("ProblemFormulation.jl")
export MCPGame

include("Solve.jl")
export solve







end # module