module RobustBeliefGame

using TrajectoryGamesBase
using TrajectoryGamesExamples
using MixedComplementarityProblems
using Symbolics
using LinearAlgebra
using DifferentiableTrajectoryOptimization: get_constraints_from_box_bounds
using BlockArrays
using Infiltrator


include("ProblemFormulation.jl")
export MCPGame

include("Solve.jl")
export solve







end # module