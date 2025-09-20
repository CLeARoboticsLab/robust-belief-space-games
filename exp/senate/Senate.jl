module Congress
using Infiltrator
using RobustBeliefGame
using LinearAlgebra
using BlockArrays
using Distributions
using Random
using Statistics

using JLD2
using FileIO

export receding_horizon_main


@enum ActivistID begin
    non_robust_activist = 1
    robust_activist = 2
end
cost_params = Dict(
    non_robust_activist => [(;a = 0.0, b = 1.0, c = 0.0, d = 1.0, control_weight=(;direction=1, effort=1))],
    robust_activist => [(;a = 0.0, b = 1.0, c = 0.0, d = 1.0, control_weight=(;direction=1, effort=1))],
) 
# Ideally, we can "save" cost functions by storing the parameters of components used to generate the cost.
# This can somewhat approximate multi-modal preferences by generating multiple ellipsoids.
num_senators = 3
state_dim_per_senator = (;mean=2, covariance=4)
# x = [x_pos, y_pos] per senator
control_dim_per_senator = (;effort=1)
# u = [direction, effort] per senator per activist

# All changes to game parameters should flow from info above


ϵ = eps()
num_activists = length(instances(ActivistID))


function ellipsoidal_preference_generator(a::Float64, b::Float64, c::Float64, d::Float64)
    function (x, y)
        return (x-a)^2/b^2 + (y-c)^2/d^2
    end
end
function control_cost_generator(effort::Float64)
    function (u::BlockVector)
        effort * sum(control[2] for control in u.blocks)^2
        # Some smooth function of total effort, derivative=0 near 1 (and below 1), positive above 1 and below 0.
        #i.e. the derivative should be a bowl shape, but I'm not sure i want a parabola-like thing since that just encourages control effort to the minimum.
        # Also, I don't want negative control effort elements, which means we should impose some sort of soft constraint.
    end
end

# TODO: Some function to generate the nonterminal and terminal cost functions
# For now, terminal should just be preference cost, nonterminal is preference cost + control cost

function f(x::BlockVector, u::BlockVector, ms::BlockVector)
    us_per_senator = [vcat([u[Block(sum(control_dim_per_senator) * num_senators * (j-1) + i)] for j in 1:num_activists]) for i in 1:num_senators]
    BlockVector(mapreduce(vcat, zip(x.blocks, us_per_senator, ms.blocks)) do (x, us, m)
        x_move = sum([cos(u[1]) * u[2] for u in us])
        y_move = sum([sin(u[1]) * u[2] for u in us])
        [1 0; 0 1] * x + [x_move; y_move] + m # Maybe some scalar for noise?
    end, [state_dim_per_senator.mean for _ in 1:num_senators])
end

function sensor_model(x::BlockVector, ns::BlockVector)
    return x + ns
end

function receding_horizon_main()
    # TODO
end
end # module