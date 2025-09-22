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
num_senators = 3
dims = 2 #opinion space dimensions

cost_params = Dict(
    non_robust_activist => [(;pos = [0,0], scale = [0,1], control_weight=(;direction=1, effort=1))],
    robust_activist => [(;pos = [0,1], scale = [1,0], control_weight=(;direction=1, effort=1))],
) 
# Ideally, we can "save" cost functions by storing the parameters of components used to generate the cost.
# This can somewhat approximate multi-modal preferences by generating multiple ellipsoids.
state_dim_per_senator = (;mean=dims, covariance=dims^2)
# x = [x_pos, y_pos] per senator; From Henry: Consider adding uncertainty along each axes
control_dim_per_senator = (;effort=1)
# u = [direction, effort] per senator per activist; From Henry: Do we need direction, isn't it just towards the activist position determined by the effort distribution

# All changes to game parameters should flow from info above

ϵ = eps()
random_seed = 1
num_activists = length(instances(ActivistID))

function ellipsoidal_preference_generator(pos::Vector, scale:: Vector)
    function (point::Vector)
        if length(point) != dims || length(scale) != dims || length(point) != dims #Assert equal dimensions
            throw(DimensionMismatch("Opinion dimensions are not uniform, $(size(pos)), $(size(scale)), $(size(point))"))
        end
        return sum((point[i]-pos[i])^2/scale[i] for i in 1:length(point))
    end
end
function control_cost_generator(effort::Float64)
    function (u::BlockVector)
        effort * sum(control[2] for control in u.blocks)^2
        # Some smooth function of total effort, derivative=0 near 1 (and below 1), positive above 1 and below 0.
        #i.e. the derivative should be a bowl shape, but I'm not sure i want a parabola-like thing since that just encourages control effort to the minimum.
        # Also, I don't want negative control effort elements, which means we should impose some sort of soft constraint.

        #Henry: Not a fan of the naming of effort, confuses whether its an actual value or a multiplier, also isn't [2] OutOfBounds
    end
end

# For now, terminal should just be preference cost, nonterminal is preference cost + control cost
function non_terminal_cost_components(preference_ellipsoid::Function, control_function::Function, senator_beliefs::BlockVector, u::BlockVector )
    preference = sum(preference_ellipsoid(pos[0],pos[1]) for pos in senator_beliefs.blocks)
    control = control_function(u)
    return (;preference, control)
end

function terminal_cost_components(preference_ellipsoid::Function, senator_beliefs::BlockVector)
    preference = sum(preference_ellipsoid(pos[0],pos[1]) for pos in senator_beliefs.blocks)
    return (;preference)
end

non_terminal_cost(preference_ellipsoid::Function, control::Function, senator_beliefs::BlockVector, u::BlockVector) = 
    sum(non_terminal_cost_components(preference_ellipsoid, control, senator_beliefs,u))
    
terminal_cost(preference_ellipsoid::Function, senator_beliefs::BlockVector) = 
    sum(terminal_cost_components(preference_ellipsoid, senator_beliefs))

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

#Assertions for global variables
function init_checks()
    if (length(non_robust_activist["pos"]) != dims || length(robust_activist["pos"]) != dims ||
         length(non_robust_activist["scale"]) != dims || length(robust_activist["scale"]) != dims)
        throw(ErrorException("Position or scale dimension mismatch with opinion"))
    end
    if sum(non_robust_activist["scale"]) != 1 || sum(robust_activist["scale"]) != 1
        throw(ErrorException("Scale does not add up to 1")) 
        #Consider helping normalize instead of throwing an exception
        # if sum(non_robust_activist["scale"]) != 1 || sum(robust_activist["scale"]) == 0
        #     throw(ErrorException("Scale is zero vector"))
        # end
        # non_robust_activist["scale"] /= norm(non_robust_activist["scale"])
        # robust_activist["scale"] /= norm(robust_activist["scale"])
    end
end
    

function receding_horizon_main()
    init_checks()
    Random.seed!(random_seed)
end
end # module