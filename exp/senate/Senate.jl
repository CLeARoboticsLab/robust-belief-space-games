module Senate
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
num_activists = length(instances(ActivistID))
dims = 2 #opinion space dimensions

cost_params = Dict(
    non_robust_activist => (;pos = [[1,1]], scale = [[1,2]], control_weight=(;direction=1.0, control_cost=1.0)),
    robust_activist => (;pos = [[3,0]], scale = [[2,1]], control_weight=(;direction=1.0, control_cost=1.0)),
) 
# Ideally, we can "save" cost functions by storing the parameters of components used to generate the cost.
# This can somewhat approximate multi-modal preferences by generating multiple ellipsoids.
state_dim_per_senator = (;mean=dims, covariance=dims^2)
# x = [x_pos, y_pos] per senator;
control_dim_per_senator = (;direction=1, effort=1)
# u = [direction, effort] per senator per activist; From Henry: Do we need direction, isn't it just towards the activist position determined by the effort distribution
ground_truth_senator_states = mortar([
        [2, 0.5],
        [1.5, 0],
        [2, -0.5],
    ])
num_senators = length(ground_truth_senator_states.blocks)
initial_beliefs = Beliefs(vcat([[Belief(ground_truth_senator_states[Block(i)], 0.2 * Symmetric(I(dims))) for i in 1:num_senators] for _ in 1:num_activists]...))


# All changes to game parameters should flow from info above

ϵ = eps()
random_seed = 1

function ellipsoidal_preference_generator(pos::Vector, scale::Vector; nature=false)
    function (point::Vector)
        if length(point) != dims || length(scale[1]) != dims || length(point) != dims #Assert equal dimensions
            throw(DimensionMismatch("Opinion dimensions are not uniform, $(size(pos)), $(size(scale)), $(size(point))"))
        end
        mapreduce(+, zip(pos, scale)) do (pos_mode, scale_mode)
            mapreduce(+, 1:dims) do i
                (nature ? -1 : 1) * (point[i]-pos_mode[i])^2/scale_mode[i]
            end
        end
    end
end

function u_transform(u::BlockVector)
    BlockVector(mapreduce(vcat, u.blocks) do u_i
        [u_i[1], u_i[2]^2]
    end, length.(u.blocks))
end

function control_cost_generator(control_effort; nature=false)
    if nature
        function (u::BlockVector)
            control_effort * sum([dot(u[Block(i)], u[Block(i)]) for i in 1:num_activists]...)
        end
    else
        function (u::BlockVector)
            transformed_u = u_transform(u)
            control_effort * sum(u[2] for u in transformed_u.blocks)
        end
    end
end

# For now, terminal should just be preference cost, nonterminal is preference cost + control cost
function non_terminal_cost_components(preference_ellipsoids::Function, control_function::Function, senator_beliefs::BlockVector, u::BlockVector )
    preference = sum(preference_ellipsoids(pos) for pos in senator_beliefs.blocks)
    control = control_function(u)
    return (;preference, control)
end

function terminal_cost_components(preference_ellipsoids::Function, senator_beliefs::BlockVector)
    preference = sum(preference_ellipsoids(pos) for pos in senator_beliefs.blocks)
    return (;preference)
end

function non_terminal_cost_generator(cost_params::NamedTuple, ellipsoids::Function; nature=false)
    function (beliefs::Beliefs, u::BlockVector)
        sum(non_terminal_cost_components(ellipsoids,
        control_cost_generator(cost_params.control_weight.control_cost; nature=nature),
        means(beliefs), u))
    end
end

function terminal_cost_generator(cost_params::NamedTuple, ellipsoids::Function; nature=false)
    function (beliefs::Beliefs)
        sum(terminal_cost_components(ellipsoids, means(beliefs)))
    end
end

function f(x::BlockVector, u::BlockVector, ms::BlockVector)
    transformed_u = u_transform(u)
    # us_per_senator = [[transformed_u[Block(num_senators * (j-1) + i)] for j in 1:num_activists] for i in 1:num_senators]
    BlockVector(mapreduce(vcat, enumerate(zip(x.blocks, ms.blocks))) do (i, (x, m))
        senator = 1 + (i-1) % num_senators
        us = BlockVector(vcat([transformed_u[Block((j-1) * num_senators + senator)] for j in 1:num_activists]...), [sum(control_dim_per_senator) for _ in 1:num_activists])

        x_move = sum([cos(u[1]) * u[2] for u in us.blocks])
        y_move = sum([sin(u[1]) * u[2] for u in us.blocks])
        [1 0; 0 1] * x + [x_move; y_move] + m # Maybe some scalar for noise?
    end, length.(x.blocks))
end

function h(x::BlockVector, ns::BlockVector)
    return x + ns
end

#Assertions for global variables
function init_checks()
    if (length(non_robust_activist.pos) != dims || length(robust_activist.pos) != dims ||
         length(non_robust_activist.scale) != dims || length(robust_activist.scale) != dims)
        throw(ErrorException("Position or scale dimension mismatch with opinion"))
    end
    if sum(non_robust_activist.scale) != 1 || sum(robust_activist.scale) != 1
        throw(ErrorException("Scale does not add up to 1")) 
        #Consider helping normalize instead of throwing an exception
        # if sum(non_robust_activist.scale) != 1 || sum(robust_activist.scale) == 0
        #     throw(ErrorException("Scale is zero vector"))
        # end
        # non_robust_activist.scale /= norm(non_robust_activist.scale)
        # robust_activist.scale /= norm(robust_activist.scale)
    end
end
    

function receding_horizon_main(file_id::String=""; horizon=10, planning_horizon=5, override=false, random_seed=1, explicit_covariance=false, trials=10)
    solution_filename = "exp/senate/outputs/rh_$file_id.jld2"

    if isfile(solution_filename) && !override
        println("Loading solution from $solution_filename")
        @load solution_filename robust_sol non_robust_sol
        return
    end
    # init_checks()
    Random.seed!(random_seed)
    ellipsoids = [ellipsoidal_preference_generator(cost_params[non_robust_activist].pos, cost_params[non_robust_activist].scale),
                  ellipsoidal_preference_generator(cost_params[robust_activist].pos, cost_params[robust_activist].scale),
                  ellipsoidal_preference_generator(cost_params[robust_activist].pos, cost_params[robust_activist].scale; nature=true)]
    costs = [BeliefCost(non_terminal_cost_generator(cost_params[non_robust_activist], ellipsoids[1]), terminal_cost_generator(cost_params[non_robust_activist], ellipsoids[1])),
              BeliefCost(non_terminal_cost_generator(cost_params[robust_activist], ellipsoids[2]), terminal_cost_generator(cost_params[robust_activist], ellipsoids[2])),
              BeliefCost(non_terminal_cost_generator(cost_params[robust_activist], ellipsoids[3]; nature=true), terminal_cost_generator(cost_params[robust_activist], ellipsoids[3]; nature=true))]

    environment = BeliefEnvironment(f, ground_truth_senator_states, h)
    non_robust_senate_game = BeliefGame(
            environment,
            [costs[1], costs[2]],
            initial_beliefs,
            horizon,
            (; n=2, states=length.(ground_truth_senator_states.blocks), controls=[sum(control_dim_per_senator) for _ in 1:(num_senators*num_activists)], belief=vcat([length.(ground_truth_senator_states.blocks) for _ in 1:num_activists]...), sensor=[state_dim_per_senator.mean * num_senators for _ in 1:num_activists]),
            ground_truth_senator_states,
            false,
        )
    robust_senate_game = BeliefGame(
            environment,
            [costs[1], costs[2], costs[3]],
            initial_beliefs,
            horizon,
            (; n=2, states=length.(ground_truth_senator_states.blocks), controls=[sum(control_dim_per_senator) for _ in 1:(num_senators*num_activists)], belief=length.(ground_truth_senator_states.blocks), sensor=[state_dim_per_senator.mean * num_senators for _ in 1:num_activists]),
            ground_truth_senator_states,
            true,
        )
    non_robust_sol = solve(non_robust_senate_game; debug=true)
    # robust_sol = solve(robust_senate_game; debug=true)
    println("Saving solution to $solution_filename")
    @save solution_filename robust_sol non_robust_sol
end
end # module