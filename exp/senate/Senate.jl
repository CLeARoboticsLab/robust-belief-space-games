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

include("./SenateVisuals.jl")
using .SenateVisuals

export receding_horizon_main

@enum ActivistID begin
    non_robust_activist = 1
    robust_activist = 2
end
num_activists = length(instances(ActivistID))
opinion_dim=2
@enum NatureID begin
    nature_activist = 3
end

cost_params = Dict(
    non_robust_activist => (;pos = [[1,1]], scale = [[1,2]], terminal_weight=5.0, control_weight=(;direction=1.0, control_cost=10.0)),
    robust_activist => (;pos = [[3,0]], scale = [[2,1]], terminal_weight=10.0, control_weight=(;direction=1.0, control_cost=1.0)),
    nature_activist => (;terminal_weight=1.0, control_weight=(;direction=1.0, control_cost=20.0)),

) 
# Ideally, we can "save" cost functions by storing the parameters of components used to generate the cost.
# This can somewhat approximate multi-modal preferences by generating multiple ellipsoids.
state_dim_per_senator = (;mean=opinion_dim, covariance=opinion_dim^2)
# x = [x_pos, y_pos] per senator;
control_dim_per_senator = (;direction=1, effort=1)
#xmove, ymove
ground_truth_senator_states = mortar([
        [2, 0.5],
        [1.5, 0],
        [2, -0.5],
    ])
num_senators = length(ground_truth_senator_states.blocks)
initial_beliefs = Beliefs(vcat([[Belief(ground_truth_senator_states[Block(i)], 0.2 * Symmetric(I(opinion_dim))) for i in 1:num_senators] for _ in 1:num_activists]...))
dims = (;
    n=2,
    num_beliefs_per_activist=num_senators,
    num_senators=num_senators,
    num_activists=num_activists,
    states=length.(ground_truth_senator_states.blocks),
    controls=[sum(control_dim_per_senator) for _ in 1:(num_senators*num_activists)],
    controls_per_activist=[sum(control_dim_per_senator)*num_senators for _ in 1:num_activists],
    belief=vcat([length.(ground_truth_senator_states.blocks) for _ in 1:num_activists]...),
    sensor=[state_dim_per_senator.mean * num_senators for _ in 1:num_activists],
    opinion_dim=opinion_dim
)


# All changes to game parameters should flow from info above

ϵ = eps()
random_seed = 1

function ellipsoidal_preference_generator(pos::Vector, scale::Vector; nature=false)
    function (point::Vector)
        if length(point) != opinion_dim || length(scale[1]) != opinion_dim || length(point) != opinion_dim #Assert equal dimensions
            throw(DimensionMismatch("Opinion dimensions are not uniform, $(size(pos)), $(size(scale)), $(size(point))"))
        end
        mapreduce(+, zip(pos, scale)) do (pos_mode, scale_mode)
            mapreduce(+, 1:opinion_dim) do i
                (nature ? -1 : 1) * 0.1 * (point[i]-pos_mode[i])^2/scale_mode[i]
            end
        end
    end
end

function u_transform(u::BlockVector)
    BlockVector(mapreduce(vcat, u.blocks) do u_i
        [u_i[1], log(exp(u_i[2]) + 1)]
    end, length.(u.blocks))
end

function control_cost_generator(control_effort; nature=false)
    if nature
        function (u::BlockVector)
            nature_u = u.blocks[end]
            control_effort * dot(nature_u, nature_u)
        end
    else
        function (u::BlockVector)
            # Only use lobbyist controls, not nature's controls
            lobbyist_u = u[Block(1):Block(num_senators*num_activists)]
            # transformed_u = u_transform(lobbyist_u)
            control_effort * sum(dot(u, u) for u in lobbyist_u.blocks)
            # control_effort * sum(u_i[1]^2+u_i[2]^2 for u_i in transformed_u.blocks)
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
        cost_params.terminal_weight * sum(terminal_cost_components(ellipsoids, means(beliefs)))
    end
end

function f(x::BlockVector, u::BlockVector, ms::BlockVector)
    # transformed_u = u_transform(u)
    # us_per_senator = [[transformed_u[Block(num_senators * (j-1) + i)] for j in 1:num_activists] for i in 1:num_senators]
    BlockVector(mapreduce(vcat, enumerate(zip(x.blocks, ms.blocks))) do (i, (x, m))
        senator = 1 + (i-1) % num_senators
        us = BlockVector(vcat([u[Block((j-1) * num_senators + senator)] for j in 1:num_activists]...), [sum(control_dim_per_senator) for _ in 1:num_activists])
        transformed_us = u_transform(us)

        x_move = sum([u[1] for u in us.blocks])
        y_move = sum([u[2] for u in us.blocks])
        [1 0; 0 1] * x + [x_move; y_move] + m # Maybe some scalar for noise?
    end, length.(x.blocks))
end

function h(x::BlockVector, ns::BlockVector)
    return x + ns
end

#Assertions for global variables
function init_checks()
    if (length(non_robust_activist.pos) != opinion_dim || length(robust_activist.pos) != opinion_dim ||
        length(non_robust_activist.scale) != opinion_dim || length(robust_activist.scale) != opinion_dim)
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
    solutions = Dict()
    games = Dict()

    if isfile(solution_filename) && !override
        println("Loading solution from $solution_filename")
        @load solution_filename solutions games
        SenateVisuals.visualize_receding_horizon_solution(solutions, games; dims=dims)
        return
    end
    # init_checks()
    Random.seed!(random_seed)
    ellipsoids = [ellipsoidal_preference_generator(cost_params[non_robust_activist].pos, cost_params[non_robust_activist].scale),
                ellipsoidal_preference_generator(cost_params[robust_activist].pos, cost_params[robust_activist].scale),
                ellipsoidal_preference_generator(cost_params[robust_activist].pos, cost_params[robust_activist].scale; nature=true)]
    costs = [BeliefCost(non_terminal_cost_generator(cost_params[non_robust_activist], ellipsoids[1]), terminal_cost_generator(cost_params[non_robust_activist], ellipsoids[1])),
            BeliefCost(non_terminal_cost_generator(cost_params[robust_activist], ellipsoids[2]), terminal_cost_generator(cost_params[robust_activist], ellipsoids[2])),
            BeliefCost(non_terminal_cost_generator(cost_params[nature_activist], ellipsoids[3]; nature=true), terminal_cost_generator(cost_params[nature_activist], ellipsoids[3]; nature=true))]

    environment = BeliefEnvironment(f, ground_truth_senator_states, h)
    games["non_robust"] = BeliefGame(
            environment,
            [costs[1], costs[2]],
            initial_beliefs,
            horizon,
            dims,
            ground_truth_senator_states,
            false,
        )
    games["robust"] = BeliefGame(
            environment,
            [costs[1], costs[2], costs[3]],
            initial_beliefs,
            horizon,
            dims,
            ground_truth_senator_states,
            true,
        )
    solutions["non_robust"] = solve(games["non_robust"]; debug=false)
    solutions["robust"] = solve(games["robust"]; debug=false)
    println("Saving solution to $solution_filename")
    @save solution_filename solutions games

    SenateVisuals.visualize_receding_horizon_solution(solutions, games; dims=dims)
end
end # module