module Senate
using Infiltrator
using RobustBeliefGame
using LinearAlgebra
using BlockArrays
using Distributions
using Random
using Statistics
using Serialization

# include("./SenateVisuals.jl")
# using .SenateVisuals

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
    non_robust_activist => (;pos = [[1,1]], scale = [[1,2]], terminal_weight=2.0, control_weight=(;direction=1.0, control_cost=2.0)),
    robust_activist => (;pos = [[3,0]], scale = [[2,1]], terminal_weight=2.0, control_weight=(;direction=1.0, control_cost=2.0)),
    nature_activist => (;terminal_weight=1.0, control_weight=(;direction=2.0, control_cost=10.0)),

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
    sensor=[state_dim_per_senator.mean for _ in 1:num_activists*num_senators],
    opinion_dim=opinion_dim
)


# All changes to game parameters should flow from info above

ϵ = eps()
random_seed = 1

function ellipsoidal_preference_generator(pos::Vector, scale::Vector; nature=false)
    function preference(point::Vector)
        if length(point) != opinion_dim || length(scale[1]) != opinion_dim || length(point) != opinion_dim #Assert equal dimensions
            throw(DimensionMismatch("Opinion dimensions are not uniform, $(size(pos)), $(size(scale)), $(size(point))"))
        end
        mapreduce(+, zip(pos, scale)) do (pos_mode, scale_mode)
            mapreduce(+, 1:opinion_dim) do i
                (nature ? -1 : 1) * 0.1 * (point[i]-pos_mode[i])^2/scale_mode[i]
            end
        end
    end
    return preference
end

function u_transform(u::BlockVector)
    BlockVector(mapreduce(vcat, u.blocks) do u_i
        [u_i[1], log(exp(u_i[2]) + 1)]
    end, length.(u.blocks))
end

function control_cost_generator(control_effort; nature=false)
    if nature
        function control_cost_function_nature(u::BlockVector)
            nature_u = u.blocks[end]
            control_effort * dot(nature_u, nature_u)
        end
        return control_cost_function_nature
    else
        function control_cost_function(u::BlockVector)
            # Only use lobbyist controls, not nature's controls
            lobbyist_u = u[Block(1):Block(num_senators*num_activists)]
            # transformed_u = u_transform(lobbyist_u)
            control_effort * sum(dot(u, u) for u in lobbyist_u.blocks)
            # control_effort * sum(u_i[1]^2+u_i[2]^2 for u_i in transformed_u.blocks)
        end
        return control_cost_function
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
    function non_terminal_cost_function(beliefs::Beliefs, u::BlockVector)
        sum(non_terminal_cost_components(ellipsoids,
        control_cost_generator(cost_params.control_weight.control_cost; nature=nature),
        means(beliefs), u))
    end
    return non_terminal_cost_function
end

function terminal_cost_generator(cost_params::NamedTuple, ellipsoids::Function; nature=false)
    function terminal_cost_function(beliefs::Beliefs)
        cost_params.terminal_weight * sum(terminal_cost_components(ellipsoids, means(beliefs)))
    end
    return terminal_cost_function
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
    BlockVector(x + ns, length.(x.blocks))
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
    

function run_receding_horizon_trial(;horizon=10, planning_horizon=5, random_seed=1,
    scale_scale_factors = [1.0, 1.0],
    terminal_weight_scale_factors = [1.0, 1.0, 1.0],
    control_cost_scale_factors = [1.0, 1.0, 1.0],
)
    current_cost_params = deepcopy(cost_params)
    current_cost_params[non_robust_activist] = (;
        pos=current_cost_params[non_robust_activist].pos,
        scale=current_cost_params[non_robust_activist].scale .* scale_scale_factors[1],
        terminal_weight=current_cost_params[non_robust_activist].terminal_weight * terminal_weight_scale_factors[1],
        control_weight=(;direction=current_cost_params[non_robust_activist].control_weight.direction, control_cost=current_cost_params[non_robust_activist].control_weight.control_cost * control_cost_scale_factors[1])
    )
    current_cost_params[robust_activist] = (;
        pos=current_cost_params[robust_activist].pos,
        scale=current_cost_params[robust_activist].scale .* scale_scale_factors[2],
        terminal_weight=current_cost_params[robust_activist].terminal_weight * terminal_weight_scale_factors[2],
        control_weight=(;direction=current_cost_params[robust_activist].control_weight.direction, control_cost=current_cost_params[robust_activist].control_weight.control_cost * control_cost_scale_factors[2])
    )
    current_cost_params[nature_activist] = (;
        terminal_weight=current_cost_params[nature_activist].terminal_weight * terminal_weight_scale_factors[3],
        control_weight=(;direction=current_cost_params[nature_activist].control_weight.direction, control_cost=current_cost_params[nature_activist].control_weight.control_cost * control_cost_scale_factors[3])
    )

    ellipsoids = [ellipsoidal_preference_generator(current_cost_params[non_robust_activist].pos, current_cost_params[non_robust_activist].scale),
                ellipsoidal_preference_generator(current_cost_params[robust_activist].pos, current_cost_params[robust_activist].scale),
                ellipsoidal_preference_generator(current_cost_params[robust_activist].pos, current_cost_params[robust_activist].scale; nature=true)]
    costs = [BeliefCost(non_terminal_cost_generator(current_cost_params[non_robust_activist], ellipsoids[1]), terminal_cost_generator(current_cost_params[non_robust_activist], ellipsoids[1])),
            BeliefCost(non_terminal_cost_generator(current_cost_params[robust_activist], ellipsoids[2]), terminal_cost_generator(current_cost_params[robust_activist], ellipsoids[2])),
            BeliefCost(non_terminal_cost_generator(current_cost_params[nature_activist], ellipsoids[3]; nature=true), terminal_cost_generator(current_cost_params[nature_activist], ellipsoids[3]; nature=true))]

    
    # --- Receding Horizon Loop ---
    
    gt_state_history = [ground_truth_senator_states]
    observation_history = []
    solution_history = Dict("non_robust"=>[], "robust"=>[])
    environments::Vector{BeliefEnvironment} = [BeliefEnvironment(f, ground_truth_senator_states, h) for _ in 1:dims.num_activists]
    
    current_beliefs = initial_beliefs
    current_gt_state = ground_truth_senator_states
    
    warm_starts = Dict{String, Any}("non_robust"=>nothing, "robust"=>nothing)
    representative_games = Dict{String, Any}("non_robust"=>nothing, "robust"=>nothing)

    Random.seed!(random_seed)
    
    process_noise_dist = MvNormal(zeros(sum(dims.states)), I(sum(dims.states))) 
    sensor_noise_dist = MvNormal(zeros(sum(dims.states)), I(sum(dims.states)))

    for t in 1:horizon-1
        println("Receding Horizon Step $t / $(horizon-1)")

        for type in ["non_robust", "robust"]
            is_robust = type == "robust"
            
            game_horizon = min(planning_horizon, horizon - t + 1)
            
            game = BeliefGame(
                BeliefEnvironment(f, current_gt_state, h),
                is_robust ? costs : costs[1:2],
                current_beliefs,
                game_horizon,
                dims,
                current_gt_state,
                is_robust
            )

            if t == 1
                representative_games[type] = game
            end

            nominal_beliefs, nominal_controls, _ = solve(game; debug=false, warm_start=warm_starts[type])

            push!(solution_history[type], (nominal_beliefs, nominal_controls))
            
            if length(nominal_beliefs) > 1
                shifted_beliefs = nominal_beliefs[2:end]
                shifted_controls = nominal_controls[2:end]

                zero_control = if is_robust
                    control_block_sizes = vcat(dims.controls, sum(dims.states))
                    BlockVector(zeros(sum(control_block_sizes)), control_block_sizes)
                else
                    BlockVector(zeros(sum(dims.controls)), dims.controls)
                end
                
                last_belief = shifted_beliefs[end]
                g, _ = ekf_update(last_belief, zero_control, game.environment.dynamics, game.environment.sensor_models; is_robust=is_robust, n_players=dims.num_activists)
                extended_belief = unvec(g, game.dims.belief)

                warm_start_beliefs = vcat(shifted_beliefs, [extended_belief])
                warm_start_controls = vcat(shifted_controls, [zero_control])
                warm_starts[type] = (warm_start_beliefs, warm_start_controls)
            else
                warm_starts[type] = (nominal_beliefs, nominal_controls)
            end
        end

        u_non_robust = solution_history["non_robust"][end][2][1]
        u_robust = solution_history["robust"][end][2][1]
        
        u1_controls = u_non_robust[Block(1):Block(dims.num_senators)]
        u2_controls = u_robust[Block(dims.num_senators + 1):Block(dims.num_activists * dims.num_senators)]

        activist_u = mortar([u1_controls.blocks..., u2_controls.blocks...])
        
        process_noise_vec = rand(process_noise_dist)
        process_noise = BlockVector(process_noise_vec, dims.states)
        current_gt_state = f(current_gt_state, activist_u, process_noise)
        push!(gt_state_history, current_gt_state)

        observations = [h(current_gt_state, BlockVector(rand(sensor_noise_dist), dims.states)) for _ in 1:dims.num_activists]
        observations = mortar(observations)
        push!(observation_history, observations)
        
        current_beliefs = ekf_update_with_observations(current_beliefs, activist_u, environments, observations)
    end

    solutions_dict = Dict(
        "non_robust" => (
            gt_state_history=gt_state_history,
            observation_history=observation_history,
            solution_history=solution_history["non_robust"]
        ),
        "robust" => (
            gt_state_history=gt_state_history,
            observation_history=observation_history,
            solution_history=solution_history["robust"]
        )
    )
    return solutions_dict, representative_games
end


function receding_horizon_main(file_id::String=""; horizon=10, min_planning_horizon=5, override=false, random_seed=1, trials=2)
    solution_filename = "exp/senate/outputs/$file_id.dat"
    solutions = Dict()
    games = Dict()

    if isfile(solution_filename) && !override
        println("Loading solution from $solution_filename")
        open(solution_filename, "r") do f
            solutions, games = deserialize(f)
        end
        # SenateVisuals.visualize_receding_horizon_solution(solutions, games; dims=dims)
        return
    end

    scale_scale_factors = [1.0]
    terminal_weight_scale_factors = [1.0]
    control_cost_scale_factors = [1.0]
    
    num_non_robust_runs = length(scale_scale_factors)^2 * length(terminal_weight_scale_factors)^2 * length(control_cost_scale_factors)^2
    num_robust_runs = num_non_robust_runs * length(terminal_weight_scale_factors) * length(control_cost_scale_factors)
    total_runs = (num_non_robust_runs + num_robust_runs) * trials

    println("This experiment will run $total_runs simulations.")
    println("Breakdown: $(num_non_robust_runs*trials) non-robust runs and $(num_robust_runs*trials) robust runs.")
    println("Do you want to continue? (y/n)")
    user_input = readline()
    if user_input != "y"
        println("Aborting.")
        return
    end


    non_robust_params = Iterators.product(
        scale_scale_factors, # non_robust_activist scale
        scale_scale_factors, # robust_activist scale
        terminal_weight_scale_factors, # non_robust_activist terminal_weight
        terminal_weight_scale_factors, # robust_activist terminal_weight
        control_cost_scale_factors, # non_robust_activist control_cost
        control_cost_scale_factors  # robust_activist control_cost
    )

    nature_params = Iterators.product(
        terminal_weight_scale_factors, # nature_activist terminal_weight
        control_cost_scale_factors  # nature_activist control_cost
    )

    for (s_nr, s_r, tw_nr, tw_r, cc_nr, cc_r) in non_robust_params
        for (tw_n, cc_n) in nature_params
            _random_seed = random_seed
            for trial in 1:trials
                println("Running trial $trial with scale_scale_factors: $s_nr, $s_r, terminal_weight_scale_factors: $tw_nr, $tw_r, $tw_n, control_cost_scale_factors: $cc_nr, $cc_r, $cc_n")
                
                rh_solutions, rh_games = run_receding_horizon_trial(
                    scale_scale_factors=[s_nr, s_r],
                    terminal_weight_scale_factors=[tw_nr, tw_r, tw_n],
                    control_cost_scale_factors=[cc_nr, cc_r, cc_n],
                    horizon=horizon,
                    planning_horizon=min_planning_horizon,
                    random_seed=_random_seed
                )
                
                non_robust_key = "nr_s_$(s_nr)_$(s_r)_tw_$(tw_nr)_$(tw_r)_cc_$(cc_nr)_$(cc_r)_$trial"
                solutions[non_robust_key] = rh_solutions["non_robust"]
                games[non_robust_key] = rh_games["non_robust"]

                robust_key = "r_s_$(s_nr)_$(s_r)_tw_$(tw_nr)_$(tw_r)_$(tw_n)_cc_$(cc_nr)_$(cc_r)_$(cc_n)_$trial"
                solutions[robust_key] = rh_solutions["robust"]
                games[robust_key] = rh_games["robust"]

                _random_seed += 1
            end
        end
    end


    println("Saving solution to $solution_filename")
    open(solution_filename, "w") do f
        serialize(f, (solutions, games))
    end
    # SenateVisuals.visualize_receding_horizon_solution(solutions, games; dims=dims)
end
end # module