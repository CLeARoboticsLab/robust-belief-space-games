using Infiltrator
using TrajectoryGamesBase
using TrajectoryGamesExamples
using RobustBeliefGame
using LinearAlgebra
using BlockArrays
using Makie
using Makie.GeometryBasics
using Symbolics
# using CairoMakie
using GLMakie
using JLD2
using FileIO
using Distributions
using Random

struct DummyEnvironment end

include("HockeyVisuals.jl")

function TrajectoryGamesBase.get_constraints(::DummyEnvironment, player_index)
    (state) -> Symbolics.Num[]
end

function hockey_game(;
        n=2,
        player_costs = [
            (x, u) -> x[1]^2 + x[2]^2 + u[1]^2 + u[2]^2,
            (x, u) -> x[1]^2 + x[2]^2 + u[1]^2 + u[2]^2,
        ],
        environment = PolygonEnvironment(4, 50),
        single_dynamics = planar_double_integrator(;
            state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 4, 4]),
            control_bounds = (; lb = [-10, -10], ub = [3, 3]),
            dt = 0.3
        ),
        goal_position = nothing,
        horizon = 20, # actually set in main
        cost = [
            (xs, us) -> attacker_cost(xs, us; goal_position = goal_position),
            (xs, us) -> defender_cost(xs, us; goal_position = goal_position),
        ],
    )

    dynamics = ProductDynamics([single_dynamics for _ in 1:n])
    return TrajectoryGame(
        dynamics,
        cost,
        environment,
        nothing # constraints, add initial conditions as constraints
    )
end


function attacker_cost(xs, us; goal_position)
    return mapreduce(+, eachindex(xs)) do t
        -1 * shot_probability(xs[t][1:2], xs[t][5:6], goal_position[1], goal_position[2]) +
        -2 * dot(xs[t][1:2]-xs[t][5:6],xs[t][1:2]-xs[t][5:6]) + 
        0.05 * dot(us[t][1:2], us[t][1:2])
        # dot(xs[t][1:4], xs[t][1:4]) + 0.1*dot(us[t][1:2], us[t][1:2])
    end
end

function defender_cost(xs, us; goal_position)
    return mapreduce(+, eachindex(xs)) do t
        shot_probability(xs[t][1:2], xs[t][5:6], goal_position[1], goal_position[2]) +
        2 * dot(xs[t][1:2]-xs[t][5:6], xs[t][1:2]-xs[t][5:6]) +
        0.05 * dot(us[t][3:4], us[t][3:4])
        # dot(xs[t][5:8], xs[t][5:8]) + 0.1*dot(us[t][3:4], us[t][3:4])
    end
end

function shot_probability(attacker_pos, defender_pos, goal_p1, goal_p2)
    u = defender_pos - attacker_pos
    v = (goal_p1 + goal_p2) / 2 - attacker_pos

    nu = dot(u, u)
    nv = dot(v, v)

    return -0.1 * dot(u, v) / (nv + nu + 1e-9) + -1 * nv
    # return dot(attacker_pos,attacker_pos)
end

function main()
    horizon = 20
    initial_states = [
        [0.75, 5.0, 0.0, 0.0],  # Attacker
        [-0.75, 1.5, 0.0, 0.0],
    ]
    # A single goal defined by its two posts.
    goal_position = [
            [0.25, -1.5],
            [-0.25, -1.5],
        ]

    # Visual
    goal_line_width = 5

    game = hockey_game(;
        horizon = horizon,
        goal_position = goal_position,
    )    
    mcp_game = MCPGame(game, horizon, vcat(initial_states...);debug=true)
    
    sol = solve(mcp_game; debug=true, warm_start=false)
    
    # Create figure
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1],
        title="Hockey Game Solution",
        xlabel="x position",
        ylabel="y position",
        aspect=1,
    )
    
    # Plot trajectories
    traj1 = lines!(ax, [x[Block(1)][1] for x in sol.xs], [x[Block(1)][2] for x in sol.xs], label="Attacker", color=:blue, linewidth=2)
    traj2 = lines!(ax, [x[Block(2)][1] for x in sol.xs], [x[Block(2)][2] for x in sol.xs], label="Defender", color=:red, linewidth=2)
    
    # Plot goal positions
    goal_posts = [[p[1] for p in goal_position], [p[2] for p in goal_position]]
    goal = lines!(ax, goal_posts[1], goal_posts[2], label="Goal", color=:green, linewidth=goal_line_width)

    
    # Plot initial positions
    init_atk = scatter!(ax, [sol.xs[1][Block(1)][1]], [sol.xs[1][Block(1)][2]], label="Attacker Start", color=:blue, markersize=15)
    init_def = scatter!(ax, [sol.xs[1][Block(2)][1]], [sol.xs[1][Block(2)][2]], label="Defender Start", color=:red, markersize=15)
    
    # Add arrows to show direction of movement
    for i in 2:2:length(sol.xs)-1
        arrows!(ax, 
            [sol.xs[i][Block(1)][1]], [sol.xs[i][Block(1)][2]], 
            [sol.xs[i][Block(1)][3] / norm(sol.xs[i][Block(1)][3:4])], [sol.xs[i][Block(1)][4] / norm(sol.xs[i][Block(1)][3:4])], 
            color=:blue, arrowsize=10, lengthscale=0.1)
        arrows!(ax, 
            [sol.xs[i][Block(2)][1]], [sol.xs[i][Block(2)][2]], 
            [sol.xs[i][Block(2)][3] / norm(sol.xs[i][Block(2)][3:4])], [sol.xs[i][Block(2)][4] / norm(sol.xs[i][Block(2)][3:4])], 
            color=:red, arrowsize=10, lengthscale=0.1)
    end

    Legend(
        fig[1, 2], 
        # [traj1, traj2, goal, [init_atk, init_def], [final_atk, final_def]],
        [traj1, traj2, goal, [init_atk, init_def]],
        # ["Attacker Trajectory", "Defender Trajectory", "Goals", "Initial Positions", "Final Positions"]
        ["Attacker Trajectory", "Defender Trajectory", "Goals", "Initial Positions"]
    )

    
    # Display and save
    # display(fig)
    save("exp/hockey/outputs/hockey_solution.png", fig)
end

dt = 0.3
dt = 0.3
n=2
goal_position = [
    [0.25, -1.5],
    [-0.25, -1.5],
]
# Environment
    # Dynamics
function f(xs::BlockVector, us::BlockVector, ms::BlockVector)
    BlockVector(
            mapreduce(vcat, zip(xs.blocks, us.blocks, ms.blocks)) do (xᵢ, uᵢ, mᵢ)
            [1 0; 0 1] * xᵢ +
            [1 0; 0 1] * uᵢ +
            [0.1 0; 0 0.1] * mᵢ
        end,
        length.(xs.blocks)
    )
end

    # Sensor Models
function h(xs::BlockVector, ns::BlockVector)
    BlockVector(
        mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
            [1 0; 0 1] * xᵢ + [0.1 0; 0 0.1] * nᵢ
        end,
        length.(xs.blocks)
    )
end
# Cost
function steal_liklihood(belief_over_attacker::Belief, belief_over_defender::Belief)

    attacker_pos = belief_over_attacker.belief_mean
    defender_pos = belief_over_defender.belief_mean

    sq_dist = dot(attacker_pos - defender_pos, attacker_pos - defender_pos)

    goal_center = (goal_position[1] + goal_position[2]) / 2
    v_attacker_to_goal = attacker_pos - goal_center
    v_attacker_to_defender = attacker_pos - defender_pos

    cos_block_angle = dot(v_attacker_to_goal, v_attacker_to_defender) / (norm(v_attacker_to_goal) * norm(v_attacker_to_defender) + 1e-9)
    
    blocking_factor = max(0, cos_block_angle)

    proximity_factor = exp(-0.5 * sq_dist)
    
    geometric_bonus = blocking_factor * proximity_factor
    
    attacker_pos_uncertainty = tr(belief_over_attacker.belief_covariance)
    defender_pos_uncertainty = tr(belief_over_defender.belief_covariance)
    total_pos_uncertainty = attacker_pos_uncertainty + defender_pos_uncertainty
    
    return 5.0 * geometric_bonus - 0.5 * total_pos_uncertainty
end
function defender_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us)
    # The defender's cost is based on their own belief about the world.
    # Here, we assume the defender is player 2.
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender)
    control_effort = dot(us[Block(2)], us[Block(2)]) # Defender is player 2
    return -2 * steal_prob + 2 * control_effort
end
function attacker_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us)
    # The attacker's cost is based on their own belief about the world.
    # Here, we assume the attacker is player 1.
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender)
    goal_center = (goal_position[1] + goal_position[2]) / 2
    dist_to_goal_sq = dot(belief_over_attacker.belief_mean - goal_center, belief_over_attacker.belief_mean - goal_center)
    control_effort = dot(us[Block(1)], us[Block(1)]) # Attacker is player 1
    return 1 * steal_prob + 4 * control_effort + 0.1 * dist_to_goal_sq
end    
function shot_probability(belief_over_attacker::Belief, belief_over_defender::Belief)
    dist_penalty = 0.1
    block_max = 3
    block_falloff = 0.8
    attacker_uncertainty_penalty = 0.3  
    defender_uncertainty_penalty = 0.5 
    goal_center = (goal_position[1] + goal_position[2]) / 2

    attacker_pos = belief_over_attacker.belief_mean # Player 1 is Attacker
    defender_pos = belief_over_defender.belief_mean # Player 2 is Defender
    attacker_pos_uncertainty = tr(belief_over_attacker.belief_covariance)
    defender_pos_uncertainty = tr(belief_over_defender.belief_covariance)

    # Term 1: Base score, penalized by distance to goal and attacker's own uncertainty.
    dist_sq_to_goal = dot(attacker_pos - goal_center, attacker_pos - goal_center)
    distance_penalty = dist_penalty * atan(dist_sq_to_goal)
    attacker_uncertainty_penalty_term = attacker_uncertainty_penalty * attacker_pos_uncertainty

    # Term 2: Defender blocking penalty, hindered by defender's own uncertainty.
    v_attacker_to_goal = goal_center - attacker_pos
    v_attacker_to_defender = defender_pos - attacker_pos
    dist_sq_to_defender = dot(v_attacker_to_defender, v_attacker_to_defender)

    cos_block_angle =
        dot(v_attacker_to_goal, v_attacker_to_defender) /
        (norm(v_attacker_to_goal) * norm(v_attacker_to_defender) + 1e-9)

    # Defender's blocking power is reduced by their positional uncertainty
    block_effectiveness = (block_max * atan(-0.1 * block_falloff * dist_sq_to_defender)) /
                        (1 + defender_uncertainty_penalty * defender_pos_uncertainty)

    defender_block_penalty = block_effectiveness * log(1 + exp(cos_block_angle))

    # Final score calculation
    final_score = 1.0 - distance_penalty - attacker_uncertainty_penalty_term - defender_block_penalty
    return final_score
end
function attacker_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief)

    return -10 * shot_probability(belief_over_attacker, belief_over_defender)
end
function defender_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief)

    return 10 * shot_probability(belief_over_attacker, belief_over_defender)
end
function nature_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us::BlockVector)
    steal_prob = dot(belief_over_attacker.belief_mean[1:2] - belief_over_defender.belief_mean[1:2], belief_over_attacker.belief_mean[1:2] - belief_over_defender.belief_mean[1:2])
    return steal_prob + 1000*dot(us[Block(3)], us[Block(3)])
end
function nature_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief)
    return -defender_terminal_cost(belief_over_attacker, belief_over_defender)
end

function belief_main(sol_number=2, override_solution=false)
    solution_filename = "exp/hockey/outputs/hockey_solution_$sol_number.jld2"

    local robust_sol, non_robust_sol
    global goal_position

    if isfile(solution_filename) && !override_solution
        println("Loading solution from $solution_filename")
        @load solution_filename robust_sol non_robust_sol goal_position
    else
        !override_solution && println("No solution file found. Running solver...")
        override_solution && println("Overriding solution...")
        # Game Params
        horizon = 5

        # Initial States/Beliefs
        gt_initial_state = mortar([ # gt = ground truth
        [0.75, 5.0, 0.0, 0.0],  # Attacker
        [-0.75, 1.5, 0.0, 0.0],
        ])
        initial_belief_covariance = [
            [0 0 0    0;
            0 0 0    0;
            0 0 0.25 0;
            0 0 0    0.25],
            [0 0 0    0;
            0 0 0    0;
            0 0 0.25 0;
            0 0 0    0.25],
        ]
        initial_beliefs = Beliefs([Belief(gt_initial_state[Block(i)], initial_belief_covariance[i]) for i in 1:2])

        environment = BeliefEnvironment(f, gt_initial_state, h)

        attacker_cost = BeliefCost(
            attacker_non_terminal_cost,
            attacker_terminal_cost,
        )
        defender_cost = BeliefCost(
            defender_non_terminal_cost,
            defender_terminal_cost,
        )
        nature_cost = BeliefCost(
            nature_non_terminal_cost,
            nature_terminal_cost,
        )
        non_robust_hockey_game = BeliefGame(
            environment,
            [attacker_cost, defender_cost],
            initial_beliefs,
            horizon,
            (; n=2, states=length.(gt_initial_state.blocks), controls=[2, 2], belief=length.(gt_initial_state.blocks), sensor=[2, 2]),
            gt_initial_state,
            false,
        )
        robust_hockey_game = BeliefGame(
            environment,
            [attacker_cost, defender_cost, nature_cost],
            initial_beliefs,
            horizon,
            (; n=2, states=length.(gt_initial_state.blocks), controls=[2, 2], belief=length.(gt_initial_state.blocks), sensor=[2, 2]),
            gt_initial_state,
            true,
            )
        non_robust_sol = solve(non_robust_hockey_game; debug=true, α=1.0)
        robust_sol = solve(robust_hockey_game; debug=true, α=1.0)
        println("Saving solution to $solution_filename")
        @save solution_filename robust_sol non_robust_sol goal_position
    end
    plot_feed_forward_norms(robust_sol[4])
    visualize_belief_hockey_solution(robust_sol, non_robust_sol, goal_position)
end

function safe_eigen(A)
    # try
        A_reg = A + 1e-8 * I(size(A, 1))
        E = eigen(A_reg)
        return (values = max.(E.values, 1e-6), vectors = E.vectors)
    # catch
    #     n = size(A, 1)
    #     return (values = fill(1e-6, n), vectors = Matrix(I, n, n))
    # end
end

function receding_horizon_main(file_id::String=""; horizon=20, plotting_horizon=10, override=false, random_seed=1, ff_cond=false)
    global goal_position
    if isfile("exp/hockey/outputs/rh_$file_id.jld2") && !override
        println("Loading solution from exp/hockey/outputs/rh_$file_id.jld2") # TODO handle missing solution history
        @load "exp/hockey/outputs/rh_$file_id.jld2" gt_state_history belief_history planned_trajectories all_observations goal_position robust intermediate_planned_trajectories us_history nature_us_history planned_us_history solution_history
        visualize_receding_horizon_solution(
            gt_state_history, 
            belief_history, 
            planned_trajectories, 
            all_observations, 
            goal_position; 
            is_robust=any(robust), 
            intermediate_planned_trajectories=intermediate_planned_trajectories, 
            us_history=us_history, 
            nature_us_history=nature_us_history,
            planned_us_history=planned_us_history,
            solution_history=solution_history
        )
        return
    end

    gt_initial_state = mortar([
        [0.75, 5.0],  # Attacker
        [-0.75, 1.5], # Defender
    ])
    initial_belief_covariance = [
        [0.1 0; 0 0.1],
        [0.1 0; 0 0.1],
    ]
    initial_beliefs = Beliefs([
        Belief(gt_initial_state[Block(1)], initial_belief_covariance[1]), # Attacker's belief of attacker
        Belief(gt_initial_state[Block(2)], initial_belief_covariance[2]), # Attacker's belief of defender
        Belief(gt_initial_state[Block(1)], initial_belief_covariance[1]), # Defender's belief of attacker
        Belief(gt_initial_state[Block(2)], initial_belief_covariance[2]), # Defender's belief of defender
    ])
    attacker_cost = BeliefCost(
            (bs, us) -> attacker_non_terminal_cost(bs.beliefs[1], bs.beliefs[2], us),
            (bs) -> attacker_terminal_cost(bs.beliefs[1], bs.beliefs[2]),
        )
        defender_cost = BeliefCost(
            (bs, us) -> defender_non_terminal_cost(bs.beliefs[3], bs.beliefs[4], us),
            (bs) -> defender_terminal_cost(bs.beliefs[3], bs.beliefs[4]),
        )
        nature_cost = BeliefCost(
            (bs, us) -> nature_non_terminal_cost(bs.beliefs[3], bs.beliefs[4], us),
            (bs) -> nature_terminal_cost(bs.beliefs[3], bs.beliefs[4]),
        )
    environment = BeliefEnvironment(f, gt_initial_state, h)

    costs = [[attacker_cost, defender_cost], [attacker_cost, defender_cost, nature_cost]]
    robust = [false, true]
    dims = (; n=2, states=length.(gt_initial_state.blocks), controls=[2, 2], belief=[2, 2, 2, 2], sensor=[2, 2, 2, 2])

    current_beliefs = initial_beliefs
    current_gt_state = gt_initial_state
    all_observations = []
    gt_state_history = [current_gt_state]
    belief_history = [current_beliefs]
    planned_trajectories = []
    intermediate_planned_trajectories = []
    us_history = []
    nature_us_history = []
    planned_us_history = []
    solution_history = []
    warm_starts = Vector{Any}([nothing, nothing])

    Random.seed!(random_seed)
    # normal_distribution = MvNormal(zeros(sum(dims.states)), I(sum(dims.states)))
    # draw_from_normal = () -> BlockVector(rand(normal_distribution), dims.states)
    draw_from_normal = () -> BlockVector(zeros(sum(dims.states)), dims.states)

    αs = [1.0, 1.0]
    
    for t in 1:horizon-1
        println("--- Receding Horizon Step $t / $horizon ---")
        
        intermediate_sols_at_t = []
        # graph all beliefs
        # no rh, j do 1 solve.
        # graph nature's actions.

        # NAture has too much power?
        sols = Vector{Any}(undef, dims.n)
        for ii in 1:dims.n
            game = BeliefGame(
                environment,
                costs[ii],
                current_beliefs,
                10,
                dims,
                current_gt_state,
                robust[ii])
            nominal_beliefs, nominal_controls, intermediate_beliefs = solve(game; debug=false, α=αs[ii], warm_start=warm_starts[ii], ff_cond=ff_cond)
            warm_starts[ii] = (nominal_beliefs, nominal_controls)
            push!(intermediate_sols_at_t, intermediate_beliefs)
            sols[ii] = (nominal_beliefs, nominal_controls)
        end
        push!(solution_history, sols)
        push!(planned_us_history, [sols[ii][2] for ii in 1:dims.n])
        u = mortar([sols[ii][2][1][Block(ii)] for ii in 1:dims.n])
        push!(us_history, u)
        
        robust_sol_controls = sols[2][2]
        nature_u = robust_sol_controls[1][Block(3)]
        push!(nature_us_history, nature_u)

        current_gt_state = f(current_gt_state, u, draw_from_normal())

        # TODO different sensor models per player
        observations = mortar([h(current_gt_state, draw_from_normal()) for ii in 1:dims.n])
        current_beliefs = ekf_update_with_observations(current_beliefs, u, environment.dynamics, environment.sensor_models, observations)
        push!(gt_state_history, current_gt_state)
        push!(all_observations, observations)
        push!(belief_history, current_beliefs)
        push!(planned_trajectories, [sols[ii][1] for ii in 1:dims.n])
        push!(intermediate_planned_trajectories, intermediate_sols_at_t)
    end

    @save "exp/hockey/outputs/rh_$file_id.jld2" gt_state_history belief_history planned_trajectories all_observations goal_position robust intermediate_planned_trajectories us_history nature_us_history planned_us_history solution_history
    
    # 6. Visualize
    visualize_receding_horizon_solution(
        gt_state_history, 
        belief_history, 
        planned_trajectories,
        all_observations,
        goal_position; 
        is_robust=any(robust),
        intermediate_planned_trajectories=intermediate_planned_trajectories,
        us_history=us_history,
        nature_us_history=nature_us_history,
        planned_us_history=planned_us_history,
        solution_history=solution_history
    )
end