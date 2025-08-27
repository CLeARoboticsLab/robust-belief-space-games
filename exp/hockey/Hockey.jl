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
using Statistics

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
n=2
goal_position = [
    [0.25, -1.5],
    [-0.25, -1.5],
]
goal_center = (goal_position[1] + goal_position[2]) / 2
# Environment
    # Dynamics
function M(u)
    return 0.1 * I
end
function f(xs::BlockVector, us::BlockVector, ms::BlockVector)
    BlockVector(
            mapreduce(vcat, zip(xs.blocks, us.blocks, ms.blocks)) do (xᵢ, uᵢ, mᵢ)
            [1 0; 0 1] * xᵢ +
            [1 0; 0 1] * uᵢ +
            M(uᵢ) * mᵢ
        end,
        length.(xs.blocks)
    )
end
    # Sensor Models
function N(x)
    dist = dot(x - goal_center, x - goal_center)
    return 0.01 * I * dist
end
function h₁(xs::BlockVector, ns::BlockVector)
    BlockVector(
        mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
            [1 0; 0 1] * xᵢ + N(xᵢ) * nᵢ
        end,
        length.(xs.blocks)
    )
end

function h₂(xs::BlockVector, ns::BlockVector)
    BlockVector(
        mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
            [1 0; 0 1] * xᵢ + 0.1 * I * nᵢ
        end,
        length.(xs.blocks)
    )
end
# Cost
function box_bounds(belief::Belief)
    # bottom = max(100 * exp(-(belief.belief_mean[2] + 5)) - 1, 0)
    bottom = (belief.belief_mean[2] < 0) ? 5 * belief.belief_mean[2]^2 : 0
    # top = max(100 * exp(belief.belief_mean[2] - 10) - 1, 0)
    top = (belief.belief_mean[2] > 3) ? 5 * belief.belief_mean[2]^2 : 0
    # left = max(100 * exp(-(belief.belief_mean[1]+8)) - 1, 0)
    left = (belief.belief_mean[1] < -3) ? 5 * belief.belief_mean[1]^2 : 0
    # right = max(100 * exp(belief.belief_mean[1] - 8) - 1, 0)
    right = (belief.belief_mean[1] > 3) ? 5 * belief.belief_mean[1]^2 : 0
    return 1 * (bottom + top + left + right)
end
function steal_liklihood(belief_over_attacker::Belief, belief_over_defender::Belief)

    attacker_pos = length(belief_over_attacker.belief_mean) == 4 ? belief_over_attacker.belief_mean[1:2] : belief_over_attacker.belief_mean
    defender_pos = length(belief_over_defender.belief_mean) == 4 ? belief_over_defender.belief_mean[1:2] : belief_over_defender.belief_mean

    sq_dist = dot(attacker_pos - defender_pos, attacker_pos - defender_pos)

    # goal_center = (goal_position[1] + goal_position[2]) / 2
    # v_attacker_to_goal = attacker_pos - goal_center
    # v_attacker_to_defender = attacker_pos - defender_pos

    # cos_block_angle = dot(v_attacker_to_goal, v_attacker_to_defender) / (norm(v_attacker_to_goal) * norm(v_attacker_to_defender) + 1e-9)
    
    # blocking_factor = max(0, cos_block_angle)

    # proximity_factor = exp(-0.5 * sq_dist)
    
    # geometric_bonus = blocking_factor * proximity_factor
    
    # attacker_pos_uncertainty = tr(belief_over_attacker.belief_covariance)
    # defender_pos_uncertainty = tr(belief_over_defender.belief_covariance)
    # total_pos_uncertainty = attacker_pos_uncertainty + defender_pos_uncertainty
    
    # return 5.0 * geometric_bonus - 0.5 * total_pos_uncertainty
    return max(0, 10 - sq_dist)
end
function shot_probability(belief_over_attacker::Belief, belief_over_defender::Belief)
    attacker_pos = length(belief_over_attacker.belief_mean) == 4 ? belief_over_attacker.belief_mean[1:2] : belief_over_attacker.belief_mean
    defender_pos = length(belief_over_defender.belief_mean) == 4 ? belief_over_defender.belief_mean[1:2] : belief_over_defender.belief_mean
    
    return shot_probability(attacker_pos, defender_pos, goal_position[1], goal_position[2])
end
function attacker_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us)
    # The attacker's cost is based on their own belief about the world.
    # Here, we assume the attacker is player 1.
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender)
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender)
    control_effort = dot(us[Block(1)], us[Block(1)]) # Attacker is player 1
    return 2 * steal_prob + -2 * shot_prob + 2 * control_effort + box_bounds(belief_over_attacker)
end    
function defender_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us)
    # The defender's cost is based on their own belief about the world.
    # Here, we assume the defender is player 2.
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender)
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender)
    control_effort = dot(us[Block(2)], us[Block(2)]) # Defender is player 2
    return -2 * steal_prob + 2 * shot_prob + 2 * control_effort + box_bounds(belief_over_defender)
end
function nature_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us::BlockVector)
    # steal_prob = dot(belief_over_attacker.belief_mean[1:2] - belief_over_defender.belief_mean[1:2], belief_over_attacker.belief_mean[1:2] - belief_over_defender.belief_mean[1:2])
    return -defender_non_terminal_cost(belief_over_attacker, belief_over_defender, us) + 5*dot(us[Block(3)], us[Block(3)]) + box_bounds(belief_over_attacker) + box_bounds(belief_over_defender)
end
function attacker_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief)

    return -4 * shot_probability(belief_over_attacker, belief_over_defender) + box_bounds(belief_over_attacker)
end
function defender_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief)

    return 4 * shot_probability(belief_over_attacker, belief_over_defender) + box_bounds(belief_over_defender)
end
function nature_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief)
    return -defender_terminal_cost(belief_over_attacker, belief_over_defender) + box_bounds(belief_over_attacker) + box_bounds(belief_over_defender)
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

function receding_horizon_main(file_id::String=""; horizon=15, planning_horizon=5, override=false, random_seed=1)
    global goal_position
    if isfile("exp/hockey/outputs/rh_$file_id.jld2") && !override
        println("Loading solution from exp/hockey/outputs/rh_$file_id.jld2")
        @load "exp/hockey/outputs/rh_$file_id.jld2" solutions goal_position
        visualize_receding_horizon_solution(
            solutions, 
            goal_position;
            dims=(; n=2, states=[2, 2], controls=[2, 2], belief=[2, 2, 2, 2], sensor=[2, 2, 2, 2])
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
    
    # --- Shared Parameters ---
    dims = (; n=2, states=length.(gt_initial_state.blocks), controls=[2, 2], belief=[2, 2, 2, 2], sensor=[2, 2, 2, 2])
    costs = [[attacker_cost, defender_cost], [attacker_cost, defender_cost, nature_cost]]
    robust = [false, true]
    
    # --- Run Scenarios ---
    solutions = Dict()

    println("--- Running Nominal Scenario ---")
    solutions["nominal"] = run_receding_horizon_scenario(
        gt_initial_state, initial_beliefs, costs, robust, dims,
        horizon, planning_horizon, random_seed,
        (f, gt_initial_state, [h₁, h₁]), # environment
        (current_beliefs, u, environments, observations) -> ekf_update_with_observations(current_beliefs, u, environments, observations) # ekf_update
    )

    println("\n--- Running Mismatched Sensor Scenario ---")
    solutions["mismatched_sensor"] = run_receding_horizon_scenario(
        gt_initial_state, initial_beliefs, costs, robust, dims,
        horizon, planning_horizon, random_seed,
        (f, gt_initial_state, [h₂, h₁]), # environment
        (current_beliefs, u, environments, observations) -> ekf_update_with_observations(current_beliefs, u, environments, observations) # ekf_update with h₂
    )

    @save "exp/hockey/outputs/rh_$file_id.jld2" solutions goal_position
    
    visualize_receding_horizon_solution(
        solutions,
        goal_position;
        dims=dims
    )
end

function run_receding_horizon_scenario(
    gt_initial_state, initial_beliefs, costs, robust, dims,
    horizon, planning_horizon, random_seed,
    environment_params, ekf_update_fn
)
    f, _, h = environment_params
    environments::Vector{BeliefEnvironment} = [BeliefEnvironment(f, gt_initial_state, h) for h in h]

    # --- Solve LQ Game ---
    lq_sol_history = []
    
    current_beliefs = initial_beliefs
    current_gt_state = gt_initial_state
    all_observations = []
    gt_state_history = [current_gt_state]
    solution_history = []
    cond_history = []
    warm_starts = Vector{Any}([nothing, nothing])

    Random.seed!(random_seed)
    normal_distribution = MvNormal(zeros(sum(dims.states)), 0.01*I(sum(dims.states)))
    draw_from_normal = () -> BlockVector(rand(normal_distribution), dims.states)

    for t in 1:horizon-planning_horizon
        println("Receding Horizon Step $t / $(horizon-planning_horizon)")
        
        lq_horizon = 10
        lq_initial_states = [
            [current_gt_state[Block(1)]..., 0.0, 0.0],
            [current_gt_state[Block(2)]..., 0.0, 0.0]
        ]
        lq_game = hockey_game(; horizon = lq_horizon, goal_position = goal_position)
        mcp_game = MCPGame(lq_game, lq_horizon, vcat(lq_initial_states...); debug=false)
        lq_sol = solve(mcp_game; debug=false, warm_start=false)
        push!(lq_sol_history, lq_sol)

        sols = Vector{Any}(undef, dims.n)
        for ii in 1:dims.n
            game = BeliefGame(
                environments[ii],
                costs[ii],
                current_beliefs,
                planning_horizon,
                dims,
                current_gt_state,
                robust[ii])
            nominal_beliefs, nominal_controls, intermediate_solutions, _, _, cond = solve(game; debug=true)
            warm_starts[ii] = (nominal_beliefs, nominal_controls)
            sols[ii] = (nominal_beliefs, nominal_controls, intermediate_solutions)
            push!(cond_history, cond)
        end
        push!(solution_history, sols)
        u = mortar([sols[ii][2][1][Block(ii)] for ii in 1:dims.n])
        current_gt_state = f(current_gt_state, u, draw_from_normal())
        
        observations = mortar([h[ii](current_gt_state, draw_from_normal()) for ii in 1:dims.n])
        current_beliefs = ekf_update_fn(current_beliefs, u, environments, observations)
        push!(gt_state_history, current_gt_state)
        push!(all_observations, observations)
    end

    return (gt_state_history, all_observations, solution_history, cond_history, lq_sol_history)
end