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

function save_solution(filename, robust_sol, non_robust_sol, goal_position)  
    @save filename robust_sol non_robust_sol goal_position
end

function load_solution(filename)
    @load filename robust_sol non_robust_sol goal_position
    return robust_sol, non_robust_sol, goal_position
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
            [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1] * xᵢ +
            [0.5*dt^2 0; 0 0.5*dt^2; dt 0; 0 dt] * uᵢ +
            [.01 0 0 0; 0 .01 0 0; 0 0 .05*uᵢ[1] 0; 0 0 0 .05*uᵢ[2]] * mᵢ
        end,
        [4, 4]
    )
end

    # Sensor Models
function h(xs::BlockVector, ns::BlockVector)
    BlockVector(
        mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
            [1 0 0 0; 0 1 0 0] * xᵢ + [0 0 .05*xᵢ[3]+.025 0; 0 0 0 .05*xᵢ[4]+.025] * nᵢ
        end,
        [2, 2]
    )
end
# Cost
function steal_liklihood(bs::Beliefs)
    sq_dist = dot(bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2], bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2])
    sq_vel_dist = dot(bs.beliefs[1].belief_mean[3:4] - bs.beliefs[2].belief_mean[3:4], bs.beliefs[1].belief_mean[3:4] - bs.beliefs[2].belief_mean[3:4])
    dist_uncertainty = dot(bs.beliefs[1].belief_covariance[1, 1:2], bs.beliefs[1].belief_covariance[2, 1:2])
    vel_uncertainty = dot(bs.beliefs[1].belief_covariance[3, 3:4], bs.beliefs[1].belief_covariance[4, 3:4])
    return 1/(dist_uncertainty + vel_uncertainty + 1e-9) * exp(-5 * sq_dist^2) * exp(-sq_vel_dist)
end
function defender_non_terminal_cost(bs::Beliefs, us)
    steal_prob = steal_liklihood(bs)
    # steal_prob = dot(bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2], bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2])
    control_effort = dot(us[Block(1)], us[Block(1)])
    return -2 * steal_prob + 2 * control_effort + dot(bs.beliefs[2].belief_mean[1:2], bs.beliefs[2].belief_mean[1:2])^6
end
function attacker_non_terminal_cost(bs::Beliefs, us)
    steal_prob = steal_liklihood(bs)
    # steal_prob = dot(bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2], bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2])
    control_effort = dot(us[Block(2)], us[Block(2)])
    return steal_prob + 4 * control_effort + dot(bs.beliefs[1].belief_mean[1:2], bs.beliefs[1].belief_mean[1:2])^6
end    
function shot_probability(bs::Beliefs)
    dist_penalty = 0.1
    block_max = 3
    block_falloff = 0.8
    attacker_uncertainty_penalty = 0.3  
    defender_uncertainty_penalty = 0.5 
    goal_center = (goal_position[1] + goal_position[2]) / 2

    attacker_pos = bs.beliefs[1].belief_mean[1:2] # Player 1 is Attacker
    defender_pos = bs.beliefs[2].belief_mean[1:2] # Player 2 is Defender
    attacker_pos_uncertainty = tr(bs.beliefs[1].belief_covariance[1:2, 1:2])
    defender_pos_uncertainty = tr(bs.beliefs[2].belief_covariance[1:2, 1:2])

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
    block_effectiveness = (block_max * exp(-block_falloff * dist_sq_to_defender)) /
                        (1 + defender_uncertainty_penalty * defender_pos_uncertainty)

    defender_block_penalty = block_effectiveness * max(0, cos_block_angle)

    # Final score calculation
    final_score = 1.0 - distance_penalty - attacker_uncertainty_penalty_term - defender_block_penalty
    return final_score
end
function attacker_terminal_cost(bs::Beliefs)
    # Attacker wants to max shot quality, so we min its negative.
    # Don't let attacker get too far away from origin (area of play). This game construction
    #   doesn't allow for hard constraints.
    return -8 * shot_probability(bs)
end
function defender_terminal_cost(bs::Beliefs)
    return 10 * shot_probability(bs)
end
function nature_non_terminal_cost(bs::Beliefs, us::BlockVector)
    steal_prob = dot(bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2], bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2])
    return steal_prob + exp(dot(us[Block(3)], us[Block(3)]))
end
function nature_terminal_cost(bs::Beliefs)
    return -defender_terminal_cost(bs)
end



function belief_main(sol_number=2, override_solution=false)
    solution_filename = "exp/hockey/outputs/hockey_solution_$sol_number.jld2"

    local robust_sol, non_robust_sol
    global goal_position

    if isfile(solution_filename) && !override_solution
        println("Loading solution from $solution_filename")
        robust_sol, non_robust_sol, goal_position = load_solution(solution_filename)
    else
        !override_solution && println("No solution file found. Running solver...")
        override_solution && println("Overriding solution...")
        # Game Params
        horizon = 5 # Reduced from 20 to make the problem smaller

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
            
        non_robust_sol = solve(non_robust_hockey_game; debug=true)
        robust_sol = solve(robust_hockey_game; debug=true)
        println("Saving solution to $solution_filename")
        save_solution(solution_filename, robust_sol, non_robust_sol, goal_position)
    end
    
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

function receding_horizon_main(; robust=true, horizon=20, plotting_horizon=10, override=false)
    global goal_position
    if isfile("exp/hockey/outputs/receding_horizon_$(robust ? "robust" : "non_robust").jld2") && !override
        println("Loading solution from file")
        @load "exp/hockey/outputs/receding_horizon_$(robust ? "robust" : "non_robust").jld2" gt_state_history belief_history planned_trajectories goal_position robust
        visualize_receding_horizon_solution(gt_state_history, belief_history, planned_trajectories, goal_position; is_robust=robust)
        return
    end

    # --- Game Setup ---
    # Mostly copied from belief_main, could be refactored
 
    gt_initial_state = mortar([
        [0.75, 5.0, 0.0, 0.0],  # Attacker
        [-0.75, 1.5, 0.0, 0.0], # Defender
    ])
    initial_belief_covariance = [
        [0.1 0 0 0; 0 0.1 0 0; 0 0 0.25 0; 0 0 0 0.25],
        [0.1 0 0 0; 0 0.1 0 0; 0 0 0.25 0; 0 0 0 0.25],
    ]
    initial_beliefs = Beliefs([Belief(gt_initial_state[Block(i)], initial_belief_covariance[i]) for i in 1:2])

    attacker_cost = BeliefCost(attacker_non_terminal_cost, attacker_terminal_cost)
    defender_cost = BeliefCost(defender_non_terminal_cost, defender_terminal_cost)
    nature_cost = BeliefCost(nature_non_terminal_cost, nature_terminal_cost)
    environment = BeliefEnvironment(f, gt_initial_state, h)
    
    game_template = BeliefGame(
        environment,
        robust ? [attacker_cost, defender_cost, nature_cost] : [attacker_cost, defender_cost],
        initial_beliefs, # This will be updated each step
        plotting_horizon,
        (; n=2, states=length.(gt_initial_state.blocks), controls=[2, 2], belief=length.(gt_initial_state.blocks), sensor=[2, 2]),
        gt_initial_state,
        robust,
    )

    # --- Receding Horizon Loop ---
    current_beliefs = initial_beliefs
    current_gt_state = gt_initial_state
    
    gt_state_history = [current_gt_state]
    belief_history = [current_beliefs]
    planned_trajectories = []
    executed_controls = []
    
    for t in 1:horizon
        println("--- Receding Horizon Step $t / $horizon ---")
        
        # 1. Update the game with the current belief and solve
        game = BeliefGame(
            game_template.environment,
            game_template.costs,
            current_beliefs,
            game_template.horizon,
            game_template.dims,
            current_gt_state,
            game_template.is_robust
        )
        
        sol = solve(game; debug=false)
        push!(planned_trajectories, sol[1]) # Store the planned belief trajectory
        
        # 2. Get first action
        u = sol[2][1]
        push!(executed_controls, u)
        
        # 3. Simulate one step in the "real" environment
        m_true = BlockVector(randn(sum(game_template.dims.states)), game_template.dims.states)
        next_gt_state = f(current_gt_state, u, m_true)
        
        # 4. Update belief using the solver's internal EKF-based propagation
        g, W = RobustBeliefGame.ekf_update(current_beliefs, u, environment.dynamics, environment.sensor_models; is_robust=robust)
        
        # Sample noise and apply it to get the next belief state
        noise = randn(size(W, 2))
        next_belief_vec = g + W * noise
        
        # 5. Update state for next iteration
        current_gt_state = next_gt_state
        current_beliefs = unvec(next_belief_vec, game_template.dims.belief)
        
        push!(gt_state_history, current_gt_state)
        push!(belief_history, current_beliefs)
    end

    @save "exp/hockey/outputs/receding_horizon_$(robust ? "robust" : "non_robust").jld2" gt_state_history belief_history planned_trajectories goal_position robust
    
    # 6. Visualize
    visualize_receding_horizon_solution(
        gt_state_history, 
        belief_history, 
        planned_trajectories, 
        goal_position; 
        is_robust=robust
    )
end