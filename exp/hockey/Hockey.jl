using Infiltrator
using TrajectoryGamesBase
using TrajectoryGamesExamples
using RobustBeliefGame
using LinearAlgebra
using CairoMakie
using BlockArrays



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
        horizon = 20,
        cost = [
            (xs, us) -> attacker_cost(xs, us; goal_position = goal_position),
            (xs, us) -> defender_cost(xs, us; goal_position = goal_position),
        ],
    )

    # simple cost for now:
    # cost = [
    #     (xs, us) -> mapreduce(+, eachindex(xs)) do t
    #         norm(xs[t][1:4]) + 0.1*norm(us[t][1:2])
    #     end,
    #     (xs, us) -> mapreduce(+, eachindex(xs)) do t
    #         norm(xs[t][5:8]) + 0.1*norm(us[t][3:4])
    #     end
    # ]

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
        shot_probability(xs[t][1:2], xs[t][5:6], goal_position) + 0.1 * norm(us[t][1:2])
    end
end

function defender_cost(xs, us; goal_position)
    return mapreduce(+, eachindex(xs)) do t
        -1 * shot_probability(xs[t][1:2], xs[t][5:6], goal_position) + 0.1 * norm(us[t][3:4])
    end
end

function shot_probability(attacker_pos, defender_pos, goal_pos)
    attacker_shooting_angle = find_angle(attacker_pos, goal_pos[1], goal_pos[2])

    # Some multiple of the distance between the attacker and defender. i.e. the defender can react/block further when the attacker is farther
    # defender_range = 0.01 * norm(defender_pos - attacker_pos)
    defender_range = 0.1

    defender_guard = defender_range * [0.0 1.0; -1.0 0.0] * ((defender_pos - attacker_pos) ./ norm(defender_pos - attacker_pos))
    defender_blocking_angle = find_angle(attacker_pos, defender_pos + defender_guard, defender_pos - defender_guard)
    return attacker_shooting_angle - defender_blocking_angle
end

function find_angle(p1, p2, p3)
    v1 = p2 .- p1
    v2 = p3 .- p1
    n1 = norm(v1)
    n2 = norm(v2)
    
    # Should be 2 * atan(norm(v1*n2 - n1*v2), norm(v1*n2 + n1*v2))
    return 2 * atan(norm(v1*n2 - n1*v2), norm(v1*n2 + n1*v2))
end

function main()
    horizon = 10
    initial_states = [
        [0.75, 5.0, 0.0, 0.0],  # Attacker
        [-0.75, 1.5, 0.0, 0.0],
    ]
    goal_position = [
            [-1.5, 0.0],
            [1.5, 0.0],
        ]


    # Visual
    goal_width = 0.5
    goal_line_width = 0.5

    game = hockey_game(;
        horizon = horizon,
        goal_position = goal_position,
    )    
    mcp_game = MCPGame(game, horizon, vcat(initial_states...);debug=true)
    
    sol = solve(mcp_game; debug=true)
    
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
    goal = lines!(ax, [goal_position[i][1] for i in eachindex(goal_position)], [goal_position[i][2]-goal_width for i in eachindex(goal_position)], label="Goals", color=:green, linewidth=goal_line_width)
    lines!(ax, [goal_position[1][1], goal_position[1][1]], [goal_position[1][2], goal_position[1][2]-goal_width], color=:green, linewidth=goal_line_width)
    lines!(ax, [goal_position[2][1], goal_position[2][1]], [goal_position[2][2], goal_position[2][2]-goal_width], color=:green, linewidth=goal_line_width)

    
    # Plot initial positions
    init_atk = scatter!(ax, [sol.xs[1][Block(1)][1]], [sol.xs[1][Block(1)][2]], label="Attacker Start", color=:blue, markersize=15)
    init_def = scatter!(ax, [sol.xs[1][Block(2)][1]], [sol.xs[1][Block(2)][2]], label="Defender Start", color=:red, markersize=15)
    
    # # Plot final positions
    # final_atk = scatter!(ax, [sol.xs[end][Block(1)][1]], [sol.xs[end][Block(1)][2]], label="Attacker End", 
    #     color=:blue, marker=:diamond, markersize=15)
    # final_def = scatter!(ax, [sol.xs[end][Block(2)][1]], [sol.xs[end][Block(2)][2]], label="Defender End", 
    #     color=:red, marker=:diamond, markersize=15)
    
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