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
        environment = PolygonEnvironment(4, 8),
        single_dynamics = planar_double_integrator(;
            state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
            control_bounds = (; lb = [-10, -10], ub = [10, 10]),
        ),
        goal_position = [
            [-1.5, 0.0],
            [1.5, 0.0],
        ],
        horizon = 20,
        cost = [
            (xs, us) -> attacker_cost(xs, us; goal_position = goal_position, horizon = horizon),
            (xs, us) -> defender_cost(xs, us; goal_position = goal_position, horizon = horizon),
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


function attacker_cost(xs, us; goal_position, horizon)
    return mapreduce(+, 1:horizon) do t
        shot_probability(xs[t][1:2], xs[t][3:4], goal_position) + 0.1 * norm(us[t][1:2])
    end
end

function defender_cost(xs, us; goal_position, horizon)
    return mapreduce(+, 1:horizon) do t
        -1 * shot_probability(xs[t][1:2], xs[t][3:4], goal_position) + 0.1 * norm(us[t][1:2])
    end
end

function shot_probability(attacker_pos, defender_pos, goal_pos)
    attacker_shooting_angle = find_angle(attacker_pos, goal_pos[1], goal_pos[2])

    # Some multiple of the distance between the attacker and defender. i.e. the defender can react/block further when the attacker is farther
    defender_range = 0.5 * norm(defender_pos - attacker_pos)

    defender_guard = defender_range * [0.0 1.0; -1.0 0.0] * ((defender_pos - attacker_pos) ./ norm(defender_pos - attacker_pos))
    defender_blocking_angle = find_angle(defender_pos, attacker_pos + defender_guard, attacker_pos - defender_guard)
    return attacker_shooting_angle - defender_blocking_angle
end

function find_angle(p1, p2, p3)
    v1 = p2 .- p1
    v2 = p3 .- p1
    
    dot_product = dot(v1, v2)
    mag_v1 = norm(v1)
    mag_v2 = norm(v2)
    
    cos_angle = dot_product / (mag_v1 * mag_v2)
    
    return acos(cos_angle)
end


function main()
    horizon ::Int = 2  
    initial_states = [
        [2.0, 0.0, 0.0, 0.0],  # Attacker
        [1.5, 0.0, 0.0, 0.0],
    ]

    game = hockey_game(;horizon = horizon)    
    mcp_game = MCPGame(game, horizon, vcat(initial_states...))
    
    sol = solve(mcp_game)
    
    # Create figure
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1],
        title="Hockey Game Solution",
        xlabel="x position",
        ylabel="y position",
        aspect=1
    )
    
    # Extract trajectories
    attacker_x = [x[Block(1)][1] for x in sol.xs]
    attacker_y = [x[Block(1)][2] for x in sol.xs]
    defender_x = [x[Block(2)][1] for x in sol.xs]
    defender_y = [x[Block(2)][2] for x in sol.xs]
    
    # Plot trajectories
    lines!(ax, attacker_x, attacker_y, label="Attacker", color=:blue, linewidth=2)
    lines!(ax, defender_x, defender_y, label="Defender", color=:red, linewidth=2)
    
    # Plot goal positions
    lines!(ax, [-1.5, 1.5], [0.0, 0.0], label="Goals", color=:green, linewidth=0.5)
    
    # Plot initial positions
    scatter!(ax, [attacker_x[1]], [attacker_y[1]], label="Attacker Start", color=:blue, markersize=15)
    scatter!(ax, [defender_x[1]], [defender_y[1]], label="Defender Start", color=:red, markersize=15)
    
    # Plot final positions
    scatter!(ax, [attacker_x[end]], [attacker_y[end]], label="Attacker End", 
        color=:blue, marker=:diamond, markersize=15)
    scatter!(ax, [defender_x[end]], [defender_y[end]], label="Defender End", 
        color=:red, marker=:diamond, markersize=15)
    
    # Add arrows to show direction of movement
    for i in 1:10:length(sol.xs)
        arrows!(ax, 
            [attacker_x[i]], [attacker_y[i]], 
            [sol.xs[i][5]], [sol.xs[i][6]], 
            color=:blue, arrowsize=10)
        arrows!(ax, 
            [defender_x[i]], [defender_y[i]], 
            [sol.xs[i][7]], [sol.xs[i][8]], 
            color=:red, arrowsize=10)
    end
    
    # Add legend
    axislegend(ax, position=:lt)
    
    # Display and save
    display(fig)
    save("exp/hockey/outputs/hockey_solution.png", fig)
end