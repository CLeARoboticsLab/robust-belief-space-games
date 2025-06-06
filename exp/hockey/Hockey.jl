using Infiltrator
using TrajectoryGamesBase
using TrajectoryGamesExamples
using RobustBeliefGame
using LinearAlgebra
using GLMakie
using BlockArrays
using Makie



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
        prob = shot_probability(xs[t][1:2], xs[t][5:6], goal_position[1], goal_position[2])
        -1 * prob + 0.05 * norm(us[t][1:2])
    end
end

function defender_cost(xs, us; goal_position)
    return mapreduce(+, eachindex(xs)) do t
        prob = shot_probability(xs[t][1:2], xs[t][5:6], goal_position[1], goal_position[2])
        1 * prob + 0.05 * norm(us[t][3:4])
    end
end

function attacker_shooting_angle(attacker_pos, goal_p1, goal_p2)
    return find_angle(attacker_pos, goal_p1, goal_p2)
end

function defender_blocking_angle(attacker_pos, defender_pos)
    defender_range = 0.1
    v_ad = defender_pos - attacker_pos
    defender_guard = defender_range * [0.0 1.0; -1.0 0.0] * (v_ad / (norm(v_ad) + 1e-9))
    return find_angle(attacker_pos, defender_pos + defender_guard, defender_pos - defender_guard)
end

function signed_angle(v1, v2)
    return atan(v1[1]*v2[2] - v1[2]*v2[1], v1[1]*v2[1] + v1[2]*v2[2])
end

function soft_max(xs::AbstractArray; alpha=20)
    return log(sum(exp.(alpha .* xs))) / alpha
end

function shot_probability(attacker_pos, defender_pos, goal_p1, goal_p2)
    shooting_angle = attacker_shooting_angle(attacker_pos, goal_p1, goal_p2)

    blocking_angle = defender_blocking_angle(attacker_pos, defender_pos)

    goal_center = (goal_p1 + goal_p2) / 2
    separation_angle = find_angle(attacker_pos, goal_center, defender_pos)
    alignment_factor = exp(-5 * separation_angle)
    
    effective_block = blocking_angle * alignment_factor
    
    # The final probability is the shooting angle, reduced by the effective block.
    # soft_max ensures it's a smooth function and >= 0.
    return soft_max([0.0, shooting_angle - effective_block]; alpha=100)
end

function find_angle(p1, p2, p3)
    v1 = p2 .- p1
    v2 = p3 .- p1
    n1 = norm(v1)
    n2 = norm(v2)
    
    return 2 * atan(norm(v1*n2 - n1*v2), norm(v1*n2 + n1*v2))
end

"""
    test_shot_probability_viz()

An interactive visualization to test the `shot_probability` function.
You can drag the attacker (blue) and defender (red) around to see how the shooting and blocking angles change.
The green line represents the goal.

To run, simply call `test_shot_probability_viz()` from the REPL after loading the file.
"""
function test_shot_probability_viz()
    fig = Figure(resolution=(1000, 800))
    ax = Axis(fig[1, 1],
        title="Shot Probability Visualization",
        xlabel="x position",
        ylabel="y position",
        aspect=1
    )
    deregister_interaction!(ax, :rectanglezoom)

    attacker_pos = Observable(Point2f(0.75, 5.0))
    defender_pos = Observable(Point2f(-0.75, 1.5))
    goal_p1 = Point2f(-1.5, 0.25)
    goal_p2 = Point2f(-1.5, -0.25)

    # The goal
    lines!(ax, [goal_p1, goal_p2], color=:green, linewidth=5, label="Goal")
    
    # The players
    attacker_plot = scatter!(ax, attacker_pos, color=:blue, markersize=20, label="Attacker")
    defender_plot = scatter!(ax, defender_pos, color=:red, markersize=20, label="Defender")
    
    # The angles
    shooting_angle = @lift(attacker_shooting_angle($attacker_pos, goal_p1, goal_p2))
    blocking_angle = @lift(defender_blocking_angle($attacker_pos, $defender_pos))
    shot_prob = @lift(shot_probability($attacker_pos, $defender_pos, goal_p1, goal_p2))

    # Angle visualizations
    poly!(ax, @lift([$attacker_pos, goal_p1, goal_p2]), color=(:blue, 0.2), strokecolor=:blue, strokewidth=1, pickable=false)

    defender_range = 0.1
    poly_blocking_points = @lift begin
        v = $defender_pos - $attacker_pos
        # Avoid division by zero if points are on top of each other
        if norm(v) > 1e-9
            vn = v / norm(v)
            # rotate by 90 degrees
            guard_vec = defender_range * [0.0 1.0; -1.0 0.0] * vn
            p1 = $defender_pos + guard_vec
            p2 = $defender_pos - guard_vec
            [$attacker_pos, p1, p2]
        else
            [$attacker_pos, $attacker_pos, $attacker_pos]
        end
    end
    poly!(ax, poly_blocking_points, color=(:red, 0.2), strokecolor=:red, strokewidth=1, pickable=false)
    
    # Manual dragging interaction
    dragged_plot = Observable{Any}(nothing)
    register_interaction!(ax, :manual_drag) do event::MouseEvent, axis
        if event.type === MouseEventTypes.leftdown
            plt, _ = pick(axis)
            if plt === attacker_plot || plt === defender_plot
                dragged_plot[] = plt
                return Consume(true)
            end
        elseif event.type === MouseEventTypes.leftdrag
            if !isnothing(dragged_plot[])
                pos = Point2f(mouseposition(axis))
                if dragged_plot[] === attacker_plot
                    attacker_pos[] = pos
                elseif dragged_plot[] === defender_plot
                    defender_pos[] = pos
                end
                return Consume(true)
            end
        elseif event.type === MouseEventTypes.leftup
            if !isnothing(dragged_plot[])
                dragged_plot[] = nothing
                return Consume(true)
            end
        end
        return Consume(false)
    end

    # Text display
    angle_text = @lift """
    Attacker Angle: $(round(rad2deg($shooting_angle), digits=1))°
    Defender Angle: $(round(rad2deg($blocking_angle), digits=1))°
    Shot Likelihood: $(round($shot_prob, digits=3))
    """
    
    Label(fig[2, 1], angle_text, fontsize=20, tellwidth=false)

    axislegend(ax)
    display(fig)
    return fig
end

function main()
    horizon = 10
    initial_states = [
        [0.75, 5.0, 0.0, 0.0],  # Attacker
        [-0.75, 1.5, 0.0, 0.0],
    ]
    # A single goal defined by its two posts.
    goal_position = [
            [-1.5, 0.25],
            [-1.5, -0.25],
        ]


    # Visual
    goal_line_width = 5

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
    goal_posts = [[p[1] for p in goal_position], [p[2] for p in goal_position]]
    goal = lines!(ax, goal_posts[1], goal_posts[2], label="Goal", color=:green, linewidth=goal_line_width)

    
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