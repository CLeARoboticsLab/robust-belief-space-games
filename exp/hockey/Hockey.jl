using Infiltrator
using TrajectoryGamesBase
using TrajectoryGamesExamples
using RobustBeliefGame
using LinearAlgebra
using BlockArrays
using Makie
using Symbolics

struct DummyEnvironment end

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
    # shooting_angle = @lift(attacker_shooting_angle($attacker_pos, goal_p1, goal_p2))
    # blocking_angle = @lift(defender_blocking_angle($attacker_pos, $defender_pos))
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
    Shot Likelihood: $(dual_round($shot_prob, digits=3))
    """
        # Attacker Angle: $(dual_round(rad2deg($shooting_angle), digits=1))°
    # Defender Angle: $(dual_round(rad2deg($blocking_angle), digits=1))°
    
    Label(fig[2, 1], angle_text, fontsize=20, tellwidth=false)

    axislegend(ax)
    display(fig)
    return fig
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

function belief_main()
    # Game Params
    horizon = 20
    dt = 0.3
    n=2
    goal_position = [
        [0.25, -1.5],
        [-0.25, -1.5],
    ]

    # Initial States/Beliefs
    gt_initial_state = mortar([ # gt = ground truth
        [-0.75, 1.5, 0.0, 0.0],
        [0.75, 5.0, 0.0, 0.0],  # Attacker
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


    # Environment
        # Dynamics
    function f(xs::BlockVector, us::BlockVector, ms::BlockVector)
        BlockVector(
                mapreduce(vcat, zip(xs.blocks, us.blocks, ms.blocks)) do (xᵢ, uᵢ, mᵢ)
                [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1] * xᵢ +
                [0.5*dt^2 0; 0 0.5*dt^2; dt 0; 0 dt] * uᵢ +
                [0 0 0 0; 0 0 0 0; 0 0 0.1*uᵢ[1] 0; 0 0 0 0.1*uᵢ[2]] * mᵢ
            end,
            [4, 4]
        )
    end

        # Sensor Models
    function h(xs::BlockVector, ns::BlockVector)
        BlockVector(
            mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
                [1 0 0 0; 0 1 0 0] * xᵢ + [0 0 xᵢ[3] 0; 0 0 0 xᵢ[4]] * nᵢ
            end,
            [2, 2]
        )
    end

    environment = BeliefEnvironment(f, gt_initial_state, h)

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
        control_effort = dot(us[Block(1)], us[Block(1)])
        return -1 * steal_prob + 3 * control_effort
    end
    function attacker_non_terminal_cost(bs::Beliefs, us)
        steal_prob = steal_liklihood(bs)
        control_effort = dot(us[Block(2)], us[Block(2)])
        return steal_prob + 3 * control_effort
    end    
    function shot_probability(bs::Beliefs)
        dist_penalty = 0.1
        block_max = 0.8
        block_falloff = 0.5
        attacker_uncertainty_penalty = 0.3  
        defender_uncertainty_penalty = 0.5 
        goal_center = (goal_position[1] + goal_position[2]) / 2

        attacker_pos = bs.beliefs[2].belief_mean[1:2] # Player 2 is Attacker
        defender_pos = bs.beliefs[1].belief_mean[1:2] # Player 1 is Defender
        attacker_pos_uncertainty = tr(bs.beliefs[2].belief_covariance[1:2, 1:2])
        defender_pos_uncertainty = tr(bs.beliefs[1].belief_covariance[1:2, 1:2])

        # Term 1: Base score, penalized by distance to goal and attacker's own uncertainty.
        dist_sq_to_goal = dot(attacker_pos - goal_center, attacker_pos - goal_center)
        distance_penalty = dist_penalty * dist_sq_to_goal
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
        shot_qual = shot_probability(bs)
        # Attacker wants to max shot quality, so we min its negative.
        # Don't let attacker get too far away from origin (area of play). This game construction
        #   doesn't allow for hard constraints.
        return -shot_qual + 0.01 * dot(bs.beliefs[2].belief_mean[1:2], bs.beliefs[2].belief_mean[1:2])
    end
    function defender_terminal_cost(bs::Beliefs)
        shot_qual = shot_probability(bs)
        return shot_qual + 0.01 * dot(bs.beliefs[1].belief_mean[1:2], bs.beliefs[1].belief_mean[1:2])
    end

    attacker_cost = BeliefCost(
        attacker_non_terminal_cost,
        attacker_terminal_cost,
    )
    defender_cost = BeliefCost(
        defender_non_terminal_cost,
        defender_terminal_cost,
    )

    bs_hockey_game = BeliefGame(
        environment,
        [defender_cost, attacker_cost],
        initial_beliefs,
        horizon,
        (; n=2, states=length.(gt_initial_state.blocks), controls=[2, 2], belief=length.(gt_initial_state.blocks), sensor=[2, 2]),
        gt_initial_state,
    )

    sol = solve(bs_hockey_game; debug=true)

    print("[Hockey] Ran, please implement vis.")
end