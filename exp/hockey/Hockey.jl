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

function save_solution(filename, robust_sol, non_robust_sol, goal_position)
    @save filename robust_sol non_robust_sol goal_position
end

function load_solution(filename)
    @load filename robust_sol non_robust_sol goal_position
    return robust_sol, non_robust_sol, goal_position
end

function belief_main(override_solution=false)
    solution_filename = "exp/hockey/outputs/hockey_solution.jld2"

    local robust_sol, non_robust_sol, goal_position

    if isfile(solution_filename) && !override_solution
        println("Loading solution from $solution_filename")
        robust_sol, non_robust_sol, goal_position = load_solution(solution_filename)
    else
        println("No solution file found. Running solver...")
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
                    [0.1 0 0 0; 0 0.1 0 0; 0 0 0.2*uᵢ[1] 0; 0 0 0 0.2*uᵢ[2]] * mᵢ
                end,
                [4, 4]
            )
        end

            # Sensor Models
        function h(xs::BlockVector, ns::BlockVector)
            BlockVector(
                mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
                    [1 0 0 0; 0 1 0 0] * xᵢ + [0 0 2*xᵢ[3] 0; 0 0 0 2*xᵢ[4]] * nᵢ
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
            # steal_prob = steal_liklihood(bs)
            steal_prob = dot(bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2], bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2])
            control_effort = dot(us[Block(1)], us[Block(1)])
            return -2 * steal_prob + 1 * control_effort
        end
        function attacker_non_terminal_cost(bs::Beliefs, us)
            # steal_prob = steal_liklihood(bs)
            steal_prob = dot(bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2], bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2])
            control_effort = dot(us[Block(2)], us[Block(2)])
            return exp(0.9 * steal_prob) + 2 * control_effort
        end    
        function shot_probability(bs::Beliefs)
            dist_penalty = 0.1
            block_max = 3
            block_falloff = 0.8
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
            # Attacker wants to max shot quality, so we min its negative.
            # Don't let attacker get too far away from origin (area of play). This game construction
            #   doesn't allow for hard constraints.
            return -shot_probability(bs)
        end
        function defender_terminal_cost(bs::Beliefs)
            return 10 * shot_probability(bs)
        end
        function nature_non_terminal_cost(bs::Beliefs, us::BlockVector)
            steal_prob = dot(bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2], bs.beliefs[1].belief_mean[1:2] - bs.beliefs[2].belief_mean[1:2])
            return steal_prob + exp(dot(us[Block(3)], us[Block(3)]))
        end
        function nature_terminal_cost(bs::Beliefs)
            return -10 * shot_probability(bs)
        end

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
            [defender_cost, attacker_cost],
            initial_beliefs,
            horizon,
            (; n=2, states=length.(gt_initial_state.blocks), controls=[2, 2], belief=length.(gt_initial_state.blocks), sensor=[2, 2]),
            gt_initial_state,
            false,
        )
        robust_hockey_game = BeliefGame(
            environment,
            [defender_cost, attacker_cost, nature_cost],
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

function visualize_belief_hockey_solution(sol, non_robust_sol, goal_position; graph_name="belief_hockey")
    robust_beliefs = sol[1]
    non_robust_beliefs = non_robust_sol[1]

    # Plotting Vars
    robust_attacker_color = :blue
    robust_defender_color = :red
    non_robust_attacker_color = :darkblue
    non_robust_defender_color = :darkred
    warning_color = :darkorange

    robust_attacker_means = [bs.beliefs[2].belief_mean for bs in robust_beliefs]
    robust_defender_means = [bs.beliefs[1].belief_mean for bs in robust_beliefs]
    non_robust_attacker_means = [bs.beliefs[2].belief_mean for bs in non_robust_beliefs]
    non_robust_defender_means = [bs.beliefs[1].belief_mean for bs in non_robust_beliefs]

    fig = Figure()

    # --- Top Row: Axis and Legend ---
    ax = Axis(fig[1, 1],
        title="Hockey Game Trajectories",
        xlabel="x position",
        ylabel="y position",
    )
    
    # --- Bottom Row: Controls ---
    control_grid = fig[2, 1:2] = GridLayout(tellheight=false)
    
    # Observables
    current_step = Observable(1)
    robust_opacity = Observable(1.0)
    non_robust_opacity = Observable(0.3)

    lines!(ax, [m[1] for m in robust_attacker_means], [m[2] for m in robust_attacker_means], label="Robust Attacker", color=robust_attacker_color, linewidth=3, alpha=robust_opacity)
    lines!(ax, [m[1] for m in robust_defender_means], [m[2] for m in robust_defender_means], label="Robust Defender", color=robust_defender_color, linewidth=3, alpha=robust_opacity)
    lines!(ax, [m[1] for m in non_robust_attacker_means], [m[2] for m in non_robust_attacker_means], label="Non-Robust Attacker", color=non_robust_attacker_color, linewidth=2, alpha=non_robust_opacity)
    lines!(ax, [m[1] for m in non_robust_defender_means], [m[2] for m in non_robust_defender_means], label="Non-Robust Defender", color=non_robust_defender_color, linewidth=2, alpha=non_robust_opacity)

    robust_attacker_pos = @lift(Point2f(robust_attacker_means[$current_step][1:2]))
    robust_defender_pos = @lift(Point2f(robust_defender_means[$current_step][1:2]))
    non_robust_attacker_pos = @lift(Point2f(non_robust_attacker_means[$current_step][1:2]))
    non_robust_defender_pos = @lift(Point2f(non_robust_defender_means[$current_step][1:2]))
    
    scatter!(ax, robust_attacker_pos, color=robust_attacker_color, markersize=20, alpha=robust_opacity)
    scatter!(ax, robust_defender_pos, color=robust_defender_color, markersize=20, alpha=robust_opacity)
    scatter!(ax, non_robust_attacker_pos, color=non_robust_attacker_color, markersize=15, alpha=non_robust_opacity)
    scatter!(ax, non_robust_defender_pos, color=non_robust_defender_color, markersize=15, alpha=non_robust_opacity)

    # Goal
    goal_posts = [[p[1] for p in goal_position], [p[2] for p in goal_position]]
    lines!(ax, goal_posts[1], goal_posts[2], label="Goal", color=:green, linewidth=5)

    # Velocity arrows
    velocity_scale = 0.5
    arrows!(ax, 
        @lift([robust_attacker_means[$current_step][1]]), @lift([robust_attacker_means[$current_step][2]]),
        @lift([velocity_scale * robust_attacker_means[$current_step][3]]), @lift([velocity_scale * robust_attacker_means[$current_step][4]]),
        color=robust_attacker_color, arrowsize=15, lengthscale=1.0, alpha=robust_opacity, linewidth=3)
    arrows!(ax, 
        @lift([robust_defender_means[$current_step][1]]), @lift([robust_defender_means[$current_step][2]]),
        @lift([velocity_scale * robust_defender_means[$current_step][3]]), @lift([velocity_scale * robust_defender_means[$current_step][4]]),
        color=robust_defender_color, arrowsize=15, lengthscale=1.0, alpha=robust_opacity, linewidth=3)
    arrows!(ax, 
        @lift([non_robust_attacker_means[$current_step][1]]), @lift([non_robust_attacker_means[$current_step][2]]),
        @lift([velocity_scale * non_robust_attacker_means[$current_step][3]]), @lift([velocity_scale * non_robust_attacker_means[$current_step][4]]),
        color=non_robust_attacker_color, arrowsize=12, lengthscale=1.0, alpha=non_robust_opacity, linewidth=2)
    arrows!(ax, 
        @lift([non_robust_defender_means[$current_step][1]]), @lift([non_robust_defender_means[$current_step][2]]),
        @lift([velocity_scale * non_robust_defender_means[$current_step][3]]), @lift([velocity_scale * non_robust_defender_means[$current_step][4]]),
        color=non_robust_defender_color, arrowsize=12, lengthscale=1.0, alpha=non_robust_opacity, linewidth=2)

    function get_uncertainty_visuals(mean, covariance, base_color)
        E = safe_eigen(covariance)
        scale = sqrt(-2 * log(1 - 0.95)) # 95% confidence
        t = range(0, 2π, 100)
        ellipse_points = [Point2f(scale * sqrt(E.values[1]) * E.vectors[1,1] * cos(θ) + scale * sqrt(E.values[2]) * E.vectors[1,2] * sin(θ) + mean[1],
                                    scale * sqrt(E.values[1]) * E.vectors[2,1] * cos(θ) + scale * sqrt(E.values[2]) * E.vectors[2,2] * sin(θ) + mean[2]) for θ in t]
        return (ellipse_points, (base_color, 0.2))
    end

    function get_velocity_uncertainty_visuals(mean, covariance, base_color)
        # Extract velocity components (indices 3-4) and their covariance
        vel_mean = mean[3:4]
        vel_cov = covariance[3:4, 3:4]
        
        velocity_scale = 0.5
        arrow_tip = mean[1:2] + velocity_scale * vel_mean        
        E = safe_eigen(vel_cov)
        scale = sqrt(-2 * log(1 - 0.95)) # 95% confidence
        t = range(0, 2π, 50)
        ellipse_points = [Point2f(velocity_scale * scale * sqrt(E.values[1]) * E.vectors[1,1] * cos(θ) + velocity_scale * scale * sqrt(E.values[2]) * E.vectors[1,2] * sin(θ) + arrow_tip[1],
                                    velocity_scale * scale * sqrt(E.values[1]) * E.vectors[2,1] * cos(θ) + velocity_scale * scale * sqrt(E.values[2]) * E.vectors[2,2] * sin(θ) + arrow_tip[2]) for θ in t]
        
        return (ellipse_points, (base_color, 0.2))
    end

    rob_att_ellipse_pts = Observable(Point2f[])
    rob_att_ellipse_color_tuple = Observable((robust_attacker_color, 0.2))
    rob_def_ellipse_pts = Observable(Point2f[])
    rob_def_ellipse_color_tuple = Observable((robust_defender_color, 0.2))
    non_rob_att_ellipse_pts = Observable(Point2f[])
    non_rob_att_ellipse_color_tuple = Observable((non_robust_attacker_color, 0.2))
    non_rob_def_ellipse_pts = Observable(Point2f[])
    non_rob_def_ellipse_color_tuple = Observable((non_robust_defender_color, 0.2))

    # Velocity uncertainty ellipses
    rob_att_vel_ellipse_pts = Observable(Point2f[])
    rob_att_vel_ellipse_color_tuple = Observable((robust_attacker_color, 0.15))
    rob_def_vel_ellipse_pts = Observable(Point2f[])
    rob_def_vel_ellipse_color_tuple = Observable((robust_defender_color, 0.15))
    non_rob_att_vel_ellipse_pts = Observable(Point2f[])
    non_rob_att_vel_ellipse_color_tuple = Observable((non_robust_attacker_color, 0.15))
    non_rob_def_vel_ellipse_pts = Observable(Point2f[])
    non_rob_def_vel_ellipse_color_tuple = Observable((non_robust_defender_color, 0.15))

    # Position uncertainty ellipses
    poly!(ax, rob_att_ellipse_pts, color=@lift(to_color($rob_att_ellipse_color_tuple)), strokecolor=@lift(to_color($rob_att_ellipse_color_tuple)), strokewidth=2, alpha=robust_opacity)
    poly!(ax, rob_def_ellipse_pts, color=@lift(to_color($rob_def_ellipse_color_tuple)), strokecolor=@lift(to_color($rob_def_ellipse_color_tuple)), strokewidth=2, alpha=robust_opacity)
    poly!(ax, non_rob_att_ellipse_pts, color=@lift(to_color($non_rob_att_ellipse_color_tuple)), strokecolor=@lift(to_color($non_rob_att_ellipse_color_tuple)), strokewidth=2, alpha=non_robust_opacity)
    poly!(ax, non_rob_def_ellipse_pts, color=@lift(to_color($non_rob_def_ellipse_color_tuple)), strokecolor=@lift(to_color($non_rob_def_ellipse_color_tuple)), strokewidth=2, alpha=non_robust_opacity)

    # Velocity uncertainty ellipses (dashed lines to distinguish from position uncertainty)
    poly!(ax, rob_att_vel_ellipse_pts, color=@lift(to_color($rob_att_vel_ellipse_color_tuple)), strokecolor=@lift(to_color($rob_att_vel_ellipse_color_tuple)), strokewidth=1, alpha=robust_opacity, linestyle=:dash)
    poly!(ax, rob_def_vel_ellipse_pts, color=@lift(to_color($rob_def_vel_ellipse_color_tuple)), strokecolor=@lift(to_color($rob_def_vel_ellipse_color_tuple)), strokewidth=1, alpha=robust_opacity, linestyle=:dash)
    poly!(ax, non_rob_att_vel_ellipse_pts, color=@lift(to_color($non_rob_att_vel_ellipse_color_tuple)), strokecolor=@lift(to_color($non_rob_att_vel_ellipse_color_tuple)), strokewidth=1, alpha=non_robust_opacity, linestyle=:dash)
    poly!(ax, non_rob_def_vel_ellipse_pts, color=@lift(to_color($non_rob_def_vel_ellipse_color_tuple)), strokecolor=@lift(to_color($non_rob_def_vel_ellipse_color_tuple)), strokewidth=1, alpha=non_robust_opacity, linestyle=:dash)

    Legend(fig[1, 2], ax, tellheight=false, tellwidth=true)
    
    slider = Slider(control_grid[1, 1], range=1:length(robust_attacker_means), startvalue=1)
    on(slider.value) do val; current_step[] = val; end
    
    Label(control_grid[1, 2], "Time:")
    Label(control_grid[1, 3], @lift("$(Int($current_step))"))

    button_grid = control_grid[2, 1:3] = GridLayout(tellwidth = false)
    focus_robust_btn = Button(button_grid[1, 1], label="Focus Robust")
    focus_non_robust_btn = Button(button_grid[1, 2], label="Focus Non-Robust")
    show_both_btn = Button(button_grid[1, 3], label="Show Both")
    
    on(focus_robust_btn.clicks) do n; robust_opacity[] = 1.0; non_robust_opacity[] = 0.1; end
    on(focus_non_robust_btn.clicks) do n; robust_opacity[] = 0.1; non_robust_opacity[] = 1.0; end
    on(show_both_btn.clicks) do n; robust_opacity[] = 0.7; non_robust_opacity[] = 0.7; end
    on(current_step) do val
        # Update position uncertainty ellipses
        rob_att_ellipse_pts[], rob_att_ellipse_color_tuple[] = get_uncertainty_visuals(robust_attacker_pos[], robust_beliefs[val].beliefs[2].belief_covariance, robust_attacker_color)
        rob_def_ellipse_pts[], rob_def_ellipse_color_tuple[] = get_uncertainty_visuals(robust_defender_pos[], robust_beliefs[val].beliefs[1].belief_covariance, robust_defender_color)
        non_rob_att_ellipse_pts[], non_rob_att_ellipse_color_tuple[] = get_uncertainty_visuals(non_robust_attacker_pos[], non_robust_beliefs[val].beliefs[2].belief_covariance, non_robust_attacker_color)
        non_rob_def_ellipse_pts[], non_rob_def_ellipse_color_tuple[] = get_uncertainty_visuals(non_robust_defender_pos[], non_robust_beliefs[val].beliefs[1].belief_covariance, non_robust_defender_color)
        
        # Update velocity uncertainty ellipses
        rob_att_vel_ellipse_pts[], rob_att_vel_ellipse_color_tuple[] = get_velocity_uncertainty_visuals(robust_attacker_means[val], robust_beliefs[val].beliefs[2].belief_covariance, robust_attacker_color)
        rob_def_vel_ellipse_pts[], rob_def_vel_ellipse_color_tuple[] = get_velocity_uncertainty_visuals(robust_defender_means[val], robust_beliefs[val].beliefs[1].belief_covariance, robust_defender_color)
        non_rob_att_vel_ellipse_pts[], non_rob_att_vel_ellipse_color_tuple[] = get_velocity_uncertainty_visuals(non_robust_attacker_means[val], non_robust_beliefs[val].beliefs[2].belief_covariance, non_robust_attacker_color)
        non_rob_def_vel_ellipse_pts[], non_rob_def_vel_ellipse_color_tuple[] = get_velocity_uncertainty_visuals(non_robust_defender_means[val], non_robust_beliefs[val].beliefs[1].belief_covariance, non_robust_defender_color)
    end
    
    set_close_to!(slider, 1)

    display(fig)
    save("exp/hockey/outputs/$graph_name.png", fig)
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