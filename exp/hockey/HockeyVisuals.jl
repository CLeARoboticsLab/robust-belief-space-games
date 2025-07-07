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

function get_position_uncertainty_ellipse(mean_pos, cov, confidence=0.95)
    # Ensure we only use the position components of the covariance
    pos_cov = cov[1:2, 1:2]
    E = safe_eigen(pos_cov)
    scale = sqrt(-2 * log(1 - confidence)) 
    t = range(0, 2π, 100)
    
    # Parametric equation for an ellipse
    return [Point2f(scale * E.vectors * [sqrt(E.values[1])*cos(θ), sqrt(E.values[2])*sin(θ)] + mean_pos[1:2]) for θ in t]
end

function get_velocity_uncertainty_ellipse(full_mean, cov, confidence=0.95, velocity_scale=0.5)
    # Extract velocity components
    vel_mean = full_mean[3:4]
    vel_cov = cov[3:4, 3:4]
    
    # The center of the ellipse is at the tip of the velocity vector
    arrow_tip = full_mean[1:2] + velocity_scale * vel_mean        
    
    E = safe_eigen(vel_cov)
    scale = sqrt(-2 * log(1 - confidence))
    t = range(0, 2π, 50)
    
    # Parametric equation for an ellipse, translated to the arrow tip
    return [Point2f(velocity_scale * scale * E.vectors * [sqrt(E.values[1])*cos(θ), sqrt(E.values[2])*sin(θ)] + arrow_tip) for θ in t]
end

function visualize_belief_hockey_solution(sol, non_robust_sol, goal_position; graph_name="belief_hockey")
    robust_beliefs = sol[1]
    non_robust_beliefs = non_robust_sol[1]

    # Plotting Vars
    robust_attacker_color = :blue
    robust_defender_color = :red
    non_robust_attacker_color = :darkblue
    non_robust_defender_color = :darkred

    robust_attacker_means = [bs.beliefs[1].belief_mean for bs in robust_beliefs]
    robust_defender_means = [bs.beliefs[2].belief_mean for bs in robust_beliefs]
    non_robust_attacker_means = [bs.beliefs[1].belief_mean for bs in non_robust_beliefs]
    non_robust_defender_means = [bs.beliefs[2].belief_mean for bs in non_robust_beliefs]

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

    rob_att_ellipse_pts = Observable(Point2f[])
    rob_def_ellipse_pts = Observable(Point2f[])
    non_rob_att_ellipse_pts = Observable(Point2f[])
    non_rob_def_ellipse_pts = Observable(Point2f[])

    # Velocity uncertainty ellipses
    rob_att_vel_ellipse_pts = Observable(Point2f[])
    rob_def_vel_ellipse_pts = Observable(Point2f[])
    non_rob_att_vel_ellipse_pts = Observable(Point2f[])
    non_rob_def_vel_ellipse_pts = Observable(Point2f[])

    # Position uncertainty ellipses
    poly!(ax, rob_att_ellipse_pts, color=(robust_attacker_color, 0.2), strokecolor=(robust_attacker_color, 0.2), strokewidth=2, alpha=robust_opacity)
    poly!(ax, rob_def_ellipse_pts, color=(robust_defender_color, 0.2), strokecolor=(robust_defender_color, 0.2), strokewidth=2, alpha=robust_opacity)
    poly!(ax, non_rob_att_ellipse_pts, color=(non_robust_attacker_color, 0.2), strokecolor=(non_robust_attacker_color, 0.2), strokewidth=2, alpha=non_robust_opacity)
    poly!(ax, non_rob_def_ellipse_pts, color=(non_robust_defender_color, 0.2), strokecolor=(non_robust_defender_color, 0.2), strokewidth=2, alpha=non_robust_opacity)

    # Velocity uncertainty ellipses (dashed lines to distinguish from position uncertainty)
    poly!(ax, rob_att_vel_ellipse_pts, color=(robust_attacker_color, 0.15), strokecolor=(robust_attacker_color, 0.15), strokewidth=1, alpha=robust_opacity, linestyle=:dash)
    poly!(ax, rob_def_vel_ellipse_pts, color=(robust_defender_color, 0.15), strokecolor=(robust_defender_color, 0.15), strokewidth=1, alpha=robust_opacity, linestyle=:dash)
    poly!(ax, non_rob_att_vel_ellipse_pts, color=(non_robust_attacker_color, 0.15), strokecolor=(non_robust_attacker_color, 0.15), strokewidth=1, alpha=non_robust_opacity, linestyle=:dash)
    poly!(ax, non_rob_def_vel_ellipse_pts, color=(non_robust_defender_color, 0.15), strokecolor=(non_robust_defender_color, 0.15), strokewidth=1, alpha=non_robust_opacity, linestyle=:dash)

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
        rob_att_ellipse_pts[] = get_position_uncertainty_ellipse(robust_attacker_pos[], robust_beliefs[val].beliefs[1].belief_covariance)
        rob_def_ellipse_pts[] = get_position_uncertainty_ellipse(robust_defender_pos[], robust_beliefs[val].beliefs[2].belief_covariance)
        non_rob_att_ellipse_pts[] = get_position_uncertainty_ellipse(non_robust_attacker_pos[], non_robust_beliefs[val].beliefs[1].belief_covariance)
        non_rob_def_ellipse_pts[] = get_position_uncertainty_ellipse(non_robust_defender_pos[], non_robust_beliefs[val].beliefs[2].belief_covariance)
        
        # Update velocity uncertainty ellipses
        rob_att_vel_ellipse_pts[] = get_velocity_uncertainty_ellipse(robust_attacker_means[val], robust_beliefs[val].beliefs[1].belief_covariance)
        rob_def_vel_ellipse_pts[] = get_velocity_uncertainty_ellipse(robust_defender_means[val], robust_beliefs[val].beliefs[2].belief_covariance)
        non_rob_att_vel_ellipse_pts[] = get_velocity_uncertainty_ellipse(non_robust_attacker_means[val], non_robust_beliefs[val].beliefs[1].belief_covariance)
        non_rob_def_vel_ellipse_pts[] = get_velocity_uncertainty_ellipse(non_robust_defender_means[val], non_robust_beliefs[val].beliefs[2].belief_covariance)
    end
    
    set_close_to!(slider, 1)

    display(fig)
    save("exp/hockey/outputs/$graph_name.png", fig)
end

function visualize_receding_horizon_solution(gt_state_history, belief_history, planned_trajectories, observations,goal_position; is_robust)
    fig = Figure()
    
    # --- Top Row: Axis and Legend ---
    ax = Axis(fig[1, 1],
        title="Receding Horizon Hockey Game",
        xlabel="x position",
        ylabel="y position",
    )
    
    # --- Bottom Row: Controls ---
    control_grid = fig[2, 1:2] = GridLayout(tellheight=false)
    
    current_step = Observable(1)
    horizon = length(gt_state_history) - 1
    
    # Colors
    attacker_color = :blue
    defender_color = :red
    gt_color = :black
    
    # Opacities
    belief_opacity = 0.3
    plan_opacity = 0.8
    attacker_plan_opacity = Observable(plan_opacity)
    defender_plan_opacity = Observable(plan_opacity)
    
    # --- Static trajectory plotting ---
    # Plot full ground truth trajectories as static lines
    attacker_gt_x = [s[Block(1)][1] for s in gt_state_history]
    attacker_gt_y = [s[Block(1)][2] for s in gt_state_history]
    defender_gt_x = [s[Block(2)][1] for s in gt_state_history]
    defender_gt_y = [s[Block(2)][2] for s in gt_state_history]
    
    lines!(ax, attacker_gt_x, attacker_gt_y, color=gt_color, linewidth=3, label="Attacker Ground Truth")
    lines!(ax, defender_gt_x, defender_gt_y, color=gt_color, linewidth=3, linestyle=:dash, label="Defender Ground Truth")
    
    # Plot full belief trajectories as static lines
    attacker_belief_x = [b.beliefs[1].belief_mean[1] for b in belief_history]
    attacker_belief_y = [b.beliefs[1].belief_mean[2] for b in belief_history]
    defender_belief_x = [b.beliefs[2].belief_mean[1] for b in belief_history]
    defender_belief_y = [b.beliefs[2].belief_mean[2] for b in belief_history]

    lines!(ax, attacker_belief_x, attacker_belief_y, color=attacker_color, linewidth=2, alpha=belief_opacity, label="Attacker Belief Trajectory")
    lines!(ax, defender_belief_x, defender_belief_y, color=defender_color, linewidth=2, alpha=belief_opacity, label="Defender Belief Trajectory")
    
    # --- Goal ---
    lines!(ax, [p[1] for p in goal_position], [p[2] for p in goal_position], color=:green, linewidth=5, label="Goal")
    
    # --- Current belief means and planned trajectory (observables) ---
    current_belief_state = @lift belief_history[$current_step]
    attacker_pos = @lift Point2f($current_belief_state.beliefs[1].belief_mean[1:2])
    defender_pos = @lift Point2f($current_belief_state.beliefs[2].belief_mean[1:2])
    
    # Planned trajectories (what they plan to do from the current step)
    planned_attacker_traj = @lift [Point2f(m.beliefs[1].belief_mean[1:2]) for m in planned_trajectories[$current_step][1]]
    planned_defender_traj = @lift [Point2f(m.beliefs[2].belief_mean[1:2]) for m in planned_trajectories[$current_step][2]]
    
    # --- Planned trajectories for the current step (higher opacity) ---
    lines!(ax, planned_attacker_traj, color=attacker_color, linestyle=:dash, linewidth=3, alpha=attacker_plan_opacity, label="Attacker Plan")
    lines!(ax, planned_defender_traj, color=defender_color, linestyle=:dot, linewidth=3, alpha=defender_plan_opacity, label="Defender Plan")
    
    # --- Current belief positions (as markers) ---
    scatter!(ax, attacker_pos, color=attacker_color, markersize=20, label="Current Attacker Belief")
    scatter!(ax, defender_pos, color=defender_color, markersize=20, label="Current Defender Belief")

    # --- Observations ---
    # Plot all observations with low opacity
    attacker_obs_x = [obs[1] for obs in observations]
    attacker_obs_y = [obs[2] for obs in observations]
    defender_obs_x = [obs[3] for obs in observations]
    defender_obs_y = [obs[4] for obs in observations]
    scatter!(ax, attacker_obs_x, attacker_obs_y, color=attacker_color, markersize=15, alpha=0.3, label="Attacker Observations")
    scatter!(ax, defender_obs_x, defender_obs_y, color=defender_color, markersize=15, alpha=0.3, label="Defender Observations")

    # Plot current observation with full opacity
    current_attacker_obs = @lift Point2f(observations[$current_step][1:2])
    current_defender_obs = @lift Point2f(observations[$current_step][3:4])
    scatter!(ax, current_attacker_obs, color=attacker_color, markersize=20, label="Current Attacker Observation") 
    scatter!(ax, current_defender_obs, color=defender_color, markersize=20, label="Current Defender Observation")
    
    # --- Belief uncertainty ellipses ---
    attacker_ellipse_pts = Observable(Point2f[])
    defender_ellipse_pts = Observable(Point2f[])
    attacker_vel_ellipse_pts = Observable(Point2f[])
    defender_vel_ellipse_pts = Observable(Point2f[])

    poly!(ax, attacker_ellipse_pts, color=(attacker_color, 0.2), strokecolor=(attacker_color, 0.2), strokewidth=2)
    poly!(ax, defender_ellipse_pts, color=(defender_color, 0.2), strokecolor=(defender_color, 0.2), strokewidth=2)
    poly!(ax, attacker_vel_ellipse_pts, color=(attacker_color, 0.15), strokecolor=(attacker_color, 0.15), strokewidth=1, linestyle=:dash)
    poly!(ax, defender_vel_ellipse_pts, color=(defender_color, 0.15), strokecolor=(defender_color, 0.15), strokewidth=1, linestyle=:dash)

    on(current_step) do val
        current_belief = belief_history[val]
        # Update position uncertainty ellipses
        attacker_ellipse_pts[] = get_position_uncertainty_ellipse(current_belief.beliefs[1].belief_mean[1:2], current_belief.beliefs[1].belief_covariance)
        defender_ellipse_pts[] = get_position_uncertainty_ellipse(current_belief.beliefs[2].belief_mean[1:2], current_belief.beliefs[2].belief_covariance)
        
        # Update velocity uncertainty ellipses
        attacker_vel_ellipse_pts[] = get_velocity_uncertainty_ellipse(current_belief.beliefs[1].belief_mean, current_belief.beliefs[1].belief_covariance)
        defender_vel_ellipse_pts[] = get_velocity_uncertainty_ellipse(current_belief.beliefs[2].belief_mean, current_belief.beliefs[2].belief_covariance)
    end
    
    # --- Controls ---
    slider = Slider(control_grid[1, 1], range=1:horizon, startvalue=1)
    on(slider.value) do val; current_step[] = val; end
    
    Label(control_grid[1, 2], "Time:")
    Label(control_grid[1, 3], @lift("$(Int($current_step))"))

    button_grid = control_grid[2, 1:3] = GridLayout(tellwidth = false)
    highlight_attacker_btn = Button(button_grid[1, 1], label="Highlight Attacker Plan")
    highlight_defender_btn = Button(button_grid[1, 2], label="Highlight Defender Plan")
    show_both_btn = Button(button_grid[1, 3], label="Show Both Plans")
    
    on(highlight_attacker_btn.clicks) do n
        attacker_plan_opacity[] = 1.0
        defender_plan_opacity[] = 0.1
    end
    on(highlight_defender_btn.clicks) do n
        attacker_plan_opacity[] = 0.1
        defender_plan_opacity[] = 1.0
    end
    on(show_both_btn.clicks) do n
        attacker_plan_opacity[] = 0.8
        defender_plan_opacity[] = 0.8
    end
    
    # Put legend in top-right corner without taking too much space
    Legend(fig[1, 2], ax, tellheight=false, tellwidth=true)
    
    set_close_to!(slider, 1)
    
    display(fig)
    save("exp/hockey/outputs/receding_horizon_$(is_robust ? "robust" : "non_robust").png", fig)
end