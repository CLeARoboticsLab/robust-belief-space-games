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

function visualize_belief_hockey_solution(sol, non_robust_sol, goal_position; graph_name="belief_hockey")
    robust_beliefs = sol[1]
    non_robust_beliefs = non_robust_sol[1]

    # Plotting Vars
    robust_attacker_color = :blue
    robust_defender_color = :red
    non_robust_attacker_color = :purple
    non_robust_defender_color = :purple

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
    lines!(ax, [m[1] for m in non_robust_attacker_means], [m[2] for m in non_robust_attacker_means], label="Non-Robust", color=non_robust_attacker_color, linewidth=2, alpha=non_robust_opacity)
    lines!(ax, [m[1] for m in non_robust_defender_means], [m[2] for m in non_robust_defender_means], color=non_robust_defender_color, linewidth=2, alpha=non_robust_opacity)

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

    rob_att_ellipse_pts = Observable(Point2f[])
    rob_def_ellipse_pts = Observable(Point2f[])
    non_rob_att_ellipse_pts = Observable(Point2f[])
    non_rob_def_ellipse_pts = Observable(Point2f[])

    # Position uncertainty ellipses
    poly!(ax, rob_att_ellipse_pts, color=(robust_attacker_color, 0.2), strokecolor=(robust_attacker_color, 0.2), strokewidth=2, alpha=robust_opacity)
    poly!(ax, rob_def_ellipse_pts, color=(robust_defender_color, 0.2), strokecolor=(robust_defender_color, 0.2), strokewidth=2, alpha=robust_opacity)
    poly!(ax, non_rob_att_ellipse_pts, color=(non_robust_attacker_color, 0.2), strokecolor=(non_robust_attacker_color, 0.2), strokewidth=2, alpha=non_robust_opacity)
    poly!(ax, non_rob_def_ellipse_pts, color=(non_robust_defender_color, 0.2), strokecolor=(non_robust_defender_color, 0.2), strokewidth=2, alpha=non_robust_opacity)

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
        
    end
    
    set_close_to!(slider, 1)

    display(fig)
    save("exp/hockey/outputs/$graph_name.png", fig)
end

function visualize_receding_horizon_solution(gt_state_history, belief_history, planned_trajectories, observations, goal_position; is_robust, intermediate_planned_trajectories, us_history, nature_us_history, planned_us_history, solution_history)
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
    attacker_color = :red
    defender_color = :blue
    gt_color = :black
    nature_color = :green
    non_robust_plan_color = :purple
    robust_plan_color = :orange

    # Opacities & Visibilities
    plan_opacity = 1.0
    gt_opacity = Observable(1.0)
    belief_opacity = Observable(1.0)
    non_robust_plan_opacity = Observable(plan_opacity)
    robust_plan_opacity = Observable(plan_opacity)
    observation_opacity = Observable(1.0)
    
    # --- Solver Iteration Controls ---
    show_solver_iterations = Observable(false)
    
    num_iterations = @lift begin
        if $current_step <= length(intermediate_planned_trajectories)
            sols_at_t = intermediate_planned_trajectories[$current_step]
            if length(sols_at_t) >= 2 && !isempty(sols_at_t[1]) && !isempty(sols_at_t[2])
                min(length(sols_at_t[1]), length(sols_at_t[2]))
            else
                0
            end
        else
            0
        end
    end

    current_iteration = Observable(1)

    non_robust_plan = @lift begin
        if $show_solver_iterations
            if $num_iterations > 0
                iter_idx = clamp($current_iteration, 1, $num_iterations)
                intermediate_planned_trajectories[$current_step][1][iter_idx]
            else
                []
            end
        else
            # Ensure we don't error if planned_trajectories is short
            $current_step <= length(planned_trajectories) ? planned_trajectories[$current_step][1] : []
        end
    end
    
    robust_plan = @lift begin
        if $show_solver_iterations
            if $num_iterations > 0
                iter_idx = clamp($current_iteration, 1, $num_iterations)
                intermediate_planned_trajectories[$current_step][2][iter_idx]
            else
                []
            end
        else
            # Ensure we don't error if planned_trajectories is short
            $current_step <= length(planned_trajectories) && length(planned_trajectories[$current_step]) > 1 ? planned_trajectories[$current_step][2] : []
        end
    end

    # --- Static trajectory plotting ---
    attacker_gt_x = [s[Block(1)][1] for s in gt_state_history]
    attacker_gt_y = [s[Block(1)][2] for s in gt_state_history]
    defender_gt_x = [s[Block(2)][1] for s in gt_state_history]
    defender_gt_y = [s[Block(2)][2] for s in gt_state_history]
    
    lines!(ax, attacker_gt_x, attacker_gt_y, color=gt_color, linewidth=3, alpha=@lift($gt_opacity * ($show_solver_iterations ? 0.2 : 1.0)))
    lines!(ax, defender_gt_x, defender_gt_y, color=gt_color, linewidth=3, alpha=@lift($gt_opacity * ($show_solver_iterations ? 0.2 : 1.0)))
    
    # Beliefs from Attacker's perspective
    attacker_belief_self_x = [b.beliefs[1].belief_mean[1] for b in belief_history]
    attacker_belief_self_y = [b.beliefs[1].belief_mean[2] for b in belief_history]
    attacker_belief_other_x = [b.beliefs[2].belief_mean[1] for b in belief_history]
    attacker_belief_other_y = [b.beliefs[2].belief_mean[2] for b in belief_history]
    
    # Beliefs from Defender's perspective
    defender_belief_other_x = [b.beliefs[3].belief_mean[1] for b in belief_history]
    defender_belief_other_y = [b.beliefs[3].belief_mean[2] for b in belief_history]
    defender_belief_self_x = [b.beliefs[4].belief_mean[1] for b in belief_history]
    defender_belief_self_y = [b.beliefs[4].belief_mean[2] for b in belief_history]

    lines!(ax, attacker_belief_self_x, attacker_belief_self_y, color=attacker_color, linewidth=2, label="Attacker's Belief (executed)", alpha=@lift($belief_opacity * ($show_solver_iterations ? 0.2 : 1.0)))
    lines!(ax, defender_belief_self_x, defender_belief_self_y, color=defender_color, linewidth=2, label="Defender's Belief (executed)", alpha=@lift($belief_opacity * ($show_solver_iterations ? 0.2 : 1.0)))
    
    # --- Goal ---
    lines!(ax, [p[1] for p in goal_position], [p[2] for p in goal_position], color=:green, linewidth=5, label="Goal")
    
    # --- Current belief means and planned trajectory (observables) ---
    current_belief_state = @lift belief_history[$current_step]
    attacker_pos_self = @lift Point2f($current_belief_state.beliefs[1].belief_mean[1:2])
    attacker_pos_other = @lift Point2f($current_belief_state.beliefs[2].belief_mean[1:2])
    defender_pos_other = @lift Point2f($current_belief_state.beliefs[3].belief_mean[1:2])
    defender_pos_self = @lift Point2f($current_belief_state.beliefs[4].belief_mean[1:2])
    
    # Planned trajectories from both non-robust and robust solves
    non_robust_attacker_plan = @lift isempty($non_robust_plan) ? Point2f[] : [Point2f(m.beliefs[1].belief_mean[1:2]) for m in $non_robust_plan]
    non_robust_defender_plan = @lift isempty($non_robust_plan) ? Point2f[] : [Point2f(m.beliefs[2].belief_mean[1:2]) for m in $non_robust_plan]
    robust_attacker_plan = @lift isempty($robust_plan) ? Point2f[] : [Point2f(m.beliefs[1].belief_mean[1:2]) for m in $robust_plan]
    robust_defender_plan = @lift isempty($robust_plan) ? Point2f[] : [Point2f(m.beliefs[4].belief_mean[1:2]) for m in $robust_plan]
    robust_attacker_plan_other = @lift isempty($robust_plan) ? Point2f[] : [Point2f(m.beliefs[2].belief_mean[1:2]) for m in $robust_plan]
    robust_defender_plan_other = @lift isempty($robust_plan) ? Point2f[] : [Point2f(m.beliefs[3].belief_mean[1:2]) for m in $robust_plan]

    # Planned actions
    planned_us = @lift $current_step <= length(planned_us_history) ? planned_us_history[$current_step] : []
    
    non_robust_planned_us = @lift isempty($planned_us) ? [] : $planned_us[1]
    non_robust_attacker_actions = @lift isempty($non_robust_planned_us) ? Point2f[] : [Point2f(u[Block(1)]) for u in $non_robust_planned_us]
    non_robust_defender_actions = @lift isempty($non_robust_planned_us) ? Point2f[] : [Point2f(u[Block(2)]) for u in $non_robust_planned_us]
    
    arrows!(ax, non_robust_attacker_plan, non_robust_attacker_actions, color=attacker_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($non_robust_plan_opacity > 0.1))
    arrows!(ax, non_robust_defender_plan, non_robust_defender_actions, color=defender_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($non_robust_plan_opacity > 0.1))

    robust_planned_us = @lift isempty($planned_us) || length($planned_us) < 2 ? [] : $planned_us[2]
    robust_attacker_actions = @lift isempty($robust_planned_us) ? Point2f[] : [Point2f(u[Block(1)]) for u in $robust_planned_us]
    robust_defender_actions = @lift isempty($robust_planned_us) ? Point2f[] : [Point2f(u[Block(2)]) for u in $robust_planned_us]
    
    arrows!(ax, robust_attacker_plan, robust_attacker_actions, color=attacker_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1))
    arrows!(ax, robust_defender_plan, robust_defender_actions, color=defender_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1))

    # --- Arrows on attacker's belief of other plan
    arrows!(ax, robust_attacker_plan_other, robust_defender_actions, color=defender_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1))

    # --- Arrows on defender's belief of other plan
    arrows!(ax, robust_defender_plan_other, robust_attacker_actions, color=attacker_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1))

    # --- Nature's actions on robust plan
    nature_actions_on_robust_plan = @lift begin
        if $robust_plan_opacity > 0.1 && $current_step <= length(solution_history)
            sols = solution_history[$current_step]
            if length(sols) >= 2
                robust_us = sols[2][2] # us
                map(u -> Point2f(u[Block(3)]), robust_us)
            else
                Point2f[]
            end
        else
            Point2f[]
        end
    end
    arrows!(ax, robust_attacker_plan, nature_actions_on_robust_plan, color=nature_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1))
    arrows!(ax, robust_defender_plan_other, nature_actions_on_robust_plan, color=nature_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1))



    lines!(ax, robust_attacker_plan_other, color=attacker_color, linestyle=:dash, linewidth=2, alpha=robust_plan_opacity, label="Robust Attacker Plan (other)")
    scatter!(ax, robust_attacker_plan_other, color=attacker_color, marker=:xcross, markersize=8, alpha=robust_plan_opacity)

    lines!(ax, robust_defender_plan_other, color=defender_color, linestyle=:dash, linewidth=2, alpha=robust_plan_opacity, label="Robust Defender Plan (other)")
    scatter!(ax, robust_defender_plan_other, color=defender_color, marker=:cross, markersize=8, alpha=robust_plan_opacity)


    lines!(ax, non_robust_attacker_plan, color=attacker_color, linewidth=3, alpha=non_robust_plan_opacity, label="Non-Robust Attacker Plan")
    scatter!(ax, non_robust_attacker_plan, color=attacker_color, markersize=10, alpha=non_robust_plan_opacity)
    lines!(ax, non_robust_defender_plan, color=defender_color, linewidth=3, alpha=non_robust_plan_opacity, label="Non-Robust Defender Plan")
    scatter!(ax, non_robust_defender_plan, color=defender_color, marker=:xcross, markersize=10, alpha=non_robust_plan_opacity)
    lines!(ax, robust_attacker_plan, color=attacker_color, linestyle=:dash, linewidth=3, alpha=robust_plan_opacity, label="Robust Attacker Plan (self)")
    scatter!(ax, robust_attacker_plan, color=attacker_color, markersize=10, alpha=robust_plan_opacity)
    lines!(ax, robust_defender_plan, color=defender_color, linestyle=:dash, linewidth=3, alpha=robust_plan_opacity, label="Robust Defender Plan (self)")
    scatter!(ax, robust_defender_plan, color=defender_color, marker=:xcross, markersize=10, alpha=robust_plan_opacity)

    # --- Current belief positions (as markers) ---
    scatter!(ax, attacker_pos_self, color=attacker_color, markersize=20)
    scatter!(ax, defender_pos_self, color=defender_color, markersize=20)

    # --- Player Actions ---
    current_us = @lift us_history[$current_step]
    attacker_action = @lift Point2f($current_us[Block(1)])
    defender_action = @lift Point2f($current_us[Block(2)])
    
    arrows!(ax, @lift([$attacker_pos_self]), @lift([$attacker_action]), color=attacker_color, linewidth=3, arrowsize=15, label="Attacker Action", alpha=0.7)
    arrows!(ax, @lift([$defender_pos_self]), @lift([$defender_action]), color=defender_color, linewidth=3, arrowsize=15, label="Defender Action", alpha=0.7)

    if is_robust
        nature_action_on_attacker = @lift Point2f(nature_us_history[$current_step][1:2])
        nature_action_on_defender = @lift Point2f(nature_us_history[$current_step][3:4])

        nature_start_pos_attacker = @lift begin
            if !isempty($robust_attacker_plan) && !isempty($robust_attacker_actions)
                $robust_attacker_plan[1] + $robust_attacker_actions[1]
            else
                gt_s = gt_state_history[$current_step]
                Point2f(gt_s[Block(1)])
            end
        end
        nature_start_pos_defender = @lift begin
            if !isempty($robust_defender_plan) && !isempty($robust_defender_actions)
                $robust_defender_plan[1] + $robust_defender_actions[1]
            else
                gt_s = gt_state_history[$current_step]
                Point2f(gt_s[Block(2)])
            end
        end

        arrows!(ax, @lift([$nature_start_pos_attacker]), @lift([$nature_action_on_attacker]), color=nature_color, linewidth=3, arrowsize=15, alpha=0.7, visible=@lift($robust_plan_opacity > 0.1))
        arrows!(ax, @lift([$nature_start_pos_defender]), @lift([$nature_action_on_defender]), color=nature_color, linewidth=3, arrowsize=15, label="Nature", alpha=0.7, visible=@lift($robust_plan_opacity > 0.1))
    end

    # --- Observations ---
    attacker_obs_x = [obs[1] for obs in observations]
    attacker_obs_y = [obs[2] for obs in observations]
    defender_obs_x = [obs[3] for obs in observations]
    defender_obs_y = [obs[4] for obs in observations]
    scatter!(ax, attacker_obs_x, attacker_obs_y, color=attacker_color, markersize=15, alpha=@lift(0.3 * $observation_opacity), label="Attacker Observations")
    scatter!(ax, defender_obs_x, defender_obs_y, color=defender_color, markersize=15, alpha=@lift(0.3 * $observation_opacity), label="Defender Observations")

    current_attacker_obs = @lift Point2f(observations[$current_step][1:2])
    current_defender_obs = @lift Point2f(observations[$current_step][3:4])
    scatter!(ax, current_attacker_obs, color=attacker_color, markersize=20, marker=:utriangle, alpha=observation_opacity, label="Current Attacker Observation") 
    scatter!(ax, current_defender_obs, color=defender_color, markersize=20, marker=:utriangle, alpha=observation_opacity, label="Current Defender Observation")
    
    # --- Belief uncertainty ellipses ---
    attacker_ellipse_self_pts = Observable(Point2f[])
    attacker_ellipse_other_pts = Observable(Point2f[])
    defender_ellipse_other_pts = Observable(Point2f[])
    defender_ellipse_self_pts = Observable(Point2f[])

    poly!(ax, attacker_ellipse_self_pts, color=(attacker_color, 0.2), strokecolor=(attacker_color, 0.2), strokewidth=2)
    poly!(ax, attacker_ellipse_other_pts, color=(attacker_color, 0.2), strokecolor=(attacker_color, 0.2), strokewidth=2)
    poly!(ax, defender_ellipse_other_pts, color=(defender_color, 0.2), strokecolor=(defender_color, 0.2), strokewidth=2)
    poly!(ax, defender_ellipse_self_pts, color=(defender_color, 0.2), strokecolor=(defender_color, 0.2), strokewidth=2)

    on(current_step) do val
        current_belief = belief_history[val]
        attacker_ellipse_self_pts[] = get_position_uncertainty_ellipse(current_belief.beliefs[1].belief_mean[1:2], current_belief.beliefs[1].belief_covariance)
        attacker_ellipse_other_pts[] = get_position_uncertainty_ellipse(current_belief.beliefs[2].belief_mean[1:2], current_belief.beliefs[2].belief_covariance)
        defender_ellipse_other_pts[] = get_position_uncertainty_ellipse(current_belief.beliefs[3].belief_mean[1:2], current_belief.beliefs[3].belief_covariance)
        defender_ellipse_self_pts[] = get_position_uncertainty_ellipse(current_belief.beliefs[4].belief_mean[1:2], current_belief.beliefs[4].belief_covariance)
        
    end
    
    # --- Controls ---
    time_slider_grid = control_grid[1, 1] = GridLayout(tellwidth=false)
    time_slider = Slider(time_slider_grid[1, 2], range=1:horizon, startvalue=1)
    on(time_slider.value) do val; current_step[] = val; end
    Label(time_slider_grid[1, 1], "Time:")
    Label(time_slider_grid[1, 3], @lift("$(Int($current_step))"))

    iteration_slider_grid = control_grid[2, 1] = GridLayout(tellwidth=false)
    iteration_slider = Slider(iteration_slider_grid[1, 2], range=1:101, startvalue=1)

    on(lift(tuple, iteration_slider.value, num_iterations)) do (slider_val, n_iter)
        if n_iter > 0
            percent = (slider_val - 1) / 100.0
            index = round(Int, 1 + percent * (n_iter - 1))
            current_iteration[] = clamp(index, 1, n_iter)
        else
            current_iteration[] = 1
        end
    end

    Label(iteration_slider_grid[1, 1], "Iteration:")
    Label(iteration_slider_grid[1, 3], @lift("$(show_solver_iterations[] ? ($num_iterations > 0 ? string(Int($current_iteration) - 1) : "N/A") : "Final")"))

    toggle_grid = control_grid[1:2, 2] = GridLayout(tellwidth=false)

    gt_toggle = Toggle(toggle_grid[1, 2], active=true)
    Label(toggle_grid[1, 1], "Ground Truth")
    on(gt_toggle.active) do active; gt_opacity[] = active ? 1.0 : 0.0; end

    belief_toggle = Toggle(toggle_grid[2, 2], active=true)
    Label(toggle_grid[2, 1], "Belief Traj")
    on(belief_toggle.active) do active; belief_opacity[] = active ? 1.0 : 0.0; end

    non_robust_toggle = Toggle(toggle_grid[3, 2], active=true)
    Label(toggle_grid[3, 1], "Non-Robust Plan")
    on(non_robust_toggle.active) do active; non_robust_plan_opacity[] = active ? plan_opacity : 0.0; end

    robust_toggle = Toggle(toggle_grid[4, 2], active=true)
    Label(toggle_grid[4, 1], "Robust Plan")
    on(robust_toggle.active) do active; robust_plan_opacity[] = active ? plan_opacity : 0.0; end

    solver_iter_toggle = Toggle(toggle_grid[5, 2], active=false)
    Label(toggle_grid[5, 1], "Show Iters")
    on(solver_iter_toggle.active) do active; show_solver_iterations[] = active; end

    obs_toggle = Toggle(toggle_grid[6, 2], active=true)
    Label(toggle_grid[6, 1], "Observations")
    on(obs_toggle.active) do active; observation_opacity[] = active ? 1.0 : 0.0; end

    Legend(fig[1, 2], ax, tellheight=false, tellwidth=true)
    
    set_close_to!(time_slider, 1)
    
    display(fig)
    save("exp/hockey/outputs/receding_horizon_$(is_robust ? "robust" : "non_robust").png", fig)
end