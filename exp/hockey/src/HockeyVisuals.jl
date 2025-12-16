function get_position_uncertainty_ellipse(mean_pos, cov, confidence=0.95)
    # Ensure we only use the position components of the covariance
    pos_cov = cov[1:2, 1:2]
    E = eigen(pos_cov)
    scale = sqrt(-2 * log(1 - confidence)) 
    t = range(0, 2π, 100)
    
    # Parametric equation for an ellipse
    return [Point2f(scale * E.vectors * [sqrt(E.values[1])*cos(θ), sqrt(E.values[2])*sin(θ)] + mean_pos[1:2]) for θ in t]
end

function visualize_receding_horizon_solutions_multi_figure(solutions::Dict, goal_position; dims = (; n=2, states=[2, 2], controls=[2, 2], belief=[2, 2, 2, 2], sensor=[2, 2, 2, 2]))
    # Create separate figures for each solution
    figures = Dict{String, Figure}()
    axes = Dict{String, Axis}()
    screens = []
    
    for (sol_name, sol_data) in solutions
        # Create a new screen/window for each figure
        screen = GLMakie.Screen(title="Receding Horizon Hockey Game - $sol_name")
        push!(screens, screen)
        
        fig = Figure(size=(700, 500))
        figures[sol_name] = fig
        
        # --- Top Row: Axis and Legend ---
        ax = Axis(fig[1, 1],
            title="Receding Horizon Hockey Game - $sol_name",
            xlabel="x position",
            ylabel="y position",
        )
        axes[sol_name] = ax
        
        # Create individual visualization for this solution
        create_individual_solution_plot(fig, ax, sol_name, sol_data, goal_position, dims)
        
        # Display each figure in its own window
        display(screen, fig)
    end
    
    return figures
end

function visualize_receding_horizon_solution(solutions::Dict, goal_position; dims = (; n=2, states=[2, 2], controls=[2, 2], belief=[2, 2, 2, 2], sensor=[2, 2, 2, 2]))
    # Call the new multi-figure function
    return visualize_receding_horizon_solutions_multi_figure(solutions, goal_position; dims=dims)
end

function create_individual_solution_plot(fig, ax, sol_name, sol_data, goal_position, dims)
    # Handle both NamedTuple (new) and potentially Tuple (old/other)
    # Handle both NamedTuple (new), Dict (loaded), and potentially Tuple (old/other)
    if sol_data isa NamedTuple
        gt_state_history = sol_data.gt_state_history
        observations = sol_data.observation_history # correct name
        solution_history = sol_data.solution_history
        lq_sol_history = [] # Not present in new
    elseif sol_data isa Dict
        gt_state_history = sol_data["gt_state_history"]
        observations = sol_data["observation_history"]
        solution_history = sol_data["solution_history"]
        lq_sol_history = get(sol_data, "lq_sol_history", [])
    else
        # Fallback for old format if needed
        gt_state_history, observations, solution_history, cond_history, lq_sol_history = sol_data
    end
    
    # --- Bottom Row: Controls ---
    control_grid = fig[2, 1] = GridLayout(tellheight=false)

    current_timestep = Observable(1)
    horizon = length(gt_state_history) - 1
    
    # Colors
    attacker_color = :red
    defender_color = :blue
    nature_color = :green
    lq_sol_color = :cyan

    # Opacities & Visibilities
    plan_opacity = 1.0
    gt_opacity = Observable(0.0)
    belief_opacity = Observable(0.0)
    non_robust_plan_opacity = Observable(plan_opacity)
    robust_plan_opacity = Observable(plan_opacity)
    observation_opacity = Observable(0.0)
    nature_opacity = Observable(0.0)
    show_arrows = Observable(false)
    show_ellipses = Observable(false)

    attacker_belief_opacity = Observable(1.0)
    defender_belief_opacity = Observable(0.1)
    plan_timestep = Observable(1)
    show_lq_sol = Observable(false)

    # Convert solution_history to a time-indexed format if it's a Dict (New)
    # Dict: idx => [step1, step2, ...]
    # We want: [step1_all_players, step2_all_players, ...]
    # step_i_all_players = [step_i_p1, step_i_p2]
    
    formatted_solution_history = if solution_history isa Dict
         len = length(solution_history[1])
         map(1:len) do t
             [solution_history[i][t] for i in 1:length(solution_history)]
         end
    else
         solution_history # Assume old format
    end

    # Belief history for visualization (just first player's belief at each step, first belief in that?)
    # New: formatted_solution_history[t][1].beliefs[1]
    belief_history = [sols[1].beliefs[1] for sols in formatted_solution_history]

    non_robust_plan = @lift begin
         if $current_timestep <= length(formatted_solution_history)
             formatted_solution_history[$current_timestep][1].beliefs # Attacker's plan
         else
             []
         end
    end
    # Robust plan usually mapped to defender? Or checking if p2 is robust?
    # Hockey visualizer assumed p2 result was robust?
    # Let's assume P2 is the robust one we want to visualize if robust
    robust_plan = @lift begin
         if $current_timestep <= length(formatted_solution_history)
             formatted_solution_history[$current_timestep][2].beliefs # Defender's plan
         else
             []
         end
    end

    # --- Static trajectory plotting ---
    attacker_gt_x = [s[Block(1)][1] for s in gt_state_history]
    attacker_gt_y = [s[Block(1)][2] for s in gt_state_history]
    defender_gt_x = [s[Block(2)][1] for s in gt_state_history]
    defender_gt_y = [s[Block(2)][2] for s in gt_state_history]

    lines!(ax, attacker_gt_x, attacker_gt_y, color=attacker_color, linewidth=3, alpha=gt_opacity, label="$sol_name Attacker GT")
    lines!(ax, defender_gt_x, defender_gt_y, color=defender_color, linewidth=3, alpha=gt_opacity, label="$sol_name Defender GT")

    # Attacker's beliefs
    attacker_belief_self_x = [b.beliefs[1].belief_mean[1] for b in belief_history]
    attacker_belief_self_y = [b.beliefs[1].belief_mean[2] for b in belief_history]
    attacker_belief_other_x = [b.beliefs[2].belief_mean[1] for b in belief_history]
    attacker_belief_other_y = [b.beliefs[2].belief_mean[2] for b in belief_history]
    
    # Defender's beliefs
    defender_belief_other_x = [b.beliefs[3].belief_mean[1] for b in belief_history]
    defender_belief_other_y = [b.beliefs[3].belief_mean[2] for b in belief_history]
    defender_belief_self_x = [b.beliefs[4].belief_mean[1] for b in belief_history]
    defender_belief_self_y = [b.beliefs[4].belief_mean[2] for b in belief_history]

    lines!(ax, attacker_belief_self_x, attacker_belief_self_y, color=attacker_color, linewidth=2, label="$sol_name Attacker's Belief (self)", alpha=belief_opacity)
    # lines!(ax, attacker_belief_other_x, attacker_belief_other_y, color=defender_color, linestyle=:dash, linewidth=2, label="$sol_name Attacker's Belief (other)", alpha=belief_opacity)
    lines!(ax, defender_belief_self_x, defender_belief_self_y, color=defender_color, linewidth=2, label="$sol_name Defender's Belief (self)", alpha=belief_opacity)
    # lines!(ax, defender_belief_other_x, defender_belief_other_y, color=attacker_color, linestyle=:dash, linewidth=2, label="$sol_name Defender's Belief (other)", alpha=belief_opacity)
    
    # --- LQ Solution Trajectory ---
    # lq_sol = @lift lq_sol_history[$current_timestep]
    # lq_attacker_x = @lift [s[Block(1)][1] for s in $lq_sol.xs]
    # lq_attacker_y = @lift [s[Block(1)][2] for s in $lq_sol.xs]
    # lq_defender_x = @lift [s[Block(2)][1] for s in $lq_sol.xs]
    # lq_defender_y = @lift [s[Block(2)][2] for s in $lq_sol.xs]

    # lines!(ax, lq_attacker_x, lq_attacker_y, color=lq_sol_color, linewidth=2, linestyle=:dot, label="LQ Attacker", visible=show_lq_sol)
    # lines!(ax, lq_defender_x, lq_defender_y, color=lq_sol_color, linewidth=2, linestyle=:dot, label="LQ Defender", visible=show_lq_sol)

    # --- Goal ---
    lines!(ax, [p[2] for p in goal_position], [p[1] for p in goal_position], color=:green, linewidth=5, label="Goal")
    
    # --- Current belief means and planned trajectory (observables) ---
    current_belief_state = @lift belief_history[$current_timestep]
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
    planned_us_history = [(sols[1][2], sols[2][2]) for sols in formatted_solution_history]
    planned_us = @lift $current_timestep <= length(planned_us_history) ? planned_us_history[$current_timestep] : []
    
    non_robust_planned_us = @lift isempty($planned_us) ? [] : $planned_us[1]
    non_robust_attacker_actions = @lift isempty($non_robust_planned_us) ? Point2f[] : [Point2f(u[Block(1)]) for u in $non_robust_planned_us]
    non_robust_defender_actions = @lift isempty($non_robust_planned_us) ? Point2f[] : [Point2f(u[Block(2)]) for u in $non_robust_planned_us]
    
    arrows!(ax, non_robust_attacker_plan, non_robust_attacker_actions, color=attacker_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($non_robust_plan_opacity > 0.1 && $show_arrows))
    arrows!(ax, non_robust_defender_plan, non_robust_defender_actions, color=defender_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($non_robust_plan_opacity > 0.1 && $show_arrows))

    robust_planned_us = @lift isempty($planned_us) || length($planned_us) < 2 ? [] : $planned_us[2]
    robust_attacker_actions = @lift begin
        if isempty($robust_planned_us)
            Point2f[]
        else
            # Check if the controls have at least 2 blocks before accessing them
            [Point2f(u[Block(1)]) for u in $robust_planned_us if length(u.blocks) >= 1]
        end
    end
    robust_defender_actions = @lift begin
        if isempty($robust_planned_us)
            Point2f[]
        else
            # Check if the controls have at least 2 blocks before accessing them
            [Point2f(u[Block(2)]) for u in $robust_planned_us if length(u.blocks) >= 2]
        end
    end
    
    arrows!(ax, robust_attacker_plan, robust_attacker_actions, color=attacker_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1 && $attacker_belief_opacity > 0.1 && $show_arrows))

    # --- Arrows on attacker's belief of other plan
    arrows!(ax, robust_attacker_plan_other, robust_defender_actions, color=defender_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1 && $attacker_belief_opacity > 0.1 && $show_arrows))
    arrows!(ax, robust_defender_plan, robust_defender_actions, color=defender_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1 && $defender_belief_opacity > 0.1 && $show_arrows))
    arrows!(ax, robust_defender_plan_other, robust_attacker_actions, color=attacker_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1 && $defender_belief_opacity > 0.1 && $show_arrows))

    # --- Nature's actions on robust plan
    nature_actions_on_robust_plan_split = @lift begin
        if $robust_plan_opacity > 0.1 && $current_timestep <= length(formatted_solution_history)
            sols = formatted_solution_history[$current_timestep]

            robust_controls = sols[2][2]
            # Check if nature controls exist (robust scenario with 3 players)
            if !isempty(robust_controls) && length(robust_controls[1].blocks) >= 3
                nature_controls = [u[Block(3)] for u in robust_controls]
                actions_on_attacker = [Point2f(nature_control[1:dims.belief[1]]) for nature_control in nature_controls]
                actions_on_defender = [Point2f(nature_control[dims.belief[1] + 1:end]) for nature_control in nature_controls]
                (actions_on_attacker, actions_on_defender)
            else
                (Point2f[], Point2f[])
            end
        else
            (Point2f[], Point2f[])
        end
    end

    nature_actions_on_attacker_plan = @lift $nature_actions_on_robust_plan_split[1]
    nature_actions_on_defender_plan = @lift $nature_actions_on_robust_plan_split[2]

    nature_start_pos_attacker_plan = @lift begin
        plan = $robust_attacker_plan
        actions = $robust_attacker_actions
        if !isempty(actions) && length(plan) > length(actions)
            plan[1:length(actions)] .+ actions
        else
            Point2f[]
        end
    end
    nature_start_pos_defender_plan = @lift begin
        plan = $robust_attacker_plan_other
        actions = $robust_defender_actions
        if !isempty(actions) && length(plan) > length(actions)
            plan[1:length(actions)] .+ actions
        else
            Point2f[]
        end
    end
    
    arrows!(ax, nature_start_pos_attacker_plan, nature_actions_on_attacker_plan, color=nature_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1 && $nature_opacity > 0.1 && $attacker_belief_opacity > 0.1 && $show_arrows))
    arrows!(ax, nature_start_pos_defender_plan, nature_actions_on_defender_plan, color=nature_color, linewidth=2, arrowsize=10, alpha=0.5, visible=@lift($robust_plan_opacity > 0.1 && $nature_opacity > 0.1 && $attacker_belief_opacity > 0.1 && $show_arrows))

    # --- Plot planned trajectories ---
    lines!(ax, robust_attacker_plan_other, color=attacker_color, linestyle=:dash, linewidth=2, alpha=@lift($robust_plan_opacity * $attacker_belief_opacity), label="Robust Attacker Plan (other)")
    scatter!(ax, robust_attacker_plan_other, color=attacker_color, marker=:xcross, markersize=8, alpha=@lift($robust_plan_opacity * $attacker_belief_opacity))

    lines!(ax, robust_defender_plan_other, color=defender_color, linestyle=:dash, linewidth=2, alpha=@lift($robust_plan_opacity * $defender_belief_opacity), label="Robust Defender Plan (other)")
    scatter!(ax, robust_defender_plan_other, color=defender_color, marker=:cross, markersize=8, alpha=@lift($robust_plan_opacity * $defender_belief_opacity))

    lines!(ax, non_robust_attacker_plan, color=attacker_color, linewidth=3, alpha=non_robust_plan_opacity, label="Non-Robust Attacker Plan")
    scatter!(ax, non_robust_attacker_plan, color=attacker_color, markersize=10, alpha=non_robust_plan_opacity)
    lines!(ax, non_robust_defender_plan, color=defender_color, linewidth=3, alpha=non_robust_plan_opacity, label="Non-Robust Defender Plan")
    scatter!(ax, non_robust_defender_plan, color=defender_color, marker=:xcross, markersize=10, alpha=non_robust_plan_opacity)
    lines!(ax, robust_attacker_plan, color=attacker_color, linestyle=:dash, linewidth=3, alpha=@lift($robust_plan_opacity * $attacker_belief_opacity), label="Robust Attacker Plan (self)")
    scatter!(ax, robust_attacker_plan, color=attacker_color, markersize=10, alpha=@lift($robust_plan_opacity * $attacker_belief_opacity))
    lines!(ax, robust_defender_plan, color=defender_color, linestyle=:dash, linewidth=3, alpha=@lift($robust_plan_opacity * $defender_belief_opacity), label="Robust Defender Plan (self)")
    scatter!(ax, robust_defender_plan, color=defender_color, marker=:xcross, markersize=10, alpha=@lift($robust_plan_opacity * $defender_belief_opacity))

    # --- HIGHLIGHT CURRENT PLAN TIMESTEP ---
    current_non_robust_attacker_pos = @lift if !isempty($non_robust_attacker_plan) && $(plan_timestep) <= length($non_robust_attacker_plan); $non_robust_attacker_plan[$(plan_timestep)]; else; Point2f(NaN, NaN); end
    current_non_robust_defender_pos = @lift if !isempty($non_robust_defender_plan) && $(plan_timestep) <= length($non_robust_defender_plan); $non_robust_defender_plan[$(plan_timestep)]; else; Point2f(NaN, NaN); end
    
    current_robust_attacker_pos = @lift if !isempty($robust_attacker_plan) && $(plan_timestep) <= length($robust_attacker_plan); $robust_attacker_plan[$(plan_timestep)]; else; Point2f(NaN, NaN); end
    current_robust_defender_pos = @lift if !isempty($robust_defender_plan) && $(plan_timestep) <= length($robust_defender_plan); $robust_defender_plan[$(plan_timestep)]; else; Point2f(NaN, NaN); end
    
    current_robust_attacker_pos_other = @lift if !isempty($robust_attacker_plan_other) && $(plan_timestep) <= length($robust_attacker_plan_other); $robust_attacker_plan_other[$(plan_timestep)]; else; Point2f(NaN, NaN); end
    current_robust_defender_pos_other = @lift if !isempty($robust_defender_plan_other) && $(plan_timestep) <= length($robust_defender_plan_other); $robust_defender_plan_other[$(plan_timestep)]; else; Point2f(NaN, NaN); end
    
    scatter!(ax, current_non_robust_attacker_pos, color=attacker_color, markersize=25, marker=:star5, alpha=non_robust_plan_opacity)
    scatter!(ax, current_non_robust_defender_pos, color=defender_color, markersize=25, marker=:star5, alpha=non_robust_plan_opacity)
    
    scatter!(ax, current_robust_attacker_pos, color=attacker_color, markersize=25, marker=:star5, alpha=@lift($robust_plan_opacity * $attacker_belief_opacity))
    scatter!(ax, current_robust_defender_pos, color=defender_color, markersize=25, marker=:star5, alpha=@lift($robust_plan_opacity * $defender_belief_opacity))
    
    scatter!(ax, current_robust_attacker_pos_other, color=attacker_color, markersize=25, marker=:star5, alpha=@lift($robust_plan_opacity * $attacker_belief_opacity))
    scatter!(ax, current_robust_defender_pos_other, color=defender_color, markersize=25, marker=:star5, alpha=@lift($robust_plan_opacity * $defender_belief_opacity))

    # --- Current belief positions (as markers) ---
    scatter!(ax, attacker_pos_self, color=attacker_color, markersize=20)
    scatter!(ax, defender_pos_self, color=defender_color, markersize=20)

    # --- Player Actions ---
    executed_us_history = [(sols[1][2][1], sols[2][2][1]) for sols in formatted_solution_history]
    current_us = @lift executed_us_history[$current_timestep]
    attacker_action = @lift Point2f($current_us[1][Block(1)])
    defender_action = @lift Point2f($current_us[2][Block(2)])
    
    arrows!(ax, @lift([$attacker_pos_self]), @lift([$attacker_action]), color=attacker_color, linewidth=3, arrowsize=15, label="Executed Attacker Action", alpha=0.5, visible=show_arrows)
    arrows!(ax, @lift([$defender_pos_self]), @lift([$defender_action]), color=defender_color, linewidth=3, arrowsize=15, label="Executed Defender Action", alpha=0.5, visible=show_arrows)

    # --- Observations ---
    attacker_obs_x = [obs[1] for obs in observations]
    attacker_obs_y = [obs[2] for obs in observations]
    defender_obs_x = [obs[3] for obs in observations]
    defender_obs_y = [obs[4] for obs in observations]
    scatter!(ax, attacker_obs_x, attacker_obs_y, color=attacker_color, markersize=15, alpha=@lift(0.3 * $observation_opacity), label="Attacker Observations")
    scatter!(ax, defender_obs_x, defender_obs_y, color=defender_color, markersize=15, alpha=@lift(0.3 * $observation_opacity), label="Defender Observations")

    # --- Belief uncertainty ellipses (at planned time) ---
    non_robust_belief_at_plan_time = @lift if !isempty($non_robust_plan) && !isempty($non_robust_plan) && $(plan_timestep) <= length($non_robust_plan); $non_robust_plan[$(plan_timestep)]; else; nothing; end
    robust_belief_at_plan_time = @lift if !isempty($robust_plan) && !isempty($robust_plan) && $(plan_timestep) <= length($robust_plan); $robust_plan[$(plan_timestep)]; else; nothing; end

    # Non-robust ellipses
    nr_attacker_self_ellipse = @lift if !isnothing($non_robust_belief_at_plan_time); get_position_uncertainty_ellipse($non_robust_belief_at_plan_time.beliefs[1].belief_mean[1:2], $non_robust_belief_at_plan_time.beliefs[1].belief_covariance); else; Point2f[]; end
    nr_attacker_other_ellipse = @lift if !isnothing($non_robust_belief_at_plan_time); get_position_uncertainty_ellipse($non_robust_belief_at_plan_time.beliefs[2].belief_mean[1:2], $non_robust_belief_at_plan_time.beliefs[2].belief_covariance); else; Point2f[]; end
    
    poly!(ax, nr_attacker_self_ellipse, color=(attacker_color, 0.2), strokecolor=(attacker_color, 0.2), visible=@lift($non_robust_plan_opacity > 0.1 && $show_ellipses))
    poly!(ax, nr_attacker_other_ellipse, color=(defender_color, 0.2), strokecolor=(defender_color, 0.2), visible=@lift($non_robust_plan_opacity > 0.1 && $show_ellipses))

    # Robust ellipses
    r_attacker_self_ellipse = @lift if !isnothing($robust_belief_at_plan_time); get_position_uncertainty_ellipse($robust_belief_at_plan_time.beliefs[1].belief_mean[1:2], $robust_belief_at_plan_time.beliefs[1].belief_covariance); else; Point2f[]; end
    r_attacker_other_ellipse = @lift if !isnothing($robust_belief_at_plan_time); get_position_uncertainty_ellipse($robust_belief_at_plan_time.beliefs[2].belief_mean[1:2], $robust_belief_at_plan_time.beliefs[2].belief_covariance); else; Point2f[]; end
    r_defender_other_ellipse = @lift if !isnothing($robust_belief_at_plan_time); get_position_uncertainty_ellipse($robust_belief_at_plan_time.beliefs[3].belief_mean[1:2], $robust_belief_at_plan_time.beliefs[3].belief_covariance); else; Point2f[]; end
    r_defender_self_ellipse = @lift if !isnothing($robust_belief_at_plan_time); get_position_uncertainty_ellipse($robust_belief_at_plan_time.beliefs[4].belief_mean[1:2], $robust_belief_at_plan_time.beliefs[4].belief_covariance); else; Point2f[]; end

    poly!(ax, r_attacker_self_ellipse, color=(attacker_color, 0.2), strokecolor=(attacker_color, 0.2), visible=@lift($robust_plan_opacity > 0.1 && $attacker_belief_opacity > 0.1 && $show_ellipses))
    poly!(ax, r_attacker_other_ellipse, color=(defender_color, 0.2), strokecolor=(defender_color, 0.2), visible=@lift($robust_plan_opacity > 0.1 && $attacker_belief_opacity > 0.1 && $show_ellipses))
    poly!(ax, r_defender_other_ellipse, color=(attacker_color, 0.2), strokecolor=(attacker_color, 0.2), visible=@lift($robust_plan_opacity > 0.1 && $defender_belief_opacity > 0.1 && $show_ellipses))
    poly!(ax, r_defender_self_ellipse, color=(defender_color, 0.2), strokecolor=(defender_color, 0.2), visible=@lift($robust_plan_opacity > 0.1 && $defender_belief_opacity > 0.1 && $show_ellipses))
    
    # --- Controls ---
    # Sliders on the left
    slider_grid = control_grid[1, 1] = GridLayout(tellwidth=false)

    time_slider_grid = slider_grid[1, 1] = GridLayout(tellwidth=false)
    time_slider = Slider(time_slider_grid[1, 2], range=1:horizon, startvalue=1)
    on(time_slider.value) do val; current_timestep[] = val; end
    Label(time_slider_grid[1, 1], "Time:")
    Label(time_slider_grid[1, 3], @lift("$(Int($current_timestep))"))

    # iteration_slider_grid = slider_grid[2, 1] = GridLayout(tellwidth=false)
    # iteration_slider = Slider(iteration_slider_grid[1, 2], range=0:1, startvalue=1)
    # on(iteration_slider.value) do val; current_iteration[] = val; end
    # Label(iteration_slider_grid[1, 1], "Iteration:")
    # Label(iteration_slider_grid[1, 3], @lift("$(show_solver_iterations[] ? ($num_iterations > 0 ? string(round($current_iteration, digits=2)) : "N/A") : "Final")"))

    plan_time_slider_grid = slider_grid[3, 1] = GridLayout(tellwidth=false)
    plan_length = @lift begin
        nrp = $non_robust_plan
        if !isempty(nrp)
            length(nrp)
        else
            1
        end
    end
    plan_time_slider = Slider(plan_time_slider_grid[1, 2], range=@lift(1:$plan_length), startvalue=1)
    on(plan_time_slider.value) do val; plan_timestep[] = val; end
    Label(plan_time_slider_grid[1, 1], "Plan Time:")
    Label(plan_time_slider_grid[1, 3], @lift("$(Int($plan_timestep))"))

    # Buttons and Toggles on the right, in two columns
    right_controls = control_grid[1, 2] = GridLayout(tellwidth=false)

    belief_focus_grid = right_controls[1, 1:2] = GridLayout(tellwidth=false)
    focus_attacker_btn = Button(belief_focus_grid[1, 1], label="Focus Attacker Beliefs")
    focus_defender_btn = Button(belief_focus_grid[1, 2], label="Focus Defender Beliefs")
    show_all_beliefs_btn = Button(belief_focus_grid[1, 3], label="Show All Beliefs")
    on(focus_attacker_btn.clicks) do n; attacker_belief_opacity[] = 1.0; defender_belief_opacity[] = 0.1; end
    on(focus_defender_btn.clicks) do n; attacker_belief_opacity[] = 0.1; defender_belief_opacity[] = 1.0; end
    on(show_all_beliefs_btn.clicks) do n; attacker_belief_opacity[] = 1.0; defender_belief_opacity[] = 1.0; end

    toggle_grid = right_controls[2, 1:2] = GridLayout(tellwidth=false)
    
    toggles = [
        ("Ground Truth", gt_opacity, false, 1.0, 0.0),
        ("Belief Traj", belief_opacity, false, 1.0, 0.0),
        ("Non-Robust Plan", non_robust_plan_opacity, true, plan_opacity, 0.0),
        ("Robust Plan", robust_plan_opacity, true, plan_opacity, 0.0),
        # ("Show Iters", show_solver_iterations, false, true, false),
        ("Observations", observation_opacity, false, 1.0, 0.0),
        ("Nature", nature_opacity, false, 1.0, 0.0),
        ("Show Arrows", show_arrows, false, true, false),
        ("LQ Solution", show_lq_sol, false, true, false),
        ("Uncertainty Ellipses", show_ellipses, false, true, false)
    ]

    for (i, (label, obs, is_active, active_val, inactive_val)) in enumerate(toggles)
        row = (i + 1) ÷ 2
        col = (i % 2 == 1) ? 1 : 3
        
        Label(toggle_grid[row, col], label)
        toggle = Toggle(toggle_grid[row, col + 1], active=is_active)
        on(toggle.active) do active
            obs[] = active ? active_val : inactive_val
        end
    end

    # Legend(fig[1, 2], ax, tellheight=false, tellwidth=true)
    
    set_close_to!(time_slider, 1)
end

function load_hockey_results(results_dir::String; visualize::Bool=false, goal_position=[[-1.5, 0.25], [-1.5, -0.25]])
    # Regex to match: hockey_robust_Def_cc_0p01_trial_1.jld2 (case insensitive for Def)
    regex = r"hockey_robust_(?:def|Def)_cc_(\d+p\d+)_trial_(\d+)\.jld2"
    
    results = Dict{String, Any}()
    
    if !isdir(results_dir)
        @warn "Directory not found: $results_dir"
        return results
    end
    
    for file in readdir(results_dir)
        m = match(regex, file)
        if !isnothing(m)
            cc_str = m.captures[1]
            trial_str = m.captures[2]
            
            # Parse cc value: 0p01 -> 0.01
            cc_val = tryparse(Float64, replace(cc_str, "p" => "."))
            trial_val = tryparse(Int, trial_str)
            
            if !isnothing(cc_val) && !isnothing(trial_val)
                path = joinpath(results_dir, file)
                try
                    key = "CC: $(cc_val), Trial: $(trial_val)"
                    results[key] = load(path)
                catch e
                    @warn "Failed to load $file: $e"
                end
            end
        end
    end
    
    if visualize
        visualize_receding_horizon_solutions_multi_figure(results, goal_position)
    end
    
    return results
end