module SenateVisuals

using GLMakie
using LinearAlgebra
using RobustBeliefGame
using BlockArrays
using Serialization
using Infiltrator

const senate_module_path = joinpath(@__DIR__, "Senate.jl")
if !isdefined(Main, :Senate)
    @eval Main begin
        include($senate_module_path)
    end
end
const Senate = Main.Senate

# Register UUID for deserialization at module load time
const senate_uuid = Base.UUID("a2515029-de12-424a-9371-454911e3b6f1")
const senate_pkgid = Base.PkgId(senate_uuid, "Senate")
if !haskey(Base.loaded_modules, senate_pkgid)
    Base.loaded_modules[senate_pkgid] = Senate
end

export visualize_receding_horizon_solution, load_solution

function load_solution(folder, filename, type = "mass_results")
    
    
    path = "exp/senate/outputs/$folder/$(filename)_$type.dat"
    results = open(deserialize, path, "r")
    experiments = Dict{String, Dict{String, Tuple{Dict, Dict, Main.Senate.SenateParams}}}()
    for exp_data in results
        params, fixed, trial_dict, exp_name = exp_data
        if filename == SubString(exp_name,1,length(filename))
            exp_name = SubString(exp_name,length(filename)+1,length(exp_name)) #Shorten to only parameters
        end
        experiments[exp_name] = Dict{String, Tuple{Dict, Dict, Main.Senate.SenateParams}}()
        for (trial_id, trial_data) in trial_dict
            solutions, games, cost_params = trial_data # ,Dict {player_idx -> RBG.BeliefGame}, SenateParams
            experiments[exp_name][trial_id] = (solutions, games, cost_params)
        end
    end
    visualize_receding_horizon_solution(experiments, filename)
end

function plot_ellipse!(ax, center, a, b; n=100, label="", color=:black)
    t = range(0, 2*pi, length=n)
    x = center[1] .+ a .* cos.(t)
    y = center[2] .+ b .* sin.(t)
    lines!(ax, x, y, label=label, color=color)
end

function get_ellipse_points(center, cov; n=50, conf=1.0)
    eig_decomp = eigen(cov)
    λ = eig_decomp.values
    v = eig_decomp.vectors
    a = conf * sqrt(λ[2]) # Major axis
    b = conf * sqrt(λ[1]) # Minor axis
    θ = atan(v[2, 2], v[1, 2]) # Angle of major axis eigenvector

    t_range = range(0, 2*pi, length=n)
    
    pts = Point2f[]
    for t_ellipse in t_range
        x_r = a * cos(t_ellipse)
        y_r = b * sin(t_ellipse)
        x = center[1] + x_r * cos(θ) - y_r * sin(θ)
        y = center[2] + x_r * sin(θ) + y_r * cos(θ)
        push!(pts, Point2f(x, y))
    end
    # close the ellipse
    push!(pts, pts[1])
    return pts
end

function visualize_receding_horizon_solution(experiments::Dict, filename::String = "Test")
    screen = GLMakie.Screen()
    fig = Figure()
    ax = Axis(fig[1, 1:2], title="$filename graph", xlabel="Opinion Dimension 1", ylabel="Opinion Dimension 2", aspect=DataAspect())
    create_individual_solution_plot(fig, ax, experiments)
    display(screen, fig)
end

function create_individual_solution_plot(fig, ax, experiments::Dict)# sol_data, dims, cost_params)
    selection_state = Dict{String, Tuple{Observable{Bool}, Dict{String, Observable{Bool}}}}()    
    for exp_name in keys(experiments)
        trial_selections = Dict{String, Observable{Bool}}()
        for trial_id in keys(experiments[exp_name])
            trial_selections[trial_id] = Observable(false)
        end
        selection_state[exp_name] = (Observable(false), trial_selections)
    end

    # Set first experiment/trial selected to true
    
    # Create lifted dictionaries that filter by selection booleans
    # Create combined observables for each (exp_name, trial_id) pair
    all_observables = Observable[]
    for (exp_name, (exp_obs, trial_obs_dict)) in selection_state
        push!(all_observables, exp_obs)
        for trial_obs in values(trial_obs_dict)
            push!(all_observables, trial_obs)
        end
    end

    # Then lift on all of them
    exp_trial_pairs = lift(all_observables...) do _...
        [
            (exp_name, trial_id)
            for exp_name in keys(experiments)
            for trial_id in keys(experiments[exp_name])
            if selection_state[exp_name][1][] && selection_state[exp_name][2][trial_id][]
        ]
    end
    # #Toggle the first experiment on afterwards (trigger callback)
    selection_state[first(keys(experiments))][1][] = true
    selection_state[first(keys(experiments))][2][first(keys(experiments[first(keys(experiments))]))][] = true

    solutions = @lift Dict(
        (exp_name, trial_id) => experiments[exp_name][trial_id][1]
        for (exp_name, trial_id) in $exp_trial_pairs
    )

    games = @lift Dict(
        (exp_name, trial_id) => experiments[exp_name][trial_id][2]
        for (exp_name, trial_id) in $exp_trial_pairs
    )

    cost_params = @lift Dict(
        (exp_name, trial_id) => experiments[exp_name][trial_id][3]
        for (exp_name, trial_id) in $exp_trial_pairs
    )
    # Helper observables for accessing player data across all selected trials
    p1_sols = @lift Dict(k => $solutions[k][1] for k in keys($solutions))
    p2_sols = @lift Dict(k => $solutions[k][2] for k in keys($solutions))
    dims_dict = @lift Dict(k => Senate.dims($cost_params[k]) for k in keys($cost_params))
    params_dict = @lift Dict(k => $cost_params[k] for k in keys($cost_params))
    
    # Helper observables for solution history and costs
    p1_solution_history_dict = @lift Dict(k => $p1_sols[k].solution_history for k in keys($p1_sols))
    p2_solution_history_dict = @lift Dict(k => $p2_sols[k].solution_history for k in keys($p2_sols))
    p1_costs_dict = @lift Dict(k => $p1_sols[k].cost_history for k in keys($p1_sols))
    p2_costs_dict = @lift Dict(k => $p2_sols[k].cost_history for k in keys($p2_sols))
    
    # --- Selection Panel ---
    selection_panel = fig[2, 1] = GridLayout(tellwidth=false)
    colsize!(fig.layout, 1, Relative(0.15))
    
    # Add title
    Label(selection_panel[1, 1:5], "Selection", fontsize=14, font=:bold, halign=:left)
    
    # Create nested toggles for experiments and trials
    row_idx = 2  # Start after title
    first_trial_placed = false
    for (exp_name, (exp_obs, trial_obs_dict)) in selection_state
        # Experiment-level toggle
        exp_toggle = Toggle(selection_panel[row_idx, 1], active=exp_obs[])
        on(exp_toggle.active) do active; exp_obs[] = active; end
        Label(selection_panel[row_idx, 3], exp_name, fontsize=12)  # Col 2 provides spacing
        row_idx += 1
        
        # Trial-level toggles (nested under experiment)
        # Column layout: col 2 = indentation spacer, col 3 = toggle, col 4 = spacing, col 5 = label
        for (trial_id, trial_obs) in trial_obs_dict
            # Place spacer in column 2 for indentation
            Label(selection_panel[row_idx, 2], "")  # Empty label as spacer
            trial_toggle = Toggle(selection_panel[row_idx, 3], active=trial_obs[])
            on(trial_toggle.active) do active; trial_obs[] = active; end
            Label(selection_panel[row_idx, 5], trial_id, fontsize=10)  # Col 4 is spacing
            # Set column widths for proper spacing after first trial row is complete
            if !first_trial_placed
                colsize!(selection_panel, 2, Fixed(20))  # Indentation spacer for trials (also spacing for experiments)
                colsize!(selection_panel, 4, Fixed(10))   # Spacing between toggle and label
                first_trial_placed = true
            end
            row_idx += 1
        end
    end
    
    # --- Layout ---
    controls_grid = fig[2, 2] = GridLayout(tellwidth=false)
    colsize!(fig.layout, 2, Relative(0.60))
    current_time_step = Observable(1)
    plan_time_step = Observable(1)
    show_p1_planned_trajectory = Observable(true)
    show_p2_planned_trajectory = Observable(true)
    show_executed_trajectory = Observable(true)
    colors = [:blue, :red]

    # --- Sliders ---
    time_slider_grid = controls_grid[1, 1] = GridLayout(tellwidth=false)
    Label(time_slider_grid[1, 1], "Receding Horizon Time")
    time_steps = @lift isempty($p1_solution_history_dict) ? 1 : minimum([length(val) for val in values($p1_solution_history_dict)])
    slider_range = @lift 1:max(1, $time_steps)
    slider = Slider(time_slider_grid[2, 1], range = slider_range, startvalue = 1)
    on(slider.value) do val; current_time_step[] = val; end
    Label(time_slider_grid[3, 1], @lift(string(Int($current_time_step))))

    p1_planned_belief_trajectory = @lift begin
        result = Dict()
        for k in $exp_trial_pairs
            if haskey($p1_solution_history_dict, k)
                result[k] = $p1_solution_history_dict[k][$current_time_step].beliefs
            end
        end
        result
    end
    p2_planned_belief_trajectory = @lift begin
        result = Dict()
        for k in $exp_trial_pairs
            if haskey($p2_solution_history_dict, k)
                result[k] = $p2_solution_history_dict[k][$current_time_step].beliefs
            end
        end
        result
    end
    p1_planned_controls = @lift begin
        result = Dict()
        for k in $exp_trial_pairs
            if haskey($p1_solution_history_dict, k)
                result[k] = $p1_solution_history_dict[k][$current_time_step].controls
            end
        end
        result
    end
    p2_planned_controls = @lift begin
        result = Dict()
        for k in $exp_trial_pairs
            if haskey($p2_solution_history_dict, k)
                result[k] = $p2_solution_history_dict[k][$current_time_step].controls
            end
        end
        result
    end
    # Observables for planned trajectories at each receding horizon step `t` (dict keyed by trial)
    p1_means_trajectory = @lift begin
        result = Dict()
        for k in $exp_trial_pairs
            if haskey($p1_planned_belief_trajectory, k)
                result[k] = [means(b) for b in $p1_planned_belief_trajectory[k]]
            end
        end
        result
    end
    p2_means_trajectory = @lift begin
        result = Dict()
        for k in $exp_trial_pairs
            if haskey($p2_planned_belief_trajectory, k)
                result[k] = [means(b) for b in $p2_planned_belief_trajectory[k]]
            end
        end
        result
    end
    p1_covariances_trajectory = @lift begin
        result = Dict()
        for k in $exp_trial_pairs
            if haskey($p1_planned_belief_trajectory, k)
                result[k] = [covs(b) for b in $p1_planned_belief_trajectory[k]]
            end
        end
        result
    end
    p2_covariances_trajectory = @lift begin
        result = Dict()
        for k in $exp_trial_pairs
            if haskey($p2_planned_belief_trajectory, k)
                result[k] = [covs(b) for b in $p2_planned_belief_trajectory[k]]
            end
        end
        result
    end
    # Use first trial for planning horizon (they should all be the same)
    p1_planning_horizon = @lift isempty($p1_means_trajectory) ? 0 : minimum(length.(values($p1_means_trajectory)))
    p2_planning_horizon = @lift isempty($p2_means_trajectory) ? 0 : minimum(length.(values($p2_means_trajectory)))

    plan_slider_grid = controls_grid[2, 1] = GridLayout(tellwidth=false)
    Label(plan_slider_grid[1, 1], "Plan Time")
    plan_slider_range = @lift(1:($p1_planning_horizon > 0 ? $p1_planning_horizon : 1))
    plan_slider = Slider(plan_slider_grid[2, 1], range = plan_slider_range, startvalue = 1)
    on(plan_slider.value) do val; plan_time_step[] = val; end
    Label(plan_slider_grid[3, 1], @lift(string(Int($plan_time_step))))

    # Plot activist preferences (use first trial's params, they should all be the same)
    # Plot activist preference ellipses for each trial independently, using on() callback to manage plot elements
    preference_ellipse_plots = Dict{Tuple{String, String}, Vector{Any}}()
    
    on(exp_trial_pairs) do pairs
        # Clear all previous preference ellipse plots
        for plots in values(preference_ellipse_plots)
            for plot in plots
                delete!(ax, plot)
            end
        end
        empty!(preference_ellipse_plots)
        
        # Create preference ellipses for currently selected trials
        for trial_key in pairs
            if !haskey(params_dict[], trial_key)
                @warn "preference_ellipses: params_dict[] missing key: $trial_key. Available keys: $(keys(params_dict[]))"
                continue
            end
            if !haskey(dims_dict[], trial_key)
                @warn "preference_ellipses: dims_dict[] missing key: $trial_key. Available keys: $(keys(dims_dict[]))"
                continue
            end
            trial_params = params_dict[][trial_key]
            trial_dims = dims_dict[][trial_key]
            ellipse_list = Any[]
            for activist_id in 1:trial_dims.num_activists
                params = trial_params.player_configs[activist_id]
                center = params.ellipsoid_centers[1]
                scale = params.ellipsoid_radii[1]
                a = sqrt(1 / scale[1])
                b = sqrt(1 / scale[2])
                plot = plot_ellipse!(ax, center, a, b, label="Activist $activist_id Pref. (Trial $trial_key)", color=colors[activist_id])
                push!(ellipse_list, plot)
            end
            preference_ellipse_plots[trial_key] = ellipse_list
        end
    end

    min_num_senators = @lift isempty($dims_dict) ? 0 : minimum(d.num_senators for d in values($dims_dict))
    min_num_activists = @lift isempty($dims_dict) ? 0 : minimum(d.num_activists for d in values($dims_dict))
    point_colors = @lift vcat([fill(c, $min_num_senators) for c in colors]...)

    #executed_trajectory holds the solved trajectory for each time_step for each senator, based on each activist (we only care about gt first state, which is repeated twice)
    # Use on() callback to manage plot elements per trial
    executed_trajectory_plots = Dict{Tuple{String, String}, Vector{Tuple{Any, Any, Observable{Vector{Point2f}}}}}()  # (scatter_plot, line_plot, trajectory_obs)
    
    on(exp_trial_pairs) do pairs
        # Clear all previous executed trajectory plots
        for plot_list in values(executed_trajectory_plots)
            for (scatter_plot, line_plot, _) in plot_list
                delete!(ax, scatter_plot)
                delete!(ax, line_plot)
            end
        end
        empty!(executed_trajectory_plots)
        
        # Create plots and Observables for currently selected trials
        for (exp_name, trial_id) in pairs
            trial_key = (exp_name, trial_id)
            plot_list = Tuple{Any, Any, Observable{Vector{Point2f}}}[]
            
            for senator_id in 1:min_num_senators[]
                trajectory_obs = Observable(Point2f[])
                scatter_plot = scatter!(ax, trajectory_obs, color=:green, markersize=8, visible=show_executed_trajectory)
                line_plot = lines!(ax, trajectory_obs, color=:green, visible=show_executed_trajectory)
                push!(plot_list, (scatter_plot, line_plot, trajectory_obs))
            end
            
            executed_trajectory_plots[trial_key] = plot_list
        end
    end
    
    # Update executed trajectory Observables when time step changes
    on(current_time_step) do t
        for (trial_key, plot_list) in executed_trajectory_plots
            if haskey(p1_solution_history_dict[], trial_key)
                solution_history = p1_solution_history_dict[][trial_key]
                for (senator_idx, (_, _, trajectory_obs)) in enumerate(plot_list)
                    if senator_idx <= length(solution_history) && t <= length(solution_history)
                        senator_states = [solution_history[time][1][1].beliefs[senator_idx].belief_mean for time in 1:t]
                        gt_trajectory = [Point2f(state[1], state[2]) for state in senator_states]
                        trajectory_obs[] = gt_trajectory
                    end
                end
            end
        end
    end
    # Plot full planned trajectories as lines - use on() callback to manage plot elements per trial
    # Store as Dict{trial_key => Dict{(activist_id, senator_id) => (plot_handle, trajectory_obs)}}
    p1_planned_trajectory_plots = Dict{Tuple{String, String}, Dict{Tuple{Int, Int}, Tuple{Any, Observable{Vector{Point2f}}}}}()
    p2_planned_trajectory_plots = Dict{Tuple{String, String}, Dict{Tuple{Int, Int}, Tuple{Any, Observable{Vector{Point2f}}}}}()
    
    # Function to update trajectory plots
    function update_trajectory_plots(plan_t, curr_t, p1_horizon, p2_horizon, pairs)
        # println("=== DEBUG update_trajectory_plots START ===")
        # println("  plan_t=$plan_t, curr_t=$curr_t, p1_horizon=$p1_horizon, p2_horizon=$p2_horizon")
        # println("  pairs=$(pairs)")
        # println("  p1_means_trajectory[] keys: $(keys(p1_means_trajectory[]))")
        # println("  p2_means_trajectory[] keys: $(keys(p2_means_trajectory[]))")
        # println("  p1_planned_trajectory_plots keys: $(keys(p1_planned_trajectory_plots))")
        # println("  p2_planned_trajectory_plots keys: $(keys(p2_planned_trajectory_plots))")
        
        # Create plots lazily if they don't exist
        for (exp_name, trial_id) in pairs
            trial_key = (exp_name, trial_id)
            
            # Create P1 plots if they don't exist
            if !haskey(p1_planned_trajectory_plots, trial_key)
                # println("  Creating P1 plots lazily for trial: $trial_key")
                p1_dict = Dict{Tuple{Int, Int}, Tuple{Any, Observable{Vector{Point2f}}}}()
                for activist_id in 1:min_num_activists[]
                    for senator_id in 1:min_num_senators[]
                        key = (activist_id, senator_id)
                        p1_obs = Observable(Point2f[])
                        p1_plot = lines!(ax, p1_obs, color=colors[activist_id], visible=show_p1_planned_trajectory)
                        p1_dict[key] = (p1_plot, p1_obs)
                    end
                end
                p1_planned_trajectory_plots[trial_key] = p1_dict
                # println("    Created P1 plots_dict with $(length(p1_dict)) entries")
            end
            
            # Create P2 plots if they don't exist
            if !haskey(p2_planned_trajectory_plots, trial_key)
                # println("  Creating P2 plots lazily for trial: $trial_key")
                p2_dict = Dict{Tuple{Int, Int}, Tuple{Any, Observable{Vector{Point2f}}}}()
                for activist_id in 1:min_num_activists[]
                    for senator_id in 1:min_num_senators[]
                        key = (activist_id, senator_id)
                        p2_obs = Observable(Point2f[])
                        p2_plot = lines!(ax, p2_obs, color=colors[activist_id], visible=show_p2_planned_trajectory, linestyle=:dash)
                        p2_dict[key] = (p2_plot, p2_obs)
                    end
                end
                p2_planned_trajectory_plots[trial_key] = p2_dict
                # println("    Created P2 plots_dict with $(length(p2_dict)) entries")
            end
        end
        
        p1_update_count = 0
        p2_update_count = 0
        
        for (exp_name, trial_id) in pairs
            trial_key = (exp_name, trial_id)
            # println("  Checking P1 trial: $trial_key")
            plots_dict = p1_planned_trajectory_plots[trial_key]
            # println("    Found plots_dict with $(length(plots_dict)) entries")
            if haskey(p1_means_trajectory[], trial_key)
                means_traj_p1 = p1_means_trajectory[][trial_key]
                # println("    Found means_traj_p1: length=$(length(means_traj_p1)), p1_horizon=$p1_horizon, plan_t=$plan_t")
                # println("    Conditions check: !isempty=$(!isempty(means_traj_p1)), p1_horizon>0=$(p1_horizon > 0), plan_t<=length=$(plan_t <= length(means_traj_p1))")
                if !isempty(means_traj_p1) && p1_horizon > 0 && plan_t <= length(means_traj_p1)
                    # println("    Iterating over $(length(plots_dict)) plots in plots_dict")
                    for ((activist_id, senator_id), (_, trajectory_obs)) in plots_dict
                        belief_idx_local = (activist_id-1)*min_num_senators[] + senator_id
                        # println("      Processing (activist=$activist_id, senator=$senator_id) -> belief_idx=$belief_idx_local")
                        # println("        blocks length=$(length(means_traj_p1[plan_t].blocks))")
                        if belief_idx_local <= length(means_traj_p1[plan_t].blocks)
                            p1_pts = [Point2f(means_traj_p1[time][Block(belief_idx_local)][1], means_traj_p1[time][Block(belief_idx_local)][2]) for time in 1:p1_horizon]
                            # println("  DEBUG: Setting P1 trajectory_obs[] for ($activist_id, $senator_id) with $(length(p1_pts)) points")
                            trajectory_obs[] = p1_pts
                            p1_update_count += 1
                        else
                            # println("        SKIP: belief_idx_local=$belief_idx_local > length(blocks)=$(length(means_traj_p1[plan_t].blocks))")
                        end
                    end
                else
                    # println("    SKIP: Conditions not met")
                end
            else
                # println("    SKIP: !haskey(p1_means_trajectory[], $trial_key)")
            end
        end
        
        for (exp_name, trial_id) in pairs
            trial_key = (exp_name, trial_id)
            # println("  Checking P2 trial: $trial_key")
            plots_dict = p2_planned_trajectory_plots[trial_key]
            # println("    Found plots_dict with $(length(plots_dict)) entries")
            if haskey(p2_means_trajectory[], trial_key)
                means_traj_p2 = p2_means_trajectory[][trial_key]
                # println("    Found means_traj_p2: length=$(length(means_traj_p2)), p2_horizon=$p2_horizon, plan_t=$plan_t")
                # println("    Conditions check: !isempty=$(!isempty(means_traj_p2)), p2_horizon>0=$(p2_horizon > 0), plan_t<=length=$(plan_t <= length(means_traj_p2))")
                if !isempty(means_traj_p2) && p2_horizon > 0 && plan_t <= length(means_traj_p2)
                    # println("    Iterating over $(length(plots_dict)) plots in plots_dict")
                    for ((activist_id, senator_id), (_, trajectory_obs)) in plots_dict
                        belief_idx_local = (activist_id-1)*min_num_senators[] + senator_id
                        # println("      Processing (activist=$activist_id, senator=$senator_id) -> belief_idx=$belief_idx_local")
                        # println("        blocks length=$(length(means_traj_p2[plan_t].blocks))")
                        if belief_idx_local <= length(means_traj_p2[plan_t].blocks)
                            p2_pts = [Point2f(means_traj_p2[time][Block(belief_idx_local)][1], means_traj_p2[time][Block(belief_idx_local)][2]) for time in 1:p2_horizon]
                            # println("  DEBUG: Setting P2 trajectory_obs[] for ($activist_id, $senator_id) with $(length(p2_pts)) points")
                            trajectory_obs[] = p2_pts
                            p2_update_count += 1
                        else
                            # println("        SKIP: belief_idx_local=$belief_idx_local > length(blocks)=$(length(means_traj_p2[plan_t].blocks))")
                        end
                    end
                else
                    # println("    SKIP: Conditions not met")
                end
            else
                # println("    SKIP: !haskey(p2_means_trajectory[], $trial_key)")
            end
        end
        
        # println("  DEBUG: Updated $p1_update_count P1 trajectories and $p2_update_count P2 trajectories")
        # println("=== DEBUG update_trajectory_plots END ===\n")
    end
    
    on(exp_trial_pairs) do pairs
        # println("=== DEBUG on(exp_trial_pairs) START ===")
        # println("  pairs=$(pairs)")
        # println("  BEFORE clear: p1_planned_trajectory_plots keys: $(keys(p1_planned_trajectory_plots))")
        # println("  BEFORE clear: p2_planned_trajectory_plots keys: $(keys(p2_planned_trajectory_plots))")
        
        # Clear all previous planned trajectory plots
        for plots_dict in values(p1_planned_trajectory_plots)
            for (plot_handle, _) in values(plots_dict)
                delete!(ax, plot_handle)
            end
        end
        for plots_dict in values(p2_planned_trajectory_plots)
            for (plot_handle, _) in values(plots_dict)
                delete!(ax, plot_handle)
            end
        end
        empty!(p1_planned_trajectory_plots)
        empty!(p2_planned_trajectory_plots)
        
        # println("  AFTER clear: p1_planned_trajectory_plots keys: $(keys(p1_planned_trajectory_plots))")
        # println("  AFTER clear: p2_planned_trajectory_plots keys: $(keys(p2_planned_trajectory_plots))")
        # println("  min_num_activists[]=$(min_num_activists[]), min_num_senators[]=$(min_num_senators[])")
        
        # Create plots and Observables for currently selected trials
        for (exp_name, trial_id) in pairs
            trial_key = (exp_name, trial_id)
            # println("  Creating plots for trial_key: $trial_key")
            p1_dict = Dict{Tuple{Int, Int}, Tuple{Any, Observable{Vector{Point2f}}}}()
            p2_dict = Dict{Tuple{Int, Int}, Tuple{Any, Observable{Vector{Point2f}}}}()
            
            for activist_id in 1:min_num_activists[]
                for senator_id in 1:min_num_senators[]
                    belief_idx_local = (activist_id-1)*min_num_senators[] + senator_id
                    key = (activist_id, senator_id)
                    
                    p1_obs = Observable(Point2f[])
                    p1_plot = lines!(ax, p1_obs, color=colors[activist_id], visible=show_p1_planned_trajectory)
                    p1_dict[key] = (p1_plot, p1_obs)
                    
                    p2_obs = Observable(Point2f[])
                    p2_plot = lines!(ax, p2_obs, color=colors[activist_id], visible=show_p2_planned_trajectory, linestyle=:dash)
                    p2_dict[key] = (p2_plot, p2_obs)
                end
            end
            
            # println("  p1_dict has $(length(p1_dict)) entries, p2_dict has $(length(p2_dict)) entries")
            p1_planned_trajectory_plots[trial_key] = p1_dict
            p2_planned_trajectory_plots[trial_key] = p2_dict
            # println("  AFTER assignment: p1_planned_trajectory_plots keys: $(keys(p1_planned_trajectory_plots))")
            # println("  AFTER assignment: p2_planned_trajectory_plots keys: $(keys(p2_planned_trajectory_plots))")
        end
        
        # println("  FINAL: p1_planned_trajectory_plots keys: $(keys(p1_planned_trajectory_plots))")
        # println("  FINAL: p2_planned_trajectory_plots keys: $(keys(p2_planned_trajectory_plots))")
        # println("  Calling update_trajectory_plots from on(exp_trial_pairs)")
        # Manually trigger updates after populating dictionaries
        update_trajectory_plots(plan_time_step[], current_time_step[], p1_planning_horizon[], p2_planning_horizon[], pairs)
        # println("=== DEBUG on(exp_trial_pairs) END ===\n")
    end
    
    # Update planned trajectory Observables when time steps change (but not when exp_trial_pairs changes)
    onany(plan_time_step, current_time_step, p1_planning_horizon, p2_planning_horizon) do plan_t, curr_t, p1_horizon, p2_horizon
        # println("=== DEBUG onany(plan_time_step, ...) FIRED ===")
        # println("  plan_t=$plan_t, curr_t=$curr_t, p1_horizon=$p1_horizon, p2_horizon=$p2_horizon")
        # println("  exp_trial_pairs[]=$(exp_trial_pairs[])")
        # println("  p1_planned_trajectory_plots keys: $(keys(p1_planned_trajectory_plots))")
        # println("  p2_planned_trajectory_plots keys: $(keys(p2_planned_trajectory_plots))")
        # println("  Calling update_trajectory_plots from onany")
        update_trajectory_plots(plan_t, curr_t, p1_horizon, p2_horizon, exp_trial_pairs[])
        # println("=== DEBUG onany(plan_time_step, ...) END ===\n")
    end

    # --- Toggles ---
    toggles_grid = controls_grid[1, 2] = GridLayout(tellwidth=false)
    
    show_p1_activist_controls = Observable(true)
    p1_activist_toggle = Toggle(toggles_grid[1, 1], active=true)
    on(p1_activist_toggle.active) do active; show_p1_activist_controls[] = active; end
    Label(toggles_grid[1, 2], "Show Player 1 Activist Controls")

    show_p2_activist_controls = Observable(true)
    p2_activist_toggle = Toggle(toggles_grid[2, 1], active=true)
    on(p2_activist_toggle.active) do active; show_p2_activist_controls[] = active; end
    Label(toggles_grid[2, 2], "Show Player 2 Activist Controls")

    show_nature_controls = Observable(false)
    nature_toggle = Toggle(toggles_grid[3, 1], active=false)
    on(nature_toggle.active) do active; show_nature_controls[] = active; end
    Label(toggles_grid[3, 2], "Show Nature Controls")

    show_executed_trajectory_toggle = Toggle(toggles_grid[4, 1], active=true)
    on(show_executed_trajectory_toggle.active) do active; show_executed_trajectory[] = active; end
    Label(toggles_grid[4, 2], "Show Executed Trajectory")

    show_p1_planned_trajectory_toggle = Toggle(toggles_grid[5, 1], active=true)
    on(show_p1_planned_trajectory_toggle.active) do active; show_p1_planned_trajectory[] = active; end
    Label(toggles_grid[5, 2], "Show Player 1 Planned Trajectory")

    show_p2_planned_trajectory_toggle = Toggle(toggles_grid[6, 1], active=true)
    on(show_p2_planned_trajectory_toggle.active) do active; show_p2_planned_trajectory[] = active; end
    Label(toggles_grid[6, 2], "Show Player 2 Planned Trajectory")

    # --- Arrow Plotting ---

    # Draw arrows for robust activist controls - aggregate from all trials into single Observables
    all_p1_arrow_starts = @lift begin
        starts = Point2f[]
        for (exp_name, trial_id) in $exp_trial_pairs
            trial_key = (exp_name, trial_id)
            if haskey($p1_means_trajectory, trial_key)
                p1_means_traj = $p1_means_trajectory[trial_key]
                if !isempty(p1_means_traj) && $plan_time_step <= length(p1_means_traj)
                    for activist_id in 1:$min_num_activists
                        for senator_id in 1:$min_num_senators
                            belief_idx_local = (activist_id-1)*$min_num_senators + senator_id
                            if belief_idx_local <= length(p1_means_traj[$plan_time_step].blocks)
                                push!(starts, Point2f(p1_means_traj[$plan_time_step][Block(belief_idx_local)]))
                            else
                                push!(starts, Point2f(0, 0))
                            end
                        end
                    end
                end
            end
        end
        starts
    end
    
    all_p1_arrow_vectors = @lift begin
        vectors = Point2f[]
        for (exp_name, trial_id) in $exp_trial_pairs
            trial_key = (exp_name, trial_id)
            if haskey($p1_planned_controls, trial_key)
                p1_controls = $p1_planned_controls[trial_key]
                if !isempty(p1_controls) && $plan_time_step <= length(p1_controls)
                    for activist_id in 1:$min_num_activists
                        for senator_id in 1:$min_num_senators
                            belief_idx_local = (activist_id-1)*$min_num_senators + senator_id
                            if belief_idx_local <= length(p1_controls[$plan_time_step].blocks)
                                push!(vectors, Point2f(p1_controls[$plan_time_step][Block(belief_idx_local)]))
                            else
                                push!(vectors, Point2f(0, 0))
                            end
                        end
                    end
                end
            end
        end
        vectors
    end
    
    all_p2_arrow_starts = @lift begin
        starts = Point2f[]
        for (exp_name, trial_id) in $exp_trial_pairs
            trial_key = (exp_name, trial_id)
            if haskey($p2_means_trajectory, trial_key)
                p2_means_traj = $p2_means_trajectory[trial_key]
                if !isempty(p2_means_traj) && $plan_time_step <= length(p2_means_traj)
                    for activist_id in 1:$min_num_activists
                        for senator_id in 1:$min_num_senators
                            belief_idx_local = (activist_id-1)*$min_num_senators + senator_id
                            if belief_idx_local <= length(p2_means_traj[$plan_time_step].blocks)
                                push!(starts, Point2f(p2_means_traj[$plan_time_step][Block(belief_idx_local)]))
                            else
                                push!(starts, Point2f(0, 0))
                            end
                        end
                    end
                end
            end
        end
        starts
    end
    
    all_p2_arrow_vectors = @lift begin
        vectors = Point2f[]
        for (exp_name, trial_id) in $exp_trial_pairs
            trial_key = (exp_name, trial_id)
            if haskey($p2_planned_controls, trial_key)
                p2_controls = $p2_planned_controls[trial_key]
                if !isempty(p2_controls) && $plan_time_step <= length(p2_controls)
                    for activist_id in 1:$min_num_activists
                        for senator_id in 1:$min_num_senators
                            belief_idx_local = (activist_id-1)*$min_num_senators + senator_id
                            if belief_idx_local <= length(p2_controls[$plan_time_step].blocks)
                                push!(vectors, Point2f(p2_controls[$plan_time_step][Block(belief_idx_local)]))
                            else
                                push!(vectors, Point2f(0, 0))
                            end
                        end
                    end
                end
            end
        end
        vectors
    end
    
    # Create arrow color array matching the structure
    arrow_colors = @lift vcat([fill(colors[activist_id], $min_num_senators) for activist_id in 1:$min_num_activists]...)
    
    # Create arrow plots ONCE with aggregated Observables
    arrows!(ax, all_p1_arrow_starts, all_p1_arrow_vectors, color=arrow_colors, visible=show_p1_activist_controls)
    arrows!(ax, all_p2_arrow_starts, all_p2_arrow_vectors, color=arrow_colors, visible=show_p2_activist_controls, linestyle=:dash)

    # Draw arrows for nature's controls #TODO: Fix nature controls
    # is_robust = @lift begin
    #     if !isempty($p1_planned_controls)
    #         first_controls = first(values($p1_planned_controls))
    #         !isempty(first_controls) && !isempty(first_controls[1].blocks) && length(first_controls[1]) > sum($first_trial_dims.control_dims_per_activist)
    #     else
    #         false
    #     end
    # end
    # for senator_id in 1:first_trial_dims_val.num_senators
    #     local senator_id_local = senator_id
    #     arrow_starts = @lift begin
    #         starts = Point2f[]
    #         for (k, means_traj) in $p1_means_trajectory
    #             if !isempty(means_traj) && $plan_time_step <= length(means_traj) && senator_id_local <= length(means_traj[$plan_time_step].blocks)
    #                 push!(starts, Point2f(means_traj[$plan_time_step][Block(senator_id_local)]))
    #             end
    #         end
    #         starts
    #     end
    #     arrow_vectors = @lift begin
    #         vectors = Point2f[]
    #         if $is_robust
    #             for (k, controls) in $p1_planned_controls
    #                 if !isempty(controls) && $plan_time_step <= length(controls)
    #                     control_vec = controls[$plan_time_step]
    #                     last_block_idx = length(control_vec.blocks)
    #                     nature_control_vec = control_vec[Block(last_block_idx)]
    #                     if length(nature_control_vec) == sum($first_trial_dims.state_dims_per_activist)
    #                         nature_control_block = BlockVector(nature_control_vec, $first_trial_dims.state_dims_per_activist)
    #                         if senator_id_local <= length(nature_control_block.blocks)
    #                             push!(vectors, Point2f(nature_control_block[Block(senator_id_local)]))
    #                         end
    #                     end
    #                 end
    #             end
    #         end
    #         vectors
    #     end
    #     arrows!(ax, arrow_starts, arrow_vectors, color=:green, visible=show_nature_controls)
    # end

    # Create observables and plots for covariance ellipses
    # Store as Dict{Tuple{String, String}, Vector{Observable{Vector{Point2f}}}}
    # Keyed by (exp_name, trial_id), each value is a vector of observables (one per activist/senator pair)
    p1_ellipse_observables_dict = Dict{Tuple{String, String}, Vector{Observable{Vector{Point2f}}}}()
    p2_ellipse_observables_dict = Dict{Tuple{String, String}, Vector{Observable{Vector{Point2f}}}}()
    
    # Function to update ellipses
    function update_ellipses(time_step, pairs)
        # Create ellipse observables lazily if they don't exist
        for (exp_name, trial_id) in pairs
            key = (exp_name, trial_id)
            
            # Create P1 ellipse observables if they don't exist
            if !haskey(p1_ellipse_observables_dict, key)
                # println("  Creating P1 ellipse observables lazily for trial: $key")
                p1_list = Observable{Vector{Point2f}}[]
                min_num_activists_val = min_num_activists[]
                min_num_senators_val = min_num_senators[]
                for activist_id in 1:min_num_activists_val
                    for senator_id in 1:min_num_senators_val
                        p1_obs = Observable(Point2f[])
                        lines!(ax, p1_obs, color=colors[activist_id], visible=show_p1_planned_trajectory)
                        push!(p1_list, p1_obs)
                    end
                end
                p1_ellipse_observables_dict[key] = p1_list
                # println("    Created P1 ellipse list with $(length(p1_list)) entries")
            end
            
            # Create P2 ellipse observables if they don't exist
            if !haskey(p2_ellipse_observables_dict, key)
                # println("  Creating P2 ellipse observables lazily for trial: $key")
                p2_list = Observable{Vector{Point2f}}[]
                min_num_activists_val = min_num_activists[]
                min_num_senators_val = min_num_senators[]
                for activist_id in 1:min_num_activists_val
                    for senator_id in 1:min_num_senators_val
                        p2_obs = Observable(Point2f[])
                        lines!(ax, p2_obs, color=colors[activist_id], visible=show_p2_planned_trajectory, linestyle=:dash)
                        push!(p2_list, p2_obs)
                    end
                end
                p2_ellipse_observables_dict[key] = p2_list
                # # println("    Created P2 ellipse list with $(length(p2_list)) entries")
            end
        end
        
        # Now update the ellipses (they should all exist now)
        for (exp_name, trial_id) in pairs
            key = (exp_name, trial_id)
            
            # Check all required keys before accessing
            if !haskey(p1_covariances_trajectory[], key) || !haskey(p1_means_trajectory[], key) ||
               !haskey(p2_covariances_trajectory[], key) || !haskey(p2_means_trajectory[], key)
                continue
            end
            
            p1_covs_traj = p1_covariances_trajectory[][key]
            p2_covs_traj = p2_covariances_trajectory[][key]
            p1_means_traj = p1_means_trajectory[][key]
            p2_means_traj = p2_means_trajectory[][key]
            if time_step < 1 || isempty(p1_covs_traj) || isempty(p1_means_traj) || 
               time_step > min(length(p1_covs_traj), length(p1_means_traj))
                continue
            end
            if time_step < 1 || isempty(p2_covs_traj) || isempty(p2_means_traj) ||
               time_step > min(length(p2_covs_traj), length(p2_means_traj))
                continue
            end
            current_p1_covs = p1_covs_traj[time_step]
            current_p2_covs = p2_covs_traj[time_step]
            current_p1_means = p1_means_traj[time_step]
            current_p2_means = p2_means_traj[time_step]
            trial_dims = dims_dict[][key]
            p1_plot_idx = 1
            p2_plot_idx = 1
            for activist_id in 1:trial_dims.num_activists
                for senator_id in 1:trial_dims.num_senators
                    belief_idx = (activist_id - 1) * trial_dims.num_senators + senator_id
                    if belief_idx <= length(current_p1_means.blocks)
                        center = Point2f(current_p1_means[Block(belief_idx)][1], current_p1_means[Block(belief_idx)][2])
                        cov = current_p1_covs[belief_idx]
                        ellipse_points = get_ellipse_points(center, cov)
                        p1_ellipse_observables_dict[key][p1_plot_idx][] = ellipse_points
                        p1_plot_idx += 1
                    end
                    if belief_idx <= length(current_p2_means.blocks)
                        center = Point2f(current_p2_means[Block(belief_idx)][1], current_p2_means[Block(belief_idx)][2])
                        cov = current_p2_covs[belief_idx]
                        ellipse_points = get_ellipse_points(center, cov)
                        p2_ellipse_observables_dict[key][p2_plot_idx][] = ellipse_points
                        p2_plot_idx += 1
                    end
                end
            end
        end
    end
    
    # Use on() callback instead of @lift to create plot elements only when trials change
    on(exp_trial_pairs) do pairs
        for (exp_name, trial_id) in pairs
            trial_key = (exp_name, trial_id)
            for p1_obs in values(p1_ellipse_observables_dict[trial_key])
                p1_obs[] = Point2f[]
            end
            for p2_obs in values(p2_ellipse_observables_dict[trial_key])
                p2_obs[] = Point2f[]
            end
        end
        empty!(p1_ellipse_observables_dict)
        empty!(p2_ellipse_observables_dict)
        for (exp_name, trial_id) in pairs
            trial_key = (exp_name, trial_id)
            p1_list = Observable{Vector{Point2f}}[]
            p2_list = Observable{Vector{Point2f}}[]
            
            min_num_activists_val = min_num_activists[]
            min_num_senators_val = min_num_senators[]
            
            for activist_id in 1:min_num_activists_val
                for senator_id in 1:min_num_senators_val
                    p1_obs = Observable(Point2f[])
                    p2_obs = Observable(Point2f[])
                    lines!(ax, p1_obs, color=colors[activist_id], visible=show_p1_planned_trajectory)
                    lines!(ax, p2_obs, color=colors[activist_id], visible=show_p2_planned_trajectory, linestyle=:dash)
                    push!(p1_list, p1_obs)
                    push!(p2_list, p2_obs)
                end
            end
            
            p1_ellipse_observables_dict[trial_key] = p1_list
            p2_ellipse_observables_dict[trial_key] = p2_list
        end
        
        # Manually trigger updates after populating dictionaries
        update_ellipses(plan_time_step[], pairs)
    end
    # Collect points from all selected trials into single observables
    p1_points = @lift begin
        pts = Point2f[]
        for (exp_name, trial_id) in $exp_trial_pairs
            key = (exp_name, trial_id)
            if haskey($p1_means_trajectory, key)
                p1_means_traj = $p1_means_trajectory[key]
                if $plan_time_step <= length(p1_means_traj) && !isempty(p1_means_traj)
                    current_means = p1_means_traj[$plan_time_step]
                    for activist_id in 1:$min_num_activists
                        for senator_id in 1:$min_num_senators
                            belief_idx = (activist_id-1)*$min_num_senators + senator_id
                            if belief_idx <= length(current_means.blocks)
                                push!(pts, Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2]))
                            end
                        end
                    end
                end
            end
        end
        pts
    end
    
    p2_points = @lift begin
        pts = Point2f[]
        for (exp_name, trial_id) in $exp_trial_pairs
            key = (exp_name, trial_id)
            if haskey($p2_means_trajectory, key)
                p2_means_traj = $p2_means_trajectory[key]
                if $plan_time_step <= length(p2_means_traj) && !isempty(p2_means_traj)
                    current_means = p2_means_traj[$plan_time_step]
                    for activist_id in 1:$min_num_activists
                        for senator_id in 1:$min_num_senators
                            belief_idx = (activist_id-1)*$min_num_senators + senator_id
                            if belief_idx <= length(current_means.blocks)
                                push!(pts, Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2]))
                            end
                        end
                    end
                end
            end
        end
        pts
    end
    # Create scatter plots once with reactive observables
    scatter!(ax, p1_points, color=point_colors, markersize=8, visible=show_p1_planned_trajectory)
    scatter!(ax, p2_points, color=point_colors, markersize=8, visible=show_p2_planned_trajectory)
    # Compute component costs for all shown trials, using lifted observables
    p1_total_costs = @lift [c.total for (key, trial_costs) in $p1_costs_dict for c in trial_costs]
    p1_terminal_costs = @lift [c.terminal for (key, trial_costs) in $p1_costs_dict for c in trial_costs]
    p1_non_terminal_costs = @lift [c.non_terminal for (key, trial_costs) in $p1_costs_dict for c in trial_costs]

    p2_total_costs = @lift [c.total for (key, trial_costs) in $p2_costs_dict for c in trial_costs]
    p2_terminal_costs = @lift [c.terminal for (key, trial_costs) in $p2_costs_dict for c in trial_costs]
    p2_non_terminal_costs = @lift [c.non_terminal for (key, trial_costs) in $p2_costs_dict for c in trial_costs]

    # Create graphs once during setup with reactive data
    add_multi_line_graph!(fig;
        series=[p1_total_costs, p2_total_costs],
        labels=["player 1","player 2"],
        current_time_step=current_time_step,
        title="Cost Over Time",
        ylabel="cost",
        # timesteps = 0:4,                     # optional custom x-axis
        scalarizer = to_scalar_cost,            # no-op for numbers; handy for cost structs
        loc=(1,3)                               # put it in column 3, row 1 (side graph)
    )

    add_multi_line_graph!(fig;
        series=[p1_terminal_costs, p1_non_terminal_costs, p2_terminal_costs, p2_non_terminal_costs],
        labels=["p1 terminal", "p1 non-terminal", "p2 terminal", "p2 non-terminal"],
        current_time_step=current_time_step,
        title="Component Cost Over Time",
        ylabel="cost",
        # timesteps = 0:4,                     # optional custom x-axis
        scalarizer = to_scalar_cost,            # no-op for numbers; handy for cost structs
        loc=(2,3)                               # put it in column 3, row 2 (side graph)
    )
    colsize!(fig.layout, 3, Relative(0.25))



    # Update ellipses on slider change (but not when exp_trial_pairs changes)
    on(plan_time_step) do time_step
        update_ellipses(time_step, exp_trial_pairs[])
    end
    
    # Also update ellipses when the main time slider changes
    # The planned belief trajectory depends on current_time_step, so ellipses need to update
    on(current_time_step) do new_t
        update_ellipses(plan_time_step[], exp_trial_pairs[])
        # Reset plan slider to 1 when receding horizon step changes
        set_close_to!(plan_slider, 1)
    end

    if !isempty(keys(experiments))
        first_exp = first(keys(experiments))
        if haskey(selection_state, first_exp)
            first_obs = selection_state[first_exp][1]
            first_obs[] = first_obs[]  # Force trigger by setting to itself
        end
    end
    #test scatter
    #axislegend(ax)
end

# ---- Optional: convert complex "cost" objects to scalars ----
to_scalar_cost(c) = c isa Number ? float(c) :
                    c isa AbstractArray ? sum(skipmissing(vec(c))) :
                    c isa NamedTuple && hasproperty(c, :total) ? float(c.total) :
                    c isa AbstractDict && haskey(c, :total) ? float(c[:total]) :
                    try
                        float(getfield(c, :total))
                    catch
                        missing
                    end

"""
    add_multi_line_graph!(
        parent;
        series::Vector{<:AbstractVector},
        labels::Vector{<:AbstractString},
        current_time_step::Observable{Int}=Observable(typemax(Int)),
        title::AbstractString = "Series over time",
        xlabel::AbstractString = "t",
        ylabel::AbstractString = "value",
        timesteps::Union{Nothing,AbstractVector}=nothing,
        scalarizer::Function = identity,
        loc::Tuple{Int,Int} = (1, 2)
    ) -> Axis

Plot multiple time-aligned series on a single axis. Each element of `series` is a
vector of values at the same discrete timesteps. Use `scalarizer` (e.g., `to_scalar_cost`)
if your elements aren’t plain numbers.

- `parent`: either a `Figure` (an axis will be placed at `loc`) or an existing `Axis`.
- `current_time_step`: if you pass the same Observable you use for your main slider,
  the plot reveals points up to that step; otherwise it shows all points.
- `timesteps`: optional x-values (defaults to `1:N`).
- Returns the created/used `Axis`.
"""
function add_multi_line_graph!(parent;
    series::Vector,
    labels::Vector{<:AbstractString},
    current_time_step::Observable{Int}=Observable(typemax(Int)),
    title::AbstractString = "Series over time",
    xlabel::AbstractString = "t",
    ylabel::AbstractString = "value",
    timesteps::Union{Nothing,AbstractVector}=nothing,
    scalarizer::Function = identity,
    loc::Tuple{Int,Int} = (1, 2)
)
    @assert length(series) == length(labels) "series and labels must have same length"

    # Create or use an axis
    ax = parent isa Figure ? Axis(parent[loc...], title=title, xlabel=xlabel, ylabel=ylabel) :
                             (parent isa Axis ? parent :
                              error("parent must be a Figure or an Axis"))

    # Scalarize & sanitize each series, wrap in Observables for reactive updates
    obs_series = Vector{Observable{Vector{Float64}}}(undef, length(series))
    for i in eachindex(series)
        if series[i] isa Observable
            # If already an observable, create a reactive chain that scalarizes the values
            obs_series[i] = @lift begin
                sc = scalarizer.($(series[i]))
                sc = collect(skipmissing(sc))
                Float64.(sc)
            end
        else
            # Regular vector, convert to observable
            sc = scalarizer.(series[i])
            sc = collect(skipmissing(sc))
            obs_series[i] = Observable(Float64.(sc))
        end
    end
    maxval = map(obs_series...) do series...
        combined = vcat(series...)
        isempty(combined) ? 0 : maximum(combined)
    end

    # All series should share the same time grid; we use the shortest length
    # Make these reactive since obs_series are observables that may update
    minlen = map(obs_series...) do series...
        lengths = length.(series)
        isempty(lengths) ? 0 : minimum(lengths)
    end
    
    maxval = map(obs_series...) do series...
        combined = vcat(series...)
        isempty(combined) ? 0 : maximum(combined)
    end
    xt = timesteps === nothing ? nothing : Observable(vec(timesteps))

for i in eachindex(obs_series)
    local oi = obs_series[i]
    
    # Create a single observable that computes points directly
    points_obs = if timesteps === nothing
        @lift begin
            oi_val = $oi
            ml = $minlen
            actual_len = min(ml, length(oi_val))
            if actual_len > 0
                Point2f.(1:actual_len, oi_val[1:actual_len])
            else
                Point2f[]
            end
        end
    else
        @lift begin
            oi_val = $oi
            xt_val = $xt
            ml = $minlen
            actual_len = min(ml, length(oi_val), length(xt_val))
            if actual_len > 0
                Point2f.(xt_val[1:actual_len], oi_val[1:actual_len])
            else
                Point2f[]
            end
        end
    end
    
    lines!(ax, points_obs, label=labels[i])
end
    # # Keep y-limits comfy as we reveal more points
    # on(current_time_step) do _
    #     k_now = min(current_time_step[], minlen())
    #     ys = Float64[]
    #     for oi in obs_series
    #         append!(ys, oi[][1:k_now])
    #     end
    #     if !isempty(ys)
    #         ymin, ymax = extrema(ys)
    #         pad = max(1e-9, 0.05 * (ymax - ymin + 1e-12))
    #         ax.ylimits = (ymin - pad, ymax + pad)
    #     end
    # end
    return ax
end
end
