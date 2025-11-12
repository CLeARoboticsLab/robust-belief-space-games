module SenateVisuals

using GLMakie
using Makie.Colors
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

# Helper function to convert color symbols to RGBA with 50% opacity
function color_with_alpha(color, alpha=0.4)
    c = color isa Symbol ? to_color(color) : color
    return RGBA(c.r, c.g, c.b, alpha)
end

function load_solution(folder, filename, type = "mass_results")
    results = nothing
    if filename isa Vector{String}
       for file in filename
            if type == "mass_results"
                path = "exp/senate/outputs/$folder/$(file)_$type.dat"
            else
                path = "exp/senate/outputs/$folder/$(file).dat"
            end
            if isnothing(results)
                results = open(deserialize, path, "r")
            else
                new_results = open(deserialize, path, "r")
                append!(results, new_results)
            end
        end
        filename = filename[1]
    elseif filename isa String
        if type == "mass_results"
            path = "exp/senate/outputs/$folder/$(filename)_$type.dat"
        else
            path = "exp/senate/outputs/$folder/$(filename).dat"
        end
        results = open(deserialize, path, "r")
    else
        error("filename must be a String or Array of Strings")
    end
    experiments = Dict{String, Dict{String, Tuple{Dict, Dict, Main.Senate.SenateParams}}}()
    
    # Handle case where results itself is a Dict (direct from outputs/runs)
    if results isa Dict{String, Any}
        # This is a single experiment file from outputs/runs
        # Use the filename as the experiment name
        exp_name = filename != "" ? filename : "experiment"
        experiments[exp_name] = Dict{String, Tuple{Dict, Dict, Main.Senate.SenateParams}}()
        for (trial_id, trial_data) in results
            if trial_data isa Tuple && length(trial_data) == 3
                solutions, games, cost_params = trial_data
                experiments[exp_name][trial_id] = (solutions, games, cost_params)
            else
                @warn "Unexpected format for trial_data in $exp_name/$trial_id: $(typeof(trial_data)). Skipping."
            end
        end
    elseif results isa Vector
        # This is the expected format - a Vector of experiment entries
        for exp_data in results
            # Handle both NamedTuple and Tuple formats
            if exp_data isa NamedTuple
                params = exp_data.params
                fixed = exp_data.fixed
                trial_results = exp_data.results
                exp_name = exp_data.name
            elseif exp_data isa Tuple && length(exp_data) == 4
                params, fixed, trial_results, exp_name = exp_data
            else
                @warn "Unexpected format for exp_data: $(typeof(exp_data)). Skipping."
                continue
            end
            
            if isnothing(trial_results)
                @warn "No results for experiment: $(exp_name). Skipping."
                continue
            end
            
            # Handle filename shortening if needed
            if filename != "" && startswith(exp_name, filename)
                exp_name = exp_name[length(filename)+1:end] # Shorten to only parameters
            end
            
            # Handle case where trial_results might be a Dict (from outputs/runs format)
            # or already in the expected format
            if trial_results isa Dict{String, Any}
                # This is the format from outputs/runs - need to extract the tuple from each trial
                experiments[exp_name] = Dict{String, Tuple{Dict, Dict, Main.Senate.SenateParams}}()
                for (trial_id, trial_data) in trial_results
                    if trial_data isa Tuple && length(trial_data) == 3
                        solutions, games, cost_params = trial_data
                        experiments[exp_name][trial_id] = (solutions, games, cost_params)
                    else
                        @warn "Unexpected format for trial_data in $exp_name/$trial_id: $(typeof(trial_data)). Skipping."
                    end
                end
            else
                @warn "Unexpected format for trial_results in $exp_name: $(typeof(trial_results)). Skipping."
            end
        end
    else
        error("Unexpected format for results: $(typeof(results)). Expected Vector or Dict{String, Any}.")
    end
    visualize_receding_horizon_solution(experiments, filename)
end

function plot_ellipse!(ax, center, a, b; n=100, label="", color=color_with_alpha(:black))
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

# Parse experiment name to extract parameters
# Format: prefix_param1_value1_param2_value2_...
# Example: asym_p2t_robust_p2nm_2.0_dmt_default_p2bpdss_0.0_h_7_p1t_non_robust_gtis_[0.75, 0.75, 1.75, 1.0, 1.0, 1.75]
function parse_experiment_name(exp_name::String)
    params = Dict{String, Any}()
    parts = split(exp_name, "_")
    
    # Define known parameter keys to identify where the parameters start
    known_keys = ["p1t", "p2t", "p1nm", "p2nm", "dmt", "p1bpdss", "p2bpdss", "gtis", "h"]
    
    key_indices = findall(part -> part in known_keys, parts)
    
    if isempty(key_indices)
        @warn "Could not find any known parameter keys in experiment name: $exp_name"
        return params
    end
    
    for i in 1:length(key_indices)
        start_idx = key_indices[i]
        param_key = parts[start_idx]
        
        end_idx = (i < length(key_indices)) ? key_indices[i+1] - 1 : length(parts)
        
        value_parts = parts[start_idx+1 : end_idx]
        
        if isempty(value_parts)
            @warn "No value found for key '$param_key' in experiment name: $exp_name"
            continue
        end
        
        value_str = join(value_parts, "_")
        
        # Array parsing
        if startswith(value_str, "[")
            params[param_key] = value_str
            continue
        end
        
        # Number parsing
        try
            if occursin(".", value_str)
                params[param_key] = parse(Float64, value_str)
            else
                params[param_key] = parse(Int, value_str)
            end
            continue
        catch
            # Not a number, treat as a string
        end
        
        # String value
        params[param_key] = value_str
    end
    
    return params
end

function create_individual_solution_plot(fig, ax, experiments::Dict)# sol_data, dims, cost_params)
    # Parse all experiment names to extract parameters
    exp_params = Dict{String, Dict{String, Any}}()
    for exp_name in keys(experiments)
        exp_params[exp_name] = parse_experiment_name(exp_name)
    end
    
    # Extract unique parameter values for each parameter key
    param_values = Dict{String, Set{Any}}()
    for params in values(exp_params)
        for (key, value) in params
            if !haskey(param_values, key)
                param_values[key] = Set{Any}()
            end
            push!(param_values[key], value)
        end
    end
    
    # Create parameter filter observables
    param_filters = Dict{String, Dict{Any, Observable{Bool}}}()
    for (param_key, values) in param_values
        param_filters[param_key] = Dict{Any, Observable{Bool}}()
        for value in values
            param_filters[param_key][value] = Observable(true)  # Default to all selected
        end
    end
    
    # Create trial selection state (keep trial-level toggles)
    trial_selection_state = Dict{String, Dict{String, Observable{Bool}}}()
    for exp_name in keys(experiments)
        trial_selections = Dict{String, Observable{Bool}}()
        for trial_id in keys(experiments[exp_name])
            trial_selections[trial_id] = Observable(true) #TODO revert to false
        end
        trial_selection_state[exp_name] = trial_selections
    end
    
    # Create combined observables for filtering
    all_filter_observables = Observable[]
    for param_dict in values(param_filters)
        for obs in values(param_dict)
            push!(all_filter_observables, obs)
        end
    end
    for trial_dict in values(trial_selection_state)
        for obs in values(trial_dict)
            push!(all_filter_observables, obs)
        end
    end
    
    # Filter experiments based on parameter selections
    exp_trial_pairs = lift(all_filter_observables...) do _...
        [
            (exp_name, trial_id)
            for exp_name in keys(experiments)
            for trial_id in keys(experiments[exp_name])
            if begin
                # Check if experiment matches all parameter filters
                params = exp_params[exp_name]
                matches = true
                for (param_key, value_filters) in param_filters
                    if haskey(params, param_key)
                        param_value = params[param_key]
                        if haskey(value_filters, param_value)
                            matches = matches && value_filters[param_value][]
                        else
                            matches = false  # Parameter value not in filters
                            break
                        end
                    end
                end
                # Also check trial selection
                matches && trial_selection_state[exp_name][trial_id][]
            end
        ]
    end
    
    # Set first experiment/trial selected to true
    if !isempty(experiments)
        first_exp = first(keys(experiments))
        if haskey(trial_selection_state, first_exp) && !isempty(trial_selection_state[first_exp])
            first_trial = first(keys(trial_selection_state[first_exp]))
            trial_selection_state[first_exp][first_trial][] = true
        end
    end
    
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
    colsize!(fig.layout, 1, Relative(0.35))  # Wider for two columns of controls
    
    # Add title
    Label(selection_panel[1, 1:4], "Parameter Filters", fontsize=14, font=:bold, halign=:left)
    
    # Ensure proper column separation and sizing
    colgap!(selection_panel, 15)
    rowgap!(selection_panel, 5)
    # Set explicit column widths to prevent overlap
    colsize!(selection_panel, 1, Auto())
    colsize!(selection_panel, 2, Auto())
    colsize!(selection_panel, 3, Auto())
    colsize!(selection_panel, 4, Auto())
    
    # Parameter name mappings for display
    param_display_names = Dict(
        "p2t" => "P2 Type",
        "p2nm" => "P2 Nature Mult",
        "dmt" => "Dynamics Model",
        "p2bpdss" => "P2 Belief Drift",
        "gtis" => "Ground Truth",
        "h" => "Horizon",
        "p1t" => "P1 Type"
    )
    
    # Create parameter filter controls
    sorted_param_keys = sort(collect(keys(param_values)))
    
    # Filter for parameters with more than one value
    filterable_params = filter(k -> length(param_values[k]) > 1, sorted_param_keys)
    
    if isempty(filterable_params)
        Label(selection_panel[2, 1:4], "No parameters to filter.", halign=:left)
    else
        # Balance parameters across two columns
        param_heights = Dict(k => 1 + length(param_values[k]) + 1 for k in filterable_params)
        total_height = sum(values(param_heights))
        target_col_height = total_height / 2
        
        col1_params = String[]
        col2_params = String[]
        current_col1_height = 0
        
        for p_key in filterable_params
            if current_col1_height < target_col_height || isempty(col2_params)
                push!(col1_params, p_key)
                current_col1_height += param_heights[p_key]
            else
                push!(col2_params, p_key)
            end
        end
        
        # --- Render Column 1 ---
        row_idx = 2
        for param_key in col1_params
            display_name = get(param_display_names, param_key, param_key)
            Label(selection_panel[row_idx, 1:2], display_name, fontsize=12, font=:bold, halign=:left)
            row_idx += 1
            
            sorted_values = sort(collect(param_values[param_key]), lt=(a, b) -> string(a) < string(b))
            
            for value in sorted_values
                value_str = string(value)
                if length(value_str) > 15
                    value_str = value_str[1:12] * "..."
                end
                
                toggle = Toggle(selection_panel[row_idx, 1], active=param_filters[param_key][value][])
                on(toggle.active) do active
                    param_filters[param_key][value][] = active
                end
                Label(selection_panel[row_idx, 2], value_str, fontsize=10, halign=:left)
                row_idx += 1
            end
            row_idx += 1  # Spacing
        end
        
        # --- Render Column 2 ---
        row_idx = 2
        for param_key in col2_params
            display_name = get(param_display_names, param_key, param_key)
            Label(selection_panel[row_idx, 3:4], display_name, fontsize=12, font=:bold, halign=:left)
            row_idx += 1
            
            sorted_values = sort(collect(param_values[param_key]), lt=(a, b) -> string(a) < string(b))
            
            for value in sorted_values
                value_str = string(value)
                if length(value_str) > 15
                    value_str = value_str[1:12] * "..."
                end
                
                toggle = Toggle(selection_panel[row_idx, 3], active=param_filters[param_key][value][])
                on(toggle.active) do active
                    param_filters[param_key][value][] = active
                end
                Label(selection_panel[row_idx, 4], value_str, fontsize=10, halign=:left)
                row_idx += 1
            end
            row_idx += 1  # Spacing
        end
    end
    
    # --- Layout ---
    controls_grid = fig[2, 2] = GridLayout(tellwidth=false)
    colsize!(fig.layout, 2, Relative(0.40))
    current_time_step = Observable(1)
    plan_time_step = Observable(1)
    show_p1_planned_trajectory = Observable(true)
    show_p2_planned_trajectory = Observable(true)
    show_executed_trajectory = Observable(true)
    colors = [color_with_alpha(:blue), color_with_alpha(:red)]

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
    point_colors = @lift begin
        base_colors = vcat([fill(c, $min_num_senators) for c in colors]...)
        num_trials = length($exp_trial_pairs)
        vcat([base_colors for _ in 1:num_trials]...)
    end
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
                scatter_plot = scatter!(ax, trajectory_obs, color=color_with_alpha(:green), markersize=8, visible=show_executed_trajectory) #Temporarily disabled executed trajectory points
                line_plot = lines!(ax, trajectory_obs, color=color_with_alpha(:green), visible=show_executed_trajectory)
                push!(plot_list, (scatter_plot, line_plot, trajectory_obs))
            end
            
            executed_trajectory_plots[trial_key] = plot_list
        end
    end
    
    # Update executed trajectory Observables when time step changes
    on(current_time_step) do t
        for (trial_key, plot_list) in executed_trajectory_plots
            if haskey(p1_sols[], trial_key) && haskey(p1_sols[][trial_key], :gt_state_history)
                gt_history = p1_sols[][trial_key].gt_state_history
                if t <= length(gt_history)
                    for (senator_idx, (_, _, trajectory_obs)) in enumerate(plot_list)
                        if senator_idx <= length(gt_history[t].blocks)
                            senator_states = [Point2f(gt_history[time][Block(senator_idx)][1], gt_history[time][Block(senator_idx)][2]) for time in 1:t]
                            trajectory_obs[] = senator_states
                        end
                    end
                end
            elseif haskey(p1_solution_history_dict[], trial_key)
                # Fallback: use first activist's belief about this senator
                solution_history = p1_solution_history_dict[][trial_key]
                for (senator_idx, (_, _, trajectory_obs)) in enumerate(plot_list)
                    if senator_idx <= length(solution_history) && t <= length(solution_history)
                        senator_states = [solution_history[time][1].beliefs[senator_idx].belief_mean for time in 1:t]
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
    # Create arrow color array matching the structure (replicated for each trial)
    arrow_colors = @lift begin
        base_colors = vcat([fill(colors[activist_id], $min_num_senators) for activist_id in 1:$min_num_activists]...)
        num_trials = length($exp_trial_pairs)
        vcat([base_colors for _ in 1:num_trials]...)
    end    
    
    # Create arrow plots ONCE with aggregated Observables
    arrows!(ax, all_p1_arrow_starts, all_p1_arrow_vectors, color=arrow_colors, visible=show_p1_activist_controls)
    arrows!(ax, all_p2_arrow_starts, all_p2_arrow_vectors, color=arrow_colors, visible=show_p2_activist_controls, linestyle=:dash)

    # Draw arrows for nature's controls - aggregate from all trials
    all_nature_arrow_starts = @lift begin
        starts = Point2f[]
        for (exp_name, trial_id) in $exp_trial_pairs
            trial_key = (exp_name, trial_id)
            if !haskey($dims_dict, trial_key)
                continue
            end
            trial_dims = $dims_dict[trial_key]
            
            # Check p1 controls for nature and add corresponding starts
            if haskey($p1_planned_controls, trial_key) && haskey($p1_means_trajectory, trial_key)
                p1_controls = $p1_planned_controls[trial_key]
                p1_means_traj = $p1_means_trajectory[trial_key]
                if !isempty(p1_controls) && $plan_time_step <= length(p1_controls) && 
                   !isempty(p1_means_traj) && $plan_time_step <= length(p1_means_traj)
                    control_vec = p1_controls[$plan_time_step]
                    if !isempty(control_vec.blocks)
                        total_control_dims = sum(trial_dims.control_dims_per_activist)
                        if length(control_vec) > total_control_dims
                            # Nature controls exist, add starts from p1's beliefs
                            for senator_id in 1:trial_dims.num_senators
                                # Get senator position from first activist's belief (indices 1, 2, 3 for senators 1, 2, 3)
                                belief_idx = senator_id
                                if belief_idx <= length(p1_means_traj[$plan_time_step].blocks)
                                    push!(starts, Point2f(p1_means_traj[$plan_time_step][Block(belief_idx)]))
                                else
                                    push!(starts, Point2f(0, 0))
                                end
                            end
                        end
                    end
                end
            end
            
            # Check p2 controls for nature and add corresponding starts
            if haskey($p2_planned_controls, trial_key) && haskey($p2_means_trajectory, trial_key)
                p2_controls = $p2_planned_controls[trial_key]
                p2_means_traj = $p2_means_trajectory[trial_key]
                if !isempty(p2_controls) && $plan_time_step <= length(p2_controls) &&
                   !isempty(p2_means_traj) && $plan_time_step <= length(p2_means_traj)
                    control_vec = p2_controls[$plan_time_step]
                    if !isempty(control_vec.blocks)
                        total_control_dims = sum(trial_dims.control_dims_per_activist)
                        if length(control_vec) > total_control_dims
                            # Nature controls exist, add starts from p2's beliefs
                            for senator_id in 1:trial_dims.num_senators
                                # Get senator position from first activist's belief
                                belief_idx = senator_id
                                if belief_idx <= length(p2_means_traj[$plan_time_step].blocks)
                                    push!(starts, Point2f(p2_means_traj[$plan_time_step][Block(belief_idx)]))
                                else
                                    push!(starts, Point2f(0, 0))
                                end
                            end
                        end
                    end
                end
            end
        end
        starts
    end
    
    all_nature_arrow_vectors = @lift begin
        vectors = Point2f[]
        for (exp_name, trial_id) in $exp_trial_pairs
            trial_key = (exp_name, trial_id)
            if !haskey($dims_dict, trial_key)
                continue
            end
            trial_dims = $dims_dict[trial_key]
            
            # Check p1 controls for nature
            if haskey($p1_planned_controls, trial_key)
                p1_controls = $p1_planned_controls[trial_key]
                if !isempty(p1_controls) && $plan_time_step <= length(p1_controls)
                    control_vec = p1_controls[$plan_time_step]
                    if !isempty(control_vec.blocks)
                        # Check if nature controls exist (last block should be nature if robust)
                        total_control_dims = sum(trial_dims.control_dims_per_activist)
                        if length(control_vec) > total_control_dims
                            last_block_idx = length(control_vec.blocks)
                            nature_control_vec = control_vec[Block(last_block_idx)]
                            # Validate dimensions match state_dims_per_activist
                            if length(nature_control_vec) == sum(trial_dims.state_dims_per_activist)
                                nature_control_block = BlockVector(nature_control_vec, trial_dims.state_dims_per_activist)
                                for senator_id in 1:trial_dims.num_senators
                                    if senator_id <= length(nature_control_block.blocks)
                                        push!(vectors, Point2f(nature_control_block[Block(senator_id)]))
                                    else
                                        push!(vectors, Point2f(0, 0))
                                    end
                                end
                            else
                                # Add zero vectors if dimensions don't match
                                for _ in 1:trial_dims.num_senators
                                    push!(vectors, Point2f(0, 0))
                                end
                            end
                        else
                            # No nature controls, skip (don't add vectors)
                        end
                    end
                end
            end
            
            # Check p2 controls for nature
            if haskey($p2_planned_controls, trial_key)
                p2_controls = $p2_planned_controls[trial_key]
                if !isempty(p2_controls) && $plan_time_step <= length(p2_controls)
                    control_vec = p2_controls[$plan_time_step]
                    if !isempty(control_vec.blocks)
                        # Check if nature controls exist (last block should be nature if robust)
                        total_control_dims = sum(trial_dims.control_dims_per_activist)
                        if length(control_vec) > total_control_dims
                            last_block_idx = length(control_vec.blocks)
                            nature_control_vec = control_vec[Block(last_block_idx)]
                            # Validate dimensions match state_dims_per_activist
                            if length(nature_control_vec) == sum(trial_dims.state_dims_per_activist)
                                nature_control_block = BlockVector(nature_control_vec, trial_dims.state_dims_per_activist)
                                for senator_id in 1:trial_dims.num_senators
                                    if senator_id <= length(nature_control_block.blocks)
                                        push!(vectors, Point2f(nature_control_block[Block(senator_id)]))
                                    else
                                        push!(vectors, Point2f(0, 0))
                                    end
                                end
                            else
                                # Add zero vectors if dimensions don't match
                                for _ in 1:trial_dims.num_senators
                                    push!(vectors, Point2f(0, 0))
                                end
                            end
                        else
                            # No nature controls, skip (don't add vectors)
                        end
                    end
                end
            end
        end
        vectors
    end
    # Filter out co-index pairs where vector is (0,0)
    filtered_nature_arrows = @lift begin
        filtered_starts = Point2f[]
        filtered_vectors = Point2f[]
        for (start, vec) in zip($all_nature_arrow_starts, $all_nature_arrow_vectors)
            if vec != Point2f(0, 0)
                push!(filtered_starts, start)
                push!(filtered_vectors, vec)
            end
        end
        (filtered_starts, filtered_vectors)
    end
    
    filtered_nature_arrow_starts = @lift $filtered_nature_arrows[1]
    filtered_nature_arrow_vectors = @lift $filtered_nature_arrows[2]
    
    arrows!(ax, filtered_nature_arrow_starts, filtered_nature_arrow_vectors, color=color_with_alpha(:green), visible=show_nature_controls)
    # Create observables and plots for covariance ellipses
    # Store as Dict{Tuple{String, String}, Vector{Observable{Vector{Point2f}}}}
    # Keyed by (exp_name, trial_id), each value is a vector of observables (one per activist/senator pair)
    p1_ellipse_observables_dict = Dict{Tuple{String, String}, Vector{Observable{Vector{Point2f}}}}()
    p2_ellipse_observables_dict = Dict{Tuple{String, String}, Vector{Observable{Vector{Point2f}}}}()
    # Separate dictionaries to track plot elements for deletion
    p1_ellipse_plots_dict = Dict{Tuple{String, String}, Vector{Any}}()
    p2_ellipse_plots_dict = Dict{Tuple{String, String}, Vector{Any}}()
    
    # Function to update ellipses
    function update_ellipses(time_step, pairs)
        # Create ellipse observables lazily if they don't exist
        for (exp_name, trial_id) in pairs
            key = (exp_name, trial_id)
            
            # Create P1 ellipse observables if they don't exist
            if !haskey(p1_ellipse_observables_dict, key)
                # println("  Creating P1 ellipse observables lazily for trial: $key")
                p1_list = Observable{Vector{Point2f}}[]
                p1_plots = Any[]
                min_num_activists_val = min_num_activists[]
                min_num_senators_val = min_num_senators[]
                for activist_id in 1:min_num_activists_val
                    for senator_id in 1:min_num_senators_val
                        p1_obs = Observable(Point2f[])
                        p1_plot = lines!(ax, p1_obs, color=colors[activist_id], visible=show_p1_planned_trajectory)
                        push!(p1_list, p1_obs)
                        push!(p1_plots, p1_plot)
                    end
                end
                p1_ellipse_observables_dict[key] = p1_list
                p1_ellipse_plots_dict[key] = p1_plots
                # println("    Created P1 ellipse list with $(length(p1_list)) entries")
            end
            
            # Create P2 ellipse observables if they don't exist
            if !haskey(p2_ellipse_observables_dict, key)
                # println("  Creating P2 ellipse observables lazily for trial: $key")
                p2_list = Observable{Vector{Point2f}}[]
                p2_plots = Any[]
                min_num_activists_val = min_num_activists[]
                min_num_senators_val = min_num_senators[]
                for activist_id in 1:min_num_activists_val
                    for senator_id in 1:min_num_senators_val
                        p2_obs = Observable(Point2f[])
                        p2_plot = lines!(ax, p2_obs, color=colors[activist_id], visible=show_p2_planned_trajectory, linestyle=:dash)
                        push!(p2_list, p2_obs)
                        push!(p2_plots, p2_plot)
                    end
                end
                p2_ellipse_observables_dict[key] = p2_list
                p2_ellipse_plots_dict[key] = p2_plots
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
        # Delete ALL plot elements before clearing dictionaries
        for plots in values(p1_ellipse_plots_dict)
            for plot in plots
                delete!(ax, plot)
            end
        end
        for plots in values(p2_ellipse_plots_dict)
            for plot in plots
                delete!(ax, plot)
            end
        end
        empty!(p1_ellipse_observables_dict)
        empty!(p2_ellipse_observables_dict)
        empty!(p1_ellipse_plots_dict)
        empty!(p2_ellipse_plots_dict)
        
        # Let update_ellipses lazily recreate everything
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
    p1_total_costs = @lift [[c.total for c in trial_costs] for (key, trial_costs) in $p1_costs_dict]
    p1_terminal_costs = @lift [[c.terminal for c in trial_costs] for (key, trial_costs) in $p1_costs_dict]
    p1_non_terminal_costs = @lift [[c.non_terminal for c in trial_costs] for (key, trial_costs) in $p1_costs_dict]

    p2_total_costs = @lift [[c.total for c in trial_costs] for (key, trial_costs) in $p2_costs_dict]
    p2_terminal_costs = @lift [[c.terminal for c in trial_costs] for (key, trial_costs) in $p2_costs_dict]
    p2_non_terminal_costs = @lift [[c.non_terminal for c in trial_costs] for (key, trial_costs) in $p2_costs_dict]

    # Create graphs once during setup with reactive data
    add_multi_line_graph!(fig;
        full_series=[p1_total_costs, p2_total_costs],
        labels=["player 1","player 2"], # todo: fix labels
        current_time_step=current_time_step,
        title="Cost Over Time",
        ylabel="cost",
        # timesteps = 0:4,                     # optional custom x-axis
        scalarizer = to_scalar_cost,            # no-op for numbers; handy for cost structs
        loc=(1,3)                               # put it in column 3, row 1 (side graph)
    )

    add_multi_line_graph!(fig;
        full_series=[p1_terminal_costs, p1_non_terminal_costs, p2_terminal_costs, p2_non_terminal_costs],
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
        if haskey(trial_selection_state, first_exp) && !isempty(trial_selection_state[first_exp])
            first_trial = first(keys(trial_selection_state[first_exp]))
            first_obs = trial_selection_state[first_exp][first_trial]
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
    full_series::Vector,
    labels::Vector{<:AbstractString},
    current_time_step::Observable{Int}=Observable(typemax(Int)),
    title::AbstractString = "Series over time",
    xlabel::AbstractString = "t",
    ylabel::AbstractString = "value",
    timesteps::Union{Nothing,AbstractVector}=nothing,
    scalarizer::Function = identity,
    loc::Tuple{Int,Int} = (1, 2)
)
    @assert length(full_series) == length(labels) "series and labels must have same length"

    # Create or use an axis
    ax = parent isa Figure ? Axis(parent[loc...], title=title, xlabel=xlabel, ylabel=ylabel) :
                             (parent isa Axis ? parent :
                              error("parent must be a Figure or an Axis"))
    # Create a reactive series that rebuilds when trial count changes
    # series = map(full_series[1]) do first_data
    #     n = length(first_data)
    #     [map(data -> trial_idx <= length(data) ? data[trial_idx] : Vector{Vector{Float64}}[], s) 
    #      for trial_idx in 1:n 
    #      for s in full_series]
    # end
    n = length(full_series[1][])  # Get current number of trials
    series = [Observable(s[][trial_idx]) 
            for trial_idx in 1:n 
            for s in full_series]
    #TODO: DYNAMICALLY UPDATE COST WITH THE REST OF THE GRAPH - unfinished code
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
        
        lines!(ax, points_obs) #TODO: add back labels
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
