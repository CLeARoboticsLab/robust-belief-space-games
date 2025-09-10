module TrajectoryAnalysis

using JLD2
using FileIO
using Statistics
using LinearAlgebra
using BlockArrays
using Infiltrator
using RobustBeliefGame
using Printf
using CairoMakie

include("KKTErrorTracker.jl")
using .KKTErrorTracker

include("hockey/Hockey.jl")
using .Hockey

# ========================================================================================
# TRAJECTORY ANALYSIS TRACKER SYSTEM
# ========================================================================================

export TrajectoryAnalysisEntry, TrajectoryAnalysisTracker, TRAJECTORY_TRACKER, clear_trajectory_tracker!,
    load_and_analyze_solution_files, compute_belief_covariance_traces, compute_player_distances,
    compute_belief_deviations, effect_of_nature, get_trajectory_summary, create_trajectory_analysis_plots, get_trajectory_details,
    create_yarnball_plot_for_cost_components

"""
    TrajectoryAnalysisEntry

Stores trajectory analysis data for a single scenario execution.
"""
struct TrajectoryAnalysisEntry
    scenario_name::String
    trial_number::Int
    
    # Raw trajectory data from solution file
    gt_state_history::Any 
    all_observations::Any
    solution_history::Any
    cond_history::Any
    lq_sol_history::Any
    costs::Any

    # Additional metadata
    robust::Bool
    noise_level::String
end

"""
    TrajectoryAnalysisTracker

Global tracker for trajectory analysis data, similar to KKT tracker.
"""
mutable struct TrajectoryAnalysisTracker
    entries::Vector{TrajectoryAnalysisEntry}
    auto_save::Bool
    save_file::String
end

# Global instance
const TRAJECTORY_TRACKER = TrajectoryAnalysisTracker(TrajectoryAnalysisEntry[], false, "")

"""
    clear_trajectory_tracker!()

Clear all trajectory analysis data.
"""
function clear_trajectory_tracker!()
    empty!(TRAJECTORY_TRACKER.entries)
    println("Trajectory analysis tracker cleared")
end

"""
    load_and_analyze_solution_files()

Load trajectory data from saved solution files (rh_multi-trial_*.jld2) and analyze them.
"""
function load_and_analyze_solution_files(;prefix="rh_multi-trial")
    println("Loading trajectory data from solution files...")
    
    # Clear existing trajectory data
    clear_trajectory_tracker!()
    
    # Find solution files
    output_dir = "exp/hockey/outputs"
    if !isdir(output_dir)
        println("Directory $output_dir not found")
        return false
    end
    
    all_files = readdir(output_dir)
    solution_files = [joinpath(output_dir, f) for f in all_files if startswith(f, prefix) && endswith(f, ".jld2")]
    
    println("Found $(length(solution_files)) solution files")
    
    for solution_file in solution_files
        @load solution_file solutions goal_position
        
        for (key, solution_data) in solutions
            # Parse scenario info from key (e.g., "medium_non_robust_10")
            parts = split(key, "_")
            if length(parts) >= 4  # noise_robust_type_trial
                noise_level = parts[1]  # "low", "medium", "high"
                robust_type = parts[2]  # "non" or empty for robust
                robust_str = parts[3]   # "robust"
                trial_num_str = parts[4]  # trial number as string
                
                # Handle cases where trial number might not be pure numeric
                local trial_num
                try
                    trial_num = parse(Int, trial_num_str)
                catch
                    println("Warning: Could not parse trial number from '$trial_num_str' in key '$key'")
                    continue  # Skip this entry
                end
                
                is_robust = (robust_type != "non")
                
            elseif length(parts) >= 3  # Handle case like "low_robust_1"
                noise_level = parts[1]  # "low", "medium", "high"
                robust_str = parts[2]   # "robust"
                trial_num_str = parts[3]  # trial number as string
                
                local trial_num
                try
                    trial_num = parse(Int, trial_num_str)
                catch
                    println("Warning: Could not parse trial number from '$trial_num_str' in key '$key'")
                    continue  # Skip this entry
                end
                
                is_robust = true
            else
                println("Warning: Could not parse key format '$key'")
                continue
            end
            gt_state_history, all_observations, solution_history, cond_history, lq_sol_history = solution_data
            
            # TODO: save the cost functions with the solution data.
            explicit_covariance = false
            attacker_cost = BeliefCost(
                (bs, us) -> attacker_non_terminal_cost(bs.beliefs[1], bs.beliefs[2], us; explicit_covariance=explicit_covariance),
                (bs) -> attacker_terminal_cost(bs.beliefs[1], bs.beliefs[2])
            )
            defender_cost = BeliefCost(
                (bs, us) -> defender_non_terminal_cost(bs.beliefs[3], bs.beliefs[4], us; explicit_covariance=explicit_covariance),
                (bs) -> defender_terminal_cost(bs.beliefs[3], bs.beliefs[4])
            )
            nature_cost = BeliefCost(
                (bs, us) -> nature_non_terminal_cost(bs.beliefs[3], bs.beliefs[4], us; explicit_covariance=explicit_covariance),
                (bs) -> nature_terminal_cost(bs.beliefs[3], bs.beliefs[4])
            )
            costs = [attacker_cost, defender_cost, nature_cost]
            
            if !isempty(gt_state_history)
                entry = TrajectoryAnalysisEntry(
                    key,
                    trial_num,
                    gt_state_history,
                    all_observations,
                    solution_history,
                    cond_history,
                    lq_sol_history,
                    costs,
                    is_robust,
                    noise_level,
                )
                
                push!(TRAJECTORY_TRACKER.entries, entry)
            end
        end
    end
    
    println("Loaded trajectory data for $(length(TRAJECTORY_TRACKER.entries)) scenarios")
    return true
end

function compute_belief_covariance_traces(entry::TrajectoryAnalysisEntry)
    belief_covariances = Float64[]
    if !isempty(entry.solution_history)
        try
            for (t, sols) in enumerate(entry.solution_history)
                for player_idx in 1:min(length(sols), 2)
                    if length(sols[player_idx]) >= 1
                        nominal_beliefs, _ = sols[player_idx]
                        if !isempty(nominal_beliefs)
                            current_beliefs = nominal_beliefs[1]
                            for belief in current_beliefs.beliefs
                                push!(belief_covariances, tr(belief.belief_covariance))
                            end
                        end
                    end
                end
            end
        catch e
            println("Warning: Could not extract belief covariance data for $(entry.scenario_name): $e")
        end
    end
    return belief_covariances
end

function compute_player_distances(entry::TrajectoryAnalysisEntry)
    player_distances = Float64[]
    if !isempty(entry.gt_state_history)
        for state in entry.gt_state_history
            player1_pos = state[Block(1)][1:2]
            player2_pos = state[Block(2)][1:2]
            push!(player_distances, norm(player1_pos - player2_pos))
        end
    end
    return player_distances
end

function compute_belief_deviations(entry::TrajectoryAnalysisEntry)
    belief_deviations = Float64[]
    if !isempty(entry.solution_history) && !isempty(entry.gt_state_history)
        try
            for (t, sols) in enumerate(entry.solution_history)
                if t > length(entry.gt_state_history) continue end
                gt_state = entry.gt_state_history[t]
                
                for player_idx in 1:min(length(sols), 2)
                    if length(sols[player_idx]) >= 1
                        nominal_beliefs, _ = sols[player_idx]
                        if !isempty(nominal_beliefs)
                            current_beliefs = nominal_beliefs[1]
                            for (belief_idx, belief) in enumerate(current_beliefs.beliefs)
                                if belief_idx <= length(gt_state.blocks)
                                    gt_pos = gt_state[Block(belief_idx)][1:2]
                                    belief_pos = belief.belief_mean[1:2]
                                    push!(belief_deviations, norm(belief_pos - gt_pos))
                                end
                            end
                        end
                    end
                end
            end
        catch e
            println("Warning: Could not extract belief deviation data for $(entry.scenario_name): $e")
        end
    end
    return belief_deviations
end
# Group by base scenario: "<noise_level>_<robustness>" where robustness is "robust" or "non_robust"
function base_group(name::String)
    parts = split(name, "_")
    if length(parts) >= 3 && parts[2] == "non" && parts[3] == "robust"
        return string(parts[1], "_non_robust")
    elseif length(parts) >= 2
        return string(parts[1], "_", parts[2])
    else
        return name
    end
end

"""
    get_trajectory_summary()

Get summary statistics and create analysis plots.
"""
function get_trajectory_summary()
    if isempty(TRAJECTORY_TRACKER.entries)
        println("No trajectory analysis data available")
        return nothing
    end
    
    println("\n=== TRAJECTORY ANALYSIS SUMMARY ===")
    println("Total entries: $(length(TRAJECTORY_TRACKER.entries))")
    
    base_groups = unique([base_group(e.scenario_name) for e in TRAJECTORY_TRACKER.entries])
    
    all_planned_costs = Dict()

    for base in sort(base_groups)
        group_entries = [e for e in TRAJECTORY_TRACKER.entries if base_group(e.scenario_name) == base]
        # a HACK for now
        explicit_covariance = false
        scenario_planned_trajectory_costs = [calculate_planned_trajectory_costs(e, explicit_covariance) for e in group_entries]
        all_planned_costs[base] = scenario_planned_trajectory_costs
    end

    create_yarnball_plot_for_cost_components(all_planned_costs)

    return
end

"""
Calculate the costs incurred for each planned trajectory
"""
function calculate_planned_trajectory_costs(entry::TrajectoryAnalysisEntry, explicit_covariance::Bool)
    player_names = [:attacker, :defender, :nature]
    
    map(entry.solution_history) do sols
        cost_breakdown = Dict()
        num_players = 2
        
        for (player_idx, player_name) in enumerate(player_names[1:num_players])
            # an entry for each time step, which is a named tuple of cost components
            trajectory_costs = []
            
            beliefs_traj, controls_traj = sols[player_idx]
            if isempty(beliefs_traj) || isempty(controls_traj)
                cost_breakdown[player_name] = []
                continue
            end

            planning_horizon = length(beliefs_traj)

            # Determine which beliefs to use
            belief_indices = if player_name == :attacker
                (1, 2)
            else # defender and nature use defender's beliefs
                (3, 4)
            end

            # Non-terminal costs
            for t in 1:(planning_horizon - 1)
                
                attacker_belief = beliefs_traj[t].beliefs[belief_indices[1]]
                defender_belief = beliefs_traj[t].beliefs[belief_indices[2]]

                non_terminal_costs = Hockey.player_cost_components[player_name].non_terminal(
                    attacker_belief, 
                    defender_belief,
                    controls_traj[t];
                    explicit_covariance=explicit_covariance
                )
                push!(trajectory_costs, non_terminal_costs)
            end
            
            # Terminal cost
            attacker_belief = beliefs_traj[end].beliefs[belief_indices[1]]
            defender_belief = beliefs_traj[end].beliefs[belief_indices[2]]
            terminal_costs = Hockey.player_cost_components[player_name].terminal(
                attacker_belief,
                defender_belief;
                explicit_covariance=explicit_covariance
            )
            push!(trajectory_costs, terminal_costs)

            cost_breakdown[player_name] = trajectory_costs
        end
        cost_breakdown
    end
end

"""
    create_belief_covariance_plots(scenarios)

Create plots showing belief covariance evolution over time.
"""
function create_belief_covariance_plots(scenarios)
    fig = Figure(size = (1200, 800))
    
    # Create subplots for each scenario
    n_scenarios = length(scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = ceil(Int, n_scenarios / n_cols)
    
    for (i, scenario) in enumerate(scenarios)
        row = ceil(Int, i / n_cols)
        col = mod(i - 1, n_cols) + 1
        
        ax = Axis(fig[row, col],
            title = "Belief Covariance - $(replace(scenario, "_" => " "))",
            xlabel = "Time Step",
            ylabel = "Covariance Trace"
        )
        
        scenario_entries = [e for e in TRAJECTORY_TRACKER.entries if e.scenario_name == scenario]
        
        # Plot each trial's covariance evolution
        colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
        
        for (j, entry) in enumerate(scenario_entries)
            color = colors[mod(j-1, length(colors)) + 1]
            belief_covariances = compute_belief_covariance_traces(entry)
            time_steps = 1:length(belief_covariances)
            
            lines!(ax, time_steps, belief_covariances,
                color = color,
                alpha = 0.7,
                linewidth = 2
            )
        end
        
        # Add mean line
        if !isempty(scenario_entries)
            all_covariances = [compute_belief_covariance_traces(e) for e in scenario_entries]
            max_length = maximum(length(covs) for covs in all_covariances)
            mean_covariances = Float64[]
            
            for t in 1:max_length
                values_at_t = [covs[t] for covs in all_covariances if length(covs) >= t]
                if !isempty(values_at_t)
                    push!(mean_covariances, mean(values_at_t))
                end
            end
            
            lines!(ax, 1:length(mean_covariances), mean_covariances,
                color = :black,
                linewidth = 3,
                linestyle = :dash
            )
        end
    end
    
    save("trajectory_belief_covariances.png", fig)
    println("Belief covariance plot saved as trajectory_belief_covariances.png")
end

"""
    create_player_distance_plots(scenarios)

Create plots showing player distance evolution over time.
"""
function create_player_distance_plots(scenarios)
    fig = Figure(size = (1200, 800))
    
    # Create subplots for each scenario
    n_scenarios = length(scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = ceil(Int, n_scenarios / n_cols)
    
    for (i, scenario) in enumerate(scenarios)
        row = ceil(Int, i / n_cols)
        col = mod(i - 1, n_cols) + 1
        
        ax = Axis(fig[row, col],
            title = "Player Distance - $(replace(scenario, "_" => " "))",
            xlabel = "Time Step",
            ylabel = "Distance Between Players"
        )
        
        scenario_entries = [e for e in TRAJECTORY_TRACKER.entries if e.scenario_name == scenario]
        
        # Plot each trial's distance evolution
        colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
        
        for (j, entry) in enumerate(scenario_entries)
            color = colors[mod(j-1, length(colors)) + 1]
            player_distances = compute_player_distances(entry)
            time_steps = 1:length(player_distances)
            
            lines!(ax, time_steps, player_distances,
                color = color,
                alpha = 0.7,
                linewidth = 2
            )
        end
        
        # Add mean line
        if !isempty(scenario_entries)
            all_distances = [compute_player_distances(e) for e in scenario_entries]
            max_length = maximum(length(dists) for dists in all_distances)
            mean_distances = Float64[]
            
            for t in 1:max_length
                values_at_t = [dists[t] for dists in all_distances if length(dists) >= t]
                if !isempty(values_at_t)
                    push!(mean_distances, mean(values_at_t))
                end
            end
            
            lines!(ax, 1:length(mean_distances), mean_distances,
                color = :black,
                linewidth = 3,
                linestyle = :dash
            )
        end
    end
    
    save("trajectory_player_distances.png", fig)
    println("Player distance plot saved as trajectory_player_distances.png")
end

"""
    create_planned_trajectory_costs_plots(planned_trajectory_costs)

Create plots showing planned trajectory costs over time.
"""
function create_planned_trajectory_costs_plots(planned_trajectory_costs)
    
end

function create_yarnball_plot_for_cost_components(all_planned_costs)
    println("Generating yarnball plots for cost components...")

    for (scenario, scenario_costs) in all_planned_costs # scenario_costs is for all trials of a scenario
        
        if isempty(scenario_costs) continue end
        num_rh_steps = length(scenario_costs[1])
        if num_rh_steps == 0 continue end

        for rh_step in 1:num_rh_steps
            # one trial's data to get keys
            first_trial_data = scenario_costs[1]
            if rh_step > length(first_trial_data) continue end
            rh_step_data_for_keys = first_trial_data[rh_step]
            player_names = collect(keys(rh_step_data_for_keys))

            # Get all unique components across all players for this rh_step
            all_components = Set{Symbol}()
            for player_name in player_names
                first_plan_traj = rh_step_data_for_keys[player_name]
                if !isempty(first_plan_traj)
                    for step in first_plan_traj
                        union!(all_components, keys(step))
                    end
                end
            end
            component_names = sort(collect(all_components), by=string)

            if isempty(component_names) || isempty(player_names) continue end
            
            num_components = length(component_names)
            num_cols = ceil(Int, sqrt(num_components))
            num_rows = ceil(Int, num_components / num_cols)

            fig = Figure(size=(500 * num_cols, 400 * num_rows))
            Label(fig[0, :], text = "$scenario (planning stage $rh_step)", fontsize = 24)
            
            player_colors = Dict(zip(player_names, [:blue, :red, :green, :orange, :purple]))

            for (comp_idx, component) in enumerate(component_names)
                row = ceil(Int, comp_idx / num_cols)
                col = mod1(comp_idx, num_cols)

                ax = Axis(fig[row, col], title=string(component))

                for player_name in player_names
                    color = player_colors[player_name]
                    for trial_data in scenario_costs # loop over trials
                        if rh_step > length(trial_data) continue end
                        rh_step_data = trial_data[rh_step] 
                        if !haskey(rh_step_data, player_name) continue end
                        plan_traj_costs = rh_step_data[player_name]
                        if isempty(plan_traj_costs) continue end
                        
                        component_trajectory = [get(step, component, 0.0) for step in plan_traj_costs]

                        lines!(ax, 1:length(component_trajectory), component_trajectory, color=(color, 0.3))
                    end
                end

                elements = [LineElement(color = player_colors[p], linestyle = :solid) for p in player_names]
                axislegend(ax, elements, string.(player_names), "Players")
            end
            
            save("exp/hockey/outputs/yarnball_$(scenario)_stage_$(rh_step).png", fig)
            println("Saved yarnball plot to exp/hockey/outputs/yarnball_$(scenario)_stage_$(rh_step).png")
        end
    end
end

"""
    save_trajectory_tracker(filename)

Save trajectory analysis data to file.
"""
function save_trajectory_tracker(filename::String)
    try
        @save filename TRAJECTORY_TRACKER
        println("Trajectory analysis data saved to $filename")
        return true
    catch e
        println("Error saving trajectory data: $e")
        return false
    end
end

"""
    load_trajectory_tracker(filename)

Load trajectory analysis data from file.
"""
function load_trajectory_tracker(filename::String)
    try
        if !isfile(filename)
            println("File not found: $filename")
            return false
        end
        
        loaded_data = load(filename)
        if haskey(loaded_data, "TRAJECTORY_TRACKER")
            loaded_tracker = loaded_data["TRAJECTORY_TRACKER"]
            TRAJECTORY_TRACKER.entries = loaded_tracker.entries
            TRAJECTORY_TRACKER.auto_save = loaded_tracker.auto_save
            TRAJECTORY_TRACKER.save_file = loaded_tracker.save_file
            
            println("Loaded trajectory analysis data: $(length(TRAJECTORY_TRACKER.entries)) entries")
            return true
        else
            println("No TRAJECTORY_TRACKER found in file")
            return false
        end
    catch e
        println("Error loading trajectory data: $e")
        return false
    end
end

# ========================================================================================
# HOCKEY-SPECIFIC ANALYSIS FUNCTIONS
# ========================================================================================

"""
    analyze_hockey_trajectory_data()

Analyze trajectory data from hockey receding horizon experiments.
"""
function analyze_hockey_trajectory_data(;prefix="rh_multi-trial")
    println("\n=== ANALYZING HOCKEY TRAJECTORY DATA ===")
    
    # Load trajectory data from solution files
    if load_and_analyze_solution_files(;prefix=prefix)
        get_trajectory_summary()
    else
        println("Failed to load trajectory data from solution files")
    end
end

"""
    clear_hockey_trajectory_tracker!()

Clear all trajectory tracking data for hockey experiments.
"""
function clear_hockey_trajectory_tracker!()
    clear_trajectory_tracker!()
end

"""
    save_hockey_trajectory_data(base_filename="hockey_trajectory_analysis")

Save hockey trajectory tracking data to files.
"""
function save_hockey_trajectory_data(base_filename::String="hockey_trajectory_analysis")
    save_trajectory_tracker("$base_filename.jld2")
    println("Hockey trajectory data saved to $base_filename.jld2")
end

"""
    load_hockey_trajectory_data(filename="hockey_trajectory_analysis.jld2")

Load hockey trajectory tracking data from file.
"""
function load_hockey_trajectory_data(filename::String="hockey_trajectory_analysis.jld2")
    success = load_trajectory_tracker(filename)
    if success
        println("Hockey trajectory data loaded successfully")
    else
        println("Failed to load hockey trajectory data")
    end
    return success
end

"""
    plot_spatial_trajectories()

Plot the actual x,y trajectories from the receding horizon solution data.
"""
function plot_spatial_trajectories()
    println("\n=== Plotting Spatial Trajectories ===")
    
    # Try to load the solution data
    output_dir = "exp/hockey/outputs"
    if !isdir(output_dir)
        println("Directory $output_dir not found")
        return
    end
    
    # Find all files that match the pattern rh_*.jld2
    all_files = readdir(output_dir)
    solution_files = [joinpath(output_dir, f) for f in all_files if startswith(f, "rh_multi-trial") && endswith(f, ".jld2")]
    
    if isempty(solution_files)
        println("No receding horizon solution files found in $output_dir")
        println("Available files: ", all_files)
        return
    end
    
    println("Found $(length(solution_files)) solution files")
    
    # Collect all scenarios from solution files
    all_scenarios = Set{String}()
    
    for solution_file in solution_files
        try
            @load solution_file solutions goal_position
            
            for (key, solution_data) in solutions
                # Parse scenario info from key (e.g., "low_robust_1")
                parts = split(key, "_")
                if length(parts) >= 2
                    scenario_name = "$(parts[1])_$(parts[2])"  # e.g., "low_robust", "medium_non"
                    push!(all_scenarios, scenario_name)
                end
            end
        catch e
            println("Error reading $solution_file: $e")
        end
    end
    
    println("Found scenarios: ", collect(all_scenarios))
    
    # Create separate plot for each scenario
    for scenario in all_scenarios
        println("\n--- Plotting scenario: $scenario ---")
        
        # Create figure for this scenario
        fig = Figure(size = (1000, 800))
        
        ax = Axis(fig[1, 1],
            title = "Hockey Player Trajectories - $(replace(scenario, "_" => " "))",
            xlabel = "X Position",
            ylabel = "Y Position",
            aspect = 1
        )
        
        # Plot goal
        goal_position_default = [
            [0.25, -1.5],
            [-0.25, -1.5],
        ]
        goal_posts = [[p[1] for p in goal_position_default], [p[2] for p in goal_position_default]]
        lines!(ax, goal_posts[1], goal_posts[2], color = :black, linewidth = 5)
        
        # Color scheme for different trials
        trial_colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
        
        trajectory_count = 0
        trial_counter = 1
        
        for solution_file in solution_files
            try
                @load solution_file solutions goal_position
                
                for (key, solution_data) in solutions
                    # Parse scenario info from key
                    parts = split(key, "_")
                    if length(parts) >= 2
                        key_scenario = "$(parts[1])_$(parts[2])"
                        
                        # Only plot if this matches our current scenario
                        if key_scenario == scenario
                            trial_num = length(parts) >= 3 ? parts[3] : string(trial_counter)
                            
                            # Get trajectory data
                            gt_state_history, all_observations, solution_history, cond_history, lq_sol_history = solution_data
                            
                            # Extract x,y positions for each player
                            if !isempty(gt_state_history)
                                # Player 1 (Attacker) positions
                                player1_x = [state[Block(1)][1] for state in gt_state_history]
                                player1_y = [state[Block(1)][2] for state in gt_state_history]
                                
                                # Player 2 (Defender) positions  
                                player2_x = [state[Block(2)][1] for state in gt_state_history]
                                player2_y = [state[Block(2)][2] for state in gt_state_history]
                                
                                # Assign color for this trial
                                color = trial_colors[mod(trial_counter-1, length(trial_colors)) + 1]
                                
                                # Plot trajectories (solid for attacker, dashed for defender)
                                lines!(ax, player1_x, player1_y,
                                    color = color,
                                    linestyle = :solid,
                                    linewidth = 2,
                                    alpha = 0.8
                                )
                                
                                lines!(ax, player2_x, player2_y,
                                    color = color,
                                    linestyle = :dash,
                                    linewidth = 2,
                                    alpha = 0.8
                                )
                                
                                # Mark starting positions
                                scatter!(ax, [player1_x[1]], [player1_y[1]], 
                                    color = color, marker = :circle, markersize = 10)
                                scatter!(ax, [player2_x[1]], [player2_y[1]], 
                                    color = color, marker = :rect, markersize = 10)
                                
                                trajectory_count += 1
                                trial_counter += 1
                            end
                        end
                    end
                end
            catch e
                println("Error loading $solution_file: $e")
            end
        end
        
        # Create legend
        legend_elements = [
            LineElement(color = :black, linestyle = :solid, linewidth = 3),
            LineElement(color = :black, linestyle = :dash, linewidth = 3),
            MarkerElement(color = :black, marker = :circle, markersize = 12),
            MarkerElement(color = :black, marker = :rect, markersize = 12),
            LineElement(color = :black, linewidth = 5)
        ]
        legend_labels = ["Attacker Trajectory", "Defender Trajectory", "Attacker Start", "Defender Start", "Goal"]
        
        Legend(fig[1, 2], legend_elements, legend_labels, "Legend")
        
        # Save the plot
        filename = "hockey_spatial_trajectories_$(scenario).png"
        save(filename, fig)
        println("Saved $filename with $trajectory_count trajectories")
        
        trial_counter = 1  # Reset for next scenario
    end
    
    println("\n=== All scenario plots completed ===")
end

"""
    clear_hockey_kkt_tracker!()

Clear all KKT tracking data for hockey experiments.
"""
function clear_hockey_kkt_tracker!()
    clear_rh_kkt_tracker!()
end


end
