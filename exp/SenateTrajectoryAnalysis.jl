module SenateTrajectoryAnalysis

using Serialization
using Statistics
using LinearAlgebra
using BlockArrays
using Printf
using CairoMakie
using RobustBeliefGame

# Use Senate as a registered package (has UUID in root Project.toml)
using Senate

# ========================================================================================
# SENATE TRAJECTORY ANALYSIS TRACKER SYSTEM
# ========================================================================================

export SenateTrajectoryAnalysisEntry, SenateTrajectoryAnalysisTracker, SENATE_TRAJECTORY_TRACKER,
    clear_senate_trajectory_tracker!, load_and_analyze_senate_solution_files,
    compute_senate_belief_covariance_traces, compute_senator_distances,
    get_senate_trajectory_summary, create_senate_trajectory_analysis_plots,
    compare_robust_vs_nonrobust_senate_actions, create_senate_yarnball_plot,
    analyze_senate_trajectory_data, plot_senate_spatial_trajectories

"""
    SenateTrajectoryAnalysisEntry

Stores trajectory analysis data for a single senate scenario execution.
"""
struct SenateTrajectoryAnalysisEntry
    scenario_name::String
    trial_number::Int

    # Raw trajectory data from solution file
    gt_state_history::Any
    observation_history::Any
    solution_history::Any  # Dict{Int, Vector{NamedTuple}} - indexed by player_idx
    cost_history::Any
    incurred_cost_history::Any
    params::Union{SenateParams, Nothing}

    # Additional metadata
    robust::Bool  # Whether player 2 is robust
    config_name::String  # Configuration identifier
end

"""
    SenateTrajectoryAnalysisTracker

Global tracker for senate trajectory analysis data.
"""
mutable struct SenateTrajectoryAnalysisTracker
    entries::Vector{SenateTrajectoryAnalysisEntry}
    auto_save::Bool
    save_file::String
end

# Global instance
const SENATE_TRAJECTORY_TRACKER = SenateTrajectoryAnalysisTracker(SenateTrajectoryAnalysisEntry[], false, "")

"""
    clear_senate_trajectory_tracker!()

Clear all senate trajectory analysis data.
"""
function clear_senate_trajectory_tracker!()
    empty!(SENATE_TRAJECTORY_TRACKER.entries)
    println("Senate trajectory analysis tracker cleared")
end

"""
    load_and_analyze_senate_solution_files(; directory, file_pattern)

Load trajectory data from saved senate solution files (.dat) and analyze them.
"""
function load_and_analyze_senate_solution_files(;
    directory="./exp/senate/outputs/merged/drift_test",
    file_pattern=r"")
    println("Loading senate trajectory data from solution files...")

    # Clear existing trajectory data
    clear_senate_trajectory_tracker!()

    # Find solution files
    solution_files = String[]

    if isdir(directory)
        all_files = readdir(directory)
        for f in all_files
            if file_pattern == r""
                if endswith(f, ".dat")
                    push!(solution_files, joinpath(directory, f))
                end
            elseif occursin(file_pattern, f)
                push!(solution_files, joinpath(directory, f))
            end
        end
    else
        println("Directory not found: $directory")
        return false
    end

    println("Found $(length(solution_files)) solution files")

    for solution_file in solution_files
        try
            loaded_data = open(deserialize, solution_file, "r")
            process_senate_solution_data(loaded_data, basename(solution_file))
        catch e
            println("Error loading $solution_file: $e")
            continue
        end
    end

    println("Loaded trajectory data for $(length(SENATE_TRAJECTORY_TRACKER.entries)) scenarios")
    return true
end

"""
    process_senate_solution_data(loaded_data, filename)

Process a loaded senate solution and add entries to the tracker.
Handles multiple formats:
- Vector{Any} containing NamedTuples with (params, fixed, results, name)
- (solutions_dict, params) tuple
- Dict of results
"""
function process_senate_solution_data(loaded_data, filename::String)
    # Format 1: Vector with NamedTuple entries (mass_results format)
    if loaded_data isa Vector
        for entry in loaded_data
            if entry isa NamedTuple && haskey(entry, :results)
                # Mass results format: (params=combo, fixed=fixed_params, results=results, name=exp_name)
                experiment_name = get(entry, :name, filename)
                results_dict = entry.results
                # The outer combo dict may contain :p2_type directly
                combo_params = haskey(entry, :params) ? entry.params : nothing

                for (result_key, result_value) in results_dict
                    if result_value isa Tuple && length(result_value) == 2
                        sol_dict, params = result_value
                        scenario_name = "$(experiment_name)_$(result_key)"
                        process_single_senate_solution(sol_dict, params, scenario_name, filename; combo_params=combo_params)
                    end
                end
            elseif entry isa Tuple && length(entry) == 2
                # Simple (solutions_dict, params) tuple
                sol, par = entry
                process_single_senate_solution(sol, par, filename, filename)
            end
        end
        return
    end

    # Format 2: Direct (solutions_dict, params) tuple
    if loaded_data isa Tuple && length(loaded_data) == 2
        solutions_dict, params = loaded_data
        process_single_senate_solution(solutions_dict, params, filename, filename)
        return
    end

    # Format 3: Dict of results
    if loaded_data isa Dict
        for (key, value) in loaded_data
            if value isa Tuple && length(value) == 2
                sol, par = value
                process_single_senate_solution(sol, par, string(key), filename)
            elseif value isa Dict && haskey(value, 1)
                # It's a solutions_dict directly indexed by player
                process_single_senate_solution(value, nothing, string(key), filename)
            end
        end
        return
    end

    println("Unknown format for $filename: $(typeof(loaded_data))")
end

"""
    process_single_senate_solution(solutions_dict, params, scenario_name, filename)

Process a single senate solution and add to tracker.
"""
function process_single_senate_solution(solutions_dict, params, scenario_name::String, filename::String; combo_params=nothing)
    # Extract data - solutions_dict is indexed by player_idx
    gt_state_history = nothing
    observation_history = nothing
    solution_history = Dict{Int, Any}()
    cost_history = Dict{Int, Any}()
    incurred_cost_history = Dict{Int, Any}()

    for (player_idx, player_data) in solutions_dict
        if player_data isa NamedTuple || (player_data isa Dict && haskey(player_data, :gt_state_history))
            gt_state_history = get(player_data, :gt_state_history, nothing)
            observation_history = get(player_data, :observation_history, nothing)
            solution_history[player_idx] = get(player_data, :solution_history, [])
            cost_history[player_idx] = get(player_data, :cost_history, [])
            incurred_cost_history[player_idx] = get(player_data, :incurred_cost_history, [])
        end
    end

    if isnothing(gt_state_history)
        return
    end

    # Determine robustness from params
    is_robust = false

    # Method 1: Check combo_params[:p2_type] from the sweep (most reliable)
    if !isnothing(combo_params) && combo_params isa Dict
        p2_type_val = get(combo_params, :p2_type, nothing)
        if !isnothing(p2_type_val)
            is_robust = string(p2_type_val) == "robust" || Int(p2_type_val) == Int(robust)
        end
    end

    # Method 2: Check the SenateParams from the solution
    if !is_robust && !isnothing(params)
        try
            p2_config = nothing
            if hasproperty(params, :player_configs)
                p2_config = get(params.player_configs, 2, nothing)
            elseif params isa Dict && haskey(params, 2)
                p2_config = params[2]
            end
            if !isnothing(p2_config) && hasproperty(p2_config, :type)
                is_robust = string(p2_config.type) == "robust"
            end
        catch e
        end
    end

    # Extract config name from scenario name
    config_name = extract_senate_config_name(scenario_name)

    # Parse trial number if present
    trial_num = 1
    trial_match = match(r"trial[_\s]*(\d+)", scenario_name)
    if !isnothing(trial_match)
        trial_num = parse(Int, trial_match.captures[1])
    end

    entry = SenateTrajectoryAnalysisEntry(
        scenario_name,
        trial_num,
        gt_state_history,
        observation_history,
        solution_history,
        cost_history,
        incurred_cost_history,
        params,
        is_robust,
        config_name,
    )

    push!(SENATE_TRAJECTORY_TRACKER.entries, entry)
end

"""
    extract_senate_config_name(scenario_name)

Extract the base configuration name from a scenario name.
"""
function extract_senate_config_name(scenario_name::String)
    # Remove file extension
    name = replace(scenario_name, r"\.dat$" => "")
    # Remove trial numbers and seed numbers so runs with same params group together
    name = replace(name, r"_?trial_?\d+" => "")
    name = replace(name, r"_?seed_?\d+" => "")
    name = replace(name, r"_mass_results" => "")
    # Clean up any resulting double underscores
    name = replace(name, r"__+" => "_")
    name = strip(name, '_')
    return name
end

"""
    base_senate_group(name::String)

Group scenarios by base configuration (removing robustness suffix).
"""
function base_senate_group(name::String)
    # Remove _robust or _non_robust suffix for grouping
    if endswith(name, "_robust")
        return name[1:end-7]
    elseif endswith(name, "_non_robust")
        return name[1:end-11]
    else
        return name
    end
end

# ========================================================================================
# COST COMPUTATION FUNCTIONS
# ========================================================================================

"""
    compute_senate_cost_components(beliefs, controls, config::PlayerConfig)

Compute cost components for a senate player given beliefs and controls.
Returns a NamedTuple with (preference, control, covariance, obstacle).
"""
function compute_senate_cost_components(beliefs, controls, config::PlayerConfig; is_terminal=false)
    if isnothing(beliefs) || isempty(beliefs)
        return (preference=0.0, control=0.0)
    end

    # Determine which beliefs to use based on player index
    player_idx = config.player_idx
    pretend_config = config.type == nature ? 2 : player_idx
    num_senators = config.num_senators > 0 ? config.num_senators : 3
    player_belief_indices = (pretend_config-1) * num_senators + 1:pretend_config * num_senators

    # Extract beliefs for this player's view
    belief_blocks = []
    if beliefs isa BlockVector
        belief_blocks = beliefs.blocks
    elseif hasproperty(beliefs, :beliefs)
        # Beliefs object
        for idx in player_belief_indices
            if idx <= length(beliefs.beliefs)
                push!(belief_blocks, beliefs.beliefs[idx].belief_mean)
            end
        end
    end

    # Compute preference cost (ellipsoidal)
    preference_cost = 0.0
    for pos in belief_blocks
        for (center, radii) in zip(config.ellipsoid_centers, config.ellipsoid_radii)
            opinion_dim = min(length(pos), length(center))
            for i in 1:opinion_dim
                preference_cost += config.ellipsoidal_cost_weight * (pos[i] - center[i])^2 / radii[i]
            end
        end
    end

    # Compute control cost
    control_cost = 0.0
    if !is_terminal && !isnothing(controls) && !isempty(controls)
        if controls isa BlockVector
            control_dims = config.control_dims_per_activist
            start_idx = sum(control_dims) * (player_idx - 1) + 1
            end_idx = sum(control_dims) * player_idx
            if end_idx <= length(controls)
                player_controls = controls[start_idx:end_idx]
                control_cost = config.control_cost_weight * dot(player_controls, player_controls)
            end
        else
            control_cost = config.control_cost_weight * dot(controls, controls)
        end
    end

    # Compute covariance cost if applicable
    covariance_cost = 0.0
    if config.covariance_weight > 0 && hasproperty(beliefs, :beliefs)
        for idx in player_belief_indices
            if idx <= length(beliefs.beliefs) && hasproperty(beliefs.beliefs[idx], :belief_covariance)
                covariance_cost += config.covariance_weight * tr(beliefs.beliefs[idx].belief_covariance)
            end
        end
    end

    # Compute obstacle cost if applicable
    obstacle_cost = 0.0
    if !isempty(config.obstacle_centers) && config.obstacle_weights[1] > 0
        # Use simplified obstacle cost computation
        for belief_mean in belief_blocks
            for (center, weight) in zip(config.obstacle_centers, config.obstacle_weights)
                dist = norm(belief_mean - center)
                obstacle_cost += weight * exp(-dist)  # Simplified
            end
        end
    end

    if is_terminal
        return (preference=config.terminal_cost_weight * preference_cost, covariance=covariance_cost, obstacle=obstacle_cost)
    else
        return (preference=preference_cost, control=control_cost, covariance=covariance_cost, obstacle=obstacle_cost)
    end
end

"""
    calculate_senate_planned_trajectory_costs(entry::SenateTrajectoryAnalysisEntry)

Calculate the costs for each planned trajectory in the senate game.
"""
function calculate_senate_planned_trajectory_costs(entry::SenateTrajectoryAnalysisEntry)
    if isempty(entry.solution_history) || isnothing(entry.params)
        return []
    end

    player_indices = sort(collect(keys(entry.solution_history)))

    # Get number of RH steps
    num_rh_steps = 0
    for (_, sol_hist) in entry.solution_history
        num_rh_steps = max(num_rh_steps, length(sol_hist))
    end

    all_costs = []

    for t in 1:num_rh_steps
        cost_breakdown = Dict{Int, Vector{NamedTuple}}()

        for player_idx in player_indices
            if !haskey(entry.solution_history, player_idx) || t > length(entry.solution_history[player_idx])
                cost_breakdown[player_idx] = []
                continue
            end

            step_data = entry.solution_history[player_idx][t]
            beliefs_traj = get(step_data, :beliefs, nothing)
            controls_traj = get(step_data, :controls, nothing)

            if isnothing(beliefs_traj) || isnothing(controls_traj) || isempty(beliefs_traj)
                cost_breakdown[player_idx] = []
                continue
            end

            config = get(entry.params.player_configs, player_idx, nothing)
            if isnothing(config)
                cost_breakdown[player_idx] = []
                continue
            end

            trajectory_costs = NamedTuple[]
            planning_horizon = length(beliefs_traj)

            # Non-terminal costs
            for k in 1:(planning_horizon - 1)
                if k <= length(controls_traj)
                    costs = compute_senate_cost_components(beliefs_traj[k], controls_traj[k], config; is_terminal=false)
                    push!(trajectory_costs, costs)
                end
            end

            # Terminal cost
            if !isempty(beliefs_traj)
                terminal_costs = compute_senate_cost_components(beliefs_traj[end], nothing, config; is_terminal=true)
                push!(trajectory_costs, terminal_costs)
            end

            cost_breakdown[player_idx] = trajectory_costs
        end

        push!(all_costs, cost_breakdown)
    end

    return all_costs
end

# ========================================================================================
# ANALYSIS FUNCTIONS
# ========================================================================================

"""
    compute_senate_belief_covariance_traces(entry::SenateTrajectoryAnalysisEntry)

Compute belief covariance traces over time for all beliefs.
"""
function compute_senate_belief_covariance_traces(entry::SenateTrajectoryAnalysisEntry)
    covariances = Float64[]

    for (_, sol_hist) in entry.solution_history
        for step_data in sol_hist
            beliefs_traj = get(step_data, :beliefs, nothing)
            if !isnothing(beliefs_traj) && !isempty(beliefs_traj)
                current_beliefs = beliefs_traj[1]
                if hasproperty(current_beliefs, :beliefs)
                    for belief in current_beliefs.beliefs
                        if hasproperty(belief, :belief_covariance)
                            push!(covariances, tr(belief.belief_covariance))
                        end
                    end
                end
            end
        end
    end

    return covariances
end

"""
    compute_senator_distances(entry::SenateTrajectoryAnalysisEntry)

Compute distances between senators in opinion space over time.
"""
function compute_senator_distances(entry::SenateTrajectoryAnalysisEntry)
    distances = Float64[]

    if !isempty(entry.gt_state_history)
        for state in entry.gt_state_history
            if state isa BlockVector && length(state.blocks) >= 2
                # Compute pairwise distances between senators
                for i in 1:length(state.blocks)
                    for j in (i+1):length(state.blocks)
                        push!(distances, norm(state.blocks[i] - state.blocks[j]))
                    end
                end
            end
        end
    end

    return distances
end

"""
    extract_senate_executed_controls(entry::SenateTrajectoryAnalysisEntry)

Extract the first control from each RH step for each player.
"""
function extract_senate_executed_controls(entry::SenateTrajectoryAnalysisEntry)
    executed_controls = Dict{Int, Vector{Vector{Float64}}}()

    for (player_idx, sol_hist) in entry.solution_history
        executed_controls[player_idx] = Vector{Float64}[]

        for step_data in sol_hist
            controls_traj = get(step_data, :controls, nothing)
            if !isnothing(controls_traj) && !isempty(controls_traj)
                first_control = controls_traj[1]
                if first_control isa BlockVector
                    # Extract this player's portion of the control
                    control_dims = !isnothing(entry.params) && haskey(entry.params.player_configs, player_idx) ?
                        entry.params.player_configs[player_idx].control_dims_per_activist : [2, 2, 2]
                    start_idx = sum(control_dims) * (player_idx - 1) + 1
                    end_idx = sum(control_dims) * player_idx
                    if end_idx <= length(first_control)
                        push!(executed_controls[player_idx], Vector{Float64}(first_control[start_idx:end_idx]))
                    end
                else
                    push!(executed_controls[player_idx], Vector{Float64}(first_control))
                end
            end
        end
    end

    return executed_controls
end

# ========================================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================================

"""
    get_senate_trajectory_summary(; directory)

Get summary statistics and create analysis plots for senate experiments.
"""
function get_senate_trajectory_summary(; directory="./exp/senate/outputs/analysis")
    if isempty(SENATE_TRAJECTORY_TRACKER.entries)
        println("No senate trajectory analysis data available")
        return nothing
    end

    # Create output directory if needed
    if !isdir(directory)
        mkpath(directory)
    end

    println("\n=== SENATE TRAJECTORY ANALYSIS SUMMARY ===")
    println("Total entries: $(length(SENATE_TRAJECTORY_TRACKER.entries))")

    # Group by config
    config_groups = unique([e.config_name for e in SENATE_TRAJECTORY_TRACKER.entries])

    all_planned_costs = Dict{String, Any}()

    all_config_entries = Dict{String, Vector{SenateTrajectoryAnalysisEntry}}()

    for config in sort(config_groups)
        group_entries = [e for e in SENATE_TRAJECTORY_TRACKER.entries if e.config_name == config]
        println("  Config '$config': $(length(group_entries)) entries")

        # Calculate planned trajectory costs
        scenario_planned_costs = [calculate_senate_planned_trajectory_costs(e) for e in group_entries]
        all_planned_costs[config] = scenario_planned_costs
        all_config_entries[config] = group_entries
    end

    # Create comparison plots
    compare_robust_vs_nonrobust_senate_actions(SENATE_TRAJECTORY_TRACKER.entries; directory=directory)
    create_senate_yarnball_plot(all_planned_costs, all_config_entries; directory=directory)

    return all_planned_costs
end

"""
    compare_robust_vs_nonrobust_senate_actions(all_entries; directory)

Create comparison plots for robust vs non-robust senate experiments.
"""
function compare_robust_vs_nonrobust_senate_actions(all_entries; directory="./exp/senate/outputs/analysis")
    robust_entries = [e for e in all_entries if e.robust]
    non_robust_entries = [e for e in all_entries if !e.robust]

    if isempty(robust_entries) && isempty(non_robust_entries)
        println("No entries to compare")
        return
    end

    println("Comparing $(length(robust_entries)) robust vs $(length(non_robust_entries)) non-robust entries")

    create_senate_action_difference_plots(robust_entries, non_robust_entries; directory=directory)
end

"""
    create_senate_action_difference_plots(robust_entries, non_robust_entries; directory)

Create plots comparing executed controls between robust and non-robust strategies.
"""
function create_senate_action_difference_plots(robust_entries, non_robust_entries; directory="./exp/senate/outputs/analysis")
    if isempty(robust_entries) && isempty(non_robust_entries)
        return nothing
    end

    # Extract controls
    robust_controls_all = [extract_senate_executed_controls(e) for e in robust_entries]
    non_robust_controls_all = [extract_senate_executed_controls(e) for e in non_robust_entries]

    # Determine min time steps
    min_time_steps = typemax(Int)
    for controls in vcat(robust_controls_all, non_robust_controls_all)
        for (_, ctrl_list) in controls
            if !isempty(ctrl_list)
                min_time_steps = min(min_time_steps, length(ctrl_list))
            end
        end
    end

    if min_time_steps == typemax(Int) || min_time_steps == 0
        println("No control data available for comparison")
        return nothing
    end

    # Colors
    color_p1_robust = :blue
    color_p1_non_robust = :lightblue
    color_p2_robust = :red
    color_p2_non_robust = :orange

    # ========== Figure 1: Control Norms Over Time ==========
    fig1 = Figure(size=(1200, 600))
    Label(fig1[0, :], text = "Senate Control Norms Over Time", fontsize = 16)

    ax_norm = Axis(fig1[1, 1],
        title = "Control Norms",
        xlabel = "Time Step",
        ylabel = "Control Norm (L2)"
    )

    # Helper function
    function compute_control_norms_stats(controls_all, player_idx, min_t)
        all_norms = Vector{Vector{Float64}}()
        for controls in controls_all
            if haskey(controls, player_idx) && length(controls[player_idx]) >= min_t
                norms = [norm(u) for u in controls[player_idx][1:min_t]]
                push!(all_norms, norms)
            end
        end
        if isempty(all_norms)
            return Float64[], Float64[], Float64[]
        end

        means = Float64[]
        stds = Float64[]
        for t in 1:min_t
            vals = [n[t] for n in all_norms if length(n) >= t]
            push!(means, mean(vals))
            push!(stds, length(vals) > 1 ? std(vals) : 0.0)
        end
        return 1:min_t, means, stds
    end

    # Plot P1 robust
    ts, means, stds = compute_control_norms_stats(robust_controls_all, 1, min_time_steps)
    if !isempty(ts)
        band!(ax_norm, collect(ts), means .- stds, means .+ stds, color=(color_p1_robust, 0.2))
        lines!(ax_norm, collect(ts), means, color=color_p1_robust, linewidth=2, label="P1 (Robust)")
    end

    # Plot P1 non-robust
    ts, means, stds = compute_control_norms_stats(non_robust_controls_all, 1, min_time_steps)
    if !isempty(ts)
        band!(ax_norm, collect(ts), means .- stds, means .+ stds, color=(color_p1_non_robust, 0.2))
        lines!(ax_norm, collect(ts), means, color=color_p1_non_robust, linewidth=2, linestyle=:dash, label="P1 (Non-Robust)")
    end

    # Plot P2 robust
    ts, means, stds = compute_control_norms_stats(robust_controls_all, 2, min_time_steps)
    if !isempty(ts)
        band!(ax_norm, collect(ts), means .- stds, means .+ stds, color=(color_p2_robust, 0.2))
        lines!(ax_norm, collect(ts), means, color=color_p2_robust, linewidth=2, label="P2 (Robust)")
    end

    # Plot P2 non-robust
    ts, means, stds = compute_control_norms_stats(non_robust_controls_all, 2, min_time_steps)
    if !isempty(ts)
        band!(ax_norm, collect(ts), means .- stds, means .+ stds, color=(color_p2_non_robust, 0.2))
        lines!(ax_norm, collect(ts), means, color=color_p2_non_robust, linewidth=2, linestyle=:dash, label="P2 (Non-Robust)")
    end

    axislegend(ax_norm, position=:rt)

    filename1 = joinpath(directory, "senate_control_norms.png")
    save(filename1, fig1)
    save(joinpath(directory, "senate_control_norms.pdf"), fig1)
    println("Saved control norms plot to $filename1")

    # ========== Figure 2: Opinion Space Trajectories ==========
    fig2 = Figure(size=(1000, 800))
    Label(fig2[0, :], text = "Senator Trajectories in Opinion Space", fontsize = 16)

    ax_opinion = Axis(fig2[1, 1],
        title = "Senator Positions Over Time",
        xlabel = "Opinion Dimension 1",
        ylabel = "Opinion Dimension 2"
    )

    senator_colors = [:blue, :green, :orange, :purple, :brown]

    # Plot ground truth trajectories from all entries
    for (entry_idx, entry) in enumerate(vcat(robust_entries, non_robust_entries))
        alpha = entry.robust ? 0.8 : 0.4
        linestyle = entry.robust ? :solid : :dash

        if !isempty(entry.gt_state_history)
            num_senators = length(entry.gt_state_history[1].blocks)
            for senator_idx in 1:num_senators
                xs = [state.blocks[senator_idx][1] for state in entry.gt_state_history]
                ys = [state.blocks[senator_idx][2] for state in entry.gt_state_history]

                color = senator_colors[mod1(senator_idx, length(senator_colors))]
                lines!(ax_opinion, xs, ys, color=(color, alpha), linestyle=linestyle, linewidth=1.5)

                # Mark start and end
                if entry_idx == 1 || (entry_idx == length(robust_entries) + 1 && !isempty(non_robust_entries))
                    scatter!(ax_opinion, [xs[1]], [ys[1]], color=color, marker=:circle, markersize=10)
                    scatter!(ax_opinion, [xs[end]], [ys[end]], color=color, marker=:star5, markersize=12)
                end
            end
        end
    end

    # Add legend elements
    legend_elements = [
        LineElement(color=:black, linestyle=:solid, linewidth=2),
        LineElement(color=:black, linestyle=:dash, linewidth=2),
    ]
    legend_labels = ["Robust P2", "Non-Robust P2"]
    Legend(fig2[1, 2], legend_elements, legend_labels, "Strategy")

    filename2 = joinpath(directory, "senate_opinion_trajectories.png")
    save(filename2, fig2)
    save(joinpath(directory, "senate_opinion_trajectories.pdf"), fig2)
    println("Saved opinion trajectories plot to $filename2")

    return nothing
end

"""
    create_senate_yarnball_plot(all_planned_costs, all_entries; directory)

Create yarnball plots showing cost components over planning horizons for senate games.
"""
function create_senate_yarnball_plot(all_planned_costs, all_config_entries::Dict{String, Vector{SenateTrajectoryAnalysisEntry}}; directory="./exp/senate/outputs/analysis")
    println("Generating senate yarnball plots for cost components...")

    for (config, scenario_costs) in all_planned_costs
        if isempty(scenario_costs)
            continue
        end

        config_entries = get(all_config_entries, config, SenateTrajectoryAnalysisEntry[])

        # Split costs and entries by robustness
        robust_costs = []
        non_robust_costs = []
        robust_entries = SenateTrajectoryAnalysisEntry[]
        non_robust_entries = SenateTrajectoryAnalysisEntry[]

        for (cost, entry) in zip(scenario_costs, config_entries)
            if !isempty(cost)
                if entry.robust
                    push!(robust_costs, cost)
                    push!(robust_entries, entry)
                else
                    push!(non_robust_costs, cost)
                    push!(non_robust_entries, entry)
                end
            end
        end

        has_robust = !isempty(robust_costs)
        has_non_robust = !isempty(non_robust_costs)
        if !has_robust && !has_non_robust
            continue
        end

        all_valid_costs = vcat(robust_costs, non_robust_costs)

        num_rh_steps = maximum(length(c) for c in all_valid_costs)
        if num_rh_steps == 0
            continue
        end

        # Get all unique cost components
        all_components = Set{Symbol}()
        player_indices = Set{Int}()

        for trial_data in all_valid_costs
            for rh_step in 1:min(length(trial_data), num_rh_steps)
                rh_step_data = trial_data[rh_step]
                for (player_idx, plan_costs) in rh_step_data
                    push!(player_indices, player_idx)
                    for step in plan_costs
                        union!(all_components, keys(step))
                    end
                end
            end
        end

        component_names = sort(collect(all_components), by=string)
        player_indices = sort(collect(player_indices))

        if isempty(component_names) || isempty(player_indices)
            continue
        end

        # Create figure
        num_components = length(component_names)
        num_rh_steps_display = min(num_rh_steps, 8)
        num_cols = num_components + 1  # +1 for total
        num_rows = num_rh_steps_display + 1  # +1 for executed trajectory row

        fig = Figure(size=(450 * num_cols, 400 * num_rows + 80))
        Label(fig[0, 1:num_cols], text = "$config - Cost Components", fontsize = 20)

        # Legend: player color × robustness linestyle
        player_colors = Dict(1 => :blue, 2 => :red, 3 => :green)
        legend_elements = []
        legend_labels = String[]
        for p in player_indices
            color = get(player_colors, p, :gray)
            if has_robust
                push!(legend_elements, LineElement(color=color, linewidth=2, linestyle=:solid))
                push!(legend_labels, "P$p Robust")
            end
            if has_non_robust
                push!(legend_elements, LineElement(color=color, linewidth=2, linestyle=:dash))
                push!(legend_labels, "P$p Non-Robust")
            end
        end
        Legend(fig[0, num_cols], legend_elements, legend_labels, framevisible=false)

        colgap!(fig.layout, 10)
        rowgap!(fig.layout, 10)

        # Helper to compute stats
        function get_component_stats(data_list)
            if isempty(data_list)
                return Float64[], Float64[], Float64[]
            end
            max_len = maximum(length(d) for d in data_list)
            means = Float64[]
            stds = Float64[]
            for t in 1:max_len
                vals = [d[t] for d in data_list if length(d) >= t]
                if !isempty(vals)
                    push!(means, mean(vals))
                    push!(stds, length(vals) > 1 ? std(vals) : 0.0)
                else
                    push!(means, NaN)
                    push!(stds, NaN)
                end
            end
            return 1:length(means), means, stds
        end

        # Helper to plot a set of trials onto an axis
        function plot_trials!(ax, trial_costs_list, player_idx, component, color, rh_step; linestyle=:solid)
            trajectories = Vector{Vector{Float64}}()
            for trial_data in trial_costs_list
                if rh_step > length(trial_data)
                    continue
                end
                rh_step_data = trial_data[rh_step]
                if !haskey(rh_step_data, player_idx)
                    continue
                end
                plan_costs = rh_step_data[player_idx]
                if isempty(plan_costs)
                    continue
                end
                if isnothing(component)
                    traj = [sum(values(step)) for step in plan_costs]
                else
                    traj = [get(step, component, 0.0) for step in plan_costs]
                end
                push!(trajectories, traj)
            end

            if !isempty(trajectories)
                ts, means, stds = get_component_stats(trajectories)
                valid_idx = findall(.!isnan.(means))
                if !isempty(valid_idx)
                    band!(ax, ts[valid_idx], means[valid_idx] .- stds[valid_idx],
                          means[valid_idx] .+ stds[valid_idx], color=(color, 0.15))
                    lines!(ax, ts[valid_idx], means[valid_idx], color=color, linewidth=2, linestyle=linestyle)
                end
            end
        end

        for rh_step in 1:num_rh_steps_display
            for (comp_idx, component) in enumerate(component_names)
                ax = Axis(fig[rh_step, comp_idx],
                    title = rh_step == 1 ? string(component) : "",
                    xlabel = rh_step == num_rh_steps_display ? "Planning Step" : "",
                )
                if comp_idx == 1
                    ax.ylabel = "RH $rh_step"
                end

                for player_idx in player_indices
                    color = get(player_colors, player_idx, :gray)
                    if has_robust
                        plot_trials!(ax, robust_costs, player_idx, component, color, rh_step; linestyle=:solid)
                    end
                    if has_non_robust
                        plot_trials!(ax, non_robust_costs, player_idx, component, color, rh_step; linestyle=:dash)
                    end
                end
            end

            # Total cost column
            ax_total = Axis(fig[rh_step, num_cols],
                title = rh_step == 1 ? "Total" : "",
                xlabel = rh_step == num_rh_steps_display ? "Planning Step" : "",
            )

            for player_idx in player_indices
                color = get(player_colors, player_idx, :gray)
                if has_robust
                    plot_trials!(ax_total, robust_costs, player_idx, nothing, color, rh_step; linestyle=:solid)
                end
                if has_non_robust
                    plot_trials!(ax_total, non_robust_costs, player_idx, nothing, color, rh_step; linestyle=:dash)
                end
            end
        end

        # ===== Executed trajectory row (bottom) =====
        exec_row = num_rows

        ax_exec_total = Axis(fig[exec_row, 1:num_cols],
            title = "Executed Trajectory (Incurred Cost)",
            xlabel = "Execution Step",
            ylabel = "Executed",
        )

        # Helper to plot executed costs for a set of entries
        function plot_executed!(ax, entries, player_idx, color; linestyle=:solid)
            trajectories = Vector{Vector{Float64}}()
            for entry in entries
                if !isempty(entry.incurred_cost_history) && haskey(entry.incurred_cost_history, player_idx)
                    cost_hist = entry.incurred_cost_history[player_idx]
                    traj = Float64[]
                    for step in cost_hist
                        if step isa Tuple && length(step) >= 2
                            push!(traj, Float64(step[2]))
                        elseif step isa Number
                            push!(traj, Float64(step))
                        end
                    end
                    if !isempty(traj)
                        push!(trajectories, traj)
                    end
                end
            end
            if !isempty(trajectories)
                ts, means, stds = get_component_stats(trajectories)
                valid_idx = findall(.!isnan.(means))
                if !isempty(valid_idx)
                    band!(ax, ts[valid_idx], means[valid_idx] .- stds[valid_idx],
                          means[valid_idx] .+ stds[valid_idx], color=(color, 0.15))
                    lines!(ax, ts[valid_idx], means[valid_idx], color=color, linewidth=2, linestyle=linestyle)
                end
            end
        end

        for player_idx in player_indices
            color = get(player_colors, player_idx, :gray)
            if has_robust
                plot_executed!(ax_exec_total, robust_entries, player_idx, color; linestyle=:solid)
            end
            if has_non_robust
                plot_executed!(ax_exec_total, non_robust_entries, player_idx, color; linestyle=:dash)
            end
        end

        safe_config = replace(config, r"[^a-zA-Z0-9_]" => "_")
        filename = joinpath(directory, "senate_yarnball_$(safe_config).png")
        save(filename, fig)
        save(joinpath(directory, "senate_yarnball_$(safe_config).pdf"), fig)
        println("Saved yarnball plot to $filename")
    end

    return nothing
end

"""
    plot_senate_spatial_trajectories(; directory)

Plot the opinion-space trajectories from senate experiments.
"""
function plot_senate_spatial_trajectories(; directory="./exp/senate/outputs/analysis")
    if isempty(SENATE_TRAJECTORY_TRACKER.entries)
        println("No trajectory data loaded")
        return nothing
    end

    if !isdir(directory)
        mkpath(directory)
    end

    # Group by config
    config_groups = unique([e.config_name for e in SENATE_TRAJECTORY_TRACKER.entries])

    for config in config_groups
        entries = [e for e in SENATE_TRAJECTORY_TRACKER.entries if e.config_name == config]

        fig = Figure(size=(1000, 800))
        ax = Axis(fig[1, 1],
            title = "Senator Trajectories - $config",
            xlabel = "Opinion Dimension 1",
            ylabel = "Opinion Dimension 2"
        )

        senator_colors = [:blue, :green, :orange, :purple, :brown]

        for (entry_idx, entry) in enumerate(entries)
            alpha = 0.5 + 0.5 * (entry_idx / length(entries))
            linestyle = entry.robust ? :solid : :dash

            if !isempty(entry.gt_state_history)
                num_senators = length(entry.gt_state_history[1].blocks)

                for senator_idx in 1:num_senators
                    xs = [state.blocks[senator_idx][1] for state in entry.gt_state_history]
                    ys = [state.blocks[senator_idx][2] for state in entry.gt_state_history]

                    color = senator_colors[mod1(senator_idx, length(senator_colors))]
                    lines!(ax, xs, ys, color=(color, alpha), linestyle=linestyle, linewidth=1.5)

                    # Mark start
                    scatter!(ax, [xs[1]], [ys[1]], color=color, marker=:circle, markersize=8)
                    scatter!(ax, [xs[end]], [ys[end]], color=color, marker=:star5, markersize=10)
                end
            end
        end

        # Legend
        legend_elements = [
            [LineElement(color=c, linewidth=2) for c in senator_colors[1:3]]...,
            LineElement(color=:black, linestyle=:solid, linewidth=2),
            LineElement(color=:black, linestyle=:dash, linewidth=2),
        ]
        legend_labels = ["Senator 1", "Senator 2", "Senator 3", "Robust", "Non-Robust"]
        Legend(fig[1, 2], legend_elements, legend_labels, "Legend")

        safe_config = replace(config, r"[^a-zA-Z0-9_]" => "_")
        filename = joinpath(directory, "senate_trajectories_$(safe_config).png")
        save(filename, fig)
        println("Saved trajectory plot to $filename")
    end

    return nothing
end

# ========================================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================================

"""
    analyze_senate_trajectory_data(; directory, file_pattern, output_directory)

Main entry point for analyzing senate trajectory data.

# Example usage:
```julia
include("exp/SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis

# Analyze all .dat files in the merged directory
analyze_senate_trajectory_data(
    directory="./exp/senate/outputs/merged",
    output_directory="./exp/senate/outputs/analysis"
)

# Or analyze specific files
analyze_senate_trajectory_data(
    directory="./exp/senate/outputs/merged",
    file_pattern=r"obst_block",
    output_directory="./exp/senate/outputs/analysis"
)
```
"""
function analyze_senate_trajectory_data(;
    directory="./exp/senate/outputs/merged/drift_test",
    file_pattern=r"",
    output_directory="./exp/senate/outputs/analysis")

    println("\n=== ANALYZING SENATE TRAJECTORY DATA ===")

    if load_and_analyze_senate_solution_files(; directory=directory, file_pattern=file_pattern)
        get_senate_trajectory_summary(; directory=output_directory)
        plot_senate_spatial_trajectories(; directory=output_directory)
    else
        println("Failed to load trajectory data from solution files")
    end
end

end  # module
