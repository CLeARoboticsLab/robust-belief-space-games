module SenateTrajectoryAnalysis

using Serialization
using Statistics
using LinearAlgebra
using BlockArrays
using Printf
using CairoMakie
using Distributions
using Random
using RobustBeliefGame

using Senate

# ========================================================================================
# SENATE TRAJECTORY ANALYSIS TRACKER SYSTEM
# ========================================================================================

export SenateTrajectoryAnalysisEntry, SenateTrajectoryAnalysisTracker, SENATE_TRAJECTORY_TRACKER,
    clear_senate_trajectory_tracker!, load_and_analyze_senate_solution_files,
    compute_senate_belief_covariance_traces, compute_senator_distances,
    get_senate_trajectory_summary, create_senate_trajectory_analysis_plots,
    compare_robust_vs_nonrobust_senate_actions, create_senate_yarnball_plot,
    analyze_senate_trajectory_data, plot_senate_spatial_trajectories,
    analyze_nature_control_sweep, analyze_planning_horizon_sweep,
    create_merged_executed_costs_plot, create_sweep_summary_plot,
    compute_senate_significance_report,
    analyze_drift_mismatch_sweep, analyze_robustness_comparison_sweep,
    analyze_control_planning_sweep

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
    # Remove robustness labels so robust + non_robust group together
    # (non_robust must come before robust to avoid partial match)
    name = replace(name, r"_?p2_type_non_robust" => "")
    name = replace(name, r"_?p2_type_robust" => "")
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
# SHARED HELPER FUNCTIONS (used by multiple plotting functions)
# ========================================================================================

"""Filter player indices based on hide_p1 flag."""
filter_player_indices(indices; hide_p1::Bool=false) = hide_p1 ? filter(p -> p != 1, indices) : indices

"""Compute mean and std across a list of trajectories."""
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

"""Compute mean and std of control norms across multiple trials for a given player."""
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

"""Extract executed cost trajectories from entries for a specific player."""
function extract_executed_trajectories(entries, player_idx, tuple_index; cumulative=false)
    trajectories = Vector{Vector{Float64}}()
    for entry in entries
        if !isempty(entry.incurred_cost_history) && haskey(entry.incurred_cost_history, player_idx)
            cost_hist = entry.incurred_cost_history[player_idx]
            traj = Float64[]
            for step in cost_hist
                if step isa Tuple && length(step) >= tuple_index
                    push!(traj, Float64(step[tuple_index]))
                elseif step isa Number
                    push!(traj, Float64(step))
                end
            end
            if !isempty(traj)
                push!(trajectories, cumulative ? cumsum(traj) : traj)
            end
        end
    end
    return trajectories
end

# Color palette for sweep overlay plots (colorblind-friendly)
const SWEEP_COLORS = [:blue, :red, :green, :purple, :orange, :cyan, :magenta, :brown]

# ========================================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================================

"""
    get_senate_trajectory_summary(; directory)

Get summary statistics and create analysis plots for senate experiments.
"""
function get_senate_trajectory_summary(; directory="./exp/senate/outputs/analysis", hide_p1::Bool=false)
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

    # Merge config groups by base name (so robust + non_robust go into the same yarnball)
    merged_planned_costs = Dict{String, Any}()
    merged_config_entries = Dict{String, Vector{SenateTrajectoryAnalysisEntry}}()
    for (config, costs) in all_planned_costs
        base = base_senate_group(config)
        if !haskey(merged_planned_costs, base)
            merged_planned_costs[base] = Any[]
            merged_config_entries[base] = SenateTrajectoryAnalysisEntry[]
        end
        append!(merged_planned_costs[base], costs)
        append!(merged_config_entries[base], all_config_entries[config])
    end

    # Create comparison plots
    compare_robust_vs_nonrobust_senate_actions(SENATE_TRAJECTORY_TRACKER.entries; directory=directory, hide_p1=hide_p1)
    create_senate_yarnball_plot(merged_planned_costs, merged_config_entries; directory=directory, hide_p1=hide_p1)

    return all_planned_costs
end

"""
    compare_robust_vs_nonrobust_senate_actions(all_entries; directory)

Create comparison plots for robust vs non-robust senate experiments.
"""
function compare_robust_vs_nonrobust_senate_actions(all_entries; directory="./exp/senate/outputs/analysis", hide_p1::Bool=false)
    robust_entries = [e for e in all_entries if e.robust]
    non_robust_entries = [e for e in all_entries if !e.robust]

    if isempty(robust_entries) && isempty(non_robust_entries)
        println("No entries to compare")
        return
    end

    println("Comparing $(length(robust_entries)) robust vs $(length(non_robust_entries)) non-robust entries")

    create_senate_action_difference_plots(robust_entries, non_robust_entries; directory=directory, hide_p1=hide_p1)
end

"""
    create_senate_action_difference_plots(robust_entries, non_robust_entries; directory)

Create plots comparing executed controls between robust and non-robust strategies.
"""
function create_senate_action_difference_plots(robust_entries, non_robust_entries; directory="./exp/senate/outputs/analysis", hide_p1::Bool=false)
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

    if !hide_p1
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
function create_senate_yarnball_plot(all_planned_costs, all_config_entries::Dict{String, Vector{SenateTrajectoryAnalysisEntry}}; directory="./exp/senate/outputs/analysis", hide_p1::Bool=false)
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
        player_indices = filter_player_indices(sort(collect(player_indices)); hide_p1=hide_p1)

        if isempty(component_names) || isempty(player_indices)
            continue
        end

        # Create figure
        num_components = length(component_names)
        num_rh_steps_display = min(num_rh_steps, 8)
        num_cols = num_components + 1  # +1 for total
        num_rows = num_rh_steps_display

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

        safe_config = replace(config, r"[^a-zA-Z0-9_]" => "_")
        filename = joinpath(directory, "senate_yarnball_$(safe_config).png")
        save(filename, fig)
        save(joinpath(directory, "senate_yarnball_$(safe_config).pdf"), fig)
        println("Saved yarnball plot to $filename")

        # ===== Separate figure: Executed & Cumulative costs =====
        function plot_executed_trajs!(ax, entries, player_idx, color, tuple_index; linestyle=:solid, cumulative=false)
            trajectories = extract_executed_trajectories(entries, player_idx, tuple_index; cumulative=cumulative)
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

        fig_exec = Figure(size=(2400, 1600))
        Label(fig_exec[0, :], text = "$config - Executed Costs", fontsize = 20)
        Legend(fig_exec[0, 2], legend_elements, legend_labels, framevisible=false)

        exec_titles = [
            "Per-Step (Deterministic)",
            "Per-Step (Stochastic)",
            "Cumulative (Deterministic)",
            "Cumulative (Stochastic)",
        ]
        exec_params = [(2, false), (1, false), (2, true), (1, true)]

        for (row, (title, (tuple_idx, cumul))) in enumerate(zip(exec_titles, exec_params))
            ax = Axis(fig_exec[row, 1:2],
                title = title,
                xlabel = "Execution Step",
                ylabel = cumul ? "Cumulative Cost" : "Cost",
            )
            for player_idx in player_indices
                color = get(player_colors, player_idx, :gray)
                if has_robust
                    plot_executed_trajs!(ax, robust_entries, player_idx, color, tuple_idx; linestyle=:solid, cumulative=cumul)
                end
                if has_non_robust
                    plot_executed_trajs!(ax, non_robust_entries, player_idx, color, tuple_idx; linestyle=:dash, cumulative=cumul)
                end
            end
        end

        exec_filename = joinpath(directory, "senate_executed_$(safe_config).png")
        save(exec_filename, fig_exec)
        save(joinpath(directory, "senate_executed_$(safe_config).pdf"), fig_exec)
        println("Saved executed costs plot to $exec_filename")
    end

    return nothing
end

"""
    plot_senate_spatial_trajectories(; directory)

Plot the opinion-space trajectories from senate experiments.
"""
function plot_senate_spatial_trajectories(; directory="./exp/senate/outputs/analysis", hide_p1::Bool=false)
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
    output_directory="./exp/senate/outputs/analysis",
    hide_p1::Bool=false)

    println("\n=== ANALYZING SENATE TRAJECTORY DATA ===")

    if load_and_analyze_senate_solution_files(; directory=directory, file_pattern=file_pattern)
        # Run significance analysis before plotting (this also filters outliers from the tracker)
        r_entries = [e for e in SENATE_TRAJECTORY_TRACKER.entries if e.robust]
        nr_entries = [e for e in SENATE_TRAJECTORY_TRACKER.entries if !e.robust]
        if !isempty(r_entries) && !isempty(nr_entries)
            compute_senate_significance_report(r_entries, nr_entries; directory=output_directory)
        end

        get_senate_trajectory_summary(; directory=output_directory, hide_p1=hide_p1)
        plot_senate_spatial_trajectories(; directory=output_directory, hide_p1=hide_p1)
    else
        println("Failed to load trajectory data from solution files")
    end
end

"""
    analyze_nature_control_sweep(; multipliers, directory, output_base)

Analyze nature control sweep results, split by nature multiplier value.
Each multiplier gets its own output subfolder.
"""
function analyze_nature_control_sweep(;
    multipliers=[1, 5, 25, 125, 625, 3125],
    directory="./exp/senate/outputs/merged/nature_control_sweep",
    output_base="./exp/senate/outputs/analysis/nature_control_sweep",
    hide_p1::Bool=false)

    sweep_collected = Dict{Any, NamedTuple}()

    for m in multipliers
        println("\n===== Analyzing nature_multiplier=$m =====")
        analyze_senate_trajectory_data(
            directory=directory,
            file_pattern=Regex("p2_nature_multiplier_$(m)_p2_type"),
            output_directory=joinpath(output_base, "multiplier_$(m)"),
            hide_p1=hide_p1
        )
        # Snapshot entries before next iteration clears them
        entries = copy(SENATE_TRAJECTORY_TRACKER.entries)
        sweep_collected[m] = (
            robust_entries = [e for e in entries if e.robust],
            non_robust_entries = [e for e in entries if !e.robust]
        )
    end

    # Create merged overlay plots
    merged_dir = joinpath(output_base, "merged")
    mkpath(merged_dir)
    create_merged_executed_costs_plot(sweep_collected, "Nature Multiplier";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="nature_multiplier")
    create_sweep_summary_plot(sweep_collected, "Nature Multiplier";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="nature_multiplier", xscale=log10)
end

"""
    analyze_planning_horizon_sweep(; horizons, directory, output_base)

Analyze planning horizon sweep results, split by planning horizon value.
"""
function analyze_planning_horizon_sweep(;
    horizons=[2, 5],
    directory="./exp/senate/outputs/merged/planning_horizon_sweep",
    output_base="./exp/senate/outputs/analysis/planning_horizon_sweep",
    hide_p1::Bool=false)

    sweep_collected = Dict{Any, NamedTuple}()

    for h in horizons
        println("\n===== Analyzing planning_horizon=$h =====")
        analyze_senate_trajectory_data(
            directory=directory,
            file_pattern=Regex("planning_horizon_$(h)_mass_results"),
            output_directory=joinpath(output_base, "horizon_$(h)"),
            hide_p1=hide_p1
        )
        # Snapshot entries before next iteration clears them
        entries = copy(SENATE_TRAJECTORY_TRACKER.entries)
        sweep_collected[h] = (
            robust_entries = [e for e in entries if e.robust],
            non_robust_entries = [e for e in entries if !e.robust]
        )
    end

    # Create merged overlay plots
    merged_dir = joinpath(output_base, "merged")
    mkpath(merged_dir)
    create_merged_executed_costs_plot(sweep_collected, "Planning Horizon";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="planning_horizon")
    create_sweep_summary_plot(sweep_collected, "Planning Horizon";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="planning_horizon")
end

"""
    analyze_drift_mismatch_sweep(; gt_values, directory, output_base, hide_p1)

Analyze drift mismatch sweep results, split by ground-truth drift scale.
Each gt value has its own subdirectory (gt_1.0, gt_2.0, etc.).
"""
function analyze_drift_mismatch_sweep(;
    gt_values=[1.0, 2.0, 4.0],
    directory="./exp/senate/outputs/merged/drift_mismatch_sweep",
    output_base="./exp/senate/outputs/analysis/drift_mismatch_sweep",
    hide_p1::Bool=false)

    sweep_collected = Dict{Any, NamedTuple}()

    for gt in gt_values
        println("\n===== Analyzing gt_drift=$gt =====")
        analyze_senate_trajectory_data(
            directory=joinpath(directory, "gt_$(gt)"),
            output_directory=joinpath(output_base, "gt_$(gt)"),
            hide_p1=hide_p1
        )
        entries = copy(SENATE_TRAJECTORY_TRACKER.entries)
        sweep_collected[gt] = (
            robust_entries = [e for e in entries if e.robust],
            non_robust_entries = [e for e in entries if !e.robust]
        )
    end

    merged_dir = joinpath(output_base, "merged")
    mkpath(merged_dir)
    create_merged_executed_costs_plot(sweep_collected, "GT Drift Scale";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="drift_mismatch")
    create_sweep_summary_plot(sweep_collected, "GT Drift Scale";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="drift_mismatch")
end

"""
    analyze_robustness_comparison_sweep(; p1_types, directory, output_base, hide_p1)

Analyze robustness comparison sweep results, split by P1 type.
For each P1 type, compares robust P2 vs non-robust P2.
"""
function analyze_robustness_comparison_sweep(;
    p1_types=["robust", "non_robust"],
    directory="./exp/senate/outputs/merged/robustness_comparison_sweep",
    output_base="./exp/senate/outputs/analysis/robustness_comparison_sweep",
    hide_p1::Bool=false)

    sweep_collected = Dict{Any, NamedTuple}()

    for p1t in p1_types
        println("\n===== Analyzing p1_type=$p1t =====")
        analyze_senate_trajectory_data(
            directory=directory,
            file_pattern=Regex("p1_type_$(p1t)_"),
            output_directory=joinpath(output_base, "p1_$(p1t)"),
            hide_p1=hide_p1
        )
        entries = copy(SENATE_TRAJECTORY_TRACKER.entries)
        sweep_collected[p1t] = (
            robust_entries = [e for e in entries if e.robust],
            non_robust_entries = [e for e in entries if !e.robust]
        )
    end

    merged_dir = joinpath(output_base, "merged")
    mkpath(merged_dir)
    create_merged_executed_costs_plot(sweep_collected, "P1 Type";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="robustness_comparison")
    create_sweep_summary_plot(sweep_collected, "P1 Type";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="robustness_comparison")
end

"""
    analyze_control_planning_sweep(; horizons, directory, output_base, hide_p1)

Analyze control+planning sweep results, split by planning horizon value.
"""
function analyze_control_planning_sweep(;
    horizons=[2, 5, 8, 12],
    directory="./exp/senate/outputs/merged/control_planning_sweep",
    output_base="./exp/senate/outputs/analysis/control_planning_sweep",
    hide_p1::Bool=false)

    sweep_collected = Dict{Any, NamedTuple}()

    for h in horizons
        println("\n===== Analyzing planning_horizon=$h =====")
        analyze_senate_trajectory_data(
            directory=directory,
            file_pattern=Regex("planning_horizon_$(h)_mass_results"),
            output_directory=joinpath(output_base, "horizon_$(h)"),
            hide_p1=hide_p1
        )
        entries = copy(SENATE_TRAJECTORY_TRACKER.entries)
        sweep_collected[h] = (
            robust_entries = [e for e in entries if e.robust],
            non_robust_entries = [e for e in entries if !e.robust]
        )
    end

    merged_dir = joinpath(output_base, "merged")
    mkpath(merged_dir)
    create_merged_executed_costs_plot(sweep_collected, "Planning Horizon";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="control_planning")
    create_sweep_summary_plot(sweep_collected, "Planning Horizon";
        directory=merged_dir, hide_p1=hide_p1, sweep_name="control_planning")
end

# ========================================================================================
# MERGED SWEEP OVERLAY PLOTS
# ========================================================================================

"""
    create_merged_executed_costs_plot(sweep_collected, sweep_label; directory, hide_p1, sweep_name)

Create overlay plot of executed costs across all sweep values.
Color encodes sweep value, linestyle encodes robust vs non-robust.

`sweep_collected`: Dict{Any, NamedTuple} mapping sweep_value => (robust_entries=..., non_robust_entries=...)
`sweep_label`: Human-readable label for the sweep variable (e.g. "Nature Multiplier")
"""
function create_merged_executed_costs_plot(
    sweep_collected::Dict,
    sweep_label::String;
    directory::String="./exp/senate/outputs/analysis",
    hide_p1::Bool=false,
    sweep_name::String="sweep"
)
    if isempty(sweep_collected)
        println("No sweep data to plot")
        return
    end

    mkpath(directory)
    sweep_values = sort(collect(keys(sweep_collected)))

    # Determine player indices from the data
    all_player_indices = Set{Int}()
    for (_, data) in sweep_collected
        for entry in vcat(data.robust_entries, data.non_robust_entries)
            if !isempty(entry.incurred_cost_history)
                union!(all_player_indices, keys(entry.incurred_cost_history))
            end
        end
    end
    player_indices = filter_player_indices(sort(collect(all_player_indices)); hide_p1=hide_p1)

    if isempty(player_indices)
        println("No player data available for merged plot")
        return
    end

    player_colors = Dict(1 => :blue, 2 => :red, 3 => :green)

    # 2 rows per player: per-step and cumulative deterministic
    num_players = length(player_indices)
    fig = Figure(size=(1400, 500 * num_players))
    Label(fig[0, :], text = "Merged Executed Costs by $sweep_label", fontsize = 20)

    plot_configs = [
        ("Per-Step Deterministic", 2, false),
        ("Cumulative Deterministic", 2, true),
    ]

    for (p_idx, player_idx) in enumerate(player_indices)
        for (col, (title, tuple_idx, cumul)) in enumerate(plot_configs)
            ax = Axis(fig[p_idx, col],
                title = p_idx == 1 ? title : "",
                xlabel = "Execution Step",
                ylabel = cumul ? "Cumulative Cost" : "Cost",
            )
            if col == 1
                ax.ylabel = "P$(player_idx) - " * (cumul ? "Cumulative Cost" : "Cost")
            end

            for (sv_idx, sv) in enumerate(sweep_values)
                color = SWEEP_COLORS[mod1(sv_idx, length(SWEEP_COLORS))]
                data = sweep_collected[sv]

                # Robust entries
                if !isempty(data.robust_entries)
                    trajectories = extract_executed_trajectories(data.robust_entries, player_idx, tuple_idx; cumulative=cumul)
                    if !isempty(trajectories)
                        ts, means, stds = get_component_stats(trajectories)
                        valid_idx = findall(.!isnan.(means))
                        if !isempty(valid_idx)
                            lines!(ax, ts[valid_idx], means[valid_idx], color=color, linewidth=2, linestyle=:solid,
                                label="$sv (R)")
                        end
                    end
                end

                # Non-robust entries
                if !isempty(data.non_robust_entries)
                    trajectories = extract_executed_trajectories(data.non_robust_entries, player_idx, tuple_idx; cumulative=cumul)
                    if !isempty(trajectories)
                        ts, means, stds = get_component_stats(trajectories)
                        valid_idx = findall(.!isnan.(means))
                        if !isempty(valid_idx)
                            lines!(ax, ts[valid_idx], means[valid_idx], color=color, linewidth=2, linestyle=:dash,
                                label="$sv (NR)")
                        end
                    end
                end
            end
        end
    end

    # Build legend: color per sweep value + linestyle for robust/non-robust
    legend_elements = []
    legend_labels = String[]
    for (sv_idx, sv) in enumerate(sweep_values)
        color = SWEEP_COLORS[mod1(sv_idx, length(SWEEP_COLORS))]
        push!(legend_elements, LineElement(color=color, linewidth=2))
        push!(legend_labels, "$sweep_label = $sv")
    end
    push!(legend_elements, LineElement(color=:black, linewidth=2, linestyle=:solid))
    push!(legend_labels, "Robust")
    push!(legend_elements, LineElement(color=:black, linewidth=2, linestyle=:dash))
    push!(legend_labels, "Non-Robust")
    Legend(fig[:, end+1], legend_elements, legend_labels, framevisible=true)

    filename = joinpath(directory, "merged_executed_costs_$(sweep_name).png")
    save(filename, fig)
    save(joinpath(directory, "merged_executed_costs_$(sweep_name).pdf"), fig)
    println("Saved merged executed costs plot to $filename")
end

"""
    create_sweep_summary_plot(sweep_collected, sweep_label; directory, hide_p1, sweep_name, xscale)

Create summary plot: x = sweep value, y = mean final cumulative deterministic cost.
One line per player x robustness combination, with error bars for std across trials.
"""
function create_sweep_summary_plot(
    sweep_collected::Dict,
    sweep_label::String;
    directory::String="./exp/senate/outputs/analysis",
    hide_p1::Bool=false,
    sweep_name::String="sweep",
    xscale=identity
)
    if isempty(sweep_collected)
        println("No sweep data for summary plot")
        return
    end

    mkpath(directory)
    sweep_values = sort(collect(keys(sweep_collected)))
    xs = Float64.(sweep_values)

    # Determine player indices
    all_player_indices = Set{Int}()
    for (_, data) in sweep_collected
        for entry in vcat(data.robust_entries, data.non_robust_entries)
            if !isempty(entry.incurred_cost_history)
                union!(all_player_indices, keys(entry.incurred_cost_history))
            end
        end
    end
    player_indices = filter_player_indices(sort(collect(all_player_indices)); hide_p1=hide_p1)

    if isempty(player_indices)
        return
    end

    player_colors = Dict(1 => :blue, 2 => :red, 3 => :green)

    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1],
        title = "Final Cumulative Cost vs $sweep_label",
        xlabel = sweep_label,
        ylabel = "Mean Final Cumulative Cost",
        xscale = xscale
    )

    for player_idx in player_indices
        color = get(player_colors, player_idx, :gray)

        for (robustness_label, linestyle, entry_key) in [("Robust", :solid, :robust_entries), ("Non-Robust", :dash, :non_robust_entries)]
            means_y = Float64[]
            stds_y = Float64[]
            valid_xs = Float64[]

            for (sv_idx, sv) in enumerate(sweep_values)
                data = sweep_collected[sv]
                entries = getfield(data, entry_key)
                if isempty(entries)
                    continue
                end

                # Get final cumulative deterministic cost for each trial
                trajectories = extract_executed_trajectories(entries, player_idx, 2; cumulative=true)
                final_costs = [traj[end] for traj in trajectories if !isempty(traj)]

                if !isempty(final_costs)
                    push!(valid_xs, Float64(sv))
                    push!(means_y, mean(final_costs))
                    push!(stds_y, length(final_costs) > 1 ? std(final_costs) : 0.0)
                end
            end

            if !isempty(valid_xs)
                errorbars!(ax, valid_xs, means_y, stds_y, color=(color, 0.4))
                scatterlines!(ax, valid_xs, means_y, color=color, linewidth=2, linestyle=linestyle,
                    markersize=8, label="P$(player_idx) $robustness_label")
            end
        end
    end

    axislegend(ax, position=:lt)

    filename = joinpath(directory, "sweep_summary_$(sweep_name).png")
    save(filename, fig)
    save(joinpath(directory, "sweep_summary_$(sweep_name).pdf"), fig)
    println("Saved sweep summary plot to $filename")
end

# ========================================================================================
# STATISTICAL SIGNIFICANCE ANALYSIS
# ========================================================================================

"""
    compute_senate_significance_report(robust_entries, non_robust_entries; player_idx, directory, label)

Compute statistical significance of cost differences between robust and non-robust P2.
Performs IQR-based outlier removal, generates Q-Q plots, and runs Welch's t-test,
Mann-Whitney U test, and Bootstrap test. Writes results to `significance_report.txt`.

Returns the set of scenario names that were kept after outlier removal (for downstream filtering).
"""
function compute_senate_significance_report(
    robust_entries::Vector{SenateTrajectoryAnalysisEntry},
    non_robust_entries::Vector{SenateTrajectoryAnalysisEntry};
    player_idx::Int=2,
    directory::String="./exp/senate/outputs/analysis",
    label::String="P2"
)
    mkpath(directory)

    # --- Sub-step A: Extract per-trial scalar costs ---
    function get_final_costs(entries, pidx)
        costs = Tuple{Float64, String}[]
        for entry in entries
            trajs = extract_executed_trajectories([entry], pidx, 2; cumulative=true)
            if !isempty(trajs) && !isempty(trajs[1])
                push!(costs, (trajs[1][end], entry.scenario_name))
            end
        end
        return costs
    end

    robust_cost_entries = get_final_costs(robust_entries, player_idx)
    non_robust_cost_entries = get_final_costs(non_robust_entries, player_idx)

    if length(robust_cost_entries) < 2 || length(non_robust_cost_entries) < 2
        println("Insufficient data for significance analysis (robust=$(length(robust_cost_entries)), non-robust=$(length(non_robust_cost_entries)))")
        return Set{String}()
    end

    # --- Sub-step B: IQR-based outlier removal ---
    function remove_iqr_outliers(cost_entries)
        costs = [e[1] for e in cost_entries]
        q1 = quantile(costs, 0.25)
        q3 = quantile(costs, 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = [e for e in cost_entries if lower <= e[1] <= upper]
        n_removed = length(cost_entries) - length(filtered)
        return filtered, n_removed
    end

    filtered_r_entries, n_r_removed = remove_iqr_outliers(robust_cost_entries)
    filtered_nr_entries, n_nr_removed = remove_iqr_outliers(non_robust_cost_entries)

    println("Outlier removal (IQR): Robust removed $n_r_removed/$(length(robust_cost_entries)), Non-Robust removed $n_nr_removed/$(length(non_robust_cost_entries))")

    robust_costs = [e[1] for e in filtered_r_entries]
    non_robust_costs = [e[1] for e in filtered_nr_entries]

    # Collect kept scenario names
    kept_names = Set{String}()
    for e in filtered_r_entries
        push!(kept_names, e[2])
    end
    for e in filtered_nr_entries
        push!(kept_names, e[2])
    end

    # --- Sub-step C: Q-Q plot ---
    if length(robust_costs) > 1 && length(non_robust_costs) > 1
        fig_qq = Figure(size=(1000, 500))
        Label(fig_qq[0, :], text="Q-Q Plots: Normality Assessment ($label, Outliers removed: R=$n_r_removed, NR=$n_nr_removed)", fontsize=16)

        ax_qq = Axis(fig_qq[1, 1],
            title="Cost Distribution (n_R=$(length(robust_costs)), n_NR=$(length(non_robust_costs)))",
            xlabel="Theoretical Quantiles",
            ylabel="Sample Quantiles")

        # Robust Q-Q points
        if std(robust_costs) > 0
            n_r = length(robust_costs)
            sorted_r = sort(robust_costs)
            theoretical_q_r = [quantile(Normal(0, 1), (i - 0.5) / n_r) for i in 1:n_r]
            standardized_r = (sorted_r .- mean(robust_costs)) ./ std(robust_costs)
            scatter!(ax_qq, theoretical_q_r, standardized_r, color=:blue, markersize=8, label="Robust")
        end

        # Non-robust Q-Q points
        if std(non_robust_costs) > 0
            n_nr = length(non_robust_costs)
            sorted_nr = sort(non_robust_costs)
            theoretical_q_nr = [quantile(Normal(0, 1), (i - 0.5) / n_nr) for i in 1:n_nr]
            standardized_nr = (sorted_nr .- mean(non_robust_costs)) ./ std(non_robust_costs)
            scatter!(ax_qq, theoretical_q_nr, standardized_nr, color=:red, markersize=8, label="Non-Robust")
        end

        lines!(ax_qq, [-3, 3], [-3, 3], color=:black, linestyle=:dash, linewidth=2)
        axislegend(ax_qq, position=:lt)

        qq_path = joinpath(directory, "qq_plots.png")
        save(qq_path, fig_qq)
        save(joinpath(directory, "qq_plots.pdf"), fig_qq)
        println("Saved Q-Q plot to $qq_path")
    end

    # --- Sub-step D: Statistical significance tests ---
    n_r = length(robust_costs)
    n_nr = length(non_robust_costs)

    open(joinpath(directory, "significance_report.txt"), "w") do io
        println(io, "Statistical Significance Report ($label Final Cumulative Deterministic Cost)")
        println(io, "="^70 * "\n")

        if n_r > 1 && n_nr > 1
            mean_r = mean(robust_costs)
            std_r = std(robust_costs)
            mean_nr = mean(non_robust_costs)
            std_nr = std(non_robust_costs)

            println(io, "DESCRIPTIVE STATISTICS")
            println(io, "-"^50)
            println(io, "Outliers removed (IQR): Robust=$n_r_removed, Non-Robust=$n_nr_removed")
            println(io, "Robust (n=$n_r):     Mean = $(round(mean_r, digits=4)), Std = $(round(std_r, digits=4))")
            println(io, "Non-Robust (n=$n_nr): Mean = $(round(mean_nr, digits=4)), Std = $(round(std_nr, digits=4))")
            println(io, "Difference (Robust - Non-Robust): $(round(mean_r - mean_nr, digits=4))\n")

            # Welch's t-test
            println(io, "WELCH'S T-TEST")
            println(io, "-"^50)
            se_diff = sqrt((std_r^2 / n_r) + (std_nr^2 / n_nr))
            t_stat = (mean_r - mean_nr) / se_diff

            df_num = ((std_r^2 / n_r) + (std_nr^2 / n_nr))^2
            df_den = ((std_r^2 / n_r)^2 / (n_r - 1)) + ((std_nr^2 / n_nr)^2 / (n_nr - 1))
            df = df_num / df_den

            p_val_t = 2 * (1 - cdf(TDist(df), abs(t_stat)))

            println(io, "T-Statistic: $(round(t_stat, digits=4))")
            println(io, "Degrees of Freedom: $(round(df, digits=2))")
            println(io, "P-Value: $(round(p_val_t, digits=5))")
            println(io, "Significant (p < 0.05): $(p_val_t < 0.05 ? "YES" : "NO")\n")

            # Mann-Whitney U test
            println(io, "MANN-WHITNEY U TEST")
            println(io, "-"^50)
            combined = vcat(robust_costs, non_robust_costs)
            ranks = sortperm(sortperm(combined))
            R_r = sum(ranks[1:n_r])
            U_r = R_r - n_r * (n_r + 1) / 2
            U_nr = n_r * n_nr - U_r
            U = min(U_r, U_nr)
            mu_U = n_r * n_nr / 2
            sigma_U = sqrt(n_r * n_nr * (n_r + n_nr + 1) / 12)
            z_score = (U - mu_U) / sigma_U
            p_val_mw = 2 * (1 - cdf(Normal(0, 1), abs(z_score)))

            println(io, "U-Statistic: $(round(U, digits=2))")
            println(io, "Z-Score: $(round(z_score, digits=4))")
            println(io, "P-Value: $(round(p_val_mw, digits=5))")
            println(io, "Significant (p < 0.05): $(p_val_mw < 0.05 ? "YES" : "NO")\n")

            # Bootstrap test
            println(io, "BOOTSTRAP TEST (Resampling)")
            println(io, "-"^50)

            n_bootstrap = 10000
            observed_diff = mean_r - mean_nr
            bootstrap_diffs = Float64[]

            Random.seed!(42)
            for _ in 1:n_bootstrap
                boot_r = [robust_costs[rand(1:n_r)] for _ in 1:n_r]
                boot_nr = [non_robust_costs[rand(1:n_nr)] for _ in 1:n_nr]
                push!(bootstrap_diffs, mean(boot_r) - mean(boot_nr))
            end

            ci_lower = quantile(bootstrap_diffs, 0.025)
            ci_upper = quantile(bootstrap_diffs, 0.975)

            # Permutation p-value under null
            pooled = vcat(robust_costs, non_robust_costs)
            null_diffs = Float64[]
            for _ in 1:n_bootstrap
                boot_sample = [pooled[rand(1:length(pooled))] for _ in 1:length(pooled)]
                boot_r = boot_sample[1:n_r]
                boot_nr = boot_sample[n_r+1:end]
                push!(null_diffs, mean(boot_r) - mean(boot_nr))
            end

            p_val_boot = sum(abs.(null_diffs) .>= abs(observed_diff)) / n_bootstrap

            println(io, "Bootstrap Iterations: $n_bootstrap")
            println(io, "Observed Difference: $(round(observed_diff, digits=4))")
            println(io, "95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
            println(io, "P-Value: $(round(p_val_boot, digits=5))")
            println(io, "Significant (p < 0.05): $(p_val_boot < 0.05 ? "YES" : "NO")\n")
        else
            println(io, "Insufficient data for statistical tests (n_robust=$n_r, n_non_robust=$n_nr).")
            println(io, "Need at least 2 samples each.")
        end
    end
    println("Saved significance report to $(joinpath(directory, "significance_report.txt"))")

    # --- Sub-step E: Filter tracker entries to kept scenarios ---
    if !isempty(kept_names)
        n_before = length(SENATE_TRAJECTORY_TRACKER.entries)
        filter!(e -> e.scenario_name in kept_names, SENATE_TRAJECTORY_TRACKER.entries)
        n_after = length(SENATE_TRAJECTORY_TRACKER.entries)
        if n_before != n_after
            println("Filtered SENATE_TRAJECTORY_TRACKER: $n_before -> $n_after entries (removed $(n_before - n_after) outlier trials)")
        end
    end

    return kept_names
end

end  # module
