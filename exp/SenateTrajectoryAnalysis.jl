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
    analyze_nature_control_sweep, analyze_rvr_vs_rnr_overlay, analyze_planning_horizon_sweep,
    analyze_rvr_vs_rnr_cost_components,
    create_merged_executed_costs_plot, create_sweep_summary_plot,
    compute_senate_significance_report,
    analyze_drift_mismatch_sweep, analyze_robustness_comparison_sweep,
    analyze_control_planning_sweep,
    create_sweep_violin_plot,
    create_sweep_mean_trajectory_plot,
    create_sweep_p2_control_cost_plot,
    extract_nature_diagnostics_for_sweep, plot_nature_diagnostics_sweep,
    build_seed_matched_pairs, compute_trajectory_divergence, compute_divergence_statistics,
    compute_paired_cost_decomposition,
    create_seed_matched_trajectory_plot, create_trajectory_divergence_plot,
    create_cost_decomposition_comparison_plot, create_paired_scatter_plot,
    create_sweep_paired_analysis, write_paired_analysis_report

"""
    SenateTrajectoryAnalysisEntry

Stores trajectory analysis data for a single senate scenario execution.
"""
struct SenateTrajectoryAnalysisEntry
    scenario_name::String
    trial_number::Int
    random_seed::Int

    # Raw trajectory data from solution file
    gt_state_history::Any
    observation_history::Any
    solution_history::Any  # Dict{Int, Vector{NamedTuple}} - indexed by player_idx
    cost_history::Any
    incurred_cost_history::Any
    params::Union{SenateParams, Nothing}

    # Nature diagnostics (from robust solves)
    nature_diagnostics_history::Any  # Dict{Int, Vector} or nothing
    belief_error_history::Any  # Vector{Vector{Float64}} or nothing

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
    nature_diagnostics_history = Dict{Int, Any}()
    belief_error_history = nothing

    for (player_idx, player_data) in solutions_dict
        if player_data isa NamedTuple || (player_data isa Dict && haskey(player_data, :gt_state_history))
            gt_state_history = get(player_data, :gt_state_history, nothing)
            observation_history = get(player_data, :observation_history, nothing)
            solution_history[player_idx] = get(player_data, :solution_history, [])
            cost_history[player_idx] = get(player_data, :cost_history, [])
            incurred_cost_history[player_idx] = get(player_data, :incurred_cost_history, [])
            nature_diagnostics_history[player_idx] = get(player_data, :nature_diagnostics_history, nothing)
            # belief_error_history is shared across players (same for all)
            be = get(player_data, :belief_error_history, nothing)
            if !isnothing(be)
                belief_error_history = be
            end
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

    # Extract random seed (fallback chain)
    seed = -1
    if !isnothing(params) && hasproperty(params, :random_seed)
        seed = params.random_seed
    elseif !isnothing(combo_params) && combo_params isa Dict && haskey(combo_params, :random_seed)
        seed = combo_params[:random_seed]
    else
        seed_match = match(r"seed_(\d+)", scenario_name)
        if !isnothing(seed_match)
            seed = parse(Int, seed_match.captures[1])
        end
    end

    entry = SenateTrajectoryAnalysisEntry(
        scenario_name,
        trial_num,
        seed,
        gt_state_history,
        observation_history,
        solution_history,
        cost_history,
        incurred_cost_history,
        params,
        nature_diagnostics_history,
        belief_error_history,
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

    # Compute obstacle cost if applicable (matches obstacle_cost_v4 at zero covariance)
    obstacle_cost = 0.0
    if !isempty(config.obstacle_centers) && config.obstacle_weights[1] > 0
        for belief_mean in belief_blocks
            for (center, weight, sigmoid_scale, sigmoid_offset) in zip(
                config.obstacle_centers, config.obstacle_weights,
                config.obstacle_sigmoid_scales, config.obstacle_sigmoid_offsets)
                dist = norm(belief_mean - center)
                obstacle_cost += weight / (1 + exp(sigmoid_scale * (dist - sigmoid_offset)))
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
        # Truncate to avoid Windows MAX_PATH (260 char) limit
        max_name_len = 260 - length(directory) - 20
        if length(safe_config) > max_name_len
            safe_config = safe_config[1:max_name_len]
        end
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
        # Truncate to avoid Windows MAX_PATH (260 char) limit
        max_name_len = 260 - length(directory) - 20  # leave room for path separators + extension
        if length(safe_config) > max_name_len
            safe_config = safe_config[1:max_name_len]
        end
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
            try
                compute_senate_significance_report(r_entries, nr_entries; directory=output_directory)
            catch e
                println("Warning: significance report failed: $e")
            end
        end

        try
            get_senate_trajectory_summary(; directory=output_directory, hide_p1=hide_p1)
        catch e
            println("Warning: trajectory summary plots failed: $e")
        end
        try
            plot_senate_spatial_trajectories(; directory=output_directory, hide_p1=hide_p1)
        catch e
            println("Warning: spatial trajectory plots failed: $e")
        end
    else
        println("Failed to load trajectory data from solution files")
    end
end

"""
    analyze_nature_control_sweep(; multipliers, directory, output_base, no_drift=false)

Analyze nature control sweep results, split by nature multiplier value.
Each multiplier gets its own output subfolder.

If `no_drift=true`, defaults change to use the `nature_control_sweep_no_drift` data.
"""
function analyze_nature_control_sweep(;
    no_drift::Bool=false,
    multipliers=[1, 2, 5, 10, 25, 50, 125, 250, 625, 1250, 3125, 6250],
    directory=no_drift ? "./exp/senate/outputs/merged/nature_control_sweep_no_drift" : "./exp/senate/outputs/merged/nature_control_sweep",
    output_base=no_drift ? "./exp/senate/outputs/analysis/nature_control_sweep_no_drift" : "./exp/senate/outputs/analysis/nature_control_sweep",
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
    for (plot_name, plot_fn) in [
        ("merged executed costs", () -> create_merged_executed_costs_plot(sweep_collected, "Nature Multiplier";
            directory=merged_dir, hide_p1=hide_p1, sweep_name="nature_multiplier")),
        ("sweep summary", () -> create_sweep_summary_plot(sweep_collected, "Nature Multiplier";
            directory=merged_dir, hide_p1=hide_p1, sweep_name="nature_multiplier", xscale=log10)),
        ("sweep violin", () -> create_sweep_violin_plot(sweep_collected, "Nature Multiplier";
            directory=merged_dir, sweep_name="nature_multiplier")),
        ("nature diagnostics", () -> begin
            diagnostics = extract_nature_diagnostics_for_sweep(sweep_collected)
            if !isempty(diagnostics)
                plot_nature_diagnostics_sweep(diagnostics, sweep_collected;
                    directory=merged_dir, sweep_name="nature_multiplier")
            end
        end),
        ("paired trajectory analysis", () -> create_sweep_paired_analysis(sweep_collected, "Nature Multiplier";
            directory=merged_dir, sweep_name="nature_multiplier")),
        ("mean trajectories", () -> create_sweep_mean_trajectory_plot(sweep_collected, "Nature Multiplier";
            directory=merged_dir, sweep_name="nature_multiplier")),
        ("p2 control cost", () -> create_sweep_p2_control_cost_plot(sweep_collected, "Nature Multiplier";
            directory=merged_dir, sweep_name="nature_multiplier")),
    ]
        try
            plot_fn()
        catch e
            println("Warning: $plot_name failed: $e")
        end
    end
end

"""
    analyze_rvr_vs_rnr_overlay(; multipliers, rvr_directory, rnr_directory, output_directory, hide_p1, xscale)

Overlay R-vs-R nature-control sweep against the existing R-vs-NR sweep on the
same multiplier axis. Three series per player: R-vs-R, R-vs-NR with P2=robust,
R-vs-NR with P2=non-robust. Demonstrates that any robustness advantage is not
purely a product of asymmetry between the two players' types.
"""
function analyze_rvr_vs_rnr_overlay(;
    multipliers=[1, 2, 5, 10, 25, 50, 125, 250, 625, 1250, 3125, 6250],
    rvr_directory="./exp/senate/outputs/merged/rvr_nature_control_sweep",
    rnr_directory="./exp/senate/outputs/merged/nature_control_sweep",
    output_directory="./exp/senate/outputs/analysis/rvr_vs_rnr_overlay",
    hide_p1::Bool=false,
    xscale=log10,
    preloaded=nothing,
)
    mkpath(output_directory)

    if isnothing(preloaded)
        rvr, rnr_r, rnr_nr = _load_rvr_rnr_series(multipliers;
            rvr_directory=rvr_directory, rnr_directory=rnr_directory)
    else
        rvr, rnr_r, rnr_nr = preloaded
    end

    all_player_indices = Set{Int}()
    for d in (rvr, rnr_r, rnr_nr), entries in values(d), entry in entries
        if !isempty(entry.incurred_cost_history)
            union!(all_player_indices, keys(entry.incurred_cost_history))
        end
    end
    player_indices = filter_player_indices(sort(collect(all_player_indices)); hide_p1=hide_p1)
    if isempty(player_indices)
        println("No player data available for overlay plot")
        return
    end

    series_specs = [
        ("R-vs-R", rvr, :solid),
        ("R-vs-NR (P2=Robust)", rnr_r, :dash),
        ("R-vs-NR (P2=Non-Robust)", rnr_nr, :dot),
    ]
    player_colors = Dict(1 => :blue, 2 => :red, 3 => :green)

    fig = Figure(size=(1000, 650))
    ax = Axis(fig[1, 1],
        title = "Final Cumulative Cost vs Nature Multiplier — R-vs-R vs R-vs-NR",
        xlabel = "Nature Multiplier",
        ylabel = "Mean Final Cumulative Cost",
        xscale = xscale,
    )

    for player_idx in player_indices
        color = get(player_colors, player_idx, :gray)
        for (label, dict, linestyle) in series_specs
            xs = Float64[]
            means = Float64[]
            stds = Float64[]
            for m in multipliers
                entries = get(dict, m, SenateTrajectoryAnalysisEntry[])
                isempty(entries) && continue
                trajs = extract_executed_trajectories(entries, player_idx, 2; cumulative=true)
                final_costs = [t[end] for t in trajs if !isempty(t)]
                isempty(final_costs) && continue
                push!(xs, Float64(m))
                push!(means, mean(final_costs))
                push!(stds, length(final_costs) > 1 ? std(final_costs) : 0.0)
            end
            if !isempty(xs)
                errorbars!(ax, xs, means, stds, color=(color, 0.3))
                scatterlines!(ax, xs, means; color=color, linewidth=2, linestyle=linestyle,
                    markersize=8, label="P$(player_idx) $label")
            end
        end
    end

    axislegend(ax, position=:lt)
    png_path = joinpath(output_directory, "rvr_vs_rnr_summary.png")
    save(png_path, fig)
    save(joinpath(output_directory, "rvr_vs_rnr_summary.pdf"), fig)
    println("Saved overlay summary plot to $png_path")

    _plot_overlay_violin(rvr, rnr_r, rnr_nr, multipliers;
        directory=output_directory, player_idx=2)
end

"""
    _plot_overlay_violin(rvr, rnr_r, rnr_nr, multipliers; directory, player_idx)

Three-series violin: R-vs-R and R-vs-NR-robust violins side-by-side per multiplier,
pooled R-vs-NR-non-robust violin on the right as the baseline. P2's final cumulative
deterministic cost.
"""
function _plot_overlay_violin(rvr, rnr_r, rnr_nr, multipliers;
    directory::String, player_idx::Int=2)

    mkpath(directory)
    n_sv = length(multipliers)

    function sweep_color(idx, n)
        t = n <= 1 ? 0.0 : (idx - 1) / (n - 1)
        stops = [
            (0.0,  RGBf(0.2, 0.4, 1.0)),
            (0.25, RGBf(0.2, 0.8, 0.4)),
            (0.5,  RGBf(0.9, 0.9, 0.2)),
            (0.75, RGBf(1.0, 0.6, 0.2)),
            (1.0,  RGBf(0.9, 0.2, 0.2)),
        ]
        for i in 1:length(stops)-1
            t0, c0 = stops[i]; t1, c1 = stops[i+1]
            if t <= t1
                s = (t - t0) / (t1 - t0)
                return RGBf(c0.r + s*(c1.r-c0.r), c0.g + s*(c1.g-c0.g), c0.b + s*(c1.b-c0.b))
            end
        end
        return stops[end][2]
    end
    nr_color = RGBf(0.6, 0.6, 0.6)

    function iqr_filter(costs::Vector{Float64})
        length(costs) < 4 && return costs
        q1 = quantile(costs, 0.25); q3 = quantile(costs, 0.75); iqr = q3 - q1
        return filter(c -> q1 - 1.5*iqr <= c <= q3 + 1.5*iqr, costs)
    end
    function final_costs(entries)
        trajs = extract_executed_trajectories(entries, player_idx, 2; cumulative=true)
        return Float64[t[end] for t in trajs if !isempty(t)]
    end

    # Per-multiplier: two violins offset by ±0.2 around integer x. Pooled NR at n_sv+1.
    rvr_offset = -0.22
    rnr_offset = +0.22
    violin_width = 0.40

    fig = Figure(size=(max(900, 130 * (n_sv + 1)), 650),
        backgroundcolor=:transparent, fontsize=22)
    update_theme!(fonts = (; regular = "Palatino Linotype",
                              bold = "Palatino Linotype",
                              italic = "Palatino Linotype"))
    ax = Axis(fig[1, 1],
        backgroundcolor=:transparent,
        xlabel = "Nature's Control Effort Cost (c)",
        ylabel = "Total Cost (Robust Activist, P$(player_idx))",
        title  = "R-vs-R vs R-vs-NR — advantage is not only from asymmetry",
        xlabelsize = 32, ylabelsize = 32, titlesize = 24,
        xticklabelsize = 26, yticklabelsize = 26,
        xticks = (collect(1:n_sv+1), vcat(string.(multipliers), ["NR"])),
        xticklabelrotation = π/12,
        topspinevisible = false, rightspinevisible = false,
        xgridvisible = false, ygridvisible = false,
    )
    xlims!(ax, 0.4, n_sv + 1.6)

    pooled_nr = Float64[]

    for (idx, m) in enumerate(multipliers)
        col = sweep_color(idx, n_sv)
        rvr_costs = iqr_filter(final_costs(get(rvr, m, SenateTrajectoryAnalysisEntry[])))
        rnr_costs = iqr_filter(final_costs(get(rnr_r, m, SenateTrajectoryAnalysisEntry[])))
        append!(pooled_nr, final_costs(get(rnr_nr, m, SenateTrajectoryAnalysisEntry[])))

        if !isempty(rvr_costs)
            x = idx + rvr_offset
            violin!(ax, fill(x, length(rvr_costs)), rvr_costs;
                color=(col, 0.65), width=violin_width, strokewidth=2, strokecolor=:black)
            scatter!(ax, fill(x, length(rvr_costs)) .+ randn(length(rvr_costs)).*0.025, rvr_costs;
                color=(col, 0.5), markersize=6)
            mr = mean(rvr_costs)
            lines!(ax, [x - 0.10, x + 0.10], [mr, mr]; color=:black, linewidth=2)
        end
        if !isempty(rnr_costs)
            x = idx + rnr_offset
            violin!(ax, fill(x, length(rnr_costs)), rnr_costs;
                color=(col, 0.30), width=violin_width)
            scatter!(ax, fill(x, length(rnr_costs)) .+ randn(length(rnr_costs)).*0.025, rnr_costs;
                color=(col, 0.4), markersize=6)
            mr = mean(rnr_costs)
            lines!(ax, [x - 0.10, x + 0.10], [mr, mr]; color=:black, linewidth=2, linestyle=:dash)
        end
    end

    pooled_nr = iqr_filter(pooled_nr)
    nr_x = n_sv + 1
    nr_mean = isempty(pooled_nr) ? nothing : mean(pooled_nr)
    if !isempty(pooled_nr)
        violin!(ax, fill(nr_x, length(pooled_nr)), pooled_nr;
            color=(nr_color, 0.6), width=0.9)
        scatter!(ax, fill(nr_x, length(pooled_nr)) .+ randn(length(pooled_nr)).*0.06, pooled_nr;
            color=(nr_color, 0.5), markersize=7)
        lines!(ax, [nr_x - 0.15, nr_x + 0.15], [nr_mean, nr_mean]; color=:black, linewidth=2)
    end
    if !isnothing(nr_mean)
        hlines!(ax, [nr_mean]; color=RGBAf(0,0,0,0.4), linewidth=1.5, linestyle=:dash)
    end

    # Legend swatches (synthetic for the two-series outline-vs-fill convention)
    legend_box = [
        PolyElement(color=(:gray, 0.65), strokecolor=:black, strokewidth=2),
        PolyElement(color=(:gray, 0.30), strokecolor=(:black, 0.0)),
        PolyElement(color=(nr_color, 0.6), strokecolor=(:black, 0.0)),
    ]
    Legend(fig[1, 2], legend_box,
        ["R-vs-R (solid stroke)", "R-vs-NR P2=Robust (no stroke)", "NR baseline (pooled)"];
        framevisible=false, labelsize=20)

    filename = joinpath(directory, "rvr_vs_rnr_violin")
    save(filename * ".png", fig, px_per_unit=3)
    save(filename * ".pdf", fig)
    println("Saved overlay violin plot to $(filename).png")
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
    create_sweep_violin_plot(sweep_collected, "Planning Horizon";
        directory=merged_dir, sweep_name="planning_horizon")

    # Paired trajectory analysis
    create_sweep_paired_analysis(sweep_collected, "Planning Horizon";
        directory=merged_dir, sweep_name="planning_horizon")
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
    create_sweep_violin_plot(sweep_collected, "GT Drift Scale";
        directory=merged_dir, sweep_name="drift_mismatch")

    # Paired trajectory analysis
    create_sweep_paired_analysis(sweep_collected, "GT Drift Scale";
        directory=merged_dir, sweep_name="drift_mismatch")
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
    create_sweep_violin_plot(sweep_collected, "P1 Type";
        directory=merged_dir, sweep_name="robustness_comparison")

    # Paired trajectory analysis
    create_sweep_paired_analysis(sweep_collected, "P1 Type";
        directory=merged_dir, sweep_name="robustness_comparison")
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
    create_sweep_violin_plot(sweep_collected, "Planning Horizon";
        directory=merged_dir, sweep_name="control_planning")

    # Paired trajectory analysis
    create_sweep_paired_analysis(sweep_collected, "Planning Horizon";
        directory=merged_dir, sweep_name="control_planning")
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
# VIOLIN PLOTS
# ========================================================================================

"""
    create_sweep_violin_plot(sweep_collected, sweep_label; directory, sweep_name, player_idx)

Create violin plot of P2 final cumulative deterministic cost across sweep values.
Robust P2 gets one violin per sweep value (colored blue→green→yellow→orange→red);
non-robust P2 entries are pooled into a single gray "NR" baseline on the far right.

Style: transparent background, scattered data points, white dash for mean,
horizontal white dashed line for baseline mean, light gray text/ticks.
"""
function create_sweep_violin_plot(
    sweep_collected::Dict,
    sweep_label::String;
    directory::String="./exp/senate/outputs/analysis",
    sweep_name::String="sweep",
    player_idx::Int=2
)
    if isempty(sweep_collected)
        println("No sweep data for violin plot")
        return
    end

    mkpath(directory)
    sweep_values = sort(collect(keys(sweep_collected)))
    n_sv = length(sweep_values)

    # --- Color ramp: blue → green → yellow → orange → red ---
    function sweep_color(idx, n)
        t = n <= 1 ? 0.0 : (idx - 1) / (n - 1)
        # 5-stop gradient: blue(0) → green(0.25) → yellow(0.5) → orange(0.75) → red(1)
        stops = [
            (0.0,  RGBf(0.2, 0.4, 1.0)),
            (0.25, RGBf(0.2, 0.8, 0.4)),
            (0.5,  RGBf(0.9, 0.9, 0.2)),
            (0.75, RGBf(1.0, 0.6, 0.2)),
            (1.0,  RGBf(0.9, 0.2, 0.2)),
        ]
        # Find bounding stops and lerp
        for i in 1:length(stops)-1
            t0, c0 = stops[i]
            t1, c1 = stops[i+1]
            if t <= t1
                s = (t - t0) / (t1 - t0)
                return RGBf(
                    c0.r + s * (c1.r - c0.r),
                    c0.g + s * (c1.g - c0.g),
                    c0.b + s * (c1.b - c0.b))
            end
        end
        return stops[end][2]
    end
    nr_color = RGBf(0.6, 0.6, 0.6)
    text_color = :black

    # --- IQR outlier filter ---
    function iqr_filter(costs::Vector{Float64})
        length(costs) < 4 && return costs
        q1 = quantile(costs, 0.25)
        q3 = quantile(costs, 0.75)
        iqr = q3 - q1
        return filter(c -> q1 - 1.5 * iqr <= c <= q3 + 1.5 * iqr, costs)
    end

    # --- Extract final cumulative P2 costs ---
    function final_costs(entries)
        trajs = extract_executed_trajectories(entries, player_idx, 2; cumulative=true)
        return Float64[t[end] for t in trajs if !isempty(t)]
    end

    # Collect per-group data: (position, costs, color)
    groups = Vector{NamedTuple{(:pos, :costs, :color, :label), Tuple{Int, Vector{Float64}, RGBAf, String}}}()
    tick_positions = Int[]
    tick_labels = String[]

    for (idx, sv) in enumerate(sweep_values)
        data = sweep_collected[sv]
        costs = iqr_filter(final_costs(data.robust_entries))
        push!(groups, (pos=idx, costs=costs, color=sweep_color(idx, n_sv), label=string(sv)))
        push!(tick_positions, idx)
        push!(tick_labels, string(sv))
    end

    # Non-robust pooled
    nr_position = n_sv + 1
    all_nr_costs = Float64[]
    for sv in sweep_values
        append!(all_nr_costs, final_costs(sweep_collected[sv].non_robust_entries))
    end
    all_nr_costs = iqr_filter(all_nr_costs)
    if !isempty(all_nr_costs)
        push!(groups, (pos=nr_position, costs=all_nr_costs, color=nr_color, label="NR"))
        push!(tick_positions, nr_position)
        push!(tick_labels, "NR")
    end

    if all(isempty(g.costs) for g in groups)
        println("No cost data for violin plot")
        return
    end

    # --- Figure with transparent background ---
    update_theme!(fonts = (; regular = "Palatino Linotype",
                             bold = "Palatino Linotype",
                             italic = "Palatino Linotype"))
    fig = Figure(size=(max(700, 120 * length(tick_positions)), 650),
        backgroundcolor=:transparent, fontsize=22)

    # Build descriptive axis labels
    x_label = if sweep_name == "nature_multiplier"
        "Nature's Control Effort Cost (c)"
    else
        sweep_label * " (c)"
    end

    ax = Axis(fig[1, 1],
        backgroundcolor=:transparent,
        xlabel = x_label,
        ylabel = "Total Cost (Robust Activist)",
        xlabelsize = 40, ylabelsize = 40,
        xticklabelsize = 32, yticklabelsize = 32,
        xticks = (tick_positions, tick_labels),
        xticklabelrotation = π/12,
        xlabelcolor = text_color, ylabelcolor = text_color,
        xticklabelcolor = text_color, yticklabelcolor = text_color,
        xtickcolor = text_color, ytickcolor = text_color,
        bottomspinecolor = text_color, leftspinecolor = text_color,
        topspinevisible = false, rightspinevisible = false,
        xgridvisible = false, ygridvisible = false,
    )
    xlims!(ax, 0.4, length(tick_positions) + 0.6)

    # --- Draw each violin + scatter + mean ---
    nr_mean = !isempty(all_nr_costs) ? mean(all_nr_costs) : nothing

    for g in groups
        isempty(g.costs) && continue

        # Violin (no outline)
        violin!(ax, fill(g.pos, length(g.costs)), g.costs,
            color=(g.color, 0.6), width=0.9)

        # Jittered scatter points (color matches violin)
        jitter = randn(length(g.costs)) .* 0.06
        scatter!(ax, fill(g.pos, length(g.costs)) .+ jitter, g.costs,
            color=(g.color, 0.4), markersize=8)

        # Mean bar
        m = mean(g.costs)
        lines!(ax, [g.pos - 0.15, g.pos + 0.15], [m, m],
            color=:black, linewidth=2)
    end

    # Horizontal dashed line at baseline (NR) mean
    if !isnothing(nr_mean)
        hlines!(ax, [nr_mean], color=RGBAf(0,0,0,0.5), linewidth=1.5, linestyle=:dash)
    end

    filename = joinpath(directory, "violin_$(sweep_name)")
    save(filename * ".png", fig, px_per_unit=3)
    save(filename * ".pdf", fig)
    println("Saved violin plot to $(filename).png")
end

# ========================================================================================
# MERGED MEAN TRAJECTORY PLOT
# ========================================================================================

"""
    create_sweep_mean_trajectory_plot(sweep_collected, sweep_label; directory, sweep_name)

Create a single-panel opinion-space trajectory plot showing mean trajectories across sweep values.
All senators overlaid on one axis. Each sweep value gets its own color (same ramp as violin plot).
Robust = solid, NR = dashed (gray). Senator starting positions, player goals (ellipsoid centers),
and obstacles are annotated.
"""
function create_sweep_mean_trajectory_plot(
    sweep_collected::Dict,
    sweep_label::String;
    directory::String="./exp/senate/outputs/analysis",
    sweep_name::String="sweep"
)
    if isempty(sweep_collected)
        println("No sweep data for mean trajectory plot")
        return
    end

    mkpath(directory)
    sweep_values = sort(collect(keys(sweep_collected)))
    n_sv = length(sweep_values)

    # --- Color ramp (same as violin) ---
    function sweep_color(idx, n)
        t = n <= 1 ? 0.0 : (idx - 1) / (n - 1)
        stops = [
            (0.0,  RGBf(0.2, 0.4, 1.0)),
            (0.25, RGBf(0.2, 0.8, 0.4)),
            (0.5,  RGBf(0.9, 0.9, 0.2)),
            (0.75, RGBf(1.0, 0.6, 0.2)),
            (1.0,  RGBf(0.9, 0.2, 0.2)),
        ]
        for i in 1:length(stops)-1
            t0, c0 = stops[i]
            t1, c1 = stops[i+1]
            if t <= t1
                s = (t - t0) / (t1 - t0)
                return RGBf(
                    c0.r + s * (c1.r - c0.r),
                    c0.g + s * (c1.g - c0.g),
                    c0.b + s * (c1.b - c0.b))
            end
        end
        return stops[end][2]
    end

    senator_colors = [:blue, :green, :orange, :purple, :brown]

    # --- Extract mean trajectories per senator ---
    function compute_mean_trajectories(entries)
        isempty(entries) && return Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
        valid = [e for e in entries if !isempty(e.gt_state_history)]
        isempty(valid) && return Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

        num_senators = length(valid[1].gt_state_history[1].blocks)
        min_T = minimum(length(e.gt_state_history) for e in valid)

        result = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
        for s in 1:num_senators
            all_xs = [Float64[state.blocks[s][1] for state in e.gt_state_history[1:min_T]] for e in valid]
            all_ys = [Float64[state.blocks[s][2] for state in e.gt_state_history[1:min_T]] for e in valid]
            mean_xs = [mean(xs[t] for xs in all_xs) for t in 1:min_T]
            mean_ys = [mean(ys[t] for ys in all_ys) for t in 1:min_T]
            result[s] = (mean_xs, mean_ys)
        end
        return result
    end

    # --- Find first valid entry for metadata ---
    first_entry = nothing
    for sv in sweep_values
        for e in sweep_collected[sv].robust_entries
            if !isempty(e.gt_state_history)
                first_entry = e
                break
            end
        end
        !isnothing(first_entry) && break
    end
    if isnothing(first_entry)
        for sv in sweep_values
            for e in sweep_collected[sv].non_robust_entries
                if !isempty(e.gt_state_history)
                    first_entry = e
                    break
                end
            end
            !isnothing(first_entry) && break
        end
    end
    if isnothing(first_entry)
        println("No trajectory data for mean trajectory plot")
        return
    end
    num_senators = length(first_entry.gt_state_history[1].blocks)

    # --- Set up figure (three panels side by side) ---
    update_theme!(fonts = (; regular = "Palatino Linotype",
                             bold = "Palatino Linotype",
                             italic = "Palatino Linotype"))
    fig = Figure(size=(1800, 700), backgroundcolor=:transparent, fontsize=36)

    axes = [Axis(fig[1, s],
        backgroundcolor=:transparent,
        xlabel = "Opinion Dimension 1",
        ylabel = s == 1 ? "Opinion Dimension 2" : "",
        title = "Senator $s",
        aspect = DataAspect(),
        topspinevisible = false, rightspinevisible = false,
        xgridvisible = false, ygridvisible = false,
        yticklabelsvisible = s == 1,
        ylabelvisible = s == 1,
    ) for s in 1:num_senators]

    # --- Precompute NR baseline trajectories (pooled across sweep values) ---
    all_nr_entries = SenateTrajectoryAnalysisEntry[]
    for sv in sweep_values
        append!(all_nr_entries, sweep_collected[sv].non_robust_entries)
    end
    nr_trajs = compute_mean_trajectories(all_nr_entries)

    # Use first axis for legend entries
    legend_ax = axes[1]

    # --- Draw annotations, NR baseline, and sweep trajectories per senator ---
    for s in 1:num_senators
        ax = axes[s]

        # Draw obstacle and goal annotations
        if !isnothing(first_entry.params)
            params = first_entry.params
            player_indices = sort(collect(keys(params.player_configs)))
            player_markers = [:star5, :diamond]
            player_labels = ["P1 Goal", "P2 Goal"]

            for (pi, pidx) in enumerate(player_indices)
                config = params.player_configs[pidx]
                for center in config.ellipsoid_centers
                    if length(center) >= 2
                        scatter!(ax, [center[1]], [center[2]],
                            marker=player_markers[mod1(pi, 2)],
                            markersize=36, color=:transparent,
                            strokewidth=4, strokecolor=:black,
                            label=s == 1 ? player_labels[mod1(pi, 2)] : nothing)
                    end
                end
                if !isempty(config.obstacle_centers) && config.obstacle_weights[1] > 0
                    for center in config.obstacle_centers
                        if length(center) >= 2
                            scatter!(ax, [center[1]], [center[2]],
                                marker=:xcross, markersize=32,
                                color=RGBAf(0.8, 0.0, 0.0, 0.6), strokewidth=4,
                                label=(s == 1 && pi == 1) ? "Obstacle" : nothing)
                        end
                    end
                end
            end

            # Senator initial position (this senator only)
            block = params.ground_truth_initial_states.blocks[s]
            scatter!(ax, [block[1]], [block[2]],
                marker=:circle, markersize=24,
                color=(senator_colors[mod1(s, length(senator_colors))], 0.5),
                strokewidth=3, strokecolor=:black,
                label=s == 1 ? "Start" : nothing)
        end

        # NR baseline for this senator
        if haskey(nr_trajs, s)
            xs, ys = nr_trajs[s]
            lines!(ax, xs, ys, color=RGBAf(0.4, 0.4, 0.4, 0.8),
                linewidth=5, linestyle=:dash,
                label=s == 1 ? "NR" : nothing)
            scatter!(ax, [xs[end]], [ys[end]], color=:gray,
                marker=:rect, markersize=16)
        end

        # Sweep values for this senator
        for (idx, sv) in enumerate(sweep_values)
            data = sweep_collected[sv]
            r_trajs = compute_mean_trajectories(data.robust_entries)
            col = sweep_color(idx, n_sv)

            if haskey(r_trajs, s)
                xs, ys = r_trajs[s]
                lines!(ax, xs, ys, color=(col, 0.85), linewidth=5,
                    label=s == 1 ? string(sv) : nothing)
                scatter!(ax, [xs[end]], [ys[end]], color=col,
                    marker=:rect, markersize=12)
            end
        end
    end

    # --- Legend (bottom, horizontal) ---
    Legend(fig[2, :], legend_ax, sweep_label, orientation=:horizontal,
        nbanks=2, framevisible=false, fontsize=28)

    filename = joinpath(directory, "mean_trajectories_$(sweep_name)")
    save(filename * ".png", fig, px_per_unit=3)
    save(filename * ".pdf", fig)
    println("Saved mean trajectory plot to $(filename).png")
end

# ========================================================================================
# P2 CONTROL COST SWEEP PLOT
# ========================================================================================

"""
    create_sweep_p2_control_cost_plot(sweep_collected, sweep_label; directory, sweep_name, player_idx)

Plot P2's per-step control cost (config.control_cost_weight * ‖u_t‖²) over time,
with one mean line per sweep value (e.g. nature multiplier) using the same color
ramp as `create_sweep_mean_trajectory_plot`. Two panels: per-step (left) and
cumulative (right). NR baseline is pooled across sweep values and drawn as a
dashed gray line.
"""
function create_sweep_p2_control_cost_plot(
    sweep_collected::Dict,
    sweep_label::String;
    directory::String="./exp/senate/outputs/analysis",
    sweep_name::String="sweep",
    player_idx::Int=2,
)
    if isempty(sweep_collected)
        println("No sweep data for P$(player_idx) control cost plot")
        return
    end

    mkpath(directory)
    sweep_values = sort(collect(keys(sweep_collected)))
    n_sv = length(sweep_values)

    # --- Color ramp (same as mean trajectory / violin plots) ---
    function sweep_color(idx, n)
        t = n <= 1 ? 0.0 : (idx - 1) / (n - 1)
        stops = [
            (0.0,  RGBf(0.2, 0.4, 1.0)),
            (0.25, RGBf(0.2, 0.8, 0.4)),
            (0.5,  RGBf(0.9, 0.9, 0.2)),
            (0.75, RGBf(1.0, 0.6, 0.2)),
            (1.0,  RGBf(0.9, 0.2, 0.2)),
        ]
        for i in 1:length(stops)-1
            t0, c0 = stops[i]
            t1, c1 = stops[i+1]
            if t <= t1
                s = (t - t0) / (t1 - t0)
                return RGBf(
                    c0.r + s * (c1.r - c0.r),
                    c0.g + s * (c1.g - c0.g),
                    c0.b + s * (c1.b - c0.b))
            end
        end
        return stops[end][2]
    end

    function compute_control_cost_trajs(entries)
        trajs = Vector{Vector{Float64}}()
        for e in entries
            ctrls = extract_senate_executed_controls(e)
            if !haskey(ctrls, player_idx) || isempty(ctrls[player_idx])
                continue
            end
            w = (!isnothing(e.params) && haskey(e.params.player_configs, player_idx)) ?
                e.params.player_configs[player_idx].control_cost_weight : 1.0
            traj = Float64[w * dot(u, u) for u in ctrls[player_idx]]
            push!(trajs, traj)
        end
        return trajs
    end

    function mean_curve(trajs; cumulative=false)
        isempty(trajs) && return Float64[], Float64[]
        min_T = minimum(length(t) for t in trajs)
        min_T == 0 && return Float64[], Float64[]
        series = cumulative ? [cumsum(t[1:min_T]) for t in trajs] : [t[1:min_T] for t in trajs]
        means = [mean(s[i] for s in series) for i in 1:min_T]
        return collect(1:min_T), means
    end

    all_nr_entries = SenateTrajectoryAnalysisEntry[]
    for sv in sweep_values
        append!(all_nr_entries, sweep_collected[sv].non_robust_entries)
    end
    nr_trajs = compute_control_cost_trajs(all_nr_entries)

    update_theme!(fonts = (; regular = "Palatino Linotype",
                             bold = "Palatino Linotype",
                             italic = "Palatino Linotype"))
    fig = Figure(size=(1500, 650), backgroundcolor=:transparent, fontsize=28)

    ax_step = Axis(fig[1, 1],
        backgroundcolor=:transparent,
        xlabel = "Execution Step",
        ylabel = "P$(player_idx) Control Cost",
        title = "Per-Step",
        topspinevisible = false, rightspinevisible = false,
        xgridvisible = false, ygridvisible = false,
    )
    ax_cum = Axis(fig[1, 2],
        backgroundcolor=:transparent,
        xlabel = "Execution Step",
        ylabel = "P$(player_idx) Cumulative Control Cost",
        title = "Cumulative",
        topspinevisible = false, rightspinevisible = false,
        xgridvisible = false, ygridvisible = false,
    )

    if !isempty(nr_trajs)
        ts, ms = mean_curve(nr_trajs; cumulative=false)
        if !isempty(ts)
            lines!(ax_step, ts, ms, color=RGBAf(0.4, 0.4, 0.4, 0.85),
                linewidth=4, linestyle=:dash, label="NR")
        end
        ts, ms = mean_curve(nr_trajs; cumulative=true)
        if !isempty(ts)
            lines!(ax_cum, ts, ms, color=RGBAf(0.4, 0.4, 0.4, 0.85),
                linewidth=4, linestyle=:dash, label="NR")
        end
    end

    for (idx, sv) in enumerate(sweep_values)
        r_trajs = compute_control_cost_trajs(sweep_collected[sv].robust_entries)
        isempty(r_trajs) && continue
        col = sweep_color(idx, n_sv)
        ts, ms = mean_curve(r_trajs; cumulative=false)
        if !isempty(ts)
            lines!(ax_step, ts, ms, color=(col, 0.9), linewidth=4, label=string(sv))
        end
        ts, ms = mean_curve(r_trajs; cumulative=true)
        if !isempty(ts)
            lines!(ax_cum, ts, ms, color=(col, 0.9), linewidth=4, label=string(sv))
        end
    end

    Legend(fig[2, :], ax_step, sweep_label, orientation=:horizontal,
        nbanks=2, framevisible=false)

    filename = joinpath(directory, "p$(player_idx)_control_cost_$(sweep_name)")
    save(filename * ".png", fig, px_per_unit=3)
    save(filename * ".pdf", fig)
    println("Saved P$(player_idx) control cost plot to $(filename).png")
end

# ========================================================================================
# R-vs-R vs R-vs-NR COST COMPONENT OVERLAY
# ========================================================================================

"""
    _load_rvr_rnr_series(multipliers; rvr_directory, rnr_directory)

Load the three overlay series (R-vs-R, R-vs-NR P2=Robust, R-vs-NR P2=Non-Robust)
per nature multiplier. Same loading scheme as `analyze_rvr_vs_rnr_overlay`.
"""
function _load_rvr_rnr_series(multipliers; rvr_directory, rnr_directory)
    rvr = Dict{Int, Vector{SenateTrajectoryAnalysisEntry}}()
    rnr_r = Dict{Int, Vector{SenateTrajectoryAnalysisEntry}}()
    rnr_nr = Dict{Int, Vector{SenateTrajectoryAnalysisEntry}}()

    for m in multipliers
        println("\n===== Loading entries for nature_multiplier=$m =====")

        load_and_analyze_senate_solution_files(
            directory=rvr_directory,
            file_pattern=Regex("p2_nature_multiplier_$(m)_p2_type"))
        rvr[m] = copy(SENATE_TRAJECTORY_TRACKER.entries)

        load_and_analyze_senate_solution_files(
            directory=rnr_directory,
            file_pattern=Regex("p2_nature_multiplier_$(m)_p2_type_robust"))
        rnr_r[m] = copy(SENATE_TRAJECTORY_TRACKER.entries)

        load_and_analyze_senate_solution_files(
            directory=rnr_directory,
            file_pattern=Regex("p2_nature_multiplier_$(m)_p2_type_non_robust"))
        rnr_nr[m] = copy(SENATE_TRAJECTORY_TRACKER.entries)

        println("  R-vs-R: $(length(rvr[m])), R-vs-NR robust: $(length(rnr_r[m])), R-vs-NR NR: $(length(rnr_nr[m]))")
    end
    return rvr, rnr_r, rnr_nr
end

"""
    _entry_cost_component_timeseries(entry, player_idx)

Per-execution-step deterministic cost components for one entry, evaluated on the
ground-truth state with zero-covariance beliefs (same convention as
`_decompose_entry_costs`, which reports the cumulative totals of this series).
Returns a Vector of `(preference, control, obstacle)` NamedTuples, or `nothing`.
The final step carries the terminal-weighted preference cost and no control cost.
"""
function _entry_cost_component_timeseries(entry::SenateTrajectoryAnalysisEntry, player_idx::Int)
    if isnothing(entry.params) || isempty(entry.gt_state_history)
        return nothing
    end

    config = get(entry.params.player_configs, player_idx, nothing)
    if isnothing(config)
        return nothing
    end

    executed_controls = extract_senate_executed_controls(entry)
    player_controls = get(executed_controls, player_idx, nothing)

    out = NamedTuple[]
    T = length(entry.gt_state_history)
    for t in 1:T
        gt_state = entry.gt_state_history[t]
        is_terminal = (t == T) || isnothing(player_controls) || t > length(player_controls)

        beliefs = Beliefs([
            Belief(block, zeros(length(block), length(block)))
            for _ in sort(collect(keys(entry.params.player_configs)))
            for block in gt_state.blocks
        ])

        if is_terminal
            c = compute_senate_cost_components(beliefs, nothing, config; is_terminal=true)
            push!(out, (preference=c.preference, control=0.0, obstacle=c.obstacle))
        else
            merged_ctrl_vec = Float64[]
            ctrl_block_sizes = Int[]
            for pidx in sort(collect(keys(executed_controls)))
                if t <= length(executed_controls[pidx])
                    append!(merged_ctrl_vec, executed_controls[pidx][t])
                    push!(ctrl_block_sizes, length(executed_controls[pidx][t]))
                end
            end
            if isempty(merged_ctrl_vec)
                continue
            end
            merged_controls = BlockVector(merged_ctrl_vec, ctrl_block_sizes)

            c = compute_senate_cost_components(beliefs, merged_controls, config; is_terminal=false)
            push!(out, (preference=c.preference, control=c.control, obstacle=c.obstacle))
        end
    end
    return out
end

"""
    create_rvr_vs_rnr_cost_component_plot(rvr, rnr_r, rnr_nr, multipliers;
        directory, player_idx=2, cumulative=false)

Grid of P`player_idx` cost components over execution time: rows = components
(preference, control, obstacle, total), columns = R-vs-R and R-vs-NR (P2=Robust).
One mean line per nature multiplier (sweep color ramp); pooled R-vs-NR
non-robust baseline as dashed gray in every panel.
"""
function create_rvr_vs_rnr_cost_component_plot(rvr, rnr_r, rnr_nr, multipliers;
    directory::String, player_idx::Int=2, cumulative::Bool=false)

    mkpath(directory)
    n_sv = length(multipliers)

    function sweep_color(idx, n)
        t = n <= 1 ? 0.0 : (idx - 1) / (n - 1)
        stops = [
            (0.0,  RGBf(0.2, 0.4, 1.0)),
            (0.25, RGBf(0.2, 0.8, 0.4)),
            (0.5,  RGBf(0.9, 0.9, 0.2)),
            (0.75, RGBf(1.0, 0.6, 0.2)),
            (1.0,  RGBf(0.9, 0.2, 0.2)),
        ]
        for i in 1:length(stops)-1
            t0, c0 = stops[i]; t1, c1 = stops[i+1]
            if t <= t1
                s = (t - t0) / (t1 - t0)
                return RGBf(c0.r + s*(c1.r-c0.r), c0.g + s*(c1.g-c0.g), c0.b + s*(c1.b-c0.b))
            end
        end
        return stops[end][2]
    end
    nr_color = RGBAf(0.4, 0.4, 0.4, 0.85)

    components = [:preference, :control, :obstacle, :total]
    comp_titles = Dict(
        :preference => "Preference (ellipsoidal)",
        :control    => "Control",
        :obstacle   => "Obstacle",
        :total      => "Total (sum of components)",
    )

    function decompose_entries(entries)
        out = Vector{Vector{NamedTuple}}()
        for e in entries
            series = _entry_cost_component_timeseries(e, player_idx)
            (isnothing(series) || isempty(series)) && continue
            push!(out, series)
        end
        return out
    end

    function comp_trajs(decomposed, comp)
        return [comp === :total ?
                    Float64[s.preference + s.control + s.obstacle for s in series] :
                    Float64[getfield(s, comp) for s in series]
                for series in decomposed]
    end

    function mean_curve(trajs)
        isempty(trajs) && return Float64[], Float64[]
        min_T = minimum(length(t) for t in trajs)
        min_T == 0 && return Float64[], Float64[]
        series = cumulative ? [cumsum(t[1:min_T]) for t in trajs] : [t[1:min_T] for t in trajs]
        return collect(1:min_T), [mean(s[i] for s in series) for i in 1:min_T]
    end

    series_specs = [("R-vs-R", rvr), ("R-vs-NR (P2=Robust)", rnr_r)]
    all_nr_entries = SenateTrajectoryAnalysisEntry[]
    for m in multipliers
        append!(all_nr_entries, get(rnr_nr, m, SenateTrajectoryAnalysisEntry[]))
    end

    update_theme!(fonts = (; regular = "Palatino Linotype",
                             bold = "Palatino Linotype",
                             italic = "Palatino Linotype"))
    mode = cumulative ? "Cumulative" : "Per-Step"
    fig = Figure(size=(1500, 380 * length(components) + 140),
        backgroundcolor=:transparent, fontsize=24)

    axes = Dict{Tuple{Int, Int}, Axis}()
    for (row, comp) in enumerate(components)
        for (colidx, (label, _)) in enumerate(series_specs)
            ax = Axis(fig[row, colidx],
                backgroundcolor=:transparent,
                xlabel = row == length(components) ? "Execution Step" : "",
                ylabel = colidx == 1 ? "$(comp_titles[comp])" : "",
                title = row == 1 ? label : "",
                topspinevisible = false, rightspinevisible = false,
                xgridvisible = false, ygridvisible = false,
            )
            axes[(row, colidx)] = ax
        end
    end

    nr_decomposed = decompose_entries(all_nr_entries)
    for (colidx, (_, dict)) in enumerate(series_specs)
        for (idx, m) in enumerate(multipliers)
            decomposed = decompose_entries(get(dict, m, SenateTrajectoryAnalysisEntry[]))
            isempty(decomposed) && continue
            col = sweep_color(idx, n_sv)
            for (row, comp) in enumerate(components)
                ts, ms = mean_curve(comp_trajs(decomposed, comp))
                isempty(ts) && continue
                lines!(axes[(row, colidx)], ts, ms, color=(col, 0.9),
                    linewidth=3, label=string(m))
            end
        end
        # Pooled NR baseline in every panel
        for (row, comp) in enumerate(components)
            ts, ms = mean_curve(comp_trajs(nr_decomposed, comp))
            isempty(ts) && continue
            lines!(axes[(row, colidx)], ts, ms, color=nr_color,
                linewidth=3, linestyle=:dash, label="NR")
        end
    end

    # Link y-axes across the two columns per row for direct comparison
    for row in 1:length(components)
        linkyaxes!(axes[(row, 1)], axes[(row, 2)])
    end

    Legend(fig[length(components) + 1, :], axes[(1, 1)],
        "Nature's Control Effort Cost (c)",
        orientation=:horizontal, nbanks=2, framevisible=false)

    suffix = cumulative ? "cumulative" : "perstep"
    filename = joinpath(directory, "rvr_vs_rnr_components_$(suffix)")
    save(filename * ".png", fig, px_per_unit=2)
    save(filename * ".pdf", fig)
    println("Saved $(mode) cost component overlay to $(filename).png")
end

"""
    analyze_rvr_vs_rnr_cost_components(; multipliers, rvr_directory, rnr_directory,
        output_directory, player_idx=2)

Load the R-vs-R and R-vs-NR sweeps and render P2's cost components (preference,
control, obstacle, total) over execution time, one line per nature multiplier,
in both per-step and cumulative form.
"""
function analyze_rvr_vs_rnr_cost_components(;
    multipliers=[1, 2, 5, 10, 25, 50, 125, 250, 625],
    rvr_directory="./exp/senate/outputs/merged/rvr_nature_control_sweep",
    rnr_directory="./exp/senate/outputs/merged/nature_control_sweep",
    output_directory="./exp/senate/outputs/analysis/rvr_vs_rnr_overlay",
    player_idx::Int=2,
    preloaded=nothing,
)
    if isnothing(preloaded)
        rvr, rnr_r, rnr_nr = _load_rvr_rnr_series(multipliers;
            rvr_directory=rvr_directory, rnr_directory=rnr_directory)
    else
        rvr, rnr_r, rnr_nr = preloaded
    end

    create_rvr_vs_rnr_cost_component_plot(rvr, rnr_r, rnr_nr, multipliers;
        directory=output_directory, player_idx=player_idx, cumulative=false)
    create_rvr_vs_rnr_cost_component_plot(rvr, rnr_r, rnr_nr, multipliers;
        directory=output_directory, player_idx=player_idx, cumulative=true)
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
        try
            qq_title = "Q-Q Plots: Normality Assessment ($label, Outliers removed: R=$n_r_removed, NR=$n_nr_removed)"
            fig_qq = Figure(size=(1000, 500), figure_padding=20)

            ax_qq = Axis(fig_qq[1, 1],
                title=qq_title * "\nCost Distribution (n_R=$(length(robust_costs)), n_NR=$(length(non_robust_costs)))",
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
        catch e
            println("Warning: Q-Q plot failed (Makie issue): $e")
        end
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

            p_val_t = 2 * ccdf(TDist(df), abs(t_stat))
            log10_p_t = p_val_t > 0 ? log10(p_val_t) : -Inf

            println(io, "T-Statistic: $(round(t_stat, digits=4))")
            println(io, "Degrees of Freedom: $(round(df, digits=2))")
            println(io, "P-Value: $(@sprintf("%.4e", p_val_t))$(isfinite(log10_p_t) && log10_p_t < -4 ? "  (log10 p = $(round(log10_p_t, digits=2)))" : "")")
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
            p_val_mw = 2 * ccdf(Normal(0, 1), abs(z_score))
            log10_p_mw = p_val_mw > 0 ? log10(p_val_mw) : -Inf

            println(io, "U-Statistic: $(round(U, digits=2))")
            println(io, "Z-Score: $(round(z_score, digits=4))")
            println(io, "P-Value: $(@sprintf("%.4e", p_val_mw))$(isfinite(log10_p_mw) && log10_p_mw < -4 ? "  (log10 p = $(round(log10_p_mw, digits=2)))" : "")")
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
            if p_val_boot == 0.0
                println(io, "P-Value: < $(@sprintf("%.1e", 1.0/n_bootstrap))  (0 of $n_bootstrap permutations exceeded observed)")
            else
                println(io, "P-Value: $(@sprintf("%.4e", p_val_boot))")
            end
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

# ========================================================================================
# NATURE DIAGNOSTICS ANALYSIS
# ========================================================================================

"""
    extract_nature_diagnostics_for_sweep(sweep_collected; player_idx=2)

Extract nature diagnostic quantities across a sweep of nature multiplier values.
Returns Dict mapping sweep_value => NamedTuple of per-trial diagnostic time series.
"""
function extract_nature_diagnostics_for_sweep(sweep_collected; player_idx::Int=2)
    result = Dict{Any, NamedTuple}()

    for (sv, data) in sweep_collected
        robust_entries = data.robust_entries

        all_control_norms = Vector{Vector{Float64}}()
        all_Q_traces = Vector{Vector{Float64}}()
        all_Q_min_eigvals = Vector{Vector{Float64}}()
        all_ff_norms = Vector{Vector{Float64}}()
        all_fb_gains = Vector{Vector{Float64}}()
        all_V_bb_traces = Vector{Vector{Vector{Float64}}}()  # per trial → per timestep → per player
        all_belief_errors = Vector{Vector{Float64}}()

        for entry in robust_entries
            # Access nature diagnostics from the entry struct
            ndh = entry.nature_diagnostics_history
            if isnothing(ndh)
                continue
            end
            diag_hist = get(ndh, player_idx, nothing)
            if isnothing(diag_hist) || isempty(diag_hist)
                continue
            end

            ctrl_norms = Float64[]
            q_traces = Float64[]
            q_min_eigs = Float64[]
            ff_norms = Float64[]
            fb_gains = Float64[]
            vbb_traces = Vector{Float64}[]

            for rh_step_diags in diag_hist
                if !isempty(rh_step_diags)
                    d = rh_step_diags[1]  # First planning step (the one executed)
                    push!(ctrl_norms, d.nature_control_norm)
                    push!(q_traces, d.Q_uu_nature_trace)
                    push!(q_min_eigs, minimum(d.Q_uu_nature_eigvals))
                    push!(ff_norms, d.nature_feedforward_norm)
                    push!(fb_gains, d.nature_feedback_gain_norm)
                    push!(vbb_traces, d.V_bb_traces)
                end
            end

            if !isempty(ctrl_norms)
                push!(all_control_norms, ctrl_norms)
                push!(all_Q_traces, q_traces)
                push!(all_Q_min_eigvals, q_min_eigs)
                push!(all_ff_norms, ff_norms)
                push!(all_fb_gains, fb_gains)
                push!(all_V_bb_traces, vbb_traces)
            end

            be_hist = entry.belief_error_history
            if !isnothing(be_hist) && !isempty(be_hist)
                push!(all_belief_errors, [mean(be) for be in be_hist])
            end
        end

        result[sv] = (
            nature_control_norms = all_control_norms,
            Q_uu_traces = all_Q_traces,
            Q_uu_min_eigvals = all_Q_min_eigvals,
            nature_feedforward_norms = all_ff_norms,
            nature_feedback_gains = all_fb_gains,
            V_bb_traces = all_V_bb_traces,
            belief_errors = all_belief_errors,
        )
    end

    return result
end

"""
    plot_nature_diagnostics_sweep(diagnostics_by_multiplier; directory, sweep_name)

Create multi-panel diagnostic plot for nature player analysis across sweep values.
Panels: (a) ||u_nature|| vs λ, (b) tr(Q_uu) - λ·dim vs λ, (c) tr(V_bb) vs λ,
        (d) belief error vs λ, (e) P2 cost vs λ.
"""
function plot_nature_diagnostics_sweep(
    diagnostics_by_multiplier::Dict,
    sweep_collected::Dict;
    directory::String="./exp/senate/outputs/analysis",
    sweep_name::String="nature_diagnostics"
)
    sweep_values = sort(collect(keys(diagnostics_by_multiplier)))
    n_sv = length(sweep_values)
    if n_sv == 0
        println("No diagnostics data to plot.")
        return
    end

    mkpath(directory)

    xs = Float64.(sweep_values)
    text_color = :black

    fig = Figure(size=(1800, 1200), backgroundcolor=:transparent)

    # --- Panel (a): Nature control magnitude ---
    ax_a = Axis(fig[1, 1]; xlabel="Nature Multiplier λ", ylabel="Mean ‖u_nature‖",
        title="Nature Control Magnitude", xscale=log10,
        xlabelsize=16, ylabelsize=16, titlesize=18,
        xlabelcolor=text_color, ylabelcolor=text_color, titlecolor=text_color,
        xticklabelcolor=text_color, yticklabelcolor=text_color,
        backgroundcolor=:transparent)

    means_a = Float64[]
    stds_a = Float64[]
    for sv in sweep_values
        d = diagnostics_by_multiplier[sv]
        trial_means = [mean(cn) for cn in d.nature_control_norms]
        push!(means_a, isempty(trial_means) ? NaN : mean(trial_means))
        push!(stds_a, isempty(trial_means) ? 0.0 : std(trial_means) / sqrt(length(trial_means)))
    end
    errorbars!(ax_a, xs, means_a, stds_a; color=:steelblue, whiskerwidth=6)
    scatterlines!(ax_a, xs, means_a; color=:steelblue, markersize=8)

    # --- Panel (b): Intrinsic curvature (Q_uu - λ·I) ---
    ax_b = Axis(fig[1, 2]; xlabel="Nature Multiplier λ", ylabel="Mean tr(Q_uu) − λ·dim",
        title="Intrinsic Curvature", xscale=log10,
        xlabelsize=16, ylabelsize=16, titlesize=18,
        xlabelcolor=text_color, ylabelcolor=text_color, titlecolor=text_color,
        xticklabelcolor=text_color, yticklabelcolor=text_color,
        backgroundcolor=:transparent)

    means_b = Float64[]
    stds_b = Float64[]
    for sv in sweep_values
        d = diagnostics_by_multiplier[sv]
        # nature_controls_dim can be inferred from Q_uu trace minus λ contribution
        # Q_uu_nature ≈ λ·I + C, so tr(Q_uu) - λ·dim ≈ tr(C)
        # We need the dimension; get it from the first available entry
        dim_nature = 6  # default for senate (3 senators × 2D)
        if !isempty(d.Q_uu_traces) && !isempty(d.Q_uu_traces[1])
            # Infer from eigenvalues length if available
        end
        trial_curvatures = [mean(qt) - Float64(sv) * dim_nature for qt in d.Q_uu_traces]
        push!(means_b, isempty(trial_curvatures) ? NaN : mean(trial_curvatures))
        push!(stds_b, isempty(trial_curvatures) ? 0.0 : std(trial_curvatures) / sqrt(length(trial_curvatures)))
    end
    errorbars!(ax_b, xs, means_b, stds_b; color=:coral, whiskerwidth=6)
    scatterlines!(ax_b, xs, means_b; color=:coral, markersize=8)

    # --- Panel (c): V_bb traces per player ---
    ax_c = Axis(fig[2, 1]; xlabel="Nature Multiplier λ", ylabel="Mean tr(V_bb)",
        title="Value Hessian (All Players)", xscale=log10,
        xlabelsize=16, ylabelsize=16, titlesize=18,
        xlabelcolor=text_color, ylabelcolor=text_color, titlecolor=text_color,
        xticklabelcolor=text_color, yticklabelcolor=text_color,
        backgroundcolor=:transparent)

    # Determine number of players from first available data
    n_total_players = 3  # default: p1, p2, nature
    for sv in sweep_values
        d = diagnostics_by_multiplier[sv]
        if !isempty(d.V_bb_traces) && !isempty(d.V_bb_traces[1]) && !isempty(d.V_bb_traces[1][1])
            n_total_players = length(d.V_bb_traces[1][1])
            break
        end
    end

    player_colors = [:steelblue, :coral, :gray50]
    player_labels = ["P1", "P2 (Robust)", "Nature"]
    for pi in 1:min(n_total_players, 3)
        means_c = Float64[]
        stds_c = Float64[]
        for sv in sweep_values
            d = diagnostics_by_multiplier[sv]
            trial_vbb = Float64[]
            for trial_vbb_ts in d.V_bb_traces
                if !isempty(trial_vbb_ts)
                    player_traces = [ts[pi] for ts in trial_vbb_ts if length(ts) >= pi]
                    if !isempty(player_traces)
                        push!(trial_vbb, mean(player_traces))
                    end
                end
            end
            push!(means_c, isempty(trial_vbb) ? NaN : mean(trial_vbb))
            push!(stds_c, isempty(trial_vbb) ? 0.0 : std(trial_vbb) / sqrt(length(trial_vbb)))
        end
        errorbars!(ax_c, xs, means_c, stds_c; color=player_colors[pi], whiskerwidth=6)
        scatterlines!(ax_c, xs, means_c; color=player_colors[pi], markersize=8, label=player_labels[pi])
    end
    axislegend(ax_c; position=:rt, labelcolor=text_color, framecolor=text_color)

    # --- Panel (d): Belief error ---
    ax_d = Axis(fig[2, 2]; xlabel="Nature Multiplier λ", ylabel="Mean Belief Error",
        title="Realized Estimation Error", xscale=log10,
        xlabelsize=16, ylabelsize=16, titlesize=18,
        xlabelcolor=text_color, ylabelcolor=text_color, titlecolor=text_color,
        xticklabelcolor=text_color, yticklabelcolor=text_color,
        backgroundcolor=:transparent)

    means_d = Float64[]
    stds_d = Float64[]
    for sv in sweep_values
        d = diagnostics_by_multiplier[sv]
        trial_means_be = [mean(be) for be in d.belief_errors]
        push!(means_d, isempty(trial_means_be) ? NaN : mean(trial_means_be))
        push!(stds_d, isempty(trial_means_be) ? 0.0 : std(trial_means_be) / sqrt(length(trial_means_be)))
    end
    errorbars!(ax_d, xs, means_d, stds_d; color=:forestgreen, whiskerwidth=6)
    scatterlines!(ax_d, xs, means_d; color=:forestgreen, markersize=8)

    # --- Panel (e): P2 cost overlay ---
    ax_e = Axis(fig[3, 1:2]; xlabel="Nature Multiplier λ", ylabel="Mean P2 Cost",
        title="P2 Final Cumulative Cost (with Optimal λ*)", xscale=log10,
        xlabelsize=16, ylabelsize=16, titlesize=18,
        xlabelcolor=text_color, ylabelcolor=text_color, titlecolor=text_color,
        xticklabelcolor=text_color, yticklabelcolor=text_color,
        backgroundcolor=:transparent)

    means_e = Float64[]
    stds_e = Float64[]
    for sv in sweep_values
        if haskey(sweep_collected, sv)
            entries = sweep_collected[sv].robust_entries
            costs = Float64[]
            for entry in entries
                try
                    trajs = extract_executed_trajectories([entry], 2, 2; cumulative=true)
                    if !isempty(trajs) && !isempty(trajs[1])
                        push!(costs, trajs[1][end])
                    end
                catch
                end
            end
            push!(means_e, isempty(costs) ? NaN : mean(costs))
            push!(stds_e, isempty(costs) ? 0.0 : std(costs) / sqrt(length(costs)))
        else
            push!(means_e, NaN)
            push!(stds_e, 0.0)
        end
    end
    errorbars!(ax_e, xs, means_e, stds_e; color=:purple, whiskerwidth=6)
    scatterlines!(ax_e, xs, means_e; color=:purple, markersize=8)

    # Mark empirical optimum
    valid_idx = findall(!isnan, means_e)
    if !isempty(valid_idx)
        best_idx = valid_idx[argmin(means_e[valid_idx])]
        vlines!(ax_e, [xs[best_idx]]; color=:red, linestyle=:dash, linewidth=1.5)
        text!(ax_e, xs[best_idx], means_e[best_idx]; text="λ*=$(sweep_values[best_idx])",
            color=:red, fontsize=14, align=(:left, :bottom), offset=(5, 5))
    end

    # Save
    save(joinpath(directory, "nature_diagnostics_$(sweep_name).png"), fig; px_per_unit=3)
    save(joinpath(directory, "nature_diagnostics_$(sweep_name).pdf"), fig)
    println("Saved nature diagnostics plot to $(joinpath(directory, "nature_diagnostics_$(sweep_name).png"))")

    return fig
end

# ========================================================================================
# PAIRED TRAJECTORY ANALYSIS (Seed-Matched Robust vs Non-Robust)
# ========================================================================================

"""
    build_seed_matched_pairs(robust_entries, non_robust_entries)

Match robust and non-robust entries by random_seed for paired comparison.
Returns vector of (seed, robust, non_robust) named tuples for common seeds.
"""
function build_seed_matched_pairs(robust_entries::Vector{SenateTrajectoryAnalysisEntry},
                                   non_robust_entries::Vector{SenateTrajectoryAnalysisEntry})
    robust_by_seed = Dict{Int, SenateTrajectoryAnalysisEntry}()
    for e in robust_entries
        if e.random_seed >= 0
            robust_by_seed[e.random_seed] = e
        end
    end

    nonrobust_by_seed = Dict{Int, SenateTrajectoryAnalysisEntry}()
    for e in non_robust_entries
        if e.random_seed >= 0
            nonrobust_by_seed[e.random_seed] = e
        end
    end

    common_seeds = sort(collect(intersect(keys(robust_by_seed), keys(nonrobust_by_seed))))
    pairs = [(seed=s, robust=robust_by_seed[s], non_robust=nonrobust_by_seed[s]) for s in common_seeds]

    println("Seed matching: $(length(pairs)) pairs from $(length(robust_entries)) robust + $(length(non_robust_entries)) non-robust entries")
    return pairs
end

"""
    compute_trajectory_divergence(pairs)

For each seed-matched pair, compute per-timestep per-senator trajectory divergence.
Returns vector of (seed, divergences) where divergences[t][s] = ||gt_robust[t].blocks[s] - gt_nonrobust[t].blocks[s]||.
"""
function compute_trajectory_divergence(pairs)
    result = []
    for pair in pairs
        gt_r = pair.robust.gt_state_history
        gt_nr = pair.non_robust.gt_state_history
        T = min(length(gt_r), length(gt_nr))
        num_senators = length(gt_r[1].blocks)

        divergences = Vector{Vector{Float64}}()
        for t in 1:T
            senator_divs = Float64[]
            for s in 1:num_senators
                push!(senator_divs, norm(gt_r[t].blocks[s] - gt_nr[t].blocks[s]))
            end
            push!(divergences, senator_divs)
        end
        push!(result, (seed=pair.seed, divergences=divergences))
    end
    return result
end

"""
    compute_divergence_statistics(divergence_data)

Aggregate divergence data across seed-matched pairs.
Returns named tuple with time series stats and final-timestep distribution.
"""
function compute_divergence_statistics(divergence_data)
    if isempty(divergence_data)
        return (mean_timeseries=Float64[], std_timeseries=Float64[],
                senator_mean_timeseries=Vector{Float64}[], senator_std_timeseries=Vector{Float64}[],
                final_divergences=Float64[], fraction_above_threshold=0.0)
    end

    T = minimum(length(d.divergences) for d in divergence_data)
    num_senators = length(divergence_data[1].divergences[1])

    # Per-senator time series
    senator_means = [Float64[] for _ in 1:num_senators]
    senator_stds = [Float64[] for _ in 1:num_senators]
    avg_means = Float64[]
    avg_stds = Float64[]

    for t in 1:T
        for s in 1:num_senators
            vals = [d.divergences[t][s] for d in divergence_data]
            push!(senator_means[s], mean(vals))
            push!(senator_stds[s], length(vals) > 1 ? std(vals) : 0.0)
        end
        avg_vals = [mean(d.divergences[t]) for d in divergence_data]
        push!(avg_means, mean(avg_vals))
        push!(avg_stds, length(avg_vals) > 1 ? std(avg_vals) : 0.0)
    end

    # Final timestep distribution (average across senators)
    final_divs = [mean(d.divergences[T]) for d in divergence_data]
    threshold = 0.01
    frac_above = count(d -> d > threshold, final_divs) / length(final_divs)

    return (mean_timeseries=avg_means, std_timeseries=avg_stds,
            senator_mean_timeseries=senator_means, senator_std_timeseries=senator_stds,
            final_divergences=final_divs, fraction_above_threshold=frac_above)
end

"""
    compute_paired_cost_decomposition(pairs; player_idx=2)

For each seed-matched pair, decompose cumulative costs into components using
existing `compute_senate_cost_components`. Returns per-pair component totals and deltas.
"""
function compute_paired_cost_decomposition(pairs; player_idx::Int=2)
    results = []
    for pair in pairs
        r_components = _decompose_entry_costs(pair.robust, player_idx)
        nr_components = _decompose_entry_costs(pair.non_robust, player_idx)
        if isnothing(r_components) || isnothing(nr_components)
            continue
        end
        push!(results, (
            seed=pair.seed,
            robust=r_components,
            non_robust=nr_components,
            delta=(
                preference=r_components.preference - nr_components.preference,
                control=r_components.control - nr_components.control,
                covariance=r_components.covariance - nr_components.covariance,
                obstacle=r_components.obstacle - nr_components.obstacle,
                total=r_components.total - nr_components.total,
            )
        ))
    end
    return results
end

"""Decompose an entry's incurred costs into cumulative components."""
function _decompose_entry_costs(entry::SenateTrajectoryAnalysisEntry, player_idx::Int)
    if isnothing(entry.params) || isempty(entry.gt_state_history)
        return nothing
    end

    config = get(entry.params.player_configs, player_idx, nothing)
    if isnothing(config)
        return nothing
    end

    executed_controls = extract_senate_executed_controls(entry)
    player_controls = get(executed_controls, player_idx, nothing)

    cum_preference = 0.0
    cum_control = 0.0
    cum_covariance = 0.0
    cum_obstacle = 0.0

    T = length(entry.gt_state_history)
    for t in 1:T
        gt_state = entry.gt_state_history[t]
        is_terminal = (t == T) || isnothing(player_controls) || t > length(player_controls)

        # Build deterministic beliefs (zero covariance) from GT state
        beliefs = Beliefs([
            Belief(block, zeros(length(block), length(block)))
            for _ in sort(collect(keys(entry.params.player_configs)))
            for block in gt_state.blocks
        ])

        if is_terminal
            components = compute_senate_cost_components(beliefs, nothing, config; is_terminal=true)
            cum_preference += components.preference
            cum_covariance += components.covariance
            cum_obstacle += components.obstacle
        else
            # Reconstruct merged controls as BlockVector
            merged_ctrl_vec = Float64[]
            ctrl_block_sizes = Int[]
            for pidx in sort(collect(keys(executed_controls)))
                if t <= length(executed_controls[pidx])
                    append!(merged_ctrl_vec, executed_controls[pidx][t])
                    push!(ctrl_block_sizes, length(executed_controls[pidx][t]))
                end
            end
            if isempty(merged_ctrl_vec)
                continue
            end
            merged_controls = BlockVector(merged_ctrl_vec, ctrl_block_sizes)

            components = compute_senate_cost_components(beliefs, merged_controls, config; is_terminal=false)
            cum_preference += components.preference
            cum_control += components.control
            cum_covariance += components.covariance
            cum_obstacle += components.obstacle
        end
    end

    return (preference=cum_preference, control=cum_control, covariance=cum_covariance,
            obstacle=cum_obstacle, total=cum_preference + cum_control + cum_covariance + cum_obstacle)
end

# ========================================================================================
# PAIRED ANALYSIS VISUALIZATIONS
# ========================================================================================

"""
    create_seed_matched_trajectory_plot(pairs, divergence_data; directory, num_seeds=3)

Lead visualization: 2D spatial trajectory plots for representative seed-matched pairs.
Shows robust (solid) vs non-robust (dashed) senator trajectories on the opinion plane.
"""
function create_seed_matched_trajectory_plot(pairs, divergence_data;
    directory::String="./exp/senate/outputs/analysis",
    num_seeds::Int=3)

    if isempty(pairs) || isempty(divergence_data)
        println("No pairs for seed-matched trajectory plot")
        return nothing
    end
    mkpath(directory)

    # Pick seeds: top by final divergence + 1 median
    final_divs = [(mean(d.divergences[end]), d.seed) for d in divergence_data]
    sort!(final_divs, by=x -> x[1], rev=true)
    selected_seeds = [fd[2] for fd in final_divs[1:min(num_seeds, length(final_divs))]]

    # Add median seed if room
    if length(final_divs) > num_seeds
        median_idx = div(length(final_divs), 2)
        median_seed = final_divs[median_idx][2]
        if !(median_seed in selected_seeds)
            push!(selected_seeds, median_seed)
        end
    end

    pair_lookup = Dict(p.seed => p for p in pairs)
    selected_pairs = [pair_lookup[s] for s in selected_seeds if haskey(pair_lookup, s)]

    if isempty(selected_pairs)
        return nothing
    end

    num_plots = length(selected_pairs)
    fig = Figure(size=(450 * num_plots, 400))
    senator_colors = [:blue, :green, :orange, :purple, :brown]

    for (col, pair) in enumerate(selected_pairs)
        ax = Axis(fig[1, col],
            title="Seed $(pair.seed)",
            xlabel="Opinion Dim 1",
            ylabel=col == 1 ? "Opinion Dim 2" : "",
            aspect=DataAspect()
        )

        # Draw obstacle regions if available
        config = nothing
        if !isnothing(pair.robust.params)
            config = get(pair.robust.params.player_configs, 2, nothing)
        end
        if !isnothing(config) && !isempty(config.obstacle_centers) && config.obstacle_weights[1] > 0
            for (center, offset) in zip(config.obstacle_centers, config.obstacle_sigmoid_offsets)
                if length(center) >= 2
                    # Draw obstacle as a circle at sigmoid offset radius
                    θ = range(0, 2π, length=64)
                    r = abs(offset)
                    obs_x = center[1] .+ r .* cos.(θ)
                    obs_y = center[2] .+ r .* sin.(θ)
                    poly!(ax, Point2f.(zip(obs_x, obs_y)), color=(:red, 0.1), strokecolor=(:red, 0.4), strokewidth=1)
                end
            end
        end

        # Draw goal ellipsoids
        if !isnothing(config) && !isempty(config.ellipsoid_centers)
            for center in config.ellipsoid_centers
                if length(center) >= 2
                    scatter!(ax, [center[1]], [center[2]], marker=:star5, markersize=15, color=(:gold, 0.7))
                end
            end
        end

        num_senators = length(pair.robust.gt_state_history[1].blocks)

        for (entry, linestyle, alpha) in [(pair.robust, :solid, 0.9), (pair.non_robust, :dash, 0.7)]
            for s in 1:num_senators
                xs = [state.blocks[s][1] for state in entry.gt_state_history]
                ys = [state.blocks[s][2] for state in entry.gt_state_history]
                color = senator_colors[mod1(s, length(senator_colors))]
                lines!(ax, xs, ys, color=(color, alpha), linestyle=linestyle, linewidth=2)
                scatter!(ax, [xs[1]], [ys[1]], color=color, marker=:circle, markersize=8)
                scatter!(ax, [xs[end]], [ys[end]], color=color, marker=:star5, markersize=10)
            end
        end
    end

    # Legend
    legend_elements = vcat(
        [LineElement(color=senator_colors[s], linewidth=2) for s in 1:min(3, length(senator_colors))],
        [LineElement(color=:black, linestyle=:solid, linewidth=2),
         LineElement(color=:black, linestyle=:dash, linewidth=2)]
    )
    legend_labels = vcat(
        ["Senator $s" for s in 1:min(3, length(senator_colors))],
        ["Robust P2", "Non-Robust P2"]
    )
    Legend(fig[2, 1:num_plots], legend_elements, legend_labels, orientation=:horizontal, tellwidth=false)

    filename = joinpath(directory, "seed_matched_trajectories.png")
    save(filename, fig; px_per_unit=3)
    save(joinpath(directory, "seed_matched_trajectories.pdf"), fig)
    println("Saved seed-matched trajectory plot to $filename")
    return fig
end

"""
    create_trajectory_divergence_plot(divergence_stats; directory)

Top: mean divergence over time with std band per senator.
Bottom: histogram of final-timestep divergence.
"""
function create_trajectory_divergence_plot(divergence_stats;
    directory::String="./exp/senate/outputs/analysis")

    mkpath(directory)
    senator_colors = [:blue, :green, :orange, :purple, :brown]

    fig = Figure(size=(800, 700))

    # Top: time series
    ax1 = Axis(fig[1, 1],
        title="Trajectory Divergence Over Time",
        xlabel="Timestep",
        ylabel="Mean ||GT_robust - GT_nonrobust||"
    )

    T = length(divergence_stats.mean_timeseries)
    ts = 1:T

    # Per-senator lines
    for (s, (sm, ss)) in enumerate(zip(divergence_stats.senator_mean_timeseries, divergence_stats.senator_std_timeseries))
        color = senator_colors[mod1(s, length(senator_colors))]
        band!(ax1, collect(ts), sm .- ss, sm .+ ss, color=(color, 0.15))
        lines!(ax1, collect(ts), sm, color=color, linewidth=2, label="Senator $s")
    end

    # Average
    band!(ax1, collect(ts), divergence_stats.mean_timeseries .- divergence_stats.std_timeseries,
          divergence_stats.mean_timeseries .+ divergence_stats.std_timeseries, color=(:black, 0.1))
    lines!(ax1, collect(ts), divergence_stats.mean_timeseries, color=:black, linewidth=2.5, linestyle=:dash, label="Average")
    axislegend(ax1, position=:lt)

    # Bottom: histogram of final divergences
    ax2 = Axis(fig[2, 1],
        title="Final Divergence Distribution ($(length(divergence_stats.final_divergences)) seeds)",
        xlabel="Mean Final Divergence",
        ylabel="Count"
    )
    hist!(ax2, divergence_stats.final_divergences, bins=30, color=(:steelblue, 0.7))
    vlines!(ax2, [mean(divergence_stats.final_divergences)], color=:red, linewidth=2, linestyle=:dash)
    text!(ax2, mean(divergence_stats.final_divergences), 0,
        text=@sprintf("μ=%.4f", mean(divergence_stats.final_divergences)),
        color=:red, fontsize=14, align=(:left, :bottom), offset=(5, 5))

    filename = joinpath(directory, "trajectory_divergence.png")
    save(filename, fig; px_per_unit=3)
    save(joinpath(directory, "trajectory_divergence.pdf"), fig)
    println("Saved divergence plot to $filename")
    return fig
end

"""
    create_cost_decomposition_comparison_plot(decomposition; directory)

Grouped bar chart comparing cumulative cost components (robust vs non-robust).
"""
function create_cost_decomposition_comparison_plot(decomposition;
    directory::String="./exp/senate/outputs/analysis")

    if isempty(decomposition)
        println("No decomposition data for comparison plot")
        return nothing
    end
    mkpath(directory)

    # Aggregate means and stds
    components = [:preference, :control, :covariance, :obstacle]
    component_labels = ["Preference", "Control", "Covariance", "Obstacle"]

    robust_means = Float64[]
    robust_stds = Float64[]
    nonrobust_means = Float64[]
    nonrobust_stds = Float64[]

    for comp in components
        r_vals = [getfield(d.robust, comp) for d in decomposition]
        nr_vals = [getfield(d.non_robust, comp) for d in decomposition]
        push!(robust_means, mean(r_vals))
        push!(robust_stds, length(r_vals) > 1 ? std(r_vals) : 0.0)
        push!(nonrobust_means, mean(nr_vals))
        push!(nonrobust_stds, length(nr_vals) > 1 ? std(nr_vals) : 0.0)
    end

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1],
        title="Cost Component Decomposition (P2)",
        ylabel="Cumulative Cost",
        xticks=(1:length(components), component_labels)
    )

    barwidth = 0.35
    xs = 1:length(components)

    barplot!(ax, collect(xs) .- barwidth/2, nonrobust_means, width=barwidth,
        color=(:red, 0.6), label="Non-Robust P2")
    barplot!(ax, collect(xs) .+ barwidth/2, robust_means, width=barwidth,
        color=(:blue, 0.6), label="Robust P2")
    errorbars!(ax, collect(xs) .- barwidth/2, nonrobust_means, nonrobust_stds, color=:red, whiskerwidth=8)
    errorbars!(ax, collect(xs) .+ barwidth/2, robust_means, robust_stds, color=:blue, whiskerwidth=8)

    axislegend(ax, position=:rt)

    filename = joinpath(directory, "cost_decomposition_comparison.png")
    save(filename, fig; px_per_unit=3)
    save(joinpath(directory, "cost_decomposition_comparison.pdf"), fig)
    println("Saved cost decomposition plot to $filename")
    return fig
end

"""
    create_paired_scatter_plot(decomposition; directory)

Scatter plot of robust vs non-robust total cost per seed. Points below diagonal = robustness wins.
"""
function create_paired_scatter_plot(decomposition;
    directory::String="./exp/senate/outputs/analysis")

    if isempty(decomposition)
        println("No decomposition data for scatter plot")
        return nothing
    end
    mkpath(directory)

    robust_costs = [d.robust.total for d in decomposition]
    nonrobust_costs = [d.non_robust.total for d in decomposition]

    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1],
        title="Paired Cost Comparison (P2, $(length(decomposition)) seeds)",
        xlabel="Non-Robust P2 Total Cost",
        ylabel="Robust P2 Total Cost",
        aspect=DataAspect()
    )

    scatter!(ax, nonrobust_costs, robust_costs, color=(:steelblue, 0.5), markersize=6)

    # y=x diagonal
    all_costs = vcat(robust_costs, nonrobust_costs)
    lo, hi = minimum(all_costs), maximum(all_costs)
    margin = 0.05 * (hi - lo)
    lines!(ax, [lo - margin, hi + margin], [lo - margin, hi + margin],
        color=:black, linestyle=:dash, linewidth=1.5)

    # Stats
    n_below = count(robust_costs .< nonrobust_costs)
    deltas = robust_costs .- nonrobust_costs
    mean_delta = mean(deltas)
    std_delta = std(deltas)
    cohens_d = std_delta > 0 ? mean_delta / std_delta : 0.0

    annotation = @sprintf("%d/%d below diagonal\nΔμ=%.3f, d=%.3f",
        n_below, length(decomposition), mean_delta, cohens_d)
    text!(ax, lo, hi, text=annotation, fontsize=12, align=(:left, :top), offset=(10, -10))

    filename = joinpath(directory, "paired_scatter.png")
    save(filename, fig; px_per_unit=3)
    save(joinpath(directory, "paired_scatter.pdf"), fig)
    println("Saved paired scatter plot to $filename")
    return fig
end

"""
    write_paired_analysis_report(pairs, divergence_stats, decomposition; directory)

Write text report with paired statistical tests and summary statistics.
"""
function write_paired_analysis_report(pairs, divergence_stats, decomposition;
    directory::String="./exp/senate/outputs/analysis")

    mkpath(directory)
    filename = joinpath(directory, "paired_analysis_report.txt")

    open(filename, "w") do io
        println(io, "=" ^ 60)
        println(io, "PAIRED TRAJECTORY ANALYSIS REPORT")
        println(io, "=" ^ 60)
        println(io)

        # Pair counts
        println(io, "Seed-matched pairs: $(length(pairs))")
        println(io)

        # Divergence statistics
        println(io, "--- TRAJECTORY DIVERGENCE ---")
        if !isempty(divergence_stats.final_divergences)
            fd = divergence_stats.final_divergences
            println(io, @sprintf("  Mean final divergence: %.6f ± %.6f", mean(fd), std(fd)))
            println(io, @sprintf("  Median final divergence: %.6f", median(fd)))
            println(io, @sprintf("  Fraction above 0.01 threshold: %.1f%%", 100 * divergence_stats.fraction_above_threshold))
            println(io, @sprintf("  Min/Max final divergence: %.6f / %.6f", minimum(fd), maximum(fd)))

            # Wilcoxon signed-rank test approximation (sign test as fallback)
            n_positive = count(d -> d > 0, fd)
            n_total = length(fd)
            # Under H0 (median=0), n_positive ~ Binomial(n_total, 0.5)
            # Two-sided p-value using normal approximation
            z_sign = (n_positive - n_total / 2) / sqrt(n_total / 4)
            println(io, @sprintf("  Sign test: %d/%d positive, z=%.2f", n_positive, n_total, z_sign))
        end
        println(io)

        # Cost decomposition
        println(io, "--- COST DECOMPOSITION (P2) ---")
        if !isempty(decomposition)
            for comp in [:preference, :control, :covariance, :obstacle, :total]
                r_vals = [getfield(d.robust, comp) for d in decomposition]
                nr_vals = [getfield(d.non_robust, comp) for d in decomposition]
                delta_vals = [getfield(d.delta, comp) for d in decomposition]
                println(io, @sprintf("  %-12s  Robust: %8.3f ± %6.3f  Non-Robust: %8.3f ± %6.3f  Δ: %+.3f ± %.3f",
                    string(comp), mean(r_vals), std(r_vals), mean(nr_vals), std(nr_vals),
                    mean(delta_vals), std(delta_vals)))
            end
            println(io)

            # Paired effect size on total cost
            total_deltas = [d.delta.total for d in decomposition]
            m = mean(total_deltas)
            s = std(total_deltas)
            d_cohen = s > 0 ? m / s : 0.0
            n_better = count(d -> d < 0, total_deltas)
            println(io, @sprintf("  Cohen's d (total): %.4f", d_cohen))
            println(io, @sprintf("  Seeds where robust is cheaper: %d/%d (%.1f%%)",
                n_better, length(total_deltas), 100 * n_better / length(total_deltas)))

            # Sign test on total cost
            z = (n_better - length(total_deltas) / 2) / sqrt(length(total_deltas) / 4)
            println(io, @sprintf("  Sign test (total cost): z=%.2f", z))
        end
        println(io)
        println(io, "=" ^ 60)
    end

    println("Saved paired analysis report to $filename")
end

# ========================================================================================
# SWEEP-LEVEL PAIRED ANALYSIS
# ========================================================================================

"""
    create_sweep_paired_analysis(sweep_collected, sweep_label; directory, sweep_name)

Top-level paired analysis for a sweep. For each sweep value, builds seed-matched pairs,
computes divergence and cost decomposition, and generates all paired analysis plots.
Also creates cross-sweep summary plots.
"""
function create_sweep_paired_analysis(sweep_collected::Dict, sweep_label::String;
    directory::String="./exp/senate/outputs/analysis",
    sweep_name::String="sweep")

    if isempty(sweep_collected)
        println("No sweep data for paired analysis")
        return
    end
    mkpath(directory)

    sweep_values = sort(collect(keys(sweep_collected)))
    sweep_divergence_means = Float64[]
    sweep_divergence_stds = Float64[]
    sweep_cost_deltas = Dict{Symbol, Vector{Float64}}()
    for comp in [:preference, :control, :covariance, :obstacle, :total]
        sweep_cost_deltas[comp] = Float64[]
    end
    valid_xs = Float64[]

    for sv in sweep_values
        data = sweep_collected[sv]
        pairs = build_seed_matched_pairs(data.robust_entries, data.non_robust_entries)
        if isempty(pairs)
            continue
        end

        sv_dir = joinpath(directory, "paired_$(sweep_name)_$(sv)")
        mkpath(sv_dir)

        # Divergence
        div_data = compute_trajectory_divergence(pairs)
        div_stats = compute_divergence_statistics(div_data)

        # Cost decomposition
        decomp = compute_paired_cost_decomposition(pairs)

        # Per-value plots
        create_seed_matched_trajectory_plot(pairs, div_data; directory=sv_dir)
        create_trajectory_divergence_plot(div_stats; directory=sv_dir)
        create_cost_decomposition_comparison_plot(decomp; directory=sv_dir)
        create_paired_scatter_plot(decomp; directory=sv_dir)
        write_paired_analysis_report(pairs, div_stats, decomp; directory=sv_dir)

        # Collect for cross-sweep summary
        push!(valid_xs, Float64(sv))
        push!(sweep_divergence_means, isempty(div_stats.final_divergences) ? NaN : mean(div_stats.final_divergences))
        push!(sweep_divergence_stds, isempty(div_stats.final_divergences) ? 0.0 :
            (length(div_stats.final_divergences) > 1 ? std(div_stats.final_divergences) : 0.0))

        if !isempty(decomp)
            for comp in [:preference, :control, :covariance, :obstacle, :total]
                delta_vals = [getfield(d.delta, comp) for d in decomp]
                push!(sweep_cost_deltas[comp], mean(delta_vals))
            end
        else
            for comp in [:preference, :control, :covariance, :obstacle, :total]
                push!(sweep_cost_deltas[comp], NaN)
            end
        end
    end

    if isempty(valid_xs)
        return
    end

    # Cross-sweep divergence summary
    fig_div = Figure(size=(700, 400))
    ax_div = Axis(fig_div[1, 1],
        title="Mean Final Divergence vs $sweep_label",
        xlabel=sweep_label,
        ylabel="Mean Final Trajectory Divergence"
    )
    errorbars!(ax_div, valid_xs, sweep_divergence_means, sweep_divergence_stds, color=(:steelblue, 0.5), whiskerwidth=8)
    scatterlines!(ax_div, valid_xs, sweep_divergence_means, color=:steelblue, markersize=10, linewidth=2)

    save(joinpath(directory, "sweep_divergence_$(sweep_name).png"), fig_div; px_per_unit=3)
    save(joinpath(directory, "sweep_divergence_$(sweep_name).pdf"), fig_div)
    println("Saved sweep divergence summary to $(joinpath(directory, "sweep_divergence_$(sweep_name).png"))")

    # Cross-sweep cost delta summary
    fig_cost = Figure(size=(700, 400))
    ax_cost = Axis(fig_cost[1, 1],
        title="Cost Component Deltas (Robust - Non-Robust) vs $sweep_label",
        xlabel=sweep_label,
        ylabel="Mean Δ Cost (Robust - Non-Robust)"
    )
    comp_colors = Dict(:preference => :blue, :control => :red, :obstacle => :orange, :total => :black)
    for (comp, color) in comp_colors
        vals = sweep_cost_deltas[comp]
        valid = findall(!isnan, vals)
        if !isempty(valid)
            scatterlines!(ax_cost, valid_xs[valid], vals[valid], color=color, markersize=8,
                linewidth=2, label=string(comp))
        end
    end
    hlines!(ax_cost, [0.0], color=:gray, linestyle=:dash, linewidth=1)
    axislegend(ax_cost, position=:lt)

    save(joinpath(directory, "sweep_cost_deltas_$(sweep_name).png"), fig_cost; px_per_unit=3)
    save(joinpath(directory, "sweep_cost_deltas_$(sweep_name).pdf"), fig_cost)
    println("Saved sweep cost delta summary to $(joinpath(directory, "sweep_cost_deltas_$(sweep_name).png"))")
end

end  # module
