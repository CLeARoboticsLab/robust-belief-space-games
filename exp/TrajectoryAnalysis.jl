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

# include("hockey/src/Hockey.jl")
using Hockey

# ========================================================================================
# TRAJECTORY ANALYSIS TRACKER SYSTEM
# ========================================================================================

export TrajectoryAnalysisEntry, TrajectoryAnalysisTracker, TRAJECTORY_TRACKER, clear_trajectory_tracker!,
    load_and_analyze_solution_files, compute_belief_covariance_traces, compute_player_distances,
    compute_belief_deviations, effect_of_nature, get_trajectory_summary, create_trajectory_analysis_plots, get_trajectory_details,
    create_yarnball_plot_for_cost_components, compare_robust_vs_nonrobust_actions

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
    params::HockeyParams

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

function load_and_analyze_solution_files(;
    directory="./outputs",
    file_pattern=r"")
    println("Loading trajectory data from solution files...")
    
    # Clear existing trajectory data
    clear_trajectory_tracker!()
    KKTErrorTracker.clear_rh_kkt_tracker!()
    
    # Find solution files
    solution_files = String[]
    
    if isdir(directory)
        all_files = readdir(directory)
        for f in all_files
            if file_pattern == r"" 
                if endswith(f, ".jld2")
                    push!(solution_files, joinpath(directory, f))
                end
            elseif occursin(file_pattern, f)
                push!(solution_files, joinpath(directory, f))
            end
        end
    end
    
    println("Found $(length(solution_files)) solution files")
    solutions = Dict()
    for solution_file in solution_files
        try
            loaded_data = load(solution_file)
            if haskey(loaded_data, "solutions")
                solutions = loaded_data["solutions"]
            else
                if haskey(loaded_data, "gt_state_history")
                    solutions = Dict(basename(solution_file) => loaded_data)
                else
                    continue
                end
            end
        catch e
            println("Error loading $solution_file: $e")
            continue
        end
        
        for (key, solution_data) in solutions
            noise_level = "unknown"
            is_robust = false
            trial_num = 1
            
            # Determine robustness - first try from params, then fallback to filename
            if occursin("robust", key)
                is_robust = !occursin("non_robust", key)
            end
            
            gt_state_history = nothing
            solution_history = nothing
            params = nothing
            
            if solution_data isa NamedTuple || solution_data isa Dict
                gt_state_history = get(solution_data, :gt_state_history, get(solution_data, "gt_state_history", nothing))
                solution_history = get(solution_data, :solution_history, get(solution_data, "solution_history", nothing))
                params = get(solution_data, :params, get(solution_data, "params", nothing))
                if !isnothing(solution_history)
                    extract_kkt_errors_from_history(solution_history, key, is_robust);
                end
            elseif solution_data isa Tuple
                 if length(solution_data) >= 4
                    gt_state_history = solution_data[1]
                    solution_history = solution_data[3]
                    params = solution_data[4]
                 end
            end
            
            # Update is_robust based on actual params if available
            if !isnothing(params) && hasproperty(params, :player_configs)
                defender_config = get(params.player_configs, 2, nothing)
                if !isnothing(defender_config) && hasproperty(defender_config, :type)
                    is_robust = (defender_config.type == Hockey.robust)
                end
            end
            
            costs = []
            
            if !isnothing(gt_state_history) && !isnothing(solution_history)
                entry = TrajectoryAnalysisEntry(
                    key,
                    trial_num,
                    gt_state_history,
                    [], # observations
                    solution_history,
                    [], # cond
                    [], # lq
                    costs,
                    params,
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

function extract_kkt_errors_from_history(solution_history, scenario_name, is_robust)
    for (player_idx, trajectory) in solution_history
        for (t, step_data) in enumerate(trajectory)
            if haskey(step_data, :kkt_error) && !isnothing(step_data.kkt_error)
                name = occursin(".jld2", scenario_name) ? scenario_name[1:end-5] : scenario_name
                trial_num = match(r"_trial_(\d+)", scenario_name)
                if !isnothing(trial_num)
                    trial_num = parse(Int, trial_num.captures[1])
                else
                    trial_num = -1
                end
                KKTErrorTracker.record_rh_kkt_error!(
                    name,
                    step_data.kkt_error,
                    trial_num, # trial (dummy)
                    t;
                    player=player_idx,
                    robust=is_robust,
                    additional_data=Dict{String,Any}("scenario_name" => scenario_name)
                )
            end
        end
    end
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
function get_trajectory_summary(;directory="../outputs/verification_sweep")
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
        # TODO
        explicit_covariance = false
        scenario_planned_trajectory_costs = [calculate_planned_trajectory_costs(e, explicit_covariance) for e in group_entries]
        all_planned_costs[base] = scenario_planned_trajectory_costs
    end
    
    compare_robust_vs_nonrobust_actions(TRAJECTORY_TRACKER.entries; directory=directory)
    create_yarnball_plot_for_cost_components(all_planned_costs, TRAJECTORY_TRACKER.entries; directory=directory)
    return nothing
end

"""
    extract_base_config_for_comparison(scenario_name::String)

Extract the base configuration from a scenario name by removing the 
"_robust" or "_non_robust" suffix. This allows matching robust and non-robust
entries that share the same parameter configuration.
"""
function extract_base_config_for_comparison(scenario_name::String)
    # Remove "_robust" or "_non_robust" suffix
    if endswith(scenario_name, "_robust")
        return scenario_name[1:end-7]  # Remove "_robust" (7 chars)
    elseif endswith(scenario_name, "_non_robust")
        return scenario_name[1:end-11]  # Remove "_non_robust" (11 chars)
    else
        return scenario_name
    end
end

"""
    compare_robust_vs_nonrobust_actions(all_entries)

Create grid plots showing action differences between robust and non-robust cases.
Groups entries by base configuration (shared parameters) and compares robust vs non-robust.
"""
function compare_robust_vs_nonrobust_actions(all_entries; directory="../outputs/verification_sweep")
    robust_entries = [e for e in all_entries if e.robust]
    non_robust_entries = [e for e in all_entries if !e.robust]
    
    if isempty(robust_entries) || isempty(non_robust_entries)
        println("Skipping action comparison for $base_config: missing robust or non-robust entries")
    end
    create_action_difference_plots(robust_entries, non_robust_entries, extract_base_config_for_comparison(all_entries[1].scenario_name); directory=directory)
end

"""
    get_solutions_at_time(hist, t)

Helper function to extract solutions at time t from solution_history, 
handling both player-indexed and time-indexed formats.
"""
function get_solutions_at_time(hist, t)
    is_player_indexed = isa(hist, Dict) && haskey(hist, 1) && isa(hist[1], AbstractVector)
    
    if is_player_indexed
        # Player-indexed format: hist[player_idx][time_step]
        p1 = (haskey(hist, 1) && length(hist[1]) >= t) ? hist[1][t] : ([], [])
        p2 = (haskey(hist, 2) && length(hist[2]) >= t) ? hist[2][t] : ([], [])
        return [p1, p2]
    else
        # Time-indexed format: hist[time_step][player_idx]
        return hist[t]
    end
end

"""
    get_num_time_steps(hist)

Helper function to get the number of time steps in solution_history.
"""
function get_num_time_steps(hist)
    is_player_indexed = isa(hist, Dict) && haskey(hist, 1) && isa(hist[1], AbstractVector)
    
    if is_player_indexed
        return length(hist[1])
    else
        return length(hist)
    end
end

"""
    extract_executed_controls(entry::TrajectoryAnalysisEntry)

Extract the first control from each RH step (the executed control) for each player.
- Attacker controls: sols[1][2][1][1:2] (from player 1's solution)
- Defender controls: sols[2][2][1][3:4] (from player 2's solution)

Returns Dict with :attacker => Vector of 2D controls, :defender => Vector of 2D controls
"""
function extract_executed_controls(entry::TrajectoryAnalysisEntry)
    hist = entry.solution_history
    num_time_steps = get_num_time_steps(hist)
    
    executed_controls = Dict{Symbol, Vector{Vector{Float64}}}()
    executed_controls[:attacker] = Vector{Float64}[]
    executed_controls[:defender] = Vector{Float64}[]
    
    for t in 1:num_time_steps
        sols = get_solutions_at_time(hist, t)
        
        # Attacker controls from player 1's solution: sols[1][2][1][1:2]
        if length(sols) >= 1 && length(sols[1]) >= 2
            controls_traj_p1 = sols[1][2]
            if !isempty(controls_traj_p1) && length(controls_traj_p1[1]) >= 2
                push!(executed_controls[:attacker], Vector{Float64}(controls_traj_p1[1][1:2]))
            end
        end
        
        # Defender controls from player 2's solution: sols[2][2][1][3:4]
        if length(sols) >= 2 && length(sols[2]) >= 2
            controls_traj_p2 = sols[2][2]
            if !isempty(controls_traj_p2) && length(controls_traj_p2[1]) >= 4
                push!(executed_controls[:defender], Vector{Float64}(controls_traj_p2[1][3:4]))
            end
        end
    end
    
    return executed_controls
end

"""
    control_angle(u)

Compute the angle (in radians) of a 2D control vector.
"""
function control_angle(u)
    return atan(u[2], u[1])
end

"""
    cosine_similarity(a, b)

Compute cosine similarity between two vectors.
"""
function cosine_similarity(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a < 1e-10 || norm_b < 1e-10
        return 0.0  # Avoid division by zero
    end
    return dot(a, b) / (norm_a * norm_b)
end

"""
    create_action_difference_plots(robust_entries, non_robust_entries, config_name)

Create plots comparing executed controls between robust and non-robust strategies.

Creates 3 figures:
1. Norm and angle of controls for all 4 trajectories
2. Difference in norm and angle between robust and non-robust for each player
3. 2D control space evolution
"""
function create_action_difference_plots(robust_entries, non_robust_entries, config_name; directory="../outputs/verification_sweep")
    # Determine the minimum number of time steps across all entries
    min_time_steps = minimum(vcat(
        [get_num_time_steps(e.solution_history) for e in robust_entries],
        [get_num_time_steps(e.solution_history) for e in non_robust_entries]
    ))
    
    if min_time_steps == 0
        println("Skipping action comparison for $config_name: no solution history available")
        return nothing
    end
    
    # Create short display name for title
    display_name = length(config_name) > 40 ? "..." * config_name[end-36:end] : config_name
    
    # Extract controls for all entries
    robust_controls_all = [extract_executed_controls(e) for e in robust_entries]
    non_robust_controls_all = [extract_executed_controls(e) for e in non_robust_entries]
    
    # Colors: 4 distinct colors for the 4 trajectories
    color_attacker_robust = :red
    color_attacker_non_robust = :orange
    color_defender_robust = :blue
    color_defender_non_robust = :purple
    
    # Safe filename
    safe_name = replace(config_name, r"[^a-zA-Z0-9_]" => "_")
    if length(safe_name) > 50
        safe_name = safe_name[end-49:end]
    end
    
    # ========== Figure 1: Norm and Angle for all 4 trajectories ==========
    fig1 = Figure(size=(1400, 600))
    Label(fig1[0, :], text = "Control Norm & Angle: $display_name", fontsize = 16)
    
    # Control Norms
    ax_norm = Axis(fig1[1, 1], 
        title = "Control Norms",
        xlabel = "Time Step",
        ylabel = "Control Norm (L2)"
    )
    
    # Control Angles
    ax_angle = Axis(fig1[2, 1], 
        title = "Control Angles",
        xlabel = "Time Step",
        ylabel = "Angle (radians)"
    )
    
    # Helper to compute stats
    function compute_stats_over_trials(data_list)
        if isempty(data_list) return Float64[], Float64[], Float64[] end
        max_len = maximum(length(d) for d in data_list)
        means = Float64[]
        stds = Float64[]
        for t in 1:max_len
            vals = [d[t] for d in data_list if length(d) >= t]
            if !isempty(vals)
                push!(means, mean(vals))
                push!(stds, length(vals) > 1 ? std(vals) : 0.0)
            end
        end
        return 1:length(means), means, stds
    end
    
    # Collect all norms and angles for each player/robustness combination
    robust_att_norms = Vector{Vector{Float64}}()
    robust_att_angles = Vector{Vector{Float64}}()
    robust_def_norms = Vector{Vector{Float64}}()
    robust_def_angles = Vector{Vector{Float64}}()
    
    for controls in robust_controls_all
        if length(controls[:attacker]) >= min_time_steps
            push!(robust_att_norms, [norm(u) for u in controls[:attacker][1:min_time_steps]])
            push!(robust_att_angles, [control_angle(u) for u in controls[:attacker][1:min_time_steps]])
        end
        if length(controls[:defender]) >= min_time_steps
            push!(robust_def_norms, [norm(u) for u in controls[:defender][1:min_time_steps]])
            push!(robust_def_angles, [control_angle(u) for u in controls[:defender][1:min_time_steps]])
        end
    end
    
    non_robust_att_norms = Vector{Vector{Float64}}()
    non_robust_att_angles = Vector{Vector{Float64}}()
    non_robust_def_norms = Vector{Vector{Float64}}()
    non_robust_def_angles = Vector{Vector{Float64}}()
    
    for controls in non_robust_controls_all
        if length(controls[:attacker]) >= min_time_steps
            push!(non_robust_att_norms, [norm(u) for u in controls[:attacker][1:min_time_steps]])
            push!(non_robust_att_angles, [control_angle(u) for u in controls[:attacker][1:min_time_steps]])
        end
        if length(controls[:defender]) >= min_time_steps
            push!(non_robust_def_norms, [norm(u) for u in controls[:defender][1:min_time_steps]])
            push!(non_robust_def_angles, [control_angle(u) for u in controls[:defender][1:min_time_steps]])
        end
    end
    
    # Plot robust attacker
    if !isempty(robust_att_norms)
        ts, means, stds = compute_stats_over_trials(robust_att_norms)
        band!(ax_norm, collect(ts), means .- stds, means .+ stds, color=(color_attacker_robust, 0.2))
        lines!(ax_norm, collect(ts), means, color=color_attacker_robust, linewidth=2, label="Attacker (Robust)")
        
        ts, means, stds = compute_stats_over_trials(robust_att_angles)
        band!(ax_angle, collect(ts), means .- stds, means .+ stds, color=(color_attacker_robust, 0.2))
        lines!(ax_angle, collect(ts), means, color=color_attacker_robust, linewidth=2)
    end
    
    # Plot robust defender
    if !isempty(robust_def_norms)
        ts, means, stds = compute_stats_over_trials(robust_def_norms)
        band!(ax_norm, collect(ts), means .- stds, means .+ stds, color=(color_defender_robust, 0.2))
        lines!(ax_norm, collect(ts), means, color=color_defender_robust, linewidth=2, label="Defender (Robust)")
        
        ts, means, stds = compute_stats_over_trials(robust_def_angles)
        band!(ax_angle, collect(ts), means .- stds, means .+ stds, color=(color_defender_robust, 0.2))
        lines!(ax_angle, collect(ts), means, color=color_defender_robust, linewidth=2)
    end
    
    # Plot non-robust attacker
    if !isempty(non_robust_att_norms)
        ts, means, stds = compute_stats_over_trials(non_robust_att_norms)
        band!(ax_norm, collect(ts), means .- stds, means .+ stds, color=(color_attacker_non_robust, 0.2))
        lines!(ax_norm, collect(ts), means, color=color_attacker_non_robust, linewidth=2, linestyle=:dash, label="Attacker (Non-Robust)")
        
        ts, means, stds = compute_stats_over_trials(non_robust_att_angles)
        band!(ax_angle, collect(ts), means .- stds, means .+ stds, color=(color_attacker_non_robust, 0.2))
        lines!(ax_angle, collect(ts), means, color=color_attacker_non_robust, linewidth=2, linestyle=:dash)
    end
    
    # Plot non-robust defender
    if !isempty(non_robust_def_norms)
        ts, means, stds = compute_stats_over_trials(non_robust_def_norms)
        band!(ax_norm, collect(ts), means .- stds, means .+ stds, color=(color_defender_non_robust, 0.2))
        lines!(ax_norm, collect(ts), means, color=color_defender_non_robust, linewidth=2, linestyle=:dash, label="Defender (Non-Robust)")
        
        ts, means, stds = compute_stats_over_trials(non_robust_def_angles)
        band!(ax_angle, collect(ts), means .- stds, means .+ stds, color=(color_defender_non_robust, 0.2))
        lines!(ax_angle, collect(ts), means, color=color_defender_non_robust, linewidth=2, linestyle=:dash)
    end
    
    axislegend(ax_norm, position=:rt)
    
    filename1 = "$(directory)/control_norm_angle_$(safe_name).png"
    save(filename1, fig1)
    save("$(directory)/control_norm_angle_$(safe_name).pdf", fig1)
    println("Saved control norm/angle plot to $filename1")
    
    # ========== Figure 2: Difference in Norm and Angle (robust - non-robust) ==========
    fig2 = Figure(size=(800, 800))
    Label(fig2[0, :], text = "Control Differences (Robust - Non-Robust): $display_name", fontsize = 16)
    
    n_pairs = min(length(robust_entries), length(non_robust_entries))
    
    # Norm difference
    ax_norm_diff = Axis(fig2[1, 1], 
        title = "Norm Difference (Robust - Non-Robust)",
        xlabel = "Time Step",
        ylabel = "Δ Norm"
    )
    hlines!(ax_norm_diff, [0.0], color=:gray, linestyle=:dash, linewidth=1)
    
    # Angle difference
    ax_angle_diff = Axis(fig2[2, 1], 
        title = "Angle Difference (Robust - Non-Robust)",
        xlabel = "Time Step",
        ylabel = "Δ Angle (radians)"
    )
    hlines!(ax_angle_diff, [0.0], color=:gray, linestyle=:dash, linewidth=1)
    
    # Collect all n×m pairwise differences (every robust vs every non-robust)
    attacker_norm_diffs_all = Vector{Vector{Float64}}()
    attacker_angle_diffs_all = Vector{Vector{Float64}}()
    defender_norm_diffs_all = Vector{Vector{Float64}}()
    defender_angle_diffs_all = Vector{Vector{Float64}}()
    
    for robust_controls in robust_controls_all
        for non_robust_controls in non_robust_controls_all
            # Attacker differences
            r_att = robust_controls[:attacker]
            nr_att = non_robust_controls[:attacker]
            min_len_att = min(length(r_att), length(nr_att), min_time_steps)
            
            if min_len_att > 0
                norm_diffs_att = [norm(r_att[t]) - norm(nr_att[t]) for t in 1:min_len_att]
                angle_diffs_att = [control_angle(r_att[t]) - control_angle(nr_att[t]) for t in 1:min_len_att]
                push!(attacker_norm_diffs_all, norm_diffs_att)
                push!(attacker_angle_diffs_all, angle_diffs_att)
            end
            
            # Defender differences
            r_def = robust_controls[:defender]
            nr_def = non_robust_controls[:defender]
            min_len_def = min(length(r_def), length(nr_def), min_time_steps)
            
            if min_len_def > 0
                norm_diffs_def = [norm(r_def[t]) - norm(nr_def[t]) for t in 1:min_len_def]
                angle_diffs_def = [control_angle(r_def[t]) - control_angle(nr_def[t]) for t in 1:min_len_def]
                push!(defender_norm_diffs_all, norm_diffs_def)
                push!(defender_angle_diffs_all, angle_diffs_def)
            end
        end
    end
    
    # Helper function to compute mean and std at each time step
    function compute_mean_std(diffs_all)
        if isempty(diffs_all)
            return Float64[], Float64[], Float64[]
        end
        max_len = maximum(length(d) for d in diffs_all)
        means = Float64[]
        stds = Float64[]
        for t in 1:max_len
            values_at_t = [d[t] for d in diffs_all if length(d) >= t]
            if !isempty(values_at_t)
                push!(means, mean(values_at_t))
                push!(stds, length(values_at_t) > 1 ? std(values_at_t) : 0.0)
            end
        end
        return 1:length(means), means, stds
    end
    
    # Plot attacker differences (mean ± std)
    ts_att_norm, mean_att_norm, std_att_norm = compute_mean_std(attacker_norm_diffs_all)
    if !isempty(ts_att_norm)
        band!(ax_norm_diff, collect(ts_att_norm), mean_att_norm .- std_att_norm, mean_att_norm .+ std_att_norm, 
            color=(color_attacker_robust, 0.2))
        lines!(ax_norm_diff, collect(ts_att_norm), mean_att_norm, 
            color=color_attacker_robust, linewidth=2, label="Attacker")
    end
    
    ts_att_angle, mean_att_angle, std_att_angle = compute_mean_std(attacker_angle_diffs_all)
    if !isempty(ts_att_angle)
        band!(ax_angle_diff, collect(ts_att_angle), mean_att_angle .- std_att_angle, mean_att_angle .+ std_att_angle, 
            color=(color_attacker_robust, 0.2))
        lines!(ax_angle_diff, collect(ts_att_angle), mean_att_angle, 
            color=color_attacker_robust, linewidth=2)
    end
    
    # Plot defender differences (mean ± std)
    ts_def_norm, mean_def_norm, std_def_norm = compute_mean_std(defender_norm_diffs_all)
    if !isempty(ts_def_norm)
        band!(ax_norm_diff, collect(ts_def_norm), mean_def_norm .- std_def_norm, mean_def_norm .+ std_def_norm, 
            color=(color_defender_robust, 0.2))
        lines!(ax_norm_diff, collect(ts_def_norm), mean_def_norm, 
            color=color_defender_robust, linewidth=2, label="Defender")
    end
    
    ts_def_angle, mean_def_angle, std_def_angle = compute_mean_std(defender_angle_diffs_all)
    if !isempty(ts_def_angle)
        band!(ax_angle_diff, collect(ts_def_angle), mean_def_angle .- std_def_angle, mean_def_angle .+ std_def_angle, 
            color=(color_defender_robust, 0.2))
        lines!(ax_angle_diff, collect(ts_def_angle), mean_def_angle, 
            color=color_defender_robust, linewidth=2)
    end
    
    axislegend(ax_norm_diff, position=:rt)
    
    filename2 = "$(directory)/control_differences_$(safe_name).png"
    save(filename2, fig2)
    save("$(directory)/control_differences_$(safe_name).pdf", fig2)
    println("Saved control differences plot to $filename2")
    
    # ========== Figure 3: 2D Control Trajectory Evolution ==========
    fig3 = Figure(size=(1600, 600))
    Label(fig3[0, :], text = "2D Control Evolution: $display_name", fontsize = 16)
    
    player_symbols = [:attacker, :defender]
    player_names = ["Attacker", "Defender"]
    colors_robust = [color_attacker_robust, color_defender_robust]
    colors_non_robust = [color_attacker_non_robust, color_defender_non_robust]
    
    for (col, (player_sym, player_name)) in enumerate(zip(player_symbols, player_names))
        ax = Axis(fig3[1, col], 
            title = "$player_name Control Space",
            xlabel = "Control u₁",
            ylabel = "Control u₂",
        )
        
        all_xs = Float64[]
        all_ys = Float64[]
        
        robust_xs = Float64[]
        robust_ys = Float64[]
        non_robust_xs = Float64[]
        non_robust_ys = Float64[]
        
        # Collect robust trials data
        for controls in robust_controls_all
            player_controls = controls[player_sym]
            if length(player_controls) >= min_time_steps
                xs = [u[1] for u in player_controls[1:min_time_steps]]
                ys = [u[2] for u in player_controls[1:min_time_steps]]
                append!(all_xs, xs)
                append!(all_ys, ys)
                append!(robust_xs, xs)
                append!(robust_ys, ys)
            end
        end
        
        # Collect non-robust trials data
        for controls in non_robust_controls_all
            player_controls = controls[player_sym]
            if length(player_controls) >= min_time_steps
                xs = [u[1] for u in player_controls[1:min_time_steps]]
                ys = [u[2] for u in player_controls[1:min_time_steps]]
                append!(all_xs, xs)
                append!(all_ys, ys)
                append!(non_robust_xs, xs)
                append!(non_robust_ys, ys)
            end
        end
        
        # Plot as scatter with low opacity to create colored bands
        if !isempty(robust_xs)
            scatter!(ax, robust_xs, robust_ys, 
                    color=(colors_robust[col], 0.15), 
                    markersize=10,
                    label="Robust")
        end
        
        if !isempty(non_robust_xs)
            scatter!(ax, non_robust_xs, non_robust_ys, 
                    color=(colors_non_robust[col], 0.15), 
                    markersize=10,
                    marker=:diamond,
                    label="Non-Robust")
        end
        
        # Set axis limits with padding
        if !isempty(all_xs) && !isempty(all_ys)
            x_range = maximum(all_xs) - minimum(all_xs)
            y_range = maximum(all_ys) - minimum(all_ys)
            x_pad = max(0.2 * x_range, 0.1)  # At least 0.1 padding
            y_pad = max(0.2 * y_range, 0.1)
            xlims!(ax, minimum(all_xs) - x_pad, maximum(all_xs) + x_pad)
            ylims!(ax, minimum(all_ys) - y_pad, maximum(all_ys) + y_pad)
        end
        
        axislegend(ax, position=:rb)
    end
    
    filename3 = "$(directory)/control_trajectory_2d_$(safe_name).png"
    save(filename3, fig3)
    save("$(directory)/control_trajectory_2d_$(safe_name).pdf", fig3)
    println("Saved 2D control trajectory plot to $filename3")
    
    return nothing
end

"""
    compute_executed_trajectory_costs(entry::TrajectoryAnalysisEntry, explicit_covariance::Bool)

Calculate the costs incurred for the actual executed trajectory using each player's own beliefs.
"""
function compute_executed_trajectory_costs(entry::TrajectoryAnalysisEntry, explicit_covariance::Bool)
    player_names = [:attacker, :defender, :nature]
    
    if isempty(entry.gt_state_history) || isempty(entry.solution_history)
        return []
    end
    
    executed_costs = []
    num_players = 2
    
    # Check format
    hist = entry.solution_history
    is_player_indexed = isa(hist, Dict) && haskey(hist, 1) && isa(hist[1], AbstractVector)
    
    for t in 1:length(entry.gt_state_history)
        cost_breakdown = Dict()
        
        # Determine sols for time t
        sols = nothing
        if is_player_indexed
            # Check if we have plan for time t
            if length(hist[1]) >= t
                p1 = hist[1][t]
                p2 = (haskey(hist, 2) && length(hist[2]) >= t) ? hist[2][t] : ([],[])
                sols = [p1, p2]
            end
        else
            # Old format
            if t <= length(hist) && haskey(hist, t)
                sols = hist[t]
            end
        end
        
        if isnothing(sols)
            continue
        end
        
        for (player_idx, player_name) in enumerate(player_names[1:num_players])
            
            if player_idx > length(sols)
                cost_breakdown[player_name] = NamedTuple()
                continue
            end
             
            val = sols[player_idx]
            if length(val) < 2
                cost_breakdown[player_name] = NamedTuple()
                continue
            end

            beliefs_traj, controls_traj = val
            
            if isempty(beliefs_traj) || isempty(controls_traj)
                cost_breakdown[player_name] = NamedTuple()
                continue
            else
                # Restored logic: Extract beliefs for usage below
                current_beliefs = beliefs_traj[1]
                belief_indices = if player_name == :attacker
                    (1, 2)
                else
                    (3, 4)
                end
                
                # Check bounds
                if length(current_beliefs.beliefs) >= belief_indices[2]
                    attacker_belief = current_beliefs.beliefs[belief_indices[1]]
                    defender_belief = current_beliefs.beliefs[belief_indices[2]]
                else
                    # Fallback if beliefs are missing
                    cost_breakdown[player_name] = NamedTuple()
                    continue
                end

                # Calculate cost components using the player's own beliefs
                if t == length(entry.gt_state_history)
                    costs = Hockey.player_cost_components[player_name].terminal(
                        attacker_belief,
                        defender_belief,
                        entry.params;
                        explicit_covariance=explicit_covariance
                    )
                else
                    executed_control = controls_traj[1]
                    costs = Hockey.player_cost_components[player_name].non_terminal(
                        attacker_belief, 
                        defender_belief,
                        executed_control,
                        entry.params;
                        explicit_covariance=explicit_covariance
                    )
                end
                
                cost_breakdown[player_name] = costs
            end
        end
        
        push!(executed_costs, cost_breakdown)
    end
    
    return executed_costs
end


"""
Calculate the costs incurred for each planned trajectory
"""
function calculate_planned_trajectory_costs(entry::TrajectoryAnalysisEntry, explicit_covariance::Bool)
    player_names = [:attacker, :defender, :nature]    
    
    # Determine time steps based on format
    hist = entry.solution_history
    is_player_indexed = isa(hist, Dict) && haskey(hist, 1) && isa(hist[1], AbstractVector)
    
    time_steps = is_player_indexed ? (1:length(hist[1])) : sort(collect(keys(hist)))
    
    map(time_steps) do t
        # Get solutions for this time step
        sols = nothing
        if is_player_indexed
            # Construct tuple/vector [P1_plan, P2_plan]
            p1 = length(hist[1]) >= t ? hist[1][t] : ([],[])
            p2 = (haskey(hist, 2) && length(hist[2]) >= t) ? hist[2][t] : ([],[])
            sols = [p1, p2]
        else
            sols = hist[t]
        end
        
        cost_breakdown = Dict()
        num_players = 2
        
        for (player_idx, player_name) in enumerate(player_names[1:num_players])
            # an entry for each time step, which is a named tuple of cost components
            trajectory_costs = []
            
            if player_idx > length(sols)
                cost_breakdown[player_name] = []
                continue
            end
            
            val = sols[player_idx]
            if length(val) < 2
                cost_breakdown[player_name] = []
                continue
            end
            
            beliefs_traj, controls_traj = val
            
            beliefs_traj, controls_traj = val
            
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
            for k in 1:(planning_horizon - 1)
                
                attacker_belief = beliefs_traj[k].beliefs[belief_indices[1]]
                defender_belief = beliefs_traj[k].beliefs[belief_indices[2]]

                non_terminal_costs = Hockey.player_cost_components[player_name].non_terminal(
                    attacker_belief, 
                    defender_belief,
                    controls_traj[k],
                    entry.params;
                    explicit_covariance=explicit_covariance
                )
                
                push!(trajectory_costs, non_terminal_costs)
            end
            
            # Terminal cost
            attacker_belief = beliefs_traj[end].beliefs[belief_indices[1]]
            defender_belief = beliefs_traj[end].beliefs[belief_indices[2]]
            terminal_costs = Hockey.player_cost_components[player_name].terminal(
                attacker_belief,
                defender_belief,
                entry.params;
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
    
    save("trajectory_belief_covariances.png", fig);
    println("Belief covariance plot saved as trajectory_belief_covariances.png")
    return nothing
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
    
    save("trajectory_player_distances.png", fig);
    println("Player distance plot saved as trajectory_player_distances.png")
    return nothing
end

"""
    create_planned_trajectory_costs_plots(planned_trajectory_costs)

Create plots showing planned trajectory costs over time.
"""
function create_planned_trajectory_costs_plots(planned_trajectory_costs)
    
end

function create_yarnball_plot_for_cost_components(all_planned_costs, all_entries; directory="../outputs/verification_sweep")
    println("Generating yarnball plots for cost components...")

    for (scenario, scenario_costs) in all_planned_costs # scenario_costs is for all trials of a scenario
        
        if isempty(scenario_costs) continue end
        num_rh_steps = length(scenario_costs[1])
        if num_rh_steps == 0 continue end

        # Get all unique components across all RH steps and players
        all_components = Set{Symbol}()
        player_names = Set{Symbol}()
        
        for trial_data in scenario_costs
            for rh_step in 1:min(length(trial_data), num_rh_steps)
                rh_step_data = trial_data[rh_step]
                for player_name in keys(rh_step_data)
                    push!(player_names, player_name)
                    plan_traj_costs = rh_step_data[player_name]
                    if !isempty(plan_traj_costs)
                        for step in plan_traj_costs
                            union!(all_components, keys(step))
                        end
                    end
                end
            end
        end
        
        component_names = sort(collect(all_components), by=string)
        player_names = sort(collect(player_names))
        
        if isempty(component_names) || isempty(player_names) continue end
        
        # Create grid: rows = RH steps + 1 (for executed trajectory), columns = cost components + 1 (for total cost)
        num_components = length(component_names)
        num_rh_steps_actual = min(num_rh_steps, 10)  # Limit to first 10 RH steps for readability
        num_rows = num_rh_steps_actual + 1  # +1 for executed trajectory row
        num_cols = num_components + 1  # +1 for total cost column
        
        fig = Figure(size=(400 * num_cols, 300 * num_rows))
        Label(fig[0, :], text = "$scenario - Cost Components Over Time", fontsize = 24)
        
        player_colors = Dict(zip(player_names, [:blue, :red, :green, :orange, :purple]))

        # Helper to compute stats across trials
        function get_component_stats(data_list)
            if isempty(data_list) return Float64[], Float64[], Float64[] end
            max_len = maximum(length(d) for d in data_list)
            means = Float64[]
            stds = Float64[]
            ts = 1:max_len
            
            for t in ts
                vals = [d[t] for d in data_list if length(d) >= t]
                if !isempty(vals)
                    push!(means, mean(vals))
                    push!(stds, length(vals) > 1 ? std(vals) : 0.0)
                else
                    push!(means, NaN)
                    push!(stds, NaN)
                end
            end
            return collect(ts), means, stds
        end

        for rh_step in 1:num_rh_steps_actual
            for (comp_idx, component) in enumerate(component_names)
                ax = Axis(fig[rh_step, comp_idx], 
                    title = rh_step == 1 ? string(component) : "",  # Only show component name on top row
                    xlabel = rh_step == num_rh_steps_actual ? "Planning Horizon Step" : "",  # Only show xlabel on bottom row
                    ylabel = comp_idx == 1 ? "RH Step $rh_step" : ""  # Only show ylabel on left column
                )

                for player_name in player_names
                    color = player_colors[player_name]
                    
                    # Collect trajectories for this component across all trials
                    trajectories = Vector{Vector{Float64}}()
                    
                    for trial_data in scenario_costs # loop over trials
                        if rh_step > length(trial_data) continue end
                        rh_step_data = trial_data[rh_step] 
                        if !haskey(rh_step_data, player_name) continue end
                        plan_traj_costs = rh_step_data[player_name]
                        if isempty(plan_traj_costs) continue end
                        
                        traj = [get(step, component, 0.0) for step in plan_traj_costs]
                        push!(trajectories, traj)
                    end
                    
                    if !isempty(trajectories)
                        ts, means, stds = get_component_stats(trajectories)
                        valid_idx = findall(.!isnan.(means))
                        if !isempty(valid_idx)
                             band!(ax, ts[valid_idx], means[valid_idx] .- stds[valid_idx], means[valid_idx] .+ stds[valid_idx], 
                                   color=(color, 0.2))
                             lines!(ax, ts[valid_idx], means[valid_idx], color=color, linewidth=2)
                        end
                    end
                end

                # Add legend only to the first subplot
                if rh_step == 1 && comp_idx == 1
                    elements = [LineElement(color = player_colors[p], linestyle = :solid, linewidth=2) for p in player_names]
                    axislegend(ax, elements, string.(player_names), "Players")
                end
            end
            
            # Plot total cost column for planned trajectories
            ax_total = Axis(fig[rh_step, num_cols], 
                title = rh_step == 1 ? "Total Cost" : "",
                xlabel = rh_step == num_rh_steps_actual ? "Planning Horizon Step" : "",
                ylabel = "RH Step $rh_step"
            )
            
            for player_name in player_names
                color = player_colors[player_name]
                trajectories = Vector{Vector{Float64}}()

                for trial_data in scenario_costs
                    if rh_step > length(trial_data) continue end
                    rh_step_data = trial_data[rh_step] 
                    if !haskey(rh_step_data, player_name) continue end
                    plan_traj_costs = rh_step_data[player_name]
                    if isempty(plan_traj_costs) continue end
                    
                    # Calculate total cost for each step
                    total_trajectory = Float64[]
                    for step in plan_traj_costs
                        comp_sum = sum(values(step))
                        push!(total_trajectory, comp_sum)
                    end
                    push!(trajectories, total_trajectory)
                end
                
                if !isempty(trajectories)
                    ts, means, stds = get_component_stats(trajectories)
                    valid_idx = findall(.!isnan.(means))
                    if !isempty(valid_idx)
                         band!(ax_total, ts[valid_idx], means[valid_idx] .- stds[valid_idx], means[valid_idx] .+ stds[valid_idx], 
                               color=(color, 0.2))
                         lines!(ax_total, ts[valid_idx], means[valid_idx], color=color, linewidth=2)
                    end
                end
            end
        end
        
        # Get entries for this scenario
        scenario_entries = [e for e in all_entries if base_group(e.scenario_name) == scenario]
        
        # Plot executed trajectory costs (bottom row)
        for (comp_idx, component) in enumerate(component_names)
            ax = Axis(fig[num_rows, comp_idx], 
                title = string(component),
                xlabel = "Execution Time Step",
                ylabel = "Executed"
            )
            
            for player_name in player_names
                color = player_colors[player_name]
                trajectories = Vector{Vector{Float64}}()
                
                for entry in scenario_entries
                    # Compute executed trajectory costs
                    executed_costs = compute_executed_trajectory_costs(entry, false)
                    
                    if !isempty(executed_costs) && haskey(executed_costs[1], player_name)
                        component_trajectory = Float64[]
                        for time_step_costs in executed_costs
                            if haskey(time_step_costs, player_name) && !isempty(time_step_costs[player_name])
                                cost_value = get(time_step_costs[player_name], component, 0.0)
                                push!(component_trajectory, cost_value)
                            end
                        end
                        if !isempty(component_trajectory)
                            push!(trajectories, component_trajectory)
                        end
                    end
                end
                
                if !isempty(trajectories)
                    ts, means, stds = get_component_stats(trajectories)
                    valid_idx = findall(.!isnan.(means))
                    if !isempty(valid_idx)
                        band!(ax, ts[valid_idx], means[valid_idx] .- stds[valid_idx], means[valid_idx] .+ stds[valid_idx], 
                            color=(color, 0.2))
                        lines!(ax, ts[valid_idx], means[valid_idx], color=color, linewidth=2)
                    end
                end
            end
        end
        
        # Plot total cost for executed trajectory (bottom right)
        ax_executed_total = Axis(fig[num_rows, num_cols], 
            title = "Total Cost",
            xlabel = "Execution Time Step",
            ylabel = "Executed"
        )
        
        for player_name in player_names
            color = player_colors[player_name]
            trajectories = Vector{Vector{Float64}}()
            
            for entry in scenario_entries
                executed_costs = compute_executed_trajectory_costs(entry, false)
                
                if !isempty(executed_costs) && haskey(executed_costs[1], player_name)
                    total_trajectory = Float64[]
                    for time_step_costs in executed_costs
                        if haskey(time_step_costs, player_name) && !isempty(time_step_costs[player_name])
                            step = time_step_costs[player_name]
                            comp_sum = sum(values(step))
                            push!(total_trajectory, comp_sum)
                        end
                    end
                    if !isempty(total_trajectory)
                        push!(trajectories, total_trajectory)
                    end
                end
            end
            
            if !isempty(trajectories)
                ts, means, stds = get_component_stats(trajectories)
                valid_idx = findall(.!isnan.(means))
                if !isempty(valid_idx)
                     band!(ax_executed_total, ts[valid_idx], means[valid_idx] .- stds[valid_idx], means[valid_idx] .+ stds[valid_idx], 
                           color=(color, 0.2))
                     lines!(ax_executed_total, ts[valid_idx], means[valid_idx], color=color, linewidth=2)
                end
            end
        end
        
        save("$(directory)/yarnball_$(scenario)_cost_grid.png", fig);
        save("$(directory)/yarnball_$(scenario)_cost_grid.pdf", fig);
        println("Saved yarnball cost grid plot to $(directory)/yarnball_$(scenario)_cost_grid.png")
    end
    return nothing
end

"""
    create_defender_yarnball_comparison(r_entries, nr_entries; output_dir=".")

Create a side-by-side or overlaid yarnball plot comparing Robust vs Non-Robust defender costs.
Breakdown by component. Now supports multiple trials with mean ± std bands.
"""
function create_defender_yarnball_comparison(r_entries, nr_entries; output_dir=".")
    # Handle both single entries and lists
    r_entries_list = r_entries isa Vector ? r_entries : [r_entries]
    nr_entries_list = nr_entries isa Vector ? nr_entries : [nr_entries]
    
    if isempty(r_entries_list) || isempty(nr_entries_list) return end
    
    println("Generating Defender Yarnball Comparison...")
    
    # Helper to compute stats
    function compute_stats(data_list)
        if isempty(data_list) return Float64[], Float64[], Float64[] end
        max_len = maximum(length(d) for d in data_list)
        means = Float64[]
        stds = Float64[]
        for t in 1:max_len
            vals = [d[t] for d in data_list if length(d) >= t]
            if !isempty(vals)
                push!(means, mean(vals))
                push!(stds, length(vals) > 1 ? std(vals) : 0.0)
            end
        end
        return 1:length(means), means, stds
    end
    
    # Collect all components from all entries
    all_comps = Set{Symbol}()
    for entry in vcat(r_entries_list, nr_entries_list)
        planned = calculate_planned_trajectory_costs(entry, false)
        for step_data in planned
            if haskey(step_data, :defender)
                for plan_step in step_data[:defender]
                    union!(all_comps, keys(plan_step))
                end
            end
        end
    end
    component_names = sort(collect(all_comps), by=string)
    
    if isempty(component_names) return end
    
    # Determine layout
    max_t = 0
    for entry in vcat(r_entries_list, nr_entries_list)
        planned = calculate_planned_trajectory_costs(entry, false)
        max_t = max(max_t, length(planned))
    end
    
    num_plan_rows = min(max_t, 10)
    num_rows = num_plan_rows + 1
    num_cols = length(component_names) + 1
    
    fig = Figure(size=(300 * (num_cols-1) + 100, 200 * num_rows))
    Label(fig[0, :], text = "Defender Cost Comparison: Robust (Blue) vs Non-Robust (Red)", fontsize = 20)
    
    colsize!(fig.layout, 1, Relative(0.12))
    
    for row_idx in 1:num_rows
        is_exec_row = (row_idx == num_rows)
        t_plan = row_idx
        
        for col_idx in 1:num_cols
            is_total_col = (col_idx == num_cols)
            
            comp_sym = nothing
            comp_str = ""
            if is_total_col
                comp_str = "Total"
            else
                comp_sym = component_names[col_idx]
                comp_str = string(comp_sym)
            end
            
            ax_title = (row_idx == 1) ? comp_str : ""
            ax_ylabel = ""
            if col_idx == 1
                ax_ylabel = is_exec_row ? "Executed" : "Exec Step $t_plan"
            end
            ax_xlabel = is_exec_row ? "Execution Time" : ((row_idx == num_plan_rows) ? "Horizon Step" : "")
            
            ax = Axis(fig[row_idx, col_idx],
                title=ax_title,
                ylabel=ax_ylabel,
                xlabel=ax_xlabel
            )
            
            if is_exec_row
                # Executed trajectory - aggregate across trials
                r_trajectories = Vector{Vector{Float64}}()
                nr_trajectories = Vector{Vector{Float64}}()
                
                for entry in r_entries_list
                    exec_costs = compute_executed_trajectory_costs(entry, false)
                    vals = Float64[]
                    for step_costs in exec_costs
                        if haskey(step_costs, :defender)
                            if is_total_col
                                push!(vals, sum(values(step_costs[:defender])))
                            elseif haskey(step_costs[:defender], comp_sym)
                                push!(vals, step_costs[:defender][comp_sym])
                            else
                                push!(vals, 0.0)
                            end
                        end
                    end
                    if !isempty(vals)
                        push!(r_trajectories, vals)
                    end
                end
                
                for entry in nr_entries_list
                    exec_costs = compute_executed_trajectory_costs(entry, false)
                    vals = Float64[]
                    for step_costs in exec_costs
                        if haskey(step_costs, :defender)
                            if is_total_col
                                push!(vals, sum(values(step_costs[:defender])))
                            elseif haskey(step_costs[:defender], comp_sym)
                                push!(vals, step_costs[:defender][comp_sym])
                            else
                                push!(vals, 0.0)
                            end
                        end
                    end
                    if !isempty(vals)
                        push!(nr_trajectories, vals)
                    end
                end
                
                # Plot with bands
                if !isempty(r_trajectories)
                    ts, means, stds = compute_stats(r_trajectories)
                    band!(ax, collect(ts), means .- stds, means .+ stds, color=(:blue, 0.2))
                    lines!(ax, collect(ts), means, color=:blue, linewidth=2)
                end
                
                if !isempty(nr_trajectories)
                    ts, means, stds = compute_stats(nr_trajectories)
                    band!(ax, collect(ts), means .- stds, means .+ stds, color=(:red, 0.2))
                    lines!(ax, collect(ts), means, color=:red, linewidth=2)
                end
                
            else
                # Planned trajectory at step t_plan - aggregate across trials
                r_trajectories = Vector{Vector{Float64}}()
                nr_trajectories = Vector{Vector{Float64}}()
                
                for entry in r_entries_list
                    planned = calculate_planned_trajectory_costs(entry, false)
                    if t_plan <= length(planned)
                        step_data = planned[t_plan]
                        if haskey(step_data, :defender)
                            plan = step_data[:defender]
                            vals = Float64[]
                            for p_step in plan
                                if is_total_col
                                    push!(vals, sum(values(p_step)))
                                else
                                    push!(vals, get(p_step, comp_sym, 0.0))
                                end
                            end
                            if !isempty(vals)
                                push!(r_trajectories, vals)
                            end
                        end
                    end
                end
                
                for entry in nr_entries_list
                    planned = calculate_planned_trajectory_costs(entry, false)
                    if t_plan <= length(planned)
                        step_data = planned[t_plan]
                        if haskey(step_data, :defender)
                            plan = step_data[:defender]
                            vals = Float64[]
                            for p_step in plan
                                if is_total_col
                                    push!(vals, sum(values(p_step)))
                                else
                                    push!(vals, get(p_step, comp_sym, 0.0))
                                end
                            end
                            if !isempty(vals)
                                push!(nr_trajectories, vals)
                            end
                        end
                    end
                end
                
                # Plot with bands
                if !isempty(r_trajectories)
                    ts, means, stds = compute_stats(r_trajectories)
                    band!(ax, collect(ts), means .- stds, means .+ stds, color=(:blue, 0.2))
                    lines!(ax, collect(ts), means, color=:blue, linewidth=2)
                end
                
                if !isempty(nr_trajectories)
                    ts, means, stds = compute_stats(nr_trajectories)
                    band!(ax, collect(ts), means .- stds, means .+ stds, color=(:red, 0.2))
                    lines!(ax, collect(ts), means, color=:red, linewidth=2)
                end
            end
        end
    end
    
    rank = r_entries_list[1].trial_number
    file_path = joinpath(output_dir, "rank_$(rank)_defender_yarnball.png")
    save(file_path, fig)
    save("$(directory)/rank_$(rank)_defender_yarnball.pdf", fig)
    println("Saved Defender Yarnball to $file_path")
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
        save(filename, fig);
        println("Saved $filename with $trajectory_count trajectories")
        
        trial_counter = 1  # Reset for next scenario
    end
    
    println("\n=== All scenario plots completed ===")
    return nothing
end

"""
    clear_hockey_kkt_tracker!()

Clear all KKT tracking data for hockey experiments.
"""
function clear_hockey_kkt_tracker!()
    clear_rh_kkt_tracker!()
end


end
