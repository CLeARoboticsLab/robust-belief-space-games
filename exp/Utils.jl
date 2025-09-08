module Utils
using JLD2
using FileIO
using Statistics
using LinearAlgebra
using BlockArrays
using Infiltrator
using RobustBeliefGame
using Printf
using CairoMakie

# ========================================================================================
# KKT ERROR TRACKING SYSTEM
# ========================================================================================

"""
KKT Error Tracking System for Receding Horizon

This module provides functionality to track KKT errors specifically from receding horizon solver calls.
"""

mutable struct RHKKTEntry
    solver_type::String  # "MCP" or "BeliefGame" 
    kkt_error_mean::Float64  # Mean of all KKT errors in trajectory
    kkt_errors_trajectory::Vector{Float64}  # Individual KKT errors for each time step
    trial::Int
    time_step::Int
    player::Union{Int, Nothing}  # For BeliefGame solver calls
    robust::Bool
    iteration_count::Int
    convergence_status::Symbol
    additional_data::Dict{String, Any}
end

"""
Receding Horizon KKT error tracker
"""
mutable struct RHKKTTracker
    entries::Vector{RHKKTEntry}
    auto_save::Bool
    save_file::String
    
    function RHKKTTracker(auto_save::Bool=true, save_file::String="rh_kkt_errors.jld2")
        new(RHKKTEntry[], auto_save, save_file)
    end
end

# Global instance for receding horizon tracking
const RH_KKT_TRACKER = RHKKTTracker()

"""
    record_rh_kkt_error!(solver_type, kkt_errors_trajectory, trial, time_step; kwargs...)

Record KKT errors from a receding horizon solver call.
Can accept either a single Float64 (for backward compatibility) or Vector{Float64} for trajectory.
"""
function record_rh_kkt_error!(
    solver_type::String, 
    kkt_errors::Union{Float64, Vector{Float64}},
    trial::Int,
    time_step::Int;
    player::Union{Int, Nothing}=nothing,
    robust::Bool=false,
    iteration_count::Int=0,
    convergence_status::Symbol=:unknown,
    additional_data::Dict{String, Any}=Dict{String, Any}()
)
    # Handle both single value and trajectory input
    if kkt_errors isa Float64
        kkt_errors_trajectory = [kkt_errors]
        kkt_error_mean = kkt_errors
    else
        kkt_errors_trajectory = kkt_errors
        kkt_error_mean = mean(kkt_errors_trajectory)
    end
    
    entry = RHKKTEntry(
        solver_type,
        kkt_error_mean,
        kkt_errors_trajectory,
        trial,
        time_step,
        player,
        robust,
        iteration_count,
        convergence_status,
        additional_data
    )
    
    push!(RH_KKT_TRACKER.entries, entry)
    
    if RH_KKT_TRACKER.auto_save
        save_rh_kkt_tracker()
    end
    
    player_str = isnothing(player) ? "" : " Player $player"
    robust_str = robust ? " (Robust)" : " (Non-Robust)"
    trajectory_info = length(kkt_errors_trajectory) > 1 ? " ($(length(kkt_errors_trajectory)) steps)" : ""
    println("[RHKKTTracker] Trial $trial, Step $time_step$player_str$robust_str: $solver_type KKT mean = $(round(kkt_error_mean, digits=8))$trajectory_info")
    
    return entry
end

"""
    save_rh_kkt_tracker(filename=RH_KKT_TRACKER.save_file)

Save the RH KKT tracker to a file.
"""
function save_rh_kkt_tracker(filename::String=RH_KKT_TRACKER.save_file)
    @save filename RH_KKT_TRACKER
end

"""
    load_rh_kkt_tracker(filename="rh_multi-trial-kkt-error-data.jld2")

Load an RH KKT tracker from a file and update the global tracker.
Handles migration from old RobustBeliefGame module types to new Utils module types.
"""
function load_rh_kkt_tracker(filename::String="rh_multi-trial-kkt-error-data.jld2")
    if isfile(filename)
        try
            # Load into a temporary variable to avoid conflicts with the global const
            loaded_data = load(filename)
            
            if haskey(loaded_data, "RH_KKT_TRACKER")
                loaded_tracker = loaded_data["RH_KKT_TRACKER"]
                
                # Clear the global tracker
                empty!(RH_KKT_TRACKER.entries)
                
                # Handle both old (RobustBeliefGame) and new (Utils) type formats
                try
                    # Try direct assignment first (for new format)
                    append!(RH_KKT_TRACKER.entries, loaded_tracker.entries)
                catch e1
                    # If that fails, try manual conversion (for old format)
                    println("Converting from old format...")
                    for (i, entry) in enumerate(loaded_tracker.entries)
                        try
                            # Handle JLD2 reconstructed types by accessing fields via property access
                            new_entry = RHKKTEntry(
                                entry.solver_type,
                                entry.kkt_error_mean,
                                Vector{Float64}(entry.kkt_errors_trajectory),
                                entry.trial,
                                entry.time_step,
                                entry.player,
                                entry.robust,
                                entry.iteration_count,
                                entry.convergence_status,
                                Dict{String, Any}(entry.additional_data)
                            )
                            push!(RH_KKT_TRACKER.entries, new_entry)
                        catch e2
                            # Try even more manual extraction if property access fails
                            try
                                # Extract using reflection for JLD2 reconstructed objects
                                fields = fieldnames(typeof(entry))
                                values = [getfield(entry, f) for f in fields]
                                
                                new_entry = RHKKTEntry(
                                    values[1],  # solver_type
                                    values[2],  # kkt_error_mean
                                    Vector{Float64}(values[3]),  # kkt_errors_trajectory
                                    values[4],  # trial
                                    values[5],  # time_step
                                    values[6],  # player
                                    values[7],  # robust
                                    values[8],  # iteration_count
                                    values[9],  # convergence_status
                                    Dict{String, Any}(values[10])  # additional_data
                                )
                                push!(RH_KKT_TRACKER.entries, new_entry)
                            catch e3
                                println("Warning: Could not convert entry $i: $e3")
                                continue
                            end
                        end
                    end
                end
                
                RH_KKT_TRACKER.auto_save = loaded_tracker.auto_save
                RH_KKT_TRACKER.save_file = loaded_tracker.save_file
                
                println("Loaded $(length(RH_KKT_TRACKER.entries)) KKT entries from $filename")
                return RH_KKT_TRACKER
            else
                println("Warning: File $filename does not contain RH_KKT_TRACKER data")
                return RH_KKT_TRACKER
            end
        catch e
            println("Error loading KKT data from $filename: $e")
            println("Consider deleting the old file and re-running experiments to generate new data")
            return RH_KKT_TRACKER
        end
    else
        println("File $filename not found, returning empty tracker")
        return RHKKTTracker()
    end
end

"""
    get_rh_kkt_summary()

Get a comprehensive summary of receding horizon KKT errors.
"""
function get_rh_kkt_summary()
    if isempty(RH_KKT_TRACKER.entries)
        println("No receding horizon KKT errors recorded yet.")
        return nothing
    end
    
    println("\n=== Receding Horizon KKT Error Summary ===")
    println("Total entries: $(length(RH_KKT_TRACKER.entries))")
    
    # Group by solver type
    mcp_entries = [e for e in RH_KKT_TRACKER.entries if e.solver_type == "MCP"]
    belief_entries = [e for e in RH_KKT_TRACKER.entries if e.solver_type == "BeliefGame"]
    
    if !isempty(mcp_entries)
        mcp_errors = [e.kkt_error_mean for e in mcp_entries]
        println("\nMCP Solver ($(length(mcp_entries)) calls):")
        println("  Min: $(minimum(mcp_errors))")
        println("  Max: $(maximum(mcp_errors))")
        println("  Mean: $(round(mean(mcp_errors), digits=8))")
        println("  Std: $(round(std(mcp_errors), digits=8))")
    end
    
    if !isempty(belief_entries)
        belief_errors = [e.kkt_error_mean for e in belief_entries]
        println("\nBeliefGame Solver ($(length(belief_entries)) calls):")
        println("  Min: $(minimum(belief_errors))")
        println("  Max: $(maximum(belief_errors))")
        println("  Mean: $(round(mean(belief_errors), digits=8))")
        println("  Std: $(round(std(belief_errors), digits=8))")
        
        # Analyze trajectory lengths
        trajectory_lengths = [length(e.kkt_errors_trajectory) for e in belief_entries]
        total_kkt_measurements = sum(trajectory_lengths)
        println("  Total KKT measurements across all trajectories: $total_kkt_measurements")
        
        # Break down by robust/non-robust
        robust_entries = [e for e in belief_entries if e.robust]
        non_robust_entries = [e for e in belief_entries if !e.robust]
        
        if !isempty(robust_entries)
            robust_errors = [e.kkt_error_mean for e in robust_entries]
            println("  Robust ($(length(robust_entries)) calls): mean = $(round(mean(robust_errors), digits=8))")
        end
        
        if !isempty(non_robust_entries)
            non_robust_errors = [e.kkt_error_mean for e in non_robust_entries]
            println("  Non-Robust ($(length(non_robust_entries)) calls): mean = $(round(mean(non_robust_errors), digits=8))")
        end
    end
    
    # Analyze by scenario - statistics across trials for each scenario type
    scenarios = unique([e.additional_data["scenario_name"] for e in RH_KKT_TRACKER.entries if haskey(e.additional_data, "scenario_name")])
    if !isempty(scenarios)
        # Group scenarios by base type (remove trial numbers)
        base_scenarios = unique([
            join(split(scenario, "_")[1:end-1], "_") 
            for scenario in scenarios
        ])
        
        println("\nScenario Statistics Across Trials:")
        
        for base_scenario in sort(base_scenarios)
            # Find all trials for this base scenario
            trial_scenarios = [s for s in scenarios if startswith(s, base_scenario)]
            
            if !isempty(trial_scenarios)
                println("\n  $base_scenario:")
                
                # Calculate mean KKT error for each trial
                trial_means = Float64[]
                
                for trial_scenario in trial_scenarios
                    trial_entries = [e for e in RH_KKT_TRACKER.entries if haskey(e.additional_data, "scenario_name") && e.additional_data["scenario_name"] == trial_scenario]
                    if !isempty(trial_entries)
                        trial_errors = [e.kkt_error_mean for e in trial_entries]
                        trial_mean = mean(trial_errors)
                        push!(trial_means, trial_mean)
                    end
                end
                
                # Calculate statistics across trials
                if !isempty(trial_means)
                    println("    Trials: $(length(trial_means))")
                    println("    Mean: $(round(mean(trial_means), digits=8))")
                    println("    Median: $(round(median(trial_means), digits=8))")
                    println("    Std: $(round(std(trial_means), digits=8))")
                    println("    Min: $(round(minimum(trial_means), digits=8))")
                    println("    Max: $(round(maximum(trial_means), digits=8))")
                end
            end
        end
    end
    
    # Create yarnball plots
    create_kkt_yarnball_plots()
    
    return RH_KKT_TRACKER.entries
end

"""
    create_kkt_yarnball_plots()

Create temporal KKT error plots showing evolution across receding horizon execution time.
"""
function create_kkt_yarnball_plots()
    if isempty(RH_KKT_TRACKER.entries)
        println("No data to plot.")
        return
    end
    
    # Get BeliefGame entries only (they have trajectories)
    belief_entries = [e for e in RH_KKT_TRACKER.entries if e.solver_type == "BeliefGame"]
    if isempty(belief_entries)
        println("No BeliefGame entries to plot.")
        return
    end
    
    # Group by base scenario type (remove trial numbers)
    base_scenarios = unique([
        join(split(e.additional_data["scenario_name"], "_")[1:end-1], "_") 
        for e in belief_entries if haskey(e.additional_data, "scenario_name")
    ])
    
    if isempty(base_scenarios)
        println("No scenario information available for plotting.")
        return
    end
    
    println("\nCreating KKT temporal evolution plots...")
    
    # Create overall figure with subplots for each base scenario type
    n_scenarios = length(base_scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = ceil(Int, n_scenarios / n_cols)
    
    fig = Figure(size = (400 * n_cols, 300 * n_rows))
    
    # Define colors for different conditions
    robust_color = :blue
    non_robust_color = :red
    alpha_val = 0.7
    
    for (i, base_scenario) in enumerate(sort(base_scenarios))
        row = div(i - 1, n_cols) + 1
        col = mod(i - 1, n_cols) + 1
        
        ax = Axis(fig[row, col], 
            title = base_scenario,
            xlabel = "Execution Time Step",
            ylabel = "Mean KKT Error",
            yscale = log10
        )
        
        # Get all entries that match this base scenario type
        base_scenario_entries = [
            e for e in belief_entries 
            if haskey(e.additional_data, "scenario_name") && 
               startswith(e.additional_data["scenario_name"], base_scenario)
        ]
        
        # Group by trial, player, robust to create separate lines
        trial_player_robust_groups = unique([(e.trial, e.player, e.robust) for e in base_scenario_entries])
        
        for (trial, player, is_robust) in trial_player_robust_groups
            # Get all entries for this specific trial/player/robust combination
            group_entries = [e for e in base_scenario_entries if e.trial == trial && e.player == player && e.robust == is_robust]
            
            if !isempty(group_entries)
                # Sort by time step to ensure correct temporal order
                sort!(group_entries, by = e -> e.time_step)
                
                # For each time step, take the mean KKT error across the planned trajectory
                x_vals = [e.time_step for e in group_entries]
                y_vals = [e.kkt_error_mean for e in group_entries]
                
                color = is_robust ? robust_color : non_robust_color
                
                # Plot temporal evolution (no individual labels to avoid clutter)
                lines!(ax, x_vals, y_vals, 
                    color = (color, alpha_val),
                    linewidth = 1.5
                )
                
                # Add points at each execution time step
                scatter!(ax, x_vals, y_vals,
                    color = (color, alpha_val),
                    markersize = 4
                )
            end
        end
        
        # Add legend only to first subplot with generic labels
        if i == 1 && !isempty(base_scenario_entries)
            robust_exists = any(e.robust for e in base_scenario_entries)
            non_robust_exists = any(!e.robust for e in base_scenario_entries)
            
            legend_elements = []
            legend_labels = []
            
            if robust_exists
                push!(legend_elements, LineElement(color = robust_color, linewidth = 2))
                push!(legend_labels, "Robust")
            end
            if non_robust_exists
                push!(legend_elements, LineElement(color = non_robust_color, linewidth = 2))
                push!(legend_labels, "Non-Robust")
            end
            
            if !isempty(legend_elements)
                Legend(fig[1, n_cols + 1], legend_elements, legend_labels, "Method")
            end
        end
    end
    
    # Save the plot
    save("kkt_temporal_evolution.png", fig)
    println("Temporal KKT evolution plots saved to kkt_temporal_evolution.png")
end


"""
    clear_rh_kkt_tracker!()

Clear all recorded RH KKT errors.
"""
function clear_rh_kkt_tracker!()
    empty!(RH_KKT_TRACKER.entries)
    println("Receding horizon KKT tracker cleared.")
end

"""
    export_rh_kkt_csv(filename="rh_kkt_errors.csv")

Export RH KKT errors to CSV format.
"""
function export_rh_kkt_csv(filename::String="rh_kkt_errors.csv")
    if isempty(RH_KKT_TRACKER.entries)
        println("No KKT errors to export.")
        return
    end
    
    open(filename, "w") do f
        # Header
        println(f, "solver_type,kkt_error_mean,trial,time_step,player,robust,iteration_count,convergence_status,trajectory_length,kkt_errors_trajectory")
        
        # Data
        for entry in RH_KKT_TRACKER.entries
            player_str = isnothing(entry.player) ? "" : string(entry.player)
            trajectory_str = join(entry.kkt_errors_trajectory, ";")
            println(f, "$(entry.solver_type),$(entry.kkt_error_mean),$(entry.trial),$(entry.time_step),$(player_str),$(entry.robust),$(entry.iteration_count),$(entry.convergence_status),$(length(entry.kkt_errors_trajectory)),$(trajectory_str)")
        end
    end
    
    println("Receding horizon KKT errors exported to $filename")
end


"""
    get_trajectory_details(trial::Int, time_step::Int, player::Union{Int, Nothing}=nothing)

Get detailed trajectory information for a specific solver call.
"""
function get_trajectory_details(trial::Int, time_step::Int, player::Union{Int, Nothing}=nothing)
    matching_entries = [e for e in RH_KKT_TRACKER.entries if 
                       e.trial == trial && e.time_step == time_step && 
                       (isnothing(player) || e.player == player)]
    
    if isempty(matching_entries)
        println("No entries found for trial $trial, step $time_step" * 
                (isnothing(player) ? "" : ", player $player"))
        return nothing
    end
    
    for entry in matching_entries
        player_str = isnothing(entry.player) ? "" : " Player $(entry.player)"
        robust_str = entry.robust ? " (Robust)" : " (Non-Robust)"
        
        println("\n=== Trajectory Details ===")
        println("Trial $(entry.trial), Step $(entry.time_step)$player_str$robust_str")
        println("Solver: $(entry.solver_type)")
        println("Mean KKT Error: $(round(entry.kkt_error_mean, digits=8))")
        println("Trajectory Length: $(length(entry.kkt_errors_trajectory))")
        println("Individual KKT Errors:")
        
        for (i, kkt_val) in enumerate(entry.kkt_errors_trajectory)
            println("  Time $i: $(round(kkt_val, digits=8))")
        end
        
        if length(entry.kkt_errors_trajectory) > 1
            println("Variance: $(round(var(entry.kkt_errors_trajectory), digits=10))")
            
            # Calculate trend
            x = collect(1:length(entry.kkt_errors_trajectory))
            y = entry.kkt_errors_trajectory
            n = length(x)
            slope = (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x .^ 2) - sum(x)^2)
            println("Trend (slope): $(round(slope, digits=10))")
        end
    end
    
    return matching_entries
end

# ========================================================================================
# TRAJECTORY ANALYSIS TRACKER SYSTEM
# ========================================================================================

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
function load_and_analyze_solution_files()
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
    solution_files = [joinpath(output_dir, f) for f in all_files if startswith(f, "rh_multi-trial") && endswith(f, ".jld2")]
    
    if isempty(solution_files)
        println("No solution files found in $output_dir")
        return false
    end
    
    println("Found $(length(solution_files)) solution files")
    
    for solution_file in solution_files
        try
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
                    
                # Extract trajectory data
                gt_state_history, all_observations, solution_history, cond_history, lq_sol_history = solution_data
                
                if !isempty(gt_state_history)
                    # Create trajectory analysis entry with raw data
                    entry = TrajectoryAnalysisEntry(
                        key,  # Full scenario name including trial
                        trial_num,
                        gt_state_history,
                        all_observations,
                        solution_history,
                        cond_history,
                        lq_sol_history,
                        is_robust,
                        noise_level
                    )
                    
                    push!(TRAJECTORY_TRACKER.entries, entry)
                end
            end
            
        catch e
            println("Error loading $solution_file: $e")
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

function effect_of_nature(rh_file_ids; save_to_file=false)
    if isempty(rh_file_ids)
        output_dir = "exp/hockey/outputs"
        rh_file_ids = [replace(f, ".jld2" => "") for f in readdir(output_dir) if startswith(f, "rh_")]
    end

    for file_id in rh_file_ids
        filepath = "exp/hockey/outputs/$file_id.jld2"
        if !isfile(filepath)
            println("File not found: $filepath")
            continue
        end

        @load filepath gt_state_history all_observations goal_position solution_history
        
        analysis_text = "============================================================\n"
        analysis_text *= @sprintf("ANALYSIS FOR: %s.jld2\n", file_id)
        analysis_text *= "============================================================\n\n"

        for t in 1:length(solution_history)
            non_robust_traj = solution_history[t][1][1]
            robust_traj = solution_history[t][2][1]
            nature_us_history = [solution_history[t][2][2][k][Block(3)] for k in 1:length(solution_history[t][2][2])]
            non_robust_planned_us_history = [solution_history[t][1][2][k][Block(1):Block(2)] for k in 1:length(solution_history[t][1][2])]
            robust_planned_us_history = [solution_history[t][2][2][k][Block(1):Block(2)] for k in 1:length(solution_history[t][2][2])]
            
            # 1. Absolute L2 norm difference between trajectories
            traj_diff = sum(norm(means(robust_traj[k])[Block(1):Block(2)] - means(robust_traj[k])[Block(3):Block(4)]) for k in eachindex(robust_traj))
            
            # 2. Average L2 norm difference per unit control cost of nature
            nature_control_cost = sum(
                2_000*dot(nature_us_history[k], nature_us_history[k]) 
                for k in eachindex(nature_us_history)
                )
            diff_per_cost = nature_control_cost > 1e-9 ? traj_diff / nature_control_cost : traj_diff

            # 3. Control costs for players
            attacker_control_cost = sum(
                4 * dot(robust_planned_us_history[k][1:2], robust_planned_us_history[k][1:2])
                for k in eachindex(robust_planned_us_history)
            )
            defender_control_cost = sum(
                2 * dot(robust_planned_us_history[k][3:4], robust_planned_us_history[k][3:4])
                for k in eachindex(robust_planned_us_history)
            )
            attacker_diff_per_cost = attacker_control_cost > 1e-9 ? traj_diff / attacker_control_cost : traj_diff
            defender_diff_per_cost = defender_control_cost > 1e-9 ? traj_diff / defender_control_cost : traj_diff

            # 4. Difference in robust and non-robust actions
            attacker_action_diff = (
                sum(
                    norm(non_robust_planned_us_history[t][1] - robust_planned_us_history[t][1])
                    for k in eachindex(non_robust_planned_us_history[t])
                ),
                sum(
                    norm(non_robust_planned_us_history[t][2] - robust_planned_us_history[t][2])
                    for k in eachindex(non_robust_planned_us_history[t])
                )
            )
            defender_action_diff = (
                sum(
                    norm(non_robust_planned_us_history[t][3] - robust_planned_us_history[t][3])
                    for k in eachindex(non_robust_planned_us_history[t])
                ),
                sum(
                    norm(non_robust_planned_us_history[t][4] - robust_planned_us_history[t][4])
                    for k in eachindex(non_robust_planned_us_history[t])
                )
            )

            analysis_text *= @sprintf("----------------- Timestep t=%d -----------------\n", t)
            analysis_text *= "Trajectory Analysis:\n"
            analysis_text *= @sprintf("  - Trajectory Difference    : %.5f\n", traj_diff)
            analysis_text *= @sprintf("  - Nature's Control Cost    : %.5f\n", nature_control_cost)
            analysis_text *= @sprintf("  - Attacker's Control Cost  : %.5f\n", attacker_control_cost)
            analysis_text *= @sprintf("  - Defender's Control Cost  : %.5f\n\n", defender_control_cost)
            analysis_text *= @sprintf("  - Nature's Diff per Cost   : %.5f\n", diff_per_cost)
            analysis_text *= @sprintf("  - Attacker's Diff per Cost : %.5f\n", attacker_diff_per_cost)
            analysis_text *= @sprintf("  - Defender's Diff per Cost : %.5f\n", defender_diff_per_cost)
            analysis_text *= "\n"
            analysis_text *= "Action Difference (Non-Robust vs. Robust):\n"
            analysis_text *= @sprintf("  - Attacker (accel, steer)  : %.5f, %.5f\n", attacker_action_diff[1], attacker_action_diff[2])
            analysis_text *= @sprintf("  - Defender (accel, steer)  : %.5f, %.5f\n", defender_action_diff[1], defender_action_diff[2])
            analysis_text *= "\n"

        end

        println(analysis_text)

        if save_to_file
            open("exp/hockey/outputs/analysis_$file_id.txt", "w") do f
                write(f, analysis_text)
            end
            println("Analysis saved to exp/hockey/outputs/analysis_$file_id.txt")
        end
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
    
    # Group by base scenario: "<noise_level>_<robustness>" where robustness is "robust" or "non_robust"
    base_group = function(name::String)
        parts = split(name, "_")
        if length(parts) >= 3 && parts[2] == "non" && parts[3] == "robust"
            return string(parts[1], "_non_robust")
        elseif length(parts) >= 2
            return string(parts[1], "_", parts[2])
        else
            return name
        end
    end
    
    base_groups = unique([base_group(e.scenario_name) for e in TRAJECTORY_TRACKER.entries])
    for base in sort(base_groups)
        group_entries = [e for e in TRAJECTORY_TRACKER.entries if base_group(e.scenario_name) == base]
        println("\nScenario Group: $base ($(length(group_entries)) trials)")
        
        # Belief covariance stats (aggregated across trials)
        all_cov_traces = vcat([compute_belief_covariance_traces(e) for e in group_entries]...)
        if !isempty(all_cov_traces)
            println("  Belief Covariance Traces:")
            println("    Mean:   $(round(mean(all_cov_traces), digits=4))")
            println("    Std:    $(round(std(all_cov_traces), digits=4))")
            println("    Range: [$(round(minimum(all_cov_traces), digits=4)), $(round(maximum(all_cov_traces), digits=4))]")
        else
            println("  Belief Covariance Traces: Not available")
        end
        
        # Player distance stats (aggregated across trials)
        all_player_distances = vcat([compute_player_distances(e) for e in group_entries]...)
        if !isempty(all_player_distances)
            avg_distances = [mean(compute_player_distances(e)) for e in group_entries if !isempty(compute_player_distances(e))]
            println("  Player Distances:")
            println("    Mean:   $(round(mean(avg_distances), digits=4))")
            println("    Std:    $(round(std(avg_distances), digits=4))")
            println("    Range: [$(round(minimum(all_player_distances), digits=4)), $(round(maximum(all_player_distances), digits=4))]")
        else
            println("  Player Distances: Not available")
        end
        
        # Belief deviation stats (aggregated across trials)
        all_belief_deviations = vcat([compute_belief_deviations(e) for e in group_entries]...)
        if !isempty(all_belief_deviations)
            total_deviations = [sum(compute_belief_deviations(e)) for e in group_entries]
            println("  Belief Deviations:")
            println("    Mean:   $(round(mean(total_deviations), digits=4))")
            println("    Std:    $(round(std(total_deviations), digits=4))")
            println("    Range: [$(round(minimum(all_belief_deviations), digits=4)), $(round(maximum(all_belief_deviations), digits=4))]")
        else
            println("  Belief Deviations: Not available")
        end
    end
    
    # Create analysis plots
    create_trajectory_analysis_plots()
    
    # return TRAJECTORY_TRACKER.entries
end

"""
    create_trajectory_analysis_plots()

Create comprehensive plots for trajectory analysis.
"""
function create_trajectory_analysis_plots()
    if isempty(TRAJECTORY_TRACKER.entries)
        println("No data to plot")
        return
    end
    
    # Group entries by scenario for plotting
    scenarios = unique([entry.scenario_name for entry in TRAJECTORY_TRACKER.entries])
    
    # Plot 1: Belief Covariance Evolution
    create_belief_covariance_plots(scenarios)
    
    # Plot 2: Player Distance Evolution  
    create_player_distance_plots(scenarios)
    
    # Plot 3: Belief Deviation Evolution
    create_belief_deviation_plots(scenarios)
    
    println("Trajectory analysis plots created successfully")
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
    create_belief_deviation_plots(scenarios)

Create plots showing belief deviation from ground truth over time.
"""
function create_belief_deviation_plots(scenarios)
    fig = Figure(size = (1200, 800))
    
    # Create subplots for each scenario
    n_scenarios = length(scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = ceil(Int, n_scenarios / n_cols)
    
    for (i, scenario) in enumerate(scenarios)
        row = ceil(Int, i / n_cols)
        col = mod(i - 1, n_cols) + 1
        
        ax = Axis(fig[row, col],
            title = "Belief Deviation - $(replace(scenario, "_" => " "))",
            xlabel = "Time Step", 
            ylabel = "Deviation from Ground Truth"
        )
        
        scenario_entries = [e for e in TRAJECTORY_TRACKER.entries if e.scenario_name == scenario]
        
        # Plot each trial's deviation evolution
        colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
        
        for (j, entry) in enumerate(scenario_entries)
            color = colors[mod(j-1, length(colors)) + 1]
            belief_deviations = compute_belief_deviations(entry)
            time_steps = 1:length(belief_deviations)
            
            lines!(ax, time_steps, belief_deviations,
                color = color,
                alpha = 0.7,
                linewidth = 2
            )
        end
        
        # Add mean line
        if !isempty(scenario_entries)
            all_deviations = [compute_belief_deviations(e) for e in scenario_entries]
            max_length = maximum(length(devs) for devs in all_deviations)
            mean_deviations = Float64[]
            
            for t in 1:max_length
                values_at_t = [devs[t] for devs in all_deviations if length(devs) >= t]
                if !isempty(values_at_t)
                    push!(mean_deviations, mean(values_at_t))
                end
            end
            
            lines!(ax, 1:length(mean_deviations), mean_deviations,
                color = :black,
                linewidth = 3,
                linestyle = :dash
            )
        end
    end
    
    save("trajectory_belief_deviations.png", fig)
    println("Belief deviation plot saved as trajectory_belief_deviations.png")
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
    analyze_hockey_kkt_errors()

Analyze KKT errors from hockey receding horizon experiments.
"""
function analyze_hockey_kkt_errors()
    println("=== Hockey Receding Horizon KKT Error Analysis ===")
    load_rh_kkt_tracker("rh_multi-trial-kkt-error-data.jld2")
    
    # Debug: Check if data was actually loaded
    println("Debug: Global tracker has $(length(RH_KKT_TRACKER.entries)) entries")
    
    get_rh_kkt_summary()
        
    # Also plot spatial trajectories
    plot_spatial_trajectories()
    
    # Analyze trajectory data
    analyze_hockey_trajectory_data()
end

"""
    analyze_hockey_trajectory_data()

Analyze trajectory data from hockey receding horizon experiments.
"""
function analyze_hockey_trajectory_data()
    println("\n=== ANALYZING HOCKEY TRAJECTORY DATA ===")
    
    # Load trajectory data from solution files
    if load_and_analyze_solution_files()
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

"""
    save_hockey_kkt_data(base_filename="hockey_rh_kkt_errors")

Save hockey KKT tracking data to files, organized by scenario.
"""
function save_hockey_kkt_data(base_filename::String="hockey_rh_kkt_errors")
    # Save all data in one comprehensive file
    save_rh_kkt_tracker("$base_filename.jld2")
    export_rh_kkt_csv("$base_filename.csv")
    
    # Also save separate files for each scenario
    if !isempty(RH_KKT_TRACKER.entries)
        scenarios = unique([entry.additional_data["scenario_name"] for entry in RH_KKT_TRACKER.entries if haskey(entry.additional_data, "scenario_name")])
        
        for scenario in scenarios
            scenario_entries = [e for e in RH_KKT_TRACKER.entries if haskey(e.additional_data, "scenario_name") && e.additional_data["scenario_name"] == scenario]
            
            if !isempty(scenario_entries)
                # Create a temporary tracker with just this scenario's data
                temp_tracker = RHKKTTracker(false, "")
                temp_tracker.entries = scenario_entries
                
                # Save scenario-specific files
                @save "kkt_$scenario.jld2" temp_tracker
                
                # Export scenario-specific CSV
                open("kkt_$scenario.csv", "w") do f
                    println(f, "solver_type,kkt_error_mean,trial,time_step,player,robust,iteration_count,convergence_status,trajectory_length,kkt_errors_trajectory")
                    
                    for entry in scenario_entries
                        player_str = isnothing(entry.player) ? "" : string(entry.player)
                        trajectory_str = join(entry.kkt_errors_trajectory, ";")
                        println(f, "$(entry.solver_type),$(entry.kkt_error_mean),$(entry.trial),$(entry.time_step),$(player_str),$(entry.robust),$(entry.iteration_count),$(entry.convergence_status),$(length(entry.kkt_errors_trajectory)),$(trajectory_str)")
                    end
                end
                
                println("Scenario $scenario: saved to kkt_$scenario.jld2 and kkt_$scenario.csv")
            end
        end
    end
    
    println("All hockey KKT data saved to $base_filename.jld2 and $base_filename.csv")
end

end