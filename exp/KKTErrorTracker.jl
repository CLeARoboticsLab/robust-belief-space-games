module KKTErrorTracker
using JLD2
using FileIO
using Statistics
using LinearAlgebra
using Printf
using CairoMakie

export RHKKTEntry, RHKKTTracker, RH_KKT_TRACKER, record_rh_kkt_error!, 
       save_rh_kkt_tracker, load_rh_kkt_tracker, get_rh_kkt_summary, 
       create_kkt_yarnball_plots, clear_rh_kkt_tracker!, 
       export_rh_kkt_csv, save_hockey_kkt_data, analyze_hockey_kkt_errors

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
    player::Union{Int, Nothing}
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
    
    # player_str = isnothing(player) ? "" : " Player $player"
    # robust_str = robust ? " (Robust)" : " (Non-Robust)"
    # trajectory_info = length(kkt_errors_trajectory) > 1 ? " ($(length(kkt_errors_trajectory)) steps)" : ""
    # println("[RHKKTTracker] Trial $trial, Step $time_step$player_str$robust_str: $solver_type KKT mean = $(round(kkt_error_mean, digits=8))$trajectory_info")
    
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
                
                # Handle old format
                try
                    # new format
                    append!(RH_KKT_TRACKER.entries, loaded_tracker.entries)
                catch e1
                    # old format
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
                            # even more old format
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
    analyze_hockey_kkt_errors()

Analyze KKT errors from hockey receding horizon experiments.
"""
function analyze_hockey_kkt_errors()
    println("=== Hockey Receding Horizon KKT Error Analysis ===")
    load_rh_kkt_tracker("rh_multi-trial-kkt-error-data.jld2")    
    get_rh_kkt_summary()
    plot_spatial_trajectories()
    analyze_hockey_trajectory_data()
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
    save("kkt_temporal_evolution.png", fig);
    println("Temporal KKT evolution plots saved to kkt_temporal_evolution.png")
    return nothing
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
        println(f, "solver_type,kkt_error_mean,trial,time_step,player,robust,iteration_count,convergence_status,trajectory_length,kkt_errors_trajectory")
        for entry in RH_KKT_TRACKER.entries
            player_str = isnothing(entry.player) ? "" : string(entry.player)
            trajectory_str = join(entry.kkt_errors_trajectory, ";")
            println(f, "$(entry.solver_type),$(entry.kkt_error_mean),$(entry.trial),$(entry.time_step),$(player_str),$(entry.robust),$(entry.iteration_count),$(entry.convergence_status),$(length(entry.kkt_errors_trajectory)),$(trajectory_str)")
        end
    end
    
    println("Receding horizon KKT errors exported to $filename")
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
end
