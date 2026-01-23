using Distributed
using Test
using FileIO
using Infiltrator
using JLD2
using Statistics
using Logging
using GLMakie
using LinearAlgebra

# Add procs if needed
if nprocs() < 2
    addprocs(1)
end

# Resolve paths relative to this script
const SCRIPT_DIR = @__DIR__
const PARAM_SWEEP_PATH = joinpath(SCRIPT_DIR, "param_sweep.jl")
# TrajectoryAnalysis is in exp/ (two levels up from exp/hockey/versions)
const TRAJ_ANALYSIS_PATH = joinpath(dirname(dirname(SCRIPT_DIR)), "TrajectoryAnalysis.jl")

@everywhere include($PARAM_SWEEP_PATH)
if !@isdefined(TrajectoryAnalysis)
    @eval include(TRAJ_ANALYSIS_PATH)
end
using .TrajectoryAnalysis
# Ensure Hockey is loaded for JLD2 type resolution
using .TrajectoryAnalysis.Hockey

const HOCKEY_VISUALS_PATH = joinpath(dirname(SCRIPT_DIR), "src", "HockeyVisuals.jl")
if isfile(HOCKEY_VISUALS_PATH)
    include(HOCKEY_VISUALS_PATH)
end


function generate_config_list()
    configs = Dict{Symbol, Any}[]
    
    # Generate logarithmic scale: 0.1x to 10x
    function log_scale(default_val; n_points=3)
        return [default_val * 10^exp for exp in range(-1, 1, length=n_points)]
    end
    
    # Helper for clean value strings
    fmt(v) = replace(string(round(v, sigdigits=2)), "." => "p")
    
    # Parameter ranges (same defaults, used independently per player)
    ccw_range = log_scale(0.5)   # control_cost_weight
    tcw_range = log_scale(2.0)   # terminal_cost_weight
    # bcw_range = log_scale(10.0)  # boundary_cost_weight
    sdw_range = log_scale(0.1)   # steal_dist_weight
    # suw_range = log_scale(20.0)  # shot_uncertainty_weight
    nccw_range = log_scale(3000.0) # nature_control_cost_weight (only for robust P2)
    
    # Sensor models
    sensor_models = ["low", "medium", "high"]
    
    # Full factorial sweep - independent loops for P1 and P2 params
    for p1_ccw in ccw_range, p1_tcw in tcw_range, p1_sdw in sdw_range
        for p2_ccw in ccw_range, p2_tcw in tcw_range, p2_sdw in sdw_range
            for sensor in sensor_models
                # Base name WITHOUT nccw (for non-robust baseline)
                base_name_nr = "p1_$(fmt(p1_ccw))_$(fmt(p1_tcw))_$(fmt(p1_sdw))_" *
                               "p2_$(fmt(p2_ccw))_$(fmt(p2_tcw))_$(fmt(p2_sdw))_" *
                               "sensor_$(sensor)"
                
                # Non-robust config - only ONE per base config
                push!(configs, Dict{Symbol, Any}(
                    :name => "$(base_name_nr)_non_robust",
                    :sensor_model => sensor,
                    :player_configs => Dict(
                        1 => Dict(
                            :type => non_robust,
                            :control_cost_weight => p1_ccw,
                            :terminal_cost_weight => p1_tcw,
                            :steal_dist_weight => p1_sdw,
                        ),
                        2 => Dict(
                            :type => non_robust,
                            :control_cost_weight => p2_ccw,
                            :terminal_cost_weight => p2_tcw,
                            :steal_dist_weight => p2_sdw,
                        )
                    ),
                    :planning_horizon => 5,
                    :horizon => 10,
                    :trials => 3
                ))
                
                # Robust configs - one per nccw value
                for nccw in nccw_range
                    # nature_bounds_cost_weight is 5x P2's control_cost_weight
                    nbcw = 5 * p2_ccw
                    
                    # Base name WITH nccw (for robust)
                    base_name_r = "p1_$(fmt(p1_ccw))_$(fmt(p1_tcw))_$(fmt(p1_sdw))_" *
                                  "p2_$(fmt(p2_ccw))_$(fmt(p2_tcw))_$(fmt(p2_sdw))_" *
                                  "nccw_$(fmt(nccw))_sensor_$(sensor)"
                    
                    push!(configs, Dict{Symbol, Any}(
                        :name => "$(base_name_r)_robust",
                        :sensor_model => sensor,
                        :player_configs => Dict(
                            1 => Dict(
                                :control_cost_weight => p1_ccw,
                                :terminal_cost_weight => p1_tcw,
                                :steal_dist_weight => p1_sdw,
                            ),
                            2 => Dict(
                                :control_cost_weight => p2_ccw,
                                :terminal_cost_weight => p2_tcw,
                                :steal_dist_weight => p2_sdw,
                                :type => robust,
                                :nature_control_cost_weight => nccw,
                                :nature_bounds_cost_weight => nbcw
                            )
                        ),
                        :planning_horizon => 5,
                        :horizon => 10,
                        :trials => 3
                    ))
                end
            end
        end
    end
    
    return configs
end

"""
Generate only non-robust configs to run baselines for existing robust data.
"""
function generate_non_robust_config_list()
    configs = Dict{Symbol, Any}[]
    
    function log_scale(default_val; n_points=3)
        return [default_val * 10^exp for exp in range(-1, 1, length=n_points)]
    end
    
    fmt(v) = replace(string(round(v, sigdigits=2)), "." => "p")
    
    ccw_range = log_scale(0.5)
    tcw_range = log_scale(2.0)
    sdw_range = log_scale(0.1)
    sensor_models = ["low", "medium", "high"]
    
    for p1_ccw in ccw_range, p1_tcw in tcw_range, p1_sdw in sdw_range
        for p2_ccw in ccw_range, p2_tcw in tcw_range, p2_sdw in sdw_range
            for sensor in sensor_models
                base_name = "p1_$(fmt(p1_ccw))_$(fmt(p1_tcw))_$(fmt(p1_sdw))_" *
                            "p2_$(fmt(p2_ccw))_$(fmt(p2_tcw))_$(fmt(p2_sdw))_" *
                            "sensor_$(sensor)"
                
                push!(configs, Dict{Symbol, Any}(
                    :name => "$(base_name)_non_robust",
                    :sensor_model => sensor,
                    :player_configs => Dict(
                        1 => Dict(
                            :type => non_robust,
                            :control_cost_weight => p1_ccw,
                            :terminal_cost_weight => p1_tcw,
                            :steal_dist_weight => p1_sdw,
                        ),
                        2 => Dict(
                            :type => non_robust,
                            :control_cost_weight => p2_ccw,
                            :terminal_cost_weight => p2_tcw,
                            :steal_dist_weight => p2_sdw,
                        )
                    ),
                    :planning_horizon => 5,
                    :horizon => 10,
                    :trials => 3
                ))
            end
        end
    end
    
    return configs
end

"""
Run only non-robust baselines to complement existing robust data in sweep/.
Saves to sweep/ directory alongside existing robust directories.
"""
function run_non_robust_baselines(; output_subdir="sweep")
    config_list = generate_non_robust_config_list()
    println("Number of non-robust baseline configs: $(length(config_list))")
    println("Total trials: $(length(config_list) * 3)")
    println("Continue? (y/n)")
    
    if readline() != "y"
        return
    end
    
    results = Main.run_param_sweep(config_list; cores=40, output_subdir=output_subdir)
    println("Non-robust baselines completed.")
    return results
end

function run_sweep()
    # Define a small config
    config_list = generate_config_list()
    println("Number of configs: $(length(config_list) * 3). Continue? (y/n)")

    if readline() != "y"
        return
    end
    
    results = Main.run_param_sweep(config_list; cores=40, output_subdir="sweepv2")
    
    output_dir = joinpath(@__DIR__, "..", "outputs", "sweepv2")

    
    # Iterate over subdirectories in output_dir (each corresponds to a distinct case)
    if isdir(output_dir)
        for sub_dir in readdir(output_dir)
            full_path = joinpath(output_dir, sub_dir)
            if isdir(full_path)
                println("\n" * "="^40)
                println("Analyzing directory: $sub_dir")
                println("="^40)
                load_and_analyze_solution_files(;
                    directory=full_path
                )
                get_trajectory_summary(;directory=full_path);
            end
        end
    else
        println("Output directory $output_dir does not exist.")
    end

    println("Verification complete.")
end
# ==============================================================================
# Analysis Tools
# ==============================================================================

"""
Calculate the sum of defender costs from a solution history using pre-computed costs.
The defender is player 2 (index 2).
Uses the first step of each RH iteration (i.e., the executed trajectory costs).
"""

function calculate_defender_total_cost(data)
    # Extract necessary data
    sol_hist = get(data, "solution_history", nothing)
    gt_hist = get(data, "gt_state_history", nothing)
    params = get(data, "params", nothing)
    
    if isnothing(sol_hist) || isnothing(gt_hist) || isnothing(params)
        return 0.0
    end
    
    # Construct dummy entry for TrajectoryAnalysis
    # We use "dummy" for fields that don't matter for cost calculation
    entry = TrajectoryAnalysis.TrajectoryAnalysisEntry(
        "temp", 0, gt_hist, [], sol_hist, [], [], [], params, false, "temp"
    )
    
    # Compute executed costs using standardized logic
    executed_costs = TrajectoryAnalysis.compute_executed_trajectory_costs(entry, false)
    
    total_cost = 0.0
    if !isempty(executed_costs)
        for step_costs in executed_costs
            if haskey(step_costs, :defender)
                step = step_costs[:defender]
                # Sum values
                step_total = sum(values(step))
                total_cost += step_total
            end
        end
    end

    return total_cost
end

function calculate_kkt_stats(solution_history)
    total_samples = 0
    high_error_samples = 0
    threshold = 0.01

    if isnothing(solution_history) return 0.0 end
    
    sorted_keys = sort(collect(keys(solution_history)))
    for t in sorted_keys
        sols = solution_history[t]
        for (i, player_data) in enumerate(sols)
            if hasproperty(player_data, :kkt_error)
                err = getproperty(player_data, :kkt_error)
                if !isnothing(err)
                    total_samples += length(err)
                    high_error_samples += sum(err .> threshold)
                end
            end
        end
    end
    return total_samples > 0 ? (high_error_samples / total_samples) : 0.0
end

"""
Load and process a single jld2 file, returning cost, KKT stats, and GT history.
"""
function analyze_trial_file(filepath, verbose=false)
    try
        data = with_logger(NullLogger()) do
            load(filepath)
        end
        
        if !haskey(data, "solution_history")
            return nothing
        end
        
        sol_hist = data["solution_history"]
        gt_hist = get(data, "gt_state_history", nothing)
        
        cost = calculate_defender_total_cost(data)
        kkt_pct = calculate_kkt_stats(sol_hist)
        
        return (; cost=cost, kkt_pct=kkt_pct, gt_hist=gt_hist)
    catch e
        println("  Error processing $filepath: $e")
        return nothing
    end
end

function extract_base_config(dirname::String)
    result = dirname
    result = replace(result, r"_p2_nbc[\d.]+(?=_)" => "")
    result = replace(result, r"_p2_ncc[\d.]+(?=_)" => "")
    result = replace(result, r"_p2_nbc[\d.]+$" => "")
    result = replace(result, r"_p2_ncc[\d.]+$" => "")
    return result
end

function is_robust_config(dirname::String)
    return occursin("_nbc", dirname) && occursin("_ncc", dirname)
end

function get_dir_stats(dirpath::String, verbose=false)
    if !isdir(dirpath) return nothing end
    files = filter(f -> endswith(f, ".jld2"), readdir(dirpath))
    if isempty(files) return nothing end
    
    costs = Float64[]
    kkt_pcts = Float64[]
    trajectories = []
    
    for f in files
        res = analyze_trial_file(joinpath(dirpath, f), verbose)
        if !isnothing(res)
            push!(costs, res.cost)
            push!(kkt_pcts, res.kkt_pct)
            if !isnothing(res.gt_hist)
                push!(trajectories, res.gt_hist)
            end
        end
    end
    
    if isempty(costs) return nothing end
    
    return (; 
        mean_cost = mean(costs), 
        mean_kkt = mean(kkt_pcts), 
        trajectories = trajectories,
        n_trials = length(costs)
    )
end

function calculate_mean_trajectory_difference(traj_list_1, traj_list_2)
    # Average pairwise distance between trajectories    
    total_diff = 0.0
    valid_pairs = 0
    
    # for i in 1:n
    for t1 in traj_list_1
        for t2 in traj_list_2
            
            # len = min(length(t1), length(t2))
            @assert length(t1) == length(t2)
            dist_sum = sum(norm(t1[t] - t2[t]) for t in 1:length(t1))
            total_diff += dist_sum / length(t1)
            valid_pairs += 1
        end
    end
    
    return valid_pairs > 0 ? (total_diff / valid_pairs) : 0.0
end

"""
Analyze sweep results to find configurations where the robust defender outperforms the non-robust version.
Also computes KKT error stats and trajectory differences.
"""
function analyze_sweep_results(; 
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep"),
    top_k::Int = 10,
    verbose::Bool = false
    )
    if !isdir(sweep_dir)
        println("Sweep directory not found: $sweep_dir")
        return []
    end
    
    println("Analyzing sweep results in: $sweep_dir")
    
    all_dirs = filter(d -> isdir(joinpath(sweep_dir, d)), readdir(sweep_dir))
    robust_dirs = filter(is_robust_config, all_dirs)
    non_robust_dirs = filter(d -> !is_robust_config(d), all_dirs)
    
    # Map base config -> non-robust dir
    non_robust_map = Dict{String, String}()
    for d in non_robust_dirs
        base = extract_base_config(d)
        non_robust_map[base] = d
    end
    
    results = [] 
    
    for (i, robust_dir) in enumerate(robust_dirs)
        if verbose && i % 100 == 0; print("."); end
        base_config = extract_base_config(robust_dir)
        
        if haskey(non_robust_map, base_config)
            non_robust_dir = non_robust_map[base_config]
            
            r_stats = get_dir_stats(joinpath(sweep_dir, robust_dir), verbose)
            nr_stats = get_dir_stats(joinpath(sweep_dir, non_robust_dir), verbose)
            
            if !isnothing(r_stats) && !isnothing(nr_stats)
                if abs(nr_stats.mean_cost) > 1e-10
                    traj_diff = calculate_mean_trajectory_difference(r_stats.trajectories, nr_stats.trajectories)
                    
                    # Cost improvement: positive means robust has lower (better) cost
                    cost_improvement = nr_stats.mean_cost - r_stats.mean_cost
                    
                    # Only include if robust actually outperforms non-robust
                        push!(results, (
                            cost_improvement = cost_improvement,
                            traj_diff = traj_diff,
                            robust_dir = robust_dir,
                            non_robust_dir = non_robust_dir,
                            r_cost = r_stats.mean_cost,
                            nr_cost = nr_stats.mean_cost,
                            r_kkt = r_stats.mean_kkt,
                            nr_kkt = nr_stats.mean_kkt,
                            r_trials = r_stats.n_trials,
                            nr_trials = nr_stats.n_trials
                        ))
                end
            end
        end
    end
    
    println("\nProcessed $(length(results)) valid pairs.")
    
    if isempty(results)
        println("No valid results found.")
        return []
    end
    
    # Sort by Cost Improvement (Largest improvement first = robust defender saves most cost)
    sort!(results, by = x -> x.cost_improvement, rev=true)
    
    println("\n" * "="^100)
    println("TOP $top_k ROBUST CONFIGURATIONS (Largest Cost Improvement)")
    println("="^100)
    
    for (i, res) in enumerate(results[1:min(top_k, length(results))])
        println("\n[$i] Cost Improvement: $(round(res.cost_improvement, digits=3)) (robust: $(round(res.r_cost, digits=3)), non-robust: $(round(res.nr_cost, digits=3)))")
        println("    Robust:     $(res.robust_dir)")
        println("    Non-Robust: $(res.non_robust_dir)")
        println("    Statistics: Improvement: $(round(res.cost_improvement, digits=3)) | Traj Diff: $(round(res.traj_diff, digits=4)) | Robust KKT >0.01: $(round(res.r_kkt*100, digits=1))% | Non-Robust KKT >0.01: $(round(res.nr_kkt*100, digits=1))%")
    end
    
    # Sort by Trajectory Difference (Largest)
    sort!(results, by = x -> x.traj_diff, rev=true)
    
    println("\n" * "="^100)
    println("TOP $top_k CONFIGURATIONS BY TRAJECTORY DIFFERENCE (Largest Impact)")
    println("="^100)
    
    for (i, res) in enumerate(results[1:min(top_k, length(results))])
        println("\n[$i] Traj Diff: $(round(res.traj_diff, digits=4))")
        println("    Robust:     $(res.robust_dir)")
        println("    Non-Robust: $(res.non_robust_dir)")
        println("    Statistics: Improvement: $(round(res.cost_improvement, digits=3)) | Traj Diff: $(round(res.traj_diff, digits=4)) | Robust KKT >0.01: $(round(res.r_kkt*100, digits=1))% | Non-Robust KKT >0.01: $(round(res.nr_kkt*100, digits=1))%")
    end
    
    # Return sorted by cost improvement (default preference)
    sort!(results, by = x -> x.cost_improvement, rev=true)
    return results
end

"""
Visualize the trajectories for a specific rank from the best robust runs.
Loads 1 robust and 1 non-robust trial.
"""
function visualize_sweep_rank(rank::Int;
    k::Int = 10,
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep"),
    results = nothing
    )
    if isnothing(results)
        println("Analyzing sweep results to find top configurations...")
        results = analyze_sweep_results(sweep_dir=sweep_dir, verbose=false, top_k=k)
        sort!(results, by = x -> x.traj_diff, rev=true)
    end
    
    if isempty(results) || rank > length(results)
        println("Rank $rank out of bounds (found $(length(results)) results)")
        return
    end
    
    entry = results[rank]
    println("\nVisualizing Rank $rank:")
    println("  Robust Dir:     $(entry.robust_dir)")
    println("  Non-Robust Dir: $(entry.non_robust_dir)")
    println("  Cost Improvement: $(round(entry.cost_improvement, digits=3)) (robust: $(round(entry.r_cost, digits=3)), non-robust: $(round(entry.nr_cost, digits=3)))")
    println("  Traj Diff:        $(round(entry.traj_diff, digits=4))")
    
    solutions = Dict{String, Any}()
    
    function load_dir_to_sol(base_dir, prefix, subdir, limit=1)
        path = joinpath(base_dir, subdir)
        if !isdir(path)
            println("Warning: Directory not found: $path")
            return
        end
        
        count = 0
        for f in readdir(path)
            if endswith(f, ".jld2")
                try
                    key_name = "$prefix | $f"
                    solutions[key_name] = load(joinpath(path, f))
                    count += 1
                    if count >= limit
                        break
                    end
                catch e
                     println("Failed to load $f: $e")
                end
            end
        end
    end
    
    load_dir_to_sol(sweep_dir, "Robust", entry.robust_dir, 1)
    load_dir_to_sol(sweep_dir, "Non-Robust", entry.non_robust_dir, 1)
    
    if isempty(solutions)
        println("No valid solution files found to visualize.")
        return
    end
    
    println("Loaded $(length(solutions)) trials. Launching visualizer...")
    
    # Generate Analysis Plots - organize by robust config name
    analysis_dir = joinpath(dirname(sweep_dir), "..", "analysis", entry.robust_dir)
    generate_comparison_plots(solutions, rank, analysis_dir)
    
    if @isdefined(visualize_receding_horizon_solutions_multi_figure)
        visualize_receding_horizon_solutions_multi_figure(solutions, [[-1.5, 0.25], [-1.5, -0.25]])
    else
        println("Error: visualize_receding_horizon_solutions_multi_figure not defined. Check HockeyVisuals.jl include.")
    end
end

"""
Interactive visualization tool.
Prompts the user to choose a sorting criteria (Cost Improvement vs Trajectory Diff)
and a rank, then visualizes the selected result.
"""
function visualize_sweep(; 
    k::Int = 10,
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep")
)
    # 1. Run Analysis
    println("Running sweep analysis to gather candidates...")
    # Get top 50 candidates to allow user some range
    results = analyze_sweep_results(sweep_dir=sweep_dir, verbose=false, top_k=k)
    
    if isempty(results)
        println("No results found.")
        return
    end

    while true    
        # 2. Ask for Sorting Criteria
        println("\n" * "="^60)
        println("SELECT RANKING CRITERIA")
        println("="^60)
        println("[1] Largest Cost Improvement (Robust defender saves most cost compared to non-robust)")
        println("[2] Largest Trajectory Difference (Robust strategy causes biggest change)")
        print("\nEnter choice [default: 1]: ")
        
        input = readline()
        if strip(input) == "2"
            sort!(results, by = x -> x.traj_diff, rev=true)
            println("\nSorted by: Largest Trajectory Difference")
        elseif strip(input) == "1"
            sort!(results, by = x -> x.cost_improvement, rev=true)
            println("\nSorted by: Largest Cost Improvement")
        elseif strip(input) == "q"
            continue
        end
        
        # 3. Show Top Options
        println("\nTop Options:")
        for i in 1:min(k, length(results))
            res = results[i]
            imp_str = "Improvement: $(rpad(round(res.cost_improvement, digits=2), 7))"
            d_str = "Diff: $(rpad(round(res.traj_diff, digits=3), 6))"
            kkt_str = "KKT>0.01: $(round(res.r_kkt*100, digits=0))%"
            trials_str = "Trials: $(res.r_trials)/$(res.nr_trials)"
            println("[$i] $imp_str | $d_str | $kkt_str | $trials_str" )
        end
    
        # 4. Ask for Rank
        print("\nEnter rank to visualize [default: 1]: ")
        rank_input = readline()
        rank = 0
        if strip(rank_input) == "q"
            continue
        elseif typeof(tryparse(Int, rank_input)) <: Int
            rank = tryparse(Int, rank_input)
        end
        
        # 5. Ask for extra trials
        print("\nHow many extra trials to run for this config? [default: 0]: ")
        extra_trials_input = readline()
        extra_trials = 0
        if strip(extra_trials_input) == "q"
            continue
        elseif typeof(tryparse(Int, extra_trials_input)) <: Int
            extra_trials = tryparse(Int, extra_trials_input)
        end
        
        # 5b. Run extra trials if requested
        if extra_trials > 0
            entry = results[rank]
            println("\nRunning $extra_trials extra trials for both robust and non-robust...")
            run_extra_trials(entry, extra_trials, sweep_dir)
            
            # Re-analyze to include new trials
            println("\nRe-analyzing sweep results with new trials...")
            results = analyze_sweep_results(sweep_dir=sweep_dir, verbose=false, top_k=k)
            sort!(results, by = x -> x.traj_diff, rev=true)
        end
        
        # 6. Visualize
        visualize_sweep_rank(rank, results=results, sweep_dir=sweep_dir; k=k)
    end
end

"""
    run_extra_trials(entry, num_trials, sweep_dir)

Run additional trials for a given configuration (both robust and non-robust).
Uses the exact params from existing trial files directly.
Counts existing trials and uses trial_offset to avoid overwriting existing files.
"""
function run_extra_trials(entry, num_trials::Int, sweep_dir::String)
    robust_path = joinpath(sweep_dir, entry.robust_dir)
    non_robust_path = joinpath(sweep_dir, entry.non_robust_dir)
    
    println("\n=== DEBUG: run_extra_trials ===")
    println("Robust Path:     $robust_path")
    println("Non-Robust Path: $non_robust_path")

    # Count existing trials in each directory
    function count_existing_trials(dir_path)
        if !isdir(dir_path)
            println("  Directory not found: $dir_path")
            return 0
        end
        ate = count(f -> endswith(f, ".jld2"), readdir(dir_path))
        println("  Found $ate existing trials in $dir_path")
        return ate
    end
    
    robust_existing = count_existing_trials(robust_path)
    non_robust_existing = count_existing_trials(non_robust_path)
    
    # Load an existing trial to get the params
    robust_params = nothing
    non_robust_params = nothing
    
    println("Attempting to load base params...")
    for path in [robust_path, non_robust_path]
        if isdir(path)
            for f in readdir(path)
                if endswith(f, ".jld2")
                    try
                        data = load(joinpath(path, f))
                        if haskey(data, "params")
                            if path == robust_path
                                robust_params = deepcopy(data["params"])
                                println("  [SUCCESS] Loaded robust params from $f")
                                # DEBUG: Check output dir in loaded params
                                println("    > robust_params.output_dir: $(robust_params.output_dir)")
                                println("    > robust_params.name: $(robust_params.name)")
                            else
                                non_robust_params = deepcopy(data["params"])
                                println("  [SUCCESS] Loaded non-robust params from $f")
                                println("    > non-robust_params.output_dir: $(non_robust_params.output_dir)")
                                println("    > non-robust_params.name: $(non_robust_params.name)")
                            end
                            break
                        end
                    catch e
                        println("Warning: Failed to load $f: $e")
                    end
                end
            end
        end
    end
    
    if isnothing(robust_params) && isnothing(non_robust_params)
        println("Error: Could not load params from existing trials.")
        return
    end
    
    if !isnothing(robust_params)
        old_robust_dir = robust_params.output_dir
        robust_params.output_dir = robust_path
        if old_robust_dir != robust_path
            println("  [FIX] Updated robust output_dir:")
            println("    OLD: $old_robust_dir")
            println("    NEW: $robust_path")
        end
    end
    if !isnothing(non_robust_params)
        old_nr_dir = non_robust_params.output_dir
        non_robust_params.output_dir = non_robust_path
        if old_nr_dir != non_robust_path
            println("  [FIX] Updated non-robust output_dir:")
            println("    OLD: $old_nr_dir")
            println("    NEW: $non_robust_path")
        end
    end
    
    # Determine the trial offset (use max of existing counts to ensure unique trial numbers)
    trial_offset = max(robust_existing, non_robust_existing)
    println("Found $robust_existing existing robust trials, $non_robust_existing existing non-robust trials.")
    println("New trials will start from trial $(trial_offset + 1).")
    
    params_list = HockeyParams[]
    
    if !isnothing(robust_params)
        robust_params.trials = num_trials
        push!(params_list, robust_params)
    end
    
    if !isnothing(non_robust_params)
        non_robust_params.trials = num_trials
        push!(params_list, non_robust_params)
    end
    
    if isempty(params_list)
        println("No params to run.")
        return
    end
    
    # Count total trials for display
    total_trials = sum(p.trials for p in params_list)
    r_count = robust_params !== nothing ? num_trials : 0
    nr_count = non_robust_params !== nothing ? num_trials : 0
    println("Running $total_trials new trials ($r_count robust + $nr_count non-robust)...")
    
    run_experiment_batch(params_list; cores=min(40, total_trials), trial_offset=trial_offset)
    
    println("Extra trials completed. Total trials now: $(trial_offset + num_trials) each.")
end

function generate_comparison_plots(solutions, rank, output_dir)
    # Ensure directory exists
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # 1. Clear Trackers
    TrajectoryAnalysis.clear_trajectory_tracker!()
    TrajectoryAnalysis.KKTErrorTracker.clear_rh_kkt_tracker!()
    
    println("Populating analysis trackers...")
    
    # 2. Populate from solutions
    for (key_name, solution_data) in solutions
        # Robustness check
        is_robust = occursin("Robust", key_name) && !occursin("Non-Robust", key_name)
        
        gt_state_history = get(solution_data, "gt_state_history", nothing)
        solution_history = get(solution_data, "solution_history", nothing)
        params = get(solution_data, "params", nothing)
        
        if !isnothing(solution_history)
            # Internal call to populate KKT
            TrajectoryAnalysis.extract_kkt_errors_from_history(solution_history, key_name, is_robust)
            
            if !isnothing(gt_state_history) && !isnothing(params)
                try
                    entry = TrajectoryAnalysis.TrajectoryAnalysisEntry(
                        key_name,
                        rank,
                        gt_state_history,
                        [], # observations
                        solution_history,
                        [], # cond
                        [], # lq
                        [], # costs
                        params,
                        is_robust,
                        "rank_$rank"
                    )
                    push!(TrajectoryAnalysis.TRAJECTORY_TRACKER.entries, entry)
                catch e
                    println("Warning: Failed to create TrajectoryAnalysisEntry for $key_name: $e")
                end
            end
        end
    end
    
    # 3. Generate Plots
    println("Generating plots via TrajectoryAnalysis...")
    
    # KKT
    TrajectoryAnalysis.KKTErrorTracker.create_kkt_yarnball_plots()
    if isfile("kkt_temporal_evolution.png")
        mv("kkt_temporal_evolution.png", joinpath(output_dir, "kkt.png"), force=true)
        println("Saved KKT plot to $(joinpath(output_dir, "kkt.png"))")
    end
    
    # --- Statistical Significance Report ---
    println("Generating statistical significance report...")
    robust_costs = Float64[]
    non_robust_costs = Float64[]
    
    for (key_name, solution_data) in solutions
        # Robustness check
        is_robust = occursin("Robust", key_name) && !occursin("Non-Robust", key_name)
        cost = calculate_defender_total_cost(solution_data)
        
        if is_robust
            push!(robust_costs, cost)
        else
            push!(non_robust_costs, cost)
        end
    end
    
    open(joinpath(output_dir, "significance_report.txt"), "w") do io
        println(io, "Statistical Significance Report (Defender Total Cost)")
        println(io, "===================================================")
        
        n_r = length(robust_costs)
        n_nr = length(non_robust_costs)
        
        if n_r > 1 && n_nr > 1
            mean_r = mean(robust_costs)
            std_r = std(robust_costs)
            mean_nr = mean(non_robust_costs)
            std_nr = std(non_robust_costs)
            
            # Welch's t-test
            se_diff = sqrt((std_r^2 / n_r) + (std_nr^2 / n_nr))
            t_stat = (mean_r - mean_nr) / se_diff
            
            # Degrees of freedom (Welch-Satterthwaite equation)
            df_num = ((std_r^2 / n_r) + (std_nr^2 / n_nr))^2
            df_den = ((std_r^2 / n_r)^2 / (n_r - 1)) + ((std_nr^2 / n_nr)^2 / (n_nr - 1))
            df = df_num / df_den
            
            # P-value (approximate using normal distribution for large degrees of freedom, or simple lookup/heuristic)
            # Since we don't have Distributions.jl loaded explicitly here, we'll report t-stat.
            # However, Distributions IS loaded in ExperimentRunner.jl, let's see if it's available.
            # ExperimentRunner is a module... let's check imports.
            # param_sweep.jl uses Distributions. Let's assume Main has it or we can just report t-stat.
            # For now, just reporting stats is good, t >= 1.96 roughly significant at 0.05.
            
            println(io, "Robust (n=$n_r):     Mean = $(round(mean_r, digits=4)), Std = $(round(std_r, digits=4))")
            println(io, "Non-Robust (n=$n_nr): Mean = $(round(mean_nr, digits=4)), Std = $(round(std_nr, digits=4))")
            println(io, "---------------------------------------------------")
            println(io, "Difference (Robust - Non-Robust): $(round(mean_r - mean_nr, digits=4))")
            println(io, "T-Statistic: $(round(t_stat, digits=4))")
            println(io, "Degrees of Freedom: $(round(df, digits=2))")
            
            critical_val_05 = 1.96 # Approx cost large df
            is_sig = abs(t_stat) > critical_val_05
            
            println(io, "Significant Difference (p < 0.05 approx): $(is_sig ? "YES" : "NO")")
            if is_sig
                better = mean_r < mean_nr ? "Robust is better (lower cost)" : "Non-Robust is better (lower cost)"
                println(io, "Conclusion: $better")
            else
                println(io, "Conclusion: No significant difference detected.")
            end
        else
            println(io, "Insufficient data for t-test (n_robust=$n_r, n_non_robust=$n_nr). Need at least 2 samples each.")
        end
    end
    println("Saved significance report to $(joinpath(output_dir, "significance_report.txt"))")
    
    # Costs & Actions
    # Pass directory to save plots there
    TrajectoryAnalysis.get_trajectory_summary(directory=output_dir)
    println("Saved Cost and Action comparisons to $output_dir")
    
    # 4. Generate Specific Defender Cost Comparison
    println("Generating Defender-only cost comparison...")
    try
        robust_entry = nothing
        non_robust_entry = nothing
        
        for entry in TrajectoryAnalysis.TRAJECTORY_TRACKER.entries
            if entry.robust
                robust_entry = entry
            else
                non_robust_entry = entry
            end
        end
        
        if !isnothing(robust_entry) && !isnothing(non_robust_entry)
            
            function get_defender_cost_series(entry)
                executed_costs = TrajectoryAnalysis.compute_executed_trajectory_costs(entry, false)
                
                costs = Float64[]
                for step_costs in executed_costs
                    if haskey(step_costs, :defender)
                        
                        step = step_costs[:defender]
                        
                        # Sum all components
                        total_val = sum(values(step))
                        push!(costs, total_val)
                    else
                        push!(costs, 0.0)
                    end
                end
                if !isnothing(robust_entry) && !isnothing(non_robust_entry)
                    try
                        TrajectoryAnalysis.create_defender_yarnball_comparison(robust_entry, non_robust_entry; output_dir=output_dir)
                    catch e
                        println("Error creating defender yarnball: $e")
                    end
                end
                
                return costs
            end
            
            r_costs = get_defender_cost_series(robust_entry)
            nr_costs = get_defender_cost_series(non_robust_entry)
            
            fig = Figure(size=(1000, 500))
            
            ax1 = Axis(fig[1, 1], title="Defender Instantaneous Cost", xlabel="Time Step", ylabel="Cost")
            if !isempty(r_costs)
                lines!(ax1, 1:length(r_costs), r_costs, color=:blue, label="Robust", linewidth=2)
            end
            if !isempty(nr_costs)
                lines!(ax1, 1:length(nr_costs), nr_costs, color=:red, label="Non-Robust", linewidth=2)
            end
            axislegend(ax1)
            
            ax2 = Axis(fig[1, 2], title="Defender Cumulative Cost", xlabel="Time Step", ylabel="Total Cost")
            if !isempty(r_costs)
                lines!(ax2, 1:length(r_costs), cumsum(r_costs), color=:blue, label="Robust", linewidth=2)
            end
            if !isempty(nr_costs)
                lines!(ax2, 1:length(nr_costs), cumsum(nr_costs), color=:red, label="Non-Robust", linewidth=2)
            end
            axislegend(ax2)
            
            save(joinpath(output_dir, "defender_cost_comparison.png"), fig)
            println("Saved defender cost comparison to $(joinpath(output_dir, "defender_cost_comparison.png"))")
        else
            println("Could not find both robust and non-robust entries for cost comparison.")
        end
    catch e
        println("Error generating defender cost plot: $e")
    end
end
