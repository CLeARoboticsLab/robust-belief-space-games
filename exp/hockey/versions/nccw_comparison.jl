"""
    nccw_comparison.jl

Two-pass batch analysis of parameter combinations across different nature_control_cost_weight
(nccw) values. Groups configs that are identical except for nccw, requires minimum trial counts.

Workflow:
    include("test_sweep.jl")
    include("nccw_comparison.jl")

    # Pass 1: Generate Q-Q plots for all nccw variants to inspect outliers
    groups = generate_qq_plots_for_nccw_groups()

    # Inspect the Q-Q plots in analysis/<base_config>/nccw_<value>/qq_plots.png
    # Then fill in the outlier counts:

    # Per-nccw robust outlier counts
    robust_outliers = Dict(
        10.0   => 4,
        100.0  => 4,
        # ... etc
    )

    # Single non-robust outlier count (shared baseline)
    non_robust_outliers = 1

    # Pass 2: Run full analysis with outlier removal
    analyze_nccw_groups(robust_outliers=robust_outliers, non_robust_outliers=non_robust_outliers)
"""

using JLD2
using Statistics
using Logging

"""
    extract_nccw(dirname::String) -> Union{Float64, Nothing}

Extract the nature_control_cost_weight value from a robust directory name.
Returns `nothing` for non-robust directories.
"""
function extract_nccw(dirname::String)
    m = match(r"p2_ncc([\d.]+)", dirname)
    isnothing(m) && return nothing
    return parse(Float64, m.captures[1])
end

"""
    find_nccw_groups(; sweep_dir, min_trials) -> Vector{NamedTuple}

Scan the sweep directory and find groups of parameter configurations that:
1. Share the same base config (everything except nccw/nbc)
2. Have a non-robust baseline with >= `min_trials` trials
3. Have multiple robust variants (different nccw) each with >= `min_trials` trials

Returns a vector of named tuples, each containing:
- `base_config`: The shared base config string
- `non_robust_dir`: Directory name of the non-robust baseline
- `nr_trials`: Number of non-robust trials
- `robust_variants`: Vector of (nccw, dir_name, n_trials) sorted by nccw
"""
function find_nccw_groups(;
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep"),
    min_trials::Int = 50
)
    if !isdir(sweep_dir)
        println("Sweep directory not found: $sweep_dir")
        return []
    end

    all_dirs = filter(d -> isdir(joinpath(sweep_dir, d)), readdir(sweep_dir))

    function count_trials(dirname)
        path = joinpath(sweep_dir, dirname)
        return count(f -> endswith(f, ".jld2"), readdir(path))
    end

    non_robust_map = Dict{String, Tuple{String, Int}}()
    robust_map = Dict{String, Vector{Tuple{Float64, String, Int}}}()

    for d in all_dirs
        n = count_trials(d)
        n < min_trials && continue

        if is_robust_config(d)
            base = extract_base_config(d)
            nccw = extract_nccw(d)
            isnothing(nccw) && continue
            group = get!(robust_map, base, Tuple{Float64, String, Int}[])
            push!(group, (nccw, d, n))
        else
            base = extract_base_config(d)
            non_robust_map[base] = (d, n)
        end
    end

    groups = NamedTuple[]
    for (base, variants) in robust_map
        haskey(non_robust_map, base) || continue
        length(variants) < 2 && continue

        nr_dir, nr_n = non_robust_map[base]
        sort!(variants, by = v -> v[1])

        push!(groups, (
            base_config = base,
            non_robust_dir = nr_dir,
            nr_trials = nr_n,
            robust_variants = variants
        ))
    end

    sort!(groups, by = g -> length(g.robust_variants), rev=true)
    return groups
end

"""
    load_dir_solutions(sweep_dir, subdir, prefix) -> Dict{String, Any}

Load all .jld2 trial files from a single directory into a solutions dict
keyed by "<prefix> | <filename>".
"""
function load_dir_solutions(sweep_dir::String, subdir::String, prefix::String)
    solutions = Dict{String, Any}()
    path = joinpath(sweep_dir, subdir)
    isdir(path) || return solutions
    for f in readdir(path)
        endswith(f, ".jld2") || continue
        try
            solutions["$prefix | $f"] = with_logger(NullLogger()) do
                load(joinpath(path, f))
            end
        catch e
            println("  Failed to load $f: $e")
        end
    end
    return solutions
end

"""
    compute_nr_baseline(nr_solutions, non_robust_outliers) -> NamedTuple

Compute the non-robust baseline data once: costs (with outlier removal),
kept keys, and tracker entries. This is reused across all nccw variants.
"""
function compute_nr_baseline(nr_solutions::Dict{String, Any}, non_robust_outliers::Int)
    # Compute costs
    cost_entries = Tuple{Float64, String}[]
    for (key_name, solution_data) in nr_solutions
        cost = calculate_defender_total_cost(solution_data)
        push!(cost_entries, (cost, key_name))
    end

    # Sort and remove outliers
    sort!(cost_entries, by=x->x[1])
    n_remove = min(non_robust_outliers, length(cost_entries) - 2)
    filtered = n_remove > 0 ? cost_entries[1:end-n_remove] : cost_entries

    costs = [e[1] for e in filtered]
    kept_keys = Set(e[2] for e in filtered)

    # Create tracker entries for kept trials
    tracker_entries = TrajectoryAnalysis.TrajectoryAnalysisEntry[]
    for (key_name, solution_data) in nr_solutions
        key_name in kept_keys || continue
        gt_hist = get(solution_data, "gt_state_history", nothing)
        sol_hist = get(solution_data, "solution_history", nothing)
        params = get(solution_data, "params", nothing)
        (isnothing(gt_hist) || isnothing(sol_hist) || isnothing(params)) && continue
        try
            entry = TrajectoryAnalysis.TrajectoryAnalysisEntry(
                key_name, 0, gt_hist, [], sol_hist, [], [], [],
                params, false, "non_robust"
            )
            push!(tracker_entries, entry)
        catch e
            println("    Warning: Failed to create NR entry for $key_name: $e")
        end
    end

    return (
        costs = costs,
        n_total = length(cost_entries),
        n_removed = n_remove,
        kept_keys = kept_keys,
        tracker_entries = tracker_entries,
        solutions = nr_solutions
    )
end

# ============================================================================
# Pass 1: Generate Q-Q plots for outlier inspection
# ============================================================================

"""
    generate_qq_plots_for_nccw_groups(;
        sweep_dir, analysis_base_dir, min_trials,
        robust_outliers = Dict{Float64, Int}(),
        non_robust_outliers = 0
    ) -> Vector{NamedTuple}

Pass 1: For each qualifying nccw variant, loads trials, computes defender costs,
and saves a Q-Q plot to `analysis_base_dir/<base_config>/nccw_<value>/qq_plots.png`.

If outlier counts are provided, the Q-Q plots show data after removing the specified
number of highest-cost outliers. `robust_outliers` maps nccw → count (per-variant).
`non_robust_outliers` is a single Int (shared baseline, computed once).

Returns the groups vector so you can inspect it and build your outlier config.
"""
function generate_qq_plots_for_nccw_groups(;
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep"),
    analysis_base_dir::String = joinpath(@__DIR__, "..", "analysis"),
    min_trials::Int = 50,
    robust_outliers::Dict = Dict{Float64, Int}(),
    non_robust_outliers::Int = 0
)
    groups = find_nccw_groups(sweep_dir=sweep_dir, min_trials=min_trials)

    if isempty(groups)
        println("No parameter groups found with >= $min_trials trials.")
        return groups
    end

    println("="^80)
    println("PASS 1: Generating Q-Q plots for outlier inspection")
    println("="^80)

    for (gi, group) in enumerate(groups)
        println("\nGroup $gi: $(group.base_config)")
        println("  Non-robust baseline: $(group.non_robust_dir) ($(group.nr_trials) trials)")
        group_output_dir = joinpath(analysis_base_dir, group.base_config)

        # Load and process non-robust baseline once per group
        println("  Loading non-robust baseline...")
        nr_solutions = load_dir_solutions(sweep_dir, group.non_robust_dir, "Non-Robust")
        nr_all_costs = sort([calculate_defender_total_cost(d) for d in values(nr_solutions)])
        n_total_nr = length(nr_all_costs)

        n_nr_remove = min(non_robust_outliers, n_total_nr - 2)
        nr_costs = n_nr_remove > 0 ? nr_all_costs[1:end-n_nr_remove] : nr_all_costs

        for (nccw, robust_dir, n_robust) in group.robust_variants
            nccw_label = nccw == floor(nccw) ? string(Int(nccw)) : string(nccw)
            output_dir = joinpath(group_output_dir, "nccw_$nccw_label")
            mkpath(output_dir)

            println("  nccw=$nccw: loading $n_robust robust trials...")

            r_solutions = load_dir_solutions(sweep_dir, robust_dir, "Robust")
            robust_costs = sort([calculate_defender_total_cost(d) for d in values(r_solutions)])
            n_total_r = length(robust_costs)

            # Apply robust outlier removal
            n_r_remove = min(get(robust_outliers, nccw, 0), length(robust_costs) - 2)
            if n_r_remove > 0
                robust_costs = robust_costs[1:end-n_r_remove]
            end

            # Use the shared (already-filtered) non-robust costs
            non_robust_costs = nr_costs

            # Generate Q-Q plot
            outlier_str = (n_r_remove > 0 || n_nr_remove > 0) ?
                " | Outliers removed: R=$n_r_remove, NR=$n_nr_remove" : ""
            fig_qq = Figure(size=(1200, 500))
            Label(fig_qq[0, :],
                text="nccw=$nccw | Robust n=$(length(robust_costs))/$n_total_r, Non-Robust n=$(length(non_robust_costs))/$n_total_nr$outlier_str",
                fontsize=16)

            # Robust Q-Q
            ax_r = Axis(fig_qq[1, 1],
                title="Robust (n=$(length(robust_costs)))",
                xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")
            if length(robust_costs) > 1 && std(robust_costs) > 0
                n_r = length(robust_costs)
                tq = [quantile(Normal(0, 1), (i - 0.5) / n_r) for i in 1:n_r]
                standardized = (sort(robust_costs) .- mean(robust_costs)) ./ std(robust_costs)
                scatter!(ax_r, tq, standardized, color=:blue, markersize=8)
                lines!(ax_r, [-3, 3], [-3, 3], color=:red, linestyle=:dash, linewidth=2)
            end

            # Non-Robust Q-Q
            ax_nr = Axis(fig_qq[1, 2],
                title="Non-Robust (n=$(length(non_robust_costs)))",
                xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")
            if length(non_robust_costs) > 1 && std(non_robust_costs) > 0
                n_nr = length(non_robust_costs)
                tq = [quantile(Normal(0, 1), (i - 0.5) / n_nr) for i in 1:n_nr]
                standardized = (sort(non_robust_costs) .- mean(non_robust_costs)) ./ std(non_robust_costs)
                scatter!(ax_nr, tq, standardized, color=:red, markersize=8)
                lines!(ax_nr, [-3, 3], [-3, 3], color=:red, linestyle=:dash, linewidth=2)
            end

            qq_path = joinpath(output_dir, "qq_plots.png")
            save(qq_path, fig_qq)
            println("    Saved: $qq_path")
            println("    Robust  costs (after removal): min=$(round(robust_costs[1], digits=3)), max=$(round(robust_costs[end], digits=3)), mean=$(round(mean(robust_costs), digits=3)), std=$(round(std(robust_costs), digits=3))")
            println("    Non-Rob costs (after removal): min=$(round(non_robust_costs[1], digits=3)), max=$(round(non_robust_costs[end], digits=3)), mean=$(round(mean(non_robust_costs), digits=3)), std=$(round(std(non_robust_costs), digits=3))")
        end
    end

    # Print template
    println("\n" * "="^80)
    println("Q-Q plots saved. Inspect them, then create your outlier config:")
    println("="^80)
    println()
    println("robust_outliers = Dict(")
    for group in groups
        for (nccw, _, _) in group.robust_variants
            nccw_str = nccw == floor(nccw) ? "$(Int(nccw)).0" : string(nccw)
            println("    $nccw_str => 0,")
        end
    end
    println(")")
    println("non_robust_outliers = 0")
    println()
    println("Then run:  analyze_nccw_groups(robust_outliers=robust_outliers, non_robust_outliers=non_robust_outliers)")

    return groups
end

# ============================================================================
# Pass 2: Full analysis with outlier removal
# ============================================================================

"""
    analyze_nccw_groups(;
        sweep_dir, analysis_base_dir, min_trials,
        robust_outliers = Dict{Float64, Int}(),
        non_robust_outliers = 0
    )

Pass 2: For each nccw variant, runs the full analysis pipeline (same as `visualize_sweep_rank`):
- KKT yarnball plots
- Q-Q plots (updated with outlier removal applied)
- Statistical significance report (Welch's t-test, Mann-Whitney U, Bootstrap)
- Trajectory summary (action comparisons, cost component plots)
- Defender cost comparison (instantaneous + cumulative)
- Defender yarnball comparison

`robust_outliers` maps nccw values to robust outlier counts (per-variant):
```julia
robust_outliers = Dict(
    10.0   => 4,
    100.0  => 4,
    200.0  => 3,
    300.0  => 1,
    500.0  => 0,
    1000.0 => 2,
    2000.0 => 2,
    3000.0 => 1,
    5000.0 => 1,
    10000.0 => 2,
)
```
`non_robust_outliers` is a single Int for the shared baseline (default 1).
Any nccw not in `robust_outliers` defaults to 0.
"""
function analyze_nccw_groups(;
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep"),
    analysis_base_dir::String = joinpath(@__DIR__, "..", "analysis"),
    min_trials::Int = 50,
    robust_outliers::Dict = Dict(
    10.0   => 4,
    100.0  => 4,
    200.0  => 3,
    300.0  => 1,
    500.0  => 0,
    1000.0 => 2,
    2000.0 => 2,
    3000.0 => 1,
    5000.0 => 1,
    10000.0 => 2,
),
    non_robust_outliers::Int = 1
)
    groups = find_nccw_groups(sweep_dir=sweep_dir, min_trials=min_trials)

    if isempty(groups)
        println("No parameter groups found with >= $min_trials trials.")
        return
    end

    println("="^80)
    println("PASS 2: Full NCCW Comparison Analysis")
    println("="^80)
    println("Found $(length(groups)) group(s). Non-robust outliers: $non_robust_outliers\n")

    for (gi, group) in enumerate(groups)
        println("─"^80)
        println("Group $gi: $(group.base_config)")
        println("  Non-robust baseline: $(group.non_robust_dir) ($(group.nr_trials) trials, removing $non_robust_outliers outliers)")
        println("  Robust variants ($(length(group.robust_variants))):")
        for (nccw, _, nt) in group.robust_variants
            r_out = get(robust_outliers, nccw, 0)
            println("    nccw=$(lpad(nccw, 8)) → $nt trials | robust outliers: $r_out")
        end
        println()

        # Load and process non-robust baseline ONCE per group
        println("  Loading non-robust baseline...")
        nr_solutions = load_dir_solutions(sweep_dir, group.non_robust_dir, "Non-Robust")
        nr_baseline = compute_nr_baseline(nr_solutions, non_robust_outliers)
        println("  Non-robust: $(length(nr_baseline.costs))/$(nr_baseline.n_total) trials ($(nr_baseline.n_removed) outliers removed)")
        println("  Non-robust mean=$(round(mean(nr_baseline.costs), digits=4)), std=$(round(std(nr_baseline.costs), digits=4))")

        group_output_dir = joinpath(analysis_base_dir, group.base_config)

        for (vi, (nccw, robust_dir, n_robust)) in enumerate(group.robust_variants)
            nccw_label = nccw == floor(nccw) ? string(Int(nccw)) : string(nccw)
            output_dir = joinpath(group_output_dir, "nccw_$nccw_label")

            r_out = get(robust_outliers, nccw, 0)

            println("\n  [$vi/$(length(group.robust_variants))] nccw = $nccw")
            println("    Robust dir:     $robust_dir ($n_robust trials, removing $r_out outliers)")
            println("    Output:         $output_dir")

            r_solutions = load_dir_solutions(sweep_dir, robust_dir, "Robust")
            println("    Loaded $(length(r_solutions)) robust trials")

            if isempty(r_solutions)
                println("    SKIPPING: no robust trials loaded")
                continue
            end

            generate_nccw_comparison_plots(
                r_solutions, nr_baseline, vi, output_dir;
                robust_outliers_to_remove = r_out
            )
            println("    Analysis complete.")
        end
    end

    # Generate cross-nccw comparison plot
    println("\nGenerating cross-nccw comparison plot...")
    generate_cross_nccw_cost_plot(
        sweep_dir=sweep_dir, analysis_base_dir=analysis_base_dir,
        min_trials=min_trials, robust_outliers=robust_outliers,
        non_robust_outliers=non_robust_outliers
    )

    println("\n" * "="^80)
    println("All analyses complete.")
    println("="^80)
end

# ============================================================================
# Core analysis (non-interactive generate_comparison_plots)
# ============================================================================

"""
    generate_nccw_comparison_plots(r_solutions, nr_baseline, rank, output_dir; kwargs...)

Non-interactive version of `generate_comparison_plots` from test_sweep.jl.

Takes robust solutions and a pre-computed non-robust baseline (from `compute_nr_baseline`).
The non-robust costs/entries are NOT recomputed — they are identical across all nccw variants.
"""
function generate_nccw_comparison_plots(
    r_solutions::Dict{String, Any},
    nr_baseline::NamedTuple,
    rank::Int,
    output_dir::String;
    robust_outliers_to_remove::Int = 0
)
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    # Clear Trackers
    TrajectoryAnalysis.clear_trajectory_tracker!()
    TrajectoryAnalysis.KKTErrorTracker.clear_rh_kkt_tracker!()

    # Populate trackers from robust solutions
    for (key_name, solution_data) in r_solutions
        gt_state_history = get(solution_data, "gt_state_history", nothing)
        solution_history = get(solution_data, "solution_history", nothing)
        params = get(solution_data, "params", nothing)

        if !isnothing(solution_history)
            TrajectoryAnalysis.extract_kkt_errors_from_history(solution_history, key_name, true)

            if !isnothing(gt_state_history) && !isnothing(params)
                try
                    entry = TrajectoryAnalysis.TrajectoryAnalysisEntry(
                        key_name, rank,
                        gt_state_history, [], solution_history, [], [], [],
                        params, true, "rank_$rank"
                    )
                    push!(TrajectoryAnalysis.TRAJECTORY_TRACKER.entries, entry)
                catch e
                    println("    Warning: Failed to create entry for $key_name: $e")
                end
            end
        end
    end

    # Populate trackers from non-robust baseline
    for (key_name, solution_data) in nr_baseline.solutions
        key_name in nr_baseline.kept_keys || continue
        solution_history = get(solution_data, "solution_history", nothing)
        gt_state_history = get(solution_data, "gt_state_history", nothing)
        params = get(solution_data, "params", nothing)

        if !isnothing(solution_history)
            TrajectoryAnalysis.extract_kkt_errors_from_history(solution_history, key_name, false)

            if !isnothing(gt_state_history) && !isnothing(params)
                try
                    entry = TrajectoryAnalysis.TrajectoryAnalysisEntry(
                        key_name, rank,
                        gt_state_history, [], solution_history, [], [], [],
                        params, false, "rank_$rank"
                    )
                    push!(TrajectoryAnalysis.TRAJECTORY_TRACKER.entries, entry)
                catch e
                    println("    Warning: Failed to create NR entry for $key_name: $e")
                end
            end
        end
    end

    # KKT plots
    TrajectoryAnalysis.KKTErrorTracker.create_kkt_yarnball_plots()
    if isfile("kkt_temporal_evolution.png")
        mv("kkt_temporal_evolution.png", joinpath(output_dir, "kkt.png"), force=true)
    end

    # Collect robust costs and apply outlier removal
    robust_cost_entries = Tuple{Float64, String}[]
    for (key_name, solution_data) in r_solutions
        cost = calculate_defender_total_cost(solution_data)
        push!(robust_cost_entries, (cost, key_name))
    end

    sort!(robust_cost_entries, by=x->x[1])
    n_r_remove = min(robust_outliers_to_remove, length(robust_cost_entries) - 2)
    filtered_r = n_r_remove > 0 ? robust_cost_entries[1:end-n_r_remove] : robust_cost_entries

    robust_costs = [e[1] for e in filtered_r]
    robust_kept_keys = Set(e[2] for e in filtered_r)

    # Non-robust costs are pre-computed (identical across all nccw variants)
    non_robust_costs = nr_baseline.costs

    # Build combined kept_keys and filter tracker
    all_kept_keys = union(robust_kept_keys, nr_baseline.kept_keys)
    filter!(e -> e.scenario_name in all_kept_keys, TrajectoryAnalysis.TRAJECTORY_TRACKER.entries)
    println("    After outlier removal: $(length(robust_costs)) robust, $(length(non_robust_costs)) non-robust")

    # Q-Q plot
    if length(robust_costs) > 1 && length(non_robust_costs) > 1
        fig_qq = Figure(size=(1200, 500))
        Label(fig_qq[0, :],
            text="Q-Q Plots (Outliers removed: R=$(n_r_remove), NR=$(nr_baseline.n_removed))",
            fontsize=16)

        ax_r = Axis(fig_qq[1, 1],
            title="Robust (n=$(length(robust_costs)))",
            xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")
        if std(robust_costs) > 0
            n_r = length(robust_costs)
            tq = [quantile(Normal(0, 1), (i - 0.5) / n_r) for i in 1:n_r]
            standardized = (sort(robust_costs) .- mean(robust_costs)) ./ std(robust_costs)
            scatter!(ax_r, tq, standardized, color=:blue, markersize=8)
            lines!(ax_r, [-3, 3], [-3, 3], color=:red, linestyle=:dash, linewidth=2)
        end

        ax_nr = Axis(fig_qq[1, 2],
            title="Non-Robust (n=$(length(non_robust_costs)))",
            xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")
        if std(non_robust_costs) > 0
            n_nr = length(non_robust_costs)
            tq = [quantile(Normal(0, 1), (i - 0.5) / n_nr) for i in 1:n_nr]
            standardized = (sort(non_robust_costs) .- mean(non_robust_costs)) ./ std(non_robust_costs)
            scatter!(ax_nr, tq, standardized, color=:red, markersize=8)
            lines!(ax_nr, [-3, 3], [-3, 3], color=:red, linestyle=:dash, linewidth=2)
        end

        save(joinpath(output_dir, "qq_plots.png"), fig_qq)
    end

    # Statistical significance report
    open(joinpath(output_dir, "significance_report.txt"), "w") do io
        println(io, "Statistical Significance Report (Defender Total Cost)")
        println(io, "===================================================\n")

        n_r = length(robust_costs)
        n_nr = length(non_robust_costs)

        if n_r > 1 && n_nr > 1
            mean_r = mean(robust_costs)
            std_r = std(robust_costs)
            mean_nr = mean(non_robust_costs)
            std_nr = std(non_robust_costs)

            println(io, "DESCRIPTIVE STATISTICS")
            println(io, "---------------------------------------------------")
            println(io, "Outliers removed: Robust=$(n_r_remove), Non-Robust=$(nr_baseline.n_removed)")
            println(io, "Robust (n=$n_r):     Mean = $(round(mean_r, digits=4)), Std = $(round(std_r, digits=4))")
            println(io, "Non-Robust (n=$n_nr): Mean = $(round(mean_nr, digits=4)), Std = $(round(std_nr, digits=4))")
            println(io, "Difference (Robust - Non-Robust): $(round(mean_r - mean_nr, digits=4))\n")

            # Welch's t-test
            println(io, "WELCH'S T-TEST")
            println(io, "---------------------------------------------------")
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
            println(io, "---------------------------------------------------")
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
            println(io, "---------------------------------------------------")

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
        end
    end
    println("    Saved significance report.")

    # Trajectory summary (action comparisons, cost component yarnball)
    TrajectoryAnalysis.get_trajectory_summary(directory=output_dir)

    # Defender cost comparison
    try
        robust_entries = TrajectoryAnalysis.TrajectoryAnalysisEntry[]
        non_robust_entries = TrajectoryAnalysis.TrajectoryAnalysisEntry[]

        for entry in TrajectoryAnalysis.TRAJECTORY_TRACKER.entries
            if entry.robust
                push!(robust_entries, entry)
            else
                push!(non_robust_entries, entry)
            end
        end

        if !isempty(robust_entries) && !isempty(non_robust_entries)
            try
                TrajectoryAnalysis.create_defender_yarnball_comparison(
                    robust_entries, non_robust_entries; output_dir=output_dir
                )
            catch e
                println("    Error creating defender yarnball: $e")
            end

            function get_defender_cost_series(entry)
                executed_costs = TrajectoryAnalysis.compute_executed_trajectory_costs(entry, false)
                costs = Float64[]
                for step_costs in executed_costs
                    if haskey(step_costs, :defender)
                        push!(costs, sum(values(step_costs[:defender])))
                    else
                        push!(costs, 0.0)
                    end
                end
                return costs
            end

            function compute_stats(data_list)
                isempty(data_list) && return (1:0, Float64[], Float64[])
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

            r_costs_all = [get_defender_cost_series(e) for e in robust_entries]
            nr_costs_all = [get_defender_cost_series(e) for e in non_robust_entries]

            fig = Figure(size=(1000, 500), fontsize=40)

            ax1 = Axis(fig[1, 1], title="Defender Instantaneous Cost",
                xlabel="Time Step", ylabel="Cost")
            if !isempty(r_costs_all)
                ts, means, stds = compute_stats(r_costs_all)
                if !isempty(ts)
                    band!(ax1, collect(ts), means .- stds, means .+ stds, color=(:blue, 0.2))
                    lines!(ax1, collect(ts), means, color=:blue, label="Robust", linewidth=2)
                end
            end
            if !isempty(nr_costs_all)
                ts, means, stds = compute_stats(nr_costs_all)
                if !isempty(ts)
                    band!(ax1, collect(ts), means .- stds, means .+ stds, color=(:red, 0.2))
                    lines!(ax1, collect(ts), means, color=:red, label="Non-Robust", linewidth=2)
                end
            end
            axislegend(ax1)

            ax2 = Axis(fig[1, 2], title="Defender Cumulative Cost",
                xlabel="Time Step", ylabel="Total Cost")
            if !isempty(r_costs_all)
                r_cumsum_all = [cumsum(c) for c in r_costs_all]
                ts, means, stds = compute_stats(r_cumsum_all)
                if !isempty(ts)
                    band!(ax2, collect(ts), means .- stds, means .+ stds, color=(:blue, 0.2))
                    lines!(ax2, collect(ts), means, color=:blue, label="Robust", linewidth=2)
                end
            end
            if !isempty(nr_costs_all)
                nr_cumsum_all = [cumsum(c) for c in nr_costs_all]
                ts, means, stds = compute_stats(nr_cumsum_all)
                if !isempty(ts)
                    band!(ax2, collect(ts), means .- stds, means .+ stds, color=(:red, 0.2))
                    lines!(ax2, collect(ts), means, color=:red, label="Non-Robust", linewidth=2)
                end
            end
            axislegend(ax2)

            save(joinpath(output_dir, "defender_cost_comparison.png"), fig)
            save(joinpath(output_dir, "defender_cost_comparison.pdf"), fig)
            println("    Saved defender cost comparison.")

            # Defender cost components comparison
            function get_defender_component_series(entry)
                executed_costs = TrajectoryAnalysis.compute_executed_trajectory_costs(entry, false)
                comp_series = Dict{Symbol, Vector{Float64}}()
                for step_costs in executed_costs
                    if haskey(step_costs, :defender)
                        for (comp, val) in pairs(step_costs[:defender])
                            series = get!(comp_series, comp, Float64[])
                            push!(series, val)
                        end
                    end
                end
                return comp_series
            end

            # Collect component series for all robust and non-robust entries
            r_comp_all = [get_defender_component_series(e) for e in robust_entries]
            nr_comp_all = [get_defender_component_series(e) for e in non_robust_entries]

            # Discover all component names
            all_comps = Set{Symbol}()
            for cs in vcat(r_comp_all, nr_comp_all)
                union!(all_comps, keys(cs))
            end
            comp_names = sort(collect(all_comps))

            if !isempty(comp_names)
                n_comps = length(comp_names)
                fig_comp = Figure(size=(500 * n_comps, 450), fontsize=16)

                for (ci, comp) in enumerate(comp_names)
                    ax_c = Axis(fig_comp[1, ci],
                        title=string(comp),
                        xlabel="Time Step", ylabel="Cost")

                    r_series = [get(cs, comp, Float64[]) for cs in r_comp_all]
                    filter!(!isempty, r_series)
                    if !isempty(r_series)
                        ts, means, stds = compute_stats(r_series)
                        if !isempty(ts)
                            band!(ax_c, collect(ts), means .- stds, means .+ stds, color=(:blue, 0.2))
                            lines!(ax_c, collect(ts), means, color=:blue, label="Robust", linewidth=2)
                        end
                    end

                    nr_series = [get(cs, comp, Float64[]) for cs in nr_comp_all]
                    filter!(!isempty, nr_series)
                    if !isempty(nr_series)
                        ts, means, stds = compute_stats(nr_series)
                        if !isempty(ts)
                            band!(ax_c, collect(ts), means .- stds, means .+ stds, color=(:red, 0.2))
                            lines!(ax_c, collect(ts), means, color=:red, label="Non-Robust", linewidth=2)
                        end
                    end

                    ci == n_comps && axislegend(ax_c)
                end

                save(joinpath(output_dir, "defender_cost_components.png"), fig_comp)
                save(joinpath(output_dir, "defender_cost_components.pdf"), fig_comp)
                println("    Saved defender cost components comparison.")
            end
        end
    catch e
        println("    Error generating defender cost plot: $e")
    end

    # Belief mean trajectory yarnball (per-nccw)
    try
        _generate_belief_trajectory_plot(r_solutions, nr_baseline, output_dir;
            robust_outliers_to_remove=robust_outliers_to_remove)
    catch e
        println("    Error generating belief trajectory plot: $e")
    end
end

"""
Helper: extract 2D belief mean trajectories for a single player from solution data.
`player_idx`: 1 = attacker, 2 = defender.
Belief indices: attacker sees beliefs (1,2), defender sees beliefs (3,4).
Within each pair: first = belief about attacker, second = belief about defender.
Returns (attacker_xy, defender_xy) vectors of (x,y) tuples.
"""
function _extract_player_belief_xy(sol_hist, player_idx::Int)
    is_player_indexed = isa(sol_hist, Dict) && haskey(sol_hist, 1) && isa(sol_hist[1], AbstractVector)

    belief_about_att = Tuple{Float64, Float64}[]
    belief_about_def = Tuple{Float64, Float64}[]

    # Belief indices per player
    att_belief_idx, def_belief_idx = player_idx == 1 ? (1, 2) : (3, 4)

    n_steps = if is_player_indexed
        haskey(sol_hist, player_idx) ? length(sol_hist[player_idx]) : 0
    else
        length(sol_hist)
    end

    for t in 1:n_steps
        player_sol = if is_player_indexed
            haskey(sol_hist, player_idx) && length(sol_hist[player_idx]) >= t ? sol_hist[player_idx][t] : nothing
        else
            haskey(sol_hist, t) && length(sol_hist[t]) >= player_idx ? sol_hist[t][player_idx] : nothing
        end

        isnothing(player_sol) && continue
        length(player_sol) < 2 && continue

        beliefs_traj, _ = player_sol
        isempty(beliefs_traj) && continue

        current_beliefs = beliefs_traj[1]
        if length(current_beliefs.beliefs) >= max(att_belief_idx, def_belief_idx)
            att_mean = current_beliefs.beliefs[att_belief_idx].belief_mean
            def_mean = current_beliefs.beliefs[def_belief_idx].belief_mean
            push!(belief_about_att, (att_mean[1], att_mean[2]))
            push!(belief_about_def, (def_mean[1], def_mean[2]))
        end
    end

    return (belief_about_att, belief_about_def)
end

"""
Generate a per-nccw belief mean trajectory yarnball plot.

Left panel: Defender's beliefs (about attacker + about itself).
Right panel: Attacker's beliefs (about attacker + about defender).
Robust = solid blue, Non-robust = dashed red.
"""
function _generate_belief_trajectory_plot(
    r_solutions::Dict{String, Any},
    nr_baseline::NamedTuple,
    output_dir::String;
    robust_outliers_to_remove::Int = 0,
    figsize::Tuple{Int,Int} = (1000, 500),
    fontsize::Int = 14,
    linewidth::Real = 0.8,
    alpha::Real = 0.25
)
    mkpath(output_dir)

    # Outlier filtering for robust solutions
    cost_key_pairs = Tuple{Float64, String}[]
    for (key_name, solution_data) in r_solutions
        cost = calculate_defender_total_cost(solution_data)
        push!(cost_key_pairs, (cost, key_name))
    end
    sort!(cost_key_pairs, by=x->x[1])
    n_remove = min(robust_outliers_to_remove, length(cost_key_pairs) - 2)
    if n_remove > 0
        cost_key_pairs = cost_key_pairs[1:end-n_remove]
    end
    robust_kept = Set(p[2] for p in cost_key_pairs)

    fig = Figure(size=figsize, fontsize=fontsize)

    ax_def = Axis(fig[1, 1],
        title="Defender's Beliefs",
        xlabel="x", ylabel="y",
        aspect=DataAspect())

    ax_att = Axis(fig[1, 2],
        title="Attacker's Beliefs",
        xlabel="x", ylabel="",
        aspect=DataAspect())

    # Plot robust solutions (solid)
    for (key_name, solution_data) in r_solutions
        key_name in robust_kept || continue
        sol_hist = get(solution_data, "solution_history", nothing)
        isnothing(sol_hist) && continue

        try
            # Defender's beliefs
            d_att, d_def = _extract_player_belief_xy(sol_hist, 2)
            if length(d_att) > 1
                lines!(ax_def, [p[1] for p in d_att], [p[2] for p in d_att],
                    color=(:blue, alpha), linewidth=linewidth)
            end
            if length(d_def) > 1
                lines!(ax_def, [p[1] for p in d_def], [p[2] for p in d_def],
                    color=(:dodgerblue, alpha), linewidth=linewidth, linestyle=:dot)
            end

            # Attacker's beliefs
            a_att, a_def = _extract_player_belief_xy(sol_hist, 1)
            if length(a_att) > 1
                lines!(ax_att, [p[1] for p in a_att], [p[2] for p in a_att],
                    color=(:blue, alpha), linewidth=linewidth)
            end
            if length(a_def) > 1
                lines!(ax_att, [p[1] for p in a_def], [p[2] for p in a_def],
                    color=(:dodgerblue, alpha), linewidth=linewidth, linestyle=:dot)
            end
        catch; end
    end

    # Plot non-robust solutions (dashed)
    for (key_name, solution_data) in nr_baseline.solutions
        key_name in nr_baseline.kept_keys || continue
        sol_hist = get(solution_data, "solution_history", nothing)
        isnothing(sol_hist) && continue

        try
            # Defender's beliefs
            d_att, d_def = _extract_player_belief_xy(sol_hist, 2)
            if length(d_att) > 1
                lines!(ax_def, [p[1] for p in d_att], [p[2] for p in d_att],
                    color=(:red, alpha), linewidth=linewidth, linestyle=:dash)
            end
            if length(d_def) > 1
                lines!(ax_def, [p[1] for p in d_def], [p[2] for p in d_def],
                    color=(:salmon, alpha), linewidth=linewidth, linestyle=:dashdot)
            end

            # Attacker's beliefs
            a_att, a_def = _extract_player_belief_xy(sol_hist, 1)
            if length(a_att) > 1
                lines!(ax_att, [p[1] for p in a_att], [p[2] for p in a_att],
                    color=(:red, alpha), linewidth=linewidth, linestyle=:dash)
            end
            if length(a_def) > 1
                lines!(ax_att, [p[1] for p in a_def], [p[2] for p in a_def],
                    color=(:salmon, alpha), linewidth=linewidth, linestyle=:dashdot)
            end
        catch; end
    end

    # Legend
    legend_elements = [
        LineElement(color=:blue, linestyle=:solid, linewidth=2),
        LineElement(color=:dodgerblue, linestyle=:dot, linewidth=2),
        LineElement(color=:red, linestyle=:dash, linewidth=2),
        LineElement(color=:salmon, linestyle=:dashdot, linewidth=2),
    ]
    legend_labels = [
        "Robust (about attacker)",
        "Robust (about defender)",
        "Non-Robust (about attacker)",
        "Non-Robust (about defender)",
    ]
    Legend(fig[2, :], legend_elements, legend_labels,
        orientation=:horizontal, nbanks=1, framevisible=false)

    save(joinpath(output_dir, "belief_trajectories.png"), fig)
    save(joinpath(output_dir, "belief_trajectories.pdf"), fig)
    println("    Saved belief trajectory yarnball.")
end

# ============================================================================
# Cross-nccw comparison plot
# ============================================================================

"""
Helper: compute mean and std time series from a list of cost vectors.
"""
function _compute_timeseries_stats(data_list)
    isempty(data_list) && return (1:0, Float64[], Float64[])
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

"""
Helper: extract defender cost time series from a solutions dict, applying outlier removal.
Returns a vector of per-trial cost vectors (one Float64[] per trial).
"""
function _extract_defender_cost_series(solutions::Dict{String, Any}, is_robust_filter::Bool;
    outliers_to_remove::Int = 0
)
    # First compute total costs per trial for outlier identification
    cost_key_pairs = Tuple{Float64, String}[]
    for (key_name, solution_data) in solutions
        is_robust = occursin("Robust", key_name) && !occursin("Non-Robust", key_name)
        is_robust == is_robust_filter || continue
        cost = calculate_defender_total_cost(solution_data)
        push!(cost_key_pairs, (cost, key_name))
    end

    # Remove highest-cost outliers
    sort!(cost_key_pairs, by=x->x[1])
    n_remove = min(outliers_to_remove, length(cost_key_pairs) - 2)
    if n_remove > 0
        cost_key_pairs = cost_key_pairs[1:end-n_remove]
    end
    kept_keys = Set(p[2] for p in cost_key_pairs)

    # Now extract per-timestep cost series for kept trials
    all_series = Vector{Float64}[]
    for (key_name, solution_data) in solutions
        key_name in kept_keys || continue

        gt_hist = get(solution_data, "gt_state_history", nothing)
        sol_hist = get(solution_data, "solution_history", nothing)
        params = get(solution_data, "params", nothing)
        (isnothing(gt_hist) || isnothing(sol_hist) || isnothing(params)) && continue

        try
            entry = TrajectoryAnalysis.TrajectoryAnalysisEntry(
                key_name, 0, gt_hist, [], sol_hist, [], [], [],
                params, is_robust_filter, ""
            )
            executed_costs = TrajectoryAnalysis.compute_executed_trajectory_costs(entry, false)
            costs = Float64[]
            for step_costs in executed_costs
                if haskey(step_costs, :defender)
                    push!(costs, sum(values(step_costs[:defender])))
                else
                    push!(costs, 0.0)
                end
            end
            push!(all_series, costs)
        catch
        end
    end

    return all_series
end

"""
Helper: extract per-component defender cost time series from a solutions dict.
Returns `Dict{Symbol, Vector{Vector{Float64}}}` where each key is a component name
and the value is a list of per-trial time series for that component.
Outlier removal is based on total defender cost (same as `_extract_defender_cost_series`).
"""
function _extract_defender_component_series(solutions::Dict{String, Any}, is_robust_filter::Bool;
    outliers_to_remove::Int = 0
)
    # Compute total costs for outlier ranking
    cost_key_pairs = Tuple{Float64, String}[]
    for (key_name, solution_data) in solutions
        is_robust = occursin("Robust", key_name) && !occursin("Non-Robust", key_name)
        is_robust == is_robust_filter || continue
        cost = calculate_defender_total_cost(solution_data)
        push!(cost_key_pairs, (cost, key_name))
    end

    sort!(cost_key_pairs, by=x->x[1])
    n_remove = min(outliers_to_remove, length(cost_key_pairs) - 2)
    if n_remove > 0
        cost_key_pairs = cost_key_pairs[1:end-n_remove]
    end
    kept_keys = Set(p[2] for p in cost_key_pairs)

    result = Dict{Symbol, Vector{Vector{Float64}}}()
    for (key_name, solution_data) in solutions
        key_name in kept_keys || continue

        gt_hist = get(solution_data, "gt_state_history", nothing)
        sol_hist = get(solution_data, "solution_history", nothing)
        params = get(solution_data, "params", nothing)
        (isnothing(gt_hist) || isnothing(sol_hist) || isnothing(params)) && continue

        try
            entry = TrajectoryAnalysis.TrajectoryAnalysisEntry(
                key_name, 0, gt_hist, [], sol_hist, [], [], [],
                params, is_robust_filter, ""
            )
            executed_costs = TrajectoryAnalysis.compute_executed_trajectory_costs(entry, false)
            comp_series = Dict{Symbol, Vector{Float64}}()
            for step_costs in executed_costs
                if haskey(step_costs, :defender)
                    for (comp, val) in pairs(step_costs[:defender])
                        series = get!(comp_series, comp, Float64[])
                        push!(series, val)
                    end
                end
            end
            for (comp, series) in comp_series
                trials = get!(result, comp, Vector{Float64}[])
                push!(trials, series)
            end
        catch
        end
    end

    return result
end

"""
    load_dir_solutions_lite(sweep_dir, subdir, prefix) -> Dict{String, Any}

Like `load_dir_solutions` but only loads keys needed for cost computation
(gt_state_history, solution_history, params), skipping observation_history.
"""
function load_dir_solutions_lite(sweep_dir::String, subdir::String, prefix::String)
    solutions = Dict{String, Any}()
    path = joinpath(sweep_dir, subdir)
    isdir(path) || return solutions
    for f in readdir(path)
        endswith(f, ".jld2") || continue
        try
            d = with_logger(NullLogger()) do
                jldopen(joinpath(path, f), "r") do file
                    Dict{String, Any}(
                        "gt_state_history" => file["gt_state_history"],
                        "solution_history" => file["solution_history"],
                        "params"           => file["params"]
                    )
                end
            end
            solutions["$prefix | $f"] = d
        catch e
            println("  Failed to load $f: $e")
        end
    end
    return solutions
end

"""
Helper: extract per-player cost time series from a solutions dict, applying outlier removal.
Returns Dict{Symbol, Vector{Vector{Float64}}} keyed by player name (:attacker, :defender).
Outlier removal is based on total defender cost (same ranking as before).
"""
function _extract_all_player_cost_series(solutions::Dict{String, Any}, is_robust_filter::Bool;
    outliers_to_remove::Int = 0
)
    # Compute total defender costs for outlier ranking
    cost_key_pairs = Tuple{Float64, String}[]
    for (key_name, solution_data) in solutions
        is_robust = occursin("Robust", key_name) && !occursin("Non-Robust", key_name)
        is_robust == is_robust_filter || continue
        cost = calculate_defender_total_cost(solution_data)
        push!(cost_key_pairs, (cost, key_name))
    end

    sort!(cost_key_pairs, by=x->x[1])
    n_remove = min(outliers_to_remove, length(cost_key_pairs) - 2)
    if n_remove > 0
        cost_key_pairs = cost_key_pairs[1:end-n_remove]
    end
    kept_keys = Set(p[2] for p in cost_key_pairs)

    player_names = [:attacker, :defender]
    all_series = Dict{Symbol, Vector{Vector{Float64}}}(p => Vector{Float64}[] for p in player_names)

    for (key_name, solution_data) in solutions
        key_name in kept_keys || continue

        gt_hist = get(solution_data, "gt_state_history", nothing)
        sol_hist = get(solution_data, "solution_history", nothing)
        params = get(solution_data, "params", nothing)
        (isnothing(gt_hist) || isnothing(sol_hist) || isnothing(params)) && continue

        try
            entry = TrajectoryAnalysis.TrajectoryAnalysisEntry(
                key_name, 0, gt_hist, [], sol_hist, [], [], [],
                params, is_robust_filter, ""
            )
            executed_costs = TrajectoryAnalysis.compute_executed_trajectory_costs(entry, false)

            for pname in player_names
                costs = Float64[]
                for step_costs in executed_costs
                    if haskey(step_costs, pname)
                        push!(costs, sum(values(step_costs[pname])))
                    else
                        push!(costs, 0.0)
                    end
                end
                push!(all_series[pname], costs)
            end
        catch
        end
    end

    return all_series
end

"""
    load_cross_nccw_data(; kwargs...) -> Vector{NamedTuple}

Load all data needed for cross-nccw plots. Returns a cacheable vector (one per group)
containing pre-computed cost time series for all players and nccw variants.

Call this once, then pass the result to `plot_cross_nccw_cost` repeatedly to iterate
on the visualization without reloading.

Usage:
    data = load_cross_nccw_data(robust_outliers=..., non_robust_outliers=...)
    plot_cross_nccw_cost(data)                    # default: defender only
    plot_cross_nccw_cost(data, players=[:attacker, :defender])  # both players
"""
function load_cross_nccw_data(;
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep"),
    min_trials::Int = 50,
    robust_outliers::Dict = Dict(
        10.0   => 4,
        100.0  => 4,
        200.0  => 3,
        300.0  => 1,
        500.0  => 0,
        1000.0 => 2,
        2000.0 => 2,
        3000.0 => 1,
        5000.0 => 1,
        10000.0 => 2,
        20000.0 => 2,
        30000.0 => 1,
    ),
    non_robust_outliers::Int = 1
)
    groups = find_nccw_groups(sweep_dir=sweep_dir, min_trials=min_trials)

    if isempty(groups)
        println("No parameter groups found with >= $min_trials trials.")
        return []
    end

    result = []

    for (gi, group) in enumerate(groups)
        println("Group $gi: $(group.base_config)")

        # Load non-robust baseline once per group
        println("  Loading non-robust baseline: $(group.non_robust_dir)")
        nr_solutions = load_dir_solutions_lite(sweep_dir, group.non_robust_dir, "Non-Robust")
        nr_player_series = _extract_all_player_cost_series(nr_solutions, false;
            outliers_to_remove=non_robust_outliers)
        println("  Non-robust: $(length(first(values(nr_player_series)))) trials after outlier removal")

        # Load each robust variant
        variants = []
        for (nccw, robust_dir, n_robust) in group.robust_variants
            r_out = get(robust_outliers, nccw, 0)
            println("  Loading nccw=$nccw: $robust_dir ($n_robust trials, removing $r_out)")

            r_solutions = load_dir_solutions_lite(sweep_dir, robust_dir, "Robust")
            r_player_series = _extract_all_player_cost_series(r_solutions, true;
                outliers_to_remove=r_out)
            n_kept = length(first(values(r_player_series)))
            println("    $n_kept trials after outlier removal")

            push!(variants, (nccw=nccw, player_series=r_player_series))
        end

        push!(result, (
            base_config=group.base_config,
            nccw_values=[v[1] for v in group.robust_variants],
            nr_player_series=nr_player_series,
            variants=variants
        ))
    end

    println("Done loading. Pass result to plot_cross_nccw_cost() to plot.")
    return result
end

"""
    plot_cross_nccw_cost(data;
        players = [:defender],
        analysis_base_dir, figsize, fontsize
    )

Generate cross-nccw violin plots from pre-loaded data (from `load_cross_nccw_data`).
Each nccw value gets a violin showing the distribution of total defender cost across trials,
with semi-transparent scatter points overlaid. Non-robust baseline shown as the last category.

`players` can be `:defender`, `:attacker`, or both `[:attacker, :defender]`.
"""
function plot_cross_nccw_cost(data;
    players::Union{Symbol, Vector{Symbol}} = [:defender],
    analysis_base_dir::String = joinpath(@__DIR__, "..", "analysis"),
    figsize::Tuple{Int,Int} = (2000, 800),
    fontsize::Int = 50,
    xticklabelrotation=pi/9,
    top_panel_nccws::Vector{Float64} = Float64[],
    top_panel_height_ratio::Real = 0.3,
    skip_nccws::Vector{Float64} = Float64[],
)
    players = players isa Symbol ? [players] : players
    broken = !isempty(top_panel_nccws)

    for group_data in data
        group_output_dir = joinpath(analysis_base_dir, group_data.base_config)
        mkpath(group_output_dir)

        # Filter out skipped nccws
        kept_variants = filter(v -> v.nccw ∉ skip_nccws, group_data.variants)
        nccw_values = [v.nccw for v in kept_variants]

        # Color scheme
        log_nccw = log10.(nccw_values)
        log_min, log_max = extrema(log_nccw)
        log_range = log_max - log_min
        if log_range == 0; log_range = 1.0; end

        function nccw_color(nccw_val)
            t = (log10(nccw_val) - log_min) / log_range
            h = 240.0 * (1.0 - t)
            s, l = 0.75, 0.45
            c = (1 - abs(2*l - 1)) * s
            x = c * (1 - abs((h/60) % 2 - 1))
            m = l - c/2
            r, g, b = if h < 60;      (c, x, 0.0)
                       elseif h < 120; (x, c, 0.0)
                       elseif h < 180; (0.0, c, x)
                       elseif h < 240; (0.0, x, c)
                       elseif h < 300; (x, 0.0, c)
                       else;           (c, 0.0, x)
                       end
            return RGBf(r+m, g+m, b+m)
        end

        for player in players
            n_variants = length(kept_variants)

            # Build tick labels: nccw values + "Baseline"
            tick_positions = collect(1:n_variants+1)
            tick_labels = String[]
            for vd in kept_variants
                nccw_str = vd.nccw == floor(vd.nccw) ? string(Int(vd.nccw)) : string(vd.nccw)
                push!(tick_labels, nccw_str)
            end
            push!(tick_labels, "Baseline")

            player_title = titlecase(string(player))
            fig = Figure(size=figsize, fontsize=fontsize,
                # fonts=(; regular="TeX Gyre Pagella"))
                fonts=(; regular="Palatino Roman"))

            local ax_top, ax_bot
            if broken
                ax_top = Axis(fig[1, 1],
                    xticks=(tick_positions, tick_labels),
                    xticklabelsvisible=false, xticksvisible=false,
                    yticklabelsize=fontsize,
                    xgridvisible=false, ygridvisible=false,
                    topspinevisible=false, rightspinevisible=false,
                    bottomspinevisible=false)
                ax_bot = Axis(fig[2, 1],
                    xlabel="Nature's Control Effort Cost (c)",
                    ylabel="Total Cost\nIncurred by Defender",
                    xticks=(tick_positions, tick_labels),
                    xticklabelrotation=xticklabelrotation,
                    xticklabelsize=fontsize,
                    yticklabelsize=fontsize,
                    xgridvisible=false, ygridvisible=false,
                    topspinevisible=false, rightspinevisible=false)
                rowsize!(fig.layout, 1, Auto(top_panel_height_ratio))
                rowgap!(fig.layout, 8)
                linkxaxes!(ax_top, ax_bot)
            else
                ax_top = nothing
                ax_bot = Axis(fig[1, 1],
                    xlabel="Nature's Control Effort Cost (c)", ylabel="Total Cost\nIncurred by Defender",
                    xticks=(tick_positions, tick_labels),
                    xticklabelrotation=xticklabelrotation,
                    xticklabelsize=fontsize,
                    yticklabelsize=fontsize,
                    xgridvisible=false, ygridvisible=false,
                    topspinevisible=false, rightspinevisible=false)
            end

            pick_ax(nccw_val) = (broken && nccw_val in top_panel_nccws) ? ax_top : ax_bot

            # Plot each nccw variant as a violin + boxplot quartile bars + jittered scatter
            for (i, vd) in enumerate(kept_variants)
                r_series = vd.player_series[player]
                r_totals = [sum(s) for s in r_series]
                isempty(r_totals) && continue

                ax = pick_ax(vd.nccw)
                col = nccw_color(vd.nccw)
                violin!(ax, fill(i, length(r_totals)), r_totals, color=(col, 0.6))

                # Mean bar
                bar_hw = 0.15
                m = mean(r_totals)
                lines!(ax, [i - bar_hw, i + bar_hw], [m, m], color=:black, linewidth=2)

                jitter = randn(length(r_totals)) .* 0.06
                scatter!(ax, fill(i, length(r_totals)) .+ jitter, r_totals,
                    color=(col, 0.4), markersize=5)
            end

            # Non-robust baseline (always on bottom panel)
            nr_series = group_data.nr_player_series[player]
            nr_totals = [sum(s) for s in nr_series]
            nr_pos = n_variants + 1

            if !isempty(nr_totals)
                violin!(ax_bot, fill(nr_pos, length(nr_totals)), nr_totals, color=(:gray, 0.6))

                # Mean bar for baseline
                bar_hw = 0.15
                m = mean(nr_totals)
                lines!(ax_bot, [nr_pos - bar_hw, nr_pos + bar_hw], [m, m], color=:black, linewidth=2)

                jitter = randn(length(nr_totals)) .* 0.06
                scatter!(ax_bot, fill(nr_pos, length(nr_totals)) .+ jitter, nr_totals,
                    color=(:black, 0.4), markersize=5)

                # Horizontal dashed line at baseline mean
                hlines!(ax_bot, [mean(nr_totals)], color=:black, linestyle=:dash, linewidth=1.5,
                    label="Baseline mean")
            end

            player_str = string(player)
            out_png = joinpath(group_output_dir, "cross_nccw_$(player_str)_cost.png")
            out_pdf = joinpath(group_output_dir, "cross_nccw_$(player_str)_cost.pdf")
            save(out_png, fig)
            save(out_pdf, fig)
            println("  Saved: $out_png")
            println("  Saved: $out_pdf")
        end

        # Statistical summary table across all nccw variants
        for player in players
            nr_series = group_data.nr_player_series[player]
            nr_totals = [sum(s) for s in nr_series]
            nr_mean = mean(nr_totals)
            nr_std = std(nr_totals)

            player_str = string(player)
            report_path = joinpath(group_output_dir, "cross_nccw_$(player_str)_summary.txt")
            open(report_path, "w") do io
                println(io, "Cross-NCCW $(titlecase(player_str)) Cost Summary")
                println(io, "="^65)
                println(io, "")
                println(io, "Non-Robust baseline: mean=$(round(nr_mean, digits=4)), std=$(round(nr_std, digits=4)), n=$(length(nr_totals))")
                println(io, "")
                println(io, rpad("NCCW", 10) *
                            rpad("Mean", 12) *
                            rpad("Std", 12) *
                            rpad("n", 6) *
                            rpad("Mean Impr%", 14) *
                            rpad("Std Decr%", 14))
                println(io, "-"^65)

                for vd in kept_variants
                    r_series = vd.player_series[player]
                    r_totals = [sum(s) for s in r_series]
                    r_mean = mean(r_totals)
                    r_std = std(r_totals)

                    mean_impr = (nr_mean - r_mean) / nr_mean * 100
                    std_decr = (nr_std - r_std) / nr_std * 100

                    nccw_str = vd.nccw == floor(vd.nccw) ? string(Int(vd.nccw)) : string(vd.nccw)
                    println(io,
                        rpad(nccw_str, 10) *
                        rpad(round(r_mean, digits=4), 12) *
                        rpad(round(r_std, digits=4), 12) *
                        rpad(length(r_totals), 6) *
                        rpad("$(round(mean_impr, digits=2))%", 14) *
                        rpad("$(round(std_decr, digits=2))%", 14)
                    )
                end
            end
            println("  Saved: $report_path")
        end
    end
end

"""
    generate_cross_nccw_cost_plot(;
        sweep_dir, analysis_base_dir, min_trials,
        robust_outliers, non_robust_outliers,
        zoom_padding = 0.15
    )

Generate a combined plot showing defender cost across all nccw variants on the same axes.
Each nccw value gets a color from a sequential colormap (cool→warm as nccw increases,
i.e. stronger→weaker Nature). The non-robust baseline is shown as a dashed black line.

Two panels: instantaneous cost (left) and cumulative cost (right).

The y-axis is zoomed to the range of the mean curves ± `zoom_padding` fraction of that range,
so differences between nccw values are clearly visible.

Saves to `analysis_base_dir/<base_config>/cross_nccw_defender_cost.{png,pdf}`.
"""
function generate_cross_nccw_cost_plot(;
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep"),
    analysis_base_dir::String = joinpath(@__DIR__, "..", "analysis"),
    min_trials::Int = 50,
    robust_outliers::Dict = Dict{Float64, Int}(),
    non_robust_outliers::Int = 0,
    zoom_padding::Float64 = 0.15
)
    groups = find_nccw_groups(sweep_dir=sweep_dir, min_trials=min_trials)

    if isempty(groups)
        println("No parameter groups found.")
        return
    end

    for (gi, group) in enumerate(groups)
        println("Group $gi: $(group.base_config)")
        group_output_dir = joinpath(analysis_base_dir, group.base_config)
        mkpath(group_output_dir)

        nccw_values = [v[1] for v in group.robust_variants]

        # Color scheme: use log-scaled nccw for color mapping
        # Low nccw (strong Nature) = cool/blue, high nccw (weak Nature) = warm/red
        log_nccw = log10.(nccw_values)
        log_min, log_max = extrema(log_nccw)
        log_range = log_max - log_min
        if log_range == 0; log_range = 1.0; end

        # Generate colors using a blue→purple→red palette via HSL interpolation
        function nccw_color(nccw_val)
            t = (log10(nccw_val) - log_min) / log_range  # 0 = lowest nccw, 1 = highest
            # Hue: 240° (blue) → 0° (red), Saturation: 0.75, Lightness: 0.45
            h = 240.0 * (1.0 - t)
            # Convert HSL to RGB
            s, l = 0.75, 0.45
            c = (1 - abs(2*l - 1)) * s
            x = c * (1 - abs((h/60) % 2 - 1))
            m = l - c/2
            r, g, b = if h < 60;      (c, x, 0.0)
                       elseif h < 120; (x, c, 0.0)
                       elseif h < 180; (0.0, c, x)
                       elseif h < 240; (0.0, x, c)
                       elseif h < 300; (x, 0.0, c)
                       else;           (c, 0.0, x)
                       end
            return RGBf(r+m, g+m, b+m)
        end

        # Load non-robust baseline once
        println("  Loading non-robust baseline: $(group.non_robust_dir)")
        nr_solutions_only = load_dir_solutions(sweep_dir, group.non_robust_dir, "Non-Robust")

        nr_series = _extract_defender_cost_series(nr_solutions_only, false;
            outliers_to_remove=non_robust_outliers)
        nr_ts, nr_means, nr_stds = _compute_timeseries_stats(nr_series)
        nr_cum_ts, nr_cum_means, nr_cum_stds = _compute_timeseries_stats([cumsum(c) for c in nr_series])

        # Load each robust variant
        all_inst_means = Vector{Float64}[]  # for computing zoom limits
        all_cum_means = Vector{Float64}[]
        variant_data = []

        for (nccw, robust_dir, _) in group.robust_variants
            r_out = get(robust_outliers, nccw, 0)
            println("  Loading nccw=$nccw: $robust_dir")

            r_solutions = load_dir_solutions(sweep_dir, robust_dir, "Robust")

            r_series = _extract_defender_cost_series(r_solutions, true;
                outliers_to_remove=r_out)
            inst_ts, inst_means, inst_stds = _compute_timeseries_stats(r_series)
            cum_ts, cum_means, cum_stds = _compute_timeseries_stats([cumsum(c) for c in r_series])

            push!(variant_data, (
                nccw=nccw, color=nccw_color(nccw),
                inst_ts=inst_ts, inst_means=inst_means, inst_stds=inst_stds,
                cum_ts=cum_ts, cum_means=cum_means, cum_stds=cum_stds
            ))

            !isempty(inst_means) && push!(all_inst_means, inst_means)
            !isempty(cum_means) && push!(all_cum_means, cum_means)
        end

        # Include non-robust in zoom calculation
        !isempty(nr_means) && push!(all_inst_means, nr_means)
        !isempty(nr_cum_means) && push!(all_cum_means, nr_cum_means)

        # Compute zoom limits from means only
        function compute_zoom_limits(all_means_list, padding)
            isempty(all_means_list) && return (nothing, nothing)
            all_vals = vcat(all_means_list...)
            ymin, ymax = extrema(all_vals)
            span = ymax - ymin
            if span < 1e-10; span = abs(ymax) * 0.1; end
            return (ymin - padding * span, ymax + padding * span)
        end

        inst_ylims = compute_zoom_limits(all_inst_means, zoom_padding)
        cum_ylims = compute_zoom_limits(all_cum_means, zoom_padding)

        # Create figure
        fig = Figure(size=(1400, 600), fontsize=14)

        ax1 = Axis(fig[1, 1],
            title="Defender Instantaneous Cost", ylabel="Cost")
        ax2 = Axis(fig[1, 2],
            title="Defender Cumulative Cost",
            xlabel="Time Step", ylabel="Total Cost")

        # Plot each robust variant
        for vd in variant_data
            if !isempty(vd.inst_ts)
                nccw_int = vd.nccw == floor(vd.nccw) ? string(Int(vd.nccw)) : string(vd.nccw)
                band!(ax1, collect(vd.inst_ts),
                    vd.inst_means .- vd.inst_stds, vd.inst_means .+ vd.inst_stds,
                    color=(vd.color, 0.1))
                lines!(ax1, collect(vd.inst_ts), vd.inst_means,
                    color=vd.color, linewidth=1.5, label="nccw=$nccw_int")
            end
            if !isempty(vd.cum_ts)
                band!(ax2, collect(vd.cum_ts),
                    vd.cum_means .- vd.cum_stds, vd.cum_means .+ vd.cum_stds,
                    color=(vd.color, 0.1))
                lines!(ax2, collect(vd.cum_ts), vd.cum_means,
                    color=vd.color, linewidth=1.5)
            end
        end

        # Plot non-robust baseline
        if !isempty(nr_ts)
            band!(ax1, collect(nr_ts),
                nr_means .- nr_stds, nr_means .+ nr_stds,
                color=(:black, 0.08))
            lines!(ax1, collect(nr_ts), nr_means,
                color=:black, linewidth=2, linestyle=:dash, label="Non-Robust")
        end
        if !isempty(nr_cum_ts)
            band!(ax2, collect(nr_cum_ts),
                nr_cum_means .- nr_cum_stds, nr_cum_means .+ nr_cum_stds,
                color=(:black, 0.08))
            lines!(ax2, collect(nr_cum_ts), nr_cum_means,
                color=:black, linewidth=2, linestyle=:dash)
        end

        # Apply zoom
        if !isnothing(inst_ylims[1])
            ylims!(ax1, inst_ylims...)
        end
        if !isnothing(cum_ylims[1])
            ylims!(ax2, cum_ylims...)
        end

        # Legend
        Legend(fig[2, :], ax1, orientation=:horizontal, nbanks=2, framevisible=false)

        out_png = joinpath(group_output_dir, "cross_nccw_defender_cost.png")
        out_pdf = joinpath(group_output_dir, "cross_nccw_defender_cost.pdf")
        save(out_png, fig)
        save(out_pdf, fig)
        println("  Saved: $out_png")
        println("  Saved: $out_pdf")
    end
end

# ============================================================================
# Cross-nccw cost component comparison plot
# ============================================================================

"""
    load_cross_nccw_component_data(; sweep_dir, min_trials, robust_outliers, non_robust_outliers)

Load per-component defender cost time series for all nccw variants. Returns a cacheable
vector (one per group) that can be passed to `plot_cross_nccw_components` repeatedly.

Usage:
    comp_data = load_cross_nccw_component_data()
    plot_cross_nccw_components(comp_data, components=[:steal_prob, :shot_prob])
    plot_cross_nccw_components(comp_data, components=[:control_effort], nccw_values=[100.0, 1000.0])
"""
function load_cross_nccw_component_data(;
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep"),
    min_trials::Int = 50,
    robust_outliers::Dict = Dict(
        10.0   => 4,
        100.0  => 4,
        200.0  => 3,
        300.0  => 1,
        500.0  => 0,
        1000.0 => 2,
        2000.0 => 2,
        3000.0 => 1,
        5000.0 => 1,
        10000.0 => 2,
        20000.0 => 2,
        30000.0 => 1,
    ),
    non_robust_outliers::Int = 1
)
    groups = find_nccw_groups(sweep_dir=sweep_dir, min_trials=min_trials)

    if isempty(groups)
        println("No parameter groups found with >= $min_trials trials.")
        return []
    end

    result = []

    for (gi, group) in enumerate(groups)
        println("Group $gi: $(group.base_config)")

        # Load non-robust baseline
        println("  Loading non-robust baseline: $(group.non_robust_dir)")
        nr_solutions = load_dir_solutions(sweep_dir, group.non_robust_dir, "Non-Robust")
        nr_comp = _extract_defender_component_series(nr_solutions, false;
            outliers_to_remove=non_robust_outliers)
        println("  Non-robust: loaded $(length(first(values(nr_comp)))) trials after outlier removal")

        # Load each robust variant
        variants = []
        for (nccw, robust_dir, n_robust) in group.robust_variants
            r_out = get(robust_outliers, nccw, 0)
            println("  Loading nccw=$nccw: $robust_dir ($n_robust trials, removing $r_out)")

            r_solutions = load_dir_solutions(sweep_dir, robust_dir, "Robust")
            r_comp = _extract_defender_component_series(r_solutions, true;
                outliers_to_remove=r_out)
            n_kept = isempty(r_comp) ? 0 : length(first(values(r_comp)))
            println("    $n_kept trials after outlier removal")

            push!(variants, (nccw=nccw, comp_data=r_comp))
        end

        push!(result, (
            base_config=group.base_config,
            nccw_values=[v[1] for v in group.robust_variants],
            nr_comp_data=nr_comp,
            variants=variants
        ))
    end

    println("Done loading. Pass result to plot_cross_nccw_components() to plot.")
    return result
end

"""
    plot_cross_nccw_components(data;
        components, nccw_values,
        analysis_base_dir, figsize, fontsize, linewidth, cumulative
    )

Plot defender cost components across nccw values from pre-loaded data
(from `load_cross_nccw_component_data`).

Violin plot with cost components on the x-axis and different nccw values as colors.
Each component gets a group of side-by-side violins (one per nccw value + non-robust baseline).
Colors match `plot_cross_nccw_cost`. Total cost per trial is the sum across all timesteps.

Arguments:
- `components`: Which cost components to plot (e.g. `[:steal_prob, :shot_prob, :control_effort]`).
  Available: `steal_prob`, `shot_prob`, `control_effort`, `bounds`.
- `nccw_values`: Subset of nccw values to include (default: all available).
"""
function plot_cross_nccw_components(data;
    components::Vector{Symbol},
    nccw_values::Union{Vector{Float64}, Nothing} = nothing,
    analysis_base_dir::String = joinpath(@__DIR__, "..", "analysis"),
    figsize::Tuple{Int,Int} = (800, 500),
    fontsize::Int = 20
)
    for group_data in data
        group_output_dir = joinpath(analysis_base_dir, group_data.base_config)
        mkpath(group_output_dir)

        all_nccw = group_data.nccw_values

        # Color scheme from ALL nccw values in group (matches violin plot)
        log_all = log10.(all_nccw)
        log_min, log_max = extrema(log_all)
        log_range = log_max - log_min
        if log_range == 0; log_range = 1.0; end

        function nccw_color(nccw_val)
            t = (log10(nccw_val) - log_min) / log_range
            h = 240.0 * (1.0 - t)
            s, l = 0.75, 0.45
            c = (1 - abs(2*l - 1)) * s
            x = c * (1 - abs((h/60) % 2 - 1))
            m = l - c/2
            r, g, b = if h < 60;      (c, x, 0.0)
                       elseif h < 120; (x, c, 0.0)
                       elseif h < 180; (0.0, c, x)
                       elseif h < 240; (0.0, x, c)
                       elseif h < 300; (x, 0.0, c)
                       else;           (c, 0.0, x)
                       end
            return RGBf(r+m, g+m, b+m)
        end

        # Filter variants to requested nccw values
        selected_variants = if isnothing(nccw_values)
            group_data.variants
        else
            nccw_set = Set(nccw_values)
            filter(v -> v.nccw in nccw_set, group_data.variants)
        end

        if isempty(selected_variants)
            println("  No matching nccw values. Available: $(all_nccw)")
            continue
        end

        n_comps = length(components)
        n_variants = length(selected_variants)
        n_positions = n_variants + 1  # +1 for baseline

        fig = Figure(size=figsize, fontsize=fontsize)

        # Build shared tick labels: nccw values + "Baseline"
        tick_positions = collect(1:n_positions)
        tick_labels = String[]
        for vd in selected_variants
            nccw_str = vd.nccw == floor(vd.nccw) ? string(Int(vd.nccw)) : string(vd.nccw)
            push!(tick_labels, nccw_str)
        end
        push!(tick_labels, "Baseline")

        legend_entries = []

        for (ci, comp) in enumerate(components)
            ax = Axis(fig[1, ci],
                title=string(comp),
                xlabel=ci == div(n_comps, 2) + 1 ? "Nature's Control Effort Cost" : "",
                ylabel=ci == 1 ? "Cost Incurred by Defender" : "",
                xticks=(tick_positions, tick_labels),
                xticklabelrotation=pi/4,
                xgridvisible=false, ygridvisible=false)

            mean_bars = Tuple{Float64, Float64}[]  # (position, mean)

            # Layer 1: violins
            for (vi, vd) in enumerate(selected_variants)
                col = nccw_color(vd.nccw)
                if haskey(vd.comp_data, comp)
                    totals = [sum(s) for s in vd.comp_data[comp]]
                    if !isempty(totals)
                        violin!(ax, fill(vi, length(totals)), totals, color=(col, 0.6))
                    end
                end
            end
            nr_pos = n_positions
            if haskey(group_data.nr_comp_data, comp)
                nr_totals = [sum(s) for s in group_data.nr_comp_data[comp]]
                if !isempty(nr_totals)
                    violin!(ax, fill(nr_pos, length(nr_totals)), nr_totals, color=(:gray, 0.6))
                end
            end

            # Layer 2: scatter
            for (vi, vd) in enumerate(selected_variants)
                col = nccw_color(vd.nccw)
                if haskey(vd.comp_data, comp)
                    totals = [sum(s) for s in vd.comp_data[comp]]
                    if !isempty(totals)
                        jitter = randn(length(totals)) .* 0.06
                        scatter!(ax, fill(vi, length(totals)) .+ jitter, totals,
                            color=(col, 0.4), markersize=5)
                        push!(mean_bars, (Float64(vi), mean(totals)))
                    end
                end
            end
            if haskey(group_data.nr_comp_data, comp)
                nr_totals = [sum(s) for s in group_data.nr_comp_data[comp]]
                if !isempty(nr_totals)
                    jitter = randn(length(nr_totals)) .* 0.06
                    scatter!(ax, fill(nr_pos, length(nr_totals)) .+ jitter, nr_totals,
                        color=(:black, 0.4), markersize=5)
                    push!(mean_bars, (Float64(nr_pos), mean(nr_totals)))
                end
            end

            # Layer 3: mean bars (on top)
            bar_hw = 0.15
            for (pos, m) in mean_bars
                lines!(ax, [pos - bar_hw, pos + bar_hw], [m, m], color=:black, linewidth=2)
            end
        end

        out_png = joinpath(group_output_dir, "cross_nccw_components.png")
        out_pdf = joinpath(group_output_dir, "cross_nccw_components.pdf")
        save(out_png, fig)
        save(out_pdf, fig)
        println("  Saved: $out_png")
        println("  Saved: $out_pdf")
    end
end

# ============================================================================
# Belief mean trajectory yarnball plot
# ============================================================================

"""
Helper: extract 2D belief mean trajectories from a solutions dict.
Returns a vector of named tuples, each with `attacker_xy` and `defender_xy`
(each a Vector of (x,y) tuples representing the defender's belief about that player).
Outlier removal is based on total defender cost.
"""
function _extract_belief_trajectories(solutions::Dict{String, Any}, is_robust_filter::Bool;
    outliers_to_remove::Int = 0
)
    # Outlier ranking by total defender cost
    cost_key_pairs = Tuple{Float64, String}[]
    for (key_name, solution_data) in solutions
        is_robust = occursin("Robust", key_name) && !occursin("Non-Robust", key_name)
        is_robust == is_robust_filter || continue
        cost = calculate_defender_total_cost(solution_data)
        push!(cost_key_pairs, (cost, key_name))
    end

    sort!(cost_key_pairs, by=x->x[1])
    n_remove = min(outliers_to_remove, length(cost_key_pairs) - 2)
    if n_remove > 0
        cost_key_pairs = cost_key_pairs[1:end-n_remove]
    end
    kept_keys = Set(p[2] for p in cost_key_pairs)

    trajectories = []
    for (key_name, solution_data) in solutions
        key_name in kept_keys || continue

        sol_hist = get(solution_data, "solution_history", nothing)
        isnothing(sol_hist) && continue

        is_player_indexed = isa(sol_hist, Dict) && haskey(sol_hist, 1) && isa(sol_hist[1], AbstractVector)

        attacker_xy = Tuple{Float64, Float64}[]
        defender_xy = Tuple{Float64, Float64}[]

        try
            # Determine number of timesteps
            n_steps = if is_player_indexed
                haskey(sol_hist, 2) ? length(sol_hist[2]) : 0
            else
                length(sol_hist)
            end

            for t in 1:n_steps
                # Get defender's solution at time t
                defender_sol = if is_player_indexed
                    haskey(sol_hist, 2) && length(sol_hist[2]) >= t ? sol_hist[2][t] : nothing
                else
                    haskey(sol_hist, t) && length(sol_hist[t]) >= 2 ? sol_hist[t][2] : nothing
                end

                isnothing(defender_sol) && continue
                length(defender_sol) < 2 && continue

                beliefs_traj, _ = defender_sol
                isempty(beliefs_traj) && continue

                current_beliefs = beliefs_traj[1]
                # Defender's belief indices: 3 = about attacker, 4 = about defender
                if length(current_beliefs.beliefs) >= 4
                    att_mean = current_beliefs.beliefs[3].belief_mean
                    def_mean = current_beliefs.beliefs[4].belief_mean
                    push!(attacker_xy, (att_mean[1], att_mean[2]))
                    push!(defender_xy, (def_mean[1], def_mean[2]))
                end
            end

            if !isempty(attacker_xy)
                push!(trajectories, (attacker_xy=attacker_xy, defender_xy=defender_xy))
            end
        catch
        end
    end

    return trajectories
end

"""
    load_cross_nccw_belief_data(; sweep_dir, min_trials, robust_outliers, non_robust_outliers)

Load belief mean trajectories for all nccw variants. Returns cacheable data
that can be passed to `plot_cross_nccw_belief_trajectories`.

Usage:
    belief_data = load_cross_nccw_belief_data()
    plot_cross_nccw_belief_trajectories(belief_data)
    plot_cross_nccw_belief_trajectories(belief_data, nccw_values=[100.0, 1000.0])
"""
function load_cross_nccw_belief_data(;
    sweep_dir::String = joinpath(@__DIR__, "..", "outputs", "sweep"),
    min_trials::Int = 50,
    robust_outliers::Dict = Dict(
        10.0   => 4,
        100.0  => 4,
        200.0  => 3,
        300.0  => 1,
        500.0  => 0,
        1000.0 => 2,
        2000.0 => 2,
        3000.0 => 1,
        5000.0 => 1,
        10000.0 => 2,
    ),
    non_robust_outliers::Int = 1
)
    groups = find_nccw_groups(sweep_dir=sweep_dir, min_trials=min_trials)

    if isempty(groups)
        println("No parameter groups found with >= $min_trials trials.")
        return []
    end

    result = []

    for (gi, group) in enumerate(groups)
        println("Group $gi: $(group.base_config)")

        # Load non-robust baseline
        println("  Loading non-robust baseline: $(group.non_robust_dir)")
        nr_solutions = load_dir_solutions(sweep_dir, group.non_robust_dir, "Non-Robust")
        nr_trajs = _extract_belief_trajectories(nr_solutions, false;
            outliers_to_remove=non_robust_outliers)
        println("  Non-robust: $(length(nr_trajs)) trajectories after outlier removal")

        # Load each robust variant
        variants = []
        for (nccw, robust_dir, n_robust) in group.robust_variants
            r_out = get(robust_outliers, nccw, 0)
            println("  Loading nccw=$nccw: $robust_dir ($n_robust trials, removing $r_out)")

            r_solutions = load_dir_solutions(sweep_dir, robust_dir, "Robust")
            r_trajs = _extract_belief_trajectories(r_solutions, true;
                outliers_to_remove=r_out)
            println("    $(length(r_trajs)) trajectories after outlier removal")

            push!(variants, (nccw=nccw, trajectories=r_trajs))
        end

        push!(result, (
            base_config=group.base_config,
            nccw_values=[v[1] for v in group.robust_variants],
            nr_trajectories=nr_trajs,
            variants=variants
        ))
    end

    println("Done loading. Pass result to plot_cross_nccw_belief_trajectories() to plot.")
    return result
end

"""
    plot_cross_nccw_belief_trajectories(data;
        nccw_values, analysis_base_dir, figsize, fontsize, linewidth, alpha
    )

Plot 2D belief mean trajectories (yarnball) across nccw values.

Two panels: left = defender's belief about the attacker, right = defender's belief about itself.
Robust trajectories are solid lines colored by nccw (same colors as the violin plot).
Non-robust trajectories are dashed gray lines.
"""
function plot_cross_nccw_belief_trajectories(data;
    nccw_values::Union{Vector{Float64}, Nothing} = nothing,
    analysis_base_dir::String = joinpath(@__DIR__, "..", "analysis"),
    figsize::Tuple{Int,Int} = (800, 500),
    fontsize::Int = 20,
    linewidth::Real = 1.0,
    alpha::Real = 0.3
)
    for group_data in data
        group_output_dir = joinpath(analysis_base_dir, group_data.base_config)
        mkpath(group_output_dir)

        all_nccw = group_data.nccw_values

        # Color scheme from ALL nccw values (matches violin plot)
        log_all = log10.(all_nccw)
        log_min, log_max = extrema(log_all)
        log_range = log_max - log_min
        if log_range == 0; log_range = 1.0; end

        function nccw_color(nccw_val)
            t = (log10(nccw_val) - log_min) / log_range
            h = 240.0 * (1.0 - t)
            s, l = 0.75, 0.45
            c = (1 - abs(2*l - 1)) * s
            x = c * (1 - abs((h/60) % 2 - 1))
            m = l - c/2
            r, g, b = if h < 60;      (c, x, 0.0)
                       elseif h < 120; (x, c, 0.0)
                       elseif h < 180; (0.0, c, x)
                       elseif h < 240; (0.0, x, c)
                       elseif h < 300; (x, 0.0, c)
                       else;           (c, 0.0, x)
                       end
            return RGBf(r+m, g+m, b+m)
        end

        # Filter variants
        selected_variants = if isnothing(nccw_values)
            group_data.variants
        else
            nccw_set = Set(nccw_values)
            filter(v -> v.nccw in nccw_set, group_data.variants)
        end

        if isempty(selected_variants)
            println("  No matching nccw values. Available: $(all_nccw)")
            continue
        end

        fig = Figure(size=figsize, fontsize=fontsize)

        ax_att = Axis(fig[1, 1],
            title="Belief about Attacker",
            xlabel="x", ylabel="y",
            aspect=DataAspect())

        ax_def = Axis(fig[1, 2],
            title="Belief about Defender",
            xlabel="x", ylabel="",
            aspect=DataAspect())

        # Plot non-robust baseline (dashed gray)
        for traj in group_data.nr_trajectories
            if length(traj.attacker_xy) > 1
                xs = [p[1] for p in traj.attacker_xy]
                ys = [p[2] for p in traj.attacker_xy]
                lines!(ax_att, xs, ys, color=(:gray, alpha), linewidth=linewidth, linestyle=:dash)
            end
            if length(traj.defender_xy) > 1
                xs = [p[1] for p in traj.defender_xy]
                ys = [p[2] for p in traj.defender_xy]
                lines!(ax_def, xs, ys, color=(:gray, alpha), linewidth=linewidth, linestyle=:dash)
            end
        end

        # Plot robust variants (solid, colored by nccw)
        for vd in selected_variants
            col = nccw_color(vd.nccw)
            for traj in vd.trajectories
                if length(traj.attacker_xy) > 1
                    xs = [p[1] for p in traj.attacker_xy]
                    ys = [p[2] for p in traj.attacker_xy]
                    lines!(ax_att, xs, ys, color=(col, alpha), linewidth=linewidth)
                end
                if length(traj.defender_xy) > 1
                    xs = [p[1] for p in traj.defender_xy]
                    ys = [p[2] for p in traj.defender_xy]
                    lines!(ax_def, xs, ys, color=(col, alpha), linewidth=linewidth)
                end
            end
        end

        # Legend
        legend_entries = []
        for vd in selected_variants
            nccw_str = vd.nccw == floor(vd.nccw) ? string(Int(vd.nccw)) : string(vd.nccw)
            push!(legend_entries, (color=nccw_color(vd.nccw), label="nccw=$nccw_str", style=:solid))
        end
        push!(legend_entries, (color=:gray, label="Non-Robust", style=:dash))

        legend_elements = [LineElement(color=e.color, linestyle=e.style, linewidth=2) for e in legend_entries]
        legend_labels = [e.label for e in legend_entries]
        Legend(fig[2, :], legend_elements, legend_labels,
            orientation=:horizontal, nbanks=2, framevisible=false)

        out_png = joinpath(group_output_dir, "cross_nccw_belief_trajectories.png")
        out_pdf = joinpath(group_output_dir, "cross_nccw_belief_trajectories.pdf")
        save(out_png, fig)
        save(out_pdf, fig)
        println("  Saved: $out_png")
        println("  Saved: $out_pdf")
    end
end
