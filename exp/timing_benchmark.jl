"""
Computational overhead benchmark for robust belief-space game solver.

Measures wall-clock time and iLQG iteration counts per solver invocation.
Compares P1 (non-robust, 2-player game) vs P2 (robust, 3-player game with nature)
within the same trial for a matched comparison.

Uses the same experiment infrastructure as the drift sweep experiments.

Usage:
    julia --project=. exp/timing_benchmark.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics
using Printf
using LinearAlgebra
using BlockArrays

# Load senate experiment infrastructure (includes base_experiment.jl)
include(joinpath(@__DIR__, "senate", "versions", "assymetric_experiment.jl"))

# Load hockey
include(joinpath(@__DIR__, "hockey", "src", "Hockey.jl"))

const N_TRIALS = 10

# ─────────────────────────────────────────────────────────────
# Senate — uses build_asymmetric_player_configs (matching drift sweep)
# ─────────────────────────────────────────────────────────────

function make_senate_params(; seed::Int)
    # Match the run_drift_test_sweep config from test_drift_sweep.jl
    combo = Dict{Symbol,Any}(
        :p2_type            => robust,
        :random_seed        => seed,
        :p2_nature_multiplier => 2.0,
        :p2_believes_p1_drift_sensor_scale => 0.0,
        :p1_believes_self_drift_sensor_scale => 1.0,
        :p1_ellipsoidal_cost_weight => 0.5,
        :p2_ellipsoidal_cost_weight => 0.5,
        :p1_control_cost_weight => 2.0,
        :p2_control_cost_weight => 2.0,
        :gt_drift_sensor_scale  => 1.0,
    )
    fixed_params = Dict{Symbol,Any}(
        :p1_type => non_robust,
        :p1_non_terminal_cost_model_template => obstacle_non_terminal_cost_function_generator,
        :p1_terminal_cost_model_template     => obstacle_terminal_cost_function_generator,
        :p2_non_terminal_cost_model_template => obstacle_non_terminal_cost_function_generator,
        :p2_terminal_cost_model_template     => obstacle_terminal_cost_function_generator,
        :p1_obstacle_centers  => [[1.5, 1.5]],
        :p2_obstacle_centers  => [[1.5, 1.5]],
        :p1_obstacle_weights  => Vector{Float64}([8.0]),
        :p2_obstacle_weights  => Vector{Float64}([8.0]),
        :dynamics_model_template => :default,
        :dt      => 0.75,
        :horizon => 10,
        :process_noise_covariance => 0.01 * I(6),
        :sensor_noise_covariance  => 0.01 * I(6),
        :ground_truth_initial_states => mortar([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]]),
    )
    player_configs = build_asymmetric_player_configs(combo, fixed_params)
    build_senate_params(combo, fixed_params, player_configs)
end

function extract_senate_timing_by_player(solutions_dict)
    player_timings = Dict{Int, Vector{Float64}}()
    player_iters   = Dict{Int, Vector{Int}}()
    for player_idx in sort(collect(keys(solutions_dict)))
        costs = solutions_dict[player_idx].cost_history
        player_timings[player_idx] = Float64[c.solve_time for c in costs]
        player_iters[player_idx]   = Int[c.solver_iterations for c in costs]
    end
    return player_timings, player_iters
end

# ─────────────────────────────────────────────────────────────
# Hockey — P2 (defender) robust
# ─────────────────────────────────────────────────────────────

function make_hockey_params(; seed::Int)
    params = Hockey.HockeyParams(
        name    = "timing_bench_hockey_s$(seed)",
        horizon = 10,
        planning_horizon = 5,
        random_seed = seed,
        output_dir  = mktempdir(),
    )
    params.player_configs[1].type = Hockey.non_robust
    params.player_configs[2].type = Hockey.robust
    return params
end

function extract_hockey_timing_by_player(result)
    player_timings = Dict{Int, Vector{Float64}}()
    player_iters   = Dict{Int, Vector{Int}}()
    for player_idx in sort(collect(keys(result.solution_history)))
        steps = result.solution_history[player_idx]
        player_timings[player_idx] = Float64[s.costs.solve_time for s in steps]
        player_iters[player_idx]   = Int[s.costs.solver_iterations for s in steps]
    end
    return player_timings, player_iters
end

# ─────────────────────────────────────────────────────────────
# Benchmark driver
# ─────────────────────────────────────────────────────────────

function run_benchmark(; n_trials::Int = N_TRIALS)
    experiments = [
        ("Senate", :senate),
        ("Hockey", :hockey),
    ]

    results = Dict{Tuple{String,Int}, NamedTuple}()

    for (exp_name, exp_sym) in experiments
        println("\n" * "="^60)
        println("  $exp_name  (P1=baseline, P2=robust)")
        println("="^60)

        # ── JIT warmup ──
        print("  Warmup...")
        redirect_stdout(devnull) do
            if exp_sym == :senate
                p = make_senate_params(; seed=9999)
                run_receding_horizon_trial(p; override=true)
            else
                p = make_hockey_params(; seed=9999)
                Hockey.run_receding_horizon_trial(p; override=true)
            end
        end
        println(" done.")

        # ── Measurement ──
        per_player_timings = Dict(1 => Float64[], 2 => Float64[])
        per_player_iters   = Dict(1 => Int[],     2 => Int[])

        for trial in 1:n_trials
            seed = 1000 + trial

            if exp_sym == :senate
                p = make_senate_params(; seed=seed)
                sol, _ = redirect_stdout(devnull) do
                    run_receding_horizon_trial(p; override=true)
                end
                pt, pi = extract_senate_timing_by_player(sol)
            else
                p = make_hockey_params(; seed=seed)
                result = redirect_stdout(devnull) do
                    Hockey.run_receding_horizon_trial(p; override=true)
                end
                pt, pi = extract_hockey_timing_by_player(result)
            end

            for pidx in [1, 2]
                append!(per_player_timings[pidx], pt[pidx])
                append!(per_player_iters[pidx],   pi[pidx])
            end
            @printf("  Trial %2d/%d  P1=%.4fs  P2=%.4fs  (%d solves/player)\n",
                trial, n_trials, mean(pt[1]), mean(pt[2]), length(pt[1]))
        end

        for pidx in [1, 2]
            t  = per_player_timings[pidx]
            it = Float64.(per_player_iters[pidx])
            results[(exp_name, pidx)] = (
                timings   = t,
                iters     = per_player_iters[pidx],
                mean_time = mean(t),
                std_time  = std(t),
                mean_iter = mean(it),
                std_iter  = std(it),
                n         = length(t),
            )
        end
    end

    print_results_table(results, n_trials)
    return results
end

# ─────────────────────────────────────────────────────────────
# Table output
# ─────────────────────────────────────────────────────────────

function print_results_table(results, n_trials)
    w = 100
    println("\n\n")
    println("="^w)
    println("  SOLVER COMPUTATIONAL OVERHEAD BENCHMARK")
    println("  $(n_trials) trials · per-solve() call statistics")
    println("  Baseline = P1 (non-robust, 2-player game)")
    println("  Robust   = P2 (robust, 3-player game with nature adversary)")
    println("="^w)
    println()
    @printf("%-25s  %11s  %9s  %11s  %9s  %11s  %6s\n",
        "Configuration", "Time (s)", "± Std", "Iterations", "± Std", "Overhead", "N")
    println("-"^w)

    for exp_name in ["Senate", "Hockey"]
        bk = (exp_name, 1)
        rk = (exp_name, 2)
        if !haskey(results, bk) || !haskey(results, rk)
            continue
        end
        b = results[bk]
        r = results[rk]
        overhead = (r.mean_time - b.mean_time) / b.mean_time * 100.0

        @printf("%-25s  %11.4f  %9.4f  %11.1f  %9.1f  %11s  %6d\n",
            "$exp_name P1 (Baseline)", b.mean_time, b.std_time, b.mean_iter, b.std_iter, "—", b.n)
        @printf("%-25s  %11.4f  %9.4f  %11.1f  %9.1f  %+10.1f%%  %6d\n",
            "$exp_name P2 (Robust)", r.mean_time, r.std_time, r.mean_iter, r.std_iter, overhead, r.n)
        println()
    end

    println("="^w)
end

# ── Run ──
run_benchmark()
