"""
Computational overhead benchmark for robust belief-space game solver.

Measures wall-clock time and iLQG iteration counts per solver invocation
across hockey and senate (activism) experiments, comparing robust vs baseline.

Usage:
    julia --project=. exp/timing_benchmark.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics
using Printf

import Senate
include(joinpath(@__DIR__, "hockey", "src", "Hockey.jl"))

const N_TRIALS = 10

# ─────────────────────────────────────────────────────────────
# Senate (activism) configuration
# ─────────────────────────────────────────────────────────────

function make_senate_params(; p2_robust::Bool, seed::Int)
    p2_type = p2_robust ? Senate.robust : Senate.non_robust
    Senate.DefaultSenateParams(
        name = "timing_bench_senate_r$(p2_robust)_s$(seed)",
        player_configs = Dict(
            1 => Senate.DefaultPlayerConfig(player_idx=1, type=Senate.non_robust,
                ellipsoid_centers = [[1.0, -1.0]],
                ellipsoid_radii  = [[3.0, 1.0]]),
            2 => Senate.DefaultPlayerConfig(player_idx=2, type=p2_type,
                ellipsoid_centers = [[-1.0, 1.0]],
                ellipsoid_radii  = [[1.0, 3.0]]),
        ),
        horizon = 10,
        planning_horizon = 5,
        random_seed = seed,
    )
end

function extract_senate_timing(solutions_dict)
    timings = Float64[]
    iters   = Int[]
    for player_idx in sort(collect(keys(solutions_dict)))
        for cost in solutions_dict[player_idx].cost_history
            push!(timings, cost.solve_time)
            push!(iters, cost.solver_iterations)
        end
    end
    return timings, iters
end

# ─────────────────────────────────────────────────────────────
# Hockey configuration
# ─────────────────────────────────────────────────────────────

function make_hockey_params(; p2_robust::Bool, seed::Int)
    params = Hockey.HockeyParams(
        name    = "timing_bench_hockey_r$(p2_robust)_s$(seed)",
        horizon = 10,
        planning_horizon = 5,
        random_seed = seed,
        output_dir  = mktempdir(),
    )
    params.player_configs[1].type = Hockey.non_robust
    params.player_configs[2].type = p2_robust ? Hockey.robust : Hockey.non_robust
    return params
end

function extract_hockey_timing(result)
    timings = Float64[]
    iters   = Int[]
    for player_idx in sort(collect(keys(result.solution_history)))
        for step_data in result.solution_history[player_idx]
            push!(timings, step_data.costs.solve_time)
            push!(iters, step_data.costs.solver_iterations)
        end
    end
    return timings, iters
end

# ─────────────────────────────────────────────────────────────
# Benchmark driver
# ─────────────────────────────────────────────────────────────

function run_benchmark(; n_trials::Int = N_TRIALS)
    configs = [
        ("Senate Baseline", :senate, false),
        ("Senate Robust",   :senate, true),
        ("Hockey Baseline", :hockey, false),
        ("Hockey Robust",   :hockey, true),
    ]

    results = Dict{String, NamedTuple}()

    for (label, experiment, is_robust) in configs
        println("\n" * "="^60)
        println("  $label")
        println("="^60)

        # ── JIT warmup (throwaway run) ──
        print("  Warmup...")
        redirect_stdout(devnull) do
            if experiment == :senate
                p = make_senate_params(; p2_robust=is_robust, seed=9999)
                Senate.run_receding_horizon_trial(p; override=true)
            else
                p = make_hockey_params(; p2_robust=is_robust, seed=9999)
                Hockey.run_receding_horizon_trial(p; override=true)
            end
        end
        println(" done.")

        # ── Measurement trials ──
        all_timings = Float64[]
        all_iters   = Int[]

        for trial in 1:n_trials
            seed = 1000 + trial

            if experiment == :senate
                p = make_senate_params(; p2_robust=is_robust, seed=seed)
                sol, _ = redirect_stdout(devnull) do
                    Senate.run_receding_horizon_trial(p; override=true)
                end
                t, i = extract_senate_timing(sol)
            else
                p = make_hockey_params(; p2_robust=is_robust, seed=seed)
                result = redirect_stdout(devnull) do
                    Hockey.run_receding_horizon_trial(p; override=true)
                end
                t, i = extract_hockey_timing(result)
            end

            append!(all_timings, t)
            append!(all_iters, i)
            @printf("  Trial %2d/%d  mean=%.4fs/solve  solves=%d\n",
                trial, n_trials, mean(t), length(t))
        end

        results[label] = (
            timings   = all_timings,
            iters     = all_iters,
            mean_time = mean(all_timings),
            std_time  = std(all_timings),
            mean_iter = mean(Float64.(all_iters)),
            std_iter  = std(Float64.(all_iters)),
            n         = length(all_timings),
        )
    end

    # ── Print results table ──
    print_results_table(results, n_trials)

    return results
end

# ─────────────────────────────────────────────────────────────
# Table output
# ─────────────────────────────────────────────────────────────

function print_results_table(results, n_trials)
    w = 97
    println("\n\n")
    println("="^w)
    println("  SOLVER COMPUTATIONAL OVERHEAD BENCHMARK")
    println("  $(n_trials) trials per configuration, per-solve() call statistics")
    println("="^w)
    println()
    @printf("%-22s  %11s  %9s  %11s  %9s  %11s  %6s\n",
        "Configuration", "Time (s)", "± Std", "Iterations", "± Std", "Overhead", "N")
    println("-"^w)

    for experiment in ["Senate", "Hockey"]
        bk = "$experiment Baseline"
        rk = "$experiment Robust"
        if !haskey(results, bk) || !haskey(results, rk)
            continue
        end
        b = results[bk]
        r = results[rk]
        overhead = (r.mean_time - b.mean_time) / b.mean_time * 100.0

        @printf("%-22s  %11.4f  %9.4f  %11.1f  %9.1f  %11s  %6d\n",
            bk, b.mean_time, b.std_time, b.mean_iter, b.std_iter, "—", b.n)
        @printf("%-22s  %11.4f  %9.4f  %11.1f  %9.1f  %+10.1f%%  %6d\n",
            rk, r.mean_time, r.std_time, r.mean_iter, r.std_iter, overhead, r.n)
        println()
    end

    println("="^w)
end

# ── Run ──
run_benchmark()
