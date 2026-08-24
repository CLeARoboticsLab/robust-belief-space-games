"""
Computational overhead benchmark for robust belief-space game solver.

Compares P2 non-robust vs P2 robust solve times across nature multiplier
values from the experimental sweeps.

Usage:
    julia --project=. exp/timing_benchmark.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics
using Printf
using LinearAlgebra
using BlockArrays

# Load senate experiment infrastructure
include(joinpath(@__DIR__, "senate", "versions", "assymetric_experiment.jl"))

const N_TRIALS = 5

# ═════════════════════════════════════════════════════════════
# Senate benchmark
# ═════════════════════════════════════════════════════════════

# Nature multipliers from run_nature_control_sweep in test_drift_sweep.jl
const SENATE_NATURE_MULTIPLIERS = [1, 2, 5, 10, 25, 50, 125, 250, 625, 1250, 3125, 6250]

function make_senate_params(; p2_robust::Bool, seed::Int, nature_multiplier::Real=2.0)
    p2_type = p2_robust ? robust : non_robust
    combo = Dict{Symbol,Any}(
        :p2_type            => p2_type,
        :random_seed        => seed,
        :p2_nature_multiplier => Float64(nature_multiplier),
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

function extract_p2_timing(solutions_dict)
    costs = solutions_dict[2].cost_history
    timings = Float64[c.solve_time for c in costs]
    iters   = Int[c.solver_iterations for c in costs]
    return timings, iters
end

function bench_senate_config(label, make_params_fn; n_trials)
    println("\n  $label")
    print("    Warmup...")
    redirect_stdout(devnull) do
        p = make_params_fn(9999)
        run_receding_horizon_trial(p; override=true)
    end
    println(" done.")

    all_timings = Float64[]
    all_iters   = Int[]
    for trial in 1:n_trials
        seed = 1000 + trial
        p = make_params_fn(seed)
        sol, _ = redirect_stdout(devnull) do
            run_receding_horizon_trial(p; override=true)
        end
        t, i = extract_p2_timing(sol)
        append!(all_timings, t)
        append!(all_iters, i)
        @printf("    Trial %2d/%d  mean=%.4fs/solve  mean_iter=%.2f  total_iter=%d\n",
        trial, n_trials, mean(t), mean(i), sum(i))    end
    return (
        timings   = all_timings,
        iters     = all_iters,
        mean_time = mean(all_timings),
        std_time  = std(all_timings),
        mean_iter = mean(Float64.(all_iters)),
        std_iter  = std(Float64.(all_iters)),
        n         = length(all_timings),
    )
end

function print_senate_table(nr_timings, nr_iters, r_timings, r_iters, trial, n_trials, λ)
    w = 105
    println("\n" * "="^w)
    @printf("  SENATE: Non-Robust vs Robust (λ=%d) · %d/%d trials done\n", λ, trial, n_trials)
    println("="^w)
    println()
    @printf("%-22s  %11s  %9s  %11s  %9s  %11s  %6s\n",
        "Configuration", "Time (s)", "± Std", "Iterations", "± Std", "Overhead", "N")
    println("-"^w)
    nr_mt = mean(nr_timings); nr_st = length(nr_timings) > 1 ? std(nr_timings) : 0.0
    nr_mi = mean(Float64.(nr_iters)); nr_si = length(nr_iters) > 1 ? std(Float64.(nr_iters)) : 0.0
    @printf("%-22s  %11.4f  %9.4f  %11.1f  %9.1f  %11s  %6d\n",
        "Non-Robust", nr_mt, nr_st, nr_mi, nr_si, "—", length(nr_timings))
    if !isempty(r_timings)
        r_mt = mean(r_timings); r_st = length(r_timings) > 1 ? std(r_timings) : 0.0
        r_mi = mean(Float64.(r_iters)); r_si = length(r_iters) > 1 ? std(Float64.(r_iters)) : 0.0
        overhead = (r_mt - nr_mt) / nr_mt * 100.0
        @printf("%-22s  %11.4f  %9.4f  %11.1f  %9.1f  %+10.1f%%  %6d\n",
            "Robust (λ=$λ)", r_mt, r_st, r_mi, r_si, overhead, length(r_timings))
    end
    println("="^w)
end

function run_senate_benchmark(; n_trials::Int = N_TRIALS, λ::Int = 50)
    println("\n" * "="^60)
    println("  SENATE BENCHMARK  (λ=$λ, $n_trials trials)")
    println("="^60)

    # Warmup both configs
    print("\n  Warmup (non-robust)...")
    redirect_stdout(devnull) do
        run_receding_horizon_trial(make_senate_params(; p2_robust=false, seed=9999); override=true)
    end
    println(" done.")
    print("  Warmup (robust λ=$λ)...")
    redirect_stdout(devnull) do
        run_receding_horizon_trial(make_senate_params(; p2_robust=true, seed=9999, nature_multiplier=λ); override=true)
    end
    println(" done.")

    nr_timings = Float64[]; nr_iters = Int[]
    r_timings  = Float64[]; r_iters  = Int[]

    for trial in 1:n_trials
        seed = 1000 + trial

        # --- Non-robust ---
        p_nr = make_senate_params(; p2_robust=false, seed)
        sol_nr, _ = redirect_stdout(devnull) do
            run_receding_horizon_trial(p_nr; override=true)
        end
        t_nr, i_nr = extract_p2_timing(sol_nr)
        append!(nr_timings, t_nr)
        append!(nr_iters, i_nr)
        @printf("  Trial %2d/%d  NR  mean=%.4fs  iters/step=%s\n", trial, n_trials, mean(t_nr), join(i_nr, ","))

        # --- Robust ---
        p_r = make_senate_params(; p2_robust=true, seed, nature_multiplier=λ)
        sol_r, _ = redirect_stdout(devnull) do
            run_receding_horizon_trial(p_r; override=true)
        end
        t_r, i_r = extract_p2_timing(sol_r)
        append!(r_timings, t_r)
        append!(r_iters, i_r)
        @printf("  Trial %2d/%d  R   mean=%.4fs  iters/step=%s\n", trial, n_trials, mean(t_r), join(i_r, ","))

        # --- Running summary after each pair ---
        print_senate_table(nr_timings, nr_iters, r_timings, r_iters, trial, n_trials, λ)
    end

    return (nr_timings=nr_timings, nr_iters=nr_iters, r_timings=r_timings, r_iters=r_iters)
end

# ═════════════════════════════════════════════════════════════
# Hockey benchmark
# ═════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "hockey", "src", "Hockey.jl"))

const HOCKEY_NATURE_COSTS = [10.0]#, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0]

function make_hockey_params(; p2_robust::Bool, seed::Int, nature_cost::Float64=300.0)
    params = Hockey.HockeyParams(
        name    = "timing_bench_hockey_s$(seed)",
        horizon = 10,
        planning_horizon = 5,
        random_seed = seed,
        output_dir  = mktempdir(),
    )
    params.player_configs[1].type = Hockey.non_robust
    params.player_configs[2].type = p2_robust ? Hockey.robust : Hockey.non_robust
    params.player_configs[2].nature_control_cost_weight = nature_cost
    return params
end

function extract_hockey_p2_timing(result)
    steps = result.solution_history[2]
    timings = Float64[s.costs.solve_time for s in steps]
    iters   = Int[s.costs.solver_iterations for s in steps]
    return timings, iters
end

function bench_hockey_config(label, make_params_fn; n_trials)
    println("\n  $label")
    print("    Warmup...")
    redirect_stdout(devnull) do
        p = make_params_fn(9999)
        Hockey.run_receding_horizon_trial(p; override=true)
    end
    println(" done.")

    all_timings = Float64[]
    all_iters   = Int[]
    for trial in 1:n_trials
        seed = 1000 + trial
        p = make_params_fn(seed)
        result = redirect_stdout(devnull) do
            Hockey.run_receding_horizon_trial(p; override=true)
        end
        t, i = extract_hockey_p2_timing(result)
        append!(all_timings, t)
        append!(all_iters, i)
        @printf("    Trial %2d/%d  mean=%.4fs/solve\n", trial, n_trials, mean(t))
    end
    return (
        timings   = all_timings,
        iters     = all_iters,
        mean_time = mean(all_timings),
        std_time  = std(all_timings),
        mean_iter = mean(Float64.(all_iters)),
        std_iter  = std(Float64.(all_iters)),
        n         = length(all_timings),
    )
end

function run_hockey_benchmark(; n_trials::Int = N_TRIALS)
    println("\n" * "="^60)
    println("  HOCKEY BENCHMARK")
    println("="^60)

    results = Dict{String, NamedTuple}()

    # Baseline
    results["Baseline"] = bench_hockey_config("Baseline (non-robust)",
        seed -> make_hockey_params(; p2_robust=false, seed); n_trials)

    # Each nature cost weight
    for nc in HOCKEY_NATURE_COSTS
        nc_int = Int(nc)
        results["nc=$nc_int"] = bench_hockey_config("Robust (nc=$nc_int)",
            seed -> make_hockey_params(; p2_robust=true, seed, nature_cost=nc); n_trials)
    end

    # Table
    b = results["Baseline"]
    w = 97
    println("\n\n")
    println("="^w)
    println("  HOCKEY: SOLVER OVERHEAD vs NATURE CONTROL COST WEIGHT")
    println("  $(n_trials) trials · P2 (defender) solve() calls only")
    println("="^w)
    println()
    @printf("%-22s  %11s  %9s  %11s  %9s  %11s  %6s\n",
        "Configuration", "Time (s)", "± Std", "Iterations", "± Std", "Overhead", "N")
    println("-"^w)
    @printf("%-22s  %11.4f  %9.4f  %11.1f  %9.1f  %11s  %6d\n",
        "Baseline", b.mean_time, b.std_time, b.mean_iter, b.std_iter, "—", b.n)
    for nc in HOCKEY_NATURE_COSTS
        nc_int = Int(nc)
        r = results["nc=$nc_int"]
        overhead = (r.mean_time - b.mean_time) / b.mean_time * 100.0
        @printf("%-22s  %11.4f  %9.4f  %11.1f  %9.1f  %+10.1f%%  %6d\n",
            "Robust (nc=$nc_int)", r.mean_time, r.std_time, r.mean_iter, r.std_iter, overhead, r.n)
    end
    println("="^w)
    return results
end

# ── Run ──
# Call one at a time so you can run them in parallel from separate terminals:
#   julia --project=. -e 'include("exp/timing_benchmark.jl"); run_senate_benchmark()'
#   julia --project=. -e 'include("exp/timing_benchmark.jl"); run_hockey_benchmark()'
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "hockey"
        run_hockey_benchmark()
    elseif length(ARGS) >= 1 && ARGS[1] == "senate"
        run_senate_benchmark()
    else
        println("Usage: julia --project=. exp/timing_benchmark.jl [senate|hockey]")
        println("  Run one at a time, or in parallel from two terminals.")
    end
end
