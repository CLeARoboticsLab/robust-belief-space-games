using Distributed
using BlockArrays
using LinearAlgebra
using Distributions
using Senate

# Include param_sweep for run_param_sweep
include("param_sweep.jl")

"""
    generate_drift_test_configs()

Generate config list for testing drift mismatch scenarios.
Based on v1 from drift_experiment_runner.jl.

Varies:
- P2 type: non_robust vs robust
- P2's belief about P1 drift: 0.0 (mismatch) vs 1.0 (correct)
- Nature multiplier for robust P2
"""
function generate_drift_test_configs()
    configs = Dict{Symbol, Any}[]

    # Format helper
    fmt(v) = replace(string(round(v, sigdigits=2)), "." => "p")

    # Parameter ranges
    p2_belief_drift_values = [0.0, 1.0]  # 0 = mismatch, 1 = correct
    nature_multiplier_values = [0.5, 8.0]

    # Non-robust P2 configs (one per drift belief)
    for drift in p2_belief_drift_values
        push!(configs, Dict{Symbol, Any}(
            :name => "drift_test_p2nr_drift$(fmt(drift))",
            :player_configs => Dict(
                1 => Dict(
                    :type => non_robust,
                ),
                2 => Dict(
                    :type => non_robust,
                    :drift_sensor_scale => drift,
                )
            ),
            :horizon => 7,
        ))
    end

    # Robust P2 configs (one per drift belief × nature multiplier)
    for drift in p2_belief_drift_values
        for nm in nature_multiplier_values
            push!(configs, Dict{Symbol, Any}(
                :name => "drift_test_p2r_drift$(fmt(drift))_nm$(fmt(nm))",
                :player_configs => Dict(
                    1 => Dict(
                        :type => non_robust,
                    ),
                    2 => Dict(
                        :type => robust,
                        :drift_sensor_scale => drift,
                        :nature_multiplier => nm,
                    )
                ),
                :horizon => 7,
            ))
        end
    end

    return configs
end

"""
    run_drift_test_sweep(; cores=4, override=false)

Run a small test sweep varying drift mismatch parameters.
Tests the ExperimentRunner with senate experiments.
"""
function run_drift_test_sweep(; cores=4, override=false)
    config_list = generate_drift_test_configs()

    println("Generated $(length(config_list)) drift test configurations:")
    for (i, c) in enumerate(config_list)
        println("  [$i] $(get(c, :name, "unnamed"))")
    end

    println("\nContinue? (y/n)")
    if readline() != "y"
        return nothing
    end

    results = run_param_sweep(config_list; cores=cores, output_subdir="drift_test", override=override)

    println("Drift test sweep completed.")
    return results
end
