using Pkg
Pkg.activate(".")
using Serialization
using Senate
using Statistics
using Printf

data_dir = "exp/senate/outputs/merged/nature_control_sweep"
multipliers = [1,2,5,10,25, 50,125, 250, 625,1250,3125, 6250]
seeds = collect(1001:1100)

println("=" ^ 90)
println("Nature Feedforward Norm at Convergence (Empirical Validation)")
println("=" ^ 90)
println()
@printf("%-8s %-8s %-14s %-14s %-14s %-14s\n", "λ", "n_seeds", "ff_mean", "ff_max", "ctrl_mean", "fb_gain_mean")
println("-" ^ 90)

for m in multipliers
    # Collect per-seed means, then average across seeds
    seed_ff_means = Float64[]
    seed_ff_maxes = Float64[]
    seed_ctrl_means = Float64[]
    seed_fb_means = Float64[]

    for seed in seeds
        fname = "seed_$(seed)_p2_believes_p1_drift_sensor_scale_0.0_p2_nature_multiplier_$(m)_p2_type_robust_mass_results.dat"
        fpath = joinpath(data_dir, fname)
        if !isfile(fpath)
            continue
        end
        try
            data = open(deserialize, fpath, "r")
            entry = data[1]
            results = entry.results
            if isnothing(results) || isempty(results)
                continue
            end
            all_ff = Float64[]
            all_ctrl = Float64[]
            all_fb = Float64[]
            for (trial_key, trial_val) in results
                if !(trial_val isa Tuple)
                    continue
                end
                solutions_dict = trial_val[1]
                for (player_idx, sol) in solutions_dict
                    if !hasproperty(sol, :nature_diagnostics_history) || isempty(sol.nature_diagnostics_history)
                        continue
                    end
                    for diag in sol.nature_diagnostics_history
                        if diag isa AbstractVector
                            for d in diag
                                push!(all_ff, d.nature_feedforward_norm)
                                push!(all_ctrl, d.nature_control_norm)
                                push!(all_fb, d.nature_feedback_gain_norm)
                            end
                        elseif diag isa NamedTuple
                            push!(all_ff, diag.nature_feedforward_norm)
                            push!(all_ctrl, diag.nature_control_norm)
                            push!(all_fb, diag.nature_feedback_gain_norm)
                        end
                    end
                end
            end
            if !isempty(all_ff)
                push!(seed_ff_means, mean(all_ff))
                push!(seed_ff_maxes, maximum(all_ff))
                push!(seed_ctrl_means, mean(all_ctrl))
                push!(seed_fb_means, mean(all_fb))
            end
        catch e
            # skip failed seeds
        end
    end

    if !isempty(seed_ff_means)
        @printf("%-8d %-8d %-14.8f %-14.8f %-14.8f %-14.8f\n",
            m, length(seed_ff_means),
            mean(seed_ff_means), mean(seed_ff_maxes),
            mean(seed_ctrl_means), mean(seed_fb_means))
    end
end

println()
println("=" ^ 90)
println("Summary: Nature's feedforward norm ||δu_ff|| should be near zero at convergence.")
println("The KKT convergence criterion (ε=0.01) implies ||Q_u|| < 0.01,")
println("so ||δu_ff|| = ||Q_uu^{-1} Q_u|| ≤ ||Q_u|| / σ_min(Q_uu) << 1.")
println("=" ^ 90)
