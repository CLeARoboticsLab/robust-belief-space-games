module Utils
using JLD2
using FileIO
using Statistics
using LinearAlgebra
using BlockArrays
using Infiltrator
using RobustBeliefGame



function effect_of_nature(rh_file_ids; save_to_file=false)
    if isempty(rh_file_ids)
        output_dir = "exp/hockey/outputs"
        rh_file_ids = [replace(f, ".jld2" => "") for f in readdir(output_dir) if startswith(f, "rh_")]
    end

    for file_id in rh_file_ids
        println("Analyzing file: $file_id.jld2")
        filepath = "exp/hockey/outputs/$file_id.jld2"
        if !isfile(filepath)
            println("File not found: $filepath")
            continue
        end

        @load filepath gt_state_history all_observations goal_position solution_history
        
        analysis_text = "Analysis for $file_id:\n"

        for t in 1:length(solution_history)
            non_robust_traj = solution_history[t][1]
            robust_traj = solution_history[t][2]
            
            # 1. Absolute L2 norm difference between trajectories
            traj_diff = sum(norm(means(non_robust_traj[k]) - means(robust_traj[k])) for k in eachindex(robust_traj))
            analysis_text *= "  planned @t=$t: Trajectory Difference (L2 Norm): $traj_diff\n"

            # 2. Average L2 norm difference per unit control cost of nature
            nature_control_cost = norm(nature_us_history[t])^2
            diff_per_cost = nature_control_cost > 1e-9 ? traj_diff / nature_control_cost : traj_diff
            analysis_text *= "    - Diff per Nature's Control Cost: $diff_per_cost\n"
            analysis_text *= "    - Nature's Control Cost: $nature_control_cost\n"

            # 3. Difference in robust and non-robust actions
            non_robust_us = planned_us_history[t][1]
            robust_us = planned_us_history[t][2]
            
            # We only need to look at the first action, as this is what is executed
            attacker_action_diff = norm(non_robust_us[1][Block(1)] - robust_us[1][Block(1)])
            defender_action_diff = norm(non_robust_us[1][Block(2)] - robust_us[1][Block(2)])

            analysis_text *= "    - Attacker Action Difference (accel, steer): ($(-attacker_action_diff[1]), $(-attacker_action_diff[2]))\n"
            analysis_text *= "    - Defender Action Difference (accel, steer): ($(-defender_action_diff[1]), $(-defender_action_diff[2]))\n"
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

end