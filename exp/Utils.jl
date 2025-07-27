module Utils
using JLD2
using FileIO
using Statistics
using LinearAlgebra
using BlockArrays
using Infiltrator
using RobustBeliefGame
using Printf



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

end