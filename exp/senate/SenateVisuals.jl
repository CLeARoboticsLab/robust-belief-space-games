module SenateVisuals

using GLMakie
using LinearAlgebra
using RobustBeliefGame
using BlockArrays

export visualize_receding_horizon_solution

function plot_ellipse!(ax, center, a, b; n=100, label="", color=:black)
    t = range(0, 2*pi, length=n)
    x = center[1] .+ a .* cos.(t)
    y = center[2] .+ b .* sin.(t)
    lines!(ax, x, y, label=label, color=color)
end

function visualize_receding_horizon_solution(solutions, games; dims)
    
    cost_params = Dict(
        1 => (;pos = [[1,1]], scale = [[1,2]]),
        2 => (;pos = [[3,0]], scale = [[2,1]]),
    )
    colors = [:blue, :red]

    fig = Figure(size = (1200, 600))

    # Plot for non-robust game
    ax1 = Axis(fig[1, 1], title="Non-Robust Solution", xlabel="Opinion Dimension 1", ylabel="Opinion Dimension 2", aspect=DataAspect())
    
    # Plot activist preferences
    for activist_id in 1:dims.num_activists
        params = cost_params[activist_id]
        center = params.pos[1]
        scale = params.scale[1]
        a = sqrt(10 * scale[1])
        b = sqrt(10 * scale[2])
        plot_ellipse!(ax1, center, a, b, label="Activist $activist_id Pref.", color=colors[activist_id])
    end

    # Plot belief trajectories
    belief_trajectory = solutions["non_robust"].b
    for t in 1:length(belief_trajectory)
        current_means = means(belief_trajectory[t])
        for activist_id in 1:dims.num_activists
            for senator_id in 1:dims.num_senators
                belief_idx = (activist_id-1)*dims.num_senators + senator_id
                scatter!(ax1, current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2], 
                         color=colors[activist_id], markersize=4)
                if t > 1
                    prev_means = means(belief_trajectory[t-1])
                    lines!(ax1, [prev_means[Block(belief_idx)][1], current_means[Block(belief_idx)][1]],
                          [prev_means[Block(belief_idx)][2], current_means[Block(belief_idx)][2]],
                          color=colors[activist_id])
                end
            end
        end
    end


    # Plot for robust game
    ax2 = Axis(fig[1, 2], title="Robust Solution", xlabel="Opinion Dimension 1", aspect=DataAspect())
    
    # Plot activist preferences
    for activist_id in 1:dims.num_activists
        params = cost_params[activist_id]
        center = params.pos[1]
        scale = params.scale[1]
        a = sqrt(10 * scale[1])
        b = sqrt(10 * scale[2])
        plot_ellipse!(ax2, center, a, b, label="Activist $activist_id Pref.", color=colors[activist_id])
    end

    # Plot belief trajectories
    belief_trajectory_robust = solutions["robust"].b
    for t in 1:length(belief_trajectory_robust)
        current_means = means(belief_trajectory_robust[t])
        for activist_id in 1:dims.num_activists
            for senator_id in 1:dims.num_senators
                belief_idx = (activist_id-1)*dims.num_senators + senator_id
                scatter!(ax2, current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2], 
                         color=colors[activist_id], markersize=4)
                if t > 1
                    prev_means = means(belief_trajectory_robust[t-1])
                    lines!(ax2, [prev_means[Block(belief_idx)][1], current_means[Block(belief_idx)][1]],
                          [prev_means[Block(belief_idx)][2], current_means[Block(belief_idx)][2]],
                          color=colors[activist_id])
                end
            end
        end
    end
    
    axislegend(ax1)
    axislegend(ax2)
    fig
end

end
