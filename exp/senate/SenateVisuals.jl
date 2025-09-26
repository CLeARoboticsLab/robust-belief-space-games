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

function get_ellipse_points(center, cov; n=50, conf=1.0)
    eig_decomp = eigen(cov)
    λ = eig_decomp.values
    v = eig_decomp.vectors
    a = conf * sqrt(λ[2]) # Major axis
    b = conf * sqrt(λ[1]) # Minor axis
    θ = atan(v[2, 2], v[1, 2]) # Angle of major axis eigenvector

    t_range = range(0, 2*pi, length=n)
    
    pts = Point2f[]
    for t_ellipse in t_range
        x_r = a * cos(t_ellipse)
        y_r = b * sin(t_ellipse)
        x = center[1] + x_r * cos(θ) - y_r * sin(θ)
        y = center[2] + x_r * sin(θ) + y_r * cos(θ)
        push!(pts, Point2f(x, y))
    end
    # close the ellipse
    push!(pts, pts[1])
    return pts
end

function visualize_receding_horizon_solution(solutions, games; dims)
    
    cost_params = Dict(
        1 => (;pos = [[1,1]], scale = [[1,2]]),
        2 => (;pos = [[3,0]], scale = [[2,1]]),
    )
    colors = [:blue, :red]

    fig = Figure(size = (1200, 800))

    # Time slider
    belief_trajectory = solutions["non_robust"][1]
    time_steps = length(belief_trajectory)
    slider = Slider(fig[2, 1:2], range = 1:time_steps, startvalue = 1)
    t = slider.value

    point_colors = vcat([fill(c, dims.num_senators) for c in colors]...)

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
    means_trajectory = [means(b) for b in belief_trajectory]
    covariances_trajectory = [covs(b) for b in belief_trajectory]
    
    # Plot full trajectories as lines
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id-1)*dims.num_senators + senator_id
            traj_points = [Point2f(means_trajectory[time][Block(belief_idx)][1], means_trajectory[time][Block(belief_idx)][2]) for time in 1:time_steps]
            lines!(ax1, traj_points, color=colors[activist_id])
        end
    end

    # Create observables and plots for covariance ellipses
    ellipse_observables1 = []
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            obs = Observable(Point2f[])
            lines!(ax1, obs, color=colors[activist_id])
            push!(ellipse_observables1, obs)
        end
    end

    points = @lift begin
        current_means = means_trajectory[$t]
        pts = Point2f[]
        for activist_id in 1:dims.num_activists
            for senator_id in 1:dims.num_senators
                belief_idx = (activist_id-1)*dims.num_senators + senator_id
                push!(pts, Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2]))
            end
        end
        pts
    end
    
    scatter!(ax1, points, color=point_colors, markersize=8)

    # Update ellipses on slider change
    on(t) do time_step
        current_covs = covariances_trajectory[time_step]
        current_means = means_trajectory[time_step]
        plot_idx = 1
        for activist_id in 1:dims.num_activists
            for senator_id in 1:dims.num_senators
                belief_idx = (activist_id-1)*dims.num_senators + senator_id
                center = Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2])
                cov = current_covs[belief_idx]
                
                ellipse_points = get_ellipse_points(center, cov)
                
                ellipse_observables1[plot_idx][] = ellipse_points
                plot_idx += 1
            end
        end
    end
    t[] = t[] # Trigger initial plot


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
    belief_trajectory_robust = solutions["robust"][1]
    means_trajectory_robust = [means(b) for b in belief_trajectory_robust]
    covariances_trajectory_robust = [covs(b) for b in belief_trajectory_robust]

    # Plot full trajectories as lines
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id-1)*dims.num_senators + senator_id
            traj_points = [Point2f(means_trajectory_robust[time][Block(belief_idx)][1], means_trajectory_robust[time][Block(belief_idx)][2]) for time in 1:time_steps]
            lines!(ax2, traj_points, color=colors[activist_id])
        end
    end

    # Create observables and plots for covariance ellipses
    ellipse_observables2 = []
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            obs = Observable(Point2f[])
            lines!(ax2, obs, color=colors[activist_id])
            push!(ellipse_observables2, obs)
        end
    end

    points_robust = @lift begin
        current_means = means_trajectory_robust[$t]
        pts = Point2f[]
        for activist_id in 1:dims.num_activists
            for senator_id in 1:dims.num_senators
                belief_idx = (activist_id-1)*dims.num_senators + senator_id
                push!(pts, Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2]))
            end
        end
        pts
    end
    
    scatter!(ax2, points_robust, color=point_colors, markersize=8)
    
    # Update ellipses on slider change
    on(t) do time_step
        current_covs = covariances_trajectory_robust[time_step]
        current_means = means_trajectory_robust[time_step]
        plot_idx = 1
        for activist_id in 1:dims.num_activists
            for senator_id in 1:dims.num_senators
                belief_idx = (activist_id-1)*dims.num_senators + senator_id
                center = Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2])
                cov = current_covs[belief_idx]
                
                ellipse_points = get_ellipse_points(center, cov)
                
                ellipse_observables2[plot_idx][] = ellipse_points
                plot_idx += 1
            end
        end
    end
    t[] = t[] # Trigger initial plot

    axislegend(ax1)
    axislegend(ax2)
    display(fig)
    fig
end

end
