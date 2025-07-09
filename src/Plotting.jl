
function plot_feed_forward_norms(feed_forward_norms_history)
    if isempty(feed_forward_norms_history) || isempty(feed_forward_norms_history[1])
        return Figure() # Return an empty figure if there's no data
    end

    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel="Iteration",
        ylabel="Norm of feed_forward",
        title="Feed-forward norms per iteration")

    # Convert the history into a matrix where rows are timesteps and columns are iterations
    norms_matrix = reduce(hcat, feed_forward_norms_history[2:end])
    num_timesteps, num_iterations = size(norms_matrix)
    iterations = 1:num_iterations

    for t in 1:num_timesteps
        lines!(ax, iterations, norms_matrix[t, :], label="t=$t", colormap=Reverse(:viridis), color=t, colorrange=(1, num_timesteps))
    end

    if num_timesteps <= 20
      axislegend(ax)
    end
    
    save("exp/hockey/outputs/feed_forward_norms.png", fig)
    return fig
end
