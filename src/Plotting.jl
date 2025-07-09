using CairoMakie

function plot_feed_forward_norms(feed_forward_norms_history)
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel="Iteration",
        ylabel="Norm of feed_forward",
        title="Feed-forward norms per iteration")

    for (iter, norms) in enumerate(feed_forward_norms_history)
        scatter!(ax, fill(iter, length(norms)), norms, markersize=4, color=:black)
    end
    save("exp/hockey/outputs/feed_forward_norms.png", fig)
    return fig
end
