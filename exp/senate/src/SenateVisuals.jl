module SenateVisuals

using GLMakie
using LinearAlgebra
using RobustBeliefGame
using BlockArrays
using Serialization
using Infiltrator

export visualize_receding_horizon_solution, load_solution


function load_solution(filename)
    path = "exp/senate/outputs/$filename.dat"
    solutions, games, cost_params = open(deserialize, path, "r")
    visualize_receding_horizon_solution(solutions, games, cost_params)
end

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

function visualize_receding_horizon_solution(solutions::Dict, games::Dict, cost_params::Dict)
    solution_keys = collect(keys(games)) # rh_params_etc
    screens = []
    figures = Dict{String, Figure}()
    for sol_name in solution_keys 
        sol_data = solutions[sol_name]
        dims = games[sol_name].robust.dims 
        current_cost_params = cost_params[sol_name]
        screen = GLMakie.Screen()
        fig = Figure()
        figures[sol_name] = fig
        push!(screens, screen) # Keep screen in scope to prevent it from closing

        ax = Axis(fig[1, 1], title="$sol_name Solution", xlabel="Opinion Dimension 1", ylabel="Opinion Dimension 2", aspect=DataAspect())
        
        create_individual_solution_plot(fig, ax, sol_name, sol_data, dims, current_cost_params)
        
        display(screen, fig)

    end
    return figures, screens
end

function create_individual_solution_plot(fig, ax, sol_name, sol_data, dims, cost_params)
    robust_solution_history = sol_data.robust.solution_history
    non_robust_solution_history = sol_data.non_robust.solution_history
    
    # --- Layout ---
    controls_grid = fig[2, 1] = GridLayout(tellwidth=false)
    colsize!(fig.layout, 1, Relative(0.75))
    current_time_step = Observable(1)
    plan_time_step = Observable(1)
    show_robust_planned_trajectory = Observable(true)
    show_non_robust_planned_trajectory = Observable(false)
    show_executed_trajectory = Observable(false)
    colors = [:blue, :red]

    # --- Sliders ---
    time_slider_grid = controls_grid[1, 1] = GridLayout(tellwidth=false)
    Label(time_slider_grid[1, 1], "Receding Horizon Time")
    time_steps = length(robust_solution_history)
    slider = Slider(time_slider_grid[2, 1], range = 1:time_steps, startvalue = 1)
    on(slider.value) do val; current_time_step[] = val; end
    Label(time_slider_grid[3, 1], @lift("$(Int($current_time_step))"))

    robust_planned_belief_trajectory = @lift(robust_solution_history[$current_time_step][1])
    non_robust_planned_belief_trajectory = @lift(non_robust_solution_history[$current_time_step][1])
    robust_planned_controls = @lift(robust_solution_history[$current_time_step][2])
    non_robust_planned_controls = @lift(non_robust_solution_history[$current_time_step][2])

    # Observables for planned trajectories at each receding horizon step `t`
    robust_means_trajectory = @lift [means(b) for b in $robust_planned_belief_trajectory]
    non_robust_means_trajectory = @lift [means(b) for b in $non_robust_planned_belief_trajectory]
    robust_covariances_trajectory = @lift [covs(b) for b in $robust_planned_belief_trajectory]
    non_robust_covariances_trajectory = @lift [covs(b) for b in $non_robust_planned_belief_trajectory]
    robust_planning_horizon = @lift length($robust_means_trajectory)
    non_robust_planning_horizon = @lift length($non_robust_means_trajectory)

    robust_costs = sol_data.robust.cost_history #Contains terminal, non-terminal, and total costs
    non_robust_costs = sol_data.non_robust.cost_history #Contains terminal, non-terminal, and total costs

    plan_slider_grid = controls_grid[2, 1] = GridLayout(tellwidth=false)
    Label(plan_slider_grid[1, 1], "Plan Time")
    plan_slider_range = @lift(1:($robust_planning_horizon > 0 ? $robust_planning_horizon : 1))
    plan_slider = Slider(plan_slider_grid[2, 1], range = plan_slider_range, startvalue = 1)
    on(plan_slider.value) do val; plan_time_step[] = val; end
    Label(plan_slider_grid[3, 1], @lift("$(Int($plan_time_step))"))

    # Plot activist preferences
    for (activist_id, activist_key) in [(1,Main.Senate.non_robust_activist), (2,Main.Senate.robust_activist)] #[non_robust_activist, robust_activist]
        params = cost_params[activist_key]
        center = params.pos[1]
        scale = params.scale[1]
        a = sqrt(1 / scale[1])
        b = sqrt(1 / scale[2])
        plot_ellipse!(ax, center, a, b, label="Activist $activist_key Pref.", color=colors[activist_id])
    end

    point_colors = vcat([fill(c, dims.num_senators) for c in colors]...)

    #executed_trajectory holds the solved trajectory for each time_step for each senator, based on each activist (we only care about gt first state, which is repeated twice)
    for senator_id in 1:dims.num_senators
        senator_states = [robust_solution_history[time][1][1].beliefs[senator_id].belief_mean for time in eachindex(robust_solution_history)]
        gt_trajectory = [Point2f(state[1],state[2]) for state in senator_states]
        scatter!(ax, gt_trajectory, color=:green, markersize=8, visible=show_executed_trajectory)
        lines!(ax, gt_trajectory, color=:green, visible=show_executed_trajectory)

    end
    # Plot full planned trajectories as lines
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id-1)*dims.num_senators + senator_id
            
            robust_traj_points = @lift if $robust_planning_horizon > 0
                if belief_idx <= length(($robust_means_trajectory)[1].blocks)
                    [Point2f(($robust_means_trajectory)[time][Block(belief_idx)][1], ($robust_means_trajectory)[time][Block(belief_idx)][2]) for time in 1:($robust_planning_horizon)]
                else
                    Point2f[]
                end
            else
                Point2f[]
            end
            non_robust_traj_points = @lift if $non_robust_planning_horizon > 0
                if belief_idx <= length(($non_robust_means_trajectory)[1].blocks)
                    [Point2f(($non_robust_means_trajectory)[time][Block(belief_idx)][1], ($non_robust_means_trajectory)[time][Block(belief_idx)][2]) for time in 1:($non_robust_planning_horizon)]
                else
                    Point2f[]
                end
            else
                Point2f[]
            end
            lines!(ax, robust_traj_points, color=colors[activist_id], visible=show_robust_planned_trajectory)
            lines!(ax, non_robust_traj_points, color=colors[activist_id], visible=show_non_robust_planned_trajectory, linestyle=:dash)
        end
    end

    # --- Toggles ---
    toggles_grid = controls_grid[1, 2] = GridLayout(tellwidth=false)
    
    show_robust_activist_controls = Observable(false)
    robust_activist_toggle = Toggle(toggles_grid[1, 1], active=false)
    on(robust_activist_toggle.active) do active; show_robust_activist_controls[] = active; end
    Label(toggles_grid[1, 1], "Show Robust Activist Controls")

    show_non_robust_activist_controls = Observable(false)
    non_robust_activist_toggle = Toggle(toggles_grid[2, 1], active=false)
    on(non_robust_activist_toggle.active) do active; show_non_robust_activist_controls[] = active; end
    Label(toggles_grid[2, 1], "Show Non-Robust Activist Controls")

    show_nature_controls = Observable(false)
    nature_toggle = Toggle(toggles_grid[3, 1], active=false)
    on(nature_toggle.active) do active; show_nature_controls[] = active; end
    Label(toggles_grid[3, 1], "Show Nature Controls")

    show_executed_trajectory_toggle = Toggle(toggles_grid[4, 1], active=false)
    on(show_executed_trajectory_toggle.active) do active; show_executed_trajectory[] = active; end
    Label(toggles_grid[4, 1], "Show Executed Trajectory")

    show_robust_planned_trajectory_toggle = Toggle(toggles_grid[5, 1], active=true)
    on(show_robust_planned_trajectory_toggle.active) do active; show_robust_planned_trajectory[] = active; end
    Label(toggles_grid[5, 1], "Show Robust Planned Trajectory")

    show_non_robust_planned_trajectory_toggle = Toggle(toggles_grid[6, 1], active=false)
    on(show_non_robust_planned_trajectory_toggle.active) do active; show_non_robust_planned_trajectory[] = active; end
    Label(toggles_grid[6, 1], "Show Non-Robust Planned Trajectory")

    # --- Arrow Plotting ---

    # Draw arrows for robust activist controls
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id-1)*dims.num_senators + senator_id

            arrow_starts = @lift if $plan_time_step <= length($robust_means_trajectory) && belief_idx <= length(($robust_means_trajectory)[$plan_time_step].blocks)
                [Point2f(($robust_means_trajectory)[$plan_time_step][Block(belief_idx)])]
            else
                Point2f[]
            end

            arrow_vectors = @lift if $plan_time_step <= length($robust_planned_controls) && belief_idx <= length(($robust_planned_controls)[$plan_time_step].blocks)
                [Point2f(($robust_planned_controls)[$plan_time_step][Block(belief_idx)])]
            else
                Point2f[]
            end

            arrows!(ax, arrow_starts, arrow_vectors, color=colors[activist_id], visible=show_robust_activist_controls)
        end
    end

    # Draw arrows for non-robust activist controls
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id-1)*dims.num_senators + senator_id

            arrow_starts = @lift if $plan_time_step <= length($non_robust_means_trajectory) && belief_idx <= length(($non_robust_means_trajectory)[$plan_time_step].blocks)
                [Point2f(($non_robust_means_trajectory)[$plan_time_step][Block(belief_idx)])]
            else
                Point2f[]
            end

            arrow_vectors = @lift if $plan_time_step <= length($non_robust_planned_controls) && belief_idx <= length(($non_robust_planned_controls)[$plan_time_step].blocks)
                [Point2f(($non_robust_planned_controls)[$plan_time_step][Block(belief_idx)])]
            else
                Point2f[]
            end

            arrows!(ax, arrow_starts, arrow_vectors, color=colors[activist_id], visible=show_non_robust_activist_controls, linestyle=:dash)
        end
    end

    # Draw arrows for nature's controls #TODO: FIX nature controls
    is_robust = @lift if !isempty($robust_planned_controls) && !isempty($robust_planned_controls[1].blocks)
        length(($robust_planned_controls)[1]) > sum(dims.controls_per_activist)
    else
        false
    end
    for senator_id in 1:dims.num_senators
        arrow_starts = @lift if $plan_time_step <= length($robust_means_trajectory) && senator_id <= length(($robust_means_trajectory)[$plan_time_step].blocks)
            [Point2f(($robust_means_trajectory)[$plan_time_step][Block(senator_id)])]
        else
            Point2f[]
        end
        arrow_vectors = @lift if $is_robust && $plan_time_step <= length($robust_planned_controls)
            control_vec = ($robust_planned_controls)[$plan_time_step]
            last_block_idx = length(control_vec.blocks)
            nature_control_vec = control_vec[Block(last_block_idx)]
            if length(nature_control_vec) == sum(dims.states)
                nature_control_block = BlockVector(nature_control_vec, dims.states)
                if senator_id <= length(nature_control_block.blocks)
                    [Point2f(nature_control_block[Block(senator_id)])]
                else
                    Point2f[]
                end
            else
                Point2f[]
            end
        else
            Point2f[]
        end
        arrows!(ax, arrow_starts, arrow_vectors, color=:green, visible=show_nature_controls)
    end

    # Create observables and plots for covariance ellipses
    robust_ellipse_observables = []
    for activist_id in 1:dims.num_activists
        for _ in 1:dims.num_senators
            obs = Observable(Point2f[])
            lines!(ax, obs, color=colors[activist_id], visible=show_robust_planned_trajectory)
            push!(robust_ellipse_observables, obs)
        end
    end

    non_robust_ellipse_observables = []
    for activist_id in 1:dims.num_activists
        for _ in 1:dims.num_senators
            obs = Observable(Point2f[])
            lines!(ax, obs, color=colors[activist_id], visible=show_non_robust_planned_trajectory, linestyle=:dash)
            push!(non_robust_ellipse_observables, obs)
        end
    end

    robust_points = @lift begin
        if $plan_time_step > length($robust_means_trajectory) || isempty($robust_means_trajectory)
            Point2f[]
        else
            current_means = ($robust_means_trajectory)[$plan_time_step]
            pts = Point2f[]
            for activist_id in 1:dims.num_activists
                for senator_id in 1:dims.num_senators
                    belief_idx = (activist_id-1)*dims.num_senators + senator_id
                    if belief_idx <= length(current_means.blocks)
                        push!(pts, Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2]))
                    end
                end
            end
            pts
        end
    end
    scatter!(ax, robust_points, color=point_colors, markersize=8, visible=show_robust_planned_trajectory)
    non_robust_points = @lift begin
        if $plan_time_step > length($non_robust_means_trajectory) || isempty($non_robust_means_trajectory)
            Point2f[]
        else
            current_means = ($non_robust_means_trajectory)[$plan_time_step]
            pts = Point2f[]
            for activist_id in 1:dims.num_activists
                for senator_id in 1:dims.num_senators
                    belief_idx = (activist_id-1)*dims.num_senators + senator_id
                    if belief_idx <= length(current_means.blocks)
                        push!(pts, Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2]))
                    end
                end
            end
        end
        pts
    end
    scatter!(ax, non_robust_points, color=point_colors, markersize=8, visible=show_non_robust_planned_trajectory)


    robust_total_costs = [c.total for c in robust_costs]
    robust_terminal_costs = [c.terminal for c in robust_costs]
    robust_non_terminal_costs = [c.non_terminal for c in robust_costs]

    non_robust_total_costs = [c.total for c in non_robust_costs]
    non_robust_terminal_costs = [c.terminal for c in non_robust_costs]
    non_robust_non_terminal_costs = [c.non_terminal for c in non_robust_costs]

    
    add_multi_line_graph!(fig;
        series=[robust_total_costs, non_robust_total_costs],
        labels=["robust","non_robust"],
        current_time_step=current_time_step,
        title="Cost Over Time",
        ylabel="cost",
        # timesteps = 0:4,                     # optional custom x-axis
        scalarizer = to_scalar_cost,            # no-op for numbers; handy for cost structs
        loc=(1,2)                               # put it in column 2, row 1 (side graph)
    )

    add_multi_line_graph!(fig;
        series=[robust_terminal_costs, robust_non_terminal_costs, non_robust_terminal_costs, non_robust_non_terminal_costs],
        labels=["r terminal", "r non-terminal", "nr terminal", "nr non-terminal"],
        current_time_step=current_time_step,
        title="Component Cost Over Time",
        ylabel="cost",
        # timesteps = 0:4,                     # optional custom x-axis
        scalarizer = to_scalar_cost,            # no-op for numbers; handy for cost structs
        loc=(2,2)                               # put it in column 2, row 1 (side graph)
    )
    colsize!(fig.layout, 2, Relative(0.25))



    # Update ellipses on slider change
    on(plan_time_step) do time_step
        # Update robust ellipses
        if time_step <= length(robust_covariances_trajectory[]) && time_step <= length(robust_means_trajectory[])
            current_covs = robust_covariances_trajectory[][time_step]
            current_means = robust_means_trajectory[][time_step]
            plot_idx = 1
            for activist_id in 1:dims.num_activists
                for senator_id in 1:dims.num_senators
                    belief_idx = (activist_id - 1) * dims.num_senators + senator_id
                    if belief_idx <= length(current_means.blocks)
                        center = Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2])
                        cov = current_covs[belief_idx]
                        ellipse_points = get_ellipse_points(center, cov)
                        robust_ellipse_observables[plot_idx][] = ellipse_points
                        plot_idx += 1
                    end
                end
            end
        end

        # Update non-robust ellipses
        if time_step <= length(non_robust_covariances_trajectory[]) && time_step <= length(non_robust_means_trajectory[])
            current_covs = non_robust_covariances_trajectory[][time_step]
            current_means = non_robust_means_trajectory[][time_step]
            plot_idx = 1
            for activist_id in 1:dims.num_activists
                for senator_id in 1:dims.num_senators
                    belief_idx = (activist_id - 1) * dims.num_senators + senator_id
                    if belief_idx <= length(current_means.blocks)
                        center = Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2])
                        cov = current_covs[belief_idx]
                        ellipse_points = get_ellipse_points(center, cov)
                        non_robust_ellipse_observables[plot_idx][] = ellipse_points
                        plot_idx += 1
                    end
                end
            end
        end
    end
    
    # Also update ellipses when the main time slider changes
    # prev_time_step = Ref(current_time_step[])  # store previous time step

    on(current_time_step) do new_t
        # @infiltrate
        # Δt = new_t - prev_time_step[]
        # set_close_to!(plan_slider, clamp(plan_slider[] - Δt, 1, plan_slider_range))
        # prev_time_step[] = new_t               # update for next iteration
        set_close_to!(plan_slider, 1)
    end

    axislegend(ax)
end

# ---- Optional: convert complex "cost" objects to scalars ----
to_scalar_cost(c) = c isa Number ? float(c) :
                    c isa AbstractArray ? sum(skipmissing(vec(c))) :
                    c isa NamedTuple && hasproperty(c, :total) ? float(c.total) :
                    c isa AbstractDict && haskey(c, :total) ? float(c[:total]) :
                    try
                        float(getfield(c, :total))
                    catch
                        missing
                    end

"""
    add_multi_line_graph!(
        parent;
        series::Vector{<:AbstractVector},
        labels::Vector{<:AbstractString},
        current_time_step::Observable{Int}=Observable(typemax(Int)),
        title::AbstractString = "Series over time",
        xlabel::AbstractString = "t",
        ylabel::AbstractString = "value",
        timesteps::Union{Nothing,AbstractVector}=nothing,
        scalarizer::Function = identity,
        loc::Tuple{Int,Int} = (1, 2)
    ) -> Axis

Plot multiple time-aligned series on a single axis. Each element of `series` is a
vector of values at the same discrete timesteps. Use `scalarizer` (e.g., `to_scalar_cost`)
if your elements aren’t plain numbers.

- `parent`: either a `Figure` (an axis will be placed at `loc`) or an existing `Axis`.
- `current_time_step`: if you pass the same Observable you use for your main slider,
  the plot reveals points up to that step; otherwise it shows all points.
- `timesteps`: optional x-values (defaults to `1:N`).
- Returns the created/used `Axis`.
"""
function add_multi_line_graph!(parent;
    series::Vector,
    labels::Vector{<:AbstractString},
    current_time_step::Observable{Int}=Observable(typemax(Int)),
    title::AbstractString = "Series over time",
    xlabel::AbstractString = "t",
    ylabel::AbstractString = "value",
    timesteps::Union{Nothing,AbstractVector}=nothing,
    scalarizer::Function = identity,
    loc::Tuple{Int,Int} = (1, 2)
)
    @assert length(series) == length(labels) "series and labels must have same length"

    # Create or use an axis
    ax = parent isa Figure ? Axis(parent[loc...], title=title, xlabel=xlabel, ylabel=ylabel) :
                             (parent isa Axis ? parent :
                              error("parent must be a Figure or an Axis"))

    # Scalarize & sanitize each series, wrap in Observables for reactive updates
    obs_series = Vector{Observable{Vector{Float64}}}(undef, length(series))
    for i in eachindex(series)
        sc = scalarizer.(series[i])
        sc = collect(skipmissing(sc))
        obs_series[i] = Observable(Float64.(sc))
    end

    # All series should share the same time grid; we use the shortest length
    minlen() = minimum(length.([obs[] for obs in obs_series]))
    maxvalue() = maximum(vcat([obs[] for obs in obs_series]...))
    k = minlen()#@lift(clamp($current_time_step, 1, minlen()))

    # X values (shared)
    x_all = if timesteps === nothing
        @lift(1:$k)
    else
        xt = Observable(vec(timesteps))
        @lift(xt[][1:$k])
    end

    # Plot each series; capture local Observable in the loop for @lift closures
    for i in eachindex(obs_series)
        local oi = obs_series[i]
        lines!(ax, x_all, @lift(oi[][1:$k]), label=labels[i])
    end
    axislegend(ax, position=:rb)
    ylims!(ax,0, maxvalue()*1.1)
    xlims!(ax,0.5, k+3)
    # # Keep y-limits comfy as we reveal more points
    # on(current_time_step) do _
    #     k_now = min(current_time_step[], minlen())
    #     ys = Float64[]
    #     for oi in obs_series
    #         append!(ys, oi[][1:k_now])
    #     end
    #     if !isempty(ys)
    #         ymin, ymax = extrema(ys)
    #         pad = max(1e-9, 0.05 * (ymax - ymin + 1e-12))
    #         ax.ylimits = (ymin - pad, ymax + pad)
    #     end
    # end

    return ax
end
end
