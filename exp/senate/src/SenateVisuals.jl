module SenateVisuals

using GLMakie
using LinearAlgebra
using RobustBeliefGame
using BlockArrays
using Serialization
using Infiltrator

const senate_module_path = joinpath(@__DIR__, "Senate.jl")
if !isdefined(Main, :Senate)
    @eval Main begin
        include($senate_module_path)
    end
end
const Senate = Main.Senate

# Register UUID for deserialization at module load time
const senate_uuid = Base.UUID("a2515029-de12-424a-9371-454911e3b6f1")
const senate_pkgid = Base.PkgId(senate_uuid, "Senate")
if !haskey(Base.loaded_modules, senate_pkgid)
    Base.loaded_modules[senate_pkgid] = Senate
end

export visualize_receding_horizon_solution, load_solution

function load_solution(folder, filename, type = "mass_results")
    
    
    path = "exp/senate/outputs/$folder/$(filename)_$type.dat"
    results = open(deserialize, path, "r")
    experiments = Dict{String, Dict{String, Tuple{Dict, Dict, Main.Senate.SenateParams}}}()
    for exp_data in results
        params, fixed, trial_dict, exp_name = exp_data
        if filename == SubString(exp_name,1,length(filename))
            exp_name = SubString(exp_name,length(filename)+1,length(exp_name)) #Shorten to only parameters
        end
        experiments[exp_name] = Dict{String, Tuple{Dict, Dict, Main.Senate.SenateParams}}()
        for (trial_id, trial_data) in trial_dict
            solutions, games, cost_params = trial_data # ,Dict {player_idx -> RBG.BeliefGame}, SenateParams
            experiments[exp_name][trial_id] = (solutions, games, cost_params)
        end
    end
    visualize_receding_horizon_solution(experiments, filename)
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

function visualize_receding_horizon_solution(experiments::Dict, filename::String = "Test")
    screen = GLMakie.Screen()
    fig = Figure()
    ax = Axis(fig[1, 1], title="$filename graph", xlabel="Opinion Dimension 1", ylabel="Opinion Dimension 2", aspect=DataAspect())
    create_individual_solution_plot(fig, ax, experiments)
    display(screen, fig)
end

function create_individual_solution_plot(fig, ax, experiments::Dict)# sol_data, dims, cost_params)
    selection_state = Dict{String, Tuple{Observable{Bool}, Dict{String, Observable{Bool}}}}()    
    for exp_name in keys(experiments)
        trial_selections = Dict{String, Observable{Bool}}()
        for trial_id in keys(experiments[exp_name])
            trial_selections[trial_id] = Observable(false)
        end
        selection_state[exp_name] = (Observable(false), trial_selections)
    end

    # Set first experiment/trial selected to true
    selection_state[first(keys(experiments))][1][] = true
    selection_state[first(keys(experiments))][2][first(keys(experiments[first(keys(experiments))]))][] = true
    
    solutions, _, cost_params = experiments[first(keys(experiments))][first(keys(experiments[first(keys(experiments))]))]
    #@infiltrate
    dims = Senate.dims(cost_params) #SenateParams function
    #TODO get sol_data, dims, cost_params from experiments

    p1_solution_history = solutions[1].solution_history
    p2_solution_history = solutions[2].solution_history
    
    # --- Layout ---
    controls_grid = fig[2, 1] = GridLayout(tellwidth=false)
    colsize!(fig.layout, 1, Relative(0.75))
    current_time_step = Observable(1)
    plan_time_step = Observable(1)
    show_p1_planned_trajectory = Observable(true)
    show_p2_planned_trajectory = Observable(false)
    show_executed_trajectory = Observable(false)
    colors = [:blue, :red]

    # --- Sliders ---
    time_slider_grid = controls_grid[1, 1] = GridLayout(tellwidth=false)
    Label(time_slider_grid[1, 1], "Receding Horizon Time")
    time_steps = length(p1_solution_history)
    slider = Slider(time_slider_grid[2, 1], range = 1:time_steps, startvalue = 1)
    on(slider.value) do val; current_time_step[] = val; end
    Label(time_slider_grid[3, 1], @lift("$(Int($current_time_step))"))

    p1_planned_belief_trajectory = @lift(p1_solution_history[$current_time_step].beliefs)
    p2_planned_belief_trajectory = @lift(p2_solution_history[$current_time_step].beliefs)
    p1_planned_controls = @lift(p1_solution_history[$current_time_step].controls)
    p2_planned_controls = @lift(p2_solution_history[$current_time_step].controls)
    # Observables for planned trajectories at each receding horizon step `t`
    p1_means_trajectory = @lift [means(b) for b in $p1_planned_belief_trajectory]
    p2_means_trajectory = @lift [means(b) for b in $p2_planned_belief_trajectory]
    p1_covariances_trajectory = @lift [covs(b) for b in $p1_planned_belief_trajectory]
    p2_covariances_trajectory = @lift [covs(b) for b in $p2_planned_belief_trajectory]
    p1_planning_horizon = @lift length($p1_means_trajectory)
    p2_planning_horizon = @lift length($p2_means_trajectory)

    p1_costs = solutions[1].cost_history #Contains terminal, non-terminal, and total costs
    p2_costs = solutions[2].cost_history #Contains terminal, non-terminal, and total costs

    plan_slider_grid = controls_grid[2, 1] = GridLayout(tellwidth=false)
    Label(plan_slider_grid[1, 1], "Plan Time")
    plan_slider_range = @lift(1:($p1_planning_horizon > 0 ? $p1_planning_horizon : 1))
    plan_slider = Slider(plan_slider_grid[2, 1], range = plan_slider_range, startvalue = 1)
    on(plan_slider.value) do val; plan_time_step[] = val; end
    Label(plan_slider_grid[3, 1], @lift("$(Int($plan_time_step))"))

    # Plot activist preferences
    for activist_id in [1,2] #[non_robust_activist, robust_activist]
        params = cost_params.player_configs[activist_id]
        center = params.ellipsoid_centers[1]
        scale = params.ellipsoid_radii[1]
        a = sqrt(1 / scale[1])
        b = sqrt(1 / scale[2])
        plot_ellipse!(ax, center, a, b, label="Activist $activist_id Pref.", color=colors[activist_id])
    end

    point_colors = vcat([fill(c, dims.num_senators) for c in colors]...)

    #executed_trajectory holds the solved trajectory for each time_step for each senator, based on each activist (we only care about gt first state, which is repeated twice)
    for senator_id in 1:dims.num_senators
        senator_states = [p1_solution_history[time][1][1].beliefs[senator_id].belief_mean for time in eachindex(p1_solution_history)]
        gt_trajectory = [Point2f(state[1],state[2]) for state in senator_states]
        scatter!(ax, gt_trajectory, color=:green, markersize=8, visible=show_executed_trajectory)
        lines!(ax, gt_trajectory, color=:green, visible=show_executed_trajectory)

    end
    # Plot full planned trajectories as lines
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id-1)*dims.num_senators + senator_id
            
            p1_traj_points = @lift if $p1_planning_horizon > 0
                if belief_idx <= length(($p1_means_trajectory)[1].blocks)
                    [Point2f(($p1_means_trajectory)[time][Block(belief_idx)][1], ($p1_means_trajectory)[time][Block(belief_idx)][2]) for time in 1:($p1_planning_horizon)]
                else
                    Point2f[]
                end
            else
                Point2f[]
            end
            p2_traj_points = @lift if $p2_planning_horizon > 0
                if belief_idx <= length(($p2_means_trajectory)[1].blocks)
                    [Point2f(($p2_means_trajectory)[time][Block(belief_idx)][1], ($p2_means_trajectory)[time][Block(belief_idx)][2]) for time in 1:($p2_planning_horizon)]
                else
                    Point2f[]
                end
            else
                Point2f[]
            end
            lines!(ax, p1_traj_points, color=colors[activist_id], visible=show_p1_planned_trajectory)
            lines!(ax, p2_traj_points, color=colors[activist_id], visible=show_p2_planned_trajectory, linestyle=:dash)
        end
    end

    # --- Toggles ---
    toggles_grid = controls_grid[1, 2] = GridLayout(tellwidth=false)
    
    show_p1_activist_controls = Observable(false)
    p1_activist_toggle = Toggle(toggles_grid[1, 1], active=false)
    on(p1_activist_toggle.active) do active; show_p1_activist_controls[] = active; end
    Label(toggles_grid[1, 1], "Show Robust Activist Controls")

    show_p2_activist_controls = Observable(false)
    p2_activist_toggle = Toggle(toggles_grid[2, 1], active=false)
    on(p2_activist_toggle.active) do active; show_p2_activist_controls[] = active; end
    Label(toggles_grid[2, 1], "Show Non-Robust Activist Controls")

    show_nature_controls = Observable(false)
    nature_toggle = Toggle(toggles_grid[3, 1], active=false)
    on(nature_toggle.active) do active; show_nature_controls[] = active; end
    Label(toggles_grid[3, 1], "Show Nature Controls")

    show_executed_trajectory_toggle = Toggle(toggles_grid[4, 1], active=false)
    on(show_executed_trajectory_toggle.active) do active; show_executed_trajectory[] = active; end
    Label(toggles_grid[4, 1], "Show Executed Trajectory")

    show_p1_planned_trajectory_toggle = Toggle(toggles_grid[5, 1], active=true)
    on(show_p1_planned_trajectory_toggle.active) do active; show_p1_planned_trajectory[] = active; end
    Label(toggles_grid[5, 1], "Show Robust Planned Trajectory")

    show_p2_planned_trajectory_toggle = Toggle(toggles_grid[6, 1], active=false)
    on(show_p2_planned_trajectory_toggle.active) do active; show_p2_planned_trajectory[] = active; end
    Label(toggles_grid[6, 1], "Show Non-Robust Planned Trajectory")

    # --- Arrow Plotting ---

    # Draw arrows for robust activist controls
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id-1)*dims.num_senators + senator_id

            arrow_starts = @lift if $plan_time_step <= length($p1_means_trajectory) && belief_idx <= length(($p1_means_trajectory)[$plan_time_step].blocks)
                [Point2f(($p1_means_trajectory)[$plan_time_step][Block(belief_idx)])]
            else
                Point2f[]
            end

            arrow_vectors = @lift if $plan_time_step <= length($p1_planned_controls) && belief_idx <= length(($p1_planned_controls)[$plan_time_step].blocks)
                [Point2f(($p1_planned_controls)[$plan_time_step][Block(belief_idx)])]
            else
                Point2f[]
            end

            arrows!(ax, arrow_starts, arrow_vectors, color=colors[activist_id], visible=show_p1_activist_controls)
        end
    end

    # Draw arrows for non-robust activist controls
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id-1)*dims.num_senators + senator_id

            arrow_starts = @lift if $plan_time_step <= length($p2_means_trajectory) && belief_idx <= length(($p2_means_trajectory)[$plan_time_step].blocks)
                [Point2f(($p2_means_trajectory)[$plan_time_step][Block(belief_idx)])]
            else
                Point2f[]
            end

            arrow_vectors = @lift if $plan_time_step <= length($p2_planned_controls) && belief_idx <= length(($p2_planned_controls)[$plan_time_step].blocks)
                [Point2f(($p2_planned_controls)[$plan_time_step][Block(belief_idx)])]
            else
                Point2f[]
            end

            arrows!(ax, arrow_starts, arrow_vectors, color=colors[activist_id], visible=show_p2_activist_controls, linestyle=:dash)
        end
    end

    # Draw arrows for nature's controls #TODO: FIX nature controls
    is_robust = @lift if !isempty($p1_planned_controls) && !isempty($p1_planned_controls[1].blocks)
        length(($p1_planned_controls)[1]) > sum(dims.control_dims_per_activist)
    else
        false
    end
    for senator_id in 1:dims.num_senators
        arrow_starts = @lift if $plan_time_step <= length($p1_means_trajectory) && senator_id <= length(($p1_means_trajectory)[$plan_time_step].blocks)
            [Point2f(($p1_means_trajectory)[$plan_time_step][Block(senator_id)])]
        else
            Point2f[]
        end
        arrow_vectors = @lift if $is_robust && $plan_time_step <= length($p1_planned_controls)
            control_vec = ($p1_planned_controls)[$plan_time_step]
            last_block_idx = length(control_vec.blocks)
            nature_control_vec = control_vec[Block(last_block_idx)]
            if length(nature_control_vec) == sum(dims.state_dims_per_activist)
                nature_control_block = BlockVector(nature_control_vec, dims.state_dims_per_activist)
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
    #Make ellipses dashed depending on if activist is robust or non-robust, check cost_params.player_configs.type (is of type PlayerType)
    p1_ellipse_observables = []
    for activist_id in 1:dims.num_activists
        for _ in 1:dims.num_senators
            obs = Observable(Point2f[])
            lines!(ax, obs, color=colors[activist_id], visible=show_p1_planned_trajectory)
            push!(p1_ellipse_observables, obs)
        end
    end

    p2_ellipse_observables = []
    for activist_id in 1:dims.num_activists
        for _ in 1:dims.num_senators
            obs = Observable(Point2f[])
            lines!(ax, obs, color=colors[activist_id], visible=show_p2_planned_trajectory, linestyle=:dash)
            push!(p2_ellipse_observables, obs)
        end
    end

    p1_points = @lift begin
        if $plan_time_step > length($p1_means_trajectory) || isempty($p1_means_trajectory)
            Point2f[]
        else
            current_means = ($p1_means_trajectory)[$plan_time_step]
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
    scatter!(ax, p1_points, color=point_colors, markersize=8, visible=show_p1_planned_trajectory)
    p2_points = @lift begin
        if $plan_time_step > length($p2_means_trajectory) || isempty($p2_means_trajectory)
            Point2f[]
        else
            current_means = ($p2_means_trajectory)[$plan_time_step]
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
    scatter!(ax, p2_points, color=point_colors, markersize=8, visible=show_p2_planned_trajectory)


    p1_total_costs = [c.total for c in p1_costs]
    p1_terminal_costs = [c.terminal for c in p1_costs]
    p1_non_terminal_costs = [c.non_terminal for c in p1_costs]

    p2_total_costs = [c.total for c in p2_costs]
    p2_terminal_costs = [c.terminal for c in p2_costs]
    p2_non_terminal_costs = [c.non_terminal for c in p2_costs]

    
    add_multi_line_graph!(fig;
        series=[p1_total_costs, p2_total_costs],
        labels=["robust","non_robust"],
        current_time_step=current_time_step,
        title="Cost Over Time",
        ylabel="cost",
        # timesteps = 0:4,                     # optional custom x-axis
        scalarizer = to_scalar_cost,            # no-op for numbers; handy for cost structs
        loc=(1,2)                               # put it in column 2, row 1 (side graph)
    )

    add_multi_line_graph!(fig;
        series=[p1_terminal_costs, p1_non_terminal_costs, p2_terminal_costs, p2_non_terminal_costs],
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
        if time_step <= length(p1_covariances_trajectory[]) && time_step <= length(p1_means_trajectory[])
            current_covs = p1_covariances_trajectory[][time_step]
            current_means = p1_means_trajectory[][time_step]
            plot_idx = 1
            for activist_id in 1:dims.num_activists
                for senator_id in 1:dims.num_senators
                    belief_idx = (activist_id - 1) * dims.num_senators + senator_id
                    if belief_idx <= length(current_means.blocks)
                        center = Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2])
                        cov = current_covs[belief_idx]
                        ellipse_points = get_ellipse_points(center, cov)
                        p1_ellipse_observables[plot_idx][] = ellipse_points
                        plot_idx += 1
                    end
                end
            end
        end

        # Update non-robust ellipses
        if time_step <= length(p2_covariances_trajectory[]) && time_step <= length(p2_means_trajectory[])
            current_covs = p2_covariances_trajectory[][time_step]
            current_means = p2_means_trajectory[][time_step]
            plot_idx = 1
            for activist_id in 1:dims.num_activists
                for senator_id in 1:dims.num_senators
                    belief_idx = (activist_id - 1) * dims.num_senators + senator_id
                    if belief_idx <= length(current_means.blocks)
                        center = Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2])
                        cov = current_covs[belief_idx]
                        ellipse_points = get_ellipse_points(center, cov)
                        p2_ellipse_observables[plot_idx][] = ellipse_points
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
