module SenateVisuals
using GLMakie
using LinearAlgebra
using Infiltrator
using Observables
using BlockArrays
using ..RobustBeliefGame: Beliefs, means, covs

export attach_to_senate_using_setup!, mark_run!, reset_viz!, visualize_receding_horizon_solution

const Point2 = Point2f

# --- helper: unwrap your covariance block (only the shape you said you have) ---
_cov2x2(Σblk) = Float32.(Matrix(Σblk[1,1]))  # Matrix{Symmetric{…Diagonal…}} → 2×2 Float32
function plot_ellipse!(ax, center::AbstractVector, a::Real, b::Real;
                       label::AbstractString = "", color = :black, n::Int = 200)
    cx = Float32(center[1]); cy = Float32(center[2])
    aa = Float32(a);         bb = Float32(b)
    θs = range(0f0, 2f0*pi; length = n)

    xs = cx .+ aa .* cos.(θs)
    ys = cy .+ bb .* sin.(θs)

    lines!(ax, xs, ys; label = label, color = color)
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
# Given the means/covs at `time_step`, return:
#   - `points`   :: Vector{Point2f}                (centers for scatter)
#   - `ellipses` :: Vector{Vector{Point2f}}        (polyline for each senator)
function step_geoms!(; means_t, covs_t, dims)
    points   = Point2f[]
    ellipses = Vector{Vector{Point2f}}()
    plot_idx = 1
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id - 1) * dims.num_senators + senator_id
            c = Point2f(means_t[Block(belief_idx)][1], means_t[Block(belief_idx)][2])
            push!(points, c)
            cov = covs_t[belief_idx]                # same indexing you already use
            push!(ellipses, get_ellipse_points(c, cov))
            plot_idx += 1
        end
    end
    return points, ellipses
end
# Map 1..num_activists to the key type used in `cost_params` (Int or Enum)
@inline function _activist_key(cost_params, i::Int)
    T = typeof(first(keys(cost_params)))
    if T <: Enum
        return instances(T)[i]         # e.g., ActivistID(1), ActivistID(2)
    else
        return i                        # Dict keyed by Ints
    end
end

function setup_solution_subplot!(ax, solution_tuple, t, colors, cost_params, dims, point_colors)
    belief_trajectory = solution_tuple.solution_history[1][1]  # Extract beliefs from tuple
    time_steps = length(belief_trajectory) 
    # Plot activist preferences
    for activist_id in 1:dims.num_activists - 1 #Minus nature
        key    = _activist_key(cost_params, activist_id)
        params = cost_params[key]
        @infiltrate
        center = params.pos[1]
        scale = params.scale[1]
        a = sqrt(1 / scale[1])
        b = sqrt(1 / scale[2])
        plot_ellipse!(ax, center, a, b, label="Activist $activist_id Pref.", color=colors[activist_id])
    end

    # Extract trajectories
    means_trajectory = [means(b) for b in belief_trajectory]
    covariances_trajectory = [covs(b) for b in belief_trajectory]

    centers_per_time = Vector{Vector{Point2f}}(undef, time_steps)
    for τ in 1:time_steps
        centers_per_time[τ], _ = step_geoms!(; means_t = means_trajectory[τ],
                                                covs_t  = covariances_trajectory[τ],
                                                dims    = dims)
    end
    
    # Plot full trajectories as lines
    for activist_id in 1:dims.num_activists
        for senator_id in 1:dims.num_senators
            belief_idx = (activist_id - 1) * dims.num_senators + senator_id
            traj_points = Point2f[ centers_per_time[τ][belief_idx] for τ in 1:time_steps ]
            lines!(ax, traj_points, color = colors[activist_id])
        end
    end

    # Create observables for covariance ellipses
    ellipse_observables = []
    for activist_id in 1:dims.num_activists
        for _ in 1:dims.num_senators
            obs = Observable(Point2f[])
            lines!(ax, obs, color=colors[activist_id])
            push!(ellipse_observables, obs)
        end
    end

    # Scatter points (observable)
    points = @lift begin
        if isempty(means_trajectory) || $t == 0
            Point2f[]                      # nothing to draw yet
        else
            k = min($t, length(means_trajectory))
            pts, _ = step_geoms!(; means_t = means_trajectory[k],
                                    covs_t  = covariances_trajectory[k],
                                    dims    = dims)
            pts
        end
    end
    scatter!(ax, points, color = point_colors, markersize = 8)


    return ellipse_observables, means_trajectory, covariances_trajectory
end

function visualize_receding_horizon_solution(solutions, games; dims, non_robust_key="non_robust", robust_key="robust")
    
    cost_params = Dict(
        1 => (;pos = [[1,1]], scale = [[1,2]]),
        2 => (;pos = [[3,0]], scale = [[2,1]]),
    )
    colors = [:blue, :red]
    point_colors = vcat([fill(c, dims.num_senators) for c in colors]...)

    fig = Figure(size = (1200, 800))

    # Time slider
    belief_trajectory = solutions[non_robust_key][1]
    time_steps = length(belief_trajectory)
    slider = Slider(fig[2, 1:2], range = 1:time_steps, startvalue = 1)
    t = slider.value

   # Non-robust plot
    ax1 = Axis(fig[1, 1], title="Non-Robust Solution", xlabel="Opinion Dimension 1", ylabel="Opinion Dimension 2", aspect=DataAspect())
    ellipse_obs1, means_traj1, covs_traj1 = setup_solution_subplot!(ax1, solutions[non_robust_key], t, colors, cost_params, dims, point_colors)

    # Robust plot
    ax2 = Axis(fig[1, 2], title="Robust Solution", xlabel="Opinion Dimension 1", aspect=DataAspect())
    ellipse_obs2, means_traj2, covs_traj2 = setup_solution_subplot!(ax2, solutions[robust_key], t, colors, cost_params, dims, point_colors)
    # Update ellipses on slider change
    on(t) do time_step

        # skip early calls (before first frame)
        if time_step == 0 || isempty(means_traj1)
            return
        end

        for (ellipse_observables, means_trajectory, covariances_trajectory) in [
            (ellipse_obs1, means_traj1, covs_traj1),
            (ellipse_obs2, means_traj2, covs_traj2)
        ]
            current_covs = covariances_trajectory[time_step]
            current_means = means_trajectory[time_step]
            plot_idx = 1
            for activist_id in 1:dims.num_activists
                for senator_id in 1:dims.num_senators
                    belief_idx = (activist_id-1)*dims.num_senators + senator_id
                    center = Point2f(current_means[Block(belief_idx)][1], current_means[Block(belief_idx)][2])
                    cov = current_covs[belief_idx]
                    ellipse_observables[plot_idx][] = get_ellipse_points(center, cov)
                    plot_idx += 1
                end
            end
        end
    end
    t[] = t[] # Trigger initial plot

    axislegend(ax1)
    axislegend(ax2)
    display(fig)
    fig
end
# ------------------------
# Utility conversions
# ------------------------
const _Point2 = Makie.Point{2,Float32}
_to_p2(x):: _Point2 = _Point2(Float32(x[1]), Float32(x[2]))
_to_p2pts(xs)::Vector{_Point2} = [_to_p2(p) for p in xs]

# ------------------------
# Lightweight viz state
# ------------------------
Base.@kwdef mutable struct SenateViz
    fig::Figure
    ax::Axis
    centers_obs::Observable{Vector{_Point2}} = Observable(_Point2[])
    ellipse_obs::Vector{Observable{Vector{_Point2}}}
    traj_obs::Vector{Observable{Vector{_Point2}}}
    pref_handles::Vector{Any}
    run_label_obs::Observable{String} = Observable("")
    colors::Vector{Symbol}
    point_colors::Vector{Symbol}
    dims::NamedTuple
    show_controls::Bool = false
    ctrl_handles::Vector{Any} = Any[]
end

# ------------------------
# Public API (used by your driver)
# ------------------------
function attach_to_senate_using_setup!(; dims, cost_params,
        colors::Vector{Symbol} = [:cornflowerblue, :orangered],
        neutral::Symbol = :gray60,
        show_controls::Bool = false,
        resolution::Tuple{Int,Int} = (900, 650),
    )
    fig = Figure(size = resolution)
    ax  = Axis(fig[1,1], title = "Beliefs (live)", aspect = DataAspect())

    # Scatter colors: repeat each activist color per senator
    point_colors = vcat([fill(colors[a], dims.num_senators) for a in 1:dims.num_activists]...)
    N = dims.num_activists * dims.num_senators

    # Use NaN points so nothing shows, but lengths match colors
    centers_obs = Observable([Point2f(NaN, NaN) for _ in 1:N])

    scatter!(ax, centers_obs; color = point_colors, markersize = 8)

    # Preallocate ellipse observables (one per (activist, senator))
    ellipse_obs = [Observable(_Point2[]) for _ in 1:N]
    for i in 1:N
        a = Int(fldmod1(i, dims.num_senators)[2] == 0 ? ceil(i / dims.num_senators) : ceil(i / dims.num_senators))
        # safer mapping: a = (i - 1) ÷ dims.num_senators + 1
        a = (i - 1) ÷ dims.num_senators + 1
        lines!(ax, ellipse_obs[i]; color = colors[a], linewidth = 1.5, transparency = true)
    end

    # Optional trajectories per (activist, senator)
    traj_obs = [Observable(_Point2[]) for _ in 1:N]
    for i in 1:N
        lines!(ax, traj_obs[i]; color = :black, linestyle = :dot, linewidth = 1)
    end

    # Static overlays: activist preference ellipses (draw once)
    pref_handles = _draw_preference_ellipses!(ax, cost_params; colors, dims, neutral)

    # Run label (top-right)
    run_label_obs = Observable("")
    text!(ax, run_label_obs; position = Point(1,1), space = :relative,
          align = (:right, :top), fontsize = 14, color = :gray30)

    viz = SenateViz(; fig, ax, centers_obs, ellipse_obs, traj_obs,
        pref_handles, run_label_obs, colors, point_colors, dims,
        show_controls, ctrl_handles = Any[])

    # Callback used by the solver loop (on_live_step)
    on_step = (t; beliefs, gt_state = nothing, u_non_robust = nothing, u_robust = nothing) ->
        _update_live!(viz, t; beliefs, gt_state, u_non_robust, u_robust)

    display(fig)
    return viz, on_step
end

function mark_run!(viz::SenateViz, label::AbstractString)
    viz.run_label_obs[] = String(label)
    _clear_controls!(viz)
    return nothing
end

function reset_viz!(viz::SenateViz)
    N = viz.dims.num_activists * viz.dims.num_senators
    viz.centers_obs[] = [Point2f(NaN, NaN) for _ in 1:N]   # keep lengths aligned
    for obs in viz.ellipse_obs
        obs[] = _Point2[]
    end
    for obs in viz.traj_obs
        obs[] = _Point2[]
    end
    _clear_controls!(viz)
    return nothing
end

# ------------------------
# Internal: preference ellipses from cost_params (draw once)
# ------------------------
function _draw_preference_ellipses!(ax::Axis, cost_params; colors, dims, neutral)
    handles = Any[]

    # Try to find per-activist entries; we accept any keys and draw if (:pos, :scale) exist
    _maybe_draw = function (cp, color)
        if cp === nothing
            return
        end
        haspos  = hasproperty(cp, :pos)  || (cp isa NamedTuple && haskey(cp, :pos))
        hascale = hasproperty(cp, :scale)|| (cp isa NamedTuple && haskey(cp, :scale))
        if !(haspos && hascale)
            return
        end
        pos   = cp[:pos]   isa Function ? cp.pos()   : get(cp, :pos, nothing)
        scale = cp[:scale] isa Function ? cp.scale() : get(cp, :scale, nothing)
        pos === nothing && return
        scale === nothing && return

        ns = min(length(pos), dims.num_senators)
        for s in 1:ns
            c = pos[s]
            S = scale[s]
            Σ = _as_cov(S)             # Accept 2×2 or two-length scale
            pts = get_ellipse_points(c, Σ)  # Reuse your existing helper
            push!(handles, lines!(ax, _to_p2pts(pts); color = color, linewidth = 1.2))
        end
    end

    # Iterate cost_params; try to map colors to first num_activists entries
    ks = collect(keys(cost_params))
    for (i, k) in enumerate(ks[1:min(end, dims.num_activists)])
        _maybe_draw(cost_params[k], colors[i])
    end

    return handles
end

_as_cov(S) = (S isa AbstractMatrix ? S : Diagonal(map(float, S)))

# ------------------------
# Internal: per-step updater (called by on_live_step)
# ------------------------
function _update_live!(viz::SenateViz, t;
        beliefs,
        gt_state = nothing,
        u_non_robust = nothing,
        u_robust     = nothing)

    # Reuse your geometry core for both static and live paths
    μt = means(beliefs)
    Σt = covs(beliefs)
    centers, ellipses = step_geoms!(; means_t = μt, covs_t = Σt, dims = viz.dims)

    # Centers
    viz.centers_obs[] = _to_p2pts(centers)

    # Ellipses & trajectories (preallocated observables)
    @inbounds for i in eachindex(viz.ellipse_obs)
        viz.ellipse_obs[i][] = _to_p2pts(ellipses[i])
        # Append to trajectory (no extra allocations beyond the new point)
        tr = viz.traj_obs[i][]
        push!(tr, _to_p2(centers[i]))
        viz.traj_obs[i][] = tr
    end

    # Optional: draw control arrows per step (kept tiny & togglable)
    if viz.show_controls && (u_non_robust !== nothing || u_robust !== nothing)
        _draw_control_arrows!(viz, _to_p2pts(centers), u_non_robust, u_robust)
    end

    return nothing
end

# ------------------------
# Internal: control arrows (optional)
# ------------------------
function _draw_control_arrows!(viz::SenateViz, centers::Vector{_Point2}, u_nr, u_r)
    _clear_controls!(viz)

    dims = viz.dims
    has_nr = u_nr !== nothing
    has_r  = u_r  !== nothing

    for a in 1:dims.num_activists
        for s in 1:dims.num_senators
            i = (a-1) * dims.num_senators + s
            c = centers[i]
            if a == 1 && has_nr
                u  = u_nr[s]
                p2 = _Point2(c[1] + Float32(u[1]), c[2] + Float32(u[2]))
                push!(viz.ctrl_handles, arrows!(viz.ax, [c], [p2 - c]; color = viz.colors[a], linewidth = 2))
            elseif a == 2 && has_r
                u  = u_r[s]
                p2 = _Point2(c[1] + Float32(u[1]), c[2] + Float32(u[2]))
                push!(viz.ctrl_handles, arrows!(viz.ax, [c], [p2 - c]; color = viz.colors[a], linewidth = 2))
            end
        end
    end
end

function _clear_controls!(viz::SenateViz)
    if !isempty(viz.ctrl_handles)
        for h in viz.ctrl_handles
            try delete!(h) catch; end
        end
        empty!(viz.ctrl_handles)
    end
    nothing
end
end #module
