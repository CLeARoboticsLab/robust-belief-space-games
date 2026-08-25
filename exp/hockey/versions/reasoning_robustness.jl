"""
    reasoning_robustness.jl

Quantifies *how badly the players model each other* in the hockey experiment, and how
much each player's plan churns between receding-horizon steps.

Motivation: the attacker (baseline / non-robust algorithm) appears to do worse against a
robust defender than against a non-robust one. In non-robust vs non-robust both players
model each other correctly; in non-robust vs robust both model each other incorrectly.
This script measures that mismodelling directly.

At receding-horizon step `t`, player `i` solves a game and obtains a planned control
sequence covering absolute times `t, t+1, ..., t+K-1` for *both* players. Only the first
entry is executed. So player `i`'s plan contains an explicit prediction of what player
`j` will do, and we can score it against what player `j` actually did.

    prediction  = solution_history[i][t].controls[k][block j]     # i thinks j will do this at t+k-1
    actual      = solution_history[j][t+k-1].controls[1][block j] # what j actually executed at t+k-1

Three error metrics, each shown two ways (6 figure sets total):

    L2         norm(pred - actual)
    cosine     1 - cosine_similarity(pred, actual)     (0 = same direction)
    magnitude  norm(pred) - norm(actual)               (signed; >0 = over-predicted effort)

    "over time"     -> aggregate across the planning horizon, one point per t
    "over horizon"  -> one panel per t, x-axis = planning-horizon index k

Plus a strategy-churn measure: how much player i's plan for a given absolute time changes
between the solve at t and the solve at t+1 (compared on the overlapping window).

## Usage

From `exp/hockey/` with `julia --project=.`:

    include("versions/reasoning_robustness.jl")

    # Current data: resolve the robust/non-robust pair out of an existing sweep
    run_reasoning_robustness_from_sweep(sweep_dir="outputs/sweep", ncc=3000.0)

    # Any other directories, e.g. the new-solver-cap re-run
    run_reasoning_robustness(
        robust_dir     = "outputs/sweep_newcap/<robust-config-dir>",
        non_robust_dir = "outputs/sweep_newcap/<non-robust-config-dir>",
        output_dir     = "analysis/reasoning_robustness/newcap",
    )

`max_trials` caps how many .jld2 files are read per arm (0 = all).
"""

using JLD2
using FileIO
using Statistics
using LinearAlgebra
using BlockArrays
using Logging
using Printf
using CairoMakie

# Needed so JLD2 can resolve HockeyParams / PlayerConfig when loading trial files.
using Hockey

const CTRL_DIM = 2                      # control dims per player (ax, ay)
const PLAYER_LABEL = Dict(1 => "attacker", 2 => "defender")

const COLOR_ROBUST = :steelblue
const COLOR_NON_ROBUST = :indianred

# ==============================================================================
# Loading
# ==============================================================================

quiet_load(path) = with_logger(NullLogger()) do
    load(path)
end

"""
    load_trials(dir; max_trials=0) -> Vector{NamedTuple}

Load every `.jld2` trial in `dir`. Returns entries with `solution_history`,
`gt_state_history`, `params` and the source `filename`. Files that fail to load or
that lack a solution history are skipped with a warning.
"""
function load_trials(dir::String; max_trials::Int=0)
    isdir(dir) || error("Directory not found: $dir")

    files = sort(filter(f -> endswith(f, ".jld2"), readdir(dir)))
    if max_trials > 0 && length(files) > max_trials
        files = files[1:max_trials]
    end

    trials = NamedTuple[]
    for f in files
        try
            d = quiet_load(joinpath(dir, f))
            haskey(d, "solution_history") || continue
            push!(trials, (;
                filename = f,
                solution_history = d["solution_history"],
                gt_state_history = get(d, "gt_state_history", nothing),
                params = get(d, "params", nothing),
            ))
        catch e
            @warn "Failed to load $f" exception = e
        end
    end

    isempty(trials) && error("No usable trials found in $dir")
    return trials
end

# ==============================================================================
# Plan / control accessors
# ==============================================================================

"""
    plan_controls(sh, i, t)

Planned control sequence produced by player `i` at receding-horizon step `t`.
Handles both the `(; beliefs, controls, ...)` NamedTuple format and the older
positional `(beliefs, controls)` tuple, matching `TrajectoryAnalysis`.
"""
function plan_controls(sh, i::Int, t::Int)
    entry = sh[i][t]
    return hasproperty(entry, :controls) ? entry.controls : entry[2]
end

num_rh_steps(sh) = minimum(length(sh[i]) for i in (1, 2))

"""
    player_block(u, j) -> Vector{Float64}

Player `j`'s slice of a stacked control vector. Uses linear indexing (rather than
`Block(j)`) so it is insensitive to how JLD2 restored the block structure. Robust plans
carry a trailing nature block, which this correctly ignores for j ∈ {1,2}.
"""
function player_block(u, j::Int)
    lo = (j - 1) * CTRL_DIM + 1
    hi = j * CTRL_DIM
    length(u) >= hi || return nothing
    return Vector{Float64}(u[lo:hi])
end

"""
    executed_controls(sh) -> Dict{Int, Vector{Vector{Float64}}}

The control each player actually executed at every step: the first entry of that
player's own plan, taking its own block. Same convention as
`TrajectoryAnalysis.extract_executed_controls`.
"""
function executed_controls(sh)
    T = num_rh_steps(sh)
    exec = Dict(1 => Vector{Float64}[], 2 => Vector{Float64}[])
    for j in (1, 2), t in 1:T
        controls = plan_controls(sh, j, t)
        isempty(controls) && continue
        u = player_block(controls[1], j)
        isnothing(u) || push!(exec[j], u)
    end
    return exec
end

cos_diff(a, b) = (norm(a) < 1e-10 || norm(b) < 1e-10) ? 0.0 : 1.0 - dot(a, b) / (norm(a) * norm(b))
mag_diff(a, b) = norm(a) - norm(b)

# ==============================================================================
# Prediction error: how well does player i model player j?
# ==============================================================================

"""
    prediction_errors(sh, i, j) -> Vector{NamedTuple}

For each receding-horizon step `t`, the per-horizon-index error between player `i`'s
prediction of player `j`'s controls and what `j` actually executed.

Returns one entry per `t`, each holding vectors indexed by planning-horizon step `k`
(absolute time `t + k - 1`).

Note: for `i == j` at `k == 1` the error is identically zero by construction — that
*is* the executed control. Self-prediction is only informative for `k > 1`.
"""
function prediction_errors(sh, i::Int, j::Int)
    T = num_rh_steps(sh)
    exec = executed_controls(sh)
    n_exec = length(exec[j])

    out = NamedTuple[]
    for t in 1:T
        controls = plan_controls(sh, i, t)
        l2, cd, md = Float64[], Float64[], Float64[]

        for k in eachindex(controls)
            abs_t = t + k - 1
            abs_t <= n_exec || break

            pred = player_block(controls[k], j)
            isnothing(pred) && break
            act = exec[j][abs_t]

            push!(l2, norm(pred .- act))
            push!(cd, cos_diff(pred, act))
            push!(md, mag_diff(pred, act))
        end

        push!(out, (; t, l2, cos = cd, mag = md))
    end
    return out
end

"""
    strategy_change(sh, i, j) -> Vector{NamedTuple}

How much player `i`'s plan for player `j` moves between consecutive solves. The plan at
`t` and the plan at `t+1` overlap on absolute times `t+1 …`; entry `k+1` of the former
and entry `k` of the latter refer to the same absolute time, so they are compared
directly.

Because each solve is warm-started from the previous plan (shifted by one step), this
measures how far the solver travels away from its warm start — i.e. replanning churn.
"""
function strategy_change(sh, i::Int, j::Int)
    T = num_rh_steps(sh)
    out = NamedTuple[]

    for t in 1:(T - 1)
        prev = plan_controls(sh, i, t)
        curr = plan_controls(sh, i, t + 1)
        l2, cd, md = Float64[], Float64[], Float64[]

        for k in 1:min(length(prev) - 1, length(curr))
            a = player_block(prev[k + 1], j)
            b = player_block(curr[k], j)
            (isnothing(a) || isnothing(b)) && break
            push!(l2, norm(a .- b))
            push!(cd, cos_diff(a, b))
            push!(md, mag_diff(a, b))
        end

        push!(out, (; t, l2, cos = cd, mag = md))
    end
    return out
end

# ==============================================================================
# Aggregation
# ==============================================================================

"""
    pointwise_stats(series) -> (idx, mean, std)

Mean and standard deviation across a ragged collection of vectors, computed
independently at each index over whichever series are long enough to reach it.
"""
function pointwise_stats(series::Vector{Vector{Float64}})
    series = filter(!isempty, series)
    isempty(series) && return (Int[], Float64[], Float64[])

    max_len = maximum(length, series)
    idx, mu, sd = Int[], Float64[], Float64[]
    for k in 1:max_len
        # NaN is skipped in place rather than compacted out of the series, so that
        # index k always refers to the same t / horizon step across trials.
        vals = [s[k] for s in series if length(s) >= k && !isnan(s[k])]
        isempty(vals) && continue
        push!(idx, k)
        push!(mu, mean(vals))
        push!(sd, length(vals) > 1 ? std(vals) : 0.0)
    end
    return (idx, mu, sd)
end

"""
    over_time(per_trial, metric; agg) -> Vector{Vector{Float64}}

Collapse each trial's per-`t` horizon vectors into a single number per `t`, giving one
time series per trial. `agg` is `sum` for L2 (the "total" the task asks for) or `mean`
for the scale-free metrics.
"""
function over_time(per_trial::Vector{Vector{NamedTuple}}, metric::Symbol; agg = mean)
    map(per_trial) do steps
        [isempty(getproperty(s, metric)) ? NaN : agg(getproperty(s, metric)) for s in steps]
    end
end

"""
    over_horizon(per_trial, metric, t) -> Vector{Vector{Float64}}

For a fixed receding-horizon step `t`, each trial's error profile across the planning
horizon.
"""
function over_horizon(per_trial::Vector{Vector{NamedTuple}}, metric::Symbol, t::Int)
    out = Vector{Float64}[]
    for steps in per_trial
        t <= length(steps) || continue
        v = getproperty(steps[t], metric)
        isempty(v) || push!(out, v)
    end
    return out
end

# ==============================================================================
# Plotting
# ==============================================================================

"""
One shared legend below the plot grid. Per-axis `axislegend` calls distort Makie's
automatic column sizing, letting the first panel dominate the figure width.
"""
function shared_legend!(fig, row::Int, labels)
    Legend(fig[row, :],
           [LineElement(color = COLOR_ROBUST, linewidth = 3),
            LineElement(color = COLOR_NON_ROBUST, linewidth = 3)],
           [labels[:robust], labels[:non_robust]],
           orientation = :horizontal, framevisible = false)
end

function band_series!(ax, series::Vector{Vector{Float64}}, color, label)
    idx, mu, sd = pointwise_stats(series)
    isempty(idx) && return
    band!(ax, idx, mu .- sd, mu .+ sd, color = (color, 0.18))
    lines!(ax, idx, mu, color = color, linewidth = 2.5, label = label)
    scatter!(ax, idx, mu, color = color, markersize = 7)
end

# `aggs` lists how each metric is collapsed across the planning horizon for the
# "over time" figures. L2 gets both: the total that the task asks for, and a
# per-horizon-step mean. The planning window shrinks at the end of the episode
# (4 controls for t ≤ 6, then 3, 2, 1), so a total that decays toward zero partly
# reflects having fewer terms to add up rather than better prediction; the mean
# is the length-invariant read.
const METRICS = (
    (key = :l2,  agg = sum,  aggs = (("total", sum), ("mean", mean)),
     name = "L2",        ylabel = "‖predicted − executed‖",       file = "l2"),
    (key = :cos, agg = mean, aggs = (("mean", mean),),
     name = "cosine",    ylabel = "1 − cos(predicted, executed)", file = "cosine"),
    (key = :mag, agg = mean, aggs = (("mean", mean),),
     name = "magnitude", ylabel = "‖predicted‖ − ‖executed‖",     file = "magnitude"),
)

"""
Figure set 1/3/5: one point per receding-horizon step `t`, aggregated across the
planning horizon. Both cross-prediction directions side by side.
"""
function plot_over_time(data, metric, output_dir, labels)
    nrows = length(metric.aggs)
    fig = Figure(size = (1250, 130 + 460 * nrows))
    Label(fig[0, :], "Mismodelling over time — $(metric.name)", fontsize = 18, font = :bold, tellwidth = false)

    for (row, (agg_name, agg_fn)) in enumerate(metric.aggs),
        (col, (i, j)) in enumerate(((1, 2), (2, 1)))

        ax = Axis(fig[row, col],
                  title = row == 1 ? "$(titlecase(PLAYER_LABEL[i])) predicting $(PLAYER_LABEL[j])" : "",
                  xlabel = row == nrows ? "receding-horizon step t" : "",
                  ylabel = "$agg_name  $(metric.ylabel)")
        metric.key == :mag && hlines!(ax, [0.0], color = :gray, linestyle = :dash)
        for (arm, color) in ((:robust, COLOR_ROBUST), (:non_robust, COLOR_NON_ROBUST))
            band_series!(ax, over_time(data[arm][(i, j)], metric.key; agg = agg_fn),
                         color, labels[arm])
        end
    end
    shared_legend!(fig, nrows + 1, labels)
    for c in 1:2
        colsize!(fig.layout, c, Relative(0.5))
    end

    path = joinpath(output_dir, "mismodelling_over_time_$(metric.file).png")
    save(path, fig)
    save(replace(path, ".png" => ".pdf"), fig)
    return path
end

"""
Figure set 2/4/6: one panel per receding-horizon step `t`, x-axis = planning-horizon
index. One figure per prediction direction.
"""
function plot_over_horizon(data, metric, i, j, output_dir, labels, n_steps)
    ncols = ceil(Int, sqrt(n_steps))
    nrows = ceil(Int, n_steps / ncols)

    fig = Figure(size = (400 * ncols, 320 * nrows + 130))
    Label(fig[0, :],
          "Mismodelling across the planning horizon — $(metric.name) — " *
          "$(titlecase(PLAYER_LABEL[i])) predicting $(PLAYER_LABEL[j])",
          fontsize = 18, font = :bold, tellwidth = false)

    for t in 1:n_steps
        r, c = fldmod1(t, ncols)
        ax = Axis(fig[r, c], title = "t = $t",
                  xlabel = r == nrows ? "planning-horizon step k" : "",
                  ylabel = c == 1 ? metric.ylabel : "",
                  xticks = 1:6)
        metric.key == :mag && hlines!(ax, [0.0], color = :gray, linestyle = :dash)

        for (arm, color) in ((:robust, COLOR_ROBUST), (:non_robust, COLOR_NON_ROBUST))
            band_series!(ax, over_horizon(data[arm][(i, j)], metric.key, t), color, labels[arm])
        end
    end

    shared_legend!(fig, nrows + 1, labels)
    for c in 1:ncols
        colsize!(fig.layout, c, Relative(1 / ncols))
    end

    path = joinpath(output_dir,
        "mismodelling_over_horizon_$(metric.file)_$(PLAYER_LABEL[i])_predicts_$(PLAYER_LABEL[j]).png")
    save(path, fig)
    save(replace(path, ".png" => ".pdf"), fig)
    return path
end

"""
Strategy churn: how far each solve moves from the previous one, per metric.
"""
function plot_strategy_change(churn, output_dir, labels)
    fig = Figure(size = (1250, 840))
    Label(fig[0, :], "Strategy change between consecutive solves (plan at t vs t+1)",
          fontsize = 18, font = :bold, tellwidth = false)

    for (row, metric) in enumerate(METRICS), (col, (i, j)) in enumerate(((1, 1), (2, 2)))
        agg_name = metric.agg === sum ? "total" : "mean"
        ax = Axis(fig[row, col],
                  title = row == 1 ? "$(titlecase(PLAYER_LABEL[i]))'s plan for itself" : "",
                  xlabel = row == length(METRICS) ? "receding-horizon step t" : "",
                  ylabel = "$agg_name $(metric.name)")
        metric.key == :mag && hlines!(ax, [0.0], color = :gray, linestyle = :dash)

        for (arm, color) in ((:robust, COLOR_ROBUST), (:non_robust, COLOR_NON_ROBUST))
            band_series!(ax, over_time(churn[arm][(i, j)], metric.key; agg = metric.agg),
                         color, labels[arm])
        end
    end
    shared_legend!(fig, length(METRICS) + 1, labels)
    for c in 1:2
        colsize!(fig.layout, c, Relative(0.5))
    end

    path = joinpath(output_dir, "strategy_change.png")
    save(path, fig)
    save(replace(path, ".png" => ".pdf"), fig)
    return path
end

# ==============================================================================
# Executed cost (tests the motivating premise directly)
# ==============================================================================

"""
    executed_costs_by_player(trials) -> Dict{Symbol, NamedTuple}

Executed-trajectory cost per trial for the attacker and the defender: the total, plus a
per-cost-component breakdown.

The existing sweep tooling only ever reports the *defender's* cost; the premise behind
this investigation is a claim about the *attacker*, so both are computed here. The
breakdown matters because the two players' totals live on very different scales — a
single dominant component can hide a real effect in the total. Returns empty vectors if
`TrajectoryAnalysis` is not loaded.
"""
function executed_costs_by_player(trials)
    out = Dict(p => (; total = Float64[], components = Dict{Symbol, Vector{Float64}}())
               for p in (:attacker, :defender))
    isdefined(Main, :TrajectoryAnalysis) || return out
    TA = Main.TrajectoryAnalysis

    for tr in trials
        (isnothing(tr.gt_state_history) || isnothing(tr.params)) && continue
        try
            entry = TA.TrajectoryAnalysisEntry(
                "temp", 0, tr.gt_state_history, [], tr.solution_history,
                [], [], [], tr.params, false, "temp")
            steps = TA.compute_executed_trajectory_costs(entry, false)

            for player in (:attacker, :defender)
                total = 0.0
                per_component = Dict{Symbol, Float64}()
                for s in steps
                    (haskey(s, player) && !isempty(s[player])) || continue
                    breakdown = s[player]
                    for name in keys(breakdown)
                        v = breakdown[name]
                        per_component[name] = get(per_component, name, 0.0) + v
                        total += v
                    end
                end
                push!(out[player].total, total)
                for (name, v) in per_component
                    push!(get!(out[player].components, name, Float64[]), v)
                end
            end
        catch e
            @warn "Cost computation failed for $(tr.filename)" exception = e
        end
    end
    return out
end

# ==============================================================================
# Report
# ==============================================================================

function write_report(io, data, churn, costs, labels, counts, dirs, n_steps)
    println(io, "Reasoning-Robustness Report")
    println(io, "="^78, "\n")
    println(io, "Robust arm:     $(dirs.robust)")
    println(io, "                $(counts.robust) trials")
    println(io, "Non-robust arm: $(dirs.non_robust)")
    println(io, "                $(counts.non_robust) trials")
    println(io, "Receding-horizon steps analysed: $n_steps\n")

    println(io, "PREMISE CHECK — executed cost per player")
    println(io, "-"^78)
    if isempty(costs[:robust][:attacker].total) && isempty(costs[:non_robust][:attacker].total)
        println(io, "(unavailable — include exp/TrajectoryAnalysis.jl before running to enable)\n")
    else
        # Cohen's d, so a difference can be read against trial-to-trial spread.
        function summarise(r, nr)
            (isempty(r) || isempty(nr)) && return ("n/a", "n/a", "")
            sr = length(r) > 1 ? std(r) : 0.0
            snr = length(nr) > 1 ? std(nr) : 0.0
            pooled = sqrt((sr^2 + snr^2) / 2)
            d = pooled > 1e-12 ? (mean(r) - mean(nr)) / pooled : 0.0
            (@sprintf("%11.4f ± %-9.4f", mean(r), sr),
             @sprintf("%11.4f ± %-9.4f", mean(nr), snr),
             @sprintf("%+9.4f  (d=%+.2f)", mean(r) - mean(nr), d))
        end

        @printf(io, "%-26s %-23s %-23s %s\n", "", labels[:robust], labels[:non_robust], "robust − non-robust")
        for player in (:attacker, :defender)
            rs, ns, ds = summarise(costs[:robust][player].total, costs[:non_robust][player].total)
            @printf(io, "%-26s %-23s %-23s %s\n", "$player TOTAL", rs, ns, ds)
            names = sort(collect(union(keys(costs[:robust][player].components),
                                       keys(costs[:non_robust][player].components))))
            for name in names
                r = get(costs[:robust][player].components, name, Float64[])
                nr = get(costs[:non_robust][player].components, name, Float64[])
                rs, ns, ds = summarise(r, nr)
                @printf(io, "%-26s %-23s %-23s %s\n", "    $name", rs, ns, ds)
            end
        end
        println(io, "\nPositive difference => that player pays more against the robust defender.")
        println(io, "d is Cohen's d; |d| < 0.2 is negligible against trial-to-trial spread.\n")
    end

    println(io, "MISMODELLING — averaged over all t and all planning-horizon steps")
    println(io, "-"^78)
    @printf(io, "%-34s %-16s %-16s\n", "prediction", labels[:robust], labels[:non_robust])
    for (i, j) in ((1, 2), (2, 1), (1, 1), (2, 2))
        tag = i == j ? "$(PLAYER_LABEL[i]) self (k>1)" : "$(PLAYER_LABEL[i]) → $(PLAYER_LABEL[j])"
        for metric in METRICS
            vals = Dict{Symbol, Float64}()
            for arm in (:robust, :non_robust)
                pooled = Float64[]
                for steps in data[arm][(i, j)], s in steps
                    v = getproperty(s, metric.key)
                    # self-prediction at k=1 is identically zero by construction
                    append!(pooled, i == j && length(v) > 1 ? v[2:end] : v)
                end
                vals[arm] = isempty(pooled) ? NaN : mean(pooled)
            end
            @printf(io, "%-34s %-16.5f %-16.5f\n",
                    "  $tag [$(metric.name)]", vals[:robust], vals[:non_robust])
        end
    end

    println(io, "\nSTRATEGY CHURN — mean change between consecutive solves")
    println(io, "-"^78)
    for (i, j) in ((1, 1), (2, 2))
        for metric in METRICS
            vals = Dict{Symbol, Float64}()
            for arm in (:robust, :non_robust)
                pooled = Float64[]
                for steps in churn[arm][(i, j)], s in steps
                    append!(pooled, getproperty(s, metric.key))
                end
                vals[arm] = isempty(pooled) ? NaN : mean(pooled)
            end
            @printf(io, "%-34s %-16.5f %-16.5f\n",
                    "  $(PLAYER_LABEL[i]) plan churn [$(metric.name)]", vals[:robust], vals[:non_robust])
        end
    end
end

# ==============================================================================
# Entry points
# ==============================================================================

"""
    run_reasoning_robustness(; robust_dir, non_robust_dir, output_dir, max_trials=0, labels...)

Run the full analysis on any pair of trial directories and write the figures and the
report to `output_dir`. Works on existing sweep data and on any fresh run.
"""
function run_reasoning_robustness(;
    robust_dir::String,
    non_robust_dir::String,
    output_dir::String,
    max_trials::Int = 0,
    robust_label::String = "Robust defender",
    non_robust_label::String = "Non-robust defender",
)
    CairoMakie.activate!()
    isdir(output_dir) || mkpath(output_dir)
    labels = Dict(:robust => robust_label, :non_robust => non_robust_label)

    println("Loading trials...")
    trials = Dict(:robust => load_trials(robust_dir; max_trials),
                  :non_robust => load_trials(non_robust_dir; max_trials))
    counts = (; robust = length(trials[:robust]), non_robust = length(trials[:non_robust]))
    println("  robust:     $(counts.robust) trials")
    println("  non-robust: $(counts.non_robust) trials")

    n_steps = minimum(num_rh_steps(tr.solution_history)
                      for arm in (:robust, :non_robust) for tr in trials[arm])
    println("Receding-horizon steps common to all trials: $n_steps")

    pairs = ((1, 2), (2, 1), (1, 1), (2, 2))
    println("Computing prediction errors and strategy churn...")
    data = Dict(arm => Dict(p => [prediction_errors(tr.solution_history, p...) for tr in trials[arm]]
                            for p in pairs) for arm in (:robust, :non_robust))
    churn = Dict(arm => Dict(p => [strategy_change(tr.solution_history, p...) for tr in trials[arm]]
                             for p in ((1, 1), (2, 2))) for arm in (:robust, :non_robust))

    println("Computing executed costs...")
    costs = Dict(arm => executed_costs_by_player(trials[arm]) for arm in (:robust, :non_robust))

    println("Rendering figures...")
    written = String[]
    for metric in METRICS
        push!(written, plot_over_time(data, metric, output_dir, labels))
        for (i, j) in ((1, 2), (2, 1))
            push!(written, plot_over_horizon(data, metric, i, j, output_dir, labels, n_steps))
        end
    end
    push!(written, plot_strategy_change(churn, output_dir, labels))

    report_path = joinpath(output_dir, "reasoning_robustness_report.txt")
    open(report_path, "w") do io
        write_report(io, data, churn, costs, labels, counts,
                     (; robust = robust_dir, non_robust = non_robust_dir), n_steps)
    end
    push!(written, report_path)

    println("\nWrote $(length(written)) files to $output_dir:")
    for p in written
        println("  ", basename(p))
    end
    println()
    print(read(report_path, String))

    return (; data, churn, costs, output_dir, n_steps)
end

"""
    find_config_pair(sweep_dir; ncc=3000.0, base_config=nothing) -> (robust_dir, non_robust_dir)

Locate a robust directory with the given `nature_control_cost_weight` and its matching
non-robust baseline, using the same naming convention as the rest of the sweep tooling
(`_ncc`/`_nbc` present => robust).

A full sweep contains thousands of directories sharing any given `ncc`, most of them
3-trial exploratory runs. When several match, the pair with the most trials wins — that
is the config actually under study. Pass `base_config` (the directory name with the
`_p2_nbc…`/`_p2_ncc…` tokens removed) to pin the choice explicitly.
"""
function find_config_pair(sweep_dir::String; ncc::Float64 = 3000.0, base_config = nothing)
    isdir(sweep_dir) || error("Sweep directory not found: $sweep_dir")
    dirs = filter(d -> isdir(joinpath(sweep_dir, d)), readdir(sweep_dir))

    is_robust(d) = occursin("_nbc", d) && occursin("_ncc", d)
    strip_nature(d) = replace(d, r"_p2_nbc[\d.]+" => "", r"_p2_ncc[\d.]+" => "")
    n_trials(d) = count(f -> endswith(f, ".jld2"), readdir(joinpath(sweep_dir, d)))

    ncc_matches(d) = begin
        m = match(r"p2_ncc([\d.]+)", d)
        isnothing(m) ? false : parse(Float64, rstrip(m.captures[1], '.')) == ncc
    end

    robust = filter(d -> is_robust(d) && ncc_matches(d), dirs)
    isnothing(base_config) || (robust = filter(d -> strip_nature(d) == base_config, robust))
    isempty(robust) && error("No robust directory with ncc=$ncc" *
                             (isnothing(base_config) ? "" : " and base $base_config") * " in $sweep_dir")

    # Pair each candidate with its baseline, then take the pair with the most data.
    candidates = map(robust) do r
        base = strip_nature(r)
        baseline = filter(d -> !is_robust(d) && strip_nature(d) == base, dirs)
        (; robust = r, baseline = isempty(baseline) ? nothing : only(baseline))
    end
    candidates = filter(c -> !isnothing(c.baseline), candidates)
    isempty(candidates) && error("No non-robust baseline found for any ncc=$ncc directory in $sweep_dir")

    best = argmax(c -> min(n_trials(c.robust), n_trials(c.baseline)), candidates)
    if length(candidates) > 1
        println("$(length(candidates)) candidate pairs at ncc=$ncc; " *
                "selected the one with the most trials " *
                "($(n_trials(best.robust)) robust / $(n_trials(best.baseline)) non-robust).")
    end

    return joinpath(sweep_dir, best.robust), joinpath(sweep_dir, best.baseline)
end

"""
    run_reasoning_robustness_from_sweep(; sweep_dir, ncc=3000.0, output_dir=nothing, max_trials=0)

Convenience wrapper: resolve the robust/non-robust pair out of a sweep directory, then
run the analysis. `output_dir` defaults to `analysis/reasoning_robustness/<robust dir>`.
"""
function run_reasoning_robustness_from_sweep(;
    sweep_dir::String = "outputs/sweep",
    ncc::Float64 = 3000.0,
    base_config = nothing,
    output_dir = nothing,
    max_trials::Int = 0,
)
    robust_dir, non_robust_dir = find_config_pair(sweep_dir; ncc, base_config)
    println("Robust arm:     $robust_dir")
    println("Non-robust arm: $non_robust_dir")

    out = isnothing(output_dir) ?
        joinpath("analysis", "reasoning_robustness", basename(robust_dir)) : output_dir

    return run_reasoning_robustness(;
        robust_dir, non_robust_dir, output_dir = out, max_trials,
        robust_label = "Robust defender (ncc=$ncc)",
        non_robust_label = "Non-robust defender")
end
