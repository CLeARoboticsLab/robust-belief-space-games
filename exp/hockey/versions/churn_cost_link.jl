"""
    churn_cost_link.jl

Does lower plan-churn / higher predictability actually *buy* lower cost, or do the two
merely travel together?

Three tiers of evidence, weakest to strongest:

  1. WITHIN-ARM CORRELATION. Across the trials of a single arm, does a trial that
     happened to churn less also cost less? Confounded: churn and cost are both outputs
     of the same solve, and a "hard" noise draw can raise both with no causal arrow
     between them.

  2. SEED-PAIRED (COMMON RANDOM NUMBERS). Trial number sets the RNG seed
     (`random_seed + trial - 1`) and the solver itself draws no randomness — only the
     receding-horizon loop does, twice per step in fixed order. So robust trial k and
     non-robust trial k see the *identical* noise sequence. Differencing within a seed
     (Δchurn vs Δcost) removes the noise realization exactly, which is the main
     confounder in tier 1.

  3. NCC DOSE-RESPONSE. `nature_control_cost_weight` is set by us, not by nature, and
     sweeping it moves the defender along a conservatism axis. If churn mediates cost,
     cost should track churn monotonically across ncc levels. This is the closest thing
     to a causal handle available from existing data.

## What none of these can establish

Nothing here *intervenes* on churn. Even tier 3 confounds churn with everything else ncc
changes (control effort, aggressiveness, how nature is priced). And churn is partly a
function of the executed controls — the plan at t+1 begins with the control executed at
t+1 — so regressing cost on churn is partly regressing a quantity on a transform of
itself. To limit that circularity the headline outcome is the *interaction* cost
(`shot_prob + steal_prob`, the exactly zero-sum, position-driven part) rather than
`control_effort`, which shares variables with churn directly.

A clean causal test needs an exogenous handle on churn: add a trust-region / plan-
smoothing penalty (‖uₜ₊₁ − uₜ‖ across consecutive solves) to the *solver*, leaving the
cost function alone, and sweep its weight. Then churn is manipulated directly and cost
is the response. That is a new experiment, not an analysis of existing runs.

## Usage

From `exp/hockey/` with `julia --project=.`:

    include("../TrajectoryAnalysis.jl"); using .TrajectoryAnalysis
    include("versions/reasoning_robustness.jl")
    include("versions/churn_cost_link.jl")

    run_churn_cost_link(sweep_dir="outputs/sweep", ncc=3000.0)          # tiers 1 + 2
    run_churn_cost_link(sweep_dir="outputs/sweep", ncc=3000.0,
                        dose_ncc=[10.0,100.0,300.0,1000.0,3000.0,10000.0,30000.0])  # + tier 3
"""

using Statistics
using LinearAlgebra
using Printf
using Distributions
using CairoMakie

# ==============================================================================
# Per-trial scalar summaries
# ==============================================================================

flatten_metric(steps, key) =
    isempty(steps) ? Float64[] : reduce(vcat, [getproperty(s, key) for s in steps]; init = Float64[])

safe_mean(v) = isempty(v) ? NaN : mean(v)

"""
    trial_index(filename) -> Int

Trial number parsed out of `..._trial_<n>.jld2`. This doubles as the RNG seed offset,
which is what makes the paired analysis valid.
"""
function trial_index(filename::AbstractString)
    m = match(r"_trial_(\d+)\.jld2$", filename)
    isnothing(m) ? -1 : parse(Int, m.captures[1])
end

"""
    trial_scalars(tr, TA) -> NamedTuple

Collapse one trial into the scalars used for correlation: churn and prediction-error
predictors, plus executed-cost outcomes broken out by component.
"""
function trial_scalars(tr, TA)
    sh = tr.solution_history

    churn_a = strategy_change(sh, 1, 1)
    churn_d = strategy_change(sh, 2, 2)
    pred_ad = prediction_errors(sh, 1, 2)   # attacker predicting defender
    pred_da = prediction_errors(sh, 2, 1)   # defender predicting attacker

    costs = Dict{Symbol, Dict{Symbol, Float64}}(:attacker => Dict(), :defender => Dict())
    if !isnothing(tr.gt_state_history) && !isnothing(tr.params)
        try
            entry = TA.TrajectoryAnalysisEntry("temp", 0, tr.gt_state_history, [],
                                               sh, [], [], [], tr.params, false, "temp")
            for s in TA.compute_executed_trajectory_costs(entry, false)
                for player in (:attacker, :defender)
                    (haskey(s, player) && !isempty(s[player])) || continue
                    for name in keys(s[player])
                        costs[player][name] = get(costs[player], name, 0.0) + s[player][name]
                    end
                end
            end
        catch e
            @warn "cost failed for $(tr.filename)" exception = e
        end
    end

    comp(player, name) = get(costs[player], name, NaN)
    interaction(player) = comp(player, :shot_prob) + comp(player, :steal_prob)

    return (;
        trial = trial_index(tr.filename),
        # predictors
        churn_def_l2   = safe_mean(flatten_metric(churn_d, :l2)),
        churn_def_cos  = safe_mean(flatten_metric(churn_d, :cos)),
        churn_atk_l2   = safe_mean(flatten_metric(churn_a, :l2)),
        predictability_of_def = safe_mean(flatten_metric(pred_ad, :l2)),  # lower = defender easier to predict
        pred_err_of_atk       = safe_mean(flatten_metric(pred_da, :l2)),
        # outcomes
        cost_def_total   = isempty(costs[:defender]) ? NaN : sum(values(costs[:defender])),
        cost_def_inter   = interaction(:defender),
        cost_def_effort  = comp(:defender, :control_effort),
        cost_atk_inter   = interaction(:attacker),
    )
end

# ==============================================================================
# Statistics
# ==============================================================================

ordinal_rank(x) = invperm(sortperm(x))

"""
    corr_with_p(x, y) -> (r, p, n)

Pearson correlation with a two-sided p-value from the usual t transform. Pairs where
either value is NaN are dropped.
"""
function corr_with_p(x::Vector{Float64}, y::Vector{Float64})
    keep = findall(i -> !isnan(x[i]) && !isnan(y[i]), eachindex(x))
    n = length(keep)
    n < 3 && return (NaN, NaN, n)
    xs, ys = x[keep], y[keep]
    (std(xs) < 1e-12 || std(ys) < 1e-12) && return (NaN, NaN, n)
    r = cor(xs, ys)
    abs(r) >= 1 && return (r, 0.0, n)
    t = r * sqrt((n - 2) / (1 - r^2))
    p = 2 * (1 - cdf(TDist(n - 2), abs(t)))
    return (r, p, n)
end

function spearman_with_p(x::Vector{Float64}, y::Vector{Float64})
    keep = findall(i -> !isnan(x[i]) && !isnan(y[i]), eachindex(x))
    length(keep) < 3 && return (NaN, NaN, length(keep))
    corr_with_p(Float64.(ordinal_rank(x[keep])), Float64.(ordinal_rank(y[keep])))
end

const PREDICTORS = (
    (:churn_def_l2,          "defender plan churn (L2)"),
    (:churn_def_cos,         "defender plan churn (cosine)"),
    (:churn_atk_l2,          "attacker plan churn (L2)"),
    (:predictability_of_def, "defender mispredicted by attacker (L2)"),
    (:pred_err_of_atk,       "attacker mispredicted by defender (L2)"),
)

const OUTCOMES = (
    (:cost_def_inter,  "defender interaction cost (zero-sum)"),
    (:cost_def_total,  "defender total cost"),
    (:cost_def_effort, "defender control effort  [shares variables w/ churn]"),
)

col(rows, key) = Float64[getproperty(r, key) for r in rows]

function correlation_table(io, rows, title)
    println(io, title)
    println(io, "-"^100)
    @printf(io, "%-42s %-30s %9s %9s %7s\n", "predictor", "outcome", "pearson", "spearman", "n")
    for (pkey, plabel) in PREDICTORS, (okey, olabel) in OUTCOMES
        x, y = col(rows, pkey), col(rows, okey)
        r, pr, n = corr_with_p(x, y)
        ρ, pρ, _ = spearman_with_p(x, y)
        star(p) = isnan(p) ? " " : p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : " "
        @printf(io, "%-42s %-30s %+6.3f%-3s %+6.3f%-3s %7d\n",
                plabel, olabel, r, star(pr), ρ, star(pρ), n)
    end
    println(io)
end

# ==============================================================================
# Tier 2: seed-paired differences
# ==============================================================================

"""
    paired_rows(robust_rows, non_robust_rows) -> Vector{NamedTuple}

Match trials by index (= RNG seed) and return robust-minus-non-robust differences.
Because the solver draws no randomness, both arms of a matched pair experienced the
identical process/sensor-noise sequence, so the difference is free of that confounder.
"""
function paired_rows(robust_rows, non_robust_rows)
    nr_by_trial = Dict(r.trial => r for r in non_robust_rows)
    out = NamedTuple[]
    for r in robust_rows
        haskey(nr_by_trial, r.trial) || continue
        nr = nr_by_trial[r.trial]
        push!(out, (;
            trial = r.trial,
            (k => getproperty(r, k) - getproperty(nr, k)
             for k in (first.(PREDICTORS)..., first.(OUTCOMES)...))...,
        ))
    end
    return out
end

# ==============================================================================
# Plotting
# ==============================================================================

function scatter_panel!(ax, x, y, color)
    keep = findall(i -> !isnan(x[i]) && !isnan(y[i]), eachindex(x))
    isempty(keep) && return
    xs, ys = x[keep], y[keep]
    scatter!(ax, xs, ys, color = (color, 0.65), markersize = 9)
    if length(xs) > 2 && std(xs) > 1e-12
        b = cov(xs, ys) / var(xs)
        a = mean(ys) - b * mean(xs)
        xr = [minimum(xs), maximum(xs)]
        lines!(ax, xr, a .+ b .* xr, color = color, linewidth = 2.5)
        r, p, _ = corr_with_p(xs, ys)
        text!(ax, 0.03, 0.95, text = @sprintf("r=%+.2f  p=%.3g", r, p),
              space = :relative, align = (:left, :top), fontsize = 13)
    end
end

function plot_scatter_grid(rows_by_arm, output_dir, labels, filename, suptitle)
    preds = PREDICTORS[1:4]
    fig = Figure(size = (420 * length(preds), 760))
    Label(fig[0, :], suptitle, fontsize = 18, font = :bold, tellwidth = false)

    for (col_i, (pkey, plabel)) in enumerate(preds),
        (row_i, (okey, olabel)) in enumerate(OUTCOMES[1:2])

        ax = Axis(fig[row_i, col_i],
                  xlabel = row_i == 2 ? plabel : "",
                  ylabel = col_i == 1 ? olabel : "",
                  titlesize = 13)
        for (arm, c) in ((:robust, COLOR_ROBUST), (:non_robust, COLOR_NON_ROBUST))
            haskey(rows_by_arm, arm) || continue
            scatter_panel!(ax, col(rows_by_arm[arm], pkey), col(rows_by_arm[arm], okey), c)
        end
    end
    if length(rows_by_arm) > 1
        shared_legend!(fig, 3, labels)
    end
    for c in 1:length(preds)
        colsize!(fig.layout, c, Relative(1 / length(preds)))
    end

    path = joinpath(output_dir, filename)
    save(path, fig)
    save(replace(path, ".png" => ".pdf"), fig)
    return path
end

function plot_dose_response(levels, output_dir)
    isempty(levels) && return nothing
    nccs = Float64[l.ncc for l in levels]

    fig = Figure(size = (1500, 470))
    Label(fig[0, :], "NCC dose-response — does cost track churn as the knob moves?",
          fontsize = 18, font = :bold, tellwidth = false)

    ax1 = Axis(fig[1, 1], xlabel = "nature_control_cost_weight", ylabel = "defender plan churn (L2)",
               xscale = log10, title = "churn vs knob")
    ax2 = Axis(fig[1, 2], xlabel = "nature_control_cost_weight", ylabel = "defender interaction cost",
               xscale = log10, title = "cost vs knob")
    ax3 = Axis(fig[1, 3], xlabel = "defender plan churn (L2)", ylabel = "defender interaction cost",
               title = "cost vs churn (points = ncc levels)")

    churn = Float64[l.churn for l in levels]
    cost = Float64[l.cost for l in levels]
    churn_se = Float64[l.churn_se for l in levels]
    cost_se = Float64[l.cost_se for l in levels]

    errorbars!(ax1, nccs, churn, churn_se, color = :gray)
    scatterlines!(ax1, nccs, churn, color = COLOR_ROBUST, markersize = 11)
    errorbars!(ax2, nccs, cost, cost_se, color = :gray)
    scatterlines!(ax2, nccs, cost, color = COLOR_ROBUST, markersize = 11)

    scatter!(ax3, churn, cost, color = log10.(nccs), colormap = :viridis, markersize = 15)
    for (i, l) in enumerate(levels)
        text!(ax3, churn[i], cost[i], text = " $(Int(l.ncc))", fontsize = 10, align = (:left, :bottom))
    end
    r, p, n = corr_with_p(churn, cost)
    ρ, pρ, _ = spearman_with_p(churn, cost)
    text!(ax3, 0.03, 0.95, space = :relative, align = (:left, :top), fontsize = 13,
          text = @sprintf("across %d levels: r=%+.2f (p=%.3g), ρ=%+.2f (p=%.3g)", n, r, p, ρ, pρ))

    path = joinpath(output_dir, "dose_response_ncc.png")
    save(path, fig)
    save(replace(path, ".png" => ".pdf"), fig)
    return path
end

# ==============================================================================
# Entry point
# ==============================================================================

"""
    run_churn_cost_link(; sweep_dir, ncc, dose_ncc, output_dir, max_trials)

Tiers 1 and 2 always run on the robust/non-robust pair at `ncc`. Passing `dose_ncc`
(a vector of ncc values) additionally runs tier 3 across those levels.
"""
function run_churn_cost_link(;
    sweep_dir::String = "outputs/sweep",
    ncc::Float64 = 3000.0,
    dose_ncc::Vector{Float64} = Float64[],
    output_dir = nothing,
    max_trials::Int = 0,
)
    isdefined(Main, :TrajectoryAnalysis) ||
        error("include(\"../TrajectoryAnalysis.jl\"); using .TrajectoryAnalysis first — costs need it")
    TA = Main.TrajectoryAnalysis
    CairoMakie.activate!()

    robust_dir, non_robust_dir = find_config_pair(sweep_dir; ncc)
    out = isnothing(output_dir) ?
        joinpath("analysis", "churn_cost_link", basename(robust_dir)) : output_dir
    isdir(out) || mkpath(out)

    labels = Dict(:robust => "Robust defender (ncc=$ncc)", :non_robust => "Non-robust defender")

    println("Loading arms...")
    rows = Dict(
        :robust     => [trial_scalars(tr, TA) for tr in load_trials(robust_dir; max_trials)],
        :non_robust => [trial_scalars(tr, TA) for tr in load_trials(non_robust_dir; max_trials)],
    )
    paired = paired_rows(rows[:robust], rows[:non_robust])
    println("  matched $(length(paired)) seed-paired trials")

    # Tier 3
    levels = NamedTuple[]
    for v in sort(dose_ncc)
        try
            rdir, _ = find_config_pair(sweep_dir; ncc = v)
            lrows = [trial_scalars(tr, TA) for tr in load_trials(rdir; max_trials)]
            ch = filter(!isnan, col(lrows, :churn_def_l2))
            co = filter(!isnan, col(lrows, :cost_def_inter))
            (isempty(ch) || isempty(co)) && continue
            push!(levels, (; ncc = v, n = length(lrows),
                           churn = mean(ch), churn_se = std(ch) / sqrt(length(ch)),
                           cost = mean(co), cost_se = std(co) / sqrt(length(co))))
            println("  ncc=$v: n=$(length(lrows))  churn=$(round(mean(ch), digits=4))  cost=$(round(mean(co), digits=4))")
        catch e
            @warn "dose level ncc=$v skipped" exception = e
        end
    end

    println("Rendering...")
    written = String[]
    push!(written, plot_scatter_grid(rows, out, labels, "within_arm_scatter.png",
        "Tier 1 — within-arm: churn / predictability vs cost (each point = one trial)"))
    if !isempty(paired)
        push!(written, plot_scatter_grid(Dict(:robust => paired), out, labels,
            "seed_paired_scatter.png",
            "Tier 2 — seed-paired Δ(robust − non-robust), identical noise per pair"))
    end
    dp = plot_dose_response(levels, out)
    isnothing(dp) || push!(written, dp)

    report = joinpath(out, "churn_cost_link_report.txt")
    open(report, "w") do io
        println(io, "Churn / predictability vs cost")
        println(io, "="^100, "\n")
        println(io, "Robust arm:     $robust_dir  ($(length(rows[:robust])) trials)")
        println(io, "Non-robust arm: $non_robust_dir  ($(length(rows[:non_robust])) trials)")
        println(io, "Seed-paired:    $(length(paired)) matched pairs\n")
        println(io, "Stars: * p<0.05  ** p<0.01  *** p<0.001. Correlation is not causation;")
        println(io, "see the header of churn_cost_link.jl for what these tiers can and cannot show.\n")

        correlation_table(io, rows[:robust], "TIER 1a — WITHIN ROBUST ARM (n=$(length(rows[:robust])))")
        correlation_table(io, rows[:non_robust], "TIER 1b — WITHIN NON-ROBUST ARM (n=$(length(rows[:non_robust])))")
        if !isempty(paired)
            correlation_table(io, paired,
                "TIER 2 — SEED-PAIRED DIFFERENCES, robust − non-robust (n=$(length(paired)))")
        end

        if !isempty(levels)
            println(io, "TIER 3 — NCC DOSE-RESPONSE")
            println(io, "-"^100)
            @printf(io, "%12s %6s %14s %14s\n", "ncc", "n", "churn (L2)", "interaction cost")
            for l in levels
                @printf(io, "%12.0f %6d %8.4f±%-5.4f %8.4f±%-5.4f\n",
                        l.ncc, l.n, l.churn, l.churn_se, l.cost, l.cost_se)
            end
            ch = Float64[l.churn for l in levels]
            co = Float64[l.cost for l in levels]
            r, p, n = corr_with_p(ch, co)
            ρ, pρ, _ = spearman_with_p(ch, co)
            @printf(io, "\nchurn vs cost across %d levels: pearson %+.3f (p=%.4g), spearman %+.3f (p=%.4g)\n",
                    n, r, p, ρ, pρ)
            println(io, "A strong positive association here is consistent with churn mediating cost,")
            println(io, "but ncc moves several things at once, so it is not proof of mediation.\n")
        end
    end
    push!(written, report)

    println("\nWrote $(length(written)) files to $out")
    print(read(report, String))
    return (; rows, paired, levels, output_dir = out)
end
