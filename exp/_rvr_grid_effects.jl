#=
Seed-paired effect analysis for the RvR nature-cost grid (4x4, 25 shared seeds).

Claims tested (per arm):
  1. Own-robustness effect: how does OWN cost change with OWN nature cost c
     (low c = more robust), holding opponent fixed?  Paired per seed vs own c=5.
  2. Externality: how does OWN cost change with the OPPONENT's c?  Paired vs
     opponent c=5.
  3. Strategic structure: weak dominance of high c, Nash check at (5,5) and
     (625,625), Pareto comparison (prisoner's-dilemma test).

Roles are pooled by symmetry: the sample for (own=a, opp=b) is
P1 totals at cell (a,b) plus P2 totals at cell (b,a), paired by (role, seed).

Outputs (./exp/senate/outputs/analysis/rvr_grid/):
  - effects_<arm>.png    two-panel paired effect curves with 95% CI ribbons
  - effects_report.txt   dominance / Nash / PD numbers with paired CIs

Run from repo root:
    julia --project=. exp/_rvr_grid_effects.jl
=#

using Statistics
using Printf
using CairoMakie

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: SENATE_TRAJECTORY_TRACKER,
                                  load_and_analyze_senate_solution_files
const STA = SenateTrajectoryAnalysis

const COSTS = [5, 25, 125, 625]
const ARMS = [
    ("noobs", "./exp/senate/outputs/runs/rvr_nature_grid_sym_noobs"),
    ("obs",   "./exp/senate/outputs/runs/rvr_nature_grid_sym_obs"),
]
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_grid"
mkpath(OUT_DIR)

# Wong colorblind-safe palette, fixed order by cost level
const SERIES_COLORS = [
    RGBf(0.00, 0.45, 0.70),  # blue
    RGBf(0.90, 0.62, 0.00),  # orange
    RGBf(0.00, 0.62, 0.45),  # green
    RGBf(0.80, 0.47, 0.65),  # purple-pink
]

"cell (c1,c2) -> Dict(seed -> (p1=total, p2=total))"
function load_arm(dir)
    cells = Dict{Tuple{Int, Int}, Dict{Int, NamedTuple}}()
    for c1 in COSTS, c2 in COSTS
        load_and_analyze_senate_solution_files(directory=dir,
            file_pattern=Regex("p1nm_$(c1)_p2nm_$(c2)_p1oc"))
        d = Dict{Int, NamedTuple}()
        for e in SENATE_TRAJECTORY_TRACKER.entries
            d1 = STA._decompose_entry_costs(e, 1)
            d2 = STA._decompose_entry_costs(e, 2)
            (isnothing(d1) || isnothing(d2)) && continue
            d[e.random_seed] = (p1=d1.total, p2=d2.total)
        end
        cells[(c1, c2)] = d
    end
    return cells
end

"Own cost samples for (own c = a, opponent c = b), keyed by (role, seed)."
function own_cost_samples(cells, a, b)
    out = Dict{Tuple{Int, Int}, Float64}()
    for (seed, v) in cells[(a, b)]
        out[(1, seed)] = v.p1
    end
    for (seed, v) in cells[(b, a)]
        out[(2, seed)] = v.p2
    end
    return out
end

"Paired mean difference samples2 - samples1 over shared keys: (mean, lo, hi, n)."
function paired_diff(s1, s2)
    ks = intersect(keys(s1), keys(s2))
    d = [s2[k] - s1[k] for k in ks]
    n = length(d)
    n == 0 && return (mean=NaN, lo=NaN, hi=NaN, n=0)
    m = mean(d)
    half = n > 1 ? 1.96 * std(d) / sqrt(n) : 0.0
    return (mean=m, lo=m - half, hi=m + half, n=n)
end

function effect_panel!(fig, pos, title, xlabel, series)
    # series: Vector of (label, xs, means, los, his)
    ax = Axis(fig[pos...],
        title=title, xlabel=xlabel, ylabel="Δ own executed cost (paired, 95% CI)",
        xscale=log10, xticks=(COSTS, string.(COSTS)))
    hlines!(ax, [0.0]; color=(:black, 0.4), linestyle=:dash, linewidth=1)
    for (i, (label, xs, ms, los, his)) in enumerate(series)
        c = SERIES_COLORS[i]
        band!(ax, xs, los, his; color=(c, 0.18))
        scatterlines!(ax, xs, ms; color=c, markersize=9, linewidth=2, label=label)
    end
    axislegend(ax; position=:lt, framevisible=false)
    return ax
end

report = open(joinpath(OUT_DIR, "effects_report.txt"), "w")

for (arm, dir) in ARMS
    println("\n############ ARM: $arm ############")
    cells = load_arm(dir)

    # Panel 1: own-c effect (vs own c=5), one series per opponent c
    own_series = []
    for (i, b) in enumerate(COSTS)
        base = own_cost_samples(cells, COSTS[1], b)
        ms = Float64[]; los = Float64[]; his = Float64[]
        for a in COSTS
            d = paired_diff(base, own_cost_samples(cells, a, b))
            push!(ms, d.mean); push!(los, d.lo); push!(his, d.hi)
        end
        push!(own_series, ("opponent c = $b", COSTS, ms, los, his))
    end

    # Panel 2: opponent-c effect (vs opponent c=5), one series per own c
    opp_series = []
    for (i, a) in enumerate(COSTS)
        base = own_cost_samples(cells, a, COSTS[1])
        ms = Float64[]; los = Float64[]; his = Float64[]
        for b in COSTS
            d = paired_diff(base, own_cost_samples(cells, a, b))
            push!(ms, d.mean); push!(los, d.lo); push!(his, d.hi)
        end
        push!(opp_series, ("own c = $a", COSTS, ms, los, his))
    end

    fig = Figure(size=(1250, 540))
    effect_panel!(fig, (1, 1),
        "Being less robust: effect of your own nature cost",
        "own nature cost c (higher = less robust)", own_series)
    effect_panel!(fig, (1, 2),
        "Externality: effect of the opponent's nature cost",
        "opponent nature cost c (higher = less robust)", opp_series)
    Label(fig[0, :],
        "RvR grid ($arm): seed-paired cost effects (baseline c = 5, pooled roles, n = 50 pairs)",
        fontsize=17)
    save(joinpath(OUT_DIR, "effects_$(arm).png"), fig)
    println("Saved effects_$(arm).png")

    # ---- strategic-structure report ----
    println(report, "="^90)
    println(report, "ARM: $arm  (positive Δ = costlier; paired 95% CI; n = pairs)")
    println(report, "="^90)

    println(report, "\n[1] Own-c effect, Δ vs own c=5 (per opponent c):")
    for (label, xs, ms, los, his) in own_series
        print(report, @sprintf("  %-16s:", label))
        for j in 2:length(xs)
            print(report, @sprintf("  c=%3d: %+6.2f [%+6.2f,%+6.2f]", xs[j], ms[j], los[j], his[j]))
        end
        println(report)
    end

    println(report, "\n[2] Opponent-c effect, Δ vs opponent c=5 (per own c):")
    for (label, xs, ms, los, his) in opp_series
        print(report, @sprintf("  %-16s:", label))
        for j in 2:length(xs)
            print(report, @sprintf("  c=%3d: %+6.2f [%+6.2f,%+6.2f]", xs[j], ms[j], los[j], his[j]))
        end
        println(report)
    end

    lo_c, hi_c = COSTS[1], COSTS[end]
    mut_rob = own_cost_samples(cells, lo_c, lo_c)
    mut_non = own_cost_samples(cells, hi_c, hi_c)
    dev_at_rob = own_cost_samples(cells, hi_c, lo_c)   # deviator to 625 vs a c=5 opponent
    victim     = own_cost_samples(cells, lo_c, hi_c)   # the c=5 player facing the deviator
    dev_at_non = own_cost_samples(cells, lo_c, hi_c)   # deviator to 5 vs a c=625 opponent

    d_dev_rob = paired_diff(mut_rob, dev_at_rob)
    d_victim  = paired_diff(mut_rob, victim)
    d_dev_non = paired_diff(mut_non, dev_at_non)
    d_mutual  = paired_diff(mut_rob, mut_non)

    println(report, "\n[3] Strategic structure (c=$lo_c 'robust' vs c=$hi_c 'non-robust'):")
    @printf(report, "  mean own cost: mutual-robust=%.2f  mutual-non-robust=%.2f\n",
        mean(values(mut_rob)), mean(values(mut_non)))
    @printf(report, "  deviate from (5,5) to 625:   Δ deviator = %+6.2f [%+6.2f,%+6.2f]  (negative = profitable)\n",
        d_dev_rob.mean, d_dev_rob.lo, d_dev_rob.hi)
    @printf(report, "                                Δ victim   = %+6.2f [%+6.2f,%+6.2f]\n",
        d_victim.mean, d_victim.lo, d_victim.hi)
    @printf(report, "  deviate from (625,625) to 5: Δ deviator = %+6.2f [%+6.2f,%+6.2f]  (positive = (625,625) is Nash)\n",
        d_dev_non.mean, d_dev_non.lo, d_dev_non.hi)
    @printf(report, "  mutual robust -> mutual non:  Δ both     = %+6.2f [%+6.2f,%+6.2f]  (positive = PD: Nash is Pareto-worse)\n",
        d_mutual.mean, d_mutual.lo, d_mutual.hi)
    println(report)
end

close(report)
println("\nWrote effects_report.txt")
println("Done.")
