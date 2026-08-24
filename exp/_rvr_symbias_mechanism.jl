#=
Mechanism diagnostic for the bias sweep: WHY does the robust hedge fail under
differential bias while paying under common bias?

For clean (symintent c5 t=0), common b=1.0, diff b=1.0:
  1. ownR effect decomposed into Δpreference vs Δcontrol (paired, role-pooled)
  2. per-step executed control norms, robust vs nominal (fight intensity)
  3. mean final senator positions per cell (where the tug-of-war ends)

Run from repo root:
    julia --project=. exp/_rvr_symbias_mechanism.jl
=#

using Statistics, Printf, Serialization, CairoMakie

const C = 5
const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_symbias"
const CACHE = joinpath(OUT_DIR, "symbias_mech_cache.dat")
mkpath(OUT_DIR)

const CONDS = [
    ("clean", "rvr_symintent_full_noobs_c5_t0.0"),
    ("common_b1", "rvr_symbias_full_noobs_common_b1.0"),
    ("diff_b1", "rvr_symbias_full_noobs_diff_b1.0"),
]

cell_patterns(c) = Dict(
    :RR => Regex("(?=.*p1nm_$(c)_)(?=.*p2nm_$(c)_)"),
    :NRc => Regex("(?=.*p1t_non_robust)(?=.*p2nm_$(c)_)"),
    :cNR => Regex("(?=.*p1nm_$(c)_)(?=.*p2t_non_robust)"),
    :NN => Regex("(?=.*p1t_non_robust)(?=.*p2t_non_robust)"),
)

function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)
    out = Dict{Tuple{String, Symbol}, Dict{Int, NamedTuple}}()
    for (tag, dirname) in CONDS
        dir = joinpath(RUN_ROOT, dirname)
        isdir(dir) || (println("MISSING $dir"); continue)
        for (cell, pat) in cell_patterns(C)
            Base.invokelatest(STA.load_and_analyze_senate_solution_files;
                directory=dir, file_pattern=pat)
            d = Dict{Int, NamedTuple}()
            for e in STA.SENATE_TRAJECTORY_TRACKER.entries
                d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
                d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
                (isnothing(d1) || isnothing(d2)) && continue
                ctrls = Base.invokelatest(STA.extract_senate_executed_controls, e)
                cn1 = haskey(ctrls, 1) ? [sqrt(sum(abs2, u)) for u in ctrls[1]] : Float64[]
                cn2 = haskey(ctrls, 2) ? [sqrt(sum(abs2, u)) for u in ctrls[2]] : Float64[]
                fin = isempty(e.gt_state_history) ? nothing :
                    [collect(b) for b in e.gt_state_history[end].blocks]
                d[e.random_seed] = (
                    p1=(pref=d1.preference, ctrl=d1.control, total=d1.total),
                    p2=(pref=d2.preference, ctrl=d2.control, total=d2.total),
                    cn1=cn1, cn2=cn2, fin=fin)
            end
            out[(tag, cell)] = d
            println("  $tag/$cell: $(length(d)) seeds")
        end
    end
    serialize(CACHE, out)
    return out
end

data = isfile(CACHE) ? deserialize(CACHE) : build_cache()

mstats(v) = isempty(v) ? (mean=NaN, sem=NaN) : (mean=mean(v), sem=std(v) / sqrt(length(v)))

# ownR decomposition: (NRc.P1 − RR.P1) and (cNR.P2 − RR.P2), pooled.
# Positive Δ = abandoning robustness costs you that much (robustness pays).
function ownR_decomp(tag)
    dp = Float64[]; dc = Float64[]; dt = Float64[]
    for (cellB, f) in ((:NRc, :p1), (:cNR, :p2))
        haskey(data, (tag, :RR)) && haskey(data, (tag, cellB)) || continue
        rr, other = data[(tag, :RR)], data[(tag, cellB)]
        for k in intersect(keys(rr), keys(other))
            a, b = getfield(rr[k], f), getfield(other[k], f)
            push!(dp, b.pref - a.pref); push!(dc, b.ctrl - a.ctrl); push!(dt, b.total - a.total)
        end
    end
    (pref=mstats(dp), ctrl=mstats(dc), total=mstats(dt), n=length(dt))
end

# Within-seed: does being robust make you push harder or softer?
# Compare P1's mean per-step control norm in RR vs NRc (same seed), + mirror.
function ctrl_shift(tag)
    d = Float64[]
    for (cellB, cf) in ((:NRc, :cn1), (:cNR, :cn2))
        haskey(data, (tag, :RR)) && haskey(data, (tag, cellB)) || continue
        rr, other = data[(tag, :RR)], data[(tag, cellB)]
        for k in intersect(keys(rr), keys(other))
            r = getfield(rr[k], cf); n = getfield(other[k], cf)
            (isempty(r) || isempty(n)) && continue
            push!(d, mean(r) - mean(n))  # robust minus nominal effort
        end
    end
    mstats(d)
end

open(joinpath(OUT_DIR, "symbias_mechanism_report.txt"), "w") do io
    println(io, "ownR (cost of abandoning robustness, absolute units; + = robustness pays)")
    println(io, "decomposed into preference (ground lost) vs control (fight effort):\n")
    @printf(io, "%12s | %18s %18s %18s | %s\n", "condition",
        "Δtotal", "Δpreference", "Δcontrol", "n")
    for (tag, _) in CONDS
        r = ownR_decomp(tag)
        @printf(io, "%12s | %+8.3f ± %5.3f  %+8.3f ± %5.3f  %+8.3f ± %5.3f | %d\n",
            tag, r.total.mean, 1.96 * r.total.sem, r.pref.mean, 1.96 * r.pref.sem,
            r.ctrl.mean, 1.96 * r.ctrl.sem, r.n)
    end
    println(io, "\nrobust-minus-nominal per-step control norm (same seed, same role; + = robust pushes harder):\n")
    for (tag, _) in CONDS
        r = ctrl_shift(tag)
        @printf(io, "%12s | %+7.4f ± %6.4f\n", tag, r.mean, 1.96 * r.sem)
    end
    println(io, "\nper-cell mean (preference, control) per player:\n")
    @printf(io, "%12s %5s | %14s %14s | %14s %14s\n", "condition", "cell",
        "P1 pref", "P1 ctrl", "P2 pref", "P2 ctrl")
    for (tag, _) in CONDS, cell in (:RR, :NRc, :cNR, :NN)
        haskey(data, (tag, cell)) || continue
        vs = collect(values(data[(tag, cell)]))
        @printf(io, "%12s %5s | %14.2f %14.2f | %14.2f %14.2f\n", tag, cell,
            mean(v.p1.pref for v in vs), mean(v.p1.ctrl for v in vs),
            mean(v.p2.pref for v in vs), mean(v.p2.ctrl for v in vs))
    end
end
println(read(joinpath(OUT_DIR, "symbias_mechanism_report.txt"), String))

# ---- figure: final senator positions + control-over-time ----
fig = Figure(size=(1500, 560))
for (i, (tag, _)) in enumerate(CONDS)
    ax = Axis(fig[1, i], title="$tag: mean final senator positions", xlabel="x", ylabel="y", aspect=DataAspect())
    scatter!(ax, [3.0], [1.0], marker=:star5, markersize=20, color=:firebrick)
    text!(ax, 3.0, 1.0, text="P1 goal", align=(:left, :bottom), fontsize=10, color=:firebrick)
    scatter!(ax, [1.0], [3.0], marker=:star5, markersize=20, color=:navy)
    text!(ax, 1.0, 3.0, text="P2 goal", align=(:left, :bottom), fontsize=10, color=:navy)
    for (cell, col, mk) in ((:RR, RGBf(0.0, 0.45, 0.7), :circle), (:NN, RGBf(0.85, 0.37, 0.01), :rect))
        haskey(data, (tag, cell)) || continue
        vs = [v.fin for v in values(data[(tag, cell)]) if !isnothing(v.fin)]
        isempty(vs) && continue
        for s in 1:3
            xs = mean(v[s][1] for v in vs); ys = mean(v[s][2] for v in vs)
            scatter!(ax, [xs], [ys], color=col, marker=mk, markersize=13,
                label=s == 1 ? String(cell) : nothing)
        end
    end
    # initial senator positions
    for (x, y) in ((1.0, 1.0), (2.0, 0.5), (0.5, 2.0))
        scatter!(ax, [x], [y], color=(:gray, 0.6), marker=:xcross, markersize=10)
    end
    i == 1 && axislegend(ax, position=:rb, framevisible=false)

    ax2 = Axis(fig[2, i], title="$tag: per-step ‖u‖, RR cell", xlabel="step", ylabel="mean control norm")
    haskey(data, (tag, :RR)) || continue
    vs = collect(values(data[(tag, :RR)]))
    T = minimum(length(v.cn1) for v in vs if !isempty(v.cn1); init=0)
    T == 0 && continue
    lines!(ax2, 1:T, [mean(v.cn1[t] for v in vs) for t in 1:T], color=:firebrick, label="P1")
    lines!(ax2, 1:T, [mean(v.cn2[t] for v in vs) for t in 1:T], color=:navy, label="P2")
    nnvs = haskey(data, (tag, :NN)) ? collect(values(data[(tag, :NN)])) : []
    if !isempty(nnvs)
        Tn = minimum(length(v.cn1) for v in nnvs if !isempty(v.cn1); init=0)
        Tn > 0 && lines!(ax2, 1:Tn, [mean(v.cn1[t] for v in nnvs) for t in 1:Tn],
            color=:firebrick, linestyle=:dash, label="P1 (NN)")
        Tn > 0 && lines!(ax2, 1:Tn, [mean(v.cn2[t] for v in nnvs) for t in 1:Tn],
            color=:navy, linestyle=:dash, label="P2 (NN)")
    end
    i == 1 && axislegend(ax2, position=:rt, framevisible=false, labelsize=10)
end
Label(fig[0, :], "Bias mechanism: endpoints and fight intensity (c=5, 50 seeds)", fontsize=16)
save(joinpath(OUT_DIR, "symbias_mechanism.png"), fig)
println("Saved symbias_mechanism.png")
println("Done.")
