#=
Common-mode vs differential sensor-bias sweep analysis (noobs, 50 seeds, c=5).

Question: does robustness insure against inter-player model DIVERGENCE
(:diff, biases +b/−b) rather than model error per se (:common, +b/+b)?

Reference b=0 point taken from the symintent surface cache (c=5, t=0) if present.

Run from repo root:
    julia --project=. exp/_rvr_symbias_analysis.jl
=#

using Statistics, Printf, Serialization, CairoMakie

const BS = [0.1, 0.3, 1.0]
const MODES = ["common", "diff"]
const C = 5
const PREFIX = "rvr_symbias_full_noobs"
const RUN_ROOT = "./exp/senate/outputs/runs"
const OUT_DIR = "./exp/senate/outputs/analysis/rvr_symbias"
const CACHE = joinpath(OUT_DIR, "symbias_cache.dat")
const SURF_CACHE = "./exp/senate/outputs/analysis/rvr_symintent/symintent_surface_cache.dat"
mkpath(OUT_DIR)

cell_patterns(c) = Dict(
    :RR => Regex("(?=.*p1nm_$(c)_)(?=.*p2nm_$(c)_)"),
    :NRc => Regex("(?=.*p1t_non_robust)(?=.*p2nm_$(c)_)"),
    :cNR => Regex("(?=.*p1nm_$(c)_)(?=.*p2t_non_robust)"),
    :NN => Regex("(?=.*p1t_non_robust)(?=.*p2t_non_robust)"),
)

function build_cache()
    include("./SenateTrajectoryAnalysis.jl")
    STA = Base.invokelatest(getfield, Main, :SenateTrajectoryAnalysis)
    out = Dict{Tuple{String, Float64, Symbol}, Dict{Int, NamedTuple}}()
    for mode in MODES, b in BS
        dir = joinpath(RUN_ROOT, "$(PREFIX)_$(mode)_b$(b)")
        isdir(dir) || (println("MISSING $dir"); continue)
        for (cell, pat) in cell_patterns(C)
            Base.invokelatest(STA.load_and_analyze_senate_solution_files;
                directory=dir, file_pattern=pat)
            d = Dict{Int, NamedTuple}()
            for e in STA.SENATE_TRAJECTORY_TRACKER.entries
                d1 = Base.invokelatest(STA._decompose_entry_costs, e, 1)
                d2 = Base.invokelatest(STA._decompose_entry_costs, e, 2)
                (isnothing(d1) || isnothing(d2)) && continue
                d[e.random_seed] = (p1=d1.total, p2=d2.total)
            end
            out[(mode, b, cell)] = d
        end
        println("loaded $mode b=$b")
    end
    serialize(CACHE, out)
    return out
end

data = isfile(CACHE) ? deserialize(CACHE) : build_cache()

# b=0 reference from the symintent surface cache (c=5, t=0.0): identical setup, zero bias.
if isfile(SURF_CACHE)
    surf = deserialize(SURF_CACHE)
    for cell in (:RR, :NRc, :cNR, :NN)
        haskey(surf, (5, 0.0, cell)) && for mode in MODES
            data[(mode, 0.0, cell)] = surf[(5, 0.0, cell)]
        end
    end
end
const BS_ALL = haskey(data, ("common", 0.0, :RR)) ? [0.0; BS] : BS

p1f = v -> v.p1
p2f = v -> v.p2

function pooled_pct(mode, b, specs)
    all(haskey(data, (mode, b, s[1])) && haskey(data, (mode, b, s[2])) for s in specs) ||
        return (mean=NaN, sem=NaN, n=0, base=NaN)
    rr = data[(mode, b, :RR)]
    isempty(rr) && return (mean=NaN, sem=NaN, n=0, base=NaN)
    base = mean(vcat([v.p1 for v in values(rr)], [v.p2 for v in values(rr)]))
    d = Float64[]
    for (from, to, ffrom, fto) in specs
        ks = intersect(keys(data[(mode, b, from)]), keys(data[(mode, b, to)]))
        append!(d, [(fto(data[(mode, b, to)][k]) - ffrom(data[(mode, b, from)][k])) / base * 100 for k in ks])
    end
    isempty(d) && return (mean=NaN, sem=NaN, n=0, base=base)
    (mean=mean(d), sem=std(d) / sqrt(length(d)), n=length(d), base=base)
end

const EFFECTS = [
    (:ownR, "own robustness effect vs robust opp", [(:RR, :NRc, p1f, p1f), (:RR, :cNR, p2f, p2f)]),
    (:ownN, "own robustness effect vs nominal opp", [(:NRc, :NN, p2f, p2f), (:cNR, :NN, p1f, p1f)]),
    (:ext, "externality of opp robustness on nominal", [(:NN, :NRc, p1f, p1f), (:NN, :cNR, p2f, p2f)]),
    (:mut, "mutual effect (NN − RR)", [(:RR, :NN, p1f, p1f), (:RR, :NN, p2f, p2f)]),
]

open(joinpath(OUT_DIR, "symbias_report.txt"), "w") do io
    for (key, title, specs) in EFFECTS
        println(io, "== $key: $title (% of that condition's RR baseline) ==")
        @printf(io, "%6s |%22s |%22s\n", "b", "common (+b,+b)", "diff (+b,-b)")
        for b in BS_ALL
            @printf(io, "%6.2f |", b)
            for mode in MODES
                r = pooled_pct(mode, b, specs)
                isfinite(r.mean) ?
                    @printf(io, "   %+7.2f%% ± %5.2f  ", r.mean, 1.96 * r.sem) :
                    @printf(io, "   %18s  ", "--")
            end
            println(io)
        end
        println(io)
    end
    println(io, "== baseline RR mean cost ==")
    @printf(io, "%6s |%14s |%14s\n", "b", "common", "diff")
    for b in BS_ALL
        @printf(io, "%6.2f |", b)
        for mode in MODES
            r = pooled_pct(mode, b, EFFECTS[1][3])
            isfinite(r.base) ? @printf(io, "     %8.2f  ", r.base) : @printf(io, "     %8s  ", "--")
        end
        println(io)
    end
end
println(read(joinpath(OUT_DIR, "symbias_report.txt"), String))

fig = Figure(size=(1250, 520))
for (i, (key, title, specs)) in enumerate(EFFECTS[1:2])
    ax = Axis(fig[1, i], title=title,
        xlabel="sensor bias magnitude b", ylabel="effect, % of baseline cost")
    for (mode, col) in zip(MODES, [RGBf(0.00, 0.45, 0.70), RGBf(0.85, 0.37, 0.01)])
        pts = [(b, pooled_pct(mode, b, specs)) for b in BS_ALL]
        pts = [(b, r) for (b, r) in pts if isfinite(r.mean)]
        isempty(pts) && continue
        xs = [p[1] for p in pts]
        ys = [p[2].mean for p in pts]
        es = [1.96 * p[2].sem for p in pts]
        band!(ax, xs, ys .- es, ys .+ es, color=(col, 0.18))
        scatterlines!(ax, xs, ys, color=col,
            label=mode == "common" ? "common (+b,+b): shared map error" : "diff (+b,−b): model divergence")
    end
    hlines!(ax, [0.0], color=:gray, linestyle=:dash)
    i == 1 && axislegend(ax, position=:lt, framevisible=false, labelsize=11)
end
Label(fig[0, :],
    "Sensor bias structure: common-mode vs differential (noobs, c=5, 50 seeds, paired, % of baseline)",
    fontsize=16)
save(joinpath(OUT_DIR, "symbias_effects.png"), fig)
println("Saved symbias_effects.png")
println("Done.")
