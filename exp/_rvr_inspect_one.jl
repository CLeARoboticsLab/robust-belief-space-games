using Serialization
push!(LOAD_PATH, joinpath(@__DIR__, "..", "exp", "senate"))
include("./SenateTrajectoryAnalysis.jl")

const DIR = "./exp/senate/outputs/merged/rvr_nature_control_sweep"

# Pick one "PURE R-vs-R" file (has p1_nature_multiplier) and one "contamination" file.
const PURE = "seed_1001_p1_nature_multiplier_10_p1_type_robust_p2_believes_p1_drift_sensor_scale_0.0_p2_nature_multiplier_10_p2_type_robust_mass_results.dat"
const CONTAM = "seed_1001_p2_believes_p1_drift_sensor_scale_0.0_p2_nature_multiplier_10_p2_type_robust_mass_results.dat"

function inspect(filename)
    path = joinpath(DIR, filename)
    fsize = stat(path).size
    println("="^80)
    println("FILE: $filename  ($(fsize) bytes)")
    if fsize < 100
        println("  too small — likely a stub")
        return
    end
    data = try
        open(deserialize, path, "r")
    catch e
        println("  deserialize error: $e")
        return
    end
    println("  loaded_type: $(typeof(data))")
    if data isa Vector
        println("  Vector length: $(length(data))")
        if isempty(data)
            println("  EMPTY vector — this is why 0 scenarios load")
            return
        end
        entry = data[1]
        println("  First entry type: $(typeof(entry))")
        if entry isa NamedTuple
            println("  keys: $(keys(entry))")
            for k in keys(entry)
                v = entry[k]
                println("    [:$k] => $(typeof(v))$(v isa AbstractString || v isa Number || v isa Symbol ? " = " * string(v) : "")")
            end
            if haskey(entry, :results)
                r = entry.results
                println("  results type: $(typeof(r))")
                if r isa Dict
                    println("  results keys ($(length(r))): $(collect(keys(r))[1:min(3, length(r))])")
                    if !isempty(r)
                        k1 = first(keys(r))
                        v1 = r[k1]
                        println("  results[$k1] type: $(typeof(v1))")
                        if v1 isa Tuple
                            println("    tuple length: $(length(v1))")
                            for (i, x) in enumerate(v1)
                                println("    [$i]: $(typeof(x))")
                            end
                        end
                    end
                end
            end
        end
    end
end

inspect(PURE)
inspect(CONTAM)
