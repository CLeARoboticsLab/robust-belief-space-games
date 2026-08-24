using Serialization
include("./SenateTrajectoryAnalysis.jl")
using Senate

const DIR = "./exp/senate/outputs/merged/rvr_nature_control_sweep"
const CONTAM = "seed_1001_p2_believes_p1_drift_sensor_scale_0.0_p2_nature_multiplier_10_p2_type_robust_mass_results.dat"

path = joinpath(DIR, CONTAM)
data = open(deserialize, path, "r")
entry = data[1]
println("entry.name = $(entry.name)")
println()
println("entry.params:")
for (k, v) in entry.params
    if v isa AbstractString || v isa Number || v isa Symbol
        println("  $k => $v")
    else
        println("  $k => $(typeof(v))")
    end
end
println()
println("entry.fixed:")
for (k, v) in entry.fixed
    if v isa AbstractString || v isa Number || v isa Symbol
        println("  $k => $v")
    else
        println("  $k => $(typeof(v))")
    end
end
println()
# Pull the actual SenateParams to inspect player_configs.
result_dict = entry.results
k = first(keys(result_dict))
sol_dict, sp = result_dict[k]
println("SenateParams.name = $(sp.name)")
println("SenateParams.player_configs keys: $(collect(keys(sp.player_configs)))")
for (pidx, pc) in sp.player_configs
    println("  player $pidx: type=$(pc.type)  player_idx=$(pc.player_idx)  nature_multiplier=$(pc.nature_multiplier)  drift_sensor_scale=$(pc.drift_sensor_scale)")
end
