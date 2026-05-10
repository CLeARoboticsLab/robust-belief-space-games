#=
Resume R-vs-R nature-control sweep for the multipliers that didn't complete.

Local audit of `exp/senate/outputs/merged/rvr_nature_control_sweep/` shows:
  m ∈ {1, 2, 5, 10, 25, 50}     : 100/100 seeds — done
  m = 125                        :  86/100 seeds — partial
  m ∈ {250, 625, 1250, 3125, 6250}: 0/100 seeds — only 11-byte stubs from
                                    the pre-fix ENAMETOOLONG run

`run_asymmetric_experiment` always rewrites its per-seed `_mass_results.dat`
at the end of the call, even when override=false skips the inner experiment.
That means re-running m=125 with override=false would clobber the 86 working
seeds with `results=nothing`. So this resume script defaults to skipping
m=125. The downstream analyzer already handles 86/100 fine.

If you want a complete m=125 row, set RVR_INCLUDE_125=1 and the script will
re-run m=125 with override=true (refreshes all 100 seeds).

The PlayerConfig schema added a `nature_target_player_idx` field, but the
existing R-vs-R data is still loadable thanks to the custom
Serialization.deserialize override in `exp/senate/src/SenateParams.jl` —
no full re-run is needed.

Usage (from repo root):
    julia --project=. exp/senate/versions/resume_rvr_sweep.jl
    RVR_CORES=16 julia --project=. exp/senate/versions/resume_rvr_sweep.jl
    RVR_INCLUDE_125=1 julia --project=. exp/senate/versions/resume_rvr_sweep.jl
=#

include(joinpath(@__DIR__, "test_drift_sweep.jl"))

const _CORES = parse(Int, get(ENV, "RVR_CORES", "8"))
const _MISSING_HIGH = [250, 625, 1250, 3125, 6250]

println("\n" * "="^60)
println("RESUME R-vs-R SWEEP")
println("Cores: $_CORES")
println("="^60 * "\n")

println(">>> Phase 1: missing high multipliers $_MISSING_HIGH (override=false)")
run_rvr_nature_control_sweep(
    multipliers=_MISSING_HIGH,
    cores=_CORES,
    override=false,
)

if get(ENV, "RVR_INCLUDE_125", "0") == "1"
    println("\n>>> Phase 2: m=125 refresh (override=true, re-runs all 100 seeds)")
    run_rvr_nature_control_sweep(
        multipliers=[125],
        cores=_CORES,
        override=true,
    )
else
    println("\n>>> Skipping m=125 refresh (RVR_INCLUDE_125 not set).")
    println("    Existing 86/100 seeds will be used by the analyzer as-is.")
end

println("\n" * "="^60)
println("RESUME COMPLETE — re-run analyze_rvr_vs_rnr_overlay() to refresh plots.")
println("="^60)
