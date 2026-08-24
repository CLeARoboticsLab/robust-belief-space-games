#=
Render P2 cost components over time for R-vs-R vs R-vs-NR, per nature multiplier.

Run from repo root:
    julia --project=. exp/_rvr_component_plot.jl
=#

include("./SenateTrajectoryAnalysis.jl")
using .SenateTrajectoryAnalysis: analyze_rvr_vs_rnr_cost_components

analyze_rvr_vs_rnr_cost_components(
    multipliers=[1, 2, 5, 10, 25, 50, 125, 250, 625],
)

println("\nDone.")
