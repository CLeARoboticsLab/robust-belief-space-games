
using BlockArrays
using LinearAlgebra

#region: Sensor Models
function N_state_based(x)
    # Hardcoded goal_center from Hockey.jl for now, or pass it?
    # Original Hockey.jl: goal_center = (goal_position[1] + goal_position[2]) / 2
    # goal_position = [[0.25, -1.5], [-0.25, -1.5]] -> center = [0, -1.5]
    goal_center = [0.0, -1.5] 
    
    dist = dot(x[1:2] - goal_center + [0, 3], x[1:2] - goal_center + [0, 3])
    return 1 * I * dist
end

function h_state_based(xs::BlockVector, ns::BlockVector)
    state_dim = 4 # Hardcoded for now
    BlockVector(
        mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
            I(state_dim) * xᵢ + N_state_based(xᵢ) * nᵢ
        end,
        length.(xs.blocks)
    )
end

function h_noise(xs::BlockVector, ns::BlockVector; I_mag::Float64 = 1.0) 
    state_dim = 4
    BlockVector(
        mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
            I(state_dim) * xᵢ + I_mag * I * nᵢ
        end,
        length.(xs.blocks)
    )
end

h_low_noise(xs::BlockVector, ns::BlockVector) = h_noise(xs, ns; I_mag = 0.1)
h_mid_noise(xs::BlockVector, ns::BlockVector) = h_noise(xs, ns; I_mag = 1.0)
h_high_noise(xs::BlockVector, ns::BlockVector) = h_noise(xs,ns; I_mag = 10.0)

const h_noise_dict = Dict(
    "low" => h_low_noise,
    "medium" => h_mid_noise,
    "high" => h_high_noise,
)
#endregion
