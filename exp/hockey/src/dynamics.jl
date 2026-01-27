
using BlockArrays
using LinearAlgebra

# Dynamics
function M_static(u)
    return 0.1 * I
end

function M_state_based(x)
    goal_center = [0.0, -1.5]
    dist = dot(x[1:2] - goal_center + [0, 3], x[1:2] - goal_center + [0, 3])
    return 0.1 * I * dist
end

# Renamed from rh_f
function basic_dynamics(xs::BlockVector, us::BlockVector, ms::BlockVector, player_idx::Union{Int,Nothing}=nothing)
    dt = 0.3
    dt2 = 0.5 * dt^2

    BlockVector(
        mapreduce(vcat, zip(xs.blocks, us.blocks, ms.blocks)) do (xᵢ, uᵢ, mᵢ)
            [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1] * xᵢ +
            [dt2 0; 0 dt2; dt 0; 0 dt] * uᵢ +
            M_static(xᵢ) * mᵢ
        end,
        length.(xs.blocks)
    )
end
