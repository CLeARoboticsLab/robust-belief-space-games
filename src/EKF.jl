function ekf_update(beliefs::Beliefs, control::Vector{Float64}, dynamics, sensor_model::Function)
    Aₜ=ForwardDiff.jacobian((x)-> dynamics(BlockVector(x, dims(beliefs)), control), vcat(means(beliefs)...))
end

function ekf_update_gradient(beliefs::Beliefs, control::Vector{Float64}, dynamics, sensor_model::Function)
end