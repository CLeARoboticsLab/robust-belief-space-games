function ekf_update(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function)
    zero_noise = BlockVector(zeros(sum(dims(beliefs))), dims(beliefs))
    expected_dynamics = dynamics(BlockVector(vcat(means(beliefs)...), dims(beliefs)), control, zero_noise)
    A=ForwardDiff.jacobian((x)-> Vector(dynamics(BlockVector(x, dims(beliefs)), control, zero_noise)), vcat(means(beliefs)...))
    M=ForwardDiff.jacobian((x)-> Vector(dynamics(BlockVector(vcat(means(beliefs)...), dims(beliefs)), control, x)), zero_noise)
    H=ForwardDiff.jacobian((x)-> Vector(sensor_model(dynamics(BlockVector(x, dims(beliefs)), control, zero_noise), zero_noise)), vcat(means(beliefs)...))
    N=ForwardDiff.jacobian((x)-> Vector(sensor_model(expected_dynamics, x)), zero_noise)

    Σ = BlockArray(zeros((sum(dims(beliefs)), sum(dims(beliefs)))), dims(beliefs), dims(beliefs))
    for ii in eachindex(beliefs.beliefs)
        for jj in eachindex(beliefs.beliefs)
            if ii == jj
                Σ[Block(ii), Block(jj)] = beliefs.beliefs[ii].belief_covariance
            end
        end
    end
    Γ = Symmetric(A * Σ * A' + M * Σ * M')
    K = Γ * H' * ((H * Γ * H' + N * N') \ I)
    g = [expected_dynamics; Base.vec(Symmetric(Γ - K * H * Γ))]
    W = [sqrt(Symmetric(K * H * Γ)); zeros((sum(dims(beliefs).^2), sum(dims(beliefs))))]

    return g, W
end

function ekf_update_gradient(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function)
    function mean_grad(x)
        beliefs_vec = unvec(x[1:total_size(beliefs)], dims(beliefs))
        control_vec = BlockVector(x[total_size(beliefs)+1:end], length.(blocks(control)))
        g, _ = ekf_update(Beliefs(beliefs_vec, dims(beliefs)[1]), control_vec, dynamics, sensor_model)
        return g
    end
    
    # Function to compute the gradient of the covariance update
    function cov_grad(x)
        beliefs_vec = unvec(x[1:total_size(beliefs)], dims(beliefs))
        control_vec = BlockVector(x[total_size(beliefs)+1:end], length.(blocks(control)))
        _, W = ekf_update(beliefs_vec, control_vec, dynamics, sensor_model)
        return W
    end
    x = vcat(vec(beliefs), vec(control))
    g_s = ForwardDiff.gradient(mean_grad, x)
    W_s = ForwardDiff.gradient(cov_grad, x)
    
    return g_s, W_s
end
