function ekf_update(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function; is_robust=false)
    zero_noise = BlockVector(zeros(sum(dims(beliefs))), dims(beliefs))
    stacked_controls = BlockVector(vcat(control, control), dims(beliefs))
    expected_dynamics = dynamics(means(beliefs), stacked_controls, zero_noise)
    A=ForwardDiff.jacobian((x)-> Vector(dynamics(x, stacked_controls, zero_noise)), means(beliefs))
    M=ForwardDiff.jacobian((x)-> Vector(dynamics(means(beliefs), stacked_controls, x)), zero_noise)
    H=ForwardDiff.jacobian((x)-> Vector(sensor_model(dynamics(BlockVector(x, dims(beliefs)), stacked_controls, zero_noise), zero_noise)), vcat(means(beliefs)...))
    N=ForwardDiff.jacobian((x)-> Vector(sensor_model(expected_dynamics, x)), zero_noise)

    Σ = BlockDiagonal([b.belief_covariance for b in beliefs.beliefs])
    
    if DEBUG 
        open(DEBUG_FILE, "a") do f
            println(f, "[ekf_update]")
            println(f, "A matrix:")
            display_matrix = IOContext(f, :limit=>false)
            show(display_matrix, "text/plain", A)
            println(f)
            
            println(f, "\nM matrix:")
            show(display_matrix, "text/plain", M)
            println(f)
            
            println(f, "\nH matrix:")
            show(display_matrix, "text/plain", H)
            println(f)
            
            println(f, "\nN matrix:")
            show(display_matrix, "text/plain", N)
            println(f)
            
            println(f, "\nΣ matrix:")
            show(display_matrix, "text/plain", Σ)
            println(f)
        end
    end

    Γ = Symmetric(dual_round.(A * Σ * A' + M * M' + ϵ * I, digits = 5))
    
    if DEBUG 
        open(DEBUG_FILE, "a") do f
            println(f, "\nΓ matrix:")
            display_matrix = IOContext(f, :limit=>false)
            show(display_matrix, "text/plain", Γ)
            println(f)
        end
    end

    K = dual_round.(Γ * H' * ((H * Γ * H' + N * N') \ I), digits = 5)
    
    if DEBUG
        open(DEBUG_FILE, "a") do f
            println(f, "\nK matrix:")
            display_matrix = IOContext(f, :limit=>false)
            show(display_matrix, "text/plain", K)
            println(f)
        end
    end

    temp = BlockArray(Symmetric(dual_round.(Γ - K * H * Γ, digits=5)), dims(beliefs), dims(beliefs))
    covs_extraced = mapreduce(hcat, 1:length(beliefs.beliefs)) do dim
        @view temp[Block(dim), Block(dim)]
    end
    if is_robust
        disturbed_expected_dynamics = expected_dynamics[Block(1)] + control[Block(length(beliefs.beliefs) + 1)]
        g = [vcat(disturbed_expected_dynamics, expected_dynamics[Block(2):Block(length(beliefs.beliefs))]); Base.vec(covs_extraced)]
    else
        g = [expected_dynamics; Base.vec(covs_extraced)]
    end

    W = [dual_round.(real.(sqrt(Symmetric(K * H * Γ + ϵ * I))); digits=5); zeros((sum(dims(beliefs).^2), sum(dims(beliefs))))]

    if DEBUG 
        open(DEBUG_FILE, "a") do f
            display_matrix = IOContext(f, :limit=>false)
            println(f, "\ng vector:")
            # Print belief means first
            println(f, "Belief means: ", [mean[1:dims(beliefs)[1]] for mean in blocks(expected_dynamics)])
            
            # Print covariance matrices
            println(f, "Belief covariances:")
            cov_vec = g[length(expected_dynamics)+1:end]
            cov_mat = reshape(cov_vec, (dims(beliefs)[1], sum(dims(beliefs))))
            for (i, belief) in enumerate(beliefs.beliefs)
                println(f, "Agent $i covariance:")
                show(display_matrix, "text/plain", Symmetric(cov_mat[:, dims(beliefs)[1]*(i-1)+1:dims(beliefs)[1]*i]))
                println(f)
            end
            println(f, "\nW matrix:")
            show(display_matrix, "text/plain", W)
            println(f)
        end
    end

    return g, W
end

function ekf_update_gradient(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function; is_robust=false)
    old_debug = DEBUG
    global DEBUG = false
    function mean_grad(x)
        return ekf_update(
            unvec(x[1:total_size(beliefs)], dims(beliefs)),
            BlockVector(x[total_size(beliefs)+1:end], length.(blocks(control))),
            dynamics,
            sensor_model)[1]
    end
    function cov_grad(x)
        return ekf_update(
            unvec(x[1:total_size(beliefs)], dims(beliefs)), 
            BlockVector(x[total_size(beliefs)+1:end], length.(blocks(control))),
            dynamics,
            sensor_model)[2]
    end
    x = vcat(vec(beliefs), vec(control))
    g_s = ForwardDiff.jacobian(mean_grad, x)
    fdm = FiniteDifferences.central_fdm(5, 1)
    W_s = only(FiniteDifferences.jacobian(fdm, cov_grad, x))
    
    g_s_val = clip(ForwardDiff.value.(real.(g_s)), clip_norm) # TODO fix real. being necessary...
    W_s_val = clip(real.(W_s), clip_norm)
    global DEBUG = old_debug
    if DEBUG 
        open(DEBUG_FILE, "a") do f
            println(f, "[ekf_update_gradient]")
            println(f, "\ng_s:")
            display_matrix = IOContext(f, :limit=>false)
            show(display_matrix, "text/plain", g_s_val)
            println(f, "g_s norm: $(norm(g_s_val))")
            println(f, "g_s has imaginary parts: $(any(imag.(g_s_val) .≠ 0))")
            println(f, "W_s norm: $(norm(W_s_val))")
            println(f, "W_s has imaginary parts: $(any(imag.(W_s_val) .≠ 0))")
            println(f, "W_s is nan: $(any(isnan.(W_s_val)))")
        end
    end
    return g_s_val, reshape(W_s_val,
        (total_size(beliefs),
        sum(dims(beliefs)),
        total_size(beliefs)+length(control)))
end

function ekf_update_with_observations(beliefs::Beliefs, control::BlockVector, dynamics::Function, sensor_model::Function, observations::BlockVector; is_robust=false)
    zero_noise = BlockVector(zeros(sum(dims(beliefs))), dims(beliefs))
    expected_dynamics = dynamics(means(beliefs), control, zero_noise)
    A=ForwardDiff.jacobian((x)-> Vector(dynamics(x, control, zero_noise)), means(beliefs))
    M=ForwardDiff.jacobian((x)-> Vector(dynamics(means(beliefs), control, x)), zero_noise)
    H=ForwardDiff.jacobian((x)-> Vector(sensor_model(dynamics(BlockVector(x, dims(beliefs)), control, zero_noise), zero_noise)), vcat(means(beliefs)...))
    N=ForwardDiff.jacobian((x)-> Vector(sensor_model(expected_dynamics, x)), zero_noise)

    Σ = BlockDiagonal([b.belief_covariance for b in beliefs.beliefs])
    Γ = Symmetric(dual_round.(A * Σ * A' + M * M' + ϵ * I, digits = 5))
    K = dual_round.(Γ * H' * ((H * Γ * H' + N * N') \ I), digits = 5)

    temp = BlockArray(Symmetric(dual_round.(Γ - K * H * Γ, digits=5)), dims(beliefs), dims(beliefs))

    mean_update = expected_dynamics + K * (observations - sensor_model(expected_dynamics, zero_noise))
    return Beliefs([Belief(@view(mean_update[Block(ii)]), @view(temp[Block(ii), Block(ii)])) for ii in 1:length(beliefs.beliefs)])
end