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

    Γ = Symmetric(round.(A * Σ * A' + M * Σ * M', digits = 5))
    
    if DEBUG 
        open(DEBUG_FILE, "a") do f
            println(f, "\nΓ matrix:")
            display_matrix = IOContext(f, :limit=>false)
            show(display_matrix, "text/plain", Γ)
            println(f)
        end
    end

    K = round.(Γ * H' * ((H * Γ * H' + N * N') \ I), digits = 5)
    
    if DEBUG
        open(DEBUG_FILE, "a") do f
            println(f, "\nK matrix:")
            display_matrix = IOContext(f, :limit=>false)
            show(display_matrix, "text/plain", K)
            println(f)
        end
    end

    g = [expected_dynamics; Base.vec(Symmetric(round.(Γ - K * H * Γ, digits=100)))]
    W = [sqrt(Symmetric(K * H * Γ)); zeros((sum(dims(beliefs).^2), sum(dims(beliefs))))]

    if DEBUG 
        open(DEBUG_FILE, "a") do f
            display_matrix = IOContext(f, :limit=>false)
            println(f, "\ng vector:")
            # Print belief means first
            println(f, "Belief means: ", [mean[1:dims(beliefs)[1]] for mean in blocks(expected_dynamics)])
            
            # Print covariance matrices
            println(f, "Belief covariances:")
            cov_size = dims(beliefs)[1]
            cov_vec = g[length(expected_dynamics)+1:end]
            cov_mat = reshape(cov_vec, (cov_size*length(beliefs.beliefs), cov_size*length(beliefs.beliefs)))
            show(display_matrix, "text/plain", Symmetric(cov_mat))
            # for (i, belief) in enumerate(beliefs.beliefs)
            #     println(f, "Agent $i covariance:")
            #     show(display_matrix, "text/plain", Symmetric(cov_mat[cov_size*(i-1)+1:cov_size*i, cov_size*(i-1)+1:cov_size*i]))
            #     println(f)
            # end
            println(f, "\nW matrix:")
            show(display_matrix, "text/plain", W)
            println(f)
        end
    end

    return g, W
end

function ekf_update_gradient(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function)
    old_debug = DEBUG
    global DEBUG = false
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
    global DEBUG = old_debug
    return g_s, W_s
end
