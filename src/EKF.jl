function ekf_update(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function)
    zero_noise = BlockVector(zeros(sum(dims(beliefs))), dims(beliefs))
    expected_dynamics = dynamics(BlockVector(vcat(means(beliefs)...), dims(beliefs)), control, zero_noise)
    A=ForwardDiff.jacobian((x)-> Vector(dynamics(BlockVector(x, dims(beliefs)), control, zero_noise)), vcat(means(beliefs)...))
    M=ForwardDiff.jacobian((x)-> Vector(dynamics(BlockVector(vcat(means(beliefs)...), dims(beliefs)), control, x)), zero_noise)
    H=ForwardDiff.jacobian((x)-> Vector(sensor_model(dynamics(BlockVector(x, dims(beliefs)), control, zero_noise), zero_noise)), vcat(means(beliefs)...))
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

    Γ = Symmetric(dual_round.(A * Σ * A' + M * Σ * M' + ϵ * I, digits = 5))
    
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
    covs_extraced = mapreduce(vcat, 1:length(beliefs.beliefs)) do dim
        temp[Block(dim), Block(dim)]
    end
    g = [expected_dynamics; Base.vec(covs_extraced)]

    W = [sqrt(Symmetric(K * H * Γ + ϵ * I)); zeros((sum(dims(beliefs).^2), sum(dims(beliefs))))]

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

function ekf_update_gradient(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function)
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
    W_s = ForwardDiff.jacobian(cov_grad, x)
    
    g_s_val = ForwardDiff.value.(real.(g_s)) # TODO fix real. being necessary...
    W_s_val = ForwardDiff.value.(real.(W_s))
    
    global DEBUG = old_debug
    if DEBUG 
        open(DEBUG_FILE, "a") do f
            println(f, "\ng_s:")
            display_matrix = IOContext(f, :limit=>false)
            show(display_matrix, "text/plain", g_s_val)
            println(f)
            println(f, "\nW_s:")
            show(display_matrix, "text/plain", W_s_val)
            println(f)
        end
    end
    return g_s_val, reshape(W_s_val, (total_size(beliefs), sum(dims(beliefs)), total_size(beliefs)+length(control)))
end
