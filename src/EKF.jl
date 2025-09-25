function ekf_update(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function; is_robust=false, n_players=2)
    num_beliefs_per_player = length(beliefs.beliefs) ÷ n_players
    g_player_parts = Vector{Vector{eltype(beliefs.beliefs[1].belief_mean)}}(undef, n_players)
    W_player_parts = Vector{Matrix{eltype(beliefs.beliefs[1].belief_mean)}}(undef, n_players)

    for i in 1:n_players
        player_belief_indices = (i-1)*num_beliefs_per_player+1:i*num_beliefs_per_player
        player_beliefs = Beliefs(beliefs.beliefs[player_belief_indices])
        g_player_parts[i], W_player_parts[i] = ekf_update_per_player(player_beliefs, control, dynamics, sensor_model, is_robust=(is_robust && i == 1))
    end
    
    g = vcat(g_player_parts...)
    W = BlockDiagonal(W_player_parts)

    return g, W
end

function ekf_update_per_player(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function; is_robust=false)
    non_robust_control = mortar(control.blocks[1:end-is_robust])
    zero_noise = BlockVector(zeros(sum(dims(beliefs))), dims(beliefs))
    expected_dynamics = dynamics(BlockVector(means(beliefs), dims(beliefs)), non_robust_control, zero_noise)

    A_fn(x) = Vector(dynamics(BlockVector(x, dims(beliefs)), non_robust_control, zero_noise))
    M_fn(x) = Vector(dynamics(means(beliefs), non_robust_control, x))
    H_fn(x) = Vector(sensor_model(dynamics(BlockVector(x, dims(beliefs)), non_robust_control, zero_noise), zero_noise))
    N_fn(x) = Vector(sensor_model(expected_dynamics, x))
    
    A = ForwardDiff.jacobian(A_fn, means(beliefs))
    M = ForwardDiff.jacobian(M_fn, zero_noise)
    H = ForwardDiff.jacobian(H_fn, means(beliefs))
    N = ForwardDiff.jacobian(N_fn, zero_noise)

    Σ = BlockDiagonal([b.belief_covariance for b in beliefs])
    Γ = Symmetric(dual_round.(A * Σ * A' + M * M' + ϵ * I, digits = 5))
    
    S = H * Γ * H' + N * N'
    K = dual_round.((Γ * H') / S, digits=5)

    updated_covs_matrix = Symmetric(dual_round.(Γ - K * H * Γ, digits=5))
    
    new_beliefs_for_player = Vector{Belief}(undef, length(beliefs))
    current_idx = 1
    player_belief_dims = dims(beliefs)

    for i in 1:length(beliefs)
        dim_i = player_belief_dims[i]
        cov_range = current_idx:(current_idx + dim_i - 1)
        
        mean_i = expected_dynamics.blocks[i]
        if is_robust
            mean_i += control.blocks[end][sum(dims(beliefs)[1:i-1])+1:sum(dims(beliefs)[1:i])]
        end
        cov_i = Symmetric(updated_covs_matrix[cov_range, cov_range])

        new_beliefs_for_player[i] = Belief(mean_i, cov_i)

        current_idx += dim_i
    end

    g = vec(Beliefs(new_beliefs_for_player))

    W = [dual_round.(real.(my_matrix_sqrt(K * H * Γ + ϵ * I)), digits=5); zeros(sum(d^2 for d in dims(beliefs)), sum(dims(beliefs)))]

    return g, W
end

function my_matrix_sqrt(A; max_iterations = 10)
    n = size(A, 1)
    old_norm = norm(A)
    normA = A / (old_norm + 1e-9)
    Y = copy(normA)
    Z = I(n)
    T = zeros(size(normA))

    for i in 1:max_iterations
        T = 0.5(3*I(n) - Z*Y)
        Y = Y*T
        Z = T*Z
    end
    return Y * sqrt(old_norm)
end

function ekf_update_gradient(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function; is_robust=false, n_players=2)
    old_debug = DEBUG
    global DEBUG = false #TODO: Remove use of global var. manipulation

    function g_grad_wrapper(x)
        current_beliefs = unvec(x[1:total_size(beliefs)], dims(beliefs))
        current_controls = BlockVector(x[total_size(beliefs)+1:end], length.(blocks(control)))
        g, _ = ekf_update(current_beliefs, current_controls, dynamics, sensor_model; is_robust=is_robust, n_players=n_players)
        return g
    end

    function W_grad_wrapper(x)
        current_beliefs = unvec(x[1:total_size(beliefs)], dims(beliefs))
        current_controls = BlockVector(x[total_size(beliefs)+1:end], length.(blocks(control)))
        _, W = ekf_update(current_beliefs, current_controls, dynamics, sensor_model; is_robust=is_robust, n_players=n_players)
        return vec(W) # Flatten for jacobian calculation
    end

    x = vcat(vec(beliefs), vec(control))
    g_s = ForwardDiff.jacobian(g_grad_wrapper, x)
    W_s_flat = ForwardDiff.jacobian(W_grad_wrapper, x)
    
    g_s_val = clip(ForwardDiff.value.(real.(g_s)), clip_norm)
    
    # Reshape the flattened W jacobian back into its proper 3D tensor shape
    W_shape = (total_size(beliefs), sum(dims(beliefs)))
    W_s_val = reshape(W_s_flat, (W_shape..., length(x)))
    W_s_val = clip(real.(W_s_val), clip_norm)
    
    global DEBUG = old_debug
    return g_s_val, W_s_val
end

function ekf_update_with_observations(beliefs::Beliefs, control::BlockVector, dynamics::Function, sensor_model::Function, observations::BlockVector)
    zero_noise = BlockVector(zeros(sum(dims(beliefs))), dims(beliefs))
    stacked_controls = mortar([control.blocks..., control.blocks...])
    expected_dynamics = dynamics(means(beliefs), stacked_controls, zero_noise)
    A_fn(x) = Vector(dynamics(x, stacked_controls, zero_noise))
    M_fn(x) = Vector(dynamics(means(beliefs), stacked_controls, x))
    H_fn(x) = Vector(sensor_model(dynamics(BlockVector(x, dims(beliefs)), stacked_controls, zero_noise), zero_noise))
    N_fn(x) = Vector(sensor_model(expected_dynamics, x))
    
    A = ForwardDiff.jacobian(A_fn, means(beliefs))
    M = ForwardDiff.jacobian(M_fn, zero_noise)
    H = ForwardDiff.jacobian(H_fn, vcat(means(beliefs)...))
    N = ForwardDiff.jacobian(N_fn, zero_noise)

    Σ = BlockDiagonal([b.belief_covariance for b in beliefs.beliefs])
    Γ = Symmetric(dual_round.(A * Σ * A' + M * M' + ϵ * I, digits = 5))
    
    S = H * Γ * H' + N * N'
    Q, R = qr(S)
    K_transpose = R \ (Q' * (H * Γ))
    K = dual_round.(K_transpose', digits = 5)

    temp = BlockArray(Symmetric(dual_round.(Γ - K * H * Γ, digits=5)), dims(beliefs), dims(beliefs))
    mean_update = expected_dynamics + K * (observations - sensor_model(expected_dynamics, zero_noise))
    return Beliefs([Belief(@view(mean_update[Block(ii)]), @view(temp[Block(ii), Block(ii)])) for ii in 1:length(beliefs.beliefs)])
end

function ekf_update_with_observations(beliefs::Beliefs, control::BlockVector, environments::Vector{BeliefEnvironment}, observations::BlockVector)
    num_players = length(environments)
    if length(beliefs.beliefs) % num_players != 0
        error("Number of beliefs must be a multiple of the number of players.")
    end
    beliefs_per_player = length(beliefs.beliefs) ÷ num_players
    new_beliefs = Vector{Belief}(undef, length(beliefs.beliefs))

    for p in 1:num_players
        player_belief_indices = (p-1)*beliefs_per_player+1:p*beliefs_per_player
        player_beliefs = Beliefs(beliefs.beliefs[player_belief_indices])
        player_observations = BlockVector(observations.blocks[p], dims(player_beliefs))
        updated_player_beliefs = ekf_update_with_observations(player_beliefs, control, environments[p].dynamics, environments[p].sensor_models, player_observations)
        new_beliefs[player_belief_indices] .= updated_player_beliefs.beliefs
    end
    
    return Beliefs(new_beliefs)
end