function ekf_update(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function; is_robust=false)
    zero_noise = BlockVector(zeros(sum(dims(beliefs))), dims(beliefs))
    stacked_controls = mortar([control.blocks[1:end - is_robust]..., control.blocks...])
    expected_dynamics = dynamics(means(beliefs), stacked_controls, zero_noise)

    A_fn(x) = Vector(dynamics(x, stacked_controls, zero_noise))
    M_fn(x) = Vector(dynamics(means(beliefs), stacked_controls, x))
    H_fn(x) = Vector(sensor_model(dynamics(BlockVector(x, dims(beliefs)), stacked_controls, zero_noise), zero_noise))
    N_fn(x) = Vector(sensor_model(expected_dynamics, x))
    
    A = ForwardDiff.jacobian(A_fn, means(beliefs))
    M = ForwardDiff.jacobian(M_fn, zero_noise)
    H = ForwardDiff.jacobian(H_fn, vcat(means(beliefs)...))
    N = ForwardDiff.jacobian(N_fn, zero_noise)

    Σ = BlockDiagonal([b.belief_covariance for b in beliefs]) #removed redundant .beliefs - remove comment after review
    Γ = Symmetric(dual_round.(A * Σ * A' + M * M' + ϵ * I, digits = 5))
    
    S = H * Γ * H' + N * N'
    Q, R = qr(S)
    K_transpose = R \ (Q' * (H * Γ))
    K = dual_round.(K_transpose', digits = 5)

    temp = BlockArray(Symmetric(dual_round.(Γ - K * H * Γ, digits=5)), dims(beliefs), dims(beliefs))
    covs_extraced = mapreduce(hcat, 1:length(beliefs.beliefs)) do dim
        @view temp[Block(dim), Block(dim)]
    end
    if is_robust
        n = length(control.blocks) - 1 #is_robust asserted to be true
        disturbed_expected_dynamics = expected_dynamics[Block(1):Block(n)] + control[Block(n+1)]
        g = [(disturbed_expected_dynamics; expected_dynamics[Block(n+1):Block(n^2)]); Base.vec(covs_extraced)] #removed unnecessary function call
    else
        g = [expected_dynamics; Base.vec(covs_extraced)]
    end

    W = [dual_round.(real.(my_matrix_sqrt(K * H * Γ + ϵ * I)), digits=5); zeros((sum(dims(beliefs).^2), sum(dims(beliefs))))]

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

function ekf_update_gradient(beliefs::Beliefs, control::BlockVector, dynamics, sensor_model::Function; is_robust=false)
    old_debug = DEBUG
    global DEBUG = false #TODO: Remove use of global var. manipulation
    function mean_grad(x)
        return ekf_update(
            unvec(x[1:total_size(beliefs)], dims(beliefs)),
            BlockVector(x[total_size(beliefs)+1:end], length.(blocks(control))),
            dynamics,
            sensor_model;
            is_robust=is_robust)[1]
    end
    function cov_grad(x)
        return ekf_update(
            unvec(x[1:total_size(beliefs)], dims(beliefs)), 
            BlockVector(x[total_size(beliefs)+1:end], length.(blocks(control))),
            dynamics,
            sensor_model;
            is_robust=is_robust)[2]
    end
    x = vcat(vec(beliefs), vec(control))
    g_s = ForwardDiff.jacobian(mean_grad, x)
    W_s = ForwardDiff.jacobian(cov_grad, x)
    
    g_s_val = clip(ForwardDiff.value.(real.(g_s)), clip_norm) # TODO fix real. being necessary...
    W_s_val = clip(real.(W_s), clip_norm)
    global DEBUG = old_debug
    return g_s_val, reshape(W_s_val,
        (total_size(beliefs),
        sum(dims(beliefs)),
        total_size(beliefs)+length(control)))
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