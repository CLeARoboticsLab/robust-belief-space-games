mutable struct Regularizations
    control_reg::Float64
    belief_reg::Float64
end

function solve(game::BeliefGame; debug=false, ϵ_converge=1e-3, debug_file=DEBUG_FILE, α = 0.01)
    if DEBUG
        global DEBUG_FILE = debug_file
        open(DEBUG_FILE, "w") do f end
    end
    dummy_strategy = get_dummy_strategy(game)
    nominal_beliefs, nominal_controls = rollout_strategy(game, dummy_strategy)
    new_cost = map(1:game.dims.n) do ii
        mapreduce(+, 1:game.horizon - 1, init=0.0) do t
            game.costs[ii].non_terminal_cost(nominal_beliefs[t], nominal_controls[t])
        end +
        game.costs[ii].terminal_cost(nominal_beliefs[end])
    end 
    old_cost = 1/ϵ_converge^2 * new_cost
    regularizations = Regularizations(1.0, 1.0)
    iterations = 0    
    improvement_iterations = 0
    intermediate_beliefs = [nominal_beliefs]

    while norm(new_cost - old_cost)/norm(old_cost) > ϵ_converge
    # while norm(new_cost - old_cost) > ϵ_converge
        old_cost = new_cost
        if DEBUG
            open(DEBUG_FILE, "a") do f
                println(f, "[solve] beliefs and controls")
                println(f, "Initial nominal beliefs:")
                display_matrix = IOContext(f, :limit=>false)
                show(display_matrix, "text/plain", nominal_beliefs)
                println(f)
                println(f, "Initial nominal controls:")
                display_matrix = IOContext(f, :limit=>false)
                show(display_matrix, "text/plain", nominal_controls)
                println(f)
            end
        end
        strategy = backward_pass(game, nominal_beliefs, nominal_controls, regularizations, iterations; α = α)
        candidate_beliefs, candidate_controls = rollout_strategy(game, strategy)

        new_cost = map(1:game.dims.n) do ii
            mapreduce(+, 1:game.horizon - 1, init=0.0) do t
                game.costs[ii].non_terminal_cost(candidate_beliefs[t], candidate_controls[t])
            end +
            game.costs[ii].terminal_cost(candidate_beliefs[end])
        end

        if any(map(x -> new_cost[x] < old_cost[x], 1:game.dims.n))
            nominal_beliefs, nominal_controls = candidate_beliefs, candidate_controls
            regularizations.control_reg *= 0.9
            push!(intermediate_beliefs, candidate_beliefs)
            improvement_iterations += 1
        else
            regularizations.control_reg *= 1.3
        end
        iterations += 1
    end
    println("Converged in $improvement_iterations / $iterations iterations")
    return nominal_beliefs, nominal_controls, intermediate_beliefs
end

function backward_pass(game::BeliefGame, nominal_beliefs::Vector{Beliefs}, nominal_controls::Vector{BlockVector}, regularizations::Regularizations, iteration::Int; α = 0.01)
    T = eltype(nominal_beliefs[1].beliefs[1].belief_mean)
    V = Vector{T}()
    V_b = Vector{Vector{T}}()
    V_bb = Vector{Matrix{T}}()

    cost_gradient_info = [DiffResults.HessianResult(vcat(vec(nominal_beliefs[end]), vec(nominal_controls[end]))) for _ in 1:(game.dims.n+game.is_robust)]

    joint_feedback_strategies = Vector{Any}()

    # Initialize gradient helpers
    x_val = vec(nominal_beliefs[end])
    terminal_cost_gradient_info = DiffResults.HessianResult(x_val)
    for ii in 1:(game.dims.n+game.is_robust)
        ForwardDiff.hessian!(
            terminal_cost_gradient_info,
            (x) -> game.costs[ii].terminal_cost(unvec(x, game.dims.belief)),
            x_val)
        push!(V, DiffResults.value(terminal_cost_gradient_info))
        push!(V_b, DiffResults.gradient(terminal_cost_gradient_info))
        push!(V_bb, DiffResults.hessian(terminal_cost_gradient_info))
    end

    for t in game.horizon-1:-1:1
        g, W = ekf_update(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models; is_robust=game.is_robust)
        g_s, W_s = ekf_update_gradient(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models; is_robust=game.is_robust)
        W = real.(W)
        
        for ii in 1:(game.dims.n+game.is_robust)
        ForwardDiff.hessian!(
            cost_gradient_info[ii],
            x -> game.costs[ii].non_terminal_cost(
                unvec(x[1:total_size(nominal_beliefs[t])], game.dims.belief),
                BlockVector(x[total_size(nominal_beliefs[t])+1:end], game.is_robust ? vcat(game.dims.controls..., game.dims.states[1]) : game.dims.controls)
                    ),
                vcat(vec(nominal_beliefs[t]), vec(nominal_controls[t]))
                )
        end
        if DEBUG
            open(DEBUG_FILE, "a") do f
                println(f, "[backward_pass] time: $t")
                println(f, "cost_gradient_info:")
                display_matrix = IOContext(f, :limit=>false)
                show(display_matrix, "text/plain", cost_gradient_info)
                println(f)
                println(f, "V:")
                display_matrix = IOContext(f, :limit=>false)
                show(display_matrix, "text/plain", V)
                println(f)
            end
        end
        Q = map(1:(game.dims.n+game.is_robust)) do ii
            clip(DiffResults.value(cost_gradient_info[ii]) +
            V[ii] +
            only(0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                W[:, jj, :]' *V_bb[ii] * W[:, jj]
            end), clip_norm)
        end
        Q_s = map(1:(game.dims.n+game.is_robust)) do ii
            BlockVector(
                clip(DiffResults.gradient(cost_gradient_info[ii]) +
                g_s' * V_b[ii] +
                0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                    W_s[:,jj,:]' *V_bb[ii] * W[:,jj]
                end, clip_norm), 
                [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls..., game.is_robust ? game.dims.states[1] : 0]
            )
        end
        Q_ss = map(1:(game.dims.n+game.is_robust)) do ii
            temp = BlockArray(
                clip(DiffResults.hessian(cost_gradient_info[ii]) +
                g_s' * (V_bb[ii]+regularizations.belief_reg * I) * g_s +
                0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                    W_s[:,jj,:]' * (V_bb[ii]+regularizations.belief_reg * I) * W_s[:,jj,:]
                end, clip_norm),
                [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls..., game.is_robust ? game.dims.states[1] : 0],
                [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls..., game.is_robust ? game.dims.states[1] : 0]
            )
            temp[Block(game.dims.n+1):Block(2*game.dims.n), Block(game.dims.n+1):Block(2*game.dims.n)] += regularizations.control_reg * I
            temp
        end
        if DEBUG
            open(DEBUG_FILE, "a") do f
                println(f, "[backward_pass]")
                println(f, "control reg: $(regularizations.control_reg)")
                println(f, "belief reg: $(regularizations.belief_reg)")
                # println(f, "\nQ:")
                # display_matrix = IOContext(f, :limit=>false)
                # show(display_matrix, "text/plain", Q)
                # println(f)
                # println(f, "\nQ_s:")
                # display_matrix = IOContext(f, :limit=>false)
                # show(display_matrix, "text/plain", Q_s)
                # println(f)
                # println(f, "\nQ_ss:")
                # display_matrix = IOContext(f, :limit=>false)
                # show(display_matrix, "text/plain", Q_ss)
                # println(f)
            end
        end
        Qh_u = mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
            @view Q_s[ii][Block(ii+game.dims.n)] # skip the first n belief blocks of Q_s
        end
        Qh_uu = mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
            @view Q_ss[ii][Block(ii+game.dims.n), Block(1+game.dims.n):Block(game.dims.n+game.dims.n+game.is_robust)]
        end
        Qh_ub = mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
            @view Q_ss[ii][Block(ii+game.dims.n), Block(1):Block(game.dims.n)]
        end


        strategy, feed_forward, feed_back = joint_feedback_strategy(Qh_uu, Qh_ub, Qh_u, nominal_controls[t], nominal_beliefs[t], game.dims; α = α, is_robust=game.is_robust)
        push!(joint_feedback_strategies, strategy)

        u_block_indices = Block(1+game.dims.n):Block(2*game.dims.n+game.is_robust)
        b_block_indices = Block(1):Block(game.dims.n)

        V_new = Vector{eltype(V)}(undef, game.dims.n + game.is_robust)
        V_b_new = Vector{eltype(V_b)}(undef, game.dims.n + game.is_robust)
        V_bb_new = Vector{eltype(V_bb)}(undef, game.dims.n + game.is_robust)

        for ii in 1:(game.dims.n + game.is_robust)
            Q_u = @view Q_s[ii][u_block_indices]
            Q_uu = @view Q_ss[ii][u_block_indices, u_block_indices]
            Q_b = @view Q_s[ii][b_block_indices]
            Q_ub = @view Q_ss[ii][u_block_indices, b_block_indices]
            Q_bb = @view Q_ss[ii][b_block_indices, b_block_indices]

            V_new[ii] = clip(Q[ii] + Q_u' * feed_forward +
                             0.5 * feed_forward' * Q_uu * feed_forward, clip_norm)
            
            V_b_new[ii] = clip(Q_b + # Q_b
                                feed_back' * Q_uu * feed_forward + # Q_uu
                                feed_back' * Q_u + # Q_u
                                Q_ub' * feed_forward, clip_norm)# Q_ub
            
            V_bb_new[ii] = clip(Q_bb + # Q_bb
                                 feed_back' * Q_uu * feed_back + # Q_uu
                                 feed_back' * Q_ub + # Q_ub
                                 Q_ub' * feed_back, clip_norm) # Q_ub
        end
        V, V_b, V_bb = V_new, V_b_new, V_bb_new
    end
    return reverse!(joint_feedback_strategies)
end

function joint_feedback_strategy(Qh_uu, Qh_ub, Qh_u, nominal_control, nominal_belief, dims; α = 0.01, is_robust=false)
    Qh_uu_reg = Qh_uu + ϵ * I
    Qh_uu_inv = dual_round.(clip(Qh_uu_reg \ I, clip_norm), digits=5)
    feed_forward = -1 * dual_round.(clip(Qh_uu_inv * Qh_u, clip_norm), digits=5)
    feed_back = -1 * dual_round.(clip(Qh_uu_inv * Qh_ub, clip_norm), digits=5)
    if DEBUG
        open(DEBUG_FILE, "a") do f
            println(f, "[joint_feedback_strategy]")
            println(f, "Qh_uu:")
            display_matrix = IOContext(f, :limit=>false)
            show(display_matrix, "text/plain", Qh_uu)
            println(f)
            println(f, "Qh_uu_inv:")
            show(display_matrix, "text/plain", Qh_uu_inv)
            println(f)
            println(f, "feed_forward:")
            show(display_matrix, "text/plain", feed_forward)
            println(f)
            println(f, "feed_back:")
            show(display_matrix, "text/plain", feed_back)
            println(f)
        end
    end    
    function (belief::Beliefs)
        return BlockVector(nominal_control + α * (feed_forward + feed_back * (belief - nominal_belief)), vcat(dims.controls, is_robust ? dims.states[1] : 0))
    end, feed_forward, feed_back
end

function get_dummy_strategy(game::BeliefGame)
    if game.is_robust
        return [(belief::Beliefs) -> BlockVector(fill(0.0, sum(game.dims.controls) + game.dims.states[1]), vcat(game.dims.controls, game.dims.states[1])) for _ in 1:game.horizon-1]
    else
        return [(belief::Beliefs) -> BlockVector(fill(0.0, sum(game.dims.controls)), game.dims.controls) for _ in 1:game.horizon-1]
    end
end