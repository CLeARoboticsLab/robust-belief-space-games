mutable struct Regularizations
    control_reg::Float64
    belief_reg::Float64
end

function solve(game::BeliefGame; debug=false, ϵ_converge=1e-3, debug_file=DEBUG_FILE, α = 0.01)
    if DEBUG
        global DEBUG_FILE = debug_file
        open(DEBUG_FILE, "w") do f end
    end
    nominal_beliefs, nominal_controls = rollout_strategy(game, [(x) -> BlockVector(fill(-.01, sum(game.dims.controls)), game.dims.controls) for _ in 1:game.horizon-1])
    new_cost = map(1:game.dims.n) do ii
        mapreduce(+, 1:game.horizon - 1) do t
            game.costs[ii].non_terminal_cost(nominal_beliefs[t], nominal_controls[t])
        end +
        game.costs[ii].terminal_cost(nominal_beliefs[end])
    end
    old_cost = 1/ϵ_converge^2 * new_cost
    regularizations = Regularizations(1.0, 1.0)
    iterations = 0    
    improvement_iterations = 0
    println("old_cost: $old_cost, new_cost: $new_cost, norm: $(norm(new_cost - old_cost)/norm(old_cost))")

    while norm(new_cost - old_cost)/norm(old_cost) > ϵ_converge
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
            mapreduce(+, 1:game.horizon - 1) do t
                game.costs[ii].non_terminal_cost(candidate_beliefs[t], candidate_controls[t])
            end +
            game.costs[ii].terminal_cost(candidate_beliefs[end])
        end

        if any(map(x -> new_cost[x] < old_cost[x], 1:game.dims.n))
            nominal_beliefs, nominal_controls = candidate_beliefs, candidate_controls
            regularizations.control_reg *= 0.9
            !DEBUG || println("[solve] error: $(norm(new_cost - old_cost)/norm(old_cost))")
            !DEBUG || println("[solve] old_cost: $old_cost")
            !DEBUG || println("[solve] new_cost: $new_cost")
            improvement_iterations += 1
        else
            regularizations.control_reg *= 1.2
        end
        iterations += 1
    end
    !DEBUG || println("Converged in $improvement_iterations / $iterations iterations")
    return nominal_beliefs, nominal_controls
end

function backward_pass(game::BeliefGame, nominal_beliefs::Vector{Beliefs}, nominal_controls::Vector{BlockVector}, regularizations::Regularizations, iteration::Int; α = 0.01)
    V = []
    V_b = []
    V_bb = []

    timesteps = game.horizon-1:-1:1

    cost_gradient_info = [DiffResults.HessianResult(vcat(vec(nominal_beliefs[end]), vec(nominal_controls[end]))) for _ in 1:game.dims.n]

    joint_feedback_strategies = Vector{Function}()

    # Initialize gradient helpers
    x_val = vec(nominal_beliefs[end])
    terminal_cost_gradient_info = DiffResults.HessianResult(x_val)
    for cost in game.costs
        ForwardDiff.hessian!(
            terminal_cost_gradient_info,
            (x) -> cost.terminal_cost(unvec(x, game.dims.belief)),
            x_val)
        push!(V, DiffResults.value(terminal_cost_gradient_info))
        push!(V_b, DiffResults.gradient(terminal_cost_gradient_info))
        push!(V_bb, DiffResults.hessian(terminal_cost_gradient_info))
    end

    for t in timesteps
        g, W = ekf_update(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models)
        g_s, W_s = ekf_update_gradient(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models)
        W = real.(W)
        
        for ii in 1:game.dims.n
        ForwardDiff.hessian!(
            cost_gradient_info[ii],
            x -> game.costs[ii].non_terminal_cost(
                unvec(x[1:total_size(nominal_beliefs[t])], game.dims.belief),
                BlockVector(x[total_size(nominal_beliefs[t])+1:end], game.dims.controls)
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

        Q = map(1:game.dims.n) do ii
            clip(DiffResults.value(cost_gradient_info[ii]) +
            V[ii] +
            only(0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                W[:, jj, :]' *V_bb[ii] * W[:, jj]
            end), clip_norm)
        end
        Q_s = map(1:game.dims.n) do ii
            BlockVector(
                clip(DiffResults.gradient(cost_gradient_info[ii]) +
                g_s' * V_b[ii] +
                0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                    W_s[:,jj,:]' *V_bb[ii] * W[:,jj]
                end, clip_norm), 
                [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls...]
            )
        end
        Q_ss = map(1:game.dims.n) do ii
            temp = BlockArray(
                clip(DiffResults.hessian(cost_gradient_info[ii]) +
                g_s' * (V_bb[ii]+regularizations.belief_reg * I) * g_s +
                0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                    W_s[:,jj,:]' * (V_bb[ii]+regularizations.belief_reg * I) * W_s[:,jj,:]
                end, clip_norm),
                [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls...], [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls...]
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
        Qh_u = mapreduce(vcat, 1:game.dims.n) do ii
            Q_s[ii][Block(ii+game.dims.n)] # skip the first n belief blocks of Q_s
        end
        Qh_uu = mapreduce(vcat, 1:game.dims.n) do ii
            Q_ss[ii][Block(ii+game.dims.n), Block(1+game.dims.n):Block(game.dims.n+game.dims.n)]
        end
        Qh_ub = mapreduce(vcat, 1:game.dims.n) do ii
            Q_ss[ii][Block(ii+game.dims.n), Block(1):Block(game.dims.n)]
        end

        strategy, feed_forward, feed_back = joint_feedback_strategy(Qh_uu, Qh_ub, Qh_u, nominal_controls[t], nominal_beliefs[t], game.dims; α = α)
        push!(joint_feedback_strategies, strategy)

        V = map(1:game.dims.n) do ii
            clip(Q[ii] + Q_s[ii][Block(1+game.dims.n):Block(2*game.dims.n)]' * feed_forward + # Q_u
            0.5 * feed_forward' * Q_ss[ii][Block(1+game.dims.n):Block(2*game.dims.n), Block(1+game.dims.n):Block(2*game.dims.n)] * feed_forward, clip_norm)# Q_uu
        end
        V_b = map(1:game.dims.n) do ii
            clip(Q_s[ii][Block(1):Block(game.dims.n)] + # Q_b
            feed_back' * Q_ss[ii][Block(1+game.dims.n):Block(2*game.dims.n), Block(1+game.dims.n):Block(2*game.dims.n)] * feed_forward + # Q_uu
            feed_back' * Q_s[ii][Block(1+game.dims.n):Block(2*game.dims.n)] + # Q_u
            Q_ss[ii][Block(1+game.dims.n):Block(2*game.dims.n), Block(1):Block(game.dims.n)]' * feed_forward, clip_norm)# Q_ub
        end
        V_bb = map(1:game.dims.n) do ii
            clip(Q_ss[ii][Block(1):Block(game.dims.n), Block(1):Block(game.dims.n)] + # Q_bb
            feed_back' * Q_ss[ii][Block(1+game.dims.n):Block(2*game.dims.n), Block(1+game.dims.n):Block(2*game.dims.n)] * feed_back + # Q_uu
            feed_back' * Q_ss[ii][Block(1+game.dims.n):Block(2*game.dims.n), Block(1):Block(game.dims.n)] + # Q_ub
            Q_ss[ii][Block(1+game.dims.n):Block(2*game.dims.n), Block(1):Block(game.dims.n)]' * feed_back, clip_norm) # Q_ub
        end
    end
    return joint_feedback_strategies
end

function joint_feedback_strategy(Qh_uu, Qh_ub, Qh_u, nominal_control, nominal_belief, dims; α = 0.01)
    Qh_uu_reg = Qh_uu + ϵ * I
    Qh_uu_inv = dual_round.(clip(Qh_uu_reg \ I, clip_norm), digits=5)
    feed_forward = dual_round.(clip(Qh_uu_inv * Qh_u, clip_norm), digits=5)
    feed_back = dual_round.(clip(Qh_uu_inv * Qh_ub, clip_norm), digits=5)
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
        return BlockVector(nominal_control + α * (feed_forward + feed_back * (belief - nominal_belief)), dims.controls)
    end, feed_forward, feed_back
end