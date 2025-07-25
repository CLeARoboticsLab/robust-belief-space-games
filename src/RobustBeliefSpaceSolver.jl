mutable struct Regularizations
    control_reg::Float64
    belief_reg::Float64
end

function solve(game::BeliefGame; debug=false, ϵ_converge=1e-3, debug_file=DEBUG_FILE, α = 0.01, warm_start=nothing, ff_cond=false)
    if DEBUG
        global DEBUG_FILE = debug_file
        open(DEBUG_FILE, "w") do f end
    end
    if isnothing(warm_start)
        dummy_strategy = get_dummy_strategy(game)
        nominal_beliefs, nominal_controls = rollout_strategy(game, dummy_strategy)
    else
        nominal_beliefs, nominal_controls = warm_start
    end
    new_cost = calculate_costs(game, nominal_beliefs, nominal_controls)    
    old_cost = 1/ϵ_converge^2 * new_cost
    regularizations = Regularizations(1.0, 1.0)
    iterations = 0    
    improvement_iterations = 0
    intermediate_solutions = [(nominal_beliefs, nominal_controls)]
    feed_forward_norms_history = Vector{Vector{Float64}}()
    push!(feed_forward_norms_history, [Inf])

    while true
    # while max(feed_forward_norms_history[end]...) > ϵ_converge
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
        strategy, feed_forward_norms = backward_pass(game, nominal_beliefs, nominal_controls, regularizations, iterations; α = α)
        candidate_beliefs, candidate_controls = rollout_strategy(game, strategy)

        new_cost = calculate_costs(game, candidate_beliefs, candidate_controls)

        

        push!(feed_forward_norms_history, feed_forward_norms)
        
        improvements = (old_cost .- new_cost)./abs.(old_cost)
        cost_decreased = any(improvements .> 0)
        feed_forward_norm_decreased = mean(feed_forward_norms) .< mean(feed_forward_norms_history[end])
        select = !ff_cond ? cost_decreased : feed_forward_norm_decreased


        # @printf("[s %3d]ff: cur=%10.4f new=%10.4f, reg=%10.4f, α=%10.3f\n", iterations, max(feed_forward_norms_history[end]...), max(feed_forward_norms...), regularizations.control_reg, α)
        # println("\tOld costs: ", join([@sprintf("%.3f", c) for c in old_cost], ", "))
        # println("\tNew costs: ", join([@sprintf("%.3f", c) for c in new_cost], ", "))
        # println("\tImprovements: ", join([@sprintf("%.3f", imp) for imp in improvements], ", "))
        # println("\tCost decr: $cost_decreased, ff_norm decr: $feed_forward_norm_decreased")
        if select
            nominal_beliefs, nominal_controls = candidate_beliefs, candidate_controls
            if all(improvements .< ϵ_converge) && all(feed_forward_norms .< ϵ_converge)
                break
            end
            old_cost = new_cost
            regularizations.control_reg *= 0.9
            push!(intermediate_solutions, (candidate_beliefs, candidate_controls))
            improvement_iterations += 1
        else
            if regularizations.control_reg > 1000
                break
            end
            regularizations.control_reg *= 1.3
        end
        iterations += 1
    end
    println("Converged in $improvement_iterations / $iterations iterations")
    println("Feed forward norms: max: ", round(max(feed_forward_norms_history[end]...), digits=3), " min: ", round(min(feed_forward_norms_history[end]...), digits=3), " mean: ", round(mean(feed_forward_norms_history[end]), digits=3), " std: ", round(std(feed_forward_norms_history[end]), digits=3), " median: ", round(median(feed_forward_norms_history[end]), digits=3))
    return nominal_beliefs, nominal_controls, intermediate_solutions, feed_forward_norms_history[2:end]
end

# TODO: take a gradient step on one player's control (IBR style)

function backward_pass(game::BeliefGame, nominal_beliefs::Vector{Beliefs}, nominal_controls::Vector{BlockVector}, regularizations::Regularizations, iteration::Int; α = 0.01)
    T = eltype(nominal_beliefs[1].beliefs[1].belief_mean)
    n_players = game.dims.n + game.is_robust
    belief_size = total_size(nominal_beliefs[end])
    
    V = Vector{T}(undef, n_players+game.is_robust)
    V_b = [Vector{T}(undef, belief_size) for _ in 1:n_players+game.is_robust]
    V_bb = [Matrix{T}(undef, belief_size, belief_size) for _ in 1:n_players+game.is_robust]

    cost_gradient_info = [DiffResults.HessianResult(vcat(vec(nominal_beliefs[end]), vec(nominal_controls[end]))) for _ in 1:(game.dims.n+game.is_robust)]

    joint_feedback_strategies = Vector{Any}()
    feed_forward_norms = Vector{Float64}()

    # Initialize gradient helpers
    x_val = vec(nominal_beliefs[end])
    terminal_cost_gradient_info = DiffResults.HessianResult(x_val)
    for ii in 1:(game.dims.n+game.is_robust)
        ForwardDiff.hessian!(
            terminal_cost_gradient_info,
            (x) -> game.costs[ii].terminal_cost(unvec(x, game.dims.belief)),
            x_val)
        V[ii] = DiffResults.value(terminal_cost_gradient_info)
        V_b[ii] = DiffResults.gradient(terminal_cost_gradient_info)
        V_bb[ii] = DiffResults.hessian(terminal_cost_gradient_info)
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
                BlockVector(x[total_size(nominal_beliefs[t])+1:end], game.is_robust ? vcat(game.dims.controls, sum(game.dims.states)) : game.dims.controls)
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
        belief_size = game.is_robust ? [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls..., sum(game.dims.states)] : [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls...]
        Q_s = map(1:(game.dims.n+game.is_robust)) do ii
            BlockVector(
                clip(DiffResults.gradient(cost_gradient_info[ii]) +
                g_s' * V_b[ii] + 
                0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                    W_s[:,jj,:]' *V_bb[ii] * W[:,jj]
                end, clip_norm), 
                belief_size
            )
        end
        Q_ss = map(1:(game.dims.n+game.is_robust)) do ii
            temp = BlockArray(
                clip(DiffResults.hessian(cost_gradient_info[ii]) +
                g_s' * (V_bb[ii]+regularizations.belief_reg * I) * g_s +
                0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                    W_s[:,jj,:]' * (V_bb[ii]+regularizations.belief_reg * I) * W_s[:,jj,:]
                end, clip_norm),
                belief_size,
                belief_size
            )
            temp[Block(game.dims.n+1):Block(2*game.dims.n), Block(game.dims.n+1):Block(2*game.dims.n)] += regularizations.control_reg * I
            temp
        end
        if DEBUG
            open(DEBUG_FILE, "a") do f
                println(f, "[backward_pass]")
                println(f, "control reg: $(regularizations.control_reg)")
                println(f, "belief reg: $(regularizations.belief_reg)")
            end
        end
        Qh_u = mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
            @view Q_s[ii][Block(ii+game.dims.n^2)] # skip the belief blocks of Q_s
        end
        Qh_uu = mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
            @view Q_ss[ii][Block(ii+game.dims.n^2), Block(1+game.dims.n^2):Block(game.dims.n^2+game.dims.n+game.is_robust)]
        end
        Qh_ub = mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
            @view Q_ss[ii][Block(ii+game.dims.n^2), Block(1):Block(game.dims.n^2)]
        end


        strategy, feed_forward, feed_back = joint_feedback_strategy(Qh_uu, Qh_ub, Qh_u, nominal_controls[t], nominal_beliefs[t], game.dims; α = α, is_robust=game.is_robust)
        push!(joint_feedback_strategies, strategy)
        push!(feed_forward_norms, norm(feed_forward))

        u_block_indices = Block(1+game.dims.n^2):Block(game.dims.n^2+game.dims.n+game.is_robust)
        b_block_indices = Block(1):Block(game.dims.n^2)

        for ii in 1:(game.dims.n + game.is_robust)
            Q_u = @view Q_s[ii][u_block_indices]
            Q_uu = @view Q_ss[ii][u_block_indices, u_block_indices]
            Q_b = @view Q_s[ii][b_block_indices]
            Q_ub = @view Q_ss[ii][u_block_indices, b_block_indices]
            Q_bb = @view Q_ss[ii][b_block_indices, b_block_indices]

            V[ii] = clip(Q[ii] + Q_u' * feed_forward +
                             0.5 * feed_forward' * Q_uu * feed_forward, clip_norm)
            
            V_b[ii] = clip(Q_b + # Q_b
                                feed_back' * Q_uu * feed_forward + # Q_uu
                                feed_back' * Q_u + # Q_u
                                Q_ub' * feed_forward, clip_norm)# Q_ub
            
            V_bb[ii] = clip(Q_bb + # Q_bb
                                 feed_back' * Q_uu * feed_back + # Q_uu
                                 feed_back' * Q_ub + # Q_ub
                                 Q_ub' * feed_back, clip_norm) # Q_ub
        end
    end
    return reverse!(joint_feedback_strategies), reverse!(feed_forward_norms)
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
        block_sizes = is_robust ? vcat(dims.controls, sum(dims.states)) : dims.controls
        return BlockVector(nominal_control + α * feed_forward + feed_back * (belief - nominal_belief), block_sizes)
    end, feed_forward, feed_back
end

function get_dummy_strategy(game::BeliefGame)
    if game.is_robust
        return [(belief::Beliefs) -> BlockVector(fill(0.0, sum(game.dims.controls) + sum(game.dims.states)), vcat(game.dims.controls, sum(game.dims.states))) for _ in 1:game.horizon-1]
    else
        return [(belief::Beliefs) -> BlockVector(fill(0.0, sum(game.dims.controls)), game.dims.controls) for _ in 1:game.horizon-1]
    end
end