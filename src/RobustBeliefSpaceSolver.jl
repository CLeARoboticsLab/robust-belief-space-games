mutable struct Regularizations
    control_reg::Float64
    belief_reg::Float64
end

function create_warm_start_strategy(warm_start_controls::Vector{BlockVector})
    """
    Creates a strategy that returns the warm start controls at each time step.
    This allows us to use the current beliefs as the starting point while 
    applying the warm start controls to generate the initial trajectory.
    """
    return [
        (belief::Beliefs) -> warm_start_controls[t] 
        for t in 1:length(warm_start_controls)
    ]
end

function solve(game::BeliefGame; debug=false, ϵ_converge=1e-2, debug_file=DEBUG_FILE, warm_start=nothing, save_intermediate_solutions=false)
    if DEBUG
        global DEBUG_FILE = debug_file
        open(DEBUG_FILE, "w") do f end
    end
    if isnothing(warm_start)
        dummy_strategy = get_dummy_strategy(game)
        nominal_beliefs, nominal_controls = rollout_strategy(game, dummy_strategy)
    else
        # Use current beliefs as starting point and warm start controls to generate trajectory
        warm_start_beliefs, warm_start_controls = warm_start
        warm_start_strategy = create_warm_start_strategy(warm_start_controls)
        nominal_beliefs, nominal_controls = rollout_strategy(game, warm_start_strategy)
    end
    new_cost = calculate_costs(game, nominal_beliefs, nominal_controls)    
    old_cost = 1/ϵ_converge^2 * new_cost
    regularizations = Regularizations(100.0, 1.0)
    iterations = 0    
    improvement_iterations = 0
    intermediate_solutions = [(nominal_beliefs, nominal_controls)]
    feed_forward_norms_history = Vector{Vector{Float64}}()
    kkt_error_history = Vector{Vector{Float64}}()
    # cur_ff_norm = 1
    # push!(feed_forward_norms_history, [Inf])
    # push!(kkt_error_history, [Inf])
    kkt_error_norms = nothing

    cond = Float64[]

    while true
    # while max(feed_forward_norms_history[end]...) > ϵ_converge
        # if DEBUG
        #     open(DEBUG_FILE, "a") do f
        #         println(f, "[solve] beliefs and controls")
        #         println(f, "Initial nominal beliefs:")
        #         display_matrix = IOContext(f, :limit=>false)
        #         show(display_matrix, "text/plain", nominal_beliefs)
        #         println(f)
        #         println(f, "Initial nominal controls:")
        #         display_matrix = IOContext(f, :limit=>false)
        #         show(display_matrix, "text/plain", nominal_controls)
        #         println(f)
        #     end
        # end
        feedback_terms, feed_forward_norms, new_kkt_error_norms = backward_pass(game, nominal_beliefs, nominal_controls, regularizations, iterations; kkt_component=:control)
        candidate_beliefs, candidate_controls, new_cost, step_accepted = line_search(game, nominal_beliefs, nominal_controls, feedback_terms, new_kkt_error_norms, regularizations)

        if save_intermediate_solutions
            push!(feed_forward_norms_history, feed_forward_norms)
            push!(kkt_error_history, norm.(new_kkt_error_norms))
        end
        
        # println("iter: $iterations, error: ", candidate_kkt_error)

        # @printf("[s %3d / %3d]ff: cur=%10.4f new=%10.4f, reg=%10.4f, α=%10.3f\n", iterations, improvement_iterations, mean(feed_forward_norms_history[cur_ff_norm]), mean(feed_forward_norms), regularizations.control_reg, α)
        # println("\tOld costs: ", join([@sprintf("%.3f", c) for c in old_cost], ", "))
        # println("\tNew costs: ", join([@sprintf("%.3f", c) for c in new_cost], ", "))
        # println("\tImprovements: ", join([@sprintf("%.3f", imp) for imp in improvements], ", "))
        # println("\tCost decr: $cost_decreased, ff_norm decr: $feed_forward_norm_decreased")
        if step_accepted
            nominal_beliefs, nominal_controls = candidate_beliefs, candidate_controls
            kkt_error_norms = new_kkt_error_norms

            # if all(improvements .< ϵ_converge) && mean(norm.(vcat(kkt_error_norms...))) < ϵ_converge
            if mean(norm.(kkt_error_norms)) < ϵ_converge
                break
            end
            old_cost = new_cost
            regularizations.control_reg *= 0.98
            regularizations.belief_reg *= 0.98
            if save_intermediate_solutions
                push!(intermediate_solutions, (candidate_beliefs, candidate_controls))
            end
            
            # Store the maximum feed_forward norm as a condition number proxy
            # Higher norms often indicate worse conditioning of the optimization problem
            if !isempty(feedback_terms)
                max_cond = maximum(norm.(feedback_terms[end][1]))
            else
                max_cond = 0.0
            end
            if save_intermediate_solutions
                push!(cond, max_cond)
            end
            improvement_iterations += 1
            # cur_ff_norm = length(feed_forward_norms_history)
            # println("\t step accepted, $improvement_iterations / $iterations")
            
            if DEBUG
                open(DEBUG_FILE, "a") do f
                    println(f, "[solve] Iteration $iterations - Control stationarity error: $current_stationarity_error")
                end
            end
        else
            if regularizations.control_reg > 1000
                break
            end
            regularizations.control_reg *= 1.3
            regularizations.belief_reg *= 1.3
        end
        iterations += 1
    end
    println("Converged in $improvement_iterations / $iterations iterations")
    
    # Compute final control stationarity error
    # _, _, final_kkt_error_norms = backward_pass(game, nominal_beliefs, nominal_controls, regularizations, iterations; kkt_component=:control)
    # println("Final control stationarity error: ", round(mean(norm.(final_kkt_error_norms)), digits=6))
    # println("error stats: \n\tmax: ", round(max(kkt_error_norms...), digits=3), " min: ", round(min(kkt_error_norms...), digits=3), " mean: ", round(mean(kkt_error_norms), digits=3), " std: ", round(std(kkt_error_norms), digits=3), " median: ", round(median(kkt_error_norms), digits=3))
    println("error mean: ", round(mean(norm.(kkt_error_norms)), digits=7))
    if save_intermediate_solutions
        return nominal_beliefs, nominal_controls, intermediate_solutions, feed_forward_norms_history[2:end], kkt_error_history[2:end], cond
    else
        return nominal_beliefs, nominal_controls
    end
end

# TODO: take a gradient step on one player's control (IBR style)

function backward_pass(game::BeliefGame, nominal_beliefs::Vector{Beliefs}, nominal_controls::Vector{BlockVector}, regularizations::Regularizations, iteration::Int; kkt_component::Symbol = :both)
    T = eltype(nominal_beliefs[1].beliefs[1].belief_mean)
    n_players = game.dims.n + game.is_robust
    belief_size = total_size(nominal_beliefs[end])
    
    V = Vector{T}(undef, n_players+game.is_robust)
    V_b = [Vector{T}(undef, belief_size) for _ in 1:n_players+game.is_robust]
    V_bb = [Matrix{T}(undef, belief_size, belief_size) for _ in 1:n_players+game.is_robust]

    lagrange_multipliers = [[Vector{T}(undef, belief_size) for _ in 1:n_players+game.is_robust] for _ in 1:game.horizon-1]

    cost_gradient_info = [DiffResults.HessianResult(vcat(vec(nominal_beliefs[end]), vec(nominal_controls[end]))) for _ in 1:(game.dims.n+game.is_robust)]

    joint_feedback_strategies = Vector{Any}()
    feed_forward_norms = Vector{Float64}()
    stationarity_errors = Vector{Vector{T}}()
    Q_suite = Vector{Any}()

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
        lagrange_multipliers[end][ii] = DiffResults.gradient(terminal_cost_gradient_info)
        V_bb[ii] = DiffResults.hessian(terminal_cost_gradient_info)
    end

    for t in game.horizon-1:-1:1
        g, W = ekf_update(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models; is_robust=game.is_robust)
        g_s, W_s = ekf_update_gradient(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models; is_robust=game.is_robust)
        g_s = real.(g_s)
        W_s = real.(W_s)
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
            @view Q_s[ii][Block(ii+game.dims.n^2)] # control gradient
        end
        Qh_b = mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
            @view Q_s[ii][Block(1):Block(game.dims.n^2)] # belief gradient
        end
        
        Qh_uu = mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
            @view Q_ss[ii][Block(ii+game.dims.n^2), Block(1+game.dims.n^2):Block(game.dims.n^2+game.dims.n+game.is_robust)]
        end
        Qh_ub = mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
            @view Q_ss[ii][Block(ii+game.dims.n^2), Block(1):Block(game.dims.n^2)]
        end

        stationarity_error = if kkt_component === :control
            Qh_u
        elseif kkt_component === :belief
            Qh_b
        else
            [Qh_b; Qh_u]
        end

        feed_forward, feed_back = calculate_feedback_terms(Qh_uu, Qh_ub, Qh_u)
        push!(joint_feedback_strategies, (;feed_forward, feed_back))
        push!(feed_forward_norms, norm(feed_forward))
        push!(stationarity_errors, stationarity_error)
        push!(Q_suite, (;Qh_uu, Qh_ub, Qh_u, Qh_b))

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
            lagrange_multipliers[t][ii] = V_b[ii]

            V_bb[ii] = clip(Q_bb + # Q_bb
                                 feed_back' * Q_uu * feed_back + # Q_uu
                                 feed_back' * Q_ub + # Q_ub
                                 Q_ub' * feed_back, clip_norm) # Q_ub
        end
    end
    return reverse!(joint_feedback_strategies), reverse!(feed_forward_norms), reverse!(stationarity_errors), reverse!(lagrange_multipliers)
end

function calculate_feedback_terms(Qh_uu, Qh_ub, Qh_u)
    Qh_uu_reg = Qh_uu + ϵ * I
    Qh_uu_inv = dual_round.(clip(Qh_uu_reg \ I, clip_norm), digits=5)
    feed_forward = -1 * dual_round.(clip(Qh_uu_inv * Qh_u, clip_norm), digits=5)
    feed_back = -1 * dual_round.(clip(Qh_uu_inv * Qh_ub, clip_norm), digits=5)
    if DEBUG
        open(DEBUG_FILE, "a") do f
            println(f, "[calculate_feedback_terms]")
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
    return feed_forward, feed_back
end


function joint_feedback_strategy(Qh_uu, Qh_ub, Qh_u, nominal_control, nominal_belief, dims; α = 0.01, is_robust=false)
    feed_forward, feed_back = calculate_feedback_terms(Qh_uu, Qh_ub, Qh_u)
    function (belief::Beliefs)
        block_sizes = is_robust ? vcat(dims.controls, sum(dims.states)) : dims.controls
        return BlockVector(nominal_control + α * feed_forward + feed_back * (belief - nominal_belief), block_sizes)
    end
end

function build_strategy(game::BeliefGame, nominal_beliefs, nominal_controls, feedback_terms, α)
    map(1:game.horizon-1) do t
        function (belief::Beliefs)
            block_sizes = game.is_robust ? vcat(game.dims.controls, sum(game.dims.states)) : game.dims.controls
            return BlockVector(nominal_controls[t] + α * feedback_terms[t][1] + feedback_terms[t][2] * (belief - nominal_beliefs[t]), block_sizes)
        end
    end
end

function line_search(game::BeliefGame, nominal_beliefs, nominal_controls, feedback_terms, kkt_error_norms, regularizations)
    α = 1.0
    ρ = 0.9
    c = 1e-4
    
    current_kkt_error = mean(norm.(kkt_error_norms))
    
    function loss(α_scalar)
        strategy = build_strategy(game, nominal_beliefs, nominal_controls, feedback_terms, α_scalar)
        b, u = rollout_strategy(game, strategy)

        # Compute KKT error using control stationarity only
        _, _, candidate_stationarity_errors, lagrange_multipliers = backward_pass(game, b, u, regularizations, 0; kkt_component=:control)
        # ∇ᵤL = mapreduce(vcat, 1:game.horizon-1) do t
        #     stationarity_error = candidate_stationarity_errors[t]
        #     lagrange_multiplier = lagrange_multipliers[t]
        #     mapreduce(vcat, 1:(game.dims.n+game.is_robust)) do ii
        #         g_s, _ = ekf_update_gradient(b[t], u[t], game.environment.dynamics, game.environment.sensor_models; is_robust=game.is_robust)
        #         g_s_u = (ii > game.dims.n) ? g_s[:, total_size(b[t])+sum(game.dims.controls)+1:end] : g_s[:, total_size(b[t])+sum(game.dims.controls[1:ii-1])+1:total_size(b[t])+sum(game.dims.controls[1:ii])]
        #         stat_error = (ii > game.dims.n) ? stationarity_error[sum(game.dims.controls)+1:end] : stationarity_error[sum(game.dims.controls[1:ii-1])+1:sum(game.dims.controls[1:ii])]
        #         stat_error .- g_s_u' * lagrange_multiplier[ii]
        #     end
        # end

        if DEBUG
            open(DEBUG_FILE, "a") do f
                println(f, "[line_search] KKT error breakdown for α=$α_scalar:")
                println(f, "  Stationarity error (control gradients only): $∇ᵤL")
            end
        end
        
        # return norm(∇ᵤL)
        return mean(norm.(candidate_stationarity_errors))
    end
    
    # directional_derivative = grad(central_fdm(5, 1), loss, 0.0)[1]
    # Nocedal and Wright, 11.37
    directional_derivative = -loss(0.0)

    if directional_derivative > 0
        # println("Warning: positive directional derivative. Skipping line search and increasing regularization.")
        return nominal_beliefs, nominal_controls, calculate_costs(game, nominal_beliefs, nominal_controls), false
    end

    candidate_beliefs, candidate_controls = rollout_strategy(game, build_strategy(game, nominal_beliefs, nominal_controls, feedback_terms, α))
    candidate_kkt_error = loss(α)

    iters = 0
    alpha_limit_hit = false
    while candidate_kkt_error > current_kkt_error + c * α * directional_derivative && !alpha_limit_hit
        α = ρ * α
        if α < 1e-3
            alpha_limit_hit = true
        end
        candidate_beliefs, candidate_controls = rollout_strategy(game, build_strategy(game, nominal_beliefs, nominal_controls, feedback_terms, α))
        candidate_kkt_error = loss(α)
        iters += 1
    end
    
    new_costs = calculate_costs(game, candidate_beliefs, candidate_controls)
    if DEBUG
        for ii in 1:game.horizon-1
            println("\t$(feedback_terms[ii][1])")
        end
        open(DEBUG_FILE, "a") do f
            println(f, "[line search] α=$α feedforward terms: ")
            for ii in 1:game.horizon-1
                println(f, "\t$(feedback_terms[ii][1])")
            end
        end
    end

    return candidate_beliefs, candidate_controls, new_costs, !alpha_limit_hit
end


function compute_comprehensive_kkt_error end # retained name for compatibility if referenced elsewhere, but unused

function get_dummy_strategy(game::BeliefGame)
    if game.is_robust
        return [(belief::Beliefs) -> BlockVector(fill(0.0, sum(game.dims.controls) + sum(game.dims.states)), vcat(game.dims.controls, sum(game.dims.states))) for _ in 1:game.horizon-1]
    else
        return [(belief::Beliefs) -> BlockVector(fill(0.0, sum(game.dims.controls)), game.dims.controls) for _ in 1:game.horizon-1]
    end
end
