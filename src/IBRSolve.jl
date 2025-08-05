function ibr_solve(game::BeliefGame;
    debug=false,
    ϵ_converge=1e-3,
    debug_file=DEBUG_FILE,
    warm_start=nothing,
    max_ibr_iterations=10,
    max_player_iterations=5)

    if DEBUG
        global DEBUG_FILE = debug_file
        open(DEBUG_FILE, "w") do f
        end
    end

    if isnothing(warm_start)
        dummy_strategy = get_dummy_strategy(game)
        nominal_beliefs, nominal_controls = rollout_strategy(game, dummy_strategy)
    else
        nominal_beliefs, nominal_controls = warm_start
    end

    new_cost = calculate_costs(game, nominal_beliefs, nominal_controls)
    regularizations = Regularizations(100.0, 1.0)
    intermediate_solutions = [(nominal_beliefs, nominal_controls)]
    feed_forward_norms_history = Vector{Vector{Float64}}()

    n_players = game.dims.n + game.is_robust

    for ibr_iter in 1:max_ibr_iterations
        println("IBR Iteration: $ibr_iter")
        for player_index in 1:n_players
            println("  Optimizing player $player_index")

            for player_iter in 1:max_player_iterations
                feedback_terms, feed_forward_norms, Q_suite = player_backward_pass(game, nominal_beliefs, nominal_controls, regularizations, player_index)

                candidate_beliefs, candidate_controls, new_cost, step_accepted = player_line_search(game, nominal_beliefs, nominal_controls, feedback_terms, Q_suite, feed_forward_norms, regularizations, player_index)

                if step_accepted
                    nominal_beliefs, nominal_controls = candidate_beliefs, candidate_controls
                    if all(feed_forward_norms .< ϵ_converge)
                        break
                    end
                    regularizations.control_reg *= 0.98
                else
                    if regularizations.control_reg > 10_000
                        break
                    end
                    regularizations.control_reg *= 1.3
                end
            end
            # push!(intermediate_solutions, (nominal_beliefs, nominal_controls))
        end
    end

    # To conform with the output of the original solve function, we need to compute the final feed_forward_norms
    _, feed_forward_norms, _ = backward_pass(game, nominal_beliefs, nominal_controls, regularizations, 0)
    push!(feed_forward_norms_history, feed_forward_norms)

    println("Final feedforward norms: ", feed_forward_norms, "mean: ", mean(feed_forward_norms), "std: ", std(feed_forward_norms), "max: ", maximum(feed_forward_norms), "min: ", minimum(feed_forward_norms), "median: ", median(feed_forward_norms))
    println("IBR finished.")
    return nominal_beliefs, nominal_controls, nothing, feed_forward_norms_history
end

function player_backward_pass(game::BeliefGame, nominal_beliefs::Vector{Beliefs}, nominal_controls::Vector{BlockVector}, regularizations::Regularizations, player_index::Int)
    T = eltype(nominal_beliefs[1].beliefs[1].belief_mean)
    n_players = game.dims.n + game.is_robust
    belief_size = total_size(nominal_beliefs[end])

    V = 0.0
    V_b = zeros(T, belief_size)
    V_bb = zeros(T, belief_size, belief_size)

    cost_gradient_info = DiffResults.HessianResult(vcat(vec(nominal_beliefs[end]), vec(nominal_controls[end])))

    joint_feedback_strategies = Vector{Any}()
    feed_forward_norms = Vector{Float64}()
    Q_suite = Vector{Any}()

    x_val = vec(nominal_beliefs[end])
    terminal_cost_gradient_info = DiffResults.HessianResult(x_val)

    ForwardDiff.hessian!(
        terminal_cost_gradient_info,
        (x) -> game.costs[player_index].terminal_cost(unvec(x, game.dims.belief)),
        x_val
    )
    V = DiffResults.value(terminal_cost_gradient_info)
    V_b = DiffResults.gradient(terminal_cost_gradient_info)
    V_bb = DiffResults.hessian(terminal_cost_gradient_info)
    for t in game.horizon-1:-1:1
        g, W = ekf_update(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models; is_robust=game.is_robust)
        g_s, W_s = ekf_update_gradient(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models; is_robust=game.is_robust)
        W = real.(W)

        ForwardDiff.hessian!(
            cost_gradient_info,
            x -> game.costs[player_index].non_terminal_cost(
                unvec(x[1:total_size(nominal_beliefs[t])], game.dims.belief),
                BlockVector(x[total_size(nominal_beliefs[t])+1:end], game.is_robust ? vcat(game.dims.controls, sum(game.dims.states)) : game.dims.controls)
            ),
            vcat(vec(nominal_beliefs[t]), vec(nominal_controls[t]))
        )

        Q = clip(DiffResults.value(cost_gradient_info) + V + only(0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
            W[:, jj, :]' * V_bb * W[:, jj]
        end), clip_norm)

        belief_size_t = game.is_robust ? [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls..., sum(game.dims.states)] : [[total_size(b) for b in nominal_beliefs[t].beliefs]..., game.dims.controls...]

        Q_s = BlockVector(
            clip(DiffResults.gradient(cost_gradient_info) + g_s' * V_b + 0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                W_s[:, jj, :]' * V_bb * W[:, jj]
            end, clip_norm),
            belief_size_t
        )

        temp = BlockArray(
            clip(DiffResults.hessian(cost_gradient_info) + g_s' * (V_bb + regularizations.belief_reg * I) * g_s + 0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                W_s[:, jj, :]' * (V_bb + regularizations.belief_reg * I) * W_s[:, jj, :]
            end, clip_norm),
            belief_size_t,
            belief_size_t
        )
        temp[Block(game.dims.n+player_index), Block(game.dims.n+player_index)] += regularizations.control_reg * I
        Q_ss = temp

        control_indices = Block(length(game.dims.belief) + player_index)
        Q_u = BlockVector(zeros(T, sum(game.dims.controls)), game.dims.controls)
        Q_u[Block(player_index)] = @view Q_s[control_indices]
        Q_uu = BlockArray(zeros(T, sum(game.dims.controls), sum(game.dims.controls)), game.dims.controls, game.dims.controls)
        Q_uu[Block(player_index), Block(player_index)] = @view Q_ss[control_indices, control_indices]
        Q_ub = BlockArray(zeros(T, sum(game.dims.controls), belief_size), game.dims.controls, [belief_size])
        Q_ub[Block(player_index), Block(1)] = @view Q_ss[control_indices, Block(1):Block(length(game.dims.belief))]

        feed_forward, feed_back = calculate_feedback_terms(Q_uu, Q_ub, Q_u)
        push!(joint_feedback_strategies, (; feed_forward, feed_back))
        push!(feed_forward_norms, norm(feed_forward))
        push!(Q_suite, (; Qh_uu=Q_uu, Qh_ub=Q_ub, Qh_u=Q_u))

        V = clip(Q + Q_u' * feed_forward + 0.5 * feed_forward' * Q_uu * feed_forward, clip_norm)
        V_b = clip(Q_s[Block(1):Block(length(game.dims.belief))] + feed_back' * Q_uu * feed_forward + feed_back' * Q_u + Q_ub' * feed_forward, clip_norm)
        V_bb = clip(Q_ss[Block(1):Block(length(game.dims.belief)), Block(1):Block(length(game.dims.belief))] + feed_back' * Q_uu * feed_back + feed_back' * Q_ub + Q_ub' * feed_back, clip_norm)
    end

    return reverse!(joint_feedback_strategies), reverse!(feed_forward_norms), reverse!(Q_suite)
end

function player_line_search(game::BeliefGame, nominal_beliefs, nominal_controls, feedback_terms, Q_suite, feed_forward_norms, regularizations, player_index)
    α = 1.0
    ρ = 0.5
    c = 1e-4

    current_ff_norm = mean(feed_forward_norms)

    function loss(α_scalar)
        strategy = build_player_strategy(game, nominal_beliefs, nominal_controls, feedback_terms, α_scalar, player_index)
        b, u = rollout_strategy(game, strategy)
        _, candidate_feed_forward_norms, _ = player_backward_pass(game, b, u, regularizations, player_index)
        return mean(candidate_feed_forward_norms)
    end

    directional_derivative = grad(central_fdm(5, 1), loss, 0.0)[1]

    if directional_derivative > 0
        return nominal_beliefs, nominal_controls, calculate_costs(game, nominal_beliefs, nominal_controls), false
    end

    candidate_beliefs, candidate_controls = rollout_strategy(game, build_player_strategy(game, nominal_beliefs, nominal_controls, feedback_terms, α, player_index))
    candidate_ff_norm = loss(α)

    iters = 0
    while candidate_ff_norm > current_ff_norm + c * α * directional_derivative
        α = ρ * α
        if α < 1e-8
            break
        end
        candidate_beliefs, candidate_controls = rollout_strategy(game, build_player_strategy(game, nominal_beliefs, nominal_controls, feedback_terms, α, player_index))
        candidate_ff_norm = loss(α)
        iters += 1
    end

    new_costs = calculate_costs(game, candidate_beliefs, candidate_controls)
    return candidate_beliefs, candidate_controls, new_costs, true
end

function build_player_strategy(game::BeliefGame, nominal_beliefs, nominal_controls, feedback_terms, α, player_index)
    map(1:game.horizon-1) do t
        function (belief::Beliefs)
            new_controls = deepcopy(nominal_controls[t])
            player_control_update = BlockVector(α * feedback_terms[t].feed_forward + feedback_terms[t].feed_back * (belief - nominal_beliefs[t]), game.dims.controls)
            new_controls[Block(player_index)] += player_control_update[Block(player_index)]
            return new_controls
        end
    end
end

function update_player_controls(nominal_controls, player_controls, player_index)
    new_controls = deepcopy(nominal_controls)
    for t in eachindex(new_controls)
        new_controls[t][Block(player_index)] = player_controls[t]
    end
    return new_controls
end