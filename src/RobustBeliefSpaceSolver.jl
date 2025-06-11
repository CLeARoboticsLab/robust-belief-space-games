struct Regularizations
    control_reg::Float64
    belief_reg::Float64
end

function solve(game::BeliefGame; debug=false, ϵ_converge=1e-4)
    nominal_beliefs, nominal_controls = rollout_strategy(game, [x -> BlockVector(zeros(sum(game.dims.controls)), game.dims.controls) for _ in 1:game.horizon-1])
    new_cost, old_cost = Inf, Inf
    regularizations = Regularizations(1.0, 1.0)

    while norm(new_cost - old_cost) > ϵ_converge
        old_cost = new_cost

        strategy = backward_pass(game, nominal_beliefs, nominal_controls, regularizations)
        candidate_beliefs, candidate_controls = rollout_strategy(game, strategy)

        new_cost = map(1:game.dims.n) do ii
            mapreduce(+, 1:game.horizon) do t
                game.costs[ii].non_terminal_cost(candidate_beliefs[t], candidate_controls[t])
            end +
            game.costs[ii].terminal_cost(candidate_beliefs[end])
        end

        if any(map(x -> new_cost[x] < old_cost[x], 1:game.dims.n))
            nominal_beliefs, nominal_controls = candidate_beliefs, candidate_controls
            regularizations.control_reg *= 0.8
        else
            regularizations.control_reg *= 1.2
        end
    end
end

function backward_pass(game::BeliefGame, nominal_beliefs::Vector{BlockVector}, nominal_controls::Vector{BlockVector}, regularizations::Regularizations)
    V = []
    V_b = []
    V_bb = []

    cost_gradient_info = ForwardDiff.GradientResult(vcat(vec(nominal_beliefs[end]), vec(nominal_controls[end])))

    joint_feedback_strategies = []

    # Initialize gradient helpers
    x_val = vec(nominal_beliefs[end])
    hessian_result = DiffResults.HessianResult(x_val)
    
    for cost in game.costs
        f = (x) -> cost.terminal_cost(unvec(x, game.dims.belief))
        ForwardDiff.hessian!(hessian_result, f, x_val)
        push!(V, DiffResults.value(hessian_result))
        push!(V_b, DiffResults.gradient(hessian_result))
        push!(V_bb, DiffResults.hessian(hessian_result))
    end

    for t in game.horizon-1:-1:1
        #TODO implement ekf_update_gradient
        W_s, g_s = ekf_update_gradient(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models)
        W, g = ekf_update(nominal_beliefs[t], nominal_controls[t], game.environment.dynamics, game.environment.sensor_models)
        
        ForwardDiff.hessian!(
            cost_gradient_info,
            x -> cost.non_terminal_cost(
                unvec(x[1:sum(game.dims.belief)], game.dims.belief),
                unvec(x[sum(game.dims.belief)+1:end], game.dims.controls)
                ),
            vcat(vec(nominal_beliefs[t]), vec(nominal_controls[t]))
            )

        Q = map(1:game.dims.n) do ii
            DiffResults.value(cost_gradient_info) +
            V_b[ii] +
            0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                W[jj]' *V_bb[ii] * W[jj]
            end
        end
        Q_s = map(1:game.dims.n) do ii
            BlockVector(
                DiffResults.gradient(cost_gradient_info) +
                g_s' * V_b[ii] +
                0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                    W_s[jj]' *V_bb[ii] * W[jj]
                end, 
                [sum(game.dims.belief), game.dims.controls...]
            )
        end
        Q_ss = map(1:game.dims.n) do ii
            temp = BlockArray(
                DiffResults.hessian(cost_gradient_info) +
                g_s' * (V_bb[ii]+regularizations.belief_reg * I) * g_s +
                0.5 * mapreduce(+, 1:sum(game.dims.states)) do jj
                    W_s[jj]' * (V_bb[ii]+regularizations.belief_reg * I) * W_s[jj]
                end,
                [sum(game.dims.belief), game.dims.controls...], [sum(game.dims.belief), game.dims.controls...]
            )
            temp[Block(2):Block(1+game.dims.n), Block(2):Block(1+game.dims.n)] += regularizations.belief_reg * I
        end
        Qh_u = mapreduce(vcat, 1:game.dims.n) do ii
            Q_s[ii][Blocks(2):Block(game.dims.n)]
        end
        Qh_uu = mapreduce(vcat, 1:game.dims.n) do ii
            Q_ss[ii][Block(1+ii), Block(2):Block(1+game.dims.n)]
        end
        Qh_ub = mapreduce(vcat, 1:game.dims.n) do ii
            Q_ss[ii][Block(1+ii), Block(1)]
        end
        strategy, feed_forward, feed_back = joint_feebdack_strategy(Qh_uu, Qh_ub, Qh_u, nominal_controls[t], nominal_beliefs[t])
        push!(joint_feedback_strategies, strategy)

        V = map(1:game.dims.n) do ii
            Q[ii] + Q_s[ii][Block(2):Block(1+game.dims.n)]' * feed_forward + # Q_u
            0.5 * feed_forward' * Q_ss[ii][Block(2):Block(1+game.dims.n), Block(2):Block(1+game.dims.n)] * feed_forward # Q_uu
        end
        V_b = map(1:game.dims.n) do ii
            Q_b[ii] + 
            feed_back' * Q_ss[ii][Block(2):Block(1+game.dims.n), Block(2):Block(1+game.dims.n)] * feed_forward + # Q_uu
            feed_back' * Q_s[ii][Block(2):Block(1+game.dims.n)] + # Q_u
            Q_ss[ii][Block(2):Block(1+game.dims.n), Block(1)] # Q_ub
        end
        V_bb = map(1:game.dims.n) do ii
            Q_ss[ii][Block(1), Block(1)] +
            feed_back' * Q_ss[ii][Block(2):Block(1+game.dims.n), Block(2):Block(1+game.dims.n)] * feed_back + # Q_uu
            feed_back' * Q_ss[ii][Block(2):Block(1+game.dims.n), Block(1)] + # Q_ub
            Q_ss[ii][Block(2):Block(1+game.dims.n), Block(1)]' * feed_back # Q_ub
        end
    end
    return joint_feedback_strategies
end

function joint_feedback_strategy(Qh_uu, Qh_ub, Qh_u, nominal_control, nominal_belief; α = 0.1)
    feed_forward = Qh_u \ Qh_uu
    feed_back = Qh_ub \ Qh_uu
    function (belief::Belief)
        return nominal_control + α(feed_forward + feed_back * (belief - nominal_belief))
    end
end