function solve(game::MCPGame)
    sol = MixedComplementarityProblems.solve(
        MixedComplementarityProblems.InteriorPoint(),
        game.mcp,
        [1, 1];
    )
    return interpret_variables(sol, game)
end

function interpret_variables(sol, game::MCPGame)
    sol.status == :solved || @warn "Solution is not optimal"

    dynamics = game.game.dynamics
    n_players = length(game.game.cost)
    
    state_dims = map(1:n_players) do ii
        state_dim(game.game.dynamics.subsystems[ii])
    end
    x_size = sum(state_dims) * game.horizon
    player_xs = map(1:n_players) do ii
        mapreduce(vcat, 1:game.horizon) do t
            [(t-1)*sum(state_dims)+sum(state_dims[1:ii-1])+1 : (t-1)*sum(state_dims)+sum(state_dims[1:ii])]
        end
    end
    
    control_dims = map(1:n_players) do ii
        control_dim(game.game.dynamics.subsystems[ii])
    end
    u_size = sum(control_dims) * game.horizon
    player_us = map(1:n_players) do ii
        mapreduce(vcat, 1:game.horizon) do t
            [(t-1)*sum(control_dims)+sum(control_dims[1:ii-1])+1 : (t-1)*sum(control_dims)+sum(control_dims[1:ii])]
        end
    end

    # Extract states and controls for each player
    xs = map(1:game.horizon) do t
        BlockVector(
            mapreduce(vcat, 1:n_players) do ii
                sol.x[player_xs[ii][t]]
            end,
            state_dims
        )
    end
    
    us = map(1:game.horizon) do t
        BlockVector(
            mapreduce(vcat, 1:n_players) do ii
                sol.x[x_size .+ player_us[ii][t]]
            end,
            control_dims
        )
    end

    # Extract duals
    player_λs = map(1:n_players) do ii
        sum(game.n_inequality_constraints[1:ii-1])+1:sum(game.n_inequality_constraints[1:ii])
    end
    player_μs = map(1:n_players) do ii
        sum(game.n_equality_constraints[1:ii-1])+1:sum(game.n_equality_constraints[1:ii])
    end

    # Extract equality multipliers (μ) for each player
    μs = BlockVector(
        mapreduce(vcat, 1:n_players) do ii
            sol.x[x_size + u_size .+ player_μs[ii]]
        end,
        game.n_equality_constraints
    )

    # Extract inequality multipliers (λ) for each player
    λs = BlockVector(
        mapreduce(vcat, 1:n_players) do ii
            sol.y[player_λs[ii]]
        end,
        game.n_inequality_constraints
    )

    # Extract shared inequality multipliers
    λ_sh = sol.y[sum(game.n_inequality_constraints)+1:end]

    slack = sol.s

    return (; xs, us, μs, λs, λ_sh, slack)
end