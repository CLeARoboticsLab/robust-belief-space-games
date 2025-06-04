function solve(game::MCPGame; debug::Bool = false)
    mcp_sol_raw = MixedComplementarityProblems.solve(
        MixedComplementarityProblems.InteriorPoint(),
        game.mcp,
        [0];
        x₀ = fill(0.001, (game.mcp.unconstrained_dimension,)),
        y₀ = fill(0.001, (game.mcp.constrained_dimension,)),
        verbose = debug
    )
    sol_interpreted = interpret_variables(mcp_sol_raw, game)

    if mcp_sol_raw.status != :solved
        @info "Solver status is $(mcp_sol_raw.status). Generating diagnostic heatmap for player 1, L_1 vs z_L[1] and z_L[x_size+1]."
        diagnose_problem(mcp_sol_raw, game, sol_interpreted; debug=debug)
    end

    return sol_interpreted
end

function interpret_variables(sol, game::MCPGame; debug::Bool = false)
    dims = get_dimensions(game)

    # Extract states and controls for each player
    xs = map(1:game.horizon) do t
        BlockVector(
            mapreduce(vcat, 1:dims.n_players) do ii
                sol.x[dims.player_xs[ii][t]]
            end,
            dims.state_dims
        )
    end
    
    us = map(1:game.horizon) do t
        BlockVector(
            mapreduce(vcat, 1:dims.n_players) do ii
                sol.x[dims.x_size .+ dims.player_us[ii][t]]
            end,
            dims.control_dims
        )
    end

    # Extract duals
    player_λs = map(1:dims.n_players) do ii
        sum(game.n_inequality_constraints[1:ii-1])+1:sum(game.n_inequality_constraints[1:ii])
    end
    player_μs = map(1:dims.n_players) do ii
        sum(game.n_equality_constraints[1:ii-1])+1:sum(game.n_equality_constraints[1:ii])
    end

    # Extract equality multipliers (μ) for each player
    μs = BlockVector(
        mapreduce(vcat, 1:dims.n_players) do ii
            sol.x[dims.x_size + dims.u_size .+ player_μs[ii]]
        end,
        game.n_equality_constraints
    )

    # Extract inequality multipliers (λ) for each player
    λs = BlockVector(
        mapreduce(vcat, 1:dims.n_players) do ii
            sol.y[player_λs[ii]]
        end,
        game.n_inequality_constraints
    )

    # Extract shared inequality multipliers
    λ_sh = sol.y[sum(game.n_inequality_constraints)+1:end]

    slack = sol.s

    return (; xs, us, μs, λs, λ_sh, slack)
end

function diagnose_problem(sol, game::MCPGame, sol_interpreted; debug::Bool = false)
    #TODO: something useful/human readable 
    return
end