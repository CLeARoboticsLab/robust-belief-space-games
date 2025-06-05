function solve(game::MCPGame; debug::Bool = false)
    # TODO: warm start
    !debug || println("[Solve] Solving MCP...")
    start_time = time()
    mcp_sol_raw = MixedComplementarityProblems.solve(
        MixedComplementarityProblems.InteriorPoint(),
        game.mcp,
        [0];
        verbose = debug
    )
    solve_time = time() - start_time
    !debug || println("[Solve] MCP solve took $(solve_time) seconds")
    !debug || println("[Solve]  status $(mcp_sol_raw.status)\n\t kkt_error $(mcp_sol_raw.kkt_error)\n\t outer/total iters: $(mcp_sol_raw.outer_iters)/$(mcp_sol_raw.total_iters)\n\t epsilon: $(mcp_sol_raw.ϵ)")
    sol_interpreted = interpret_variables(mcp_sol_raw, game)

    if mcp_sol_raw.status != :solved || debug
        !debug || println("[Solve] diagnosing...")
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
    open("exp/hockey/outputs/diagnostic.txt", "w") do f
        write(f, "MCP solution status: $(sol.status)\n")
        # Padding
        max_idx = max(
            length(sol_interpreted.xs),
            length(sol_interpreted.us),
            length(sol_interpreted.μs),
            length(sol_interpreted.λs),
            length(sol_interpreted.λ_sh)
        )
        idx_pad = length(string(max_idx))

        # States and controls
        for (i, x) in enumerate(sol_interpreted.xs)
            write(f, "x[$(lpad(i, idx_pad))] = [$(join(map(x -> @sprintf("% .5f", x), x), ", "))]\n")
        end
        for (i, u) in enumerate(sol_interpreted.us)
            write(f, "u[$(lpad(i, idx_pad))] = [$(join(map(u -> @sprintf("% .5f", u), u), ", "))]\n")
        end

        # Calculate and write costs for each player at each timestep
        dims = get_dimensions(game)
        costs = game.game.cost
        for ii in 1:dims.n_players
            write(f, "\nPlayer $ii costs per timestep:\n")
            for t in 1:game.horizon
                cost = costs[ii]([sol_interpreted.xs[t]], [sol_interpreted.us[t]])
                write(f, "t=$t: $(@sprintf("%.5f", cost))\n")
            end
        end
        println()

        # Show the other variables
        for (i, μ) in enumerate(sol_interpreted.μs)
            write(f, "μ[$(lpad(i, idx_pad))] = [$(join(map(μ -> @sprintf("% .5f", μ), μ), ", "))]\n")
        end
        for (i, λ) in enumerate(sol_interpreted.λs)
            write(f, "λ[$(lpad(i, idx_pad))] = [$(join(map(λ -> @sprintf("% .5f", λ), λ), ", "))]\n")
        end
        for (i, λ_sh) in enumerate(sol_interpreted.λ_sh)
            write(f, "λ_sh[$(lpad(i, idx_pad))] = [$(join(map(λ -> @sprintf("% .5f", λ), λ_sh), ", "))]\n")
        end
    end
    return
end