struct MCPGame{T1, T2}
    game::T1
    mcp::T2
    horizon::Int
    n_equality_constraints::Vector{Int}
    n_inequality_constraints::Vector{Int}
    n_shared_inequality_constraints::Int
    player_lagrangians_L::Vector{Symbolics.Num}
    initial_state::Vector{Float64}
end


function get_dimensions(game::TrajectoryGame, horizon::Int)
    n_players = length(game.cost)
    state_dims = map(1:n_players) do ii
        state_dim(game.dynamics.subsystems[ii])
    end
    x_size = sum(state_dims) * horizon
    player_xs = map(1:n_players) do ii
        mapreduce(vcat, 1:horizon) do t
            [(t-1)*sum(state_dims)+sum(state_dims[1:ii-1])+1 : (t-1)*sum(state_dims)+sum(state_dims[1:ii])]
        end
    end
    control_dims = map(1:n_players) do ii
        control_dim(game.dynamics.subsystems[ii])
    end
    u_size = sum(control_dims) * horizon
    player_us = map(1:n_players) do ii
        mapreduce(vcat, 1:horizon) do t
            [(t-1)*sum(control_dims)+sum(control_dims[1:ii-1])+1 : (t-1)*sum(control_dims)+sum(control_dims[1:ii])]
        end
    end
    (;n_players, state_dims, x_size, player_xs, control_dims, u_size, player_us)
end

function get_dimensions(game::MCPGame)
    get_dimensions(game.game, game.horizon)
end