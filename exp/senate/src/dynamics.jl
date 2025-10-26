function base_dynamics(x::BlockVector, u::BlockVector, m::BlockVector; config::PlayerConfig)
    BlockVector(mapreduce(vcat, enumerate(zip(x.blocks, m.blocks))) do (i, (x_i, m))
        senator = 1 + (i-1) % config.num_senators
        us = BlockVector(vcat([u[Block((j-1) * config.num_senators + senator)] for j in 1:config.num_activists]...), config.control_dims_per_senator[senator])
        x_move = sum([us[1] for u in us.blocks])
        y_move = sum([us[2] for u in us.blocks])
        
        x_i + [x_move; y_move] + m 
    end, length.(x.blocks))
end

function under_actuated_dynamics(x::BlockVector, u::BlockVector, m::BlockVector; config::PlayerConfig)
    BlockVector(mapreduce(vcat, enumerate(zip(x.blocks, m.blocks))) do (i, (x_i, m))
        senator = 1 + (i-1) % config.num_senators
        us = BlockVector(vcat([u[Block((j-1) * config.num_senators + senator)] for j in 1:config.num_activists]...), config.control_dims_per_senator[senator])
        
        x_move = sum([us[1] for u in us.blocks[1:config.num_activists-1]])
        y_move = sum([us[2] for u in us.blocks[2:config.num_activists]])
        
        x_i + [x_move; y_move] + m 
    end, length.(x.blocks))
end

function attraction_dynamics_model(x_all_senators::BlockVector, u::BlockVector, ms::BlockVector; config::PlayerConfig)
    BlockVector(mapreduce(vcat, enumerate(zip(x_all_senators.blocks, ms.blocks))) do (i, (x_i, m))
        senator = 1 + (i-1) % config.num_senators
        us = BlockVector(vcat([u[Block((j-1) * config.num_senators + senator)] for j in 1:config.num_activists]...), config.control_dims_per_senator[senator])

        x_move = sum([u[1] for u in us.blocks[1:config.num_activists-1]]) / config.attraction_matrix[i, i]
        y_move = sum([u[2] for u in us.blocks[2:config.num_activists]]) / config.attraction_matrix[i, i]

        party_forces = zeros(config.state_dims_per_activist[1])

        for j in 1:config.num_senators
            if i == j
                continue
            end
            x_j = x_all_senators.blocks[j]
            diff = x_j - x_i # attraction direction
            dist_sq = dot(diff, diff)

            strength_factor = config.attraction_strength*(config.attraction_numerator/(1+exp(config.attraction_steepness*(dist_sq-config.attraction_offset))))
            party_forces += config.attraction_matrix[i, j] * strength_factor * (diff / sqrt(dist_sq))                
        end

        x_i + [x_move; y_move] + party_forces + m 

    end, length.(x_all_senators.blocks))
end

function drift_dynamics_model(x::BlockVector, u::BlockVector, m::BlockVector; config::PlayerConfig, undrifted_dynamics_model::Function)
    BlockVector(
        undrifted_dynamics_model(x, u, m; config=config) +
            config.dynamics_drift_scale * ones(length(x)),
        length.(x.blocks))
end



