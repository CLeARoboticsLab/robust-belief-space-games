using Infiltrator
using ForwardDiff

function debug_nan_inf(var, name, i, j=nothing)
    val = try ForwardDiff.value.(var) catch; var end
    is_bad = any(!isfinite, val)
    if is_bad
        details = "senator i=$i"
        if !isnothing(j)
            details *= ", senator j=$j"
        end
        println("!!! BAD VALUE DETECTED in `$name` ($details) !!!")
        println("Value: ", val)
        @infiltrate
    end
    return is_bad
end

function base_dynamics(x::BlockVector, u::BlockVector, m::BlockVector; config::PlayerConfig)
    BlockVector(mapreduce(vcat, enumerate(zip(x.blocks, m.blocks))) do (i, (x_i, m))
        senator = 1 + (i-1) % config.num_senators
        us = BlockVector(vcat([u[Block((j-1) * config.num_senators + senator)] for j in 1:config.num_activists]...), config.control_dims_per_senator[senator])
        x_move = sum([u[1] for u in us.blocks])
        y_move = sum([u[2] for u in us.blocks])
        
        x_i + [x_move; y_move] + m 
    end, length.(x.blocks))
end

function under_actuated_dynamics(x::BlockVector, u::BlockVector, m::BlockVector; config::PlayerConfig)
    BlockVector(mapreduce(vcat, enumerate(zip(x.blocks, m.blocks))) do (i, (x_i, m))
        senator = 1 + (i-1) % config.num_senators
        us = BlockVector(vcat([u[Block((j-1) * config.num_senators + senator)] for j in 1:config.num_activists]...), config.control_dims_per_senator[senator])
        
        x_move = sum([u[1] for u in us.blocks[1:config.num_activists-1]])
        y_move = sum([u[2] for u in us.blocks[2:config.num_activists]])
        
        x_i + [x_move; y_move] + m 
    end, length.(x.blocks))
end

function attraction_dynamics_model(x_all_senators::BlockVector, u::BlockVector, ms::BlockVector; config::PlayerConfig)
    BlockVector(mapreduce(vcat, enumerate(zip(x_all_senators.blocks, ms.blocks))) do (i, (x_i, m))
        is_dual = eltype(x_all_senators) <: ForwardDiff.Dual
        if is_dual; debug_nan_inf(x_i, "x_i", i); end
        
        senator = 1 + (i-1) % config.num_senators
        us = BlockVector(vcat([u[Block((j-1) * config.num_senators + senator)] for j in 1:config.num_activists]...), config.control_dims_per_senator[senator])

        x_move = sum([u[1] for u in us.blocks[1:config.num_activists-1]]) / (config.attraction_matrix[i, i] + 1e-9)
        y_move = sum([u[2] for u in us.blocks[2:config.num_activists]]) / (config.attraction_matrix[i, i] + 1e-9)

        party_forces = zeros(eltype(x_i), config.state_dims_per_activist[1])

        for j in 1:config.num_senators
            if i == j
                continue
            end
            x_j = x_all_senators.blocks[j]
            if is_dual; debug_nan_inf(x_j, "x_j", i, j); end

            diff = x_j - x_i
            if is_dual; debug_nan_inf(diff, "diff", i, j); end
            
            dist_sq = dot(diff, diff)
            if is_dual; debug_nan_inf(dist_sq, "dist_sq", i, j); end

            strength_factor = config.attraction_strength*(config.attraction_numerator/(1+exp(config.attraction_steepness*(abs(dist_sq)-config.attraction_offset))))
            if is_dual; debug_nan_inf(strength_factor, "strength_factor", i, j); end

            force_update = config.attraction_matrix[i, j] * strength_factor .* sign.(diff) .* sqrt.(abs.(diff))
            if is_dual; debug_nan_inf(force_update, "force_update", i, j); end

            party_forces += force_update
        end

        x_i + [x_move; y_move] + party_forces + m 
 
    end, length.(x_all_senators.blocks))
end
function drift_dynamics_model_generator(undrifted_dynamics_model::Function)
    function(x::BlockVector, u::BlockVector, m::BlockVector; config::PlayerConfig)
        BlockVector(
            undrifted_dynamics_model(x, u, m; config=config) +
                config.drift_dynamics_scale * ones(length(x)),
            length.(x.blocks))
    end
end
function drift_dynamics_model(x::BlockVector, u::BlockVector, m::BlockVector; config::PlayerConfig, undrifted_dynamics_model::Function = base_dynamics)
    BlockVector(
        undrifted_dynamics_model(x, u, m; config=config) +
            config.drift_dynamics_scale * ones(length(x)),
        length.(x.blocks))
end



