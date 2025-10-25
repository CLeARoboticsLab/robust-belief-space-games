function non_terminal_cost_components(preference_ellipsoids::Function, control_function::Function, senator_beliefs::BlockVector, u::BlockVector )
    preference = sum(preference_ellipsoids(pos) for pos in senator_beliefs.blocks)
    control = control_function(u)
    return (;preference, control)
end

function terminal_cost_components(preference_ellipsoids::Function, senator_beliefs::BlockVector)
    preference = sum(preference_ellipsoids(pos) for pos in senator_beliefs.blocks)
    return (;preference)
end

function ellipsoidal_cost(point::Vector, pos::Vector, scale::Vector; config::PlayerConfig, nature=false)
    # a bit of a hack
    opinion_dim = config.state_dims_per_activist[1]
    if length(point) != opinion_dim || length(scale[1]) != opinion_dim || length(point) != opinion_dim #Assert equal dimensions
        throw(DimensionMismatch("Opinion dimensions are not uniform, $(size(pos)), $(size(scale)), $(size(point))"))
    end
    mapreduce(+, zip(pos, scale)) do (pos_mode, scale_mode)
        mapreduce(+, 1:opinion_dim) do i
            (nature ? -1 : 1) * config.ellipsoidal_cost_weight* (point[i]-pos_mode[i])^2/scale_mode[i]
        end
    end
end

function control_cost(u::BlockVector, control_effort, agent_idx)
    control_effort * dot(u[Block(agent_idx)], u[Block(agent_idx)])
end

function base_non_terminal_cost_function_generator(config::PlayerConfig)
    function(beliefs::Beliefs, u::BlockVector)
        preference = sum(ellipsoidal_cost(pos, config.ellipsoid_centers, config.ellipsoid_radii; config=config, nature=config.type == nature) for pos in means(beliefs).blocks)
        control = control_cost(u, config.control_cost_weight, config.player_idx)
        return preference + control
    end
end
function base_terminal_cost_function_generator(config::PlayerConfig)
    function(beliefs::Beliefs)
        preference = sum(ellipsoidal_cost(pos, config.ellipsoid_centers, config.ellipsoid_radii; config=config, nature=config.type == nature) for pos in means(beliefs).blocks)
        return (config.type == nature ? -1 : 1) * config.terminal_cost_weight * preference
    end
end