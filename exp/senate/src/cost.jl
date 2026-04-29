function non_terminal_cost_components(preference_ellipsoids::Function, control_function::Function, senator_beliefs::BlockVector, u::BlockVector )
    preference = sum(preference_ellipsoids(pos) for pos in senator_beliefs.blocks)
    control = control_function(u)
    return (;preference, control)
end

function terminal_cost_components(preference_ellipsoids::Function, senator_beliefs::BlockVector)
    preference = sum(preference_ellipsoids(pos) for pos in senator_beliefs.blocks)
    return (;preference)
end

function ellipsoidal_cost(point::Vector, pos::Vector, scale::Vector; config::PlayerConfig)
    opinion_dim = config.state_dims_per_activist[1]
    if length(point) != opinion_dim || length(scale[1]) != opinion_dim || length(point) != opinion_dim #Assert equal dimensions
        throw(DimensionMismatch("Opinion dimensions are not uniform, $(size(pos)), $(size(scale)), $(size(point))"))
    end
    mapreduce(+, zip(pos, scale)) do (pos_mode, scale_mode)
        mapreduce(+, 1:opinion_dim) do i
            config.ellipsoidal_cost_weight * (point[i]-pos_mode[i])^2/scale_mode[i]
        end
    end
end

function control_cost(u::Vector, control_effort)
    control_effort * dot(u, u)
end

function base_non_terminal_cost_function_generator(config::PlayerConfig)
    pretend_config = config.type == nature ? config.nature_target_player_idx : config.player_idx
    player_belief_indices = (pretend_config-1) * config.num_senators + 1:pretend_config * config.num_senators
    player_control_indices = sum(config.control_dims_per_activist) * (config.player_idx-1) + 1:sum(config.control_dims_per_activist) * config.player_idx
    function(beliefs::Beliefs, u::BlockVector)
        preference = sum(ellipsoidal_cost(pos, config.ellipsoid_centers, config.ellipsoid_radii; config=config) for pos in means(beliefs).blocks[player_belief_indices])
        control = config.type == nature ? control_cost(u.blocks[end], config.control_cost_weight) : control_cost(u[player_control_indices], config.control_cost_weight)
        return (config.type == nature ? -1 : 1) * preference + control
    end
end

function base_terminal_cost_function_generator(config::PlayerConfig)
    pretend_config = config.type == nature ? config.nature_target_player_idx : config.player_idx
    player_belief_indices = (pretend_config-1) * config.num_senators + 1:pretend_config * config.num_senators
    function(beliefs::Beliefs)
        preference = sum(ellipsoidal_cost(pos, config.ellipsoid_centers, config.ellipsoid_radii; config=config) for pos in means(beliefs).blocks[player_belief_indices])
        return (config.type == nature ? -1 : 1) * config.terminal_cost_weight * preference
    end
end

function covariance_non_terminal_cost_function_generator(config::PlayerConfig)
    pretend_config = config.type == nature ? config.nature_target_player_idx : config.player_idx
    player_belief_indices = (pretend_config-1) * config.num_senators + 1:pretend_config * config.num_senators
    player_control_indices = sum(config.control_dims_per_activist) * (config.player_idx-1) + 1:sum(config.control_dims_per_activist) * config.player_idx
    function(beliefs::Beliefs, u::BlockVector)
        preference = sum(ellipsoidal_cost(pos, config.ellipsoid_centers, config.ellipsoid_radii; config=config) for pos in means(beliefs).blocks[player_belief_indices])
        control = config.type == nature ? control_cost(u.blocks[end], config.control_cost_weight) : control_cost(u[player_control_indices], config.control_cost_weight)
        cov_term = (config.type == nature ? -1 : 1) * config.covariance_weight * sum(tr(belief.belief_covariance) for belief in beliefs.beliefs[player_belief_indices])
        return (config.type == nature ? -1 : 1) * preference + control + cov_term
    end
end

function covariance_terminal_cost_function_generator(config::PlayerConfig)
    pretend_config = config.type == nature ? config.nature_target_player_idx : config.player_idx
    player_belief_indices = (pretend_config-1) * config.num_senators + 1:pretend_config * config.num_senators
    function(beliefs::Beliefs)
        preference = sum(ellipsoidal_cost(pos, config.ellipsoid_centers, config.ellipsoid_radii; config=config) for pos in means(beliefs).blocks[player_belief_indices])
        cov_term = (config.type == nature ? -1 : 1) * config.covariance_weight * sum(tr(belief.belief_covariance) for belief in beliefs.beliefs[player_belief_indices])
        return (config.type == nature ? -1 : 1) * config.terminal_cost_weight * preference + cov_term
    end
end

function obstacle_non_terminal_cost_function_generator(config::PlayerConfig)
    pretend_config = config.type == nature ? config.nature_target_player_idx : config.player_idx
    player_belief_indices = (pretend_config-1) * config.num_senators + 1:pretend_config * config.num_senators
    player_control_indices = sum(config.control_dims_per_activist) * (config.player_idx-1) + 1:sum(config.control_dims_per_activist) * config.player_idx
    function(beliefs::Beliefs, u::BlockVector)
        preference = sum(ellipsoidal_cost(pos, config.ellipsoid_centers, config.ellipsoid_radii; config=config) for pos in means(beliefs).blocks[player_belief_indices])
        control = config.type == nature ? control_cost(u.blocks[end], config.control_cost_weight) : control_cost(u[player_control_indices], config.control_cost_weight)
        cov_term = config.covariance_weight * sum(tr(belief.belief_covariance) for belief in beliefs.beliefs[player_belief_indices])
        obstacle_term = sum(config.obstacle_cost_function(belief, config) for belief in beliefs.beliefs[player_belief_indices])
        return (config.type == nature ? -1 : 1) * (preference + cov_term + obstacle_term) + control
    end
end

function obstacle_terminal_cost_function_generator(config::PlayerConfig)
    pretend_config = config.type == nature ? config.nature_target_player_idx : config.player_idx
    player_belief_indices = (pretend_config-1) * config.num_senators + 1:pretend_config * config.num_senators
    function(beliefs::Beliefs)
        preference = config.terminal_cost_weight * sum(ellipsoidal_cost(pos, config.ellipsoid_centers, config.ellipsoid_radii; config=config) for pos in means(beliefs).blocks[player_belief_indices])
        cov_term = config.covariance_weight * sum(tr(belief.belief_covariance) for belief in beliefs.beliefs[player_belief_indices])
        obstacle_term = sum(config.obstacle_cost_function(belief, config) for belief in beliefs.beliefs[player_belief_indices])
        return (config.type == nature ? -1 : 1) * (preference + cov_term + obstacle_term)
    end
end

function obstacle_cost(belief::Belief, config::PlayerConfig)
    mapreduce(+, zip(config.obstacle_centers, config.obstacle_sigmoid_scales, config.obstacle_sigmoid_offsets, config.obstacle_weights)) do (obstacle_center, sigmoid_scale, sigmoid_offset, obstacle_weight)
        dist_from_obstacle_center = norm(belief.belief_mean - obstacle_center)
        cov_adjusted_sigmoid_input = dist_from_obstacle_center - tr(belief.belief_covariance) #math.max(0,distance - sqrt(covariance))
        return obstacle_weight * 1/(1+exp(sigmoid_scale * (cov_adjusted_sigmoid_input - sigmoid_offset)))
    end
end 

function obstacle_cost_v2(belief::Belief, config::PlayerConfig)
    mapreduce(+, zip(config.obstacle_centers, config.obstacle_sigmoid_scales, config.obstacle_sigmoid_offsets, config.obstacle_weights)) do (obstacle_center, sigmoid_scale, sigmoid_offset, obstacle_weight)
        dist_from_obstacle_center = norm(belief.belief_mean - obstacle_center)
        cov_adjusted_sigmoid_input = dist_from_obstacle_center - tr(belief.belief_covariance)
        return obstacle_weight * exp(-sigmoid_scale * (cov_adjusted_sigmoid_input - sigmoid_offset))
    end
end

function obstacle_cost_v3(belief::Belief, config::PlayerConfig)
    mapreduce(+, zip(config.obstacle_centers, config.obstacle_sigmoid_scales, config.obstacle_sigmoid_offsets, config.obstacle_weights)) do (obstacle_center, sigmoid_scale, sigmoid_offset, obstacle_weight)
        dist_from_obstacle_center = norm(belief.belief_mean - obstacle_center)
        cov_adjusted_sigmoid_input = dist_from_obstacle_center - tr(belief.belief_covariance) - sigmoid_offset
        return obstacle_weight * (cov_adjusted_sigmoid_input < 3 ? cov_adjusted_sigmoid_input^2 : 0)
    end
end

# Option A: Divide by covariance instead of subtracting
# Higher uncertainty → lower cost (you could be anywhere, less certain you're hitting obstacle)
# Zero uncertainty + at obstacle → full cost (certain you're there)
function obstacle_cost_v4(belief::Belief, config::PlayerConfig)
    mapreduce(+, zip(config.obstacle_centers, config.obstacle_sigmoid_scales, config.obstacle_sigmoid_offsets, config.obstacle_weights)) do (obstacle_center, sigmoid_scale, sigmoid_offset, obstacle_weight)
        dist_from_obstacle_center = norm(belief.belief_mean - obstacle_center)
        base_cost = obstacle_weight * 1/(1+exp(sigmoid_scale * (dist_from_obstacle_center - sigmoid_offset)))
        confidence_weight = 1 / (1 + config.obstacle_covariance_scale * tr(belief.belief_covariance))
        return base_cost * confidence_weight
    end
end