
using BlockArrays
using LinearAlgebra
using RobustBeliefGame 

# Helper functions
function box_bounds(belief::Belief, weight::Float64=5.0)
    # bottom = max(100 * exp(-(belief.belief_mean[2] + 5)) - 1, 0)
    bottom = (belief.belief_mean[2] < 0) ? weight * belief.belief_mean[2]^2 : 0.0
    # top = max(100 * exp(belief.belief_mean[2] - 10) - 1, 0)
    top = (belief.belief_mean[2] > 3) ? weight * belief.belief_mean[2]^2 : 0.0
    # left = max(100 * exp(-(belief.belief_mean[1]+8)) - 1, 0)
    left = (belief.belief_mean[1] < -3) ? weight * belief.belief_mean[1]^2 : 0.0
    # right = max(100 * exp(belief.belief_mean[1] - 8) - 1, 0)
    right = (belief.belief_mean[1] > 3) ? weight * belief.belief_mean[1]^2 : 0.0
    return weight * (bottom + top + left + right)
end

function steal_liklihood(belief_over_attacker::Belief, belief_over_defender::Belief, config::PlayerConfig)
    attacker_pos = length(belief_over_attacker.belief_mean) == 4 ? belief_over_attacker.belief_mean[1:2] : belief_over_attacker.belief_mean
    defender_pos = length(belief_over_defender.belief_mean) == 4 ? belief_over_defender.belief_mean[1:2] : belief_over_defender.belief_mean

    sq_dist = dot(attacker_pos - defender_pos, attacker_pos - defender_pos)
    weight = config.steal_dist_weight 
    return -weight * sq_dist
end

function shot_probability(belief_over_attacker::Belief, belief_over_defender::Belief, config::PlayerConfig; explicit_covariance=false, goal_position=[[0.25, -1.5], [-0.25, -1.5]])
    attacker_pos = length(belief_over_attacker.belief_mean) == 4 ? belief_over_attacker.belief_mean[1:2] : belief_over_attacker.belief_mean
    defender_pos = length(belief_over_defender.belief_mean) == 4 ? belief_over_defender.belief_mean[1:2] : belief_over_defender.belief_mean
    
    weight = config.shot_uncertainty_weight
    
    attacker_pos_uncertainty = explicit_covariance ? weight * tr(belief_over_attacker.belief_covariance) : 0
    defender_pos_uncertainty = explicit_covariance ? weight * tr(belief_over_defender.belief_covariance) : 0
    return shot_probability(attacker_pos, defender_pos, goal_position[1], goal_position[2]) - attacker_pos_uncertainty + defender_pos_uncertainty
end

function shot_probability(attacker_pos, defender_pos, goal_p1, goal_p2)
    u = defender_pos - attacker_pos
    v = (goal_p1 + goal_p2) / 2 - attacker_pos

    nu = dot(u, u)
    nv = dot(v, v)

    return -1 * dot(u, v) / (nv + nu + eps()) 
end

# Cost Components
function attacker_non_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, us, params::HockeyParams; explicit_covariance=false)
    config = params.player_configs[1]
    
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender, config)
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender, config; explicit_covariance, goal_position=params.goal_position)
    control_effort = dot(us[Block(1)], us[Block(1)]) # Attacker is player 1
    attacker_covariance = tr(belief_over_attacker.belief_covariance)
    bounds = box_bounds(belief_over_attacker, config.boundary_cost_weight)
    
    control_cost = config.control_cost_weight * control_effort
    
    if explicit_covariance
        return (; steal_prob, shot_prob = -1 * shot_prob, control_effort = control_cost, bounds, attacker_covariance)
    else
        return (; steal_prob, shot_prob = -1 * shot_prob, control_effort = control_cost, bounds)
    end
end

function defender_non_terminal_cost_components(
    belief_over_attacker::Belief, belief_over_defender::Belief, us, params::HockeyParams; 
    explicit_covariance=false
)
    config = params.player_configs[2]

    # Use defender's config for steal/shot weights
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender, config)
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender, config; explicit_covariance, goal_position=params.goal_position)
    control_effort = dot(us[Block(2)], us[Block(2)]) # Defender is player 2
    bounds = box_bounds(belief_over_defender, config.boundary_cost_weight)
    defender_covariance = tr(belief_over_defender.belief_covariance)
    
    control_cost = config.control_cost_weight * control_effort

    if explicit_covariance
        return (; steal_prob = -1 * steal_prob, shot_prob, control_effort = control_cost, bounds, defender_covariance)
    else
        return (; steal_prob=-1 * steal_prob, shot_prob, control_effort = control_cost, bounds)
    end
end

function nature_non_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, us::BlockVector, params::HockeyParams; explicit_covariance=false)
    # Nature parameters from Robust player (Defender, idx 2)
    config = params.player_configs[2]
    
    defender_components = defender_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us, params; explicit_covariance) 
    
    defender_cost_val = defender_components.steal_prob + defender_components.shot_prob
    if hasproperty(defender_components, :defender_covariance)
        defender_cost_val += defender_components.defender_covariance
    end

    control_effort = config.nature_control_cost_weight * dot(us[Block(3)], us[Block(3)])
    bounds = config.nature_bounds_cost_weight * (box_bounds(belief_over_attacker, config.boundary_cost_weight) + box_bounds(belief_over_defender, config.boundary_cost_weight))
    return (; defender_components = -1 * defender_cost_val, control_effort, bounds)
end

function attacker_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, params::HockeyParams; explicit_covariance=false)
    config = params.player_configs[1]
    shot_prob = -1 * shot_probability(belief_over_attacker, belief_over_defender, config; explicit_covariance, goal_position=params.goal_position)
    bounds = box_bounds(belief_over_attacker, config.boundary_cost_weight)
    return (; shot_prob, bounds)
end

function defender_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, params::HockeyParams; explicit_covariance=false)
    config = params.player_configs[2]
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender, config; explicit_covariance, goal_position=params.goal_position)
    bounds = box_bounds(belief_over_defender, config.boundary_cost_weight)
    return (; shot_prob, bounds)
end

function nature_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, params::HockeyParams)
    # Nature parameters from Robust player (Defender, idx 2)
    config = params.player_configs[2]

    defender_components = defender_terminal_cost_components(belief_over_attacker, belief_over_defender, params)
    defender_cost_val = defender_components.shot_prob
    return (; defender_components = -1 * defender_cost_val + config.nature_bounds_cost_weight * defender_components.bounds)
end

# Aggregators
attacker_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us, params::HockeyParams; explicit_covariance=false) =
    sum(attacker_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us, params; explicit_covariance))

defender_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us, params::HockeyParams; explicit_covariance=false) =
    sum(defender_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us, params; explicit_covariance))

nature_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us::BlockVector, params::HockeyParams; explicit_covariance=false) =
    sum(nature_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us, params; explicit_covariance))

attacker_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, params::HockeyParams; explicit_covariance=false) =
    sum(attacker_terminal_cost_components(belief_over_attacker, belief_over_defender, params; explicit_covariance))

defender_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, params::HockeyParams; explicit_covariance=false) =
    sum(defender_terminal_cost_components(belief_over_attacker, belief_over_defender, params; explicit_covariance))

nature_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, params::HockeyParams) =
    sum(nature_terminal_cost_components(belief_over_attacker, belief_over_defender, params))

