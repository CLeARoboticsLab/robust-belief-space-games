
using BlockArrays
using LinearAlgebra
using RobustBeliefGame 

# Helper functions
function box_bounds(belief::Belief)
    # bottom = max(100 * exp(-(belief.belief_mean[2] + 5)) - 1, 0)
    bottom = (belief.belief_mean[2] < 0) ? 5 * belief.belief_mean[2]^2 : 0
    # top = max(100 * exp(belief.belief_mean[2] - 10) - 1, 0)
    top = (belief.belief_mean[2] > 3) ? 5 * belief.belief_mean[2]^2 : 0
    # left = max(100 * exp(-(belief.belief_mean[1]+8)) - 1, 0)
    left = (belief.belief_mean[1] < -3) ? 5 * belief.belief_mean[1]^2 : 0
    # right = max(100 * exp(belief.belief_mean[1] - 8) - 1, 0)
    right = (belief.belief_mean[1] > 3) ? 5 * belief.belief_mean[1]^2 : 0
    return 5 * (bottom + top + left + right)
end

function steal_liklihood(belief_over_attacker::Belief, belief_over_defender::Belief)
    attacker_pos = length(belief_over_attacker.belief_mean) == 4 ? belief_over_attacker.belief_mean[1:2] : belief_over_attacker.belief_mean
    defender_pos = length(belief_over_defender.belief_mean) == 4 ? belief_over_defender.belief_mean[1:2] : belief_over_defender.belief_mean

    sq_dist = dot(attacker_pos - defender_pos, attacker_pos - defender_pos)
    return -0.1 * sq_dist
end

function shot_probability(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false, goal_position=[[0.25, -1.5], [-0.25, -1.5]])
    attacker_pos = length(belief_over_attacker.belief_mean) == 4 ? belief_over_attacker.belief_mean[1:2] : belief_over_attacker.belief_mean
    defender_pos = length(belief_over_defender.belief_mean) == 4 ? belief_over_defender.belief_mean[1:2] : belief_over_defender.belief_mean
    
    attacker_pos_uncertainty = explicit_covariance ? 10 * tr(belief_over_attacker.belief_covariance) : 0
    defender_pos_uncertainty = explicit_covariance ? 10 * tr(belief_over_defender.belief_covariance) : 0
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
function attacker_non_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, us; explicit_covariance=false)
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender)
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender; explicit_covariance)
    control_effort = dot(us[Block(1)], us[Block(1)]) # Attacker is player 1
    attacker_covariance = tr(belief_over_attacker.belief_covariance)
    bounds = box_bounds(belief_over_attacker)
    if explicit_covariance
        return (; steal_prob, shot_prob = -1 * shot_prob, control_effort = 2 * control_effort, bounds, attacker_covariance)
    else
        return (; steal_prob, shot_prob = -1 * shot_prob, control_effort = 2 * control_effort, bounds)
    end
end

function defender_non_terminal_cost_components(
    belief_over_attacker::Belief, belief_over_defender::Belief, us; 
    explicit_covariance=false, control_cost_weight=0.5
)
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender)
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender; explicit_covariance)
    control_effort = dot(us[Block(2)], us[Block(2)]) # Defender is player 2
    bounds = box_bounds(belief_over_defender)
    defender_covariance = tr(belief_over_defender.belief_covariance)
    if explicit_covariance
        return (; steal_prob = -1 * steal_prob, shot_prob, control_effort = control_cost_weight * control_effort, bounds, defender_covariance)
    else
        return (; steal_prob=-1 * steal_prob, shot_prob, control_effort = control_cost_weight * control_effort, bounds)
    end
end

function nature_non_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, us::BlockVector; explicit_covariance=false, control_effort_weight=3)
    # Assuming standard/default control cost for defender here or should it match?
    # Nature tries to maximize defender's cost (or minimize it if adversarial?)
    # Nature minimizes the cost that defender maximizes? Or vice versa?
    # Actually nature usually has its own cost. In Senate it has a multiplier.
    # In Hockey.jl: 
    # defender_components = defender_non_terminal_cost_components(...)
    # defender_cost_val = defender_components.steal_prob + defender_components.shot_prob
    # return -1 * defender_cost_val ...
    
    # We'll use default control_cost_weight=0.5 for now to match old code if not passed, 
    # but strictly speaking nature might not care about defender's control effort directly unless it's part of the game.
    defender_components = defender_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us; explicit_covariance) 
    # Note: using default 0.5 here unless we thread it through
    
    defender_cost_val = defender_components.steal_prob + defender_components.shot_prob
    if hasproperty(defender_components, :defender_covariance)
        defender_cost_val += defender_components.defender_covariance
    end

    control_effort = 1000 * dot(us[Block(3)], us[Block(3)])
    bounds = 10 * (box_bounds(belief_over_attacker) + box_bounds(belief_over_defender))
    return (; defender_components = -1 * defender_cost_val, control_effort, bounds)
end

function attacker_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false)
    shot_prob = -1 * shot_probability(belief_over_attacker, belief_over_defender; explicit_covariance)
    bounds = box_bounds(belief_over_attacker)
    return (; shot_prob, bounds)
end

function defender_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false)
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender; explicit_covariance)
    bounds = box_bounds(belief_over_defender)
    return (; shot_prob, bounds)
end

function nature_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief)
    defender_components = defender_terminal_cost_components(belief_over_attacker, belief_over_defender)
    defender_cost_val = defender_components.shot_prob
    return (; defender_components = -1 * defender_cost_val + 10 * defender_components.bounds)
end

# Aggregators
attacker_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us; explicit_covariance=false) =
    sum(attacker_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us; explicit_covariance))

defender_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us; explicit_covariance=false, control_cost_weight=0.5) =
    sum(defender_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us; explicit_covariance, control_cost_weight))

nature_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us::BlockVector; explicit_covariance=false, control_effort_weight=3) =
    sum(nature_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us; explicit_covariance, control_effort_weight))

attacker_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false) =
    sum(attacker_terminal_cost_components(belief_over_attacker, belief_over_defender; explicit_covariance))

defender_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false) =
    sum(defender_terminal_cost_components(belief_over_attacker, belief_over_defender; explicit_covariance))

nature_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief) =
    sum(nature_terminal_cost_components(belief_over_attacker, belief_over_defender))

