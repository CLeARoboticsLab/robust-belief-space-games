using Infiltrator
using TrajectoryGamesBase
using TrajectoryGamesBase



function hockey_game(
        n=2,
        player_costs = [
            (x, u) -> x[1]^2 + x[2]^2 + u[1]^2 + u[2]^2,
            (x, u) -> x[1]^2 + x[2]^2 + u[1]^2 + u[2]^2,
        ],
        environment = PolygonEnvironment(4, 8),
        single_dynamics = planar_double_integrator(;
            state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
            control_bounds = (; lb = [-10, -10], ub = [10, 10]),
        ),
        goal_position = [
            [-1.5, 0.0],
            [1.5, 0.0],
        ],
        costs = [
            (xs, us) -> attacker_cost(xs, us; goal_position = goal_position),
            (xs, us) -> defender_cost(xs, us; goal_position = goal_position),
        ],
    )
    dynamics = ProductDynamics([single_dynamics for _ in 1:n])
    return TrajectoryGame(
        dynamics,
        costs,
        environment,
        nothing # constraints
    )
end


function attacker_cost(xs, us; goal_position)
    return shot_probability(xs[1:2], xs[5:6], goal_position) + 0.1 * norm(us[1:2])
end

function defender_cost(xs, us; goal_position)
    return -1 * shot_probability(xs[1:2], xs[5:6], goal_position) + 0.1 * norm(us[3:4])
end

function shot_probability(attacker_pos, defender_pos, goal_pos)
    attacker_shooting_angle = find_angle(attacker_pos, goal_pos[1], goal_pos[2])

    defender_guard = 0.5 * [0.0 1.0; -1.0 0.0] * ((defender_pos - attacker_pos) ./ norm(defender_pos - attacker_pos))
    defender_blocking_angle = find_angle(defender_pos, attacker_pos + defender_guard, attacker_pos - defender_guard)

    return attacker_shooting_angle - defender_blocking_angle
end

function find_angle(p1, p2, p3)
    v1 = p2 .- p1
    v2 = p3 .- p1
    
    dot_product = dot(v1, v2)
    mag_v1 = norm(v1)
    mag_v2 = norm(v2)
    
    cos_angle = dot_product / (mag_v1 * mag_v2)
    @assert cos_angle >= -1.0 && cos_angle <= 1.0 "cos_angle is out of bounds"
    # cos_angle = clamp(cos_angle, -1.0, 1.0)
    
    return acos(cos_angle)
end


function main()
    game = hockey_game()
    
end