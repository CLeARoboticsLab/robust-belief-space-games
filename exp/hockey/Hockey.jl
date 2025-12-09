module Hockey
using Infiltrator
using TrajectoryGamesBase
using TrajectoryGamesExamples
using RobustBeliefGame
using LinearAlgebra
using BlockArrays
using Makie
using Makie.GeometryBasics
using Symbolics
# using CairoMakie
# using GLMakie
using JLD2
using FileIO
using Distributions
using Random
using Statistics

include("../KKTErrorTracker.jl")
using .KKTErrorTracker

# include("./HockeyVisuals.jl")

export hockey_game, receding_horizon_main, attacker_cost, defender_cost, attacker_non_terminal_cost, defender_non_terminal_cost, nature_non_terminal_cost, attacker_terminal_cost, defender_terminal_cost, nature_terminal_cost, player_cost_components

#region: Environment Parameters
@enum PlayerID begin
    Attacker = 0
    Defender = 1
    Nature = 2
end
dt = 0.3
ϵ = eps()
n=2
state_dim = 4
control_dim = 2
goal_position = [[0.25, -1.5], [-0.25, -1.5]]
goal_center = (goal_position[1] + goal_position[2]) / 2
gt_initial_state = mortar([
    [0.0, 5.0, 0.5, 0.0],  # Attacker
    [0.0, 1.5, 0.0, 0.0], # Defender
])
initial_belief_covariance = [
    [0.1 0 0 0; 0 0.1 0 0; 0 0 0.25 0; 0 0 0 0.25],
    [0.1 0 0 0; 0 0.1 0 0; 0 0 0.25 0; 0 0 0 0.25],
]
#endregion

struct DummyEnvironment end

function TrajectoryGamesBase.get_constraints(::DummyEnvironment, player_index)
    (state) -> Symbolics.Num[]
end

function hockey_game(;
        n=2,
        player_costs = [
            (x, u) -> x[1]^2 + x[2]^2 + u[1]^2 + u[2]^2,
            (x, u) -> x[1]^2 + x[2]^2 + u[1]^2 + u[2]^2,
        ],
        environment = PolygonEnvironment(4, 50),
        single_dynamics = planar_double_integrator(;
            state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 4, 4]),
            control_bounds = (; lb = [-10, -10], ub = [3, 3]),
            dt = 0.3
        ),
        goal_position = nothing,
        horizon = 20, # actually set in main
        cost = [
            (xs, us) -> attacker_cost(xs, us; goal_position = goal_position),
            (xs, us) -> defender_cost(xs, us; goal_position = goal_position),
        ],
    )

    dynamics = ProductDynamics([single_dynamics for _ in 1:n])
    return TrajectoryGame(
        dynamics,
        cost,
        environment,
        nothing # constraints, add initial conditions as constraints
    )
end

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
    
    # attacker_pos_uncertainty = 10 * tr(belief_over_attacker.belief_covariance)
    # defender_pos_uncertainty = 10 * tr(belief_over_defender.belief_covariance)
    
    # return max(0, 10 - sq_dist) + attacker_pos_uncertainty - defender_pos_uncertainty
    # return max(0, 10 - sq_dist)
    return -0.1 * sq_dist
end

function shot_probability(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false)
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

    return -1 * dot(u, v) / (nv + nu + eps()) #+ -1 * nv proximity_factor * coverage_factor + positioning_penalty
end

function attacker_cost(xs, us; goal_position)
    return mapreduce(+, eachindex(xs)) do t
        -1 * shot_probability(xs[t][1:2], xs[t][5:6], goal_position[1], goal_position[2]) +
        -2 * dot(xs[t][1:2]-xs[t][5:6],xs[t][1:2]-xs[t][5:6]) + 
        0.05 * dot(us[t][1:2], us[t][1:2])
        # dot(xs[t][1:4], xs[t][1:4]) + 0.1*dot(us[t][1:2], us[t][1:2])
    end
end

function defender_cost(xs, us; goal_position)
    return mapreduce(+, eachindex(xs)) do t
        shot_probability(xs[t][1:2], xs[t][5:6], goal_position[1], goal_position[2]) +
        2 * dot(xs[t][1:2]-xs[t][5:6], xs[t][1:2]-xs[t][5:6]) +
        0.05 * dot(us[t][3:4], us[t][3:4])
        # dot(xs[t][5:8], xs[t][5:8]) + 0.1*dot(us[t][3:4], us[t][3:4])
    end
end

cost_dict = Dict(
    Attacker => attacker_cost,
    Defender => defender_cost,
    Nature => defender_cost
)

function main()
    horizon = 20
    initial_states = [
        [0.75, 5.0, 0.0, 0.0],  # Attacker
        [-0.75, 1.5, 0.0, 0.0],
    ]
    # A single goal defined by its two posts.
    goal_position = [
        [0.25, -1.5],
        [-0.25, -1.5],
    ]

    # Visual
    goal_line_width = 5

    game = hockey_game(;
        horizon=horizon,
        goal_position=goal_position,
    )
    mcp_game = MCPGame(game, horizon, vcat(initial_states...); debug=true)

    sol = solve(mcp_game; debug=true, warm_start=false)

    # Create figure
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1],
        title="Hockey Game Solution",
        xlabel="x position",
        ylabel="y position",
        aspect=1,
    )

    # Plot trajectories
    traj1 = lines!(ax, [x[Block(1)][1] for x in sol.xs], [x[Block(1)][2] for x in sol.xs], label="Attacker", color=:blue, linewidth=2)
    traj2 = lines!(ax, [x[Block(2)][1] for x in sol.xs], [x[Block(2)][2] for x in sol.xs], label="Defender", color=:red, linewidth=2)

    # Plot goal positions
    goal_posts = [[p[1] for p in goal_position], [p[2] for p in goal_position]]
    goal = lines!(ax, goal_posts[1], goal_posts[2], label="Goal", color=:green, linewidth=goal_line_width)


    # Plot initial positions
    init_atk = scatter!(ax, [sol.xs[1][Block(1)][1]], [sol.xs[1][Block(1)][2]], label="Attacker Start", color=:blue, markersize=15)
    init_def = scatter!(ax, [sol.xs[1][Block(2)][1]], [sol.xs[1][Block(2)][2]], label="Defender Start", color=:red, markersize=15)

    # Add arrows to show direction of movement
    for i in 2:2:length(sol.xs)-1
        arrows!(ax,
            [sol.xs[i][Block(1)][1]], [sol.xs[i][Block(1)][2]],
            [sol.xs[i][Block(1)][3] / norm(sol.xs[i][Block(1)][3:4])], [sol.xs[i][Block(1)][4] / norm(sol.xs[i][Block(1)][3:4])],
            color=:blue, arrowsize=10, lengthscale=0.1)
        arrows!(ax,
            [sol.xs[i][Block(2)][1]], [sol.xs[i][Block(2)][2]],
            [sol.xs[i][Block(2)][3] / norm(sol.xs[i][Block(2)][3:4])], [sol.xs[i][Block(2)][4] / norm(sol.xs[i][Block(2)][3:4])],
            color=:red, arrowsize=10, lengthscale=0.1)
    end

    Legend(
        fig[1, 2],
        # [traj1, traj2, goal, [init_atk, init_def], [final_atk, final_def]],
        [traj1, traj2, goal, [init_atk, init_def]],
        # ["Attacker Trajectory", "Defender Trajectory", "Goals", "Initial Positions", "Final Positions"]
        ["Attacker Trajectory", "Defender Trajectory", "Goals", "Initial Positions"]
    )


    # Display and save
    # display(fig)
    save("exp/hockey/outputs/hockey_solution.png", fig)
end

# Environment
    # Dynamics
function M_static(u)
    return 0.1 * I
end
function M_state_based(x)
    dist = dot(x[1:2] - goal_center + [0, 3], x[1:2] - goal_center + [0, 3])
    return 0.1 * I * dist
end
function f(xs::BlockVector, us::BlockVector, ms::BlockVector, player_idx::Union{Int,Nothing}=nothing)
    dt2 = 0.5 * dt^2

    valid_us = if !isnothing(player_idx)
        # If we know which player we are, we can just grab their controls
        u_p = us[Block(player_idx)]
        mortar([u_p])
    else
        us
    end

    BlockVector(
        mapreduce(vcat, zip(xs.blocks, valid_us.blocks, ms.blocks)) do (xᵢ, uᵢ, mᵢ)
            [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1] * xᵢ +
            [dt2 0; 0 dt2; dt 0; 0 dt] * uᵢ +
            M_static(xᵢ) * mᵢ
        end,
        length.(xs.blocks)
    )
end
#region: Sensor Models
function N_state_based(x)
    dist = dot(x[1:2] - goal_center + [0, 3], x[1:2] - goal_center + [0, 3])
    return 1 * I * dist
end

function h_state_based(xs::BlockVector, ns::BlockVector)
    BlockVector(
        mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
            I(state_dim) * xᵢ + N_state_based(xᵢ) * nᵢ
        end,
        length.(xs.blocks)
    )
end

function h_noise(xs::BlockVector, ns::BlockVector; I_mag::Float64 = 1.0) 
    BlockVector(
        mapreduce(vcat, zip(xs.blocks, ns.blocks)) do (xᵢ, nᵢ)
            I(state_dim) * xᵢ + I_mag * I * nᵢ
        end,
        length.(xs.blocks)
    )
end

h_low_noise(xs::BlockVector, ns::BlockVector) = h_noise(xs, ns; I_mag = 0.1)
h_mid_noise(xs::BlockVector, ns::BlockVector) = h_noise(xs, ns; I_mag = 1.0)
h_high_noise(xs::BlockVector, ns::BlockVector) = h_noise(xs,ns; I_mag = 10.0)

h_noise_dict = Dict(
    "low" => h_low_noise,
    "medium" => h_mid_noise,
    "high" => h_high_noise,
)
#endregion

#region: Cost
#TODO: merge cost functions [differentiate using enum]
function attacker_non_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, us; explicit_covariance=false)
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender)
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender)
    control_effort = dot(us[Block(1)], us[Block(1)]) # Attacker is player 1
    attacker_covariance = tr(belief_over_attacker.belief_covariance)
    bounds = box_bounds(belief_over_attacker)
    if explicit_covariance
        return (; steal_prob, shot_prob = -2 * shot_prob, control_effort, bounds, attacker_covariance)
    else
        return (; steal_prob, shot_prob = -2 * shot_prob, control_effort, bounds)
    end
end

function defender_non_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, us; explicit_covariance=false)
    steal_prob = steal_liklihood(belief_over_attacker, belief_over_defender)
    shot_prob = shot_probability(belief_over_attacker, belief_over_defender)
    control_effort = dot(us[Block(2)], us[Block(2)]) # Defender is player 2
    bounds = box_bounds(belief_over_defender)
    defender_covariance = tr(belief_over_defender.belief_covariance)
    if explicit_covariance
        return (; steal_prob = -1 * steal_prob, shot_prob, control_effort = 0.5 * control_effort, bounds, defender_covariance)
    else
        return (; steal_prob=-1 * steal_prob, shot_prob, control_effort=5 * control_effort, bounds)
    end
end

function nature_non_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief, us::BlockVector; explicit_covariance=false, control_effort_weight=3)
    defender_components = defender_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us; explicit_covariance)
    defender_cost_val = defender_components.steal_prob + defender_components.shot_prob + defender_components.bounds
    if hasproperty(defender_components, :defender_covariance)
        defender_cost_val += defender_components.defender_covariance
    end

    control_effort = control_effort_weight * dot(us[Block(3)], us[Block(3)])
    bounds = box_bounds(belief_over_attacker) + box_bounds(belief_over_defender)
    return (; defender_components = -sum(defender_components), control_effort, bounds)
end

function attacker_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false)
    shot_prob = -5 * shot_probability(belief_over_attacker, belief_over_defender)
    bounds = box_bounds(belief_over_attacker)
    return (; shot_prob, bounds)
end

function defender_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false)
    shot_prob = 5 * shot_probability(belief_over_attacker, belief_over_defender)
    bounds = box_bounds(belief_over_defender)
    return (; shot_prob, bounds)
end

function nature_terminal_cost_components(belief_over_attacker::Belief, belief_over_defender::Belief)
    defender_components = defender_terminal_cost_components(belief_over_attacker, belief_over_defender)
    bounds = box_bounds(belief_over_attacker) + box_bounds(belief_over_defender)
    return (; defender_components = -sum(defender_components), bounds)
end

#region: Component sum wrappers
attacker_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us; explicit_covariance=false) =
    sum(attacker_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us; explicit_covariance))

defender_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us; explicit_covariance=false) =
    sum(defender_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us; explicit_covariance))

nature_non_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief, us::BlockVector; explicit_covariance=false, control_effort_weight=3) =
    sum(nature_non_terminal_cost_components(belief_over_attacker, belief_over_defender, us; explicit_covariance, control_effort_weight))

attacker_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false) =
    sum(attacker_terminal_cost_components(belief_over_attacker, belief_over_defender; explicit_covariance))

defender_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief; explicit_covariance=false) =
    sum(defender_terminal_cost_components(belief_over_attacker, belief_over_defender; explicit_covariance))

nature_terminal_cost(belief_over_attacker::Belief, belief_over_defender::Belief) =
    sum(nature_terminal_cost_components(belief_over_attacker, belief_over_defender))

non_terminal_cost_dict = Dict(
    Attacker => attacker_non_terminal_cost,
    Defender => defender_non_terminal_cost,
    Nature => nature_non_terminal_cost   
)
terminal_cost_dict = Dict(
    Attacker => attacker_terminal_cost,
    Defender => defender_terminal_cost,
    Nature => nature_terminal_cost
)
#endregion

const player_cost_components = (
    attacker = (
        non_terminal = attacker_non_terminal_cost_components,
        terminal = attacker_terminal_cost_components,
    ),
    defender = (
        non_terminal = defender_non_terminal_cost_components,
        terminal = defender_terminal_cost_components,
    ),
    nature = (
        non_terminal = nature_non_terminal_cost_components,
        terminal = nature_terminal_cost_components,
    )
)
#endregion

function belief_main(sol_number=2, override_solution=false)
    solution_filename = "exp/hockey/outputs/hockey_solution_$sol_number.jld2"

    local robust_sol, non_robust_sol
    global goal_position

    if isfile(solution_filename) && !override_solution
        println("Loading solution from $solution_filename")
        @load solution_filename robust_sol non_robust_sol goal_position
    else
        !override_solution && println("No solution file found. Running solver...")
        override_solution && println("Overriding solution...")
        # Game Params
        horizon = 5

        # Initial States/Beliefs
        gt_initial_state = mortar([ # gt = ground truth
        [0.75, 5.0, 0.0, 0.0],  # Attacker
        [-0.75, 1.5, 0.0, 0.0],
        ])
        initial_belief_covariance = [
            [0 0 0    0;
            0 0 0    0;
            0 0 0.25 0;
            0 0 0    0.25],
            [0 0 0    0;
            0 0 0    0;
            0 0 0.25 0;
            0 0 0    0.25],
        ]
        initial_beliefs = Beliefs([Belief(gt_initial_state[Block(i)], initial_belief_covariance[i]) for i in 1:2])

        environments = [BeliefEnvironment(f, gt_initial_state, h_low_noise) for _ in 1:2]

        attacker_cost = BeliefCost(
            (bs, us) -> attacker_non_terminal_cost(bs.beliefs[1], bs.beliefs[2], us),
            (bs) -> attacker_terminal_cost(bs.beliefs[1], bs.beliefs[2]),
        )
        defender_cost = BeliefCost(
            (bs, us) -> defender_non_terminal_cost(bs.beliefs[1], bs.beliefs[2], us),
            (bs) -> defender_terminal_cost(bs.beliefs[1], bs.beliefs[2]),
        )
        nature_cost = BeliefCost(
            (bs, us) -> nature_non_terminal_cost(bs.beliefs[1], bs.beliefs[2], us; control_effort_weight=10),
            (bs) -> nature_terminal_cost(bs.beliefs[1], bs.beliefs[2]),
        )
        dims = (; n=2,
            num_players=2,
            control_blocks_per_player=1,
            player_state_dims=length.(gt_initial_state.blocks),
            num_beliefs_per_player=[1, 1],
            states=length.(gt_initial_state.blocks),
            total_states_dim=length.(gt_initial_state.blocks),
            controls=[2, 2],
            total_controls_dim=[2, 2],
            belief=length.(gt_initial_state.blocks),
            sensor=[2, 2],
            nature_controls_dim=4)

        non_robust_hockey_game = BeliefGame(
            environments,
            [attacker_cost, defender_cost],
            initial_beliefs,
            horizon,
            dims,
            gt_initial_state,
            Int[],
        )
        robust_hockey_game = BeliefGame(
            environments,
            [attacker_cost, defender_cost, nature_cost],
            initial_beliefs,
            horizon,
            dims,
            gt_initial_state,
            [2],
        )
        non_robust_sol = @time solve(non_robust_hockey_game; debug=false)
        robust_sol = @time solve(robust_hockey_game; debug=false)
        println("Saving solution to $solution_filename")
        @save solution_filename robust_sol non_robust_sol goal_position
    end
    # plot_feed_forward_norms(robust_sol[4])
    # visualize_belief_hockey_solution(robust_sol, non_robust_sol, goal_position)
end

function safe_eigen(A) #Why not just override eigen. Isn't this strictly better. - Henry
    # try
        A_reg = A + ϵ * I(size(A, 1))
        E = eigen(A_reg)
        return (values = max.(E.values, ϵ), vectors = E.vectors)
    # catch
    #     n = size(A, 1)
    #     return (values = fill(ϵ, n), vectors = Matrix(I, n, n))
    # end
end

#
function receding_horizon_main(file_id::String=""; horizon=10, planning_horizon=5, override=false, random_seed=1, explicit_covariance=false, trials=10)
    global goal_position
    local gt_initial_state = deepcopy(Hockey.gt_initial_state)
    local initial_belief_covariance = deepcopy(Hockey.initial_belief_covariance)

    if isfile("exp/hockey/outputs/rh_$file_id.jld2") && !override
        println("Loading solution from exp/hockey/outputs/rh_$file_id.jld2")
        @load "exp/hockey/outputs/rh_$file_id.jld2" solutions goal_position
        # visualize_receding_horizon_solution(
        #     solutions, 
        #     goal_position;
        #     dims=(; n=2, states=[2, 2], controls=[2, 2], belief=[2, 2, 2, 2], sensor=[2, 2, 2, 2])
        # )
        return
    end

    initial_beliefs = Beliefs([
        Belief(gt_initial_state[Block(1)], initial_belief_covariance[1]), # Attacker's belief of attacker
        Belief(gt_initial_state[Block(2)], initial_belief_covariance[2]), # Attacker's belief of defender
        Belief(gt_initial_state[Block(1)], initial_belief_covariance[1]), # Defender's belief of attacker
        Belief(gt_initial_state[Block(2)], initial_belief_covariance[2]), # Defender's belief of defender
    ])
    attacker_cost = BeliefCost(
        (bs, us) -> attacker_non_terminal_cost(bs.beliefs[1], bs.beliefs[2], us; explicit_covariance=explicit_covariance),
        (bs) -> attacker_terminal_cost(bs.beliefs[1], bs.beliefs[2])
    )
    defender_cost = BeliefCost(
        (bs, us) -> defender_non_terminal_cost(bs.beliefs[3], bs.beliefs[4], us; explicit_covariance=explicit_covariance),
        (bs) -> defender_terminal_cost(bs.beliefs[3], bs.beliefs[4])
    )
    nature_cost = BeliefCost(
        (bs, us) -> nature_non_terminal_cost(bs.beliefs[3], bs.beliefs[4], us; explicit_covariance=explicit_covariance),
        (bs) -> nature_terminal_cost(bs.beliefs[3], bs.beliefs[4])
    )
    
    # --- Shared Parameters ---
    dims = (; n=2, num_players=2, control_blocks_per_player=1, player_state_dims=length.(gt_initial_state.blocks), num_beliefs_per_player=[2, 2],
        states=length.(gt_initial_state.blocks), controls=[control_dim for _ in 1:2], total_controls_dim=[control_dim for _ in 1:2], belief=[state_dim for _ in 1:4], sensor=[state_dim for _ in 1:4])
    costs = [[attacker_cost, defender_cost], [attacker_cost, defender_cost, nature_cost]]
    # --- Run Scenarios ---
    for trial in 1:trials
        solutions = Dict()
        for type in [([false,true], "robust"),([false,false], "non_robust")]
            robust, type_str = type
            for int in ["low","medium","high"]
                println("--- Running $(uppercasefirst(type_str)) $(uppercasefirst(int)) Noise Sensor Scenario (Trial $trial) ---")
                Random.seed!(random_seed)
                noise = h_noise_dict[int]
                solutions["$(int)_$(type_str)_$trial"] = run_receding_horizon_scenario(
                    gt_initial_state, initial_beliefs, costs, robust, dims,
                    horizon, planning_horizon, random_seed,
                    (f, gt_initial_state, [noise, noise]), # environment
                    (current_beliefs, u, environments, observations) -> ekf_update_with_observations(current_beliefs, u, environments, observations), # ekf_update
                    trial,
                    "$(int)_$(type_str)_$trial"
                )
            end
        end
        @save "exp/hockey/outputs/rh_$(file_id)_$trial.jld2" solutions goal_position
        println("Saved solution with file id: $file_id, trial $trial")
    end
    
    # visualize_receding_horizon_solution(
    #     solutions,
    #     goal_position;
    #     dims=dims
    # )
end

function run_receding_horizon_scenario(
    gt_initial_state, initial_beliefs, costs, robust, dims,
    horizon, planning_horizon, random_seed,
    environment_params, ekf_update_fn, trial_number::Int=1, scenario_name::String="unknown"
)
    f, _, h = environment_params
    environments::Vector{BeliefEnvironment} = [BeliefEnvironment(f, gt_initial_state, h) for h in h]

    # --- Solve LQ Game ---
    lq_sol_history = []
    
    current_beliefs = initial_beliefs
    current_gt_state = gt_initial_state
    all_observations = []
    gt_state_history = [current_gt_state]
    solution_history = []
    cond_history = []
    warm_starts = Vector{Any}([nothing, nothing])

    Random.seed!(random_seed)
    normal_distribution = MvNormal(zeros(sum(dims.states)), I(sum(dims.states)))
    draw_from_normal = () -> BlockVector(rand(normal_distribution), dims.states)
    draw_from_normal_fake = () -> BlockVector(zeros(sum(dims.states)), dims.states)

    for t in 1:horizon-1
        println("Receding Horizon Step $t / $horizon")
        
        lq_horizon = 10
        lq_initial_states = [
            [current_gt_state[Block(1)]..., 0.0, 0.0],
            [current_gt_state[Block(2)]..., 0.0, 0.0]
        ]
        lq_game = hockey_game(; horizon = lq_horizon, goal_position = goal_position)
        mcp_game = MCPGame(lq_game, lq_horizon, vcat(lq_initial_states...); debug=false)
        lq_sol = solve(mcp_game; debug=false, warm_start=false)
        push!(lq_sol_history, lq_sol)

        sols = Vector{Any}(undef, dims.n)
        for ii in 1:dims.n
            game = BeliefGame(
                environments,
                costs[ii],
                current_beliefs,
                min(planning_horizon, horizon - t + 1),
                dims,
                current_gt_state,
                robust[ii])
            # nominal_beliefs, nominal_controls, _, _, _, cond = solve(game; debug=true, warm_start=warm_starts[ii], save_intermediate_solutions=true)
            nominal_beliefs, nominal_controls, kkt_error_norms = solve(game; debug=true, warm_start=warm_starts[ii], save_intermediate_solutions=false)
            
            # Track KKT error for this BeliefGame solve
            if !isnothing(kkt_error_norms)
                # Convert the kkt_error_norms to individual trajectory values
                kkt_trajectory = norm.(kkt_error_norms)
                KKTErrorTracker.record_rh_kkt_error!(
                    "BeliefGame",
                    kkt_trajectory,
                    trial_number,
                    t;
                    player=ii,
                    robust=robust[ii],
                    iteration_count=-1,
                    convergence_status=:unknown,
                    additional_data=Dict{String,Any}(
                        "scenario_name" => scenario_name,
                        "planning_horizon" => min(planning_horizon, horizon - t + 1),
                        "warm_start_used" => !isnothing(warm_starts[ii])
                    )
                )
            end
            
            if length(nominal_beliefs) > 1 && length(nominal_controls) > 1
                shifted_beliefs = nominal_beliefs[2:end]
                shifted_controls = nominal_controls[2:end]
                # Create zero control with correct dimensions for robust/non-robust cases
                if robust[ii]
                    # For robust case, add disturbance control block
                    control_block_sizes = vcat(dims.controls, sum(dims.states))
                    zero_control = BlockVector(zeros(sum(control_block_sizes)), control_block_sizes)
                else
                    zero_control = BlockVector(zeros(sum(dims.controls)), dims.controls)
                end
                last_belief = shifted_beliefs[end]
                g, W = ekf_update(last_belief, zero_control, environments[ii].dynamics, environments[ii].sensor_models; is_robust=robust[ii])
                extended_belief = unvec(g, game.dims.belief)

                warm_start_beliefs = vcat(shifted_beliefs, [extended_belief])
                warm_start_controls = vcat(shifted_controls, [zero_control])
                warm_starts[ii] = (warm_start_beliefs, warm_start_controls)
            else
                warm_starts[ii] = (nominal_beliefs, nominal_controls)
            end
            sols[ii] = (nominal_beliefs, nominal_controls)
            #push!(cond_history, cond)
        end
        push!(solution_history, sols)
        u = mortar([sols[ii][2][1][Block(ii)] for ii in 1:dims.n])
        current_gt_state = f(current_gt_state, u, draw_from_normal_fake())
        
        observations = mortar([h[ii](current_gt_state, draw_from_normal()) for ii in 1:dims.n])
        current_beliefs = ekf_update_fn(current_beliefs, u, environments, observations)
        current_beliefs.beliefs[2] = copy(current_beliefs.beliefs[4])
        current_beliefs.beliefs[3] = copy(current_beliefs.beliefs[1])
        push!(gt_state_history, current_gt_state)
        push!(all_observations, observations)
    end
    return (gt_state_history, all_observations, solution_history, cond_history, lq_sol_history)
end
end