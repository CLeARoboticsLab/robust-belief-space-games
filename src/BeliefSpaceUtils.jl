import Base.vec
import Base: round
using ForwardDiff: Dual

struct Belief
    belief_mean::Vector
    belief_covariance::Symmetric
    belief_dim::Int
end

function Belief(belief_mean::Vector, belief_covariance::Symmetric)
    return Belief(belief_mean, belief_covariance, length(belief_mean))
end

function Belief(belief_mean::Vector, belief_covariance::Matrix)
    return Belief(belief_mean, Symmetric(belief_covariance), length(belief_mean))
end

function Belief(vectorized_belief::Vector, dims::Int)
    return Belief(vectorized_belief[1:dims], Symmetric(reshape(vectorized_belief[dims+1:end], dims, dims)), dims)
end

function vec(belief::Belief)
    return [belief.belief_mean; Base.vec(belief.belief_covariance)]
end

struct Beliefs
    beliefs::Vector{Belief}
end

function means(beliefs::Beliefs)
    return mortar([belief.belief_mean for belief in beliefs.beliefs])
end

function covs(beliefs::Beliefs)
    return [belief.belief_covariance for belief in beliefs.beliefs]
end

function dims(beliefs::Beliefs)
    return [belief.belief_dim for belief in beliefs.beliefs]
end

function total_size(belief::Belief)
    return belief.belief_dim + belief.belief_dim ^ 2
end

function total_size(beliefs::Beliefs)
    return sum(total_size(belief) for belief in beliefs.beliefs)
end

function vec(beliefs::Beliefs)
    vcat(means(beliefs), vcat([reshape(cov, (beliefs.beliefs[ii].belief_dim^2,)) for (ii, cov) in enumerate(covs(beliefs))]...))
end

function unvec(vec_beliefs::Vector, dims::Vector{Int})
    belief_means = [vec_beliefs[sum(dims[1:i-1])+1:sum(dims[1:i])] for i in eachindex(dims)]
    belief_covs = [vec_beliefs[sum(dims) + sum(dims[1:i-1].^2)+1:sum(dims)+sum(dims[1:i].^2)] for i in eachindex(dims)]
    return Beliefs(map(eachindex(belief_means)) do i
        Belief(belief_means[i], reshape(belief_covs[i], (dims[i], dims[i])))
    end)
end

function Base.:-(b1::Beliefs, b2::Beliefs)
    return vec(b1) - vec(b2)
end

function Base.:-(b1::Belief, b2::Belief)
    return vec(b1) - vec(b2)
end

struct BeliefCost{N, T}
    non_terminal_cost::N
    terminal_cost::T
end

struct BeliefEnvironment{D, S}
    dynamics::D
    gt_states::BlockVector
    sensor_models::S
end

struct BeliefGame{E, C}
    environment::E
    costs::C
    initial_beliefs::Beliefs
    horizon::Int
    dims::NamedTuple{(:n, :states, :controls, :belief, :sensor)}
    gt_initial_state::BlockVector
    is_robust::Bool # Assuming player 1 is robust
end

function clip(x, max_norm)
    norm = LinearAlgebra.norm(x)
    if norm > max_norm
        return x .* (max_norm / norm)
    else
        return x
    end
end

function dual_round(x; kwargs...)
    x isa Dual ? x : round(x; kwargs...)
end

function rollout_strategy(game::BeliefGame, strategy::Vector)
    H = length(strategy)
    beliefs = Vector{Beliefs}(undef, H + 1)
    controls = Vector{BlockVector}(undef, H)

    initial_beliefs_vec = [Belief(copy(b.belief_mean), copy(b.belief_covariance)) for b in game.initial_beliefs.beliefs]
    beliefs[1] = Beliefs(initial_beliefs_vec)

    for i in 1:H
        controls[i] = BlockVector(strategy[i](beliefs[i]),
            game.is_robust ? vcat(game.dims.controls..., game.dims.belief[1]) : game.dims.controls)
        g, W = ekf_update(beliefs[i], controls[i], game.environment.dynamics, game.environment.sensor_models; is_robust=game.is_robust)
        beliefs[i+1] = unvec(g, game.dims.belief)
    end
    
    return beliefs, controls
end
