import Base.vec
import Base: round, copy
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
function Base.copy(belief::Belief)
    return Belief(copy(belief.belief_mean), copy(belief.belief_covariance), belief.belief_dim)
end

function Base.copy(beliefs::Beliefs)
    return Beliefs([copy(belief) for belief in beliefs.beliefs])
end

Base.iterate(b::Beliefs, state...) = iterate(b.beliefs, state...)
Base.length(b::Beliefs) = length(b.beliefs)

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

function calculate_costs(game::BeliefGame, beliefs::Vector{Beliefs}, controls::Vector{BlockVector})
    return map(1:(game.dims.n+game.is_robust)) do ii
        mapreduce(+, 1:game.horizon - 1, init=0.0) do t
            game.costs[ii].non_terminal_cost(beliefs[t], controls[t])
        end +
        game.costs[ii].terminal_cost(beliefs[end])
    end
end

function clip(x, max_norm)
    norm = LinearAlgebra.norm(x)
    if norm > max_norm
        return x .* (max_norm / norm)
    else
        return x
    end
end

dual_round(x::Dual; kwargs...) = x

dual_round(x::Number; kwargs...) = _fix_neg_zero(round(x; kwargs...))

dual_round(x::AbstractArray; kwargs...) = map(y -> dual_round(y; kwargs...), x)

dual_round(x::Symmetric; kwargs...) = Symmetric(dual_round(Matrix(x); kwargs...))

_fix_neg_zero(x::AbstractFloat) = iszero(x) ? zero(x) : x
_fix_neg_zero(x) = x

function rollout_strategy(game::BeliefGame, strategy::Vector)
    H = length(strategy)
    beliefs = Vector{Beliefs}(undef, H + 1)
    controls = Vector{BlockVector}(undef, H)

    initial_beliefs_vec = [Belief(copy(b.belief_mean), copy(b.belief_covariance)) for b in game.initial_beliefs.beliefs]
    beliefs[1] = Beliefs(initial_beliefs_vec)

    for i in 1:H
        controls[i] = strategy[i](beliefs[i])
        if DEBUG
            open(DEBUG_FILE, "a") do f
                println(f, "[rollout_strategy]")
                println(f, "beliefs[$i]:")
                display_matrix = IOContext(f, :limit=>false)
                show(display_matrix, "text/plain", beliefs[i])
                println(f)
                println(f, "controls[$i]:")
                display_matrix = IOContext(f, :limit=>false)
                show(display_matrix, "text/plain", controls[i])
                println(f)
            end
        end
        g, W = ekf_update(beliefs[i], controls[i], game.environment.dynamics, game.environment.sensor_models; is_robust=game.is_robust)
        beliefs[i+1] = unvec(g, game.dims.belief)
    end
    
    return beliefs, controls
end
