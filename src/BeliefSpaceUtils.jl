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
    return [belief.belief_mean for belief in beliefs.beliefs]
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
    mapreduce(vcat, beliefs.beliefs) do belief
        vec(belief)
    end
end

function unvec(vec_beliefs::Vector, dims::Vector{Int})
    belief_dims = map(dims) do dim
        dim + dim ^2
    end
    Beliefs(map(eachindex(belief_dims)) do i
        belief_start = sum(belief_dims[1:i-1])
        Belief(vec_beliefs[belief_start+1:belief_start+dims[i]], reshape(vec_beliefs[belief_start+dims[i]+1:sum(belief_dims[1:i])], dims[i], dims[i]))
    end)
end

function Base.:-(b1::Beliefs, b2::Beliefs)
    return vec(b1) - vec(b2)
end

function Base.:-(b1::Belief, b2::Belief)
    return vec(b1) - vec(b2)
end

struct BeliefCost
    non_terminal_cost::Function
    terminal_cost::Function
end

struct BeliefEnvironment
    dynamics::Function
    gt_states::BlockVector
    sensor_models::Function
end

struct BeliefGame
    environment::BeliefEnvironment
    costs::Vector{BeliefCost}
    initial_beliefs::Beliefs
    horizon::Int
    dims::NamedTuple{(:n, :states, :controls, :belief, :sensor)}
    gt_initial_state::BlockVector
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

function rollout_strategy(game::BeliefGame, strategy::Vector{<:Function})
    beliefs = [deepcopy(game.initial_beliefs)]
    controls::Vector{BlockVector} = []
    for i in eachindex(strategy)
        push!(controls, strategy[i](beliefs[end]))
        if DEBUG
            open(DEBUG_FILE, "a") do f
                println(f, "\n[Rollout] time: $i")
                println(f, "Controls: $(controls[end])")
                println(f, "Belief means: $(means(beliefs[end]))")
                println(f, "Belief covariances:")
                for (i, cov) in enumerate(covs(beliefs[end]))
                    println(f, "Agent $i covariance:")
                    display_matrix = IOContext(f, :limit=>false)
                    show(display_matrix, "text/plain", cov)
                    println(f)
                end
                println(f, "Control: $(controls[end])")
            end
        end
        g, W = ekf_update(beliefs[end], controls[end], game.environment.dynamics, game.environment.sensor_models)
        push!(beliefs, unvec(g, game.dims.belief))
    end
    return beliefs, controls
end
