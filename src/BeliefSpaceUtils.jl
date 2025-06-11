struct Belief
    belief_mean::Vector{Float64}
    belief_covariance::Symmetric{Float64, Matrix{Float64}}
    belief_dim::Int
end

function Belief(belief_mean::Vector{Float64}, belief_covariance::Symmetric{Float64, Matrix{Float64}})
    return Belief(belief_mean, belief_covariance, length(belief_mean))
end

function Belief(belief_mean::Vector{Float64}, belief_covariance::Matrix{Float64})
    return Belief(belief_mean, Symmetric(belief_covariance), length(belief_mean))
end

function vec(belief::Belief)
    return [belief.belief_mean; vec(belief.belief_covariance)]
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

function vec(beliefs::Beliefs)
    mapreduce(vcat, beliefs.beliefs) do belief
        vec(belief)
    end
end

function unvec(vec_beliefs::Vector{Float64}, dims::Vector{Int})
    belief_dims = map(dims) do dim
        dim + dim ^2
    end
    map(eachindex(belief_dims)) do i
        belief_start = sum(belief_dims[1:i-1])
        Belief(vec_beliefs[belief_start+1:belief_start+dims[i]], reshape(vec_beliefs[belief_start+dims[i]+1:sum(belief_dims[1:i])], dims[i], dims[i]))
    end
end

function Base.:-(b1::Beliefs, b2::Beliefs)
    # Subtract corresponding belief means and covariances
    return vec(b1) - vec(b2)
end

function Base.:-(b1::Belief, b2::Belief)
    # Subtract corresponding belief means and covariances
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

function rollout_strategy(game::BeliefGame, strategy::Vector{Function})
    beliefs = [deepcopy(game.initial_beliefs)]
    controls = []
    for i in eachindex(strategy) #TODO is the order right? action -> new state -> observe -> action
        push!(controls, strategy[i](beliefs[end]))
        push!(beliefs, ekf_update(beliefs[end], controls[end], game.environment.dynamics, game.environment.sensor_models))
    end
    return beliefs, controls
end
