struct MCPGame{T1, T2}
    game::T1
    mcp::T2
    horizon::Int
    n_equality_constraints::Vector{Int}
    n_inequality_constraints::Vector{Int}
    n_shared_inequality_constraints::Int
end

function MCPGame(game::TrajectoryGame, horizon::Int, initial_conditions::Vector{Float64})
    dynamics = game.dynamics
    costs = game.cost
    n_players = length(costs)
    state_dims = map(1:n_players) do ii
        state_dim(dynamics.subsystems[ii])
    end
    x_size = sum(state_dims) * horizon
    player_xs = map(1:n_players) do ii
        mapreduce(vcat, 1:horizon) do t
            [(t-1)*sum(state_dims)+sum(state_dims[1:ii-1])+1 : (t-1)*sum(state_dims)+sum(state_dims[1:ii])]
        end
    end
    control_dims = map(1:n_players) do ii
        control_dim(dynamics.subsystems[ii])
    end
    u_size = sum(control_dims) * horizon
    player_us = map(1:n_players) do ii
        mapreduce(vcat, 1:horizon) do t
            [(t-1)*sum(control_dims)+sum(control_dims[1:ii-1])+1 : (t-1)*sum(control_dims)+sum(control_dims[1:ii])]
        end
    end

    n_eq_constr = zeros(Int, n_players)
    n_ineq_constr = zeros(Int, n_players)
    n_shared_ineq_constr = 0
    
    for ii in 1:n_players
        # Dynamics constraints
        n_eq_constr[ii] += state_dims[ii] * (horizon - 1)
        # Initial conditions
        n_eq_constr[ii] += state_dims[ii]

        # Environment constraints
        n_ineq_constr[ii] += length(get_constraints(game.env, ii)(BlockVector(zeros(state_dims[ii]), [state_dims[ii]]))) * horizon
        # State constraints
        n_ineq_constr[ii] += 2 * length(player_xs[ii][1]) * horizon
        # Control box constraints
        n_ineq_constr[ii] += 2 * length(player_us[ii][1]) * horizon
    end
    # Grab shared inequality constraints from game's coupling constraints - Assume number is not a function of time
    n_shared_ineq_constr = isnothing(game.coupling_constraints) ? 0 : length(game.coupling_constraints(BlockVector(zeros(x_size), [x_size]), BlockVector(zeros(u_size), [u_size]), 1)) * horizon

    # Define symbolic variables for the Lagrangian scope
    @variables z_L[1:x_size+u_size+sum(n_eq_constr)] λ_L[1:sum(n_ineq_constr)] λ_sh_L[1:n_shared_ineq_constr]
    z_L = Symbolics.scalarize(z_L)
    λ_L = Symbolics.scalarize(λ_L)
    λ_sh_L = Symbolics.scalarize(λ_sh_L)

    # Equality constraints (list of functions, each returning Vector{Symbolics.Num})
    equality_constr_funcs = map(1:n_players) do ii
        function(_z, _λ)
            x = _z[1:x_size]
            u = _z[x_size+1:x_size+u_size]
            
            dyn_constr = mapreduce(vcat, 1:horizon-1; init=Vector{Symbolics.Num}()) do t
                game.dynamics.subsystems[ii](x[player_xs[ii][t]], u[player_us[ii][t]], t) - x[player_xs[ii][t+1]]
            end
            init_constr = x[player_xs[ii][1]] - initial_conditions[player_xs[ii][1]]
            return vcat(dyn_constr, init_constr)
        end
    end

    # Inequality constraints (list of functions, each returning Vector{Symbolics.Num})
    inequality_constr_funcs = map(1:n_players) do ii
        environment_constraints_gen = get_constraints(game.env, ii)
        subdynamics = dynamics.subsystems[ii]
        state_box_constraints_gen = create_box_bounds(state_bounds(subdynamics))
        control_box_constraints_gen = create_box_bounds(control_bounds(subdynamics))

        function(_z, _λ)
            x = _z[1:x_size]
            u = _z[x_size+1:x_size+u_size]
            
            ec = mapreduce(vcat, 1:horizon; init=Vector{Symbolics.Num}()) do t
                environment_constraints_gen(BlockVector(x[player_xs[ii][t]], [state_dims[ii]]))
            end
            sc = mapreduce(vcat, 1:horizon; init=Vector{Symbolics.Num}()) do t
                state_box_constraints_gen(BlockVector(x[player_xs[ii][t]], [state_dims[ii]]))
            end
            cc = mapreduce(vcat, 1:horizon; init=Vector{Symbolics.Num}()) do t
                control_box_constraints_gen(BlockVector(u[player_us[ii][t]], [control_dims[ii]]))
            end
            return vcat(ec, sc, cc)
        end
    end

    # Shared inequality constraints (a single function returning Vector{Symbolics.Num})
    local shared_inequality_constr_eval_func::Function
    if game.coupling_constraints !== nothing && n_shared_ineq_constr > 0
        let captured_game_cc = game.coupling_constraints 
            shared_inequality_constr_eval_func = (_z_arg, _λ_arg) -> begin
                mapreduce(vcat, 1:horizon; init=Vector{Symbolics.Num}()) do t
                    primal_vars_z = _z_arg[1:(x_size+u_size)]
                    
                    x_slice = _z_arg[1:x_size]
                    u_slice = _z_arg[x_size+1 : x_size+u_size]
                    temp_constraints = captured_game_cc(_z_arg[1:(x_size+u_size)], _λ_arg, t)
                    return (temp_constraints isa Vector{Symbolics.Num} ? temp_constraints : convert(Vector{Symbolics.Num}, temp_constraints))
                end
            end
        end
    else
        shared_inequality_constr_eval_func = (_z_arg, _λ_arg) -> Vector{Symbolics.Num}()
    end
    
    # lagrangian_grads (list of functions, each returning Vector{Symbolics.Num})
    lagrangian_grads = let
        # Define symbolic state, control, and equality multiplier views based on z_L
        xs_L = map(1:horizon) do t
            z_L[(t-1)*sum(state_dims)+1:t*sum(state_dims)]
        end
        us_L = map(1:horizon) do t
            z_L[x_size + (t-1)*sum(control_dims)+1:x_size + t*sum(control_dims)]
        end
        μs_L = z_L[x_size+u_size+1 : x_size+u_size+sum(n_eq_constr)] 

        player_λ_indices = mapreduce(vcat, 1:n_players) do ii
            [sum(n_ineq_constr[1:ii-1])+1:sum(n_ineq_constr[1:ii])]
        end
        player_μ_indices = mapreduce(vcat, 1:n_players) do ii
            [sum(n_eq_constr[1:ii-1])+1:sum(n_eq_constr[1:ii])]
        end

        map(1:n_players) do ii
            L_ii = let
                cost_term = costs[ii](xs_L, us_L)
                eq_constr_exprs = equality_constr_funcs[ii](z_L, λ_L) 
                eq_term = dot(μs_L[player_μ_indices[ii]], eq_constr_exprs)

                ineq_constr_exprs = inequality_constr_funcs[ii](z_L, λ_L) 
                ineq_term_player = dot(λ_L[player_λ_indices[ii]], ineq_constr_exprs)
                
                ineq_term_shared = 0
                if n_shared_ineq_constr > 0
                    shared_ineq_exprs = shared_inequality_constr_eval_func(z_L, λ_L) 
                    ineq_term_shared = dot(λ_L[player_λ_indices[ii]], shared_ineq_exprs)
                end
                cost_term - eq_term - (ineq_term_player + ineq_term_shared)
            end

            player_x_vars = reduce(vcat, map(t -> z_L[player_xs[ii][t]], 1:horizon))
            player_u_vars = reduce(vcat, map(t -> z_L[x_size .+ player_us[ii][t]], 1:horizon))
            
            ∇x_L_ii = Symbolics.gradient(L_ii, player_x_vars)
            ∇u_L_ii = Symbolics.gradient(L_ii, player_u_vars)
            
            function(_z_arg, _λ_arg) 
                all_L_sym_vars = vcat(z_L, λ_L, λ_sh_L) 
                
                _λ_L_part = _λ_arg[1:sum(n_ineq_constr)]
                _λ_sh_L_part = _λ_arg[sum(n_ineq_constr)+1 : sum(n_ineq_constr)+n_shared_ineq_constr]
                all_arg_runtime_vals = vcat(_z_arg, _λ_L_part, _λ_sh_L_part) 

                if length(all_L_sym_vars) != length(all_arg_runtime_vals)
                    error("LAGRANGIAN_GRADS_SUBSTITUTION_ERROR: Mismatch between symbolic variable count (" * string(length(all_L_sym_vars)) * ") and value count (" * string(length(all_arg_runtime_vals)) * ") for substitution. Check z_L, λ_L, λ_sh_L against _z_arg and _λ_arg partitioning.")
                end

                subs_map = Dict(zip(all_L_sym_vars, all_arg_runtime_vals))

                gx = Symbolics.substitute.(∇x_L_ii, Ref(subs_map))
                gu = Symbolics.substitute.(∇u_L_ii, Ref(subs_map))
                
                vcat(gx, gu)
            end
        end
    end

    G = function(_z_mcp, _λ_mcp; θ = nothing) 
        lag_grad_components = mapreduce(vcat, lagrangian_grads; init=Vector{Symbolics.Num}()) do grad_func_for_player
            grad_func_for_player(_z_mcp, _λ_mcp) 
        end
        eq_constr_components = mapreduce(vcat, equality_constr_funcs; init=Vector{Symbolics.Num}()) do eq_func
            eq_func(_z_mcp, _λ_mcp) 
        end
        vcat(lag_grad_components, eq_constr_components)
    end
    
    H = function(_z_mcp, _λ_mcp; θ = nothing) 
        ineq_player_components = mapreduce(vcat, inequality_constr_funcs; init=Vector{Symbolics.Num}()) do ineq_func
            ineq_func(_z_mcp, _λ_mcp)
        end

        shared_ineq_components = Vector{Symbolics.Num}()
        if n_shared_ineq_constr > 0
            shared_ineq_components = shared_inequality_constr_eval_func(_z_mcp, _λ_mcp)
        end
        
        vcat(ineq_player_components, shared_ineq_components)
    end

    mcp = MixedComplementarityProblems.PrimalDualMCP(
        G,
        H;
        unconstrained_dimension = x_size + u_size + sum(n_eq_constr), 
        constrained_dimension = sum(n_ineq_constr) + n_shared_ineq_constr,
        parameter_dimension = 0
    )
    
    return MCPGame(game, mcp, horizon, n_eq_constr, n_ineq_constr, n_shared_ineq_constr)
end

function create_box_bounds(bounds)
    function (y)
        mapreduce(vcat, [(bounds.lb, 1), (bounds.ub, -1)]) do (bound, sign)
            # Don't drop constraints for unbounded variables, makes counting easier
            sign * (y - bound)
        end
    end
end

