using BlockArrays
using LinearAlgebra
using RobustBeliefGame
using Statistics
using CairoMakie
using Infiltrator

function belief_main()
    # Game Params
    horizon = 5
    dt = 0.3
    n=2
	surveillance_center = [5.0, 5.0]
	surveillance_radius = 10.0
    observation_output_dimension = 4

    gt_initial_means = BlockVector{Float64}(undef, [4 for _ in 1:2])
	gt_initial_means[Block(1)] .= [surveillance_center[1] - 8.0, surveillance_center[2] + 18.0, 0.00, 10.0] # Player 1, surveiller
	gt_initial_means[Block(2)] .= [surveillance_center[1] - 10.0, surveillance_center[2] + 15.0, 0.00, 10.0]
    gt_initial_belief_cov = [
		7.0 0.0 0.0 0.0;
		0.0 7.0 0.0 0.0;
		0.0 0.0 0.006 0.0;
		0.0 0.0 0.0 0.5;
	]
    initial_beliefs = Beliefs([Belief(gt_initial_means[Block(i)], copy(gt_initial_belief_cov)) for i in 1:2])

    function state_dynamics_noise_scaler(u) 
		return norm(u)
	end
	function f(states::BlockVector, u::BlockVector, m::BlockVector;noise_scaler::Function = state_dynamics_noise_scaler, L::Float64 = 1.0)
        BlockVector(
            mapreduce(vcat, zip(states.blocks, u.blocks, m.blocks)) do (xᵢ, uᵢ, mᵢ)
                dv = xᵢ[4] * tan(uᵢ[2]) / L
                ẋ = [xᵢ[4] * cos(xᵢ[3]), xᵢ[4] * sin(xᵢ[3]), dv, uᵢ[2]]
                xᵢ + dt * ẋ + noise_scaler(uᵢ) * mᵢ
            end
            ,
            [4, 4]
        )
	end
	function measurement_noise_scaler1(state::Vector)
		n = norm(norm(state[1:2] - surveillance_center, 2) - surveillance_radius^2)
		return [
		n 0 0 0; 
		0 n 0 0; 
		0 0 .1 0; 
		0 0 0 .1]
	end
	function measurement_noise_scaler2(state::Vector; surveillance_band_height = 0)
		n = .5 * (state[2] - surveillance_band_height)^2 + 1e-10 * state[1]
		v = 1e-10 * state[4]^2 # velocity scaled noise
        t = 1e-10 * state[3]
		noise = 
			[
			n 0 0 0; 
			0 n 0 0; 
			0 0 v 0; 
			0 0 0 t
			]
		return noise
	end
	function h(states::BlockVector, m::BlockVector; measurement_noise::Function = measurement_noise_scaler1)
        BlockVector(
            mapreduce(vcat, zip(states.blocks, m.blocks)) do (xᵢ, mᵢ)
                xᵢ + measurement_noise(xᵢ) * mᵢ
            end,
            [observation_output_dimension, observation_output_dimension]
        )
	end
	function overlap(ellipse1, ellipse2)
		c = ellipse1[1]
		d = ellipse2[1]
		A = ellipse1[2]
		B = ellipse2[2]
		expected_distance = norm(c - d)
		radius_estimate1 = (A[1, 1] + A[2, 2]) / 2.0
		radius_estimate2 = (B[1, 1] + B[2, 2]) / 2.0
		min_safety_distance = radius_estimate1 + radius_estimate2
		return expected_distance < min_safety_distance
	end
    function c_coll(β::Beliefs)
        ellipse1 = (β.beliefs[1].belief_mean[1:2], β.beliefs[1].belief_covariance[1:2, 1:2])
        ellipse2 = (β.beliefs[2].belief_mean[1:2], β.beliefs[2].belief_covariance[1:2, 1:2])
        if overlap(ellipse1, ellipse2)
            distance = β.beliefs[1].belief_mean[1:2] - β.beliefs[2].belief_mean[1:2]
            return exp(-0.1 * norm(distance, 2))
        else
            return 0
        end
    end
	function non_terminal_cost_1(β::Beliefs, u::BlockVector)
		return u[Block(1)]' * 0.001 * I(2) * u[Block(1)]
	end
	function terminal_cost_1(β::Beliefs)
		return 0.1 * prod(diag(β.beliefs[2].belief_covariance[1:2, 1:2]))
	end
    function non_terminal_cost_2(β::Beliefs, u::BlockVector)
		return u[Block(2)]' * 0.01 * I(2) * u[Block(2)] + 0.1 * (β.beliefs[2].belief_mean[4] - 10)^2 + 0.1 * c_coll(β)
	end
	function terminal_cost_2(β::Beliefs)
		return 0.1 * (β.beliefs[2].belief_mean[4] - 10)^2 + 0.1 * c_coll(β)
	end

	environment = BeliefEnvironment(
		f,
		gt_initial_means,
		h,
	)

	costs = [
		BeliefCost(non_terminal_cost_1, terminal_cost_1),
		BeliefCost(non_terminal_cost_2, terminal_cost_2),
	]

	bs_active_surveillance_game = BeliefGame(
		environment,
		costs,
		initial_beliefs,
		horizon,
		(; n=2, states=length.(gt_initial_means.blocks), controls=[2, 2], belief=length.(gt_initial_means.blocks), sensor=[observation_output_dimension, observation_output_dimension]),
		gt_initial_means,
	)

	sol = solve(bs_active_surveillance_game; debug=true, debug_file="./exp/active surveillance/outputs/belief_diagnostics.txt")

	visualize_belief_active_surveillance_solution(sol)
end

function visualize_belief_active_surveillance_solution(sol)
	println("[ActiveSurveillance] Visualizing solution...")
	
	# Extract surveillance parameters from the solution
	surveillance_center = [5.0, 5.0]
	surveillance_radius = 10.0
	
	# Create a grid for the background shading
	x_range = range(-20, 30, length=100)
	y_range = range(-20, 30, length=100)
	
	# Calculate distance from surveillance circle for each point
	function distance_from_circle(x, y)
		point = [x, y]
		return abs(norm(point - surveillance_center) - surveillance_radius)
	end
	
	# Create the background shading
	z = [distance_from_circle(x, y) for x in x_range, y in y_range]
	
	# Create the figure
	fig = Figure(size=(800, 600))
	ax = Axis(fig[1, 1], 
		title="Active Surveillance Visualization",
		xlabel="X Position",
		ylabel="Y Position")
	
	# Plot the heatmap
	heatmap!(ax, x_range, y_range, z, 
		colormap=reverse(cgrad(:grays, alpha=0.5)),
		colorrange=(0, 5))
	
	# Plot the trajectories
	colors = [:blue, :red]
	for i in 1:2
		# Extract mean positions for each player
		positions = [sol[1][t].beliefs[i].belief_mean[1:2] for t in 1:length(sol[1])]
		x_pos = [pos[1] for pos in positions]
		y_pos = [pos[2] for pos in positions]
		
		# Plot the trajectory
		lines!(ax, x_pos, y_pos, 
			color=colors[i],
			linewidth=2)
		scatter!(ax, x_pos, y_pos, 
			color=colors[i],
			markersize=8,
			label="Player $i")
	end
	
	# Add colorbar
	Colorbar(fig[1, 2], 
		colormap=reverse(cgrad(:grays, alpha=0.5)),
		label="Observation Noise Level",
		limits=(0, 5))
	
	# Add legend
	axislegend(ax, position=:rt)
	
	save("exp/active surveillance/outputs/belief_active_surveillance_solution.png", fig)
end