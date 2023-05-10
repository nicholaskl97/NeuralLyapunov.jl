using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov

############################### Define dynamics ###############################

"Pendulum Dynamics"
function dynamics(state::AbstractMatrix{T})::AbstractMatrix{T} where {T<:Number}
    pos = transpose(state[1, :])
    vel = transpose(state[2, :])
    vcat(vel, -vel - sin.(pos))
end
function dynamics(state::AbstractVector{T})::AbstractVector{T} where {T<:Number}
    pos = state[1]
    vel = state[2]
    vcat(vel, -vel - sin.(pos))
end
lb = [0.0, -10.0];
ub = [4 * pi, 10.0];
fixed_point = [2 * pi, 0.0]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 15
dim_output = 2
chain = [
    Lux.Chain(
        Dense(dim_state, dim_hidden, tanh),
        Dense(dim_hidden, dim_hidden, tanh),
        Dense(dim_hidden, 1, use_bias = false),
    ) for _ = 1:dim_output
]

# Define neural network discretization
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy)

# Define neural Lyapunov structure
structure = NonnegativeNeuralLyapunov(
        dim_output; 
        δ = 1e-6
        #grad_pos_def = (state, fixed_point) -> transpose(state - fixed_point) ./ (1.0 + (state - fixed_point) ⋅ (state - fixed_point))
        )
minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

# Define Lyapunov decrease condition
κ = 20.0
decrease_condition_log = AsymptoticDecrease(
    strict = true, 
    relu = (t) -> log(1.0 + exp(κ * t)) / κ
    )

# Construct neural Lyapunov specification
spec_log = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition_log,
    )

############################# Construct PDESystem #############################

pde_system_log, network_func = NeuralLyapunovPDESystem(
    dynamics,
    lb,
    ub,
    spec_log;
    fixed_point = fixed_point
)

######################## Construct OptimizationProblem ########################

prob_log = discretize(pde_system_log, discretization)
sym_prob_log = symbolic_discretize(pde_system_log, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

########################## Solve OptimizationProblem ##########################

# Optimize with stricter log version
res = Optimization.solve(prob_log, Adam(); callback = callback, maxiters = 300)

######################### Rebuild OptimizationProblem #########################

println("Switching from log(1 + κ exp(V̇))/κ to max(0,V̇)");

# Set up new decrease condition
decrease_condition_relu = AsymptoticDecrease(strict = true)
spec_relu = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition_relu,
    )

# Build and discretize new PDESystem
pde_system_relu, _ = NeuralLyapunovPDESystem(
    dynamics, 
    lb, 
    ub, 
    spec_relu; 
    fixed_point = fixed_point
    )
prob_relu = discretize(pde_system_relu, discretization)
sym_prob_relu = symbolic_discretize(pde_system_relu, discretization)

# Rebuild problem with weaker ReLU version
prob_relu = Optimization.remake(prob_relu, u0 = res.u);

######################## Solve new OptimizationProblem ########################

res = Optimization.solve(prob_relu, Adam(); callback = callback, maxiters = 300)
prob_relu = Optimization.remake(prob_relu, u0 = res.u);

println("Switching from Adam to BFGS");
res = Optimization.solve(prob_relu, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi, 
    res, 
    network_func, 
    structure.V,
    dynamics,
    fixed_point
    )

################################## Simulate ###################################
xs, ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))
# V_predict = [V_func([x0,y0]) for y0 in ys for x0 in xs]
# dVdt_predict  = [V̇_func([x0,y0]) for y0 in ys for x0 in xs]

# Get RoA Estimate
data = reshape(V_predict, (length(xs), length(ys)));
data = vcat(data[1, :], data[end, :], data[:, 1], data[:, end]);
ρ = minimum(data)

# Print statistics
println("V(2π, 0) = ", V_func(fixed_point))
println(
    "V ∋ [",
    min(V_func(fixed_point), minimum(V_predict)),
    ", ",
    maximum(V_predict),
    "]",
)
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func(fixed_point), maximum(dVdt_predict)),
    "]",
)
# println("Certified V ∈ [0.0, ", ρ, ")")

# Plot results
#ρ = 1

p1 = plot(xs, ys, V_predict, linetype = :contourf, title = "V", xlabel = "x", ylabel = "ẋ");
p2 = plot(
    xs,
    ys,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p2 = scatter!((lb[1]+pi):2*pi:ub[1], zeros(4), label = "Unstable equilibria");
p2 = scatter!(lb[1]:2*pi:ub[1], zeros(5), label = "Stable equilibria");
p3 = plot(
    xs,
    ys,
    V_predict .≤ ρ,
    linetype = :contourf,
    title = "Estimated RoA",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = plot(
    xs,
    ys,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dVdt<0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = scatter!((lb[1]+pi):2*pi:ub[1], zeros(4), label = "Unstable equilibria");
p4 = scatter!(lb[1]:2*pi:ub[1], zeros(5), label = "Stable equilibria");
plot(p1, p2, p3, p4)