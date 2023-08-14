using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov

######################### Define dynamics and domain ##########################

"Simple Harmonic Oscillator Dynamics"
function f(state, p, t)
    ζ, ω_0 = p
    pos = state[1]
    vel = state[2]
    vcat(vel, -2ζ * vel - ω_0^2 * pos)
end
lb = [-2 * pi, -10.0];
ub = [2 * pi, 10.0];
p = [0.5, 1.0]
dynamics = ODEFunction(f; syms = [:x, :v], paramsyms = [:ζ, :ω_0])

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
    p = p
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
pde_system_relu, _ = NeuralLyapunovPDESystem(dynamics, lb, ub, spec_relu; p = p)
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
    res.u, 
    network_func, 
    structure.V,
    dynamics,
    zeros(2);
    p = p
    )

################################## Simulate ###################################
xs, ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))

# Get RoA Estimate
data = reshape(V_predict, (length(xs), length(ys)));
data = vcat(data[1, :], data[end, :], data[:, 1], data[:, end]);
ρ = minimum(data)

# Print statistics
println("V(0.,0.) = ", V_func([0.0, 0.0]))
println("V ∋ [", min(V_func([0.0, 0.0]), minimum(V_predict)), ", ", maximum(V_predict), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func([0.0, 0.0]), maximum(dVdt_predict)),
    "]",
)

# Plot results

p1 = plot(xs, ys, V_predict, linetype = :contourf, title = "V", xlabel = "x", ylabel = "ẋ");
p1 = scatter!([0], [0], label = "Equilibrium");
p2 = plot(
    xs,
    ys,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p2 = scatter!([0], [0], label = "Equilibrium");
p3 = plot(
    xs,
    ys,
    V_predict .< ρ,
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
    title = "dV/dt < 0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = scatter!([0], [0], label = "Equilibrium");
plot(p1, p2, p3, p4)
