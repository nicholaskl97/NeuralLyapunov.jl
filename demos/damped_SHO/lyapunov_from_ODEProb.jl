using LinearAlgebra
using DifferentialEquations
using NeuralPDE, Lux
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov

# Set up the ODE
@variables t x(t)
@parameters ζ ω_0
Dt = Differential(t)
DDt = Dt^2

@named damped_sho = ODESystem(DDt(x) ~ -2ζ * ω_0 * Dt(x) - ω_0^2 * x, t)
damped_sho_simplified = structural_simplify(damped_sho)

ODEprob = ODEProblem(
    damped_sho_simplified,
    [x => 0.0, Dt(x) => 0.0],
    (0.0, 25.0),
    [ζ => 0.5, ω_0 => 1.0],
)

lb = [-2 * pi, -10.0];
ub = [2 * pi, 10.0];

# Make log version
dim_output = 1
κ = 20.0
pde_system_log, lyapunov_func = NeuralLyapunovPDESystem(
    ODEprob,
    lb,
    ub,
    dim_output,
    relu = (t) -> log(1.0 + exp(κ * t)) / κ,
)

# Set up neural net 
dim_state = length(lb)
dim_hidden = 15
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

# Build optimization problem
prob_log = discretize(pde_system_log, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Optimize log version
res = Optimization.solve(prob_log, Adam(); callback = callback, maxiters = 300)

# Optimize ReLU verion
pde_system_relu, _ = NeuralLyapunovPDESystem(ODEprob, lb, ub, dim_output)
prob_relu = discretize(pde_system_relu, discretization)
prob_relu = Optimization.remake(prob_relu, u0 = res.u);
println("Switching from log(1 + κ exp(V̇))/κ to max(0,V̇)");
res = Optimization.solve(prob_relu, Adam(); callback = callback, maxiters = 300)
prob_relu = Optimization.remake(prob_relu, u0 = res.u);
println("Switching from Adam to BFGS");
res = Optimization.solve(prob_relu, BFGS(); callback = callback, maxiters = 300)

# Get numerical numerical functions
V_func, V̇_func =
    NumericalNeuralLyapunovFunctions(discretization.phi, res, lyapunov_func, ODEprob)

# Simulate
xs, ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))

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
p2 = plot(
    xs,
    ys,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
plot(p1, p2)
