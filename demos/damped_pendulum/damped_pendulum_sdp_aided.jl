using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov

# Define dynamics
"Pendulum Dynamics"
function pendulum_dynamics(state::AbstractMatrix{T})::AbstractMatrix{T} where {T<:Number}
    pos = transpose(state[1, :])
    vel = transpose(state[2, :])
    vcat(vel, -vel - sin.(pos))
end
function pendulum_dynamics(state::AbstractVector{T})::AbstractVector{T} where {T<:Number}
    pos = @view state[1]
    vel = @view state[2]
    vcat(vel, -vel - sin.(pos))
end
lb = [0.0, -10.0];
ub = [4 * pi, 10.0];
fixed_point = [2 * pi, 0.0]

# Make log version
dim_output = 2
κ = 20.0
δ = 0.1
pde_system_log, lyapunov_func = NeuralLyapunovPDESystem(
    pendulum_dynamics,
    lb,
    ub,
    dim_output,
    δ = δ,
    relu = (t) -> log(1.0 + exp(κ * t)) / κ,
    fixed_point = fixed_point,
)

# Define neural network discretization
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
# strategy = GridTraining(0.1)
# strategy = QuasiRandomTraining(100)
strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy)

# Build optimization problem
prob_log = discretize(pde_system_log, discretization)
sym_prob_log = symbolic_discretize(pde_system_log, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Optimize with stricter log version
res = Optimization.solve(prob_log, Adam(); callback = callback, maxiters = 300)

# Rebuild with weaker ReLU version
pde_system_relu, _ = NeuralLyapunovPDESystem(
    pendulum_dynamics,
    lb,
    ub,
    dim_output,
    δ = δ,
    fixed_point = fixed_point,
)
prob_relu = discretize(pde_system_relu, discretization)
sym_prob_relu = symbolic_discretize(pde_system_relu, discretization)
prob_relu = Optimization.remake(prob_relu, u0 = res.u);
println("Switching from log(1 + κ exp(V̇))/κ to max(0,V̇)");
res = Optimization.solve(prob_relu, Adam(); callback = callback, maxiters = 300)
prob_relu = Optimization.remake(prob_relu, u0 = res.u);
println("Switching from Adam to BFGS");
res = Optimization.solve(prob_relu, BFGS(); callback = callback, maxiters = 300)

# Get numerical numerical functions
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi,
    res,
    lyapunov_func,
    pendulum_dynamics,
)

##################### Get local RoA Estimate ##########################

V_local, V̇_local, ∇V_local = local_Lyapunov(
    pendulum_dynamics, 
    length(lb); 
    fixed_point = fixed_point
)
ρ_local = get_RoA_estimate(
    V_local, 
    V̇_local, 
    lb, ub; 
    fixed_point = fixed_point, 
    ∇V=∇V_local
)

################## Get Neural Lyapunov RoA Estimate ###################

ρ = NeuralLyapunov.get_RoA_estimate_aided(
    V_func, 
    V̇_func, 
    lb, 
    ub, 
    V_local, 
    ρ_local; 
    fixed_point = fixed_point, 
    ∇V = ∇V_func, 
    ∇V_certified = ∇V_local,
)

########################### Plot results ###########################

# Simulate
xs, ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))
V_local_predict = vec(V_local(hcat(states...)))
dVdt_local_predict = vec(V̇_local(hcat(states...)))
# V_predict = [V_func([x0,y0]) for y0 in ys for x0 in xs]
# dVdt_predict  = [V̇_func([x0,y0]) for y0 in ys for x0 in xs]

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
println("Certified V ∈ [0.0, ", ρ, ")")

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
p2 = scatter!(
    (lb[1]+pi):2*pi:ub[1],
    zeros(4),
    labels = false,
    markershape = :x,
);
p2 = scatter!(lb[1]:2*pi:ub[1], zeros(5), labels = false, markershape = :+);
p3 = plot(xs, ys, V_local_predict, linetype = :contourf, title = "Local V", xlabel = "x", ylabel = "ẋ");
p4 = plot(
    xs,
    ys,
    dVdt_local_predict,
    linetype = :contourf,
    title = "Local dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p4 = scatter!(
    (lb[1]+pi):2*pi:ub[1],
    zeros(4),
    labels = false,
    markershape = :x,
);
p4 = scatter!(lb[1]:2*pi:ub[1], zeros(5), labels = false, markershape = :+);
p5 = plot(
    xs,
    ys,
    V_predict .≤ ρ,
    linetype = :contourf,
    title = "Estimated RoA",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p5 = scatter!(
    (lb[1]+pi):2*pi:ub[1],
    zeros(4),
    labels = false,
    markershape = :x,
);
p5 = scatter!(lb[1]:2*pi:ub[1], zeros(5), labels = false, markershape = :+);
p6 = plot(
    xs,
    ys,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dV/dt<0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p6 = scatter!(
    (lb[1]+pi):2*pi:ub[1],
    zeros(4),
    labels = false,
    markershape = :x,
);
p6 = scatter!(lb[1]:2*pi:ub[1], zeros(5), labels = false, markershape = :+);
p7 = plot(
    xs,
    ys,
    V_local_predict .≤ ρ_local,
    linetype = :contourf,
    title = "Estimated RoA, local",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p7 = scatter!(
    (lb[1]+pi):2*pi:ub[1],
    zeros(4),
    labels = false,
    markershape = :x,
);
p7 = scatter!(lb[1]:2*pi:ub[1], zeros(5), labels = false, markershape = :+);
p8 = plot(
    xs,
    ys,
    dVdt_local_predict .< 0,
    linetype = :contourf,
    title = "Local dV/dt < 0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p8 = scatter!(
    (lb[1]+pi):2*pi:ub[1],
    zeros(4),
    labels = false,
    markershape = :x,
);
p8 = scatter!(lb[1]:2*pi:ub[1], zeros(5), labels = false, markershape = :+);
plot(p1, p2, p3, p4, p5, p6, p7, p8, layout = (2,4))
