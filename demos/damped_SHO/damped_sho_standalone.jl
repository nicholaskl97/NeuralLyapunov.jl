using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using ForwardDiff

# Define parameters and differentials
@parameters state1 state2
x = [state1, state2]
@variables u1(..) u2(..)

# Define Lyapunov function
dim_output = 2
"Symbolic form of neural network output"
u(x) = [u1(x...), u2(x...)]
δ = 0.01
"Symobolic form of the Lyapunov function"
V_sym(x) = (u(x) - u([0.0, 0.0])) ⋅ (u(x) - u([0.0, 0.0])) + δ * log(1.0 + x ⋅ x)
#V_sym(x0,y0) = (u(x0,y0)) ⋅ (u(x0,y0)) + δ*log(1. + x0^2 + y0^2)

# Define dynamics
"Simple Harmonic Oscillator Dynamics"
function SHO_dynamics(state::AbstractMatrix{T})::AbstractMatrix{T} where {T<:Number}
    pos = transpose(state[1, :])
    vel = transpose(state[2, :])
    vcat(vel, -vel - pos)
end
function SHO_dynamics(state::AbstractVector{T})::AbstractVector{T} where {T<:Number}
    pos = state[1]
    vel = state[2]
    vcat(vel, -vel - pos)
end
NeuralPDE.dottable_(x::typeof(SHO_dynamics)) = false

# Define Lyapunov conditions
"Symbolic time derivative of the Lyapunov function"
V̇_sym(x) = SHO_dynamics(x) ⋅ Symbolics.gradient(V_sym(x), x)
eq_max = max(0.0, V̇_sym(x)) ~ 0.0
κ = 20.0
eq_log = log(1.0 + exp(κ * V̇_sym(x))) ~ 0.0 # Stricter, but max(0, V̇) still trains fine
domains = [state1 ∈ (-2 * pi, 2 * pi), state2 ∈ (-10.0, 10.0)]
bcs = [V_sym([0.0, 0.0]) ~ 0.0]

# Construct PDESystem
@named pde_system_log = PDESystem(eq_log, bcs, domains, x, u(x))

# Define neural network discretization
dim_state = length(domains)
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
#strategy = QuadratureTraining()
#strategy = QuasiRandomTraining(1000, bcs_points=3)
#strategy = StochasticTraining(1000, bcs_points=1)

discretization = PhysicsInformedNN(chain, strategy)

# Build optimization problem
prob_log = discretize(pde_system_log, discretization)
sym_prob_log = symbolic_discretize(pde_system_log, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Optimize with stricter log version 
#opt = BFGS()
#opt = Adam()
#opt = AdaGrad()
#opt = AdaMax()
#opt = Optim.SimulatedAnnealing()
res = Optimization.solve(prob_log, Adam(); callback = callback, maxiters = 300)

# Rebuild with weaker ReLU version
@named pde_system_relu = PDESystem(eq_max, bcs, domains, x, u(x))
prob_relu = discretize(pde_system_relu, discretization)
prob_relu = Optimization.remake(prob_relu, u0 = res.u);
println("Switching from log(1 + κ exp(V̇)) to max(0,V̇)");
res = Optimization.solve(prob_relu, Adam(); callback = callback, maxiters = 300)
prob_relu = Optimization.remake(prob_relu, u0 = res.u);
println("Switching from Adam to BFGS");
res = Optimization.solve(prob_relu, BFGS(); callback = callback, maxiters = 300)

# Get numerical numerical functions

phi = discretization.phi

u_func(x0, y0) = [phi[i]([x0, y0], res.u.depvar[Symbol(:u, i)])[1] for i = 1:dim_output]

"Numerical form of Lyapunov function"
function V_func(x0, y0)
    u_vec = u_func(x0, y0) - u_func(0.0, 0.0)
    #    u_vec = u_func(x0,y0)
    norm(u_vec)^2 + δ * log(1 + x0^2 + y0^2)
end

"Numerical gradient of Lyapunov function"
∇V_func(x0, y0) = ForwardDiff.gradient(p -> V_func(p[1], p[2]), [x0, y0])

"Numerical time derivative of Lyapunov function"
V̇_func(x0, y0) = SHO_dynamics([x0, y0]) ⋅ ∇V_func(x0, y0)

# Simulate
xs, ys = [
    ModelingToolkit.infimum(d.domain):0.02:ModelingToolkit.supremum(d.domain) for
    d in domains
]
V_predict = [V_func(x0, y0) for y0 in ys for x0 in xs]
dVdt_predict = [V̇_func(x0, y0) for y0 in ys for x0 in xs]

# Print statistics
println("V(0.,0.) = ", V_func(0.0, 0.0))
println("V ∋ [", min(V_func(0.0, 0.0), minimum(V_predict)), ", ", maximum(V_predict), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func(0.0, 0.0), maximum(dVdt_predict)),
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
# savefig("Lyapunov_sol")
