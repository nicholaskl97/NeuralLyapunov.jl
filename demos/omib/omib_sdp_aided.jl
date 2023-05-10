using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov

###################### Set up system #############################
# Define dynamics
r_line = 0.01 # line+gen resistance pu
x_line = 0.1 # line+gen reactance pu
y_line = 1.0 / (r_line + x_line * 1im) # line+gen admittance

g_line = real(y_line) # line conductance pu
b_line = imag(y_line) # line+gen susceptance pu

Pm = 0.5 # mechanical power from gen in pu

E = 1.01 # Generator voltage
E_∞ = 1.0 # Infinite bus voltage
H = 8.0 # Generator Inertia [seconds]
Ω_b = 2 * pi * 60 # Base Frequency
T = 0.1 # Damping Coefficient

p_omib = [
    Pm - E^2 * g_line, # P: Total Power with losses
    -E * E_∞ * b_line, # C: sine coefficient term
    -E * E_∞ * g_line, # D: cosine coefficient term
    T, # damping coefficient
    H, # inertia term
    Ω_b, # nominal frequency
]

"""
This system describes a classic machine model against an infinite bus, described as:
δ_dot = ω
ω_dot = (Ω_b / H) * (P - C * sin(δ) - D * cos(δ) - T * ω)
where δ is the rotor angle and ω is the frequency deviation from the synchronous frequency.

This model incorporates line losses and has a general Lyapunov function in the usual sense as:
V(δ, ω) = (H / Ω_b) * (ω^2 / 2) - P * δ - C * cos(δ) + D * sin(δ) + α
where α is an arbitrary constant. It can be shown that the derivative of V along the orbits satisfies:
V_dot = - T * ω^2 ≤ 0
which is a negative semi-definite function. V also satisfies the requirements of LaSalle's invariance principle
and hence V can be used to study the stability of this system in the usual way.
"""
function omib_dynamics(state::AbstractVector{S})::AbstractVector{S} where {S<:Number}
    δ, ω = @view state[1:2]
    P, C, D, T, H, Ω_b = @view p_omib[1:6]

    return [ ω; (Ω_b / H) * (P - C * sin(δ) - D * cos(δ) - T * ω)]
end
function omib_dynamics(state::AbstractMatrix{S})::AbstractMatrix{S} where {S<:Number}
    δ = @view state[1,:]
    ω = @view state[2,:]
    P, C, D, T, H, Ω_b = @view p_omib[1:6]

    return vcat(transpose(ω), transpose((Ω_b / H) .* (P .- C .* sin.(δ) .- D .* cos.(δ) .- T .* ω)))
end

lb = [-0.25, -1.0];
ub = [0.25, 1.0];
fixed_point = let (P, C, D, T, H, Ω_b) = p_omib[1:6]
    [asin(P/sqrt(C^2+D^2)) - atan(D, C), 0.0]
end

##################### Get local RoA Estimate ##########################

V_local, V̇_local, ∇V_local = local_Lyapunov(
    omib_dynamics, 
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

########################## Neural search ########################

# Make log version
dim_output = 2
κ = 20.0
δ = 0.1
pde_system_log, lyapunov_func = NeuralLyapunovPDESystem(
    omib_dynamics,
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
# strategy = GridTraining(0.01)
strategy = QuasiRandomTraining(100)
# strategy = QuadratureTraining()
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
    omib_dynamics,
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
    omib_dynamics,
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
xs, ys = [lb[i]:0.005:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))
V_local_predict = vec(V_local(hcat(states...)))
dVdt_local_predict = vec(V̇_local(hcat(states...)))
# V_predict = [V_func([x0,y0]) for y0 in ys for x0 in xs]
# dVdt_predict  = [V̇_func([x0,y0]) for y0 in ys for x0 in xs]

# Print statistics
@show V_func(fixed_point)
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
p1 = scatter!([fixed_point[1]], [fixed_point[2]], labels = false, markershape = :+, markercolor=:green);
p2 = plot(
    xs,
    ys,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p2 = scatter!([fixed_point[1]], [fixed_point[2]], labels = false, markershape = :+, markercolor=:green);
p3 = plot(xs, ys, V_local_predict, linetype = :contourf, title = "Local V", xlabel = "x", ylabel = "ẋ");
p3 = scatter!([fixed_point[1]], [fixed_point[2]], labels = false, markershape = :+, markercolor=:green);
p4 = plot(
    xs,
    ys,
    dVdt_local_predict,
    linetype = :contourf,
    title = "Local dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p4 = scatter!([fixed_point[1]], [fixed_point[2]], labels = false, markershape = :+, markercolor=:green);
p5 = plot(
    xs,
    ys,
    V_local_predict .≤ ρ_local,
    linetype = :contourf,
    title = "Estimated RoA, local",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p5 = scatter!([fixed_point[1]], [fixed_point[2]], labels = false, markershape = :+, markercolor=:green);
p6 = plot(
    xs,
    ys,
    dVdt_local_predict .< 0,
    linetype = :contourf,
    title = "Local dV/dt < 0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p6 = scatter!([fixed_point[1]], [fixed_point[2]], labels = false, markershape = :+, markercolor=:green);
p7 = plot(
    xs,
    ys,
    V_predict .≤ ρ,
    linetype = :contourf,
    title = "Certified RoA",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p7 = scatter!([fixed_point[1]], [fixed_point[2]], labels = false, markershape = :+, markercolor=:green);
p8 = plot(
    xs,
    ys,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p8 = scatter!([fixed_point[1]], [fixed_point[2]], labels = false, markershape = :+, markercolor=:green);
plot(p1, p2, p3, p4, p5, p6, p7, p8, layout = (4,2))
