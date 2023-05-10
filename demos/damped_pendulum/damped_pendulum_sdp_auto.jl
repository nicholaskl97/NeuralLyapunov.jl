using LinearAlgebra
using Hypatia, JuMP
using ForwardDiff
using Optimization
import OptimizationOptimJL
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
    pos = state[1]
    vel = state[2]
    vcat(vel, -vel - sin.(pos))
end
fixed_point = [0.0; 0.0]

# Calculate quadratic Lyapunov function of linearized system with SDP
A = ForwardDiff.jacobian(pendulum_dynamics, fixed_point)
model = Model(Hypatia.Optimizer)
set_silent(model)
@variable(model, P[1:2, 1:2], PSD)
@variable(model, Q[1:2, 1:2], PSD)
@constraint(model, (P) * A + transpose(A) * (P) .== -(Q))
JuMP.optimize!(model)
Psol = value.(P)

# Numerical form of Lyapunov function
V_func(state::AbstractVector) = dot(state, Psol, state)
V_func(states::AbstractMatrix) = mapslices(V_func, states, dims = [1])

# Numerical gradient of Lyapunov function
∇V_func(state::AbstractVector) = ForwardDiff.gradient(V_func, state)
∇V_func(states::AbstractMatrix) = mapslices(∇V_func, states, dims = [1])

# Numerical time derivative of Lyapunov function
V̇_func(state::AbstractVector) = pendulum_dynamics(state) ⋅ ∇V_func(state)
V̇_func(states::AbstractMatrix) = reshape(
    map(
        x -> x[1] ⋅ x[2],
        zip(
            eachslice(pendulum_dynamics(states), dims = 2),
            eachslice(∇V_func(states), dims = 2),
        ),
    ),
    (1, :),
)

# Get RoA Estimate
lb = [-2 * pi, -10.0];
ub = [2 * pi, 10.0];
ρ = get_RoA_estimate(V_func, V̇_func, lb, ub; fixed_point = fixed_point, ∇V = ∇V_func)

# Print statistics
println("Certified V ∈ [0.0, ", ρ, ")")

# Simulate
xs, ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))
# V_predict = [V_func([x0,y0]) for y0 in ys for x0 in xs]
# dVdt_predict  = [V̇_func([x0,y0]) for y0 in ys for x0 in xs]

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
p3 = scatter!((lb[1]+pi):2*pi:ub[1], zeros(4), label = "Unstable equilibria");
p3 = scatter!(lb[1]:2*pi:ub[1], zeros(5), label = "Stable equilibria");
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