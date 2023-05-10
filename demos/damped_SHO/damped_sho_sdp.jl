using LinearAlgebra
using Hypatia, JuMP
using ForwardDiff
using Plots

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
fixed_point = [0.0; 0.0]

# Calculate quadratic Lyapunov function of linearized system with SDP
A = ForwardDiff.jacobian(SHO_dynamics, fixed_point)
model = Model(Hypatia.Optimizer)
set_silent(model)
@variable(model, P[1:2, 1:2], PSD)
@variable(model, Q[1:2, 1:2], PSD)
@constraint(model, (P) * A + transpose(A) * (P) .== -(Q))
optimize!(model)
Psol = value.(P)

# Numerical form of Lyapunov function
V_func(state::AbstractVector) = dot(state, Psol, state)
V_func(states::AbstractMatrix) = mapslices(V_func, states, dims = [1])

# Numerical gradient of Lyapunov function
∇V_func(state::AbstractVector) = ForwardDiff.gradient(V_func, state)
∇V_func(states::AbstractMatrix) = mapslices(∇V_func, states, dims = [1])

# Numerical time derivative of Lyapunov function
V̇_func(state::AbstractVector) = SHO_dynamics(state) ⋅ ∇V_func(state)
V̇_func(states::AbstractMatrix) = reshape(
    map(
        x -> x[1] ⋅ x[2],
        zip(
            eachslice(SHO_dynamics(states), dims = 2),
            eachslice(∇V_func(states), dims = 2),
        ),
    ),
    (1, :),
)


# Simulate
lb = [-2 * pi, -10.0];
ub = [2 * pi, 10.0];
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
println("V(0.,0.) = ", V_func([0.0, 0.0]))
println(
    "V ∋ [",
    min(first(V_func([0.0, 0.0])), minimum(V_predict)),
    ", ",
    maximum(V_predict),
    "]",
)
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(first(V̇_func([0.0, 0.0])), maximum(dVdt_predict)),
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
# savefig("SHO")
