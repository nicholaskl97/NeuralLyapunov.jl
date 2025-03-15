using ModelingToolkit
import ModelingToolkit: inputs, generate_control_function
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using ControlSystemsBase: lqr, Continuous
using Test, StableRNGs

rng = StableRNG(0)

################################## Double pendulum energy ##################################
function U(x, p)
    θ1, θ2, _, _ = x
    I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    return -m1 * g * lc1 * cos(θ1) - m2 * g * (l1 * cos(θ1) + lc2 * cos(θ1 + θ2))
end
function T(x, p)
    θ1, θ2, ω1, ω2 = x
    I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2)   I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2)                        I2
    ]
    return 0.5 * dot([ω1, ω2], M, [ω1, ω2])
end
E(x, p) = T(x, p) + U(x, p)

######################### Undriven double pendulum conserve energy #########################
println("Undriven double pendulum energy conservation test")

@named double_pendulum_undriven = DoublePendulum(; actuation=:undriven)

t, = independent_variables(double_pendulum_undriven)
Dt = Differential(t)
θ1, θ2 = unknowns(double_pendulum_undriven)
x0 = Dict([θ1, θ2, Dt(θ1), Dt(θ2)] .=> vcat(2π * rand(rng, 2) .- π, zeros(2)))

# Assume uniform rods of random mass and length
m1, m2 = ones(2)
l1, l2 = ones(2)
lc1, lc2 = l1 /2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
p = Dict(parameters(double_pendulum_undriven) .=> [I1, I2, l1, l2, lc1, lc2, m1, m2, g])

prob = ODEProblem(structural_simplify(double_pendulum_undriven), x0, 100, p)
sol = solve(prob, Tsit5(), abstol = 1e-10, reltol = 1e-10)

# Test energy conservation
x = vcat(sol[θ1]', sol[θ2]', sol[Dt(θ1)]', sol[Dt(θ2)]')
p = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]
potential_energy = vec(mapslices(Base.Fix2(U, p), x; dims=1))
kinetic_energy = vec(mapslices(Base.Fix2(T, p), x; dims=1))
total_energy = vec(mapslices(Base.Fix2(E, p), x; dims=1))

#=
plot(
    sol.t,
    [potential_energy, kinetic_energy, total_energy],
    labels=["Potential energy" "Kinetic Energy" "Total Energy"],
    xlabel="Time",
    ylabel="Energy"
)
=#

avg_energy = sum(total_energy) / length(total_energy)
@test maximum(abs, total_energy .- avg_energy) / abs(avg_energy) < 1e-4

# Test plotting extension
anim = plot_double_pendulum(sol, p)
@test anim isa Plots.Animation
# gif(anim, fps=50)

########################### Feedback cancellation, PD controller ###########################
println("Double pendulum feedback cancellation test")

@named double_pendulum = DoublePendulum()

function π_cancellation(x, p)
    θ1, θ2, ω1, ω2 = x
    I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2)   I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2)                        I2
    ]
    G = [
        -m1 * g * lc1 * sin(θ1) - m2 * g * (l1 * sin(θ1) + lc2 * sin(θ1 + θ2));
        -m2 * g * lc2 * sin(θ1 + θ2)
    ]
    return -0.1 * M \ ([θ1, θ2] .- [π, π] + [ω1, ω2]) - G
end

_, x, p, double_pendulum_simplified = generate_control_function(
    double_pendulum;
    simplify=true,
    split=false
)

t, = independent_variables(double_pendulum)
Dt = Differential(t)

p = map(Base.Fix1(getproperty, double_pendulum), toexpr.(p))
u = map(
        Base.Fix1(getproperty, double_pendulum),
        toexpr.(getproperty.(inputs(double_pendulum_simplified), :f))
)
x = [double_pendulum.θ1, double_pendulum.θ2, Dt(double_pendulum.θ1), Dt(double_pendulum.θ2)]

@named cancellation_controller = ODESystem(
    u .~ π_cancellation(x, p),
    t,
    u,
    p
)
@named double_pendulum_feedback_cancellation = compose(cancellation_controller, double_pendulum)
double_pendulum_feedback_cancellation = structural_simplify(double_pendulum_feedback_cancellation)

# Swing up to upward equilibrium
# Assume uniform rods of random mass and length
m1, m2 = ones(2)
l1, l2 = ones(2)
lc1, lc2 = l1 /2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
p = Dict(p .=> [I1, I2, l1, l2, lc1, lc2, m1, m2, g])

x0 = Dict(x .=> vcat(2π * rand(rng, 2) .- π, rand(rng, 2)))

prob = ODEProblem(double_pendulum_feedback_cancellation, x0, 100, p)
sol = solve(prob, Tsit5())
θ1_end, ω1_end = sol[:double_pendulum₊θ1][end], sol.u[end][3]
x1_end, y1_end = sin(θ1_end), -cos(θ1_end)
θ2_end, ω2_end = sol[:double_pendulum₊θ2][end], sol.u[end][4]
x2_end, y2_end = sin(θ2_end), -cos(θ2_end)
@test sqrt(sum(abs2, [x1_end, y1_end] .- [0, 1])) ≈ 0 atol=1e-4
@test sqrt(sum(abs2, [x2_end, y2_end] .- [0, 1])) ≈ 0 atol=1e-4
@test ω1_end ≈ 0 atol=1e-4
@test ω2_end ≈ 0 atol=1e-4

#=
gif(
    plot_double_pendulum(
        sol,
        [I1, I2, l1, l2, lc1, lc2, m1, m2, g];
        angle1_symbol=:double_pendulum₊θ1,
        angle2_symbol=:double_pendulum₊θ2
    ),
    fps=50
)
=#

################################## LQR acrobot controller ##################################
println("Acrobot LQR test")

function acrobot_lqr_matrix(p; x_eq = [π, 0, 0, 0], Q = I(4), R = I(1))
    I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    θ1, θ2 = x_eq[1:2]

    # Assumes linearization around a fixed point
    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2)   I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2)                        I2
    ]
    B = [0, 1]
    Jτ_g = [
        -m1 * g * lc1 * cos(θ1) - m2 * g * (l1 * cos(θ1) + lc2 * cos(θ1 + θ2))  -m2 * g * lc2 * cos(θ1 + θ2);
        -m2 * g * lc2 * cos(θ1 + θ2)                                            -m2 * g * lc2 * cos(θ1 + θ2)
    ]

    A_lin = [
        zeros(2, 2)     I(2);
        M \ Jτ_g        zeros(2, 2)
    ]
    B_lin = [zeros(2); M \ B]

    return lqr(Continuous, A_lin, B_lin, Q, R)
end

function π_lqr(p; x_eq = [π, 0, 0, 0], Q = I(4), R = I(1))
    L = acrobot_lqr_matrix(p; x_eq, Q, R)
    return (x) -> -L * (x .- x_eq)
end

@named acrobot = Acrobot()

_, x, params, acrobot_simplified = generate_control_function(
    acrobot;
    simplify=true,
    split=false
)

t, = independent_variables(acrobot)
Dt = Differential(t)

params = map(Base.Fix1(getproperty, acrobot), toexpr.(params))
u = map(
        Base.Fix1(getproperty, acrobot),
        toexpr.(getproperty.(inputs(acrobot_simplified), :f))
)
x = [acrobot.θ1, acrobot.θ2, Dt(acrobot.θ1), Dt(acrobot.θ2)]

# Assume uniform rods of random mass and length
m1, m2 = ones(2)
l1, l2 = ones(2)
lc1, lc2 = l1 /2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
p = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]

@named lqr_controller = ODESystem(
    u .~ π_lqr(p; x_eq=[π,π,0,0])(x),
    t,
    vcat(x, u),
    params
)
@named acrobot_lqr = compose(lqr_controller, acrobot)
acrobot_lqr = structural_simplify(acrobot_lqr)

# Remain close to upward equilibrium
x0 = [π, π, 0, 0] + 0.005 * vcat(2π * rand(rng, 2) .- π, 2 * rand(rng, 2) .- 1)
tspan = 1000

prob = ODEProblem(acrobot_lqr, Dict(x .=> x0), tspan, Dict(params .=> p))
sol = solve(prob, Tsit5())

anim = plot_double_pendulum(sol, p; angle1_symbol=:acrobot₊θ1, angle2_symbol=:acrobot₊θ2)
@test anim isa Plots.Animation
# gif(anim, fps=50)

x1_end, y1_end, ω1_end = sin(sol[acrobot.θ1][end]), -cos(sol[acrobot.θ1][end]), sol[Dt(acrobot.θ1)][end]
x2_end, y2_end, ω2_end = sin(sol[acrobot.θ2][end]), -cos(sol[acrobot.θ2][end]), sol[Dt(acrobot.θ2)][end]
@test sqrt(sum(abs2, [x1_end, y1_end] .- [0, 1])) ≈ 0 atol=1e-4
@test sqrt(sum(abs2, [x2_end, y2_end] .- [0, 1])) ≈ 0 atol=1e-4
@test ω1_end ≈ 0 atol=1e-4
@test ω2_end ≈ 0 atol=1e-4
