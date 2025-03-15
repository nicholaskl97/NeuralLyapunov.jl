using ModelingToolkit
import ModelingToolkit: inputs, generate_control_function
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using Test, StableRNGs

rng = StableRNG(0)

################## Undriven pendulum should drop to downward equilibrium ###################
println("Undriven pendulum test")

@named pendulum_undriven = Pendulum(; driven = false)

x0 = π * rand(rng, 2)
p = rand(rng, 2)
τ = 1 / prod(p)
prob = ODEProblem(structural_simplify(pendulum_undriven), x0, 15τ, p)
sol = solve(prob, Tsit5())
x_end, y_end, ω_end = sin(sol.u[end][1]), -cos(sol.u[end][1]), sol.u[end][2]
@test sqrt(sum(abs2, [x_end, y_end] .- [0, -1])) ≈ 0 atol=1e-4
@test ω_end ≈ 0 atol=1e-4

anim = plot_pendulum(sol)
@test anim isa Plots.Animation
# gif(anim, fps=50)

############################# Feedback cancellation controller #############################
println("Simple pendulum feedback cancellation test")

@named pendulum = Pendulum()

π_cancellation(x, p) = 2 * p[2]^2 * sin(x[1])

_, x, p, pendulum_simplified = generate_control_function(
    pendulum;
    simplify=true,
    split=false
)

t, = independent_variables(pendulum)
Dt = Differential(t)

p = map(Base.Fix1(getproperty, pendulum), toexpr.(p))
u = map(
        Base.Fix1(getproperty, pendulum),
        toexpr.(getproperty.(inputs(pendulum_simplified), :f))
)

@named cancellation_controller = ODESystem(
    u .~ π_cancellation([pendulum.θ, Dt(pendulum.θ)], p),
    t,
    u,
    p
)
@named pendulum_feedback_cancellation = compose(cancellation_controller, pendulum)
pendulum_feedback_cancellation = structural_simplify(pendulum_feedback_cancellation)

# Swing up to upward equilibrium
x0 = rand(rng, 2)
p = rand(rng, 2)
τ = 1 / prod(p)
prob = ODEProblem(pendulum_feedback_cancellation, x0, 15τ, p)
sol = solve(prob, Tsit5())
x_end, y_end, ω_end = sin(sol.u[end][1]), -cos(sol.u[end][1]), sol.u[end][2]
@test sqrt(sum(abs2, [x_end, y_end] .- [0, 1])) ≈ 0 atol=1e-4
@test ω_end ≈ 0 atol=1e-4
