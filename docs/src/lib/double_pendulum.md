# Double Pendulum Model

An undamped double pendulum can be constructed using the [`DoublePendulum`](@ref) function, as shown below.
Models are provided for the fully-actuated version, the undriven version, and both of the underactuated versions (also accessible via the convenience functions [`Acrobot`](@ref) and [`Pendubot`](@ref)).
Additionally, when also using the Plots.jl package, the convenience plotting function [`plot_double_pendulum`](@ref) is provided.

![Double pendulum animation](../imgs/double_pendulum.gif)

```@docs
DoublePendulum
Acrobot
Pendubot
```

## Copy-Pastable Code

```@example plot_double_pendulum
using Random; Random.seed!(200) # hide
using ModelingToolkit, NeuralLyapunovProblemLibrary, Plots, OrdinaryDiffEq

@named double_pendulum_undriven = DoublePendulum(; actuation = :undriven)

t, = independent_variables(double_pendulum_undriven)
Dt = Differential(t)
θ1, θ2 = unknowns(double_pendulum_undriven)
x0 = Dict([θ1, θ2, Dt(θ1), Dt(θ2)] .=> vcat(2π * rand(2) .- π, zeros(2)))

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

p = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]
gif(plot_double_pendulum(sol, p); fps=50)
```

## Plotting the Double Pendulum

```@docs
plot_double_pendulum
```