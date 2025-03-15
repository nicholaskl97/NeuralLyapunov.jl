# Policy Search on the Driven Inverted Pendulum

In this demonstration, we'll search for a neural network policy to stabilize the upright equilibrium of the inverted pendulum.

The governing differential equation for the driven pendulum is:
```math
\frac{d^2 \theta}{dt^2} + 2 \zeta \omega_0 \frac{d \theta}{dt} + \omega_0^2 \sin(\theta) = \tau,
```
where ``\theta`` is the counterclockwise angle from the downward equilibrium, ``\zeta`` and ``\omega_0`` are system parameters, and ``\tau`` is our control input&mdash;the torque.

We'll jointly train a neural controller ``\tau = u \left( \theta, \frac{d\theta}{dt} \right)`` and neural Lyapunov function ``V`` which will certify the stability of the closed-loop system.

## Copy-Pastable Code

```julia
using NeuralPDE, Lux, ModelingToolkit, NeuralLyapunov
using ModelingToolkit: inputs
using NeuralLyapunovProblemLibrary
import Boltz.Layers: PeriodicEmbedding
import Optimization, OptimizationOptimisers, OptimizationOptimJL
using Random, StableRNGs

rng = StableRNG(0)
Random.seed!(200)

######################### Define dynamics and domain ##########################

@named pendulum = Pendulum(; defaults = [0.5, 1.0])

t, = independent_variables(pendulum)
Dt = Differential(t)
θ, = setdiff(unknowns(pendulum), inputs(pendulum))

bounds = [
    θ ∈ (0, 2π),
    Dt(θ) ∈ (-2.0, 2.0)
]

upright_equilibrium = [π, 0.0]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(bounds)
dim_hidden = 20
dim_phi = 2
dim_u = 1
dim_output = dim_phi + dim_u
chain = [Chain(
             PeriodicEmbedding([1], [2π]),
             Dense(3, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]
ps = Lux.initialparameters(rng, chain)

# Define neural network discretization
strategy = QuasiRandomTraining(5000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps)

# Define neural Lyapunov structure
periodic_pos_def = function (state, fixed_point)
    θ, ω = state
    θ_eq, ω_eq = fixed_point
    return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + 0.1 * (ω - ω_eq)^2
end

structure = PositiveSemiDefiniteStructure(
    dim_phi;
    pos_def = (x, x0) -> log(1.0 + periodic_pos_def(x, x0))
)
structure = add_policy_search(structure, dim_u)

minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticStability(strength = periodic_pos_def)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(
    pendulum,
    bounds,
    spec;
    fixed_point = upright_equilibrium
)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################

net = discretization.phi
_θ = res.u.depvar

(open_loop_pendulum_dynamics, _), state_order, p_order = ModelingToolkit.generate_control_function(
    pendulum; simplify = true, split = false)
p = [defaults(pendulum)[param] for param in p_order]

V_func, V̇_func = get_numerical_lyapunov_function(
    net,
    _θ,
    structure,
    open_loop_pendulum_dynamics,
    upright_equilibrium;
    p
)

u = get_policy(net, _θ, dim_output, dim_u)
```

## Detailed description

```@setup policy_search
using Random

Random.seed!(200)
```

In this example, we'll use the [`Pendulum`](@ref) model in [NeuralLyaupnovProblemLibrary.jl](../lib.md).

Since the angle ``\theta`` is periodic with period ``2\pi``, our box domain will be one period in ``\theta`` and an interval in ``\frac{d\theta}{dt}``.

```@example policy_search
using ModelingToolkit, NeuralLyapunovProblemLibrary
using ModelingToolkit: inputs

@named pendulum = Pendulum(; defaults = [0.5, 1.0])

t, = independent_variables(pendulum)
Dt = Differential(t)
θ, = setdiff(unknowns(pendulum), inputs(pendulum))

bounds = [
    θ ∈ (0, 2π),
    Dt(θ) ∈ (-2.0, 2.0)
]

upright_equilibrium = [π, 0.0]
```

We'll use an architecture that's ``2\pi``-periodic in ``\theta`` so that we can train on just one period of ``\theta`` and don't need to add any periodic boundary conditions.
To achieve that, we use `Boltz.Layers.PeriodicEmbedding([1], [2pi])`, enforces `2pi`-periodicity in input number `1`.
Additionally, we include output dimensions for both the neural Lyapunov function and the neural controller.

Other than that, setting up the neural network using Lux and NeuralPDE training strategy is no different from any other physics-informed neural network problem.
For more on that aspect, see the [NeuralPDE documentation](https://docs.sciml.ai/NeuralPDE/stable/).

```@example policy_search
using Lux
import Boltz.Layers: PeriodicEmbedding
using StableRNGs

# Stable random number generator for doc stability
rng = StableRNG(0)

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(bounds)
dim_hidden = 20
dim_phi = 2
dim_u = 1
dim_output = dim_phi + dim_u
chain = [Chain(
             PeriodicEmbedding([1], [2π]),
             Dense(3, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]
ps = Lux.initialparameters(rng, chain)
```

```@example policy_search
using NeuralPDE

# Define neural network discretization
strategy = QuasiRandomTraining(5000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
nothing # hide
```

We now define our Lyapunov candidate structure along with the form of the Lyapunov conditions we'll be using.

The default Lyapunov candidate from [`PositiveSemiDefiniteStructure`](@ref) is:
```math
V(x) = \left( 1 + \lVert \phi(x) \rVert^2 \right) \log \left( 1 + \lVert x \rVert^2 \right),
```
which structurally enforces positive definiteness.
We'll modify the second factor to be ``2\pi``-periodic in ``\theta``:

```@example policy_search
using NeuralLyapunov

# Define neural Lyapunov structure
periodic_pos_def = function (state, fixed_point)
    θ, ω = state
    θ_eq, ω_eq = fixed_point
    return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + 0.1 * (ω - ω_eq)^2
end

structure = PositiveSemiDefiniteStructure(
    dim_phi;
    pos_def = (x, x0) -> log(1.0 + periodic_pos_def(x, x0))
)
nothing # hide
```

In addition to representing the neural Lyapunov function, our neural network must also represent the controller.
For this, we use the [`add_policy_search`](@ref) function, which tells NeuralLyapunov to expect dynamics with a control input and to treat the last `dim_u` dimensions of the neural network as the output of our controller.

```@example policy_search
structure = add_policy_search(structure, dim_u)
nothing # hide
```

Since our Lyapunov candidate structurally enforces positive definiteness, we use [`DontCheckNonnegativity`](@ref).

```@example policy_search
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticStability(strength = periodic_pos_def)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

# Construct PDESystem 
@named pde_system = NeuralLyapunovPDESystem(
    pendulum,
    bounds,
    spec;
    fixed_point = upright_equilibrium
)
```

Now, we solve the PDESystem using NeuralPDE the same way we would any PINN problem.

```@example policy_search
prob = discretize(pde_system, discretization)

import Optimization, OptimizationOptimisers, OptimizationOptimJL

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 300)

net = discretization.phi
_θ = res.u.depvar
```

We can use the result of the optimization problem to build the Lyapunov candidate as a Julia function, as well as extract our controller, using the [`get_policy`](@ref) function.

```@example policy_search
(open_loop_pendulum_dynamics, _), state_order, p_order = ModelingToolkit.generate_control_function(pendulum; simplify = true, split = false)
p = [defaults(pendulum)[param] for param in p_order]

V_func, V̇_func = get_numerical_lyapunov_function(
    net,
    _θ,
    structure,
    open_loop_pendulum_dynamics,
    upright_equilibrium;
    p = p
)

u = get_policy(net, _θ, dim_output, dim_u)
nothing # hide
```

Now, let's evaluate our controller.
First, we'll get the usual summary statistics on the Lyapunov function and plot ``V``, ``\dot{V}``, and the violations of the decrease condition.

```@example policy_search
lb = [0.0, -2.0];
ub = [2π, 2.0];
xs = (-2π):0.1:(2π)
ys = lb[2]:0.1:ub[2]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_samples = vec(V_func(hcat(states...)))
V̇_samples = vec(V̇_func(hcat(states...)))

# Print statistics
println("V(π, 0) = ", V_func(upright_equilibrium))
println(
    "f([π, 0], u([π, 0])) = ",
    open_loop_pendulum_dynamics(upright_equilibrium, u(upright_equilibrium), p, 0.0)
)
println(
    "V ∋ [",
    min(V_func(upright_equilibrium),
    minimum(V_samples)),
    ", ",
    maximum(V_samples),
    "]"
)
println(
    "V̇ ∋ [",
    minimum(V̇_samples),
    ", ",
    max(V̇_func(upright_equilibrium), maximum(V̇_samples)),
    "]"
)
```

```@example policy_search
using Plots

p1 = plot(
    xs / pi,
    ys,
    V_samples,
    linetype =
    :contourf,
    title = "V",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :bone_1
);
p1 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :red, markershape = :x);
p1 = scatter!(
    [-pi, pi] / pi, [0, 0], label = "Upward Equilibria", color = :green, markershape = :+);
p2 = plot(
    xs / pi,
    ys,
    V̇_samples,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :binary
);
p2 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :red, markershape = :x);
p2 = scatter!([-pi, pi] / pi, [0, 0], label = "Upward Equilibria", color = :green,
    markershape = :+, legend = false);
p3 = plot(
    xs / pi,
    ys,
    V̇_samples .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "θ/π",
    ylabel = "ω",
    colorbar = false,
    linewidth = 0
);
p3 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :green, markershape = :+);
p3 = scatter!([-pi, pi] / pi, [0, 0], label = "Upward Equilibria",
    color = :red, markershape = :x, legend = false);
plot(p1, p2, p3)
```

Now, let's simulate the closed-loop dynamics to verify that the controller can get our system to the upward equilibrium.

First, we'll start at the downward equilibrium:

```@example policy_search
state_order = map(st -> SymbolicUtils.isterm(st) ? operation(st) : st, state_order)
state_syms = Symbol.(state_order)

closed_loop_dynamics = ODEFunction(
    (x, p, t) -> open_loop_pendulum_dynamics(x, u(x), p, t);
    sys = SciMLBase.SymbolCache(state_syms, Symbol.(p_order))
)

using OrdinaryDiffEq: Tsit5

# Starting still at bottom ...
downward_equilibrium = zeros(2)
ode_prob = ODEProblem(closed_loop_dynamics, downward_equilibrium, [0.0, 120.0], p)
sol = solve(ode_prob, Tsit5())
plot(sol)
```

```@example policy_search
# ...the system should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
[x_end, y_end, ω_end] # Should be approximately [0.0, 1.0, 0.0]
```

Then, we'll start at a random state:

```@example policy_search
# Starting at a random point ...
x0 = lb .+ rand(2) .* (ub .- lb)
ode_prob = ODEProblem(closed_loop_dynamics, x0, [0.0, 150.0], p)
sol = solve(ode_prob, Tsit5())
plot(sol)
```

```@example policy_search
# ...the system should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
[x_end, y_end, ω_end] # Should be approximately [0.0, 1.0, 0.0]
```
