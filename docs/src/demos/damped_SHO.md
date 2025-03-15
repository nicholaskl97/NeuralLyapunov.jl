# Damped Simple Harmonic Oscillator

Let's train a neural network to prove the exponential stability of the damped simple harmonic oscillator (SHO).

The damped SHO is represented by the system of differential equations
```math
\begin{align*}
    \frac{dx}{dt} &= v \\
    \frac{dv}{dt} &= -2 \zeta \omega_0 v - \omega_0^2 x
\end{align*}
```
where ``x`` is the position, ``v`` is the velocity, ``t`` is time, and ``\zeta, \omega_0`` are parameters.

We'll consider just the box domain ``x \in [-5, 5], v \in [-2, 2]``.

## Copy-Pastable Code

```julia
using NeuralPDE, Lux, NeuralLyapunov
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using StableRNGs, Random

rng = StableRNG(0)
Random.seed!(200)

######################### Define dynamics and domain ##########################

"Simple Harmonic Oscillator Dynamics"
function f(state, p, t)
    ζ, ω_0 = p
    pos = state[1]
    vel = state[2]
    vcat(vel, -2ζ * ω_0 * vel - ω_0^2 * pos)
end
lb = [-5.0, -2.0];
ub = [ 5.0,  2.0];
p = [0.5, 1.0];
fixed_point = [0.0, 0.0];
dynamics = ODEFunction(f; sys = SciMLBase.SymbolCache([:x, :v], [:ζ, :ω_0]))

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 10
dim_output = 4
chain = [Lux.Chain(
             Dense(dim_state, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]
ps = Lux.initialparameters(rng, chain)

# Define training strategy
strategy = QuasiRandomTraining(1000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps)

# Define neural Lyapunov structure
structure = NonnegativeNeuralLyapunov(
    dim_output;
    δ = 1e-6
)
minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

# Define Lyapunov decrease condition
# Damped SHO has exponential stability at a rate of k = ζ * ω_0, so we train to certify that
decrease_condition = ExponentialStability(prod(p))

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(
    dynamics,
    lb,
    ub,
    spec;
    p = p
)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 500)

###################### Get numerical numerical functions ######################
net = discretization.phi
θ = res.u.depvar

V, V̇ = get_numerical_lyapunov_function(
    net,
    θ,
    structure,
    f,
    fixed_point;
    p = p
)
```

## Detailed description

In this example, we set the dynamics up as an `ODEFunction` and use a `SciMLBase.SymbolCache` to tell the ultimate `PDESystem` what to call our state and parameter variables.

```@setup SHO
using Random

Random.seed!(200)
```

```@example SHO
using SciMLBase # for ODEFunction and SciMLBase.SymbolCache

"Simple Harmonic Oscillator Dynamics"
function f(state, p, t)
    ζ, ω_0 = p
    pos = state[1]
    vel = state[2]
    vcat(vel, -2ζ * ω_0 * vel - ω_0^2 * pos)
end
lb = [-5.0, -2.0];
ub = [ 5.0,  2.0];
p = [0.5, 1.0];
fixed_point = [0.0, 0.0];
dynamics = ODEFunction(f; sys = SciMLBase.SymbolCache([:x, :v], [:ζ, :ω_0]))
nothing # hide
```

Setting up the neural network using Lux and NeuralPDE training strategy is no different from any other physics-informed neural network problem.
For more on that aspect, see the [NeuralPDE documentation](https://docs.sciml.ai/NeuralPDE/stable/).

```@example SHO
using Lux, StableRNGs

# Stable random number generator for doc stability
rng = StableRNG(0)

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 10
dim_output = 3
chain = [Lux.Chain(
             Dense(dim_state, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]
ps = Lux.initialparameters(rng, chain)
```

```@example SHO
using NeuralPDE

# Define training strategy
strategy = QuasiRandomTraining(1000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
nothing # hide
```

We now define our Lyapunov candidate structure along with the form of the Lyapunov conditions we'll be using.

For this example, let's use a Lyapunov candidate
```math
V(x) = \lVert \phi(x) \rVert^2 + \delta \log \left( 1 + \lVert x \rVert^2 \right),
```
which structurally enforces nonnegativity, but doesn't guarantee ``V([0, 0]) = 0``.
We therefore don't need a term in the loss function enforcing ``V(x) > 0 \, \forall x \ne 0``, but we do need something enforcing ``V([0, 0]) = 0``.
So, we use [`DontCheckNonnegativity(check_fixed_point = true)`](@ref).

To train for exponential stability we use [`ExponentialStability`](@ref), but we must specify the rate of exponential decrease, which we know in this case to be ``\zeta \omega_0``.

```@example SHO
using NeuralLyapunov

# Define neural Lyapunov structure
structure = NonnegativeNeuralLyapunov(
    dim_output;
    δ = 1e-6
)
minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

# Define Lyapunov decrease condition
# Damped SHO has exponential stability at a rate of k = ζ * ω_0, so we train to certify that
decrease_condition = ExponentialStability(prod(p))

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

# Construct PDESystem
@named pde_system = NeuralLyapunovPDESystem(
    dynamics,
    lb,
    ub,
    spec;
    p = p
)
```

Now, we solve the PDESystem using NeuralPDE the same way we would any PINN problem.

```@example SHO
prob = discretize(pde_system, discretization)

using Optimization, OptimizationOptimisers, OptimizationOptimJL

res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 500)

net = discretization.phi
θ = res.u.depvar
```

We can use the result of the optimization problem to build the Lyapunov candidate as a Julia function.

```@example SHO
V, V̇ = get_numerical_lyapunov_function(
    net,
    θ,
    structure,
    f,
    fixed_point;
    p = p
)
nothing # hide
```

Now let's see how we did.
We'll evaluate both ``V`` and ``\dot{V}`` on a ``101 \times 101`` grid:

```@example SHO
Δx = (ub[1] - lb[1]) / 100
Δv = (ub[2] - lb[2]) / 100
xs = lb[1]:Δx:ub[1]
vs = lb[2]:Δv:ub[2]
states = Iterators.map(collect, Iterators.product(xs, vs))
V_samples = vec(V(hcat(states...)))
V̇_samples = vec(V̇(hcat(states...)))

# Print statistics
V_min, i_min = findmin(V_samples)
state_min = collect(states)[i_min]
V_min, state_min = if V(fixed_point) ≤ V_min
        V(fixed_point), fixed_point
    else
        V_min, state_min
    end

println("V(0.,0.) = ", V(fixed_point))
println("V ∋ [", V_min, ", ", maximum(V_samples), "]")
println("Minimial sample of V is at ", state_min)
println(
    "V̇ ∋ [",
    minimum(V̇_samples),
    ", ",
    max(V̇(fixed_point), maximum(V̇_samples)),
    "]",
)
```

At least at these validation samples, the conditions that ``\dot{V}`` be negative semi-definite and ``V`` be minimized at the origin are nearly satisfied.

```@example SHO
using Plots

p1 = plot(xs, vs, V_samples, linetype = :contourf, title = "V", xlabel = "x", ylabel = "v");
p1 = scatter!([0], [0], label = "Equilibrium");
p2 = plot(
    xs,
    vs,
    V̇_samples,
    linetype = :contourf,
    title = "V̇",
    xlabel = "x",
    ylabel = "v",
);
p2 = scatter!([0], [0], label = "Equilibrium");
plot(p1, p2)
```

Each sublevel set of ``V`` completely contained in the plot above has been verified as a subset of the region of attraction.