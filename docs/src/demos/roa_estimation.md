# Estimating the Region of Attraction

In this demonstration, we add awareness of the region of attraction (RoA) estimation task to our training.

We'll be examining the simple one-dimensional differential equation:
```math
\frac{dx}{dt} = - x + x^3.
```
This system has a fixed point at ``x = 0`` which has a RoA of ``x \in (-1, 1)``, which we will attempt to identify.

We'll train in the larger domain ``x \in [-2, 2]``.

## Copy-Pastable Code

```julia
using NeuralPDE, Lux, NeuralLyapunov
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Random, StableRNGs

rng = StableRNG(0)
Random.seed!(200)

######################### Define dynamics and domain ##########################

f(x, p, t) = -x .+ x .^ 3
lb = [-2.0];
ub = [ 2.0];
fixed_point = [0.0];

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 5
dim_output = 2
chain = [Lux.Chain(
             Dense(dim_state, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1, use_bias = false)
         ) for _ in 1:dim_output]
ps = Lux.initialparameters(rng, chain)

# Define training strategy
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps)

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(dim_output)
minimization_condition = DontCheckNonnegativity()

# Define Lyapunov decrease condition
decrease_condition = make_RoA_aware(AsymptoticStability())

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(
    f,
    lb,
    ub,
    spec
)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################
net = discretization.phi
θ = res.u.depvar

V, V̇ = get_numerical_lyapunov_function(
    net,
    θ,
    structure,
    f,
    fixed_point
)

################################## Simulate ###################################
states = lb[]:0.001:ub[]
V_samples = vec(V(states'))
V̇_samples = vec(V̇(states'))

# Calculated RoA estimate
ρ = decrease_condition.ρ
RoA_states = states[vec(V(transpose(states))) .≤ ρ]
RoA = (first(RoA_states), last(RoA_states))
```

## Detailed description

```@setup RoA
using Random

Random.seed!(200)
```

In this example, we set up the dynamics as a Julia function and don't bother specifying the symbols for the variables (so ``x`` will be called the default `state1` in the PDESystem).

```@example RoA
f(x, p, t) = -x .+ x .^ 3
lb = [-2.0];
ub = [ 2.0];
fixed_point = [0.0];
nothing # hide
```

Setting up the neural network using Lux and NeuralPDE training strategy is no different from any other physics-informed neural network problem.
For more on that aspect, see the [NeuralPDE documentation](https://docs.sciml.ai/NeuralPDE/stable/).
Since we're only considering one dimension, training on a grid isn't so bad in this case.

```@example RoA
using Lux, StableRNGs

# Stable random number generator for doc stability
rng = StableRNG(0)

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 5
dim_output = 2
chain = [Lux.Chain(
             Dense(dim_state, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1, use_bias = false)
         ) for _ in 1:dim_output]
ps = Lux.initialparameters(rng, chain)
```

```@example RoA
using NeuralPDE

# Define training strategy
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
nothing # hide
```

We now define our Lyapunov candidate structure along with the form of the Lyapunov conditions we'll be using.

For this example, let's use the default Lyapunov candidate from [`PositiveSemiDefiniteStructure`](@ref):
```math
V(x) = \left( 1 + \lVert \phi(x) \rVert^2 \right) \log \left( 1 + \lVert x \rVert^2 \right),
```
which structurally enforces positive definiteness.
We therefore use [`DontCheckNonnegativity()`](@ref).

We only require asymptotic stability in this example, but we use [`make_RoA_aware`](@ref) to only penalize positive values of ``\dot{V}(x)`` when ``V(x) \le 1``.

```@example RoA
using NeuralLyapunov

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(dim_output)
minimization_condition = DontCheckNonnegativity()

# Define Lyapunov decrease condition
decrease_condition = make_RoA_aware(AsymptoticStability())

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

# Construct PDESystem 
@named pde_system = NeuralLyapunovPDESystem(
    f,
    lb,
    ub,
    spec
)
```

Now, we solve the PDESystem using NeuralPDE the same way we would any PINN problem.

```@example RoA
prob = discretize(pde_system, discretization)

import Optimization, OptimizationOptimisers, OptimizationOptimJL

res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 300)

net = discretization.phi
θ = res.u.depvar
```

We can use the result of the optimization problem to build the Lyapunov candidate as a Julia function, then sample on a finer grid than we trained on to find the estimated region of attraction.

```@example RoA
V, V̇ = get_numerical_lyapunov_function(
    net,
    θ,
    structure,
    f,
    fixed_point
)

# Sample
states = lb[]:0.001:ub[]
V_samples = vec(V(states'))
V̇_samples = vec(V̇(states'))

# Calculate RoA estimate
ρ = decrease_condition.ρ
RoA_states = states[vec(V(transpose(states))) .≤ ρ]
RoA = (first(RoA_states), last(RoA_states))

# Print statistics
println("V(0.,0.) = ", V(fixed_point))
println("V ∋ [", min(V(fixed_point), minimum(V_samples)), ", ", maximum(V_samples), "]")
println(
    "V̇ ∋ [",
    minimum(V̇_samples),
    ", ",
    max(V̇(fixed_point), maximum(V̇_samples)),
    "]",
)
println("True region of attraction: (-1, 1)")
println("Estimated region of attraction: ", RoA)
```

The estimated region of attraction is within the true region of attraction.

```@example RoA
using Plots

p_V = plot(states, V_samples, label = "V", xlabel = "x", linewidth=2);
p_V = hline!([ρ], label = "V = ρ", legend = :top);
p_V = vspan!(collect(RoA); label = "Estimated Region of Attraction", color = :gray, fillstyle = :/);
p_V = vspan!([-1, 1]; label = "True Region of Attraction", opacity = 0.2, color = :green);

p_V̇ = plot(states, V̇_samples, label = "dV/dt", xlabel = "x", linewidth=2);
p_V̇ = hline!([0.0], label = "dV/dt = 0", legend = :top);
p_V̇ = vspan!(collect(RoA); label = "Estimated Region of Attraction", color = :gray, fillstyle = :/);
p_V̇ = vspan!([-1, 1]; label = "True Region of Attraction", opacity = 0.2, color = :green);

plt = plot(p_V, p_V̇)
```