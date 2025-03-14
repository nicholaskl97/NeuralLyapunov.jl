# Benchmarking a neural Lyapunov method

In this demonstration, we'll benchmark the neural Lyapunov method used in the [policy search demo](policy_search.md).
In that demonstration, we searched for a neural network policy to stabilize the upright equilibrium of the inverted pendulum.
Here, we will use the [`benchmark`](@ref) function to run approximately the same training, then check the performance the of the resulting controller and neural Lyapunov function by simulating the closed loop system to see (1) how well the controller drives the pendulum to the upright equilibrium, and (2) how well the neural Lyapunov function performs as a classifier of whether a state is in the region of attraction or not.
These results will be represented by a confusion matrix using the simulation results as ground truth.
(Keep in mind that training does no simulation.)

## Copy-Pastable Code

```julia
using NeuralPDE, NeuralLyapunov, Lux
import Boltz.Layers: PeriodicEmbedding
using OptimizationOptimisers
using Random

Random.seed!(200)

# Define dynamics and domain
function open_loop_pendulum_dynamics(x, u, p, t)
    θ, ω = x
    ζ, ω_0 = p
    τ = u[]
    return [ω
            -2ζ * ω_0 * ω - ω_0^2 * sin(θ) + τ]
end

lb = [0.0, -10.0];
ub = [2π, 10.0];
upright_equilibrium = [π, 0.0]
p = [0.5, 1.0]
state_syms = [:θ, :ω]
parameter_syms = [:ζ, :ω_0]

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(lb)
dim_hidden = 15
dim_phi = 2
dim_u = 1
dim_output = dim_phi + dim_u
chain = [Lux.Chain(
             PeriodicEmbedding([1], [2π]),
             Dense(3, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1, use_bias = false)
         ) for _ in 1:dim_output]

# Define neural network discretization
strategy = QuasiRandomTraining(1250)

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(
    dim_phi;
    pos_def = function (state, fixed_point)
        θ, ω = state
        θ_eq, ω_eq = fixed_point
        log(1.0 + (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (ω - ω_eq)^2)
    end
)
structure = add_policy_search(
    structure,
    dim_u
)
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticStability()

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

# Define optimization parameters
opt = OptimizationOptimisers.Adam()
optimization_args = [:maxiters => 1000]

# Run benchmark
cm, time = benchmark(
    open_loop_pendulum_dynamics,
    lb,
    ub,
    spec,
    chain,
    strategy,
    opt;
    simulation_time = 200,
    n_grid = 20,
    fixed_point = upright_equilibrium,
    p = p,
    optimization_args = optimization_args,
    state_syms = state_syms,
    parameter_syms = parameter_syms,
    policy_search = true,
    endpoint_check = (x) -> ≈([sin(x[1]), cos(x[1]), x[2]], [0, -1, 0], atol=5e-3),
)
```

## Detailed Description

Much of the set up is the same as in the [policy search demo](policy_search.md), so see that page for details.


```@example benchmarking
using NeuralPDE, NeuralLyapunov, Lux
import Boltz.Layers: PeriodicEmbedding
using Random

Random.seed!(200)

# Define dynamics and domain
function open_loop_pendulum_dynamics(x, u, p, t)
    θ, ω = x
    ζ, ω_0 = p
    τ = u[]
    return [ω
            -2ζ * ω_0 * ω - ω_0^2 * sin(θ) + τ]
end

lb = [0.0, -10.0];
ub = [2π, 10.0];
upright_equilibrium = [π, 0.0]
p = [0.5, 1.0]
state_syms = [:θ, :ω]
parameter_syms = [:ζ, :ω_0]

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(lb)
dim_hidden = 15
dim_phi = 2
dim_u = 1
dim_output = dim_phi + dim_u
chain = [Lux.Chain(
             PeriodicEmbedding([1], [2π]),
             Dense(3, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1, use_bias = false)
         ) for _ in 1:dim_output]

# Define neural network discretization
strategy = QuasiRandomTraining(1250)

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(
    dim_phi;
    pos_def = function (state, fixed_point)
        θ, ω = state
        θ_eq, ω_eq = fixed_point
        log(1.0 + (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (ω - ω_eq)^2)
    end
)
structure = add_policy_search(
    structure,
    dim_u
)
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticStability()

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)
```

At this point of the [policy search demo](policy_search.md), we constructed the PDESystem, discretized it using NeuralPDE.jl, and solved the resulting OptimizationProblem using Optimization.jl.
All of that occurs in the [`benchmark`](@ref) function, so we instead provide that function with the optimizer and optimization arguments to use.

```@example benchmarking
using OptimizationOptimisers

# Define optimization parameters
opt = OptimizationOptimisers.Adam()
optimization_args = [:maxiters => 1000]
```

Finally, we can run the [`benchmark`](@ref) function.

```@example benchmarking
endpoint_check = (x) -> ≈([sin(x[1]), cos(x[1]), x[2]], [0, -1, 0], atol=5e-3)
(confusion_matrix, training_time), (states, endpoints, actual, predicted, V_samples, V̇_samples) = benchmark(
    open_loop_pendulum_dynamics,
    lb,
    ub,
    spec,
    chain,
    strategy,
    opt;
    simulation_time = 200,
    n_grid = 20,
    fixed_point = upright_equilibrium,
    p = p,
    optimization_args = optimization_args,
    state_syms = state_syms,
    parameter_syms = parameter_syms,
    policy_search = true,
    endpoint_check = endpoint_check,
    classifier = (V, V̇, x) -> V̇ < zero(V̇) || endpoint_check(x),
    verbose = true
);
nothing # hide
```

In this case, we used the `verbose = true` option to demonstrate the outputs of that option, but if you only want the confusion matrix and training time, leave that option off (`verbose` defaults to `false`) and change the first line to:

```julia
confusion_matrix, training_time = benchmark(
```

We can observe the confusion matrix and training time:

```@example benchmarking
confusion_matrix
```

```@example benchmarking
training_time
```

The returned `actual` labels are just `endpoint_check` applied to `endpoints`, which are the results of simulating from each element of `states`.

```@example benchmarking
all(endpoint_check.(endpoints) .== actual)
```

Similarly, the `predicted` labels are the results of the neural Lyapunov classifier.
In this case, we used the default classifier, which just checks for negative values of ``\dot{V}``.

```@example benchmarking
classifier = (V, V̇, x) -> V̇ < zero(V̇) || endpoint_check(x)
all(classifier.(V_samples, V̇_samples, states) .== predicted)
```
