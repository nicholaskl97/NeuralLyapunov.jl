using NeuralPDE, NeuralLyapunov
import Optimization, OptimizationOptimisers, OptimizationOptimJL
using Random
using Lux, LuxCUDA, ComponentArrays
using Test, LinearAlgebra, ForwardDiff, StableRNGs

rng = StableRNG(0)
Random.seed!(200)

println("Damped Simple Harmonic Oscillator")

######################### Define dynamics and domain ##########################

"Simple Harmonic Oscillator Dynamics"
function f(state, p, t)
    pos = state[1]
    vel = state[2]
    vcat(vel, -vel - pos)
end
lb = [-2.0, -2.0];
ub = [2.0, 2.0];
fixed_point = [0.0, 0.0];
dynamics = ODEFunction(f; sys = SciMLBase.SymbolCache([:x, :v]))

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 20
chain = Chain(
    Dense(dim_state, dim_hidden, tanh),
    Dense(dim_hidden, dim_hidden, tanh),
    Dense(dim_hidden, dim_hidden, tanh),
    Dense(dim_hidden, 1)
)
const gpud = gpu_device()
ps = Lux.initialparameters(rng, chain) |> ComponentArray |> gpud |> f32

# Define training strategy
strategy = QuasiRandomTraining(2500)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps)

# Define neural Lyapunov structure
structure = UnstructuredNeuralLyapunov()
minimization_condition = StrictlyPositiveDefinite(C = 0.1)

# Define Lyapunov decrease condition
# This damped SHO has exponential decrease at a rate of k = 0.5, so we train to certify that
decrease_condition = ExponentialStability(0.5)

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
)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################
V, V̇ = get_numerical_lyapunov_function(
    discretization.phi,
    (; φ1 = res.u),
    structure,
    f,
    fixed_point
)

################################## Simulate ###################################
Δx = (ub[1] - lb[1]) / 100
Δv = (ub[2] - lb[2]) / 100
xs = lb[1]:Δx:ub[1]
vs = lb[2]:Δv:ub[2]
states = Iterators.map(collect, Iterators.product(xs, vs))
V_samples_gpu = vec(V(hcat(states...)))
V̇_samples_gpu = vec(V̇(hcat(states...)))

const cpud = cpu_device()
V_samples = V_samples_gpu |> cpud
V̇_samples = V̇_samples_gpu |> cpud

#################################### Tests ####################################

# Network structure should enforce nonegativeness of V
V0 = (V(fixed_point) |> cpud)[]
V_min, i_min = findmin(V_samples)
state_min = collect(states)[i_min]
V_min, state_min = if V0 ≤ V_min
    V0, fixed_point
else
    V_min, state_min
end
@test V_min ≥ -1e-2

# Trained for V's minimum to be near the fixed point
@test all(abs.(state_min .- fixed_point) .≤ 10 * [Δx, Δv])

# Check local negative semidefiniteness of V̇ at fixed point
@test (V̇(fixed_point) |> cpud)[] == 0.0
@test maximum(abs, ForwardDiff.gradient(x -> (V̇(x) |> cpud)[], fixed_point)) < 0.1
@test_broken maximum(eigvals(ForwardDiff.hessian(x -> (V̇(x) |> cpud)[], fixed_point))) ≤ 0.0

# V̇ should be negative almost everywhere
@test sum(V̇_samples .> 0) / length(V̇_samples) < 5e-3

#=
# Print statistics
println("V(0.,0.) = ", V(fixed_point))
println("V ∋ [", V_min, ", ", maximum(V_samples), "]")
println("Minimial sample of V is at ", state_min)
println(
    "V̇ ∋ [",
    minimum(V̇_samples),
    ", ",
    max((V̇(fixed_point) |> cpud)[], maximum(V̇_samples)),
    "]",
)

# Plot results
using Plots

p1 = plot(xs, vs, V_samples, linetype = :contourf, title = "V", xlabel = "x", ylabel = "ẋ");
p1 = scatter!([0], [0], label = "Equilibrium");
p2 = plot(
    xs,
    vs,
    V̇_samples,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p2 = scatter!([0], [0], label = "Equilibrium");
plot(p1, p2)
=#
