using NeuralPDE, Lux, NeuralLyapunov
import Optimization, OptimizationOptimisers, OptimizationOptimJL
using Random, StableRNGs
using Test, LinearAlgebra, ForwardDiff

rng = StableRNG(0)
Random.seed!(200)

println("Region of Attraction Estimation")

######################### Define dynamics and domain ##########################

f(x, p, t) = -x .+ x .^ 3
lb = [-2.0];
ub = [2.0];
fixed_point = [0.0];

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 5
dim_output = 2
chain = [Lux.Chain(
             Dense(dim_state, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]
ps = Lux.initialparameters(rng, chain)

# Define training strategy
strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy; init_params = ps)

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(dim_output)
minimization_condition = DontCheckNonnegativity()

# Define Lyapunov decrease condition
decrease_condition = make_RoA_aware(StabilityISL(rectifier = (t) -> log(one(t) + exp(t))))

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
sym_prob = symbolic_discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################
V, V̇ = get_numerical_lyapunov_function(
    discretization.phi,
    res.u.depvar,
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

#################################### Tests ####################################

# Network structure should enforce positive definiteness of V
@test min(V(fixed_point), minimum(V_samples)) ≥ 0.0
@test V(fixed_point) == 0.0

# Check local negative definiteness of V̇ at fixed point
@test V̇(fixed_point) == 0.0
@test ForwardDiff.gradient(V̇, fixed_point)[] == 0.0
@test ForwardDiff.hessian(V̇, fixed_point)[] ≤ 0

# V̇ should be negative everywhere in the region of attraction except the fixed point
@test maximum(V̇(transpose(RoA_states[RoA_states .!= fixed_point[]]))) < 0

# The estimated region of attraction should be a subset of the real region of attraction
@test first(RoA) ≥ -1.0 && last(RoA) ≤ 1.0

#=
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

# Plot results
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
=#
