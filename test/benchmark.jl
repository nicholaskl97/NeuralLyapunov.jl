using NeuralPDE, NeuralLyapunov, Lux
import Boltz.Layers: PeriodicEmbedding
using OptimizationOptimisers, OptimizationOptimJL
using StableRNGs, Random
using Test

###################### Damped simple harmonic oscillator ######################
@testset "Simple harmonic oscillator benchmarking" begin
    println("Benchmark: Damped SHO")

    Random.seed!(200)

    # Define dynamics and domain

    "Simple Harmonic Oscillator Dynamics"
    function sho(state, p, t)
        ζ, ω_0 = p
        pos = state[1]
        vel = state[2]
        vcat(vel, -2ζ * ω_0 * vel - ω_0^2 * pos)
    end
    lb = [-5.0, -2.0]
    ub = [5.0, 2.0]
    p = [0.5, 1.0]
    fixed_point = [0.0, 0.0]
    sho_dynamics = ODEFunction(sho; sys = SciMLBase.SymbolCache([:x, :v], [:ζ, :ω_0]))

    # Define neural network discretization
    dim_state = length(lb)
    dim_hidden = 10
    dim_output = 3
    chain = [Chain(
                 Dense(dim_state, dim_hidden, tanh),
                 Dense(dim_hidden, dim_hidden, tanh),
                 Dense(dim_hidden, 1)
             ) for _ in 1:dim_output]

    # Define training strategy
    strategy = QuasiRandomTraining(1000)

    # Define neural Lyapunov structure
    structure = NonnegativeNeuralLyapunov(
        dim_output;
        δ = 1e-6
    )
    minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

    # Define Lyapunov decrease condition
    # Damped SHO has exponential decrease at a rate of k = ζ * ω_0, so we train to certify that
    decrease_condition = ExponentialStability(prod(p))

    # Construct neural Lyapunov specification
    spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition
    )

    # Define optimization parameters
    opt = OptimizationOptimisers.Adam()
    optimization_args = [:maxiters => 450]

    # Run benchmark
    cm, time = benchmark(
        sho,
        lb,
        ub,
        spec,
        chain,
        strategy,
        opt;
        simulation_time = 100,
        n_grid = 20,
        p = p,
        optimization_args = optimization_args
    )

    # SHO is globally asymptotically stable
    @test cm.n == 0

    # Should accurately classify
    @test cm.fn / cm.p < 0.5
end

####################### Inverted pendulum policy search #######################
@testset "Policy search on inverted pendulum benchmarking" begin
    println("Benchmark: Inverted Pendulum - Policy Search")

    Random.seed!(200)

    # Define dynamics and domain
    function open_loop_pendulum_dynamics(x, u, p, t)
        θ, ω = x
        ζ, ω_0 = p
        τ = u[]
        return [ω
                -2ζ * ω_0 * ω - ω_0^2 * sin(θ) + τ]
    end

    lb = [0.0, -2.0]
    ub = [2π, 2.0]
    upright_equilibrium = [π, 0.0]
    p = [0.5, 1.0]
    state_syms = [:θ, :ω]
    parameter_syms = [:ζ, :ω_0]

    # Define neural network discretization
    # We use an input layer that is periodic with period 2π with respect to θ
    dim_state = length(lb)
    dim_hidden = 25
    dim_phi = 3
    dim_u = 1
    dim_output = dim_phi + dim_u
    chain = [Chain(
                 PeriodicEmbedding([1], [2π]),
                 Dense(3, dim_hidden, tanh),
                 Dense(dim_hidden, dim_hidden, tanh),
                 Dense(dim_hidden, 1)
             ) for _ in 1:dim_output]
    ps = Lux.initialparameters(StableRNG(0), chain)

    # Define neural network discretization
    strategy = QuasiRandomTraining(10000)

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

    # Define optimization parameters
    opt = [OptimizationOptimisers.Adam(0.05), OptimizationOptimJL.BFGS()]
    optimization_args = [[:maxiters => 300], [:maxiters => 300]]

    # Run benchmark
    endpoint_check = (x) -> ≈([sin(x[1]), cos(x[1]), x[2]], [0, -1, 0], atol = 5e-3)
    (cm, training_time), (states, endpoints, actual, predicted, V_samples, V̇_samples) = benchmark(
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
        p,
        optimization_args,
        state_syms,
        parameter_syms,
        policy_search = true,
        endpoint_check,
        classifier = (V, V̇, x) -> V̇ < zero(V̇) || endpoint_check(x),
        verbose = true,
        init_params = ps
    )

    # Resulting controller should drive more states to equilibrium than not
    @test cm.p > cm.n

    # Resulting classifier should be accurate
    @test (cm.tp + cm.tn) / (cm.p + cm.n) > 0.9
end

############################### Damped pendulum ###############################
@testset "Damped pendulum benchmarking" begin
    println("Benchmark: Damped Pendulum")

    # Define dynamics and domain
    @parameters ζ ω_0
    defaults = Dict([ζ => 5.0, ω_0 => 1.0])

    @independent_variables t
    @variables θ(t)
    Dt = Differential(t)
    DDt = Dt^2

    eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ 0.0]

    @named damped_pendulum = ODESystem(
        eqs,
        t,
        [θ],
        [ζ, ω_0];
        defaults = defaults
    )

    damped_pendulum = structural_simplify(damped_pendulum)
    bounds = [
        θ ∈ (-π, π),
        Dt(θ) ∈ (-10.0, 10.0)
    ]

    # Define neural network discretization
    # We use an input layer that is periodic with period 2π with respect to θ
    dim_state = length(bounds)
    dim_hidden = 15
    dim_output = 2
    chain = [Lux.Chain(
                 PeriodicEmbedding([1], [2π]),
                 Dense(3, dim_hidden, tanh),
                 Dense(dim_hidden, dim_hidden, tanh),
                 Dense(dim_hidden, 1)
             ) for _ in 1:dim_output]

    # Define neural network discretization
    strategy = QuadratureTraining()

    # Define neural Lyapunov structure
    structure = PositiveSemiDefiniteStructure(
        dim_output;
        pos_def = function (state, fixed_point)
            θ, ω = state
            θ_eq, ω_eq = fixed_point
            log(1.0 + (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (ω - ω_eq)^2)
        end
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
    optimization_args = [:maxiters => 600]

    # Run benchmark
    cm, time = benchmark(
        damped_pendulum,
        bounds,
        spec,
        chain,
        strategy,
        opt;
        simulation_time = 300,
        n_grid = 20,
        optimization_args = optimization_args,
        endpoint_check = (x) -> ≈([sin(x[1]), cos(x[1]), x[2]], [0, 1, 0], atol = 1e-3),
        rng = StableRNG(0)
    )

    # Damped pendulum is globally asymptotically stable, except at upright equilibrium
    @test cm.n == 2

    # Should accurately classify
    @test cm.fn / cm.p < 0.5
end
