using ModelingToolkit, LinearAlgebra, DifferentialEquations
using Symbolics: scalarize
using NeuralPDE, Lux
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov

##################### Define dynamics via ModelingToolkit #####################

@variables t
D = Differential(t)

function Mass(; name, m = 1.0, b = 10.0, xy = [0.0, 0.0], u = [0.0, 0.0])
    ps = @parameters m = m b = b
    sts = @variables pos(t)[1:2]=xy v(t)[1:2]=u
    eqs = scalarize(D.(pos) .~ v)
    ODESystem(eqs, t, [pos..., v...], ps; name)
end

function damping_force(mass)
    -mass.b .* scalarize(mass.v)
end

function Spring(; name, k = 1e4, l = 1.0)
    ps = @parameters k=k l=l
    @variables x(t), dir(t)[1:2]
    ODESystem(Equation[], t, [x, dir...], ps; name)
end

function connect_spring(spring, a, b)
    [spring.x ~ norm(scalarize(a .- b))
        scalarize(spring.dir .~ scalarize(a .- b))]
end

function spring_force(spring)
    -spring.k .* scalarize(spring.dir) .* (spring.x - spring.l) ./ (1e-6 + spring.x)
end

m = 1.0
b = 3.0
xy = [1.0, -1.0]
k = 1e4
l = 1.0
center = [0.0, 0.0]
g = [0.0, -9.81]
@named mass = Mass(m = m, b = b, xy = xy)
@named spring = Spring(k = k, l = l)

eqs = [connect_spring(spring, mass.pos, center)
    scalarize(D.(mass.v) .~ damping_force(mass) / mass.m + spring_force(spring) / mass.m .+ g)]

@named _model = ODESystem(eqs, t, [spring.x; spring.dir; mass.pos], [])
@named model = compose(_model, mass, spring)
sys = structural_simplify(model)

#=
ode_prob = ODEProblem(sys, [], (0.0, 3.0))
sol = solve(ode_prob, Rosenbrock23())
p1 = plot(sol);
p2 = plot(sol, idxs = (mass.pos[1], mass.pos[2]));
p2 = scatter!([center[1]], [center[2]], legend = false)
plot(p1,p2)
=#
################################ Define domain ################################

lb = [-10.0, -10.0, -5.0, -5.0];
ub = [10.0, 10.0, 5.0, 1.0];
fixed_point = [0.0, 0.0, center[1], center[2] + m*g[2]/k - l]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 10
dim_output = 3
chain = [
    Lux.Chain(
        Dense(dim_state, dim_hidden, tanh),
        Dense(dim_hidden, dim_hidden, tanh),
        Dense(dim_hidden, 1, use_bias = false),
    ) for _ = 1:dim_output
]

# Define neural network discretization
strategy = GridTraining(2.0)
discretization = PhysicsInformedNN(chain, strategy)

# Define neural Lyapunov structure
structure = NonnegativeNeuralLyapunov(
        dim_output; 
        δ = 1e-6
        )
structure = PositiveSemiDefiniteStructure(dim_output)
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Set up decrease condition
decrease_condition = AsymptoticDecrease(strict = true; check_fixed_point = false)
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition,
    )

############################# Construct PDESystem #############################

pde_system, network_func = NeuralLyapunovPDESystem(
        sys, 
        lb, 
        ub, 
        spec; 
        fixed_point = fixed_point
    )

######################## Construct OptimizationProblem ########################

sym_prob = symbolic_discretize(pde_system, discretization)
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 3000)
prob = Optimization.remake(prob, u0 = res.u);

println("Switching from Adam to BFGS");
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi, 
    res.u, 
    network_func, 
    structure.V,
    ODEFunction(sys),
    fixed_point;
    p = [m, b, k, l]
    )

################################## Simulate ###################################
xs = [lb[i]:0.25:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs...))
V_predict = V_func.(states)
dVdt_predict = V̇_func.(states)

lower_coords = map(((vx,vy,x,y),) -> [norm([x,y] - fixed_point[3:4]) , norm([vx,vy] - fixed_point[1:2])], states)
lower_coords = reduce(hcat, vec(lower_coords))
p1 = scatter(lower_coords[1,:], lower_coords[2,:], vec(V_predict), alpha = 0.1, xlabel = "Distance from equilibrium", ylabel = "Speed", zlabel = "V")
png(p1, "x.png")

p2 = scatter(lower_coords[1,:], lower_coords[2,:], vec(dVdt_predict), alpha = 0.1, xlabel = "Distance from equilibrium", ylabel = "Speed", zlabel = "dV/dt")
png(p2, "y.png")


#=
speeds = map(((vx,vy,x,y),) -> sqrt(vx^2 + vy^2), states)
levels = range(minimum(speeds), maximum(speeds), 4)[2:end]

V_limited = [mapslices(
        function(_V)
            maximum(_V[speeds[:,:,1,1] .≤ speed_limit])
        end, 
        V_predict, 
        dims = [1,2]
    ) for speed_limit in levels]
V_limited = map(x -> reshape(x, size(x)[3:4]), V_limited)

V̇_limited = [mapslices(
        function(_V)
            maximum(_V[speeds[:,:,1,1] .≤ speed_limit])
        end, 
        dVdt_predict, 
        dims = [1,2]
    ) for speed_limit in levels]
V̇_limited = map(x -> reshape(x, size(x)[3:4]), V̇_limited)
=#

# Print statistics
V_min = min(V_func(fixed_point), minimum(V_predict))
V_max = maximum(V_predict)
println("V(fixed_point) = ", V_func(fixed_point))
println("V ∋ [", V_min, ", ", V_max, "]")
V̇_min = minimum(dVdt_predict)
V̇_max = max(V̇_func(fixed_point), maximum(dVdt_predict))
println("V̇ ∋ [", V̇_min, ", ", V̇_max, "]")

#=
# Plot results
V_ps = map(
    function ((i, speed_limit),)
        plot(xs[3], xs[4], vec(V_limited[i]), linetype = :contourf, xlabel = "x", ylabel = "y", title = @sprintf("Speed < %.1f", speed_limit), climits = (V_min, V_max))
        scatter!([center[1]], [center[2]], markershape = :+, color = :green, legend = false)
    end,
    enumerate(levels)
)
plot(V_ps..., plot_title = "Maximum V by speed limit and position")

V̇_ps = map(
    function ((i, speed_limit),)
        plot(xs[3], xs[4], vec(V̇_limited[i]), linetype = :contourf, xlabel = "x", ylabel = "y", title = @sprintf("Speed < %.1f", speed_limit))
        scatter!([center[1]], [center[2]], markershape = :+, color = :green, legend = false)
    end,
    enumerate(levels)
)
plot(V̇_ps..., plot_title = "Maximum V̇ by speed limit and position")


# Plot results

p1 = plot(xs, ys, V_predict, linetype = :contourf, title = "V", xlabel = "x", ylabel = "ẋ");
p1 = scatter!([0], [0], label = "Equilibrium");
p2 = plot(
    xs,
    ys,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p2 = scatter!([0], [0], label = "Equilibrium");
p3 = plot(
    xs,
    ys,
    V_predict .< ρ,
    linetype = :contourf,
    title = "Estimated RoA",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = plot(
    xs,
    ys,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = scatter!([0], [0], label = "Equilibrium");
plot(p1, p2, p3, p4)
=#